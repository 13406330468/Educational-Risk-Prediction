"""Reproduce manuscript-review evidence from frozen XuetangX artifacts.

No predictor is trained. Private enrollment predictions are read only to audit
the saved folds and compute the requested within-fold prevalence comparator.
Outputs contain aggregate pair/job results, never enrollment or user IDs.
Python 3.12; numpy, pandas, pyarrow, scikit-learn.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DAYS = [7, 14, 28]
KEY = ["pair_id", "day", "seed", "base_model"]
PAIR = ["pair_id", "provider", "base_course_token", "day", "base_model"]


def bootstrap(g, measures, seed, draws=2000):
    """Pair-equal mean, preserving all pairs in resampled base-course clusters."""
    values = g[measures].to_numpy(float)
    cluster = g.provider.astype(str) + "|" + g.base_course_token.astype(str)
    unique = sorted(cluster.unique())
    sums = np.stack([values[cluster.eq(c)].sum(axis=0) for c in unique])
    ns = np.array([cluster.eq(c).sum() for c in unique])
    idx = np.random.default_rng(seed).integers(0, len(unique), size=(draws, len(unique)))
    bs = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)[:, None]
    row = {"pairs": len(g), "base_course_clusters": len(unique)}
    for j, measure in enumerate(measures):
        row[measure] = float(values[:, j].mean())
        row[measure + "_ci_low"] = float(np.quantile(bs[:, j], .025))
        row[measure + "_ci_high"] = float(np.quantile(bs[:, j], .975))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary-dir", type=Path, required=True)
    ap.add_argument("--gap-dir", type=Path, required=True)
    ap.add_argument("--prediction-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def save(frame, name):
        frame.to_csv(args.out / name, index=False)
    jobs = pd.read_csv(args.primary_dir / "primary_by_seed.csv")
    jobs = jobs[jobs.day.isin(DAYS)].copy()
    pair = pd.read_csv(args.primary_dir / "primary_pair_mean.csv")
    pair = pair[pair.day.isin(DAYS)].copy()
    assert not jobs[KEY].duplicated().any()
    audit = {"primary_jobs": len(jobs), "primary_pair_units": len(pair)}

    # The exact same bootstraps already used by the primary frozen report.
    utility, distribution = [], []
    for (day, model), g in pair.groupby(["day", "base_model"]):
        row = {"day": day, "base_model": model}
        row.update(bootstrap(g, ["G_S", "G_T", "G_T_star", "A"],
                             20261001 + 100 * day + (model == "lightgbm")))
        utility.append(row)
        gt = g.G_T
        distribution.append({"day": day, "base_model": model, "pairs": len(g),
                             "negative_pairs": int(gt.lt(0).sum()),
                             "positive_pairs": int(gt.gt(0).sum()),
                             "positive_fraction": float(gt.gt(0).mean()),
                             "min": gt.min(), "q10": gt.quantile(.1),
                             "q25": gt.quantile(.25), "median": gt.median(),
                             "q75": gt.quantile(.75), "q90": gt.quantile(.9),
                             "max": gt.max()})
    utility = pd.DataFrame(utility)
    frozen_ci = pd.read_csv(args.primary_dir / "course_cluster_bootstrap.csv")
    diffs = []
    for row in utility.itertuples():
        for measure in ["G_T", "G_T_star", "A"]:
            f = frozen_ci[(frozen_ci.day == row.day) & (frozen_ci.base_model == row.base_model)
                          & (frozen_ci.measure == measure)].iloc[0]
            diffs.extend([abs(getattr(row, measure) - f.macro_pair_mean),
                          abs(getattr(row, measure + "_ci_low") - f.cluster_ci_2_5),
                          abs(getattr(row, measure + "_ci_high") - f.cluster_ci_97_5)])
    audit["max_frozen_primary_ci_absolute_difference"] = max(diffs)
    assert max(diffs) < 1e-12
    save(utility, "primary_gain_intervals.csv")
    save(pd.DataFrame(distribution), "transfer_gain_pair_distribution.csv")
    save(pair[PAIR + ["seeds", "target_n", "G_S", "G_T", "G_T_star", "A"]],
         "primary_gain_pair_mean.csv")

    # Refit and original are compared on common jobs before averaging seeds.
    refit = pd.read_csv(args.gap_dir / "identity_refit_by_seed.csv")
    match = jobs.merge(refit, on=KEY, suffixes=("_original", "_refit"), validate="one_to_one")
    match["provider"] = match.provider_refit
    match["base_course_token"] = match.base_course_token_refit
    match["delta_A"] = match.A_refit - match.A_original
    mp = match.groupby(PAIR, as_index=False)[["A_original", "A_refit", "delta_A"]].mean()
    deltas = []
    frozen_refit = pd.read_csv(args.gap_dir / "identity_refit_summary.csv")
    deldiff = []
    for (day, model), g in mp.groupby(["day", "base_model"]):
        row = {"day": day, "base_model": model,
               "matched_jobs": int(((match.day == day) & (match.base_model == model)).sum())}
        row.update(bootstrap(g, ["A_original", "A_refit", "delta_A"],
                             20261002 + day * 11 + (model == "lightgbm")))
        deltas.append(row)
        frozen = frozen_refit[(frozen_refit.day == day) & (frozen_refit.base_model == model)].iloc[0]
        for suffix in ["", "_ci_low", "_ci_high"]:
            deldiff.append(abs(row["delta_A" + suffix] - frozen["A_refit_minus_original" + suffix]))
    assert max(deldiff) < 1e-12
    audit["max_frozen_delta_ci_absolute_difference"] = max(deldiff)
    save(pd.DataFrame(deltas), "identity_refit_delta_intervals.csv")

    # Mean 20 sampled other sources within job, then seeds, then equal pairs.
    history = pd.read_csv(args.primary_dir / "random_course_negative_control.csv")
    history = history[history.day.isin(DAYS)]
    repeats = history.groupby(KEY).size()
    assert repeats.eq(20).all()
    histjob = history.groupby(KEY, as_index=False)[["G_T_same_course_previous", "G_T_random_other_source"]].mean()
    histjob = histjob.merge(jobs[KEY + ["provider", "base_course_token", "target_n", "G_T"]], on=KEY, validate="one_to_one")
    assert np.max(abs(histjob.G_T - histjob.G_T_same_course_previous)) < 1e-12
    histjob["same_minus_random"] = histjob.G_T_same_course_previous - histjob.G_T_random_other_source
    hp = histjob.groupby(PAIR, as_index=False)[["G_T_same_course_previous", "G_T_random_other_source", "same_minus_random", "target_n"]].mean()
    histrows = []
    for (day, model), g in hp.groupby(["day", "base_model"]):
        eligible_jobs = histjob[(histjob.day == day) & (histjob.base_model == model)]
        row = {"day": day, "base_model": model, "jobs": len(eligible_jobs),
               "random_draws": 20 * len(eligible_jobs), "draws_per_job": 20,
               "target_enrollments_across_pairs": int(round(g.target_n.sum())),
               "target_n_min": int(g.target_n.min()), "target_n_max": int(g.target_n.max())}
        row.update(bootstrap(g, ["G_T_same_course_previous", "G_T_random_other_source", "same_minus_random"],
                             20261001 + 100 * day + (model == "lightgbm")))
        histrows.append(row)
    save(pd.DataFrame(histrows), "historical_source_control_intervals.csv")
    save(hp, "historical_source_control_pair_mean.csv")

    structure = []
    for (day, model), g in pair.groupby(["day", "base_model"]):
        for kind in ["intercept", "slope", "Platt"]:
            renamed = g[PAIR + [f"G_T_{kind}", f"G_T_star_{kind}", f"A_{kind}"]].rename(
                columns={f"G_T_{kind}": "G_T", f"G_T_star_{kind}": "G_T_star", f"A_{kind}": "A"})
            row = {"day": day, "base_model": model, "calibrator": kind}
            row.update(bootstrap(renamed, ["G_T", "G_T_star", "A"],
                                 20261001 + 100 * day + (model == "lightgbm")))
            structure.append(row)
    save(pd.DataFrame(structure), "recalibration_structure_intervals.csv")
    diagcols = ["source_alpha", "target_alpha", "source_joint_alpha", "source_joint_beta",
                "target_joint_alpha", "target_joint_beta", "target_raw_ECE10",
                "target_source_Platt_ECE10", "target_oof_Platt_ECE10"]
    diag = []
    for (day, model), g in pair.groupby(["day", "base_model"]):
        for measure in diagcols:
            v = g[measure]
            diag.append({"day": day, "base_model": model, "pairs": len(g), "measure": measure,
                         "mean": v.mean(), "median": v.median(), "q25": v.quantile(.25), "q75": v.quantile(.75)})
    save(pd.DataFrame(diag), "calibration_diagnostics_summary.csv")
    save(pair[PAIR + ["seeds"] + diagcols], "calibration_diagnostics_pair_mean.csv")

    # Requested new experiment: fold-training prevalence, no aggregate shortcut.
    nullrows, maxerr, verified_folds = [], 0.0, 0
    for i, row in enumerate(jobs.itertuples(), 1):
        path = args.prediction_dir / f"{row.pair_id}_d{row.day}_{row.base_model}_s{row.seed}.parquet"
        f = pd.read_parquet(path)
        t = f.loc[f.population.eq("target_primary")].reset_index(drop=True)
        y = t.y.to_numpy(int)
        fold = t.fold.to_numpy(int)
        assert len(y) == row.target_n and len(np.unique(t.username)) == len(t)
        assert set(fold) == set(range(5))
        expected = np.full(len(y), -1)
        fs = 20261001 + int(row.pair_id[1:]) * 101 + row.day
        for k, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=fs).split(np.zeros(len(y)), y)):
            expected[test] = k
        assert np.array_equal(fold, expected), path
        null = np.full(len(y), np.nan)
        for k in range(5):
            fit, test = fold != k, fold == k
            assert fit.any() and test.any()
            null[test] = float(np.mean(y[fit]))
            verified_folds += 1
        raw = np.mean((y - t.raw.to_numpy(float)) ** 2)
        target = np.mean((y - t.oof_Platt.to_numpy(float)) ** 2)
        nullbs = np.mean((y - null) ** 2)
        maxerr = max(maxerr, abs(raw - row.target_raw_Brier),
                     abs(raw - target - row.G_T_star))
        nullrows.append({"pair_id": row.pair_id, "provider": row.provider,
                         "base_course_token": row.base_course_token,
                         "day": row.day, "base_model": row.base_model, "seed": row.seed,
                         "target_n": len(y), "raw_Brier": raw, "target_Platt_Brier": target,
                         "target_null_Brier": nullbs, "Platt_benefit_over_null": nullbs - target,
                         "G_T_star_null": raw - nullbs, "G_T_star_Platt": raw - target})
        if i % 200 == 0:
            print(f"Audited {i}/{len(jobs)} prediction files", flush=True)
    assert maxerr < 1e-12
    nj = pd.DataFrame(nullrows)
    nm = ["target_n", "raw_Brier", "target_Platt_Brier", "target_null_Brier", "Platt_benefit_over_null", "G_T_star_null", "G_T_star_Platt"]
    npair = nj.groupby(PAIR, as_index=False)[nm].mean()
    nullsummary = []
    for (day, model), g in npair.groupby(["day", "base_model"]):
        row = {"day": day, "base_model": model,
               "jobs": int(((nj.day == day) & (nj.base_model == model)).sum()),
               "target_enrollments_across_pairs": int(round(g.target_n.sum())),
               "positive_Platt_benefit_pairs": int(g.Platt_benefit_over_null.gt(0).sum()),
               "positive_Platt_benefit_fraction": float(g.Platt_benefit_over_null.gt(0).mean())}
        row.update(bootstrap(g, nm[1:], 20261001 + 100 * day + (model == "lightgbm")))
        nullsummary.append(row)
    save(nj, "target_prevalence_baseline_by_job.csv")
    save(npair, "target_prevalence_baseline_pair_mean.csv")
    save(pd.DataFrame(nullsummary), "target_prevalence_baseline_summary.csv")
    audit.update({"saved_target_folds_reproduced_exactly": True,
                  "folds_verified": verified_folds, "max_saved_metric_absolute_difference": maxerr,
                  "null_baseline_formula": "For each held-out fold, predict the mean outcome in the other four folds",
                  "aggregation": "Mean seeds within pair/day/model; then equal-weight pair mean; 2000 base-course-cluster bootstrap draws",
                  "original_truth_window_verified": False,
                  "public_output_contains_individual_ids_or_predictions": False,
                  "python": platform.python_version(),
                  "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm"]}})
    (args.out / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    print(pd.DataFrame(nullsummary)[["day", "base_model", "pairs", "target_null_Brier", "target_Platt_Brier", "Platt_benefit_over_null", "Platt_benefit_over_null_ci_low", "Platt_benefit_over_null_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
