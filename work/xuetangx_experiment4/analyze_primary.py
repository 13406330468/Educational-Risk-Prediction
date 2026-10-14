"""Summarize frozen primary effects with paired and course-cluster uncertainty."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from run_integrity import validate_job_coverage


ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
WORK = ROOT / "work" / "xuetangx_experiment4"
PRED = WORK / "predictions"
OUT = ROOT / "outputs" / "xuetangx_experiment4"
PROTOCOL = json.loads((OUT / "experiment4_protocol.json").read_text(encoding="utf-8"))

metrics = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in sorted(PRED.glob("*.json"))])
expected = 2 * sum(x["admissible_pair_seeds"] for x in PROTOCOL["unit_support_summary"].values())
assert len(metrics) == expected, (len(metrics), expected)
assert metrics[["pair_id", "day", "seed", "base_model"]].duplicated().sum() == 0
validate_job_coverage(metrics, pd.read_csv(OUT / 'unit_support.csv'), label='primary summary')
metrics.to_csv(OUT / "primary_by_seed.csv", index=False)

group_keys = ["pair_id", "provider", "base_course_token", "source_course_id", "target_course_id", "day", "base_model"]
numeric = metrics.select_dtypes(include="number").columns.difference(["day", "seed"])
pair = metrics.groupby(group_keys, as_index=False)[list(numeric)].mean()
pair["seeds"] = metrics.groupby(group_keys).size().to_numpy()
pair.to_csv(OUT / "primary_pair_mean.csv", index=False)

boot_rows = []
summary_rows = []
for (day, base), g in pair.groupby(["day", "base_model"]):
    clusters = [h for _, h in g.groupby(["provider", "base_course_token"])]
    rng = np.random.default_rng(20261001 + 100 * day + (0 if base == "logistic" else 1))
    draws = []
    for _ in range(2000):
        idx = rng.integers(0, len(clusters), size=len(clusters))
        sample = pd.concat([clusters[i] for i in idx], ignore_index=True)
        draws.append(sample[["G_T", "G_T_star", "A"]].mean().to_numpy())
    draws = np.asarray(draws)
    for j, measure in enumerate(("G_T", "G_T_star", "A")):
        boot_rows.append({"day": day, "base_model": base, "measure": measure,
                          "macro_pair_mean": float(g[measure].mean()),
                          "cluster_ci_2_5": float(np.quantile(draws[:, j], .025)),
                          "cluster_ci_97_5": float(np.quantile(draws[:, j], .975)),
                          "base_course_clusters": len(clusters), "pairs": len(g)})
    summary_rows.append({
        "day": day, "base_model": base, "pairs": len(g),
        "base_course_clusters": len(clusters), "pair_seed_jobs": int(metrics[metrics.day.eq(day) & metrics.base_model.eq(base)].shape[0]),
        "G_S_macro": float(g.G_S.mean()), "G_T_macro": float(g.G_T.mean()),
        "G_T_star_macro": float(g.G_T_star.mean()), "A_macro": float(g.A.mean()),
        "A_pair_median": float(g.A.median()), "A_pair_positive_fraction": float(g.A.gt(0).mean()),
        "A_pair_ge_0_001_fraction": float(g.A.ge(.001).mean()),
        "G_T_micro_approx": float(np.average(g.G_T, weights=g.target_n)),
        "A_micro_approx": float(np.average(g.A, weights=g.target_n)),
        "G_T_intercept_macro": float(g.G_T_intercept.mean()),
        "G_T_star_intercept_macro": float(g.G_T_star_intercept.mean()),
        "G_T_slope_macro": float(g.G_T_slope.mean()),
        "G_T_star_slope_macro": float(g.G_T_star_slope.mean()),
        "target_extra_slope_gain_macro": float(g.target_extra_slope_gain.mean()),
        "G_T_Log_loss_macro": float(g.G_T_Platt_Log_loss.mean()),
        "G_T_star_Log_loss_macro": float(g.G_T_star_Platt_Log_loss.mean()),
    })
pd.DataFrame(boot_rows).to_csv(OUT / "course_cluster_bootstrap.csv", index=False)
pd.DataFrame(summary_rows).sort_values(["day", "base_model"]).to_csv(OUT / "primary_time_model_summary.csv", index=False)

# Retrospective and source-observable associations are kept separate.
moderators = ["observation_gap_days", "risk_wasserstein", "risk_ks", "alpha_drift", "beta_drift",
              "prevalence_shift", "source_cal_n", "source_prevalence", "target_prevalence",
              "source_raw_risk_mean", "target_raw_risk_mean", "target_n"]
mod_rows = []
for (day, base), g in pair.groupby(["day", "base_model"]):
    for moderator in moderators:
        valid = g[[moderator, "A", "G_T"]].dropna()
        if len(valid) < 3:
            continue
        for outcome in ("A", "G_T"):
            r, p = spearmanr(valid[moderator], valid[outcome])
            mod_rows.append({"day": day, "base_model": base, "moderator": moderator,
                             "outcome": outcome, "n_pairs": len(valid), "spearman_r": r,
                             "unadjusted_p_descriptive": p,
                             "target_labels_required_for_moderator": moderator in ("alpha_drift", "beta_drift", "prevalence_shift", "target_prevalence")})
pd.DataFrame(mod_rows).to_csv(OUT / "mechanism_correlations.csv", index=False)

student_rows = []
for i, (key, g) in enumerate(metrics.groupby(["pair_id", "day", "base_model"]), 1):
    frames = []
    for row in g.itertuples():
        f = pd.read_parquet(PRED / f"{row.pair_id}_d{row.day}_{row.base_model}_s{row.seed}.parquet")
        f = f[f.population.eq("target_primary")]
        y = f.y.to_numpy(dtype=float)
        raw = f.raw.to_numpy(dtype=float)
        transferred = f.source_Platt.to_numpy(dtype=float)
        target_oof = f.oof_Platt.to_numpy(dtype=float)
        frames.append(pd.DataFrame({"enroll_id": f.enroll_id.to_numpy(),
                                    "GT_loss_difference": (y - raw) ** 2 - (y - transferred) ** 2,
                                    "A_loss_difference": (y - transferred) ** 2 - (y - target_oof) ** 2}))
    together = pd.concat(frames, ignore_index=True).groupby("enroll_id", as_index=False).mean()
    # Every seed evaluates the same target enrollment risk set within a pair/day.
    assert len(together) == int(g.target_n.iloc[0])
    values = together[["GT_loss_difference", "A_loss_difference"]].to_numpy()
    rng = np.random.default_rng(87231 + int(key[0][1:]) * 37 + key[1] + (key[2] == "lightgbm"))
    draws = np.empty((2000, 2), dtype=float)
    for start in range(0, 2000, 100):
        idx = rng.integers(0, len(values), size=(100, len(values)))
        draws[start:start+100] = values[idx].mean(axis=1)
    student_rows.append({"pair_id": key[0], "day": key[1], "base_model": key[2],
                         "n_unique_target_users": len(values), "seeds": len(g),
                         "G_T": float(values[:, 0].mean()),
                         "G_T_ci_2_5": float(np.quantile(draws[:, 0], .025)),
                         "G_T_ci_97_5": float(np.quantile(draws[:, 0], .975)),
                         "A": float(values[:, 1].mean()),
                         "A_ci_2_5": float(np.quantile(draws[:, 1], .025)),
                         "A_ci_97_5": float(np.quantile(draws[:, 1], .975))})
    if i % 50 == 0:
        print(f"student bootstrap {i}/{pair.groupby(['pair_id','day','base_model']).ngroups}", flush=True)
student = pd.DataFrame(student_rows)
student.to_csv(OUT / "student_bootstrap_by_pair.csv", index=False)
check = student.merge(pair[["pair_id", "day", "base_model", "G_T", "A"]], on=["pair_id", "day", "base_model"], suffixes=("_student", "_pair"))
assert np.max(abs(check.G_T_student - check.G_T_pair)) < 1e-10
assert np.max(abs(check.A_student - check.A_pair)) < 1e-10
print(json.dumps({"seed_rows": len(metrics), "pair_units": len(pair), "student_bootstrap_units": len(student),
                  "course_clusters": int(pair[["provider", "base_course_token"]].drop_duplicates().shape[0])}, indent=2))
