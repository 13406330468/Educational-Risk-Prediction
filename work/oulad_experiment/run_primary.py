"""Rebuild final OULAD calibration results without running the DFMC pilot.

Prerequisite: prepare.py must build Days 7, 14, 28 and 56, even for --smoke,
because the original source ID partition was constructed across all landmarks.
Only run_experiment.py, dfmc.py and protocol.json are imported from --code-dir.
Raw rows, predictions and learner IDs remain under the private --work-dir.
Public-compatible aggregate tables go under --results-dir.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from scipy.stats import wasserstein_distance
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

KEY = ["code_module", "code_presentation", "id_student", "t"]
CELL = ["season", "course", "day", "base"]


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)


def brier(y, p):
    return float(np.mean((np.asarray(y) - clip(p)) ** 2))


def log_loss(y, p):
    p = clip(p)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1-y) * np.log1p(-p)))


def intercept_fit(p, y):
    z = logit(clip(p))
    prevalence = float(np.mean(y))
    assert 0 < prevalence < 1
    return float(brentq(lambda b: float(expit(z+b).mean()-prevalence), -30, 30))


def intercept_predict(p, b):
    return expit(logit(clip(p)) + b)


def calibration_line(p, y):
    # Identical to experiment1.calibration_line, including its weak penalty.
    x = np.column_stack([np.ones(len(p)), logit(clip(p))])
    y = np.asarray(y, dtype=float)
    initial = np.array([0., 1.])
    def objective(w):
        z = x @ w
        penalty = 1e-4 * np.sum((w-initial)**2)
        gradient = x.T @ (expit(z)-y)/len(y) + 2e-4*(w-initial)
        return np.mean(np.logaddexp(0, z)-y*z) + penalty, gradient
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12})
    assert result.success and np.isfinite(result.fun)
    return float(result.x[0]), float(result.x[1])


def crossfit(frame, calibrator, fold_seed):
    p, y, ids = frame.raw.to_numpy(), frame.y.to_numpy(), frame.id_student.to_numpy()
    platt, intercept = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    folds = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for fold, (fit, test) in enumerate(splitter.split(p.reshape(-1, 1), y, groups=ids)):
        assert len(np.unique(y[fit])) == len(np.unique(y[test])) == 2
        assert not set(ids[fit]) & set(ids[test])
        platt[test] = calibrator("Platt").fit(p[fit], y[fit]).predict(p[test])
        intercept[test] = intercept_predict(p[test], intercept_fit(p[fit], y[fit]))
        folds[test] = fold
    assert np.isfinite(platt).all() and np.isfinite(intercept).all()
    return platt, intercept, folds


def matched_target(source, target, season, day, base, seed, calibrator):
    rows = []
    nsource = int(source.id_student.nunique())
    for repeat in range(3):
        pp, pi = np.full(len(target), np.nan), np.full(len(target), np.nan)
        fit_counts = []
        for fold in range(5):
            tr = target[target.fold.ne(fold)]
            te = target.fold.eq(fold).to_numpy()
            eligible = tr.id_student.unique()
            nfit = min(nsource, len(eligible))
            rng = np.random.default_rng(20261001 + 100000 * (season == "B") +
                                        10000 * (base == "logistic") + 1000 * day +
                                        10 * seed + 3 * fold + repeat)
            selected = rng.choice(eligible, size=nfit, replace=False)
            fit = tr[tr.id_student.isin(selected)]
            assert not set(fit.id_student) & set(target.loc[te, "id_student"])
            pp[te] = calibrator("Platt").fit(fit.raw, fit.y).predict(target.loc[te, "raw"])
            pi[te] = intercept_predict(target.loc[te, "raw"], intercept_fit(fit.raw, fit.y))
            fit_counts.append(nfit)
        for course, index in target.groupby("code_module").indices.items():
            f = target.iloc[index]
            gp, gi = brier(f.y, f.raw)-brier(f.y, pp[index]), brier(f.y, f.raw)-brier(f.y, pi[index])
            rows.append(dict(season=season, course=course, day=day, base=base, seed=seed,
                             repeat=repeat, source_fit_unique_ids=nsource,
                             target_fit_unique_ids_min=min(fit_counts),
                             target_fit_unique_ids_max=max(fit_counts), target_n=len(f),
                             G_T_star_intercept_matched=gi,
                             matched_target_intercept_Brier=brier(f.y, pi[index]),
                             G_T_star_matched=gp, slope_increment_matched=gp-gi))
    return rows


def one_job(args, trainer, season, day, base, seed, source, target, split, features, modules):
    src = source[source.t.eq(day)]
    tgt = target[target.t.eq(day)].reset_index(drop=True)
    train = src[src.id_student.isin(split["train"])]
    val = src[src.id_student.isin(split["base_validation"])]
    cal = src[src.id_student.isin(split["calibration_order"])].reset_index(drop=True)
    assert not set(src.id_student) & set(tgt.id_student)
    pc, pt, metadata = trainer.train_base(train, val, cal, tgt, features, modules, base, seed)
    cache = args.work_dir / "results_verified" / season
    stem = f"{base}_s{seed}_t{day}"
    np.savez_compressed(cache / (stem+"_base.npz"), pc=pc, pt=pt,
                        cal_keys=cal[KEY].astype(str).to_numpy(dtype=str),
                        test_keys=tgt[KEY].astype(str).to_numpy(dtype=str))
    metadata.update(n_train=len(train), n_validation=len(val), n_calibration=len(cal), n_test=len(tgt))
    (cache / (stem+"_base.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    source_pred = cal[KEY + ["y", "progress_state"]].copy()
    target_pred = tgt[KEY + ["y", "progress_state"]].copy()
    source_pred["raw"], target_pred["raw"] = pc, pt
    source_pred["population"], target_pred["population"] = "source_calibration", "target"
    source_pred["Platt_oof"], source_pred["intercept_source_oof"], source_pred["fold"] = crossfit(
        source_pred, trainer.ParametricCalibrator, 1000+seed+day)
    target_pred["Platt_target_oof"], target_pred["intercept_target_oof"], target_pred["fold"] = crossfit(
        target_pred, trainer.ParametricCalibrator, 20261001+day+(100 if season == "B" else 0))
    source_model = trainer.ParametricCalibrator("Platt").fit(pc, cal.y.to_numpy())
    source_b = intercept_fit(pc, cal.y.to_numpy())
    target_pred["Platt_source_deployed"] = source_model.predict(pt)
    target_pred["intercept_source_transfer"] = intercept_predict(pt, source_b)
    pred = pd.concat([source_pred, target_pred], ignore_index=True)
    pred["season"], pred["base"], pred["seed"] = season, base, seed
    jobdir = args.work_dir / "experiment3_work"
    pred.to_parquet(jobdir / f"{season}_{stem}.parquet", index=False)
    rows = []
    for course in modules:
        s = source_pred[source_pred.code_module.eq(course)]
        t = target_pred[target_pred.code_module.eq(course)]
        sr, tr = brier(s.y, s.raw), brier(t.y, t.raw)
        gs_i = sr-brier(s.y, s.intercept_source_oof)
        gs_p = sr-brier(s.y, s.Platt_oof)
        gt_i = tr-brier(t.y, t.intercept_source_transfer)
        gt_p = tr-brier(t.y, t.Platt_source_deployed)
        gs_ti = tr-brier(t.y, t.intercept_target_oof)
        gs_tp = tr-brier(t.y, t.Platt_target_oof)
        sa, sb = calibration_line(s.raw, s.y)
        a, b = s.progress_state.value_counts(normalize=True), t.progress_state.value_counts(normalize=True)
        rows.append(dict(season=season, course=course, day=day, base=base, seed=seed,
                         source_cal_n=len(s), source_cal_unique_ids=int(s.id_student.nunique()),
                         target_n=len(t), source_full_intercept=source_b,
                         source_raw_risk_mean=float(s.raw.mean()), source_raw_risk_variance=float(s.raw.var(ddof=0)),
                         target_raw_risk_mean=float(t.raw.mean()), source_raw_Brier=sr, target_raw_Brier=tr,
                         G_S_intercept=gs_i, G_S_Platt=gs_p, G_T_intercept=gt_i, G_T_Platt=gt_p,
                         G_T_star_intercept=gs_ti, G_T_star_Platt=gs_tp, A_intercept=gs_ti-gt_i, A_Platt=gs_tp-gt_p,
                         target_slope_increment=gs_tp-gs_ti, source_slope_increment=gs_p-gs_i,
                         source_raw_Log_loss=log_loss(s.y, s.raw), source_intercept_OOF_Log_loss=log_loss(s.y, s.intercept_source_oof),
                         source_Platt_OOF_Log_loss=log_loss(s.y, s.Platt_oof), target_raw_Log_loss=log_loss(t.y, t.raw),
                         target_source_intercept_Log_loss=log_loss(t.y, t.intercept_source_transfer),
                         target_source_Platt_Log_loss=log_loss(t.y, t.Platt_source_deployed),
                         target_intercept_OOF_Log_loss=log_loss(t.y, t.intercept_target_oof),
                         target_Platt_OOF_Log_loss=log_loss(t.y, t.Platt_target_oof),
                         source_CITL_alpha=intercept_fit(s.raw, s.y), source_joint_beta=sb,
                         risk_wasserstein=float(wasserstein_distance(s.raw, t.raw)),
                         progress_TV=float(.5*sum(abs(a.get(g, 0)-b.get(g, 0)) for g in ["no_due", "due_complete", "due_missing"]))))
    matched = matched_target(source_pred, target_pred, season, day, base, seed, trainer.ParametricCalibrator)
    pd.DataFrame(rows).to_csv(jobdir / f"{season}_{stem}.csv", index=False)
    pd.DataFrame(matched).to_csv(jobdir / f"{season}_{stem}_matched.csv", index=False)
    print(f"{season} day={day} model={base} seed={seed}: completed", flush=True)
    return rows, matched


def aggregate(rows, matched, out):
    scores, matched = pd.DataFrame(rows), pd.DataFrame(matched)
    scores.to_csv(out / "decomposition_by_seed.csv", index=False)
    matched.to_csv(out / "matched_size_intercept.csv", index=False)
    numerical = scores.select_dtypes("number").columns.drop(["day", "seed"])
    cell = scores.groupby(CELL)[numerical].mean().reset_index()
    checks = scores.groupby(CELL).agg(
        target_Platt_positive_seeds=("G_T_star_Platt", lambda z: int((z>0).sum())),
        target_intercept_positive_seeds=("G_T_star_intercept", lambda z: int((z>0).sum())),
        slope_increment_positive_seeds=("target_slope_increment", lambda z: int((z>0).sum()))).reset_index()
    cell = cell.merge(checks, on=CELL, validate="one_to_one")
    cell["ratio_eligible"] = (cell.G_T_star_Platt >= .001) & (cell.target_Platt_positive_seeds >= 4)
    cell["R_int"] = np.where(cell.ratio_eligible, cell.G_T_star_intercept/cell.G_T_star_Platt, np.nan)
    cell["mechanism_label"] = "unclassified"
    dominant = cell.ratio_eligible & (cell.R_int >= .8) & (cell.target_slope_increment <= .001) & (cell.target_intercept_positive_seeds >= 4)
    material = cell.ratio_eligible & (cell.R_int < .8) & (cell.target_slope_increment >= .001) & (cell.slope_increment_positive_seeds >= 4)
    cell.loc[dominant, "mechanism_label"] = "intercept_dominant"
    cell.loc[material, "mechanism_label"] = "material_slope"
    cell.loc[cell.ratio_eligible & ~(dominant | material), "mechanism_label"] = "mixed_or_indeterminate"
    cell.to_csv(out / "adaptation_by_cell.csv", index=False)
    cols = ["G_S_intercept", "G_S_Platt", "G_T_intercept", "G_T_Platt", "G_T_star_intercept",
            "G_T_star_Platt", "A_intercept", "A_Platt", "target_slope_increment"]
    for filename, groups in [("time_model_summary", ["day", "base"]), ("course_model_summary", ["season", "course", "base"]),
                             ("season_model_summary", ["season", "base"]), ("day_summary", ["day"])]:
        cell.groupby(groups)[cols].mean().reset_index().to_csv(out / (filename+".csv"), index=False)
    matched_cell = matched.groupby(CELL)[["G_T_star_intercept_matched", "G_T_star_matched", "slope_increment_matched"]].mean().reset_index()
    matched_cell.to_csv(out / "matched_size_by_cell.csv", index=False)
    summary = dict(model_jobs=int(len(scores[["season", "day", "base", "seed"]].drop_duplicates())),
                   units=len(cell), complete_primary_analysis=(len(scores)==360),
                   overall_Brier_gains={c: float(cell[c].mean()) for c in cols},
                   mechanism_counts=cell.mechanism_label.value_counts().to_dict(),
                   matched_target_gains={c: float(matched_cell[c].mean()) for c in ["G_T_star_intercept_matched", "G_T_star_matched", "slope_increment_matched"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(args):
    sys.path.insert(0, str(args.code_dir.resolve()))
    trainer = importlib.import_module("run_experiment")
    protocol = json.loads((args.code_dir / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["probability_clip"] == 1e-6
    data = pd.read_csv(args.work_dir / "data/features.csv.gz")
    assert set(data.t) == {7, 14, 28, 56}, "Build all four landmarks before fitting, even for --smoke."
    features = json.loads((args.work_dir / "data/data_audit.json").read_text(encoding="utf-8"))["feature_columns"]
    with zipfile.ZipFile(args.oulad_zip) as z:
        info = pd.read_csv(z.open("studentInfo.csv"), usecols=["id_student", "code_presentation"])
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "experiment3_work").mkdir(parents=True, exist_ok=True)
    # Audit inputs carry a local copy of the actually used public configuration.
    if args.work_dir.resolve() != args.code_dir.resolve():
        (args.work_dir / "protocol.json").write_bytes((args.code_dir / "protocol.json").read_bytes())
    rows, matched, exploration = [], [], set()
    for season in ["J", "B"]:
        spec = protocol["exploration" if season == "J" else "confirmation"]
        source = data[data.code_presentation.eq(spec["source"]) & data.code_module.isin(spec["modules"])].copy()
        target = data[data.code_presentation.eq(spec["target"]) & data.code_module.isin(spec["modules"])].copy()
        seen = set(info.loc[info.code_presentation.eq(spec["source"]), "id_student"])
        target = target[~target.id_student.isin(seen)]
        if season == "J":
            exploration = set(source.id_student) | set(target.id_student)
        else:
            target = target[~target.id_student.isin(exploration)]
        cache = args.work_dir / "results_verified" / season
        cache.mkdir(parents=True, exist_ok=True)
        if season == "J":
            (cache / "exploration_ids.json").write_text(json.dumps(sorted(int(x) for x in exploration)), encoding="utf-8")
        if args.smoke and season == "B":
            continue
        for seed in ([17] if args.smoke else protocol["seeds"]):
            split = trainer.make_splits(source, seed, cache)
            for day in ([7] if args.smoke else protocol["times"]):
                for base in (["logistic"] if args.smoke else protocol["base_models"]):
                    with threadpool_limits(limits=2):
                        a, b = one_job(args, trainer, season, day, base, seed, source, target, split, features, spec["modules"])
                    rows.extend(a)
                    matched.extend(b)
    aggregate(rows, matched, args.results_dir)
    print(json.dumps({"aggregate_output": str(args.results_dir), "model_jobs": 1 if args.smoke else 80,
                      "smoke_only": args.smoke, "private_work_directory": str(args.work_dir)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--oulad-zip", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
