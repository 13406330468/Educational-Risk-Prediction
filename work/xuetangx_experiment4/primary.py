"""Frozen primary XuetangX same-course temporal calibration experiment."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.special import expit, logit
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
WORK = ROOT / "work" / "xuetangx_experiment4"
OUT = ROOT / "outputs" / "xuetangx_experiment4"
PRED = WORK / "predictions"
PRED.mkdir(exist_ok=True)
PROTOCOL = json.loads((OUT / "experiment4_protocol.json").read_text(encoding="utf-8"))
PAIRS = pd.read_csv(OUT / "eligible_pairs.csv")
SUPPORT = pd.read_csv(OUT / "unit_support.csv")
ROLES = pd.read_parquet(WORK / "source_roles.parquet")
ROLES["enroll_id"] = ROLES.enroll_id.astype("int64")
ENROLL = pd.read_csv(ROOT / "work" / "xuetangx_audit" / "enrollment_index.csv",
                     dtype={"enroll_id": "int64", "username": "int64", "course_id": str})
ENROLL = ENROLL[["enroll_id", "username", "course_id", "truth"]].rename(columns={"truth": "y"})
ENROLL["y"] = ENROLL.y.astype(int)
BY_COURSE = {k: g for k, g in ENROLL.groupby("course_id")}
DAY_FRAMES = {d: pd.read_parquet(WORK / f"features_day{d}.parquet") for d in (7, 14, 21, 28)}
DAY_BY_COURSE = {d: {k: g for k, g in f.groupby("course_id")} for d, f in DAY_FRAMES.items()}
ACTION_NAMES = json.loads((OUT / "feature_manifest.json").read_text())["actions"]
ACTION_COUNT_COLS = [f"action_{a}_count" for a in ACTION_NAMES]
NON_ACTION = [
    "event_count", "active_days", "unique_sessions", "unique_objects",
    "events_per_active_day", "sessions_per_active_day",
    "first_activity_lag", "last_activity_lag", "activity_span", "recency_to_landmark",
    "active_day_ratio", "events_first_half", "events_second_half",
    "event_second_half_ratio", "active_days_first_half", "active_days_second_half",
    "max_events_per_day", "mean_events_per_active_day", "std_events_per_active_day",
    "weekend_event_fraction", "unique_objects_per_event",
]
FEATURES = NON_ACTION + ACTION_COUNT_COLS + [f"action_{a}_fraction" for a in ACTION_NAMES] + ["action_OTHER_count", "action_OTHER_fraction"]
EPS = 1e-6


def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def brier(y, p):
    return float(np.mean((np.asarray(y) - clip(p)) ** 2))


def logloss(y, p):
    p = clip(p)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_cal(p, y, kind):
    p = clip(p)
    y = np.asarray(y, dtype=float)
    z = logit(p)
    if len(np.unique(y)) < 2:
        raise ValueError("calibration training lacks a class")
    if kind == "intercept":
        b = brentq(lambda x: np.mean(expit(z + x)) - np.mean(y), -30, 30)
        return np.array([1., b])
    if kind == "slope":
        res = minimize_scalar(lambda a: np.mean(np.logaddexp(0, a * z) - y * a * z)
                              + 1e-5 * (a - 1) ** 2,
                              bounds=(0, 30), method="bounded")
        if not res.success:
            raise ValueError("slope fit failed")
        return np.array([float(res.x), 0.])
    def objective(w):
        eta = w[0] * z + w[1]
        diff = expit(eta) - y
        penalty = 1e-5 * ((w[0] - 1) ** 2 + w[1] ** 2)
        return (np.mean(np.logaddexp(0, eta) - y * eta) + penalty,
                np.array([np.mean(diff * z) + 2e-5 * (w[0] - 1),
                          np.mean(diff) + 2e-5 * w[1]]))
    res = minimize(objective, [1., 0.], method="L-BFGS-B", jac=True,
                   bounds=[(0, 30), (-30, 30)], options={"maxiter": 300, "ftol": 1e-10})
    if not np.isfinite(res.fun):
        raise ValueError("Platt fit nonfinite")
    return np.asarray(res.x)


def apply_cal(p, params):
    return clip(expit(params[0] * logit(clip(p)) + params[1]))


def crossfit(p, y, usernames, kind, fold_seed):
    y = np.asarray(y, dtype=int)
    names = np.asarray(usernames)
    assert len(names) == len(np.unique(names))  # one enrollment per user within a presentation
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    result = np.empty(len(y), dtype=float)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (fit, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        if len(np.unique(y[fit])) < 2 or len(np.unique(y[test])) < 2:
            raise ValueError("crossfit fold lacks class")
        params = fit_cal(p[fit], y[fit], kind)
        result[test] = apply_cal(p[test], params)
        folds[test] = fold
    return result, folds


def ece_quantile(y, p):
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, 10)
    return float(sum(len(idx) / len(y) * abs(np.mean(y[idx]) - np.mean(p[idx])) for idx in bins if len(idx)))


def metrics(y, p):
    p = clip(p)
    y = np.asarray(y)
    return {
        "Brier": brier(y, p), "Log_loss": logloss(y, p),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "AP": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "ECE10_quantile": ece_quantile(y, p),
        "signed_calibration_bias": float(np.mean(p - y)),
        "absolute_calibration_bias": float(abs(np.mean(p - y))),
    }


def prep_pair(pair, day):
    source_course = pair.source_course_id
    target_course = pair.target_course_id
    source = BY_COURSE[source_course].merge(DAY_BY_COURSE[day].get(source_course), on=["enroll_id", "course_id"], how="inner", validate="one_to_one")
    target = BY_COURSE[target_course].merge(DAY_BY_COURSE[day].get(target_course), on=["enroll_id", "course_id"], how="left", validate="one_to_one")
    source_ids = set(BY_COURSE[source_course].username)
    target["unseen"] = ~target.username.isin(source_ids)
    target["active"] = target.event_count.notna()
    assert source.event_count.notna().all()
    assert len(source) and len(target)
    for frame in (source, target):
        for col in NON_ACTION + ACTION_COUNT_COLS:
            if col in ("first_activity_lag", "recency_to_landmark"):
                frame[col] = frame[col].fillna(day)
            else:
                frame[col] = frame[col].fillna(0)
        for action, count_col in zip(ACTION_NAMES, ACTION_COUNT_COLS):
            frame[f"action_{action}_fraction"] = np.where(frame.event_count.gt(0), frame[count_col] / frame.event_count, 0)
    return source, target


def source_vocabulary(source, role):
    train = source.loc[role == "train"]
    return [col for col in ACTION_COUNT_COLS if train[col].sum() > 0]


def apply_vocabulary(frame, vocabulary):
    frame = frame.copy()
    other = [c for c in ACTION_COUNT_COLS if c not in vocabulary]
    frame["action_OTHER_count"] = frame[other].sum(axis=1) if other else 0
    frame["action_OTHER_fraction"] = np.where(frame.event_count.gt(0), frame.action_OTHER_count / frame.event_count, 0)
    for col in other:
        frame[col] = 0
        frame[col.replace("_count", "_fraction")] = 0
    return frame


def fit_base(source, target, roles, base, seed):
    train = source.loc[roles == "train"]
    valid = source.loc[roles == "validation"]
    cal = source.loc[roles == "calibration"]
    x_train = train[FEATURES].to_numpy(dtype=float)
    x_valid = valid[FEATURES].to_numpy(dtype=float)
    x_cal = cal[FEATURES].to_numpy(dtype=float)
    x_target = target[FEATURES].to_numpy(dtype=float)
    assert np.isfinite(x_train).all() and np.isfinite(x_target).all()
    if base == "logistic":
        x_train = np.sign(x_train) * np.log1p(abs(x_train))
        x_valid = np.sign(x_valid) * np.log1p(abs(x_valid))
        x_cal = np.sign(x_cal) * np.log1p(abs(x_cal))
        x_target = np.sign(x_target) * np.log1p(abs(x_target))
        scaler = StandardScaler().fit(x_train)
        x_train, x_valid, x_cal, x_target = [scaler.transform(x) for x in (x_train, x_valid, x_cal, x_target)]
        model = LogisticRegression(C=1., max_iter=800, solver="lbfgs", random_state=seed)
        with threadpool_limits(limits=1):
            model.fit(x_train, train.y)
        best_iter = None
    else:
        model = LGBMClassifier(n_estimators=180, learning_rate=.05, num_leaves=15,
                               max_depth=5, min_child_samples=20, reg_lambda=1.,
                               verbosity=-1, n_jobs=1, random_state=seed,
                               deterministic=True, force_col_wise=True)
        model.fit(x_train, train.y, eval_X=x_valid, eval_y=valid.y,
                  eval_metric="binary_logloss", callbacks=[early_stopping(20, verbose=False)])
        best_iter = int(model.best_iteration_)
    pc = clip(model.predict_proba(x_cal)[:, 1])
    pt = clip(model.predict_proba(x_target)[:, 1])
    pv = clip(model.predict_proba(x_valid)[:, 1])
    return cal, pc, pt, {"validation_Brier": brier(valid.y, pv), "best_iteration": best_iter}


def one_job(pair, day, seed, base, source_original, target_original, role_map, riskset="active_unseen"):
    stem = f"{pair.pair_id}_d{day}_{base}_s{seed}"
    metric_path = PRED / f"{stem}.json"
    pred_path = PRED / f"{stem}.parquet"
    if metric_path.exists() and pred_path.exists():
        return json.loads(metric_path.read_text(encoding="utf-8"))
    role = source_original.enroll_id.map(role_map)
    vocabulary = source_vocabulary(source_original, role)
    source = apply_vocabulary(source_original, vocabulary)
    target = apply_vocabulary(target_original, vocabulary)
    cal, pc, pt_all, base_info = fit_base(source, target, role, base, seed)
    target = target.copy()
    target["raw"] = pt_all
    cal = cal.copy()
    cal["raw"] = pc
    eval_mask = target.unseen if riskset == "all_unseen" else (target.active & target.unseen)
    tgt = target.loc[eval_mask].copy()
    assert len(tgt) >= 50
    source_fit = {kind: fit_cal(pc, cal.y.to_numpy(), kind) for kind in ("intercept", "slope", "Platt")}
    source_oof = {}
    target_oof = {}
    folds = None
    for kind in ("intercept", "slope", "Platt"):
        source_oof[kind], _ = crossfit(pc, cal.y.to_numpy(), cal.username.to_numpy(), kind, 1000 + seed + day)
        target_oof[kind], folds = crossfit(tgt.raw.to_numpy(), tgt.y.to_numpy(), tgt.username.to_numpy(),
                                            kind, 20261001 + int(pair.pair_id[1:]) * 101 + day)
        target[f"source_{kind}"] = apply_cal(target.raw.to_numpy(), source_fit[kind])
        cal[f"oof_{kind}"] = source_oof[kind]
    tgt = target.loc[eval_mask].copy()
    for kind in ("intercept", "slope", "Platt"):
        tgt[f"oof_{kind}"] = target_oof[kind]
    tgt["fold"] = folds
    raw_source = metrics(cal.y, pc)
    raw_target = metrics(tgt.y, tgt.raw)
    result = {
        "pair_id": pair.pair_id, "provider": pair.provider, "base_course_token": pair.base_course_token,
        "source_course_id": pair.source_course_id, "target_course_id": pair.target_course_id,
        "day": day, "seed": seed, "base_model": base,
        "source_cal_n": len(cal), "source_cal_0": int(cal.y.eq(0).sum()), "source_cal_1": int(cal.y.eq(1).sum()),
        "target_n": len(tgt), "target_0": int(tgt.y.eq(0).sum()), "target_1": int(tgt.y.eq(1).sum()),
        "target_all_n": len(target), "target_returning_n": int((target.active & ~target.unseen).sum()),
        "source_prevalence": float(cal.y.mean()), "target_prevalence": float(tgt.y.mean()),
        "source_raw_risk_mean": float(np.mean(pc)), "target_raw_risk_mean": float(tgt.raw.mean()),
        "source_raw_risk_variance": float(np.var(pc)), "target_raw_risk_variance": float(tgt.raw.var(ddof=0)),
        "risk_wasserstein": float(wasserstein_distance(pc, tgt.raw)),
        "risk_ks": float(ks_2samp(pc, tgt.raw).statistic),
        "prevalence_shift": float(tgt.y.mean() - cal.y.mean()),
        "observation_gap_days": float(pair.observation_gap_days),
        "base_validation_Brier": base_info["validation_Brier"],
        "base_best_iteration": base_info["best_iteration"],
        "source_raw_Brier": raw_source["Brier"], "target_raw_Brier": raw_target["Brier"],
        "source_raw_Log_loss": raw_source["Log_loss"], "target_raw_Log_loss": raw_target["Log_loss"],
        "target_raw_AUC": raw_target["AUC"], "target_raw_AP": raw_target["AP"],
        "target_raw_ECE10": raw_target["ECE10_quantile"],
        "source_alpha": float(fit_cal(pc, cal.y.to_numpy(), "intercept")[1]),
        "target_alpha": float(fit_cal(tgt.raw.to_numpy(), tgt.y.to_numpy(), "intercept")[1]),
    }
    source_line = fit_cal(pc, cal.y.to_numpy(), "Platt")
    target_line = fit_cal(tgt.raw.to_numpy(), tgt.y.to_numpy(), "Platt")
    result.update({"source_joint_alpha": float(source_line[1]), "source_joint_beta": float(source_line[0]),
                   "target_joint_alpha": float(target_line[1]), "target_joint_beta": float(target_line[0])})
    result["alpha_drift"] = abs(result["target_alpha"] - result["source_alpha"])
    result["beta_drift"] = abs(result["target_joint_beta"] - result["source_joint_beta"])
    for kind in ("intercept", "slope", "Platt"):
        sm = metrics(cal.y, source_oof[kind])
        tm = metrics(tgt.y, tgt[f"source_{kind}"])
        om = metrics(tgt.y, tgt[f"oof_{kind}"])
        result[f"G_S_{kind}"] = raw_source["Brier"] - sm["Brier"]
        result[f"G_T_{kind}"] = raw_target["Brier"] - tm["Brier"]
        result[f"G_T_star_{kind}"] = raw_target["Brier"] - om["Brier"]
        result[f"A_{kind}"] = result[f"G_T_star_{kind}"] - result[f"G_T_{kind}"]
        result[f"G_T_{kind}_Log_loss"] = raw_target["Log_loss"] - tm["Log_loss"]
        result[f"G_T_star_{kind}_Log_loss"] = raw_target["Log_loss"] - om["Log_loss"]
        result[f"target_source_{kind}_ECE10"] = tm["ECE10_quantile"]
        result[f"target_oof_{kind}_ECE10"] = om["ECE10_quantile"]
        result[f"source_{kind}_a"] = float(source_fit[kind][0])
        result[f"source_{kind}_b"] = float(source_fit[kind][1])
    result["target_extra_slope_gain"] = result["G_T_star_Platt"] - result["G_T_star_intercept"]
    result["G_S"] = result["G_S_Platt"]
    result["G_T"] = result["G_T_Platt"]
    result["G_T_star"] = result["G_T_star_Platt"]
    result["A"] = result["A_Platt"]
    # Save enrollment-level paired predictions for later clustered intervals and sensitivity.
    tgt_frame = tgt[["enroll_id", "username", "y", "raw", "fold", "source_intercept", "source_slope", "source_Platt",
                     "oof_intercept", "oof_slope", "oof_Platt"]].copy()
    tgt_frame["population"] = "target_primary"
    all_frame = target[["enroll_id", "username", "y", "raw", "source_intercept", "source_slope", "source_Platt", "active", "unseen"]].copy()
    all_frame["population"] = "target_all"
    cal_frame = cal[["enroll_id", "username", "y", "raw", "oof_intercept", "oof_slope", "oof_Platt"]].copy()
    cal_frame["population"] = "source_calibration"
    predictions = pd.concat([tgt_frame, all_frame, cal_frame], ignore_index=True)
    predictions["pair_id"] = pair.pair_id
    predictions["day"] = day
    predictions["seed"] = seed
    predictions["base_model"] = base
    predictions.to_parquet(pred_path, index=False, compression="zstd")
    metric_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    args = sys.argv[1:]
    only_day = int(args[0]) if args else None
    all_results = []
    for day in (7, 14, 28, 21):
        if only_day is not None and day != only_day:
            continue
        day_support = SUPPORT[SUPPORT.day.eq(day) & SUPPORT.admissible]
        for i, pair in enumerate(PAIRS.itertuples(), 1):
            source, target = prep_pair(pair, day)
            day_pair_support = day_support[day_support.pair_id.eq(pair.pair_id)]
            for seed in day_pair_support.seed:
                role_subset = ROLES[ROLES.pair_id.eq(pair.pair_id) & ROLES.seed.eq(seed)]
                role_map = dict(zip(role_subset.enroll_id, role_subset.role))
                for base in ("logistic", "lightgbm"):
                    try:
                        result = one_job(pair, day, seed, base, source, target, role_map)
                        all_results.append(result)
                    except Exception as exc:
                        fail = {"pair_id": pair.pair_id, "day": day, "seed": int(seed), "base_model": base,
                                "error": str(exc)}
                        with (PRED / "failures.jsonl").open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(fail, ensure_ascii=False) + "\n")
                        print("FAILED", fail, flush=True)
            if i % 5 == 0:
                print(f"day {day}: {i}/{len(PAIRS)} pairs, {len(all_results)} jobs", flush=True)
    print(f"COMPLETE day={only_day}: {len(all_results)} completed jobs", flush=True)


if __name__ == "__main__":
    main()
