"""Label-budget matching, user-population sensitivity, Beta and isotonic checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from primary import EPS, OUT, PRED, brier, clip, fit_cal, apply_cal


def fit_other(p, y, method):
    p = clip(p)
    y = np.asarray(y, dtype=float)
    if method == "isotonic":
        return IsotonicRegression(y_min=EPS, y_max=1-EPS, out_of_bounds="clip").fit(p, y)
    x = np.column_stack([np.log(p), -np.log1p(-p), np.ones(len(p))])
    init = np.array([1., 1., 0.])
    def objective(w):
        z = x @ w
        return (np.mean(np.logaddexp(0, z) - y * z) + 1e-5 * np.sum((w-init)**2),
                x.T @ (expit(z)-y) / len(y) + 2e-5 * (w-init))
    result = minimize(objective, init, method="L-BFGS-B", jac=True,
                      bounds=[(0,30),(0,30),(-30,30)], options={"maxiter":300})
    if not np.isfinite(result.fun):
        raise ValueError("Beta calibration nonfinite")
    return result.x


def predict_other(model, p, method):
    p = clip(p)
    if method == "isotonic":
        return clip(model.predict(p))
    x = np.column_stack([np.log(p), -np.log1p(-p), np.ones(len(p))])
    return clip(expit(x @ model))


def crossfit_other(p, y, method, seed):
    p = np.asarray(p)
    y = np.asarray(y)
    pred = np.empty(len(y))
    for fit, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y):
        model = fit_other(p[fit], y[fit], method)
        pred[test] = predict_other(model, p[test], method)
    return pred


def crossfit_simple(p, y, method, seed):
    p = np.asarray(p)
    y = np.asarray(y)
    pred = np.empty(len(y))
    for fit, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y):
        pred[test] = apply_cal(p[test], fit_cal(p[fit], y[fit], method))
    return pred


def matched_target(tgt, source_n, seed, day, pair_num):
    y = tgt.y.to_numpy(dtype=int)
    p = tgt.raw.to_numpy(dtype=float)
    folds = tgt.fold.to_numpy(dtype=int)
    raw_brier = brier(y, p)
    rows = []
    for repeat in (101, 202, 303):
        pred = {kind: np.full(len(tgt), np.nan) for kind in ("intercept", "Platt")}
        fit_sizes = []
        for fold in range(5):
            train = np.where(folds != fold)[0]
            test = np.where(folds == fold)[0]
            nfit = min(source_n, len(train))
            if nfit == len(train):
                selected = train
            else:
                rng_seed = repeat * 100003 + seed * 1009 + day * 101 + pair_num * 7 + fold
                rng = np.random.default_rng(rng_seed)
                for _ in range(100):
                    selected = rng.choice(train, size=nfit, replace=False)
                    if len(np.unique(y[selected])) == 2:
                        break
                else:
                    raise ValueError("matched sample lacks a class")
            fit_sizes.append(len(selected))
            for kind in pred:
                params = fit_cal(p[selected], y[selected], kind)
                pred[kind][test] = apply_cal(p[test], params)
        rows.append({"repeat": repeat, "target_fit_n_min": min(fit_sizes),
                     "target_fit_n_max": max(fit_sizes),
                     "G_T_star_intercept_matched": raw_brier-brier(y, pred["intercept"]),
                     "G_T_star_Platt_matched": raw_brier-brier(y, pred["Platt"])})
    return rows


def one_job(row):
    stem = f"{row.pair_id}_d{row.day}_{row.base_model}_s{row.seed}"
    f = pd.read_parquet(PRED / f"{stem}.parquet")
    sc = f[f.population.eq("source_calibration")].reset_index(drop=True)
    tgt = f[f.population.eq("target_primary")].reset_index(drop=True)
    all_target = f[f.population.eq("target_all")].reset_index(drop=True)
    result = {"pair_id": row.pair_id, "day": row.day, "base_model": row.base_model, "seed": row.seed,
              "source_cal_n": len(sc), "target_primary_n": len(tgt)}
    rsc = brier(sc.y, sc.raw)
    rt = brier(tgt.y, tgt.raw)
    for method in ("beta", "isotonic"):
        source_full = fit_other(sc.raw, sc.y, method)
        source_oof = crossfit_other(sc.raw.to_numpy(), sc.y.to_numpy(), method, 1000+row.seed+row.day)
        target_oof = crossfit_other(tgt.raw.to_numpy(), tgt.y.to_numpy(), method,
                                    20261001+int(row.pair_id[1:])*101+row.day)
        deployed = predict_other(source_full, tgt.raw, method)
        result[f"G_S_{method}"] = rsc-brier(sc.y, source_oof)
        result[f"G_T_{method}"] = rt-brier(tgt.y, deployed)
        result[f"G_T_star_{method}"] = rt-brier(tgt.y, target_oof)
        result[f"A_{method}"] = result[f"G_T_star_{method}"]-result[f"G_T_{method}"]
    for riskset, sub in [
        ("all_active", all_target[all_target.active]),
        ("returning_active", all_target[all_target.active & ~all_target.unseen]),
        ("all_zero_fill", all_target),
    ]:
        sub = sub.reset_index(drop=True)
        result[f"{riskset}_n"] = len(sub)
        result[f"{riskset}_minority"] = int(sub.y.value_counts().min()) if sub.y.nunique() == 2 else 0
        if len(sub) < 50 or result[f"{riskset}_minority"] < 10:
            result[f"G_T_{riskset}"] = np.nan
            result[f"G_T_star_{riskset}"] = np.nan
            result[f"A_{riskset}"] = np.nan
            continue
        oof = crossfit_simple(sub.raw.to_numpy(), sub.y.to_numpy(), "Platt",
                              40261001+int(row.pair_id[1:])*101+row.day)
        raw = brier(sub.y, sub.raw)
        result[f"G_T_{riskset}"] = raw-brier(sub.y, sub.source_Platt)
        result[f"G_T_star_{riskset}"] = raw-brier(sub.y, oof)
        result[f"A_{riskset}"] = result[f"G_T_star_{riskset}"]-result[f"G_T_{riskset}"]
    matched = matched_target(tgt, len(sc), row.seed, row.day, int(row.pair_id[1:]))
    for m in matched:
        m.update({"pair_id": row.pair_id, "day": row.day, "base_model": row.base_model,
                  "seed": row.seed, "source_fit_n": len(sc), "G_T": row.G_T,
                  "A_matched": m["G_T_star_Platt_matched"]-row.G_T})
    return result, matched


def main(day):
    primary = pd.read_csv(OUT / "primary_by_seed.csv")
    primary = primary[primary.day.eq(day)]
    robust = []
    matched = []
    errors = []
    for i, row in enumerate(primary.itertuples(), 1):
        try:
            r, m = one_job(row)
            robust.append(r)
            matched.extend(m)
        except Exception as ex:
            errors.append({"pair_id": row.pair_id, "day": day, "seed": row.seed,
                           "base_model": row.base_model, "error": str(ex)})
        if i % 100 == 0:
            print(f"day {day}: {i}/{len(primary)} robustness jobs", flush=True)
    pd.DataFrame(robust).to_csv(OUT / f"secondary_day{day}.csv", index=False)
    pd.DataFrame(matched).to_csv(OUT / f"matched_day{day}.csv", index=False)
    if errors:
        (OUT / f"secondary_errors_day{day}.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(json.dumps({"day": day, "complete": len(robust), "errors": len(errors)}), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
