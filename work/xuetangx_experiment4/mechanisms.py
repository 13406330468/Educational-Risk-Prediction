"""Prespecified raw-risk, feature, action, and population-shift diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from primary import ACTION_COUNT_COLS, BY_COURSE, NON_ACTION, OUT, PAIRS, PRED, prep_pair, metrics


pair_mean = pd.read_csv(OUT / "primary_pair_mean.csv")
seed_rows = pd.read_csv(OUT / "primary_by_seed.csv")
feature_rows = []
feature_summary = []
for day in (7, 14, 21, 28):
    for pair in PAIRS.itertuples():
        source, target_all = prep_pair(pair, day)
        source_users = set(BY_COURSE[pair.source_course_id].username)
        target = target_all.loc[target_all.active & ~target_all.username.isin(source_users)]
        if len(source) == 0 or len(target) == 0:
            continue
        smds = []
        for feature in NON_ACTION:
            a = source[feature].to_numpy(dtype=float)
            b = target[feature].to_numpy(dtype=float)
            denom = np.sqrt((np.var(a) + np.var(b)) / 2)
            smd = float((np.mean(b) - np.mean(a)) / denom) if denom > 1e-12 else 0.
            smds.append(abs(smd))
            feature_rows.append({"pair_id": pair.pair_id, "day": day, "feature": feature,
                                 "source_mean": float(np.mean(a)), "target_mean": float(np.mean(b)),
                                 "SMD": smd, "abs_SMD": abs(smd)})
        source_action = source[ACTION_COUNT_COLS].sum().to_numpy(dtype=float)
        target_action = target[ACTION_COUNT_COLS].sum().to_numpy(dtype=float)
        action_js = float(jensenshannon(source_action/source_action.sum(),
                                        target_action/target_action.sum(), base=2) ** 2)
        feature_summary.append({"pair_id": pair.pair_id, "day": day,
                                "source_active_n": len(source), "target_unseen_active_n": len(target),
                                "mean_abs_SMD": float(np.mean(smds)), "max_abs_SMD": float(np.max(smds)),
                                "action_JS_divergence": action_js,
                                "returning_fraction_active": float(np.mean(~target_all.loc[target_all.active, "unseen"]))})
pd.DataFrame(feature_rows).to_csv(OUT / "feature_shift_detail.csv", index=False)
shift = pd.DataFrame(feature_summary)
shift.to_csv(OUT / "feature_shift_summary.csv", index=False)

metric_rows = []
risk_rows = []
for i, row in enumerate(seed_rows.itertuples(), 1):
    f = pd.read_parquet(PRED / f"{row.pair_id}_d{row.day}_{row.base_model}_s{row.seed}.parquet")
    source = f[f.population.eq("source_calibration")]
    target = f[f.population.eq("target_primary")]
    for population, frame, methods in [
        ("source_calibration", source, ["raw", "oof_intercept", "oof_slope", "oof_Platt"]),
        ("target_primary", target, ["raw", "source_intercept", "source_slope", "source_Platt", "oof_intercept", "oof_slope", "oof_Platt"]),
    ]:
        for method in methods:
            metric_rows.append({"pair_id": row.pair_id, "day": row.day, "base_model": row.base_model,
                                "seed": row.seed, "population": population, "method": method,
                                "n": len(frame), **metrics(frame.y, frame[method])})
    qs = (.1, .5, .9)
    for q in qs:
        risk_rows.append({"pair_id": row.pair_id, "day": row.day, "base_model": row.base_model,
                          "seed": row.seed, "quantile": q,
                          "source_raw_quantile": float(source.raw.quantile(q)),
                          "target_raw_quantile": float(target.raw.quantile(q)),
                          "absolute_quantile_shift": float(abs(target.raw.quantile(q)-source.raw.quantile(q)))})
    if i % 400 == 0:
        print(f"metric/shift jobs {i}/{len(seed_rows)}", flush=True)
pd.DataFrame(metric_rows).to_csv(OUT / "all_prediction_metrics.csv", index=False)
pd.DataFrame(risk_rows).to_csv(OUT / "raw_risk_quantile_shift.csv", index=False)

joined = pair_mean.merge(shift, on=["pair_id", "day"], validate="many_to_one")
assoc = []
for (day, base), g in joined.groupby(["day", "base_model"]):
    for moderator in ("mean_abs_SMD", "max_abs_SMD", "action_JS_divergence", "returning_fraction_active"):
        for outcome in ("A", "G_T"):
            r, p = spearmanr(g[moderator], g[outcome])
            assoc.append({"day": day, "base_model": base, "moderator": moderator,
                          "outcome": outcome, "pairs": len(g), "spearman_r": float(r),
                          "unadjusted_p_descriptive": float(p)})
pd.DataFrame(assoc).to_csv(OUT / "feature_shift_correlations.csv", index=False)
print(json.dumps({"feature_rows": len(feature_rows), "feature_units": len(shift),
                  "metric_rows": len(metric_rows), "risk_quantile_rows": len(risk_rows)}))
