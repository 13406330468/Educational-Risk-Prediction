"""Historical-output pooling and a fixed random different-course negative control.

Pooled raw probabilities come from separately trained source models of the same
architecture. The target base model is held fixed. This is a mapping-pool
sensitivity, not the stronger same-model historical-calibration design.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from primary import OUT, PRED, brier, fit_cal, apply_cal


rows = pd.read_csv(OUT / "primary_by_seed.csv")
inventory = pd.read_csv(OUT / "presentation_inventory.csv")
meta = inventory.set_index("course_id")[["first_observed_event", "last_observed_event", "provider", "base_course_token"]]
summary = []
random_rows = []

for (day, base, seed), jobs in rows.groupby(["day", "base_model", "seed"]):
    cache = {}
    source_params = {}
    for job in jobs.itertuples():
        source = job.source_course_id
        if source in cache:
            continue
        f = pd.read_parquet(PRED / f"{job.pair_id}_d{day}_{base}_s{seed}.parquet")
        s = f[f.population.eq("source_calibration")][["raw", "y"]]
        cache[source] = (s.raw.to_numpy(), s.y.to_numpy(dtype=int))
        source_params[source] = (job.source_Platt_a, job.source_Platt_b)
    for job in jobs.itertuples():
        f = pd.read_parquet(PRED / f"{job.pair_id}_d{day}_{base}_s{seed}.parquet")
        target = f[f.population.eq("target_primary")]
        y = target.y.to_numpy(dtype=int)
        p = target.raw.to_numpy()
        base_brier = brier(y, p)
        target_start = meta.loc[job.target_course_id, "first_observed_event"]
        base_key = (job.provider, job.base_course_token)
        historical = [course for course in cache if meta.loc[course, "last_observed_event"] < target_start]
        same = [course for course in historical
                if (meta.loc[course, "provider"], meta.loc[course, "base_course_token"]) == base_key]
        other = [course for course in historical if course not in same]
        out = {"pair_id": job.pair_id, "day": day, "base_model": base, "seed": seed,
               "target_n": len(y), "previous_G_T": job.G_T,
               "historical_sources": len(historical), "same_course_sources": len(same),
               "other_course_sources": len(other)}
        for name, choices in (("same_course_pool", same), ("global_pool", historical), ("other_course_pool", other)):
            if not choices:
                out[f"G_T_{name}"] = np.nan
                out[f"{name}_calibration_n"] = 0
                continue
            pp = np.concatenate([cache[course][0] for course in choices])
            yy = np.concatenate([cache[course][1] for course in choices])
            out[f"{name}_calibration_n"] = len(pp)
            out[f"G_T_{name}"] = base_brier - brier(y, apply_cal(p, fit_cal(pp, yy, "Platt")))
        summary.append(out)
        if other:
            rng = np.random.default_rng(509701 + int(job.pair_id[1:]) * 10007 + day * 101 + seed * 17 + (base == "lightgbm"))
            picked = rng.choice(other, size=20, replace=True)
            for repeat, course in enumerate(picked, 1):
                random_rows.append({"pair_id": job.pair_id, "day": day, "base_model": base,
                                    "seed": seed, "repeat": repeat, "random_source_course_id": course,
                                    "G_T_random_other_source": base_brier - brier(y, apply_cal(p, source_params[course])),
                                    "G_T_same_course_previous": job.G_T})
    print(f"history controls day {day} {base} seed {seed}: {len(jobs)} jobs", flush=True)

history = pd.DataFrame(summary)
history.to_csv(OUT / "historical_pooling.csv", index=False)
random = pd.DataFrame(random_rows)
random.to_csv(OUT / "random_course_negative_control.csv", index=False)
print(json.dumps({"pooling_rows": len(history), "random_rows": len(random),
                  "multi_history_jobs": int(history.same_course_sources.ge(2).sum())}, indent=2))
