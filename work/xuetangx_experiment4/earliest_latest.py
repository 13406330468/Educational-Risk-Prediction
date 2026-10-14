"""Outcome-independent earliest-to-latest pairing sensitivity for 3+ offerings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import primary as p


OUT = p.OUT
PRED = p.WORK / "earliest_latest_predictions"
PRED.mkdir(exist_ok=True)
p.PRED = PRED
inventory = pd.read_csv(OUT / "presentation_inventory.csv")
rows = []
for (provider, base), group in inventory.groupby(["provider", "base_course_token"]):
    group = group.sort_values("first_observed_event")
    if len(group) < 3:
        continue
    first, last = group.iloc[0], group.iloc[-1]
    gap = (pd.Timestamp(last.first_observed_event)-pd.Timestamp(first.last_observed_event)).total_seconds()/86400
    ok = (gap >= 7 and first.enrollments >= 300 and last.enrollments >= 300
          and min(first.truth_0_enrollments, first.truth_1_enrollments) >= 50
          and min(last.truth_0_enrollments, last.truth_1_enrollments) >= 50)
    rows.append({"provider": provider, "base_course_token": base,
                 "source_course_id": first.course_id, "target_course_id": last.course_id,
                 "observation_gap_days": gap, "presentation_count": len(group),
                 "eligible": ok})
candidates = pd.DataFrame(rows).sort_values(["provider", "base_course_token"]).reset_index(drop=True)
candidates["pair_id"] = [f"R{i:03d}" for i in range(len(candidates))]
candidates.to_csv(OUT / "earliest_latest_candidate_pairs.csv", index=False)
print(f"earliest/latest candidates {len(candidates)}, eligible {candidates.eligible.sum()}", flush=True)

results = []
unit_support = []
for row in candidates[candidates.eligible].itertuples():
    source_full = p.BY_COURSE[row.source_course_id].reset_index(drop=True)
    for day in (7,14,21,28):
        source, target = p.prep_pair(row, day)
        target_primary = target[target.active & target.unseen]
        tmin = int(target_primary.y.value_counts().min()) if target_primary.y.nunique()==2 else 0
        for seed in p.PROTOCOL["seeds"]:
            split_seed = seed + int(row.pair_id[1:])*1009 + 500000
            idx = np.arange(len(source_full))
            tr, held = train_test_split(idx, test_size=.4, random_state=split_seed, stratify=source_full.y)
            va, ca = train_test_split(held, test_size=.5, random_state=split_seed+104729,
                                      stratify=source_full.iloc[held].y)
            role = np.empty(len(source_full), dtype=object)
            role[tr], role[va], role[ca] = "train", "validation", "calibration"
            role_map = dict(zip(source_full.enroll_id, role))
            sr = source.enroll_id.map(role_map)
            counts = {}
            for name in ("train","validation","calibration"):
                y = source.loc[sr.eq(name), "y"]
                counts[name+"_n"] = len(y)
                counts[name+"_minority"] = int(y.value_counts().min()) if y.nunique()==2 else 0
            okay = (counts["train_minority"]>=5 and counts["validation_minority"]>=2
                    and counts["calibration_n"]>=30 and counts["calibration_minority"]>=10
                    and len(target_primary)>=50 and tmin>=10)
            unit_support.append({"pair_id":row.pair_id,"day":day,"seed":seed,
                                 "target_n":len(target_primary),"target_minority":tmin,
                                 **counts,"admissible":okay})
            if not okay:
                continue
            for base in ("logistic","lightgbm"):
                try:
                    results.append(p.one_job(row, day, seed, base, source, target, role_map))
                except Exception as exc:
                    raise RuntimeError(f"R4 failed: {row.pair_id} {day} {seed} {base}") from exc
    print(f"earliest/latest {row.pair_id} complete", flush=True)

pd.DataFrame(unit_support).to_csv(OUT / "earliest_latest_unit_support.csv", index=False)
frame = pd.DataFrame(results)
frame.to_csv(OUT / "earliest_latest_by_seed.csv", index=False)
summary = frame.groupby(["pair_id","day","base_model"],as_index=False)[["G_S","G_T","G_T_star","A"]].mean()
summary.to_csv(OUT / "earliest_latest_pair_mean.csv", index=False)
print(json.dumps({"jobs":len(frame),"pairs":frame.pair_id.nunique(),
                  "pair_day_model_units":len(summary)},indent=2),flush=True)
