"""Retrained all-enrollment zero-fill risk-set sensitivity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import primary as p


OUT = p.OUT
DEST = p.WORK / "zero_fill_predictions"
DEST.mkdir(exist_ok=True)
p.PRED = DEST
rows = []
unsupported = []

for day in (7,14,28,21):
    for i, pair in enumerate(p.PAIRS.itertuples(),1):
        source = p.BY_COURSE[pair.source_course_id].merge(
            p.DAY_BY_COURSE[day][pair.source_course_id],
            on=["enroll_id","course_id"],how="left",validate="one_to_one")
        target = p.BY_COURSE[pair.target_course_id].merge(
            p.DAY_BY_COURSE[day][pair.target_course_id],
            on=["enroll_id","course_id"],how="left",validate="one_to_one")
        source_users = set(p.BY_COURSE[pair.source_course_id].username)
        target["unseen"] = ~target.username.isin(source_users)
        target["active"] = target.event_count.notna()
        for frame in (source,target):
            for col in p.NON_ACTION+p.ACTION_COUNT_COLS:
                frame[col] = frame[col].fillna(day if col in ("first_activity_lag","recency_to_landmark") else 0)
            for action,count_col in zip(p.ACTION_NAMES,p.ACTION_COUNT_COLS):
                frame[f"action_{action}_fraction"] = np.where(frame.event_count.gt(0),frame[count_col]/frame.event_count,0)
        target_eval = target[target.unseen]
        if len(target_eval)<50 or target_eval.y.nunique()<2 or target_eval.y.value_counts().min()<10:
            unsupported.append({"pair_id":pair.pair_id,"day":day,"reason":"target_all_unseen_support"})
            continue
        for seed in p.PROTOCOL["seeds"]:
            role_subset = p.ROLES[p.ROLES.pair_id.eq(pair.pair_id)&p.ROLES.seed.eq(seed)]
            role_map = dict(zip(role_subset.enroll_id,role_subset.role))
            roles = source.enroll_id.map(role_map)
            cal_y = source.loc[roles.eq("calibration"),"y"]
            if cal_y.nunique()<2 or cal_y.value_counts().min()<10:
                unsupported.append({"pair_id":pair.pair_id,"day":day,"seed":seed,"reason":"source_full_calibration_support"})
                continue
            for base in ("logistic","lightgbm"):
                rows.append(p.one_job(pair,day,seed,base,source,target,role_map,riskset="all_unseen"))
        if i%10==0:
            print(f"zero-fill day {day}: {i}/{len(p.PAIRS)} pairs",flush=True)

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "zero_fill_by_seed.csv",index=False)
pair = frame.groupby(["pair_id","day","base_model"],as_index=False)[["G_S","G_T","G_T_star","A"]].mean()
pair.to_csv(OUT / "zero_fill_pair_mean.csv",index=False)
pd.DataFrame(unsupported).to_csv(OUT / "zero_fill_unsupported.csv",index=False)
print(json.dumps({"jobs":len(rows),"pair_units":len(pair),"unsupported":len(unsupported)}),flush=True)
