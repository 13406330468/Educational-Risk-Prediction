"""Independent full-pipeline sensitivity for a new explicit 7-day inactivity endpoint.

This does not identify the archived truth label's generation rule. Features end at
each landmark; the new outcome uses only events strictly after that landmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import identity_refit as r

ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
OUT = ROOT / "outputs" / "xuetangx_gap_supplement" / "future_endpoint"
WORK = ROOT / "work" / "xuetangx_experiment4" / "future_endpoint"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
future = pd.read_parquet(ROOT / "work" / "xuetangx_experiment4" / "future_activity_7day.parquet")
assert not future[["enroll_id", "course_id"]].duplicated().any()
base = r.enroll.drop(columns="y").merge(future, on=["enroll_id", "course_id"],
                                         how="inner", validate="one_to_one")
assert len(base) == len(r.enroll), (len(base), len(r.enroll))

protocol = {
    "name": "7-day future inactivity endpoint full refit with one eligible base course per user",
    "date": "2026-10-02",
    "status": "prespecified before future-endpoint model fitting",
    "label": "Y(d)=1 if no logged activity in (anchor+d,anchor+d+7]; 0 otherwise",
    "qualification": "new short-term disengagement outcome, not original truth or certified course dropout",
    "nominal_log_coverage": "all 67 original pairs have observed event spans through days 14, 21 and 35; actual provider collection window unverified",
    "feature_cutoff": "original frozen event-derived features at or before landmark d",
    "landmarks": [7, 14, 28],
    "independence": "reuse frozen earliest eligible base-course assignment for every username before source/target roles, model fitting, calibration, adaptation and evaluation",
    "roles": "fresh outcome-stratified 60/20/20 source role split for each pair, landmark and original seed",
    "models": ["logistic", "lightgbm"],
    "seeds": [17, 29, 43, 59, 71],
    "support": "same frozen thresholds as original Experiment 4, applied to new outcome",
    "estimand": "same Brier gain decomposition: G_S, G_T, G_T_star and A",
    "uncertainty": "base-course cluster bootstrap; target OOF adaptation is retrospective",
}
path = OUT / "future_endpoint_protocol.json"
if path.exists():
    assert json.loads(path.read_text(encoding="utf-8")) == protocol
else:
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

for day in (7, 14, 28):
    day_out = OUT / f"day{day}"
    day_work = WORK / f"day{day}"
    pred = day_work / "predictions"
    for folder in (day_out, day_work, pred):
        folder.mkdir(parents=True, exist_ok=True)
    r.OUT, r.WORK, r.PRED, r.DAYS = day_out, day_work, pred, (day,)
    r.p.PRED = pred
    current = base.drop(columns=[c for c in base if c.startswith("active_") or c.startswith("future_event_")]).copy()
    current["y"] = 1 - base[f"future_event_{day}"].astype(int)
    r.p.BY_COURSE = {k: g for k, g in current.groupby("course_id")}
    print(f"FUTURE_ENDPOINT_DAY {day} eligible_enrollments={len(current)} outcome_1_rate={current.y.mean():.4f}", flush=True)
    r.main()
