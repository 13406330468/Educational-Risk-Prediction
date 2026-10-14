"""Freeze external validation eligibility before fitting any model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
AUDIT = ROOT / "outputs" / "xuetangx_data_audit"
OUT = ROOT / "outputs" / "xuetangx_experiment4"
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path(__import__("os").environ["CALTRANSFER_XUETANGX_ZIP"])

inventory = pd.read_csv(AUDIT / "course_inventory.csv")
pairs = pd.read_csv(AUDIT / "consecutive_same_base_candidates.csv")
enroll = pd.read_csv(ROOT / "work" / "xuetangx_audit" / "enrollment_index.csv",
                     dtype={"enroll_id": str, "username": str, "course_id": str})
assert __import__("os").environ.get("CALTRANSFER_SMOKE") == "1" or (len(inventory) == 247 and len(enroll) == 225642 and len(pairs) == 84)

eligible_flag = (
    pairs.observation_gap_days.ge(7)
    & pairs.source_enrollments.ge(300) & pairs.target_enrollments.ge(300)
    & pairs.source_minority_count.ge(50) & pairs.target_minority_count.ge(50)
)
pairs = pairs.assign(
    pair_id=[f"P{i:03d}" for i in range(len(pairs))],
    eligible=eligible_flag,
)
pairs["exclusion_reason"] = [
    "" if ok else "; ".join(reason for condition, reason in [
        (row.observation_gap_days < 7, "observed_gap_lt_7_days"),
        (row.source_enrollments < 300 or row.target_enrollments < 300, "enrollment_lt_300"),
        (row.source_minority_count < 50 or row.target_minority_count < 50, "minority_lt_50"),
    ] if condition)
    for ok, row in zip(eligible_flag, pairs.itertuples())
]
eligible = pairs[pairs.eligible].copy()
excluded = pairs[~pairs.eligible].copy()

inventory.to_csv(OUT / "presentation_inventory.csv", index=False)
eligible.to_csv(OUT / "eligible_pairs.csv", index=False)
excluded.to_csv(OUT / "excluded_pairs.csv", index=False)

by_course = {k: g for k, g in enroll.groupby("course_id")}
overlap_rows = []
support_rows = []
for row in pairs.itertuples():
    source = by_course[row.source_course_id]
    target = by_course[row.target_course_id]
    source_users = set(source.username)
    target_users = set(target.username)
    returning = source_users & target_users
    overlap_rows.append({
        "pair_id": row.pair_id, "eligible": bool(row.eligible),
        "source_users": len(source_users), "target_users": len(target_users),
        "returning_target_users": len(returning),
        "returning_target_fraction": len(returning) / len(target_users),
    })
    source_start = pd.Timestamp(row.source_first_event)
    target_start = pd.Timestamp(row.target_first_event)
    source_first = pd.to_datetime(source.first_event)
    target_first = pd.to_datetime(target.first_event)
    for day in (7, 14, 21, 28):
        sa = source[source_first.le(source_start + pd.Timedelta(days=day))]
        ta = target[target_first.le(target_start + pd.Timedelta(days=day))]
        tu = ta[~ta.username.isin(source_users)]
        tr = ta[ta.username.isin(source_users)]
        support_rows.append({
            "pair_id": row.pair_id, "eligible": bool(row.eligible), "day": day,
            "source_active_n": len(sa), "source_active_0": int(sa.truth.eq(0).sum()),
            "source_active_1": int(sa.truth.eq(1).sum()),
            "target_unseen_active_n": len(tu), "target_unseen_active_0": int(tu.truth.eq(0).sum()),
            "target_unseen_active_1": int(tu.truth.eq(1).sum()),
            "target_returning_active_n": len(tr),
            "target_all_active_n": len(ta),
            "target_zero_fill_n": len(target),
        })
pd.DataFrame(overlap_rows).to_csv(OUT / "user_overlap_audit.csv", index=False)
support = pd.DataFrame(support_rows)
support.to_csv(OUT / "landmark_support.csv", index=False)

digest = hashlib.sha256()
with RAW.open("rb") as file:
    while block := file.read(8 * 1024 * 1024):
        digest.update(block)
pd.DataFrame([{"path": str(RAW), "sha256": digest.hexdigest(), "bytes": RAW.stat().st_size}]).to_csv(
    OUT / "input_hashes.csv", index=False)

protocol = {
    "name": "Experiment 4: XuetangX external calibration transfer validation",
    "frozen_at": "2026-10-01",
    "data_scope": "Kaggle samyakjhaveri/mooc-final attached ZIP; observed history windows, not official course calendar",
    "analysis_tiers": ["P_primary", "R_robustness", "M_mechanism", "E_exploratory"],
    "label": "truth=1 treated provisionally as dropout per Goel & Goyal 2020; truth=0 non-dropout, not certified completion",
    "presentation_key": "exact provider + base course token, legacy slash and v1 plus IDs parsed; no fuzzy matching",
    "pairing": "adjacent presentations sorted by first observed event; >=7 days from source last observed event to target first; source/target enrollments >=300 and both label classes >=50",
    "primary_risk_set": "enrollments with first event <= landmark; target usernames absent from source presentation",
    "primary_landmarks_days": [7, 14, 28],
    "secondary_landmark_days": [21],
    "anchor": "minimum observed event timestamp in each presentation; not official start date",
    "feature_cutoff": "event time <= anchor + landmark days; no post-landmark events, labels, IDs, or course IDs as predictive features",
    "source_roles": "60% train / 20% validation / 20% calibration by username, stratified by truth, fixed for all landmarks per pair and seed",
    "base_models": ["logistic", "lightgbm"],
    "seeds": [17, 29, 43, 59, 71],
    "primary_methods": ["raw", "intercept_only", "Platt"],
    "target_diagnostic": "5-fold out-of-fold grouped by username; retrospective, never deployed",
    "primary_score": "Brier gain = Brier(raw)-Brier(calibrated); G_S source calibration OOF; G_T transferred source full-calibrator; G_T_star target OOF diagnostic; A=G_T_star-G_T",
    "aggregation": "average seeds within pair/day/model, then equal-weight macro across pairs; course-cluster bootstrap resamples base course, 2000 draws",
    "no_model_selection_on_target": True,
    "known_limitations": ["official course start/end absent", "truth generation window absent", "event logs have at most 36 observed days", "Kaggle train/test not chronological"],
    "eligible_pair_ids_sha256": hashlib.sha256("|".join(eligible.pair_id).encode()).hexdigest(),
    "counts": {"adjacent_candidates": len(pairs), "eligible_pairs": len(eligible), "excluded_pairs": len(excluded)},
}
(OUT / "experiment4_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"eligible": len(eligible), "excluded": len(excluded),
                  "day7_target_unseen_min": int(support[support.eligible & support.day.eq(7)].target_unseen_active_n.min()),
                  "day7_target_unseen_minority_min": int(support[support.eligible & support.day.eq(7)][["target_unseen_active_0", "target_unseen_active_1"]].min(axis=1).min())}, indent=2), flush=True)
