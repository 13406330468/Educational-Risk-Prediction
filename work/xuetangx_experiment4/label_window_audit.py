"""Audit label provenance and derive explicit, distinct future-activity outcomes."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
ZIP = Path(__import__("os").environ["CALTRANSFER_XUETANGX_ZIP"])
OUT = ROOT / "outputs" / "xuetangx_gap_supplement"
WORK = ROOT / "work" / "xuetangx_experiment4"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = (7, 14, 28)
PAIRS = pd.read_csv(ROOT / "outputs" / "xuetangx_experiment4" / "eligible_pairs.csv")
INV = pd.read_csv(ROOT / "outputs" / "xuetangx_experiment4" / "presentation_inventory.csv")
COURSES = set(PAIRS.source_course_id) | set(PAIRS.target_course_id)

with zipfile.ZipFile(ZIP) as z:
    members = [{"name": x.filename, "bytes": x.file_size} for x in z.infolist()]
    headers = {name: z.open(name).read(4096).splitlines()[0].decode("utf-8-sig")
               for name in ("train/train.csv", "test/test.csv")}
sha = hashlib.file_digest(ZIP.open("rb"), "sha256").hexdigest()
gaps = PAIRS.observation_gap_days
nominal_coverage = {}
endpoint_maturity_lead = {}
for day in DAYS:
    i = INV.set_index("course_id")
    s = i.loc[PAIRS.source_course_id, "observed_days"].to_numpy()
    t = i.loc[PAIRS.target_course_id, "observed_days"].to_numpy()
    nominal_coverage[str(day)] = int(((s >= day + 7) & (t >= day + 7)).sum())
    lead = (pd.to_datetime(PAIRS.target_first_event)
            - (pd.to_datetime(PAIRS.source_first_event) + pd.Timedelta(days=day + 7)))
    endpoint_maturity_lead[str(day)] = float(lead.dt.total_seconds().min() / 86400)

audit = {
    "archive_sha256": sha,
    "archive_members": members,
    "csv_headers": headers,
    "label_information_present": "truth column in train and test CSV; no README, label generator, or label timestamp within attached archive",
    "original_label_generation_window_verified": False,
    "eligible_pairs": len(PAIRS),
    "observed_source_last_to_target_first_gap_days_min": float(gaps.min()),
    "observed_source_last_to_target_first_gap_days_median": float(gaps.median()),
    "pairs_with_gap_below_10_days": int(gaps.lt(10).sum()),
    "pairs_with_observed_event_span_through_landmark_plus_7_days": nominal_coverage,
    "minimum_days_from_source_new_endpoint_window_end_to_target_first_event": endpoint_maturity_lead,
    "span_caveat": "last observed event after boundary does not certify the provider's complete log collection window",
    "endpoint": "newly derived outcome is absence of logged activity during (day, day+7]; not the archive's truth label, dropout, or completion",
}
(OUT / "label_window_provenance_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

course_list = ",".join("'" + c.replace("'", "''") + "'" for c in sorted(COURSES))
con = duckdb.connect(str(WORK / "events.duckdb"), read_only=True)
sql = f"""
COPY (
  SELECT enroll_id, course_id,
    max(CASE WHEN day_exact >= 0 AND day_exact <= 7 THEN 1 ELSE 0 END)::INTEGER AS active_7,
    max(CASE WHEN day_exact >= 0 AND day_exact <= 14 THEN 1 ELSE 0 END)::INTEGER AS active_14,
    max(CASE WHEN day_exact >= 0 AND day_exact <= 28 THEN 1 ELSE 0 END)::INTEGER AS active_28,
    max(CASE WHEN day_exact > 7 AND day_exact <= 14 THEN 1 ELSE 0 END)::INTEGER AS future_event_7,
    max(CASE WHEN day_exact > 14 AND day_exact <= 21 THEN 1 ELSE 0 END)::INTEGER AS future_event_14,
    max(CASE WHEN day_exact > 28 AND day_exact <= 35 THEN 1 ELSE 0 END)::INTEGER AS future_event_28
  FROM events WHERE course_id IN ({course_list})
  GROUP BY enroll_id, course_id
) TO '{(WORK / 'future_activity_7day.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""
if not (WORK / "future_activity_7day.parquet").exists():
    con.execute(sql)
con.close()

future = pd.read_parquet(WORK / "future_activity_7day.parquet")
enroll = pd.read_csv(ROOT / "work" / "xuetangx_audit" / "enrollment_index.csv",
                     usecols=["enroll_id", "course_id", "truth"],
                     dtype={"enroll_id": "int64", "course_id": str, "truth": int})
f = future.merge(enroll, on=["enroll_id", "course_id"], validate="one_to_one")
assert len(f) == len(future)
rows = []
per_course = []
for day in DAYS:
    a = f[f[f"active_{day}"].eq(1)].copy()
    a["future_inactive"] = a[f"future_event_{day}"].eq(0).astype(int)
    ct = pd.crosstab(a.truth, a.future_inactive)
    rows.append({"day": day, "active_enrollments": len(a),
                 "original_truth_1_fraction": float(a.truth.mean()),
                 "new_future_7day_inactive_fraction": float(a.future_inactive.mean()),
                 "truth_agreement_with_future_inactivity": float(a.truth.eq(a.future_inactive).mean()),
                 "truth_1_and_future_active": int(((a.truth == 1) & (a.future_inactive == 0)).sum()),
                 "truth_0_and_future_inactive": int(((a.truth == 0) & (a.future_inactive == 1)).sum()),
                 "truth_0_future_active": int(ct.loc[0, 0]),
                 "truth_0_future_inactive": int(ct.loc[0, 1]),
                 "truth_1_future_active": int(ct.loc[1, 0]),
                 "truth_1_future_inactive": int(ct.loc[1, 1])})
    course = a.groupby("course_id").agg(
        active_enrollments=("enroll_id", "size"),
        truth_1_fraction=("truth", "mean"),
        future_7day_inactive_fraction=("future_inactive", "mean"),
        truth_future_agreement=("truth", lambda x: float(x.eq(a.loc[x.index, "future_inactive"]).mean())),
    ).reset_index()
    course["day"] = day
    per_course.append(course)
pd.DataFrame(rows).to_csv(OUT / "label_vs_future_activity_summary.csv", index=False)
pd.concat(per_course, ignore_index=True).to_csv(OUT / "label_vs_future_activity_by_course.csv", index=False)
print(json.dumps({"archive_members": [m["name"] for m in members],
                  "pairs": len(PAIRS), "gap_min": float(gaps.min()),
                  "future_outcome_rows": len(f), "summary": rows}, ensure_ascii=False), flush=True)
