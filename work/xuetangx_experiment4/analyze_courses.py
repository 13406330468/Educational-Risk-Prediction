import itertools
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
OUT = ROOT / "outputs" / "xuetangx_data_audit"
df = pd.read_csv(OUT / "course_inventory.csv", keep_default_na=False)
df["id_style"] = df.course_id.map(lambda x: "v1" if x.startswith("course-v1:") else "legacy")
legacy = df.id_style.eq("legacy")
parsed = df.loc[legacy, "course_id"].str.extract(r"^([^/]+)/([^/]+)/([^/]+)$")
df.loc[legacy, "provider"] = parsed[0].values
df.loc[legacy, "base_course_token"] = parsed[1].values
df.loc[legacy, "run_token"] = parsed[2].values
df["observed_days"] = (
    pd.to_datetime(df.last_observed_event) - pd.to_datetime(df.first_observed_event)
).dt.total_seconds() / 86400
df["first_observed_date"] = df.first_observed_event.str[:10]
df = df.sort_values(["first_observed_event", "course_id"])
df.to_csv(OUT / "course_inventory.csv", index=False)

groups = df.groupby(["provider", "base_course_token"], sort=False)
repeat = groups.filter(lambda g: len(g) > 1).copy()
repeat.to_csv(OUT / "repeated_base_token_candidates.csv", index=False)

pairs = []
for (provider, base), g in groups:
    g = g.sort_values("first_observed_event")
    if len(g) <= 1:
        continue
    rows = g.to_dict("records")
    for a, b in itertools.combinations(rows, 2):
        a_first = pd.Timestamp(a["first_observed_event"])
        a_last = pd.Timestamp(a["last_observed_event"])
        b_first = pd.Timestamp(b["first_observed_event"])
        b_last = pd.Timestamp(b["last_observed_event"])
        gap = (b_first - a_last).total_seconds() / 86400
        pairs.append({
            "provider": provider, "base_course_token": base,
            "earlier_course_id": a["course_id"], "later_course_id": b["course_id"],
            "earlier_run_token": a["run_token"], "later_run_token": b["run_token"],
            "earlier_first_event": a["first_observed_event"], "earlier_last_event": a["last_observed_event"],
            "later_first_event": b["first_observed_event"], "later_last_event": b["last_observed_event"],
            "earlier_enrollments": a["enrollments"], "later_enrollments": b["enrollments"],
            "observation_gap_days": round(gap, 2),
            "nonoverlapping_observed_windows": bool(a_last < b_first),
        })
pair_df = pd.DataFrame(pairs)
pair_df.to_csv(OUT / "repeated_course_pairs.csv", index=False)

consecutive = []
for (provider, base), g in groups:
    g = g.sort_values("first_observed_event")
    if len(g) < 2:
        continue
    rows = g.to_dict("records")
    for a, b in zip(rows[:-1], rows[1:]):
        gap = (pd.Timestamp(b["first_observed_event"]) - pd.Timestamp(a["last_observed_event"])).total_seconds() / 86400
        consecutive.append({
            "provider": provider, "base_course_token": base,
            "source_course_id": a["course_id"], "target_course_id": b["course_id"],
            "source_first_event": a["first_observed_event"], "source_last_event": a["last_observed_event"],
            "target_first_event": b["first_observed_event"], "target_last_event": b["last_observed_event"],
            "observation_gap_days": round(gap, 2),
            "source_enrollments": a["enrollments"], "target_enrollments": b["enrollments"],
            "source_minority_count": min(a["truth_0_enrollments"], a["truth_1_enrollments"]),
            "target_minority_count": min(b["truth_0_enrollments"], b["truth_1_enrollments"]),
            "source_truth_1_rate": round(a["truth_1_enrollments"] / a["enrollments"], 4),
            "target_truth_1_rate": round(b["truth_1_enrollments"] / b["enrollments"], 4),
            "source_observed_days": round(a["observed_days"], 2),
            "target_observed_days": round(b["observed_days"], 2),
        })
consecutive_df = pd.DataFrame(consecutive)
consecutive_df.to_csv(OUT / "consecutive_same_base_candidates.csv", index=False)

summary = {
    "legacy_ids": int(legacy.sum()),
    "legacy_ids_unparsed": int(df.loc[legacy, "base_course_token"].eq("").sum()),
    "repeat_base_groups": int(repeat[["provider", "base_course_token"]].drop_duplicates().shape[0]),
    "repeat_offerings": int(len(repeat)),
    "repeat_pairs": int(len(pair_df)),
    "pairs_nonoverlapping_observed_windows": int(pair_df.nonoverlapping_observed_windows.sum()),
    "pairs_overlapping_observed_windows": int((~pair_df.nonoverlapping_observed_windows).sum()),
    "repeat_groups_with_nonoverlapping_pair": int(pair_df.loc[pair_df.nonoverlapping_observed_windows, ["provider", "base_course_token"]].drop_duplicates().shape[0]),
    "consecutive_pairs": int(len(consecutive_df)),
    "consecutive_nonoverlapping": int(consecutive_df.observation_gap_days.gt(0).sum()),
    "consecutive_with_400_enrollments_and_50_minority_both": int(((consecutive_df.observation_gap_days.gt(0)) & (consecutive_df.source_enrollments.ge(400)) & (consecutive_df.target_enrollments.ge(400)) & (consecutive_df.source_minority_count.ge(50)) & (consecutive_df.target_minority_count.ge(50))).sum()),
    "observed_days_min": round(float(df.observed_days.min()), 3),
    "observed_days_median": round(float(df.observed_days.median()), 3),
    "observed_days_max": round(float(df.observed_days.max()), 3),
    "observed_days_34_to_37_count": int(df.observed_days.between(34, 37).sum()),
    "first_event_date_counts": df.first_observed_date.value_counts().head(10).to_dict(),
    "first_event_year_counts": df.first_observed_event.str[:4].value_counts().sort_index().to_dict(),
}
with (OUT / "course_structure_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nNONOVERLAPPING PAIR EXAMPLES")
print(pair_df.loc[pair_df.nonoverlapping_observed_windows, ["base_course_token", "earlier_course_id", "later_course_id", "observation_gap_days", "earlier_enrollments", "later_enrollments"]].head(25).to_string(index=False))
print("\nOVERLAPPING PAIR EXAMPLES")
print(pair_df.loc[~pair_df.nonoverlapping_observed_windows, ["base_course_token", "earlier_course_id", "later_course_id", "observation_gap_days"]].head(15).to_string(index=False))
