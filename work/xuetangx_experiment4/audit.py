import csv
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
ARCHIVE = Path(__import__("os").environ["CALTRANSFER_XUETANGX_ZIP"])
OUT = ROOT / "outputs" / "xuetangx_data_audit"
OUT.mkdir(parents=True, exist_ok=True)

COURSE_RE = re.compile(r"^course-v1:([^+]+)\+([^+]+)\+(.+)$")
COLUMNS = ["enroll_id", "username", "course_id", "session_id", "action", "time", "truth"]
CHUNK = 400_000

stats = {
    "archive": str(ARCHIVE),
    "archive_bytes": ARCHIVE.stat().st_size,
    "files": {},
    "splits": {},
    "overall": {},
}
records = {}
courses = defaultdict(lambda: {"events": 0, "first_event": None, "last_event": None, "split_events": Counter()})
actions = Counter()
truth_rows = Counter()
all_dates = set()
bad_time_examples = []
issues = Counter()


def minimum(a, b):
    return b if a is None or b < a else a


def maximum(a, b):
    return b if a is None or b > a else a


def scan_member(archive, member, split):
    start = time.monotonic()
    s = {"rows": 0, "chunks": 0, "missing": Counter(), "bad_time_shape": 0}
    with archive.open(member) as stream:
        reader = pd.read_csv(
            stream, usecols=COLUMNS, dtype=str, keep_default_na=False,
            chunksize=CHUNK, low_memory=False,
        )
        for chunk in reader:
            s["rows"] += len(chunk)
            s["chunks"] += 1
            for col in COLUMNS:
                s["missing"][col] += int(chunk[col].eq("").sum())

            # ISO timestamps sort chronologically as strings. Validate the shape
            # for every event and calendar dates once per distinct date below.
            valid = chunk["time"].str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
            s["bad_time_shape"] += int((~valid).sum())
            if len(bad_time_examples) < 5:
                bad_time_examples.extend(chunk.loc[~valid, "time"].head(5 - len(bad_time_examples)).tolist())
            all_dates.update(chunk.loc[valid, "time"].str.slice(0, 10).unique().tolist())

            actions.update(chunk["action"].value_counts(dropna=False).to_dict())
            truth_rows.update(chunk["truth"].value_counts(dropna=False).to_dict())

            course_chunk = chunk.groupby("course_id", sort=False, dropna=False)["time"].agg(["size", "min", "max"])
            for course_id, r in course_chunk.iterrows():
                c = courses[course_id]
                c["events"] += int(r["size"])
                c["split_events"][split] += int(r["size"])
                c["first_event"] = minimum(c["first_event"], r["min"])
                c["last_event"] = maximum(c["last_event"], r["max"])

            enrollment_chunk = chunk.groupby("enroll_id", sort=False, dropna=False).agg(
                course=("course_id", "first"), course_n=("course_id", "nunique"),
                user=("username", "first"), user_n=("username", "nunique"),
                truth=("truth", "first"), truth_n=("truth", "nunique"),
                first_time=("time", "min"), last_time=("time", "max"),
                events=("time", "size"),
            )
            for enroll_id, r in enrollment_chunk.iterrows():
                if not enroll_id:
                    issues["blank_enrollment_rows_grouped"] += int(r.events)
                if r.course_n > 1:
                    issues["course_conflict_within_chunk"] += 1
                if r.user_n > 1:
                    issues["user_conflict_within_chunk"] += 1
                if r.truth_n > 1:
                    issues["truth_conflict_within_chunk"] += 1
                old = records.get(enroll_id)
                if old is None:
                    records[enroll_id] = [r.course, r.user, r.truth, r.first_time, r.last_time, int(r.events), 1 if split == "train" else 2]
                else:
                    if old[0] != r.course:
                        issues["course_conflict_across_chunks"] += 1
                    if old[1] != r.user:
                        issues["user_conflict_across_chunks"] += 1
                    if old[2] != r.truth:
                        issues["truth_conflict_across_chunks"] += 1
                    old[3] = minimum(old[3], r.first_time)
                    old[4] = maximum(old[4], r.last_time)
                    old[5] += int(r.events)
                    old[6] |= 1 if split == "train" else 2
            if s["chunks"] % 10 == 0:
                print(f"{split}: {s['rows']:,} rows, {len(records):,} enrollments, {time.monotonic()-start:.0f}s", flush=True)
    s["missing"] = dict(s["missing"])
    s["elapsed_seconds"] = round(time.monotonic() - start, 1)
    print(f"{split} COMPLETE: {s['rows']:,} rows in {s['elapsed_seconds']}s", flush=True)
    return s


with zipfile.ZipFile(ARCHIVE) as z:
    stats["files"] = {
        info.filename: {"uncompressed_bytes": info.file_size, "compressed_bytes": info.compress_size}
        for info in z.infolist()
    }
    for split, member in [("train", "train/train.csv"), ("test", "test/test.csv")]:
        stats["splits"][split] = scan_member(z, member, split)

try:
    valid_dates = pd.to_datetime(sorted(all_dates), format="%Y-%m-%d", errors="raise")
    calendar_issue = None
except Exception as ex:
    calendar_issue = str(ex)

enrollments = pd.DataFrame.from_dict(
    records, orient="index",
    columns=["course_id", "username", "truth", "first_event", "last_event", "event_count", "split_mask"],
)
enrollments.index.name = "enroll_id"
enrollments["split"] = enrollments.split_mask.map({1: "train", 2: "test", 3: "both"})
enrollments.to_csv(ROOT / "work" / "xuetangx_audit" / "enrollment_index.csv")

course_rows = []
for course_id, c in courses.items():
    match = COURSE_RE.match(course_id)
    group = enrollments.loc[enrollments.course_id == course_id]
    row = {
        "course_id": course_id,
        "provider": match.group(1) if match else "",
        "base_course_token": match.group(2) if match else "",
        "run_token": match.group(3) if match else "",
        "first_observed_event": c["first_event"],
        "last_observed_event": c["last_event"],
        "event_rows": c["events"],
        "enrollments": len(group),
        "users": group.username.nunique(),
        "truth_0_enrollments": int(group.truth.eq("0").sum()),
        "truth_1_enrollments": int(group.truth.eq("1").sum()),
        "train_events": c["split_events"]["train"],
        "test_events": c["split_events"]["test"],
        "train_enrollments": int(group.split_mask.isin([1, 3]).sum()),
        "test_enrollments": int(group.split_mask.isin([2, 3]).sum()),
    }
    course_rows.append(row)
course_table = pd.DataFrame(course_rows).sort_values(["first_observed_event", "course_id"])
course_table.to_csv(OUT / "course_inventory.csv", index=False)

repeat = course_table[course_table.base_course_token.ne("")].groupby(["provider", "base_course_token"], sort=False).filter(lambda g: len(g) > 1)
repeat.to_csv(OUT / "repeated_base_token_candidates.csv", index=False)

by_split = {}
for split in ("train", "test"):
    g = enrollments[enrollments.split_mask.isin([1, 3] if split == "train" else [2, 3])]
    by_split[split] = {
        "enrollments": len(g), "users": g.username.nunique(), "courses": g.course_id.nunique(),
        "first_event": g.first_event.min(), "last_event": g.last_event.max(),
        "truth_enrollments": g.truth.value_counts().to_dict(),
    }

course_year_counts = course_table.first_observed_event.str[:4].value_counts().sort_index().to_dict()
stats["overall"] = {
    "event_rows": sum(v["rows"] for v in stats["splits"].values()),
    "unique_enrollments": len(enrollments),
    "unique_users": enrollments.username.nunique(),
    "unique_course_ids": len(course_table),
    "enrollment_truth_counts": enrollments.truth.value_counts().to_dict(),
    "row_truth_counts": dict(truth_rows),
    "actions": dict(actions),
    "issues": dict(issues),
    "bad_time_examples": bad_time_examples,
    "distinct_calendar_days": len(all_dates),
    "calendar_date_parse_issue": calendar_issue,
    "event_first": course_table.first_observed_event.min(),
    "event_last": course_table.last_observed_event.max(),
    "course_first_event_year_counts": course_year_counts,
    "course_id_regex_unparsed": int(course_table.base_course_token.eq("").sum()),
    "repeated_base_token_groups": int(repeat[["provider", "base_course_token"]].drop_duplicates().shape[0]),
    "repeated_base_token_offerings": len(repeat),
    "split_enrollment_overlap": int(enrollments.split_mask.eq(3).sum()),
    "split_user_overlap": len(set(enrollments.loc[enrollments.split_mask.isin([1, 3]), "username"]) & set(enrollments.loc[enrollments.split_mask.isin([2, 3]), "username"])),
    "split_course_overlap": int((course_table.train_events.gt(0) & course_table.test_events.gt(0)).sum()),
    "split_summary": by_split,
}
with (OUT / "audit_summary.json").open("w", encoding="utf-8") as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=2)

print(json.dumps(stats["overall"], ensure_ascii=False, indent=2), flush=True)
