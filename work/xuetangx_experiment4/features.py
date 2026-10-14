"""Build leakage-controlled, enrollment-level landmark features using DuckDB."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb


ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
WORK = ROOT / "work" / "xuetangx_experiment4"
OUT = ROOT / "outputs" / "xuetangx_experiment4"
DB = WORK / "events.duckdb"
COURSES = ROOT / "outputs" / "xuetangx_data_audit" / "course_inventory.csv"
ACTIONS = [
    "stop_video", "pause_video", "play_video", "click_courseware", "seek_video",
    "problem_get", "load_video", "close_courseware", "problem_check", "click_info",
    "problem_check_correct", "problem_check_incorrect", "click_about", "click_forum",
    "click_progress", "problem_save", "create_comment", "create_thread",
    "delete_thread", "delete_comment", "reset_problem", "close_forum",
]


def sqlpath(path):
    return path.as_posix().replace("'", "''")


con = duckdb.connect(str(DB))
con.execute("SET threads=" + __import__("os").environ.get("CALTRANSFER_THREADS", "4"))
con.execute("SET memory_limit='" + __import__("os").environ.get("CALTRANSFER_MEMORY", "4GB") + "'")
con.execute(f"SET temp_directory='{sqlpath(WORK / 'duckdb_temp')}'")
if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='events'").fetchone()[0]:
    train = sqlpath(WORK / "raw" / "train.csv")
    test = sqlpath(WORK / "raw" / "test.csv")
    course_file = sqlpath(COURSES)
    print("Materializing supplied event records", flush=True)
    con.execute(f"""
        CREATE TABLE anchors AS
        SELECT course_id, CAST(first_observed_event AS TIMESTAMP) AS anchor
        FROM read_csv('{course_file}', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE TABLE events AS
        WITH raw AS (
            SELECT enroll_id, course_id, session_id, action, "object", "time"
            FROM read_csv('{train}', header=true, all_varchar=true)
            UNION ALL
            SELECT enroll_id, course_id, session_id, action, "object", "time"
            FROM read_csv('{test}', header=true, all_varchar=true)
        )
        SELECT CAST(r.enroll_id AS BIGINT) AS enroll_id, r.course_id,
               CAST(r."time" AS TIMESTAMP) AS event_time,
               CAST(r."time" AS DATE) AS event_date,
               date_diff('second', a.anchor, CAST(r."time" AS TIMESTAMP)) / 86400.0 AS day_exact,
               r.session_id, r.action, nullif(r."object", '') AS object_id
        FROM raw r JOIN anchors a USING (course_id)
    """)
    n = con.execute("SELECT count(*) FROM events").fetchone()[0]
    assert __import__("os").environ.get("CALTRANSFER_SMOKE") == "1" or n == 42_110_402, n
    print(f"events table complete: {n:,}", flush=True)

action_sql = ",\n".join(
    f"SUM(CASE WHEN action = '{action}' THEN 1 ELSE 0 END) AS action_{action}_count"
    for action in ACTIONS
)
for landmark in (7, 14, 21, 28):
    target = WORK / f"features_day{landmark}.parquet"
    if target.exists():
        print(f"Day {landmark} feature file exists", flush=True)
        continue
    print(f"Building Day {landmark} features", flush=True)
    con.execute(f"""
        COPY (
            WITH filtered AS (
                SELECT * FROM events WHERE day_exact >= 0 AND day_exact <= {landmark}
            ), base AS (
                SELECT enroll_id, course_id,
                       count(*)::INTEGER AS event_count,
                       count(DISTINCT event_date)::INTEGER AS active_days,
                       count(DISTINCT session_id)::INTEGER AS unique_sessions,
                       count(DISTINCT object_id)::INTEGER AS unique_objects,
                       min(day_exact) AS first_activity_lag,
                       max(day_exact) AS last_activity_lag,
                       sum(CASE WHEN day_exact <= {landmark / 2} THEN 1 ELSE 0 END)::INTEGER AS events_first_half,
                       sum(CASE WHEN day_exact > {landmark / 2} THEN 1 ELSE 0 END)::INTEGER AS events_second_half,
                       count(DISTINCT CASE WHEN day_exact <= {landmark / 2} THEN event_date END)::INTEGER AS active_days_first_half,
                       count(DISTINCT CASE WHEN day_exact > {landmark / 2} THEN event_date END)::INTEGER AS active_days_second_half,
                       sum(CASE WHEN dayofweek(event_date) IN (0,6) THEN 1 ELSE 0 END)::INTEGER AS weekend_events,
                       {action_sql}
                FROM filtered GROUP BY enroll_id, course_id
            ), daily AS (
                SELECT enroll_id, event_date, count(*) AS daily_events
                FROM filtered GROUP BY enroll_id, event_date
            ), daily_summary AS (
                SELECT enroll_id, max(daily_events)::INTEGER AS max_events_per_day,
                       stddev_pop(daily_events) AS std_events_per_active_day
                FROM daily GROUP BY enroll_id
            )
            SELECT b.*,
                   b.event_count::DOUBLE / b.active_days AS events_per_active_day,
                   b.unique_sessions::DOUBLE / b.active_days AS sessions_per_active_day,
                   b.last_activity_lag - b.first_activity_lag AS activity_span,
                   {landmark} - b.last_activity_lag AS recency_to_landmark,
                   b.active_days::DOUBLE / ({landmark} + 1) AS active_day_ratio,
                   b.events_second_half::DOUBLE / b.event_count AS event_second_half_ratio,
                   b.event_count::DOUBLE / b.active_days AS mean_events_per_active_day,
                   b.weekend_events::DOUBLE / b.event_count AS weekend_event_fraction,
                   b.unique_objects::DOUBLE / b.event_count AS unique_objects_per_event,
                   d.max_events_per_day, coalesce(d.std_events_per_active_day,0) AS std_events_per_active_day
            FROM base b JOIN daily_summary d USING (enroll_id)
        ) TO '{sqlpath(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{sqlpath(target)}')").fetchone()[0]
    print(f"Day {landmark}: {n:,} enrollment features", flush=True)

manifest = {"events": con.execute("SELECT count(*) FROM events").fetchone()[0], "actions": ACTIONS,
            "feature_files": {str(d): str(WORK / f"features_day{d}.parquet") for d in (7,14,21,28)},
            "feature_cutoff": "event timestamp <= first observed event in presentation + landmark days"}
(OUT / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
con.close()
