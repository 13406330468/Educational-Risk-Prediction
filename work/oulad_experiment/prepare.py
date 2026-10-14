"""Prepare leakage-controlled OULAD landmark features directly from its ZIP.

Run: py -3.12 work/oulad_experiment/prepare.py
Dependencies: numpy, pandas. No score column is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["code_module", "code_presentation", "id_student"]
COURSE_KEYS = KEYS[:2]
STAGES = (7, 14, 28, 56)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        __import__("os").environ["CALTRANSFER_OULAD_ZIP"]))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--stages", type=int, nargs="+", default=list(STAGES),
                        help="Prediction days; defaults to the original four calendar landmarks")
    args = parser.parse_args()
    stages = tuple(sorted(set(args.stages)))
    if not stages or min(stages) < 0:
        raise ValueError("Prediction days must be nonnegative integers")
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    audit: dict = {
        "source_zip": str(args.zip.resolve()),
        "stages": list(stages),
        "prediction_time": "end of day t; date <= t is observable",
        "behavior_window": "VLE and current-course submission volume: day 0 through day t inclusive. Known prestart non-banked submissions count as already submitted for deadline status.",
        "risk_set": "registration <= t and (unregistration > t or missing)",
        "outcome": "1 = Fail or Withdrawn; 0 = Pass or Distinction",
        "schedule_assumption": "Non-exam due dates and weights in assessments.csv are assumed known at course start. Future scheduled dates are metadata, not future student evidence.",
        "activity_definition": "positive-click VLE records; active days have sum_click > 0",
        "submission_definition": "Distinct non-banked non-exam assessments with known due dates, submitted <= t (including prestart submissions for deadline status). Current-course counts restrict to days 0..t; prestart counts are separate. No score column is loaded.",
        "forbidden_predictors": ["final_result", "y", "id_student", "date_unregistration", "score", "is_banked", "future observations"],
        "recent7_definition": "max(0,t-6)..t inclusive",
        "previous7_definition": "max(0,t-13)..t-7 inclusive (partial window at day 7)",
        "warnings": [],
    }
    hasher = hashlib.sha256()
    with args.zip.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    audit["source_sha256"] = hasher.hexdigest()

    with zipfile.ZipFile(args.zip) as archive:
        def read(name: str, **kwargs) -> pd.DataFrame:
            with archive.open(name) as stream:
                return pd.read_csv(stream, na_values=["?"], **kwargs)

        names = archive.read("OULAD.names").decode("cp1252")
        audit["names_read"] = True
        audit["names_sha256"] = hashlib.sha256(names.encode("utf-8")).hexdigest()
        students = read("studentInfo.csv", usecols=KEYS + ["final_result"])
        registration = read("studentRegistration.csv")
        courses = read("courses.csv")
        assessments = read("assessments.csv")
        submissions = read("studentAssessment.csv", usecols=[
            "id_assessment", "id_student", "date_submitted", "is_banked"])
        resources = read("vle.csv", usecols=[
            "id_site", "code_module", "code_presentation", "activity_type"])

        for name, table, key in [
            ("studentInfo", students, KEYS),
            ("studentRegistration", registration, KEYS),
            ("courses", courses, COURSE_KEYS),
            ("assessments", assessments, ["id_assessment"]),
            ("vle", resources, COURSE_KEYS + ["id_site"]),
        ]:
            duplicates = int(table.duplicated(key).sum())
            audit.setdefault("duplicate_keys", {})[name] = duplicates
            if duplicates:
                raise ValueError(f"{name} has {duplicates} duplicate keys")
        if not students["final_result"].isin(["Pass", "Distinction", "Fail", "Withdrawn"]).all():
            raise ValueError("Unexpected outcome labels")

        base = students.merge(registration, on=KEYS, how="left", validate="one_to_one", indicator=True)
        audit["registration_join_unmatched"] = int((base["_merge"] != "both").sum())
        base = base.drop(columns="_merge").reset_index(drop=True)
        base["_row"] = np.arange(len(base), dtype=np.int32)
        base["y"] = base["final_result"].isin(["Fail", "Withdrawn"]).astype(np.int8)
        index = base[KEYS + ["_row"]]
        n = len(base)
        missing_registration = base["date_registration"].isna()
        missing_withdrawal = base["final_result"].eq("Withdrawn") & base["date_unregistration"].isna()
        valid_base = ~(missing_registration | missing_withdrawal)
        audit["original_enrollments"] = n
        audit["original_unique_students"] = int(base["id_student"].nunique())
        audit["original_course_runs"] = int(len(courses))
        audit["original_outcomes"] = {str(k): int(v) for k, v in base["final_result"].value_counts().items()}
        audit["exclusions"] = {
            "missing_registration": int(missing_registration.sum()),
            "withdrawn_missing_unregistration": int(missing_withdrawal.sum()),
            "overlap": int((missing_registration & missing_withdrawal).sum()),
            "union": int((~valid_base).sum()),
        }
        audit["registration_anomalies"] = {
            "non_withdrawn_with_unregistration": int((~base["final_result"].eq("Withdrawn") & base["date_unregistration"].notna()).sum()),
            "unregistration_before_registration": int((base["date_unregistration"] < base["date_registration"]).sum()),
        }

        # Exact cumulative aggregates: day-level counts and first site-access dates.
        # Retain only within-window compact (enrollment, site, first day) pairs.
        daily = np.zeros((n, max(stages) + 1), dtype=np.int64)
        type_names = sorted(resources["activity_type"].dropna().unique().tolist())
        type_codes = {name: k for k, name in enumerate(type_names)}
        resources["_type"] = resources["activity_type"].map(type_codes).astype("int16")
        type_clicks = np.zeros((n, len(stages), len(type_names)), dtype=np.int64)
        site_parts: list[pd.DataFrame] = []
        log_audit = {key: 0 for key in [
            "rows", "negative_day_rows", "after_last_stage_rows", "missing_date_rows",
            "noninteger_date_rows", "missing_click_rows", "negative_click_rows",
            "zero_click_rows", "unknown_enrollment_rows", "unknown_resource_rows",
            "retained_positive_click_rows", "retained_clicks"]}
        dtype = {"code_module": "string", "code_presentation": "string",
                 "id_student": "int64", "id_site": "int64", "date": "float64", "sum_click": "float64"}
        with archive.open("studentVle.csv") as stream:
            reader = pd.read_csv(stream, na_values=["?"], dtype=dtype, chunksize=args.chunk_size)
            for chunk_number, chunk in enumerate(reader, 1):
                log_audit["rows"] += len(chunk)
                dates, clicks = chunk["date"], chunk["sum_click"]
                log_audit["negative_day_rows"] += int(dates.lt(0).sum())
                log_audit["after_last_stage_rows"] += int(dates.gt(max(stages)).sum())
                log_audit["missing_date_rows"] += int(dates.isna().sum())
                log_audit["noninteger_date_rows"] += int((dates.notna() & dates.mod(1).ne(0)).sum())
                log_audit["missing_click_rows"] += int(clicks.isna().sum())
                log_audit["negative_click_rows"] += int(clicks.lt(0).sum())
                log_audit["zero_click_rows"] += int(clicks.eq(0).sum())
                if (clicks.notna() & clicks.mod(1).ne(0)).any():
                    raise ValueError("Non-integer VLE click count")
                keep = dates.between(0, max(stages)) & dates.mod(1).eq(0) & clicks.gt(0)
                observed = chunk.loc[keep].merge(index, on=KEYS, how="left", validate="many_to_one")
                log_audit["unknown_enrollment_rows"] += int(observed["_row"].isna().sum())
                observed = observed.loc[observed["_row"].notna()].copy()
                if observed.empty:
                    continue
                observed = observed.merge(resources[COURSE_KEYS + ["id_site", "_type"]],
                                          on=COURSE_KEYS + ["id_site"], how="left", validate="many_to_one")
                log_audit["unknown_resource_rows"] += int(observed["_type"].isna().sum())
                if observed["_type"].isna().any():
                    raise ValueError("Unmatched VLE resource; activity type cannot be assigned safely")
                row = observed["_row"].to_numpy(dtype=np.int32)
                day = observed["date"].to_numpy(dtype=np.int16)
                click = observed["sum_click"].to_numpy(dtype=np.int64)
                activity = observed["_type"].to_numpy(dtype=np.int16)
                np.add.at(daily, (row, day), click)
                for stage_index, t in enumerate(stages):
                    mask = day <= t
                    np.add.at(type_clicks[:, stage_index], (row[mask], activity[mask]), click[mask])
                site_parts.append(observed.assign(_row=row, date=day)
                                  .groupby(["_row", "id_site"], as_index=False, sort=False)["date"].min())
                log_audit["retained_positive_click_rows"] += len(observed)
                log_audit["retained_clicks"] += int(click.sum())
                print(f"VLE chunk {chunk_number}: scanned {log_audit['rows']:,}; kept {log_audit['retained_positive_click_rows']:,}", flush=True)
        audit["vle"] = log_audit
        sites = pd.concat(site_parts, ignore_index=True).groupby(["_row", "id_site"], as_index=False, sort=False)["date"].min()
        del site_parts
        if int(daily.sum()) != log_audit["retained_clicks"]:
            raise AssertionError("Click aggregation mismatch")

        # Known non-exam schedule; no score values are ever read or joined.
        nonexam = assessments["assessment_type"].ne("Exam")
        known_due = assessments["date"].notna() & assessments["date"].mod(1).eq(0)
        eligible_assessments = assessments.loc[nonexam & known_due].copy()
        schedule = eligible_assessments.groupby(COURSE_KEYS).agg(
            first_due_day=("date", "min"),
        ).reset_index()
        audit["assessments"] = {
            "metadata_rows": len(assessments),
            "exam_rows_excluded": int((~nonexam).sum()),
            "nonexam_missing_or_invalid_due_excluded": int((nonexam & ~known_due).sum()),
            "nonexam_negative_due_dates": int((nonexam & assessments["date"].lt(0)).sum()),
            "missing_weight_rows": int(eligible_assessments["weight"].isna().sum()),
            "submission_rows": len(submissions),
            "banked_rows_excluded": int(submissions["is_banked"].eq(1).sum()),
            "missing_banked_status_rows": int(submissions["is_banked"].isna().sum()),
            "missing_submission_day_rows": int(submissions["date_submitted"].isna().sum()),
            "negative_submission_day_rows": int(submissions["date_submitted"].lt(0).sum()),
            "duplicate_student_assessment_rows": int(submissions.duplicated(["id_student", "id_assessment"]).sum()),
        }
        if eligible_assessments["weight"].isna().any():
            audit["warnings"].append("Missing known schedule weights are imputed to zero; counts remain valid.")
        submitted = submissions.loc[
            submissions["is_banked"].eq(0)
            & submissions["date_submitted"].le(max(stages))
            & submissions["date_submitted"].mod(1).eq(0)
        ].merge(eligible_assessments, on="id_assessment", how="inner", validate="many_to_one")
        submitted = submitted.merge(index, on=KEYS, how="left", validate="many_to_one")
        audit["assessments"]["unknown_submission_enrollment_rows"] = int(submitted["_row"].isna().sum())
        submitted = submitted.loc[submitted["_row"].notna()].copy()
        # Deterministic first submission if a duplicate appears; prevents overcounting.
        submitted = submitted.sort_values("date_submitted").drop_duplicates(["_row", "id_assessment"], keep="first")
        submitted["_row"] = submitted["_row"].astype(np.int32)
        audit["assessments"]["by_day56_eligible_submissions"] = len(submitted)
        audit["assessments"]["prestart_eligible_submissions"] = int(submitted["date_submitted"].lt(0).sum())

        frames: list[pd.DataFrame] = []
        stage_audits: list[dict] = []
        for stage_index, t in enumerate(stages):
            risk = valid_base & base["date_registration"].le(t) & (
                base["date_unregistration"].isna() | base["date_unregistration"].gt(t))
            out = base.loc[risk, KEYS + ["_row", "y", "final_result"]].copy()
            row_ids = out["_row"].to_numpy(dtype=np.int32)
            out["t"] = t
            out["stage"] = t
            out["course"] = out["code_module"]
            day_clicks = daily[row_ids, :t + 1]
            out["total_clicks"] = day_clicks.sum(axis=1)
            out["active_days"] = (day_clicks > 0).sum(axis=1)
            site_count = np.bincount(sites.loc[sites["date"].le(t), "_row"].to_numpy(dtype=np.int32), minlength=n)
            out["unique_sites"] = site_count[row_ids]
            recent = daily[row_ids, max(0, t - 6):t + 1]
            previous = daily[row_ids, max(0, t - 13):max(0, t - 6)]
            out["recent7_clicks"] = recent.sum(axis=1)
            out["recent7_active_days"] = (recent > 0).sum(axis=1)
            out["previous7_clicks"] = previous.sum(axis=1)
            out["previous7_active_days"] = (previous > 0).sum(axis=1)
            out["recent7_observable_days"] = recent.shape[1]
            out["previous7_observable_days"] = previous.shape[1]
            out["weekly_click_change"] = out["recent7_clicks"] - out["previous7_clicks"]
            activity_days = np.where(day_clicks > 0, np.arange(t + 1)[None, :], -1)
            out["last_activity_gap"] = t - activity_days.max(axis=1)
            out["no_vle_activity"] = out["total_clicks"].eq(0).astype(np.int8)
            for activity_index, activity_name in enumerate(type_names):
                out[f"activity_clicks_{activity_name}"] = type_clicks[row_ids, stage_index, activity_index]
            out["unique_activity_types"] = (type_clicks[row_ids, stage_index] > 0).sum(axis=1)
            out = out.merge(schedule, on=COURSE_KEYS, how="left", validate="many_to_one")
            out["no_known_nonexam_schedule"] = out["first_due_day"].isna().astype(np.int8)
            out["first_due_day"] = out["first_due_day"].fillna(-1)
            due = eligible_assessments.loc[eligible_assessments["date"].le(t)]
            due_summary = due.groupby(COURSE_KEYS).agg(
                due_count=("id_assessment", "nunique"), due_weight=("weight", "sum"))
            out = out.merge(due_summary.reset_index(), on=COURSE_KEYS, how="left", validate="many_to_one")
            out[["due_count", "due_weight"]] = out[["due_count", "due_weight"]].fillna(0)
            at_t = submitted.loc[submitted["date_submitted"].le(t)]
            due_submitted = at_t.loc[at_t["date"].le(t)]
            on_time_due = due_submitted.loc[due_submitted["date_submitted"].le(due_submitted["date"])]
            for column, source in [
                ("submitted_count", at_t.loc[at_t["date_submitted"].ge(0)]),
                ("prestart_submitted_count", at_t.loc[at_t["date_submitted"].lt(0)]),
                ("due_submitted_count", due_submitted),
                ("due_on_time_count", on_time_due),
            ]:
                counts = source.groupby("_row")["id_assessment"].nunique()
                out[column] = out["_row"].map(counts).fillna(0).astype(np.int32)
            out["overdue_count"] = out["due_count"] - out["due_submitted_count"]
            out["due_late_submitted_count"] = out["due_submitted_count"] - out["due_on_time_count"]
            out["submitted_not_yet_due_count"] = out["submitted_count"] + out["prestart_submitted_count"] - out["due_submitted_count"]
            out["no_due_assessments"] = out["due_count"].eq(0).astype(np.int8)
            out["days_since_first_due"] = np.where(
                out["no_known_nonexam_schedule"].eq(0), np.maximum(t - out["first_due_day"], 0), 0)
            out["progress_state"] = np.select([
                out["due_count"].eq(0), out["overdue_count"].eq(0)], ["no_due", "due_complete"], default="due_missing")
            if (out["overdue_count"] < 0).any():
                raise AssertionError("More due submissions than scheduled assessments")
            if not np.array_equal(out.filter(regex="^activity_clicks_").sum(axis=1), out["total_clicks"]):
                raise AssertionError("Resource-type click totals do not match total clicks")
            out = out.drop(columns="_row")
            stage_audits.append({
                "t": t,
                "risk_enrollments": len(out),
                "unique_students": int(out["id_student"].nunique()),
                "positive_count": int(out["y"].sum()),
                "prevalence": float(out["y"].mean()),
                "not_yet_registered": int((valid_base & base["date_registration"].gt(t)).sum()),
                "already_unregistered": int((valid_base & base["date_registration"].le(t) & base["date_unregistration"].le(t)).sum()),
                "zero_vle_students_kept": int(out["no_vle_activity"].sum()),
                "progress_counts": {str(k): int(v) for k, v in out["progress_state"].value_counts().items()},
                "by_course_presentation": out.groupby(COURSE_KEYS).agg(
                    n=("y", "size"), positives=("y", "sum"), zero_vle=("no_vle_activity", "sum")).reset_index().to_dict("records"),
            })
            frames.append(out)

    result = pd.concat(frames, ignore_index=True).sort_values(["t"] + KEYS).reset_index(drop=True)
    metadata = KEYS + ["t", "stage", "course", "y", "final_result", "progress_state"]
    feature_columns = [column for column in result.columns if column not in metadata]
    if result[feature_columns].isna().any().any():
        raise AssertionError("Missing model feature values")
    if not np.isfinite(result[feature_columns].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite model feature values")
    if result.duplicated(KEYS + ["t"]).any():
        raise AssertionError("Duplicate landmark enrollments")
    output_file = args.output / "features.csv.gz"
    result.to_csv(output_file, index=False, compression={"method": "gzip", "mtime": 0})
    audit["stage_audits"] = stage_audits
    audit["metadata_columns"] = metadata
    audit["feature_columns"] = feature_columns
    audit["feature_count"] = len(feature_columns)
    audit["output_rows"] = len(result)
    audit["output_file"] = str(output_file.resolve())
    audit["elapsed_seconds"] = round(time.monotonic() - started, 2)
    audit["versions"] = {"pandas": pd.__version__, "numpy": np.__version__}
    with (args.output / "data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: audit[key] for key in ["output_rows", "feature_count", "elapsed_seconds", "output_file"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
