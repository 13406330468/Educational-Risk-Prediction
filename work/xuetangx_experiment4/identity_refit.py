"""Full one-base-course-per-user refit of the frozen Experiment 4 pipeline.

Run with the bundled Python. The original protocol and results are never edited.
This script is resumable: each completed pair/day/model/seed is cached separately.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import primary as p
from run_integrity import validate_job_coverage

ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
OUT = ROOT / "outputs" / "xuetangx_gap_supplement"
WORK = ROOT / "work" / "xuetangx_experiment4" / "identity_refit"
PRED = WORK / "predictions"
for path in (OUT, WORK, PRED):
    path.mkdir(parents=True, exist_ok=True)

DAYS = (7, 14, 28)
SEEDS = (17, 29, 43, 59, 71)
MODELS = ("logistic", "lightgbm")
ASSIGNMENT_PATH = p.OUT / "supplement" / "unique_user_course_assignment.csv"
ASSIGNMENT = pd.read_csv(ASSIGNMENT_PATH, dtype={"username": str, "base_course_key": str})
assert ASSIGNMENT.username.is_unique
COURSE_MAP = pd.concat([
    p.PAIRS[["provider", "base_course_token", "source_course_id"]].rename(columns={"source_course_id": "course_id"}),
    p.PAIRS[["provider", "base_course_token", "target_course_id"]].rename(columns={"target_course_id": "course_id"}),
], ignore_index=True).drop_duplicates()
assert COURSE_MAP.course_id.is_unique
COURSE_MAP["base_course_key"] = COURSE_MAP.provider + "|" + COURSE_MAP.base_course_token
course_key = dict(zip(COURSE_MAP.course_id, COURSE_MAP.base_course_key))
user_key = dict(zip(ASSIGNMENT.username, ASSIGNMENT.base_course_key))
enroll = p.ENROLL[p.ENROLL.course_id.isin(course_key)].copy()
enroll["assignment_key"] = enroll.username.astype(str).map(user_key)
enroll = enroll[enroll.assignment_key.eq(enroll.course_id.map(course_key))].drop(columns="assignment_key")
assert enroll.groupby("username").course_id.apply(lambda x: len({course_key[c] for c in x})).max() == 1
p.BY_COURSE = {k: g for k, g in enroll.groupby("course_id")}
p.PRED = PRED

protocol = {
    "name": "full independent refit after cross-base-course user assignment",
    "date": "2026-10-02",
    "source_data": "frozen Experiment 4 XuetangX course pairs and pre-landmark features",
    "assignment": "earliest observed eligible base course per username; outcome-blind; existing frozen assignment reused",
    "assignment_sha256": hashlib.sha256(ASSIGNMENT_PATH.read_bytes()).hexdigest(),
    "assignment_scope": "all source/target enrollments in the 40 eligible base courses; users can occur in only one eligible base course",
    "pair_scope": "original 67 eligible adjacent same-base-course presentation pairs, no new pairing",
    "landmarks": DAYS,
    "models": MODELS,
    "seeds": SEEDS,
    "source_roles": "regenerate 60/20/20 stratified train/validation/calibration roles on retained source enrollments for every pair and seed",
    "active_support": "original pre-model thresholds: source train minority>=5, validation minority>=2, calibration N>=30 and minority>=10; target active-unseen N>=50 and minority>=10",
    "models_and_calibrators": "refit original frozen LR and LightGBM, then source intercept/slope/Platt and target five-fold out-of-fold diagnostics on retained data",
    "primary_estimand": "pair-equal macro mean A=G_T_star-G_T after averaging available prespecified seeds within pair",
    "comparison": "evaluate original and independent refit on common supported pair-day-model-seed units; report support attrition separately",
    "uncertainty": "2000 base-course cluster bootstrap resamples, same draw for original and refit",
    "qualification": "target refit remains retrospective because target OOF calibration uses target labels",
}
protocol_path = OUT / "identity_refit_protocol.json"
if protocol_path.exists():
    assert json.loads(protocol_path.read_text(encoding="utf-8")) == json.loads(json.dumps(protocol))
else:
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def make_roles_and_support():
    role_rows = []
    support_rows = []
    for pair in p.PAIRS.itertuples():
        if pair.source_course_id not in p.BY_COURSE or pair.target_course_id not in p.BY_COURSE:
            for day in DAYS:
                for seed in SEEDS:
                    support_rows.append({"pair_id": pair.pair_id, "day": day, "seed": seed,
                                         "admissible": False, "reason": "empty_course_after_assignment"})
            continue
        full_source = p.BY_COURSE[pair.source_course_id].reset_index(drop=True)
        frames = {day: p.prep_pair(pair, day) for day in DAYS}
        for seed in SEEDS:
            split_seed = seed + int(pair.pair_id[1:]) * 1009
            if full_source.y.value_counts().min() < 5 or full_source.y.nunique() < 2:
                for day in DAYS:
                    support_rows.append({"pair_id": pair.pair_id, "day": day, "seed": seed,
                                         "admissible": False, "reason": "source_class_under_5"})
                continue
            try:
                tr, held = train_test_split(np.arange(len(full_source)), test_size=.4,
                                            random_state=split_seed, stratify=full_source.y.to_numpy())
                va, ca = train_test_split(held, test_size=.5, random_state=split_seed + 104729,
                                          stratify=full_source.iloc[held].y.to_numpy())
            except ValueError as exc:
                for day in DAYS:
                    support_rows.append({"pair_id": pair.pair_id, "day": day, "seed": seed,
                                         "admissible": False, "reason": f"split_failed:{exc}"})
                continue
            roles = np.empty(len(full_source), dtype=object)
            roles[tr], roles[va], roles[ca] = "train", "validation", "calibration"
            role_map = dict(zip(full_source.enroll_id, roles))
            role_rows.extend((pair.pair_id, seed, int(e), int(u), r)
                             for e, u, r in zip(full_source.enroll_id, full_source.username, roles))
            for day in DAYS:
                source, target = frames[day]
                active_roles = source.enroll_id.map(role_map)
                row = {"pair_id": pair.pair_id, "day": day, "seed": seed,
                       "source_n": len(source), "target_n": int((target.active & target.unseen).sum()),
                       "source_total_retained": len(full_source)}
                for role in ("train", "validation", "calibration"):
                    group = source.loc[active_roles.eq(role)]
                    row[f"source_{role}_n"] = len(group)
                    row[f"source_{role}_minority"] = int(group.y.value_counts().min()) if group.y.nunique() == 2 else 0
                tgt = target.loc[target.active & target.unseen]
                row["target_minority"] = int(tgt.y.value_counts().min()) if tgt.y.nunique() == 2 else 0
                row["admissible"] = (row["source_train_minority"] >= 5
                                     and row["source_validation_minority"] >= 2
                                     and row["source_calibration_n"] >= 30
                                     and row["source_calibration_minority"] >= 10
                                     and row["target_n"] >= 50 and row["target_minority"] >= 10)
                row["reason"] = "supported" if row["admissible"] else "below_original_support"
                support_rows.append(row)
    roles_df = pd.DataFrame(role_rows, columns=["pair_id", "seed", "enroll_id", "username", "role"])
    support_df = pd.DataFrame(support_rows)
    assert not support_df[["pair_id", "day", "seed"]].duplicated().any()
    roles_df.to_parquet(WORK / "roles.parquet", index=False)
    support_df.to_csv(OUT / "identity_refit_support.csv", index=False)
    return roles_df, support_df


def main():
    roles, support = make_roles_and_support()
    admissible = support[support.admissible].copy()
    print("REFIT_SUPPORT", json.dumps({str(day): int(g.admissible.sum()) for day, g in support.groupby("day")}), flush=True)
    results = []
    failures = []
    total = len(admissible) * len(MODELS)
    n = 0
    for day in DAYS:
        day_support = admissible[admissible.day.eq(day)]
        for pair in p.PAIRS.itertuples():
            jobs = day_support[day_support.pair_id.eq(pair.pair_id)]
            if jobs.empty:
                continue
            source, target = p.prep_pair(pair, day)
            for seed in jobs.seed:
                role_subset = roles[roles.pair_id.eq(pair.pair_id) & roles.seed.eq(seed)]
                role_map = dict(zip(role_subset.enroll_id, role_subset.role))
                for model in MODELS:
                    try:
                        result = p.one_job(pair, day, int(seed), model, source, target, role_map)
                        results.append(result)
                    except Exception as exc:
                        row = {"pair_id": pair.pair_id, "day": day, "seed": int(seed),
                               "base_model": model, "error": repr(exc)}
                        failures.append(row)
                        print("FAILED", json.dumps(row), flush=True)
                    n += 1
                    if n % 50 == 0:
                        print(f"REFIT_PROGRESS {n}/{total} ({len(results)} successful)", flush=True)
    pd.DataFrame(results).to_csv(OUT / "identity_refit_by_seed.csv", index=False)
    pd.DataFrame(failures, columns=["pair_id", "day", "seed", "base_model", "error"]).to_csv(
        OUT / "identity_refit_failures.csv", index=False)
    print("REFIT_DONE", len(results), "of", total, "failures", len(failures), flush=True)
    if failures or len(results) != total:
        raise RuntimeError(f'Incomplete supported refit: {len(results)}/{total} jobs; {len(failures)} failures. See the saved failure diagnostics.')
    validate_job_coverage(pd.DataFrame(results), support, pd.DataFrame(failures), 'identity/future refit')


if __name__ == "__main__":
    sys.exit(main())
