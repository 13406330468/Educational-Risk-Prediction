"""Summarize the two additional experiments with pair-equal cluster intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_integrity import validate_job_coverage

ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
OUT = ROOT / "outputs" / "xuetangx_gap_supplement"
PRIMARY = ROOT / "outputs" / "xuetangx_experiment4"
ASSIGNMENT = pd.read_csv(PRIMARY / "supplement" / "unique_user_course_assignment.csv",
                         dtype={"username": str})
ORIG = pd.read_csv(PRIMARY / "primary_by_seed.csv")
ORIG = ORIG[ORIG.day.isin([7, 14, 28])]
REFIT = pd.read_csv(OUT / "identity_refit_by_seed.csv")
KEY = ["pair_id", "day", "seed", "base_model"]
assert not ORIG[KEY].duplicated().any()
assert not REFIT[KEY].duplicated().any()
assert REFIT[KEY].merge(ORIG[KEY], on=KEY).shape[0] == len(REFIT)
validate_job_coverage(REFIT, pd.read_csv(OUT / 'identity_refit_support.csv'),
                      pd.read_csv(OUT / 'identity_refit_failures.csv'), 'original-label refit summary')


def bootstrap_pair_macro(pair, metrics, seed):
    pair = pair.copy()
    pair["cluster"] = pair.provider + "|" + pair.base_course_token
    clusters = sorted(pair.cluster.unique())
    arr = pair[metrics].to_numpy(dtype=float)
    sums = np.stack([arr[pair.cluster.eq(c)].sum(axis=0) for c in clusters])
    ns = np.array([pair.cluster.eq(c).sum() for c in clusters])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(2000, len(clusters)))
    draws = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)[:, None]
    out = {"pairs": len(pair), "base_course_clusters": len(clusters)}
    for j, metric in enumerate(metrics):
        out[metric] = float(arr[:, j].mean())
        out[f"{metric}_ci_low"] = float(np.quantile(draws[:, j], .025))
        out[f"{metric}_ci_high"] = float(np.quantile(draws[:, j], .975))
    return out


joined = ORIG.merge(REFIT, on=KEY, suffixes=("_original", "_refit"), validate="one_to_one")
for metric in ("G_S", "G_T", "G_T_star", "A"):
    assert np.isfinite(joined[f"{metric}_original"]).all()
    assert np.isfinite(joined[f"{metric}_refit"]).all()
joined["A_refit_minus_original"] = joined.A_refit - joined.A_original
joined["provider"] = joined.provider_refit
joined["base_course_token"] = joined.base_course_token_refit
joined[[*KEY, "provider", "base_course_token", "G_S_original", "G_T_original", "G_T_star_original",
        "A_original", "G_S_refit", "G_T_refit", "G_T_star_refit", "A_refit", "A_refit_minus_original",
        "source_cal_n_refit", "target_n_refit"]].to_csv(OUT / "identity_refit_common_jobs.csv", index=False)

pair_keys = ["pair_id", "provider", "base_course_token", "day", "base_model"]
identity_metrics = ["G_S_original", "G_T_original", "G_T_star_original", "A_original",
                    "G_S_refit", "G_T_refit", "G_T_star_refit", "A_refit", "A_refit_minus_original"]
pair = joined.groupby(pair_keys, as_index=False)[identity_metrics].mean()
seeds = joined.groupby(pair_keys).size().rename("matched_seeds").reset_index()
pair = pair.merge(seeds, on=pair_keys, validate="one_to_one")
pair.to_csv(OUT / "identity_refit_pair_mean.csv", index=False)
identity_summary = []
for (day, model), g in pair.groupby(["day", "base_model"]):
    row = {"day": int(day), "base_model": model,
           "matched_jobs": int(len(joined[(joined.day == day) & (joined.base_model == model)])),
           "original_jobs": int(len(ORIG[(ORIG.day == day) & (ORIG.base_model == model)])),
           "positive_A_pair_fraction": float(g.A_refit.gt(0).mean()),
           "target_n_refit_pair_mean": float(joined[(joined.day == day) & (joined.base_model == model)].target_n_refit.mean())}
    row.update(bootstrap_pair_macro(g, identity_metrics, 20261002 + day * 11 + (model == "lightgbm")))
    identity_summary.append(row)
identity_summary = pd.DataFrame(identity_summary).sort_values(["day", "base_model"])
identity_summary.to_csv(OUT / "identity_refit_summary.csv", index=False)

futures = []
supports = []
for day in (7, 14, 28):
    path = OUT / "future_endpoint" / f"day{day}"
    df = pd.read_csv(path / "identity_refit_by_seed.csv")
    assert set(df.day) == {day}
    futures.append(df)
    sup = pd.read_csv(path / "identity_refit_support.csv")
    validate_job_coverage(df, sup, pd.read_csv(path / 'identity_refit_failures.csv'), f'future day {day} summary')
    supports.append(sup)
FUTURE = pd.concat(futures, ignore_index=True)
FUTURE_SUPPORT = pd.concat(supports, ignore_index=True)
FUTURE.to_csv(OUT / "future_endpoint_by_seed.csv", index=False)
FUTURE_SUPPORT.to_csv(OUT / "future_endpoint_support.csv", index=False)
assert not FUTURE[KEY].duplicated().any()
assert np.allclose(FUTURE.A, FUTURE.G_T_star - FUTURE.G_T)
future_metrics = ["G_S", "G_T", "G_T_star", "A", "target_extra_slope_gain"]
future_pair = FUTURE.groupby(pair_keys, as_index=False)[future_metrics].mean()
future_pair["matched_seeds"] = FUTURE.groupby(pair_keys).size().to_numpy()
future_pair.to_csv(OUT / "future_endpoint_pair_mean.csv", index=False)
future_summary = []
for (day, model), g in future_pair.groupby(["day", "base_model"]):
    row = {"day": int(day), "base_model": model,
           "jobs": int(len(FUTURE[(FUTURE.day == day) & (FUTURE.base_model == model)])),
           "positive_A_pair_fraction": float(g.A.gt(0).mean())}
    row.update(bootstrap_pair_macro(g, future_metrics, 20261002 + 1000 + day * 11 + (model == "lightgbm")))
    future_summary.append(row)
future_summary = pd.DataFrame(future_summary).sort_values(["day", "base_model"])
future_summary.to_csv(OUT / "future_endpoint_summary.csv", index=False)

course_map = pd.concat([
    pd.read_csv(PRIMARY / "eligible_pairs.csv")[["provider", "base_course_token", "source_course_id"]]
    .rename(columns={"source_course_id": "course_id"}),
    pd.read_csv(PRIMARY / "eligible_pairs.csv")[["provider", "base_course_token", "target_course_id"]]
    .rename(columns={"target_course_id": "course_id"}),
]).drop_duplicates()
course_map["key"] = course_map.provider + "|" + course_map.base_course_token
enroll = pd.read_csv(ROOT / "work" / "xuetangx_audit" / "enrollment_index.csv",
                     usecols=["enroll_id", "username", "course_id"], dtype={"username": str})
filtered = enroll.merge(course_map[["course_id", "key"]], on="course_id", how="inner")
filtered = filtered.merge(ASSIGNMENT[["username", "base_course_key"]], on="username", how="inner")
filtered = filtered[filtered.key.eq(filtered.base_course_key)]
max_base_courses_per_user = int(filtered.groupby("username").key.nunique().max())
assert max_base_courses_per_user == 1
verification = {
    "status": "passed",
    "eligible_unique_users": int(len(ASSIGNMENT)),
    "eligible_users_with_multiple_base_courses_before_assignment": int(ASSIGNMENT.eligible_base_courses_entered.gt(1).sum()),
    "retained_enrollments": int(len(filtered)),
    "max_eligible_base_courses_per_user_after_assignment": max_base_courses_per_user,
    "original_primary_jobs_three_landmarks": int(len(ORIG)),
    "full_identity_refit_jobs": int(len(REFIT)),
    "refit_failure_rows": int(len(pd.read_csv(OUT / "identity_refit_failures.csv"))),
    "common_primary_refit_jobs": int(len(joined)),
    "max_identity_refit_A_arithmetic_error": float((REFIT.A - (REFIT.G_T_star - REFIT.G_T)).abs().max()),
    "future_endpoint_jobs": int(len(FUTURE)),
    "max_future_endpoint_A_arithmetic_error": float((FUTURE.A - (FUTURE.G_T_star - FUTURE.G_T)).abs().max()),
    "original_truth_window_verified": False,
}
(OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
print(identity_summary[["day", "base_model", "pairs", "base_course_clusters", "A_original", "A_refit",
                        "A_refit_ci_low", "A_refit_ci_high", "A_refit_minus_original"]].to_string(index=False), flush=True)
print(future_summary[["day", "base_model", "pairs", "base_course_clusters", "A", "A_ci_low", "A_ci_high"]].to_string(index=False), flush=True)
print(json.dumps(verification, indent=2), flush=True)
