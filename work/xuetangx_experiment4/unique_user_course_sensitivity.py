"""Conditionally de-duplicate target evaluation users across eligible base courses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
OUT = ROOT / "outputs" / "xuetangx_experiment4"
WORK = ROOT / "work" / "xuetangx_experiment4"
SUP = OUT / "supplement"
SUP.mkdir(exist_ok=True)
PLAN = json.loads((WORK / "supplement_protocol.json").read_text(encoding="utf-8"))
DAYS = PLAN["primary_landmarks"]

pairs = pd.read_csv(OUT / "eligible_pairs.csv")
course_map = pd.concat([
    pairs[["provider", "base_course_token", "source_course_id"]].rename(columns={"source_course_id": "course_id"}),
    pairs[["provider", "base_course_token", "target_course_id"]].rename(columns={"target_course_id": "course_id"}),
], ignore_index=True).drop_duplicates()
assert course_map.groupby("course_id")[["provider", "base_course_token"]].nunique().max().max() == 1
course_map["base_course_key"] = course_map.provider + "|" + course_map.base_course_token
enroll = pd.read_csv(ROOT / "work" / "xuetangx_audit" / "enrollment_index.csv",
                     usecols=["enroll_id", "course_id", "username", "first_event"],
                     dtype={"enroll_id": str, "course_id": str, "username": str})
enroll = enroll.merge(course_map[["course_id", "base_course_key"]], on="course_id", how="inner", validate="many_to_one")
enroll["first_event"] = pd.to_datetime(enroll.first_event, errors="raise")
per_base = enroll.groupby(["username", "base_course_key"], as_index=False).first_event.min()
base_counts = per_base.groupby("username").base_course_key.nunique().rename("eligible_base_courses_entered")
assignment = (per_base.sort_values(["username", "first_event", "base_course_key"], kind="stable")
              .drop_duplicates("username", keep="first").merge(base_counts, on="username", validate="one_to_one"))
assert len(assignment) == enroll.username.nunique()
assignment.to_csv(SUP / "unique_user_course_assignment.csv", index=False)
selected_users = {base: set(g.username) for base, g in assignment.groupby("base_course_key")}
pair_base = dict(zip(pairs.pair_id, pairs.provider + "|" + pairs.base_course_token))

primary = pd.read_csv(OUT / "primary_by_seed.csv")
primary = primary.loc[primary.day.isin(DAYS)].sort_values(["pair_id", "day", "base_model", "seed"])
rows = []
max_recompute_error = 0.0
for i, r in enumerate(primary.itertuples(index=False), 1):
    path = WORK / "predictions" / f"{r.pair_id}_d{r.day}_{r.base_model}_s{r.seed}.parquet"
    columns = ["population", "username", "y", "raw", "source_Platt", "oof_Platt", "oof_intercept"]
    f = pd.read_parquet(path, columns=columns)
    f = f.loc[f.population.eq("target_primary")].copy()
    assert len(f) == r.target_n
    mask = f.username.astype(str).isin(selected_users.get(pair_base[r.pair_id], set()))
    retained = f.loc[mask]
    y_all = f.y.to_numpy(dtype=float)
    raw_all = f.raw.to_numpy(dtype=float)
    src_all = f.source_Platt.to_numpy(dtype=float)
    oof_all = f.oof_Platt.to_numpy(dtype=float)
    original_a = float(np.mean((y_all - src_all) ** 2 - (y_all - oof_all) ** 2))
    max_recompute_error = max(max_recompute_error, abs(original_a - r.A))
    row = {"pair_id": r.pair_id, "provider": r.provider, "base_course_token": r.base_course_token,
           "day": r.day, "base_model": r.base_model, "seed": r.seed,
           "target_n_original": len(f), "target_n_retained": len(retained),
           "retained_fraction": len(retained) / len(f),
           "target_retained_0": int(retained.y.eq(0).sum()),
           "target_retained_1": int(retained.y.eq(1).sum()),
           "A_original": original_a, "G_T_original": r.G_T,
           "G_T_star_original": r.G_T_star}
    if len(retained):
        y = retained.y.to_numpy(dtype=float)
        raw = retained.raw.to_numpy(dtype=float)
        src = retained.source_Platt.to_numpy(dtype=float)
        oof = retained.oof_Platt.to_numpy(dtype=float)
        intercept = retained.oof_intercept.to_numpy(dtype=float)
        row.update({"G_T_retained": float(np.mean((y - raw) ** 2 - (y - src) ** 2)),
                    "G_T_star_retained": float(np.mean((y - raw) ** 2 - (y - oof) ** 2)),
                    "A_retained": float(np.mean((y - src) ** 2 - (y - oof) ** 2)),
                    "G_T_star_intercept_retained": float(np.mean((y - raw) ** 2 - (y - intercept) ** 2)),
                    "target_extra_slope_gain_retained": float(np.mean((y - intercept) ** 2 - (y - oof) ** 2))})
    else:
        row.update({k: np.nan for k in ("G_T_retained", "G_T_star_retained", "A_retained",
                                            "G_T_star_intercept_retained", "target_extra_slope_gain_retained")})
    row["below_original_target_support"] = (len(retained) < 50 or min(row["target_retained_0"], row["target_retained_1"]) < 10)
    rows.append(row)
    if i % 250 == 0:
        print(f"unique-user sensitivity {i}/{len(primary)}", flush=True)
jobs = pd.DataFrame(rows)
jobs.to_csv(SUP / "unique_user_course_by_seed.csv", index=False)
assert max_recompute_error < 1e-10, max_recompute_error
assert jobs.target_n_retained.gt(0).all(), "At least one original primary job had no retained target user"
assert np.allclose(jobs.A_retained, jobs.G_T_star_retained - jobs.G_T_retained)
assert jobs[["pair_id", "day", "base_model", "seed"]].duplicated().sum() == 0

group = ["pair_id", "provider", "base_course_token", "day", "base_model"]
measures = ["A_original", "G_T_original", "G_T_star_original", "A_retained", "G_T_retained",
            "G_T_star_retained", "G_T_star_intercept_retained", "target_extra_slope_gain_retained",
            "target_n_original", "target_n_retained", "retained_fraction"]
pair = jobs.groupby(group, as_index=False)[measures].mean()
pair["seeds"] = jobs.groupby(group).size().to_numpy()
pair["below_original_target_support_seed_fraction"] = jobs.groupby(group).below_original_target_support.mean().to_numpy()
pair.to_csv(SUP / "unique_user_course_pair_mean.csv", index=False)

summaries = []
bootstrap = []
for (day, model), g in pair.groupby(["day", "base_model"]):
    g = g.copy()
    g["cluster"] = g.provider + "|" + g.base_course_token
    unique_clusters = sorted(g.cluster.unique())
    metrics = ["A_original", "A_retained", "G_T_retained", "G_T_star_retained",
               "G_T_star_intercept_retained", "target_extra_slope_gain_retained"]
    values = g[metrics].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    cluster_sum = np.stack([values[g.cluster.eq(c)].sum(axis=0) for c in unique_clusters])
    cluster_n = np.asarray([g.cluster.eq(c).sum() for c in unique_clusters])
    rng = np.random.default_rng(PLAN["bootstrap_seed"] + day * 100 + (0 if model == "logistic" else 1))
    draw_idx = rng.integers(0, len(unique_clusters), size=(2000, len(unique_clusters)))
    draws = cluster_sum[draw_idx].sum(axis=1) / cluster_n[draw_idx].sum(axis=1)[:, None]
    point = values.mean(axis=0)
    diff = draws[:, 1] - draws[:, 0]
    summary = {"day": day, "base_model": model, "pairs": len(g), "base_course_clusters": len(unique_clusters),
               "pair_seed_jobs": int(jobs.day.eq(day).mul(jobs.base_model.eq(model)).sum()),
               "target_n_original_pair_mean": float(g.target_n_original.mean()),
               "target_n_retained_pair_mean": float(g.target_n_retained.mean()),
               "retained_fraction_pair_mean": float(g.retained_fraction.mean()),
               "below_original_target_support_pair_fraction": float(g.below_original_target_support_seed_fraction.gt(0).mean()),
               "A_original": point[0], "A_retained": point[1], "G_T_retained": point[2],
               "G_T_star_retained": point[3], "G_T_star_intercept_retained": point[4],
               "target_extra_slope_gain_retained": point[5],
               "A_retained_minus_original": point[1] - point[0],
               "A_retained_positive_pair_fraction": float(g.A_retained.gt(0).mean()),
               "A_retained_ci_low": float(np.quantile(draws[:, 1], .025)),
               "A_retained_ci_high": float(np.quantile(draws[:, 1], .975)),
               "A_difference_ci_low": float(np.quantile(diff, .025)),
               "A_difference_ci_high": float(np.quantile(diff, .975))}
    summaries.append(summary)
    for j, metric in enumerate(metrics):
        bootstrap.append({"day": day, "base_model": model, "measure": metric,
                          "pair_macro": point[j], "ci_low": np.quantile(draws[:, j], .025),
                          "ci_high": np.quantile(draws[:, j], .975), "pairs": len(g),
                          "base_course_clusters": len(unique_clusters)})
pd.DataFrame(summaries).sort_values(["day", "base_model"]).to_csv(SUP / "unique_user_course_summary.csv", index=False)
pd.DataFrame(bootstrap).to_csv(SUP / "unique_user_course_bootstrap.csv", index=False)

# Post hoc support check: require every retained seed of a pair/day/model to meet
# the same N and class-count thresholds as the original primary target risk set.
# Compare against the unfiltered result on exactly these selected pairs.
supported_rows = []
for (day, model), g in pair[pair.below_original_target_support_seed_fraction.eq(0)].groupby(["day", "base_model"]):
    g = g.copy()
    g["cluster"] = g.provider + "|" + g.base_course_token
    clusters = sorted(g.cluster.unique())
    values = g[["A_original", "A_retained"]].to_numpy(dtype=float)
    sums = np.stack([values[g.cluster.eq(c)].sum(axis=0) for c in clusters])
    counts = np.asarray([g.cluster.eq(c).sum() for c in clusters])
    rng = np.random.default_rng(PLAN["bootstrap_seed"] + 5000 + day * 100 + (0 if model == "logistic" else 1))
    draw_idx = rng.integers(0, len(clusters), size=(2000, len(clusters)))
    draws = sums[draw_idx].sum(axis=1) / counts[draw_idx].sum(axis=1)[:, None]
    supported_rows.append({"day": day, "base_model": model, "pairs": len(g),
                           "base_course_clusters": len(clusters),
                           "A_original_same_pairs": values[:, 0].mean(),
                           "A_retained": values[:, 1].mean(),
                           "A_retained_ci_low": np.quantile(draws[:, 1], .025),
                           "A_retained_ci_high": np.quantile(draws[:, 1], .975),
                           "retained_fraction_pair_mean": g.retained_fraction.mean()})
pd.DataFrame(supported_rows).sort_values(["day", "base_model"]).to_csv(
    SUP / "unique_user_course_posthoc_supported.csv", index=False)
verification = {"eligible_course_ids": len(course_map), "eligible_base_courses": course_map.base_course_key.nunique(),
                "eligible_enrollments": len(enroll), "eligible_unique_users": len(assignment),
                "users_in_multiple_eligible_base_courses": int(base_counts.gt(1).sum()),
                "primary_jobs_reanalyzed": len(jobs), "zero_retained_target_jobs": int(jobs.target_n_retained.eq(0).sum()),
                "jobs_below_original_target_support_after_filtering": int(jobs.below_original_target_support.sum()),
                "max_original_A_recomputation_error": max_recompute_error,
                "evaluation_only_not_refitted": True, "status": "passed"}
(SUP / "unique_user_course_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
print(json.dumps(verification, indent=2), flush=True)
