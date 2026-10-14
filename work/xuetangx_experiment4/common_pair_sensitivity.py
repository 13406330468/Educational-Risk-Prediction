"""Outcome-blind, fully paired landmark sensitivity from frozen primary results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
OUT = ROOT / "outputs" / "xuetangx_experiment4"
SUP = OUT / "supplement"
SUP.mkdir(exist_ok=True)
PLAN = json.loads((ROOT / "work" / "xuetangx_experiment4" / "supplement_protocol.json").read_text(encoding="utf-8"))
DAYS = PLAN["primary_landmarks"]
MEASURES = PLAN["common_pair_analysis"]["measures"]
N_BOOT = 2000

primary = pd.read_csv(OUT / "primary_by_seed.csv")
primary = primary.loc[primary.day.isin(DAYS)].copy()
models = PLAN["base_models"]
key_sets = [set(zip(g.pair_id, g.seed)) for _, g in primary.groupby(["day", "base_model"])]
common = set.intersection(*key_sets)
assert len(key_sets) == len(DAYS) * len(models)
assert common and all(common.issubset(s) for s in key_sets)
index = pd.MultiIndex.from_tuples(sorted(common), names=["pair_id", "seed"])
matched = primary.set_index(["pair_id", "seed"]).loc[index].reset_index()
matched = matched.sort_values(["pair_id", "seed", "day", "base_model"])
assert len(matched) == len(common) * len(DAYS) * len(models)
assert matched.groupby(["pair_id", "seed", "base_model"]).day.nunique().eq(len(DAYS)).all()
matched.to_csv(SUP / "common_pair_by_seed.csv", index=False)

group_cols = ["pair_id", "provider", "base_course_token", "day", "base_model"]
pair = matched.groupby(group_cols, as_index=False)[MEASURES].mean()
pair["matched_seeds"] = matched.groupby(group_cols).size().to_numpy()
pair.to_csv(SUP / "common_pair_pair_mean.csv", index=False)

rows = []
boot_rows = []
contrast_rows = []
for model in models:
    g = pair.loc[pair.base_model.eq(model)].copy()
    pairs = g.pair_id.nunique()
    assert g.groupby("pair_id").day.nunique().eq(len(DAYS)).all()
    g["cluster"] = g.provider + "|" + g.base_course_token
    wide = g.pivot(index=["pair_id", "cluster"], columns="day", values=MEASURES)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([MEASURES, DAYS]))
    values = wide.to_numpy(dtype=float)
    assert np.isfinite(values).all() and len(values) == pairs
    cluster = np.asarray(wide.index.get_level_values("cluster"))
    unique_clusters = np.unique(cluster)
    cluster_sums = np.stack([values[cluster == c].sum(axis=0) for c in unique_clusters])
    cluster_sizes = np.asarray([(cluster == c).sum() for c in unique_clusters])
    rng = np.random.default_rng(PLAN["bootstrap_seed"] + (0 if model == "logistic" else 1))
    draw_idx = rng.integers(0, len(unique_clusters), size=(N_BOOT, len(unique_clusters)))
    boot = cluster_sums[draw_idx].sum(axis=1) / cluster_sizes[draw_idx].sum(axis=1)[:, None]
    point = values.mean(axis=0)
    def position(measure, day):
        return MEASURES.index(measure) * len(DAYS) + DAYS.index(day)
    for measure in MEASURES:
        for day in DAYS:
            j = position(measure, day)
            boot_rows.append({"base_model": model, "day": day, "measure": measure,
                              "pair_macro": point[j], "ci_low": np.quantile(boot[:, j], .025),
                              "ci_high": np.quantile(boot[:, j], .975),
                              "pairs": pairs, "base_course_clusters": len(unique_clusters),
                              "matched_pair_seeds": len(common)})
    for day in DAYS:
        h = g[g.day.eq(day)]
        rows.append({"day": day, "base_model": model, "pairs": pairs,
                     "base_course_clusters": len(unique_clusters), "matched_pair_seeds": len(common),
                     **{m: float(h[m].mean()) for m in MEASURES},
                     "A_pair_positive_fraction": float(h.A.gt(0).mean()),
                     "slope_gain_pair_positive_fraction": float(h.target_extra_slope_gain.gt(0).mean())})
    for earlier, later in ((7, 14), (14, 28), (7, 28)):
        for measure in MEASURES:
            a, b = position(measure, earlier), position(measure, later)
            pair_diff = values[:, a] - values[:, b]
            draws = boot[:, a] - boot[:, b]
            contrast_rows.append({"base_model": model, "measure": measure,
                                  "contrast": f"day{earlier}_minus_day{later}",
                                  "pair_macro_difference": point[a] - point[b],
                                  "ci_low": np.quantile(draws, .025),
                                  "ci_high": np.quantile(draws, .975),
                                  "positive_pair_fraction": float(np.mean(pair_diff > 0)),
                                  "pairs": pairs, "base_course_clusters": len(unique_clusters)})
    a = values[:, position("A", 7)]
    b = values[:, position("A", 14)]
    c = values[:, position("A", 28)]
    print(model, "strictly decreasing A fraction", float(np.mean((a > b) & (b > c))), flush=True)

pd.DataFrame(rows).sort_values(["day", "base_model"]).to_csv(SUP / "common_pair_summary.csv", index=False)
pd.DataFrame(boot_rows).sort_values(["day", "base_model", "measure"]).to_csv(SUP / "common_pair_bootstrap.csv", index=False)
pd.DataFrame(contrast_rows).to_csv(SUP / "common_pair_contrasts.csv", index=False)
pd.DataFrame({"pair_id": sorted({p for p, _ in common})}).to_csv(SUP / "common_pair_ids.csv", index=False)
result = {"pairs": len({p for p, _ in common}), "matched_pair_seeds": len(common),
          "matched_jobs": len(matched), "models": models, "days": DAYS,
          "base_course_clusters": pair[["provider", "base_course_token"]].drop_duplicates().shape[0]}
(SUP / "common_pair_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
