"""Fixed-hyperparameter DFMC sensitivity on seed 17 only (all eligible pairs)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from primary import DAY_FRAMES, OUT, PRED, brier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oulad_experiment"))
from dfmc import DFMC


SEED = 17
ACTIONS_VIDEO = ["play_video", "load_video", "pause_video", "stop_video", "seek_video"]
ACTIONS_PROBLEM = ["problem_get", "problem_check", "problem_check_correct", "problem_check_incorrect", "problem_save", "reset_problem"]
ACTIONS_FORUM = ["click_forum", "close_forum", "create_thread", "create_comment"]


def groups(frame, features, threshold):
    merge = frame[["enroll_id"]].merge(features, on="enroll_id", how="left", validate="one_to_one")
    def count(actions):
        return merge[[f"action_{a}_count" for a in actions]].fillna(0).sum(axis=1).to_numpy()
    return pd.DataFrame({
        "all": np.ones(len(frame), dtype=int),
        "high_activity": merge.event_count.fillna(0).to_numpy() >= threshold,
        "video_any": count(ACTIONS_VIDEO) > 0,
        "problem_any": count(ACTIONS_PROBLEM) > 0,
        "forum_any": count(ACTIONS_FORUM) > 0,
    }).astype(int)


def fit_predict(p_fit, y_fit, g_fit, p_test, g_test, seed):
    idx = np.arange(len(y_fit))
    inner, valid = train_test_split(idx, test_size=.3, random_state=seed, stratify=y_fit)
    model = DFMC(n_jobs=1, seed=seed, num_boost_round=2000, early_stopping_rounds=50)
    model.fit(p_fit[inner], y_fit[inner], g_fit.iloc[inner],
              p_fit[valid], y_fit[valid], g_fit.iloc[valid])
    return model.predict(p_test, g_test), model.best_iteration_


def crossfit(p, y, g, folds, seed):
    result = np.full(len(y), np.nan)
    iterations = []
    for fold in range(5):
        fit = np.where(folds != fold)[0]
        test = np.where(folds == fold)[0]
        result[test], it = fit_predict(p[fit], y[fit], g.iloc[fit], p[test], g.iloc[test], seed+fold)
        iterations.append(it)
    return result, iterations


def one(row):
    f = pd.read_parquet(PRED / f"{row.pair_id}_d{row.day}_{row.base_model}_s{SEED}.parquet")
    sc = f[f.population.eq("source_calibration")].reset_index(drop=True)
    tgt = f[f.population.eq("target_primary")].reset_index(drop=True)
    features = DAY_FRAMES[row.day]
    source_feature = sc[["enroll_id"]].merge(features[["enroll_id","event_count"]], on="enroll_id")
    threshold = float(source_feature.event_count.median())
    gs = groups(sc, features, threshold)
    gt = groups(tgt, features, threshold)
    ps,ys = sc.raw.to_numpy(),sc.y.to_numpy(dtype=int)
    pt,yt = tgt.raw.to_numpy(),tgt.y.to_numpy(dtype=int)
    source_pred, source_iter = fit_predict(ps,ys,gs,pt,gt,100000+row.day)
    source_folds = np.full(len(ys),-1,dtype=int)
    for fold,(_,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=1000+SEED+row.day).split(ps,ys)):
        source_folds[test] = fold
    source_oof, source_oof_iter = crossfit(ps,ys,gs,source_folds,200000+row.day)
    target_oof, target_iter = crossfit(pt,yt,gt,tgt.fold.to_numpy(dtype=int),300000+row.day)
    gs_gain = brier(ys,ps)-brier(ys,source_oof)
    gt_gain = brier(yt,pt)-brier(yt,source_pred)
    gstar_gain = brier(yt,pt)-brier(yt,target_oof)
    return {"pair_id":row.pair_id,"day":row.day,"base_model":row.base_model,"seed":SEED,
            "source_cal_n":len(sc),"target_n":len(tgt),"G_S_DFMC":gs_gain,
            "G_T_DFMC":gt_gain,"G_T_star_DFMC":gstar_gain,"A_DFMC":gstar_gain-gt_gain,
            "source_deployed_out_of_range_fraction":float(np.mean((source_pred<0)|(source_pred>1))),
            "target_oof_out_of_range_fraction":float(np.mean((target_oof<0)|(target_oof>1))),
            "source_full_best_iteration":source_iter,
            "source_oof_best_iteration_mean":float(np.mean(source_oof_iter)),
            "target_oof_best_iteration_mean":float(np.mean(target_iter)),
            "group_high_activity_threshold_source_median":threshold}


def main(day,limit=None):
    primary = pd.read_csv(OUT/"primary_by_seed.csv")
    primary = primary[primary.day.eq(day)&primary.seed.eq(SEED)]
    if limit is not None:
        primary = primary.head(limit)
    out = OUT/f"dfmc_seed{SEED}_day{day}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    completed = set(zip(done.pair_id,done.base_model)) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    start = time.monotonic()
    for i,row in enumerate(primary.itertuples(),1):
        if (row.pair_id,row.base_model) in completed:
            continue
        rows.append(one(row))
        if i%10==0:
            pd.DataFrame(rows).to_csv(out,index=False)
            print(f"DFMC day {day}: {i}/{len(primary)} jobs; {time.monotonic()-start:.1f}s",flush=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(json.dumps({"day":day,"jobs":len(rows),"elapsed_seconds":round(time.monotonic()-start,1)}),flush=True)


if __name__=="__main__":
    if len(sys.argv)>2:
        SEED = int(sys.argv[2])
    main(int(sys.argv[1]),int(sys.argv[3]) if len(sys.argv)>3 else None)
