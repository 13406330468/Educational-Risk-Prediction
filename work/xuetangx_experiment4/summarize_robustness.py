"""Equal-pair summaries across all completed sensitivity families."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from primary import OUT


second = pd.concat([pd.read_csv(OUT/f"secondary_day{d}.csv") for d in (7,14,21,28)],ignore_index=True)
matched = pd.concat([pd.read_csv(OUT/f"matched_day{d}.csv") for d in (7,14,21,28)],ignore_index=True)
primary = pd.read_csv(OUT/"primary_pair_mean.csv")
history = pd.read_csv(OUT/"historical_pooling.csv")
random = pd.read_csv(OUT/"random_course_negative_control.csv")
earliest = pd.read_csv(OUT/"earliest_latest_pair_mean.csv")
zero = pd.read_csv(OUT/"zero_fill_pair_mean.csv")
expected = len(pd.read_csv(OUT/"primary_by_seed.csv"))
assert len(second)==expected and len(history)==expected and len(matched)==3*expected

second.to_csv(OUT/"secondary_calibration_by_seed.csv",index=False)
matched.to_csv(OUT/"matched_target_by_repeat.csv",index=False)

keys = ["pair_id","day","base_model"]
second_pair = second.groupby(keys,as_index=False).mean(numeric_only=True)
matched_pair = matched.groupby(keys,as_index=False).mean(numeric_only=True)
history_pair = history.groupby(keys,as_index=False).mean(numeric_only=True)
random_seed = random.groupby(keys+["seed"],as_index=False)["G_T_random_other_source"].mean()
random_pair = random_seed.groupby(keys,as_index=False)["G_T_random_other_source"].mean()
second_pair.to_csv(OUT/"secondary_calibration_pair_mean.csv",index=False)
matched_pair.to_csv(OUT/"matched_target_pair_mean.csv",index=False)
history_pair.to_csv(OUT/"historical_pooling_pair_mean.csv",index=False)

families = []
def add(name,frame,gt_col=None,gstar_col=None,a_col=None,subset=None):
    frame = frame.copy()
    if subset is not None:
        frame = frame[subset(frame)]
    for (day,base),g in frame.groupby(["day","base_model"]):
        families.append({"family":name,"day":day,"base_model":base,"pairs":g.pair_id.nunique(),
                         "G_T_macro":float(g[gt_col].mean()) if gt_col else np.nan,
                         "G_T_star_macro":float(g[gstar_col].mean()) if gstar_col else np.nan,
                         "A_macro":float(g[a_col].mean()) if a_col else np.nan,
                         "A_positive_pair_fraction":float(g[a_col].gt(0).mean()) if a_col else np.nan})

add("primary_Platt_unseen_active",primary,"G_T","G_T_star","A")
for method in ("beta","isotonic"):
    add(method+"_unseen_active",second_pair,f"G_T_{method}",f"G_T_star_{method}",f"A_{method}")
for risk in ("all_active","returning_active","all_zero_fill"):
    add("evaluation_"+risk,second_pair,f"G_T_{risk}",f"G_T_star_{risk}",f"A_{risk}",
        subset=lambda d,risk=risk: d[f"A_{risk}"].notna())
add("matched_target_label_budget",matched_pair,"G_T","G_T_star_Platt_matched","A_matched")
add("earliest_to_latest",earliest,"G_T","G_T_star","A")
add("retrained_all_enrollment_zero_fill",zero,"G_T","G_T_star","A")
for name,col in (("same_course_pooled_historical_mapping","G_T_same_course_pool"),
                 ("global_historical_mapping","G_T_global_pool"),
                 ("other_course_pooled_mapping","G_T_other_course_pool")):
    add(name,history_pair,col)
add("same_course_pool_with_2plus_history",history_pair,"G_T_same_course_pool",subset=lambda d:d.same_course_sources.ge(2))
add("previous_only_with_2plus_history",history_pair,"previous_G_T",subset=lambda d:d.same_course_sources.ge(2))
add("random_other_course_source",random_pair,"G_T_random_other_source")

summary = pd.DataFrame(families)
summary.to_csv(OUT/"robustness_family_summary.csv",index=False)
print(json.dumps({"secondary_jobs":len(second),"matched_repeats":len(matched),
                  "robustness_summary_rows":len(summary),"zero_fill_jobs":len(pd.read_csv(OUT/"zero_fill_by_seed.csv")),
                  "earliest_latest_jobs":len(pd.read_csv(OUT/"earliest_latest_by_seed.csv"))},indent=2))
