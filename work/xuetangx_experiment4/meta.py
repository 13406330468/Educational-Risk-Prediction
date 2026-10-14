"""Exploratory clustered explanation and leave-one-course-out transfer signals."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from primary import OUT


pair = pd.read_csv(OUT / "primary_pair_mean.csv")
shift = pd.read_csv(OUT / "feature_shift_summary.csv")
data = pair.merge(shift, on=["pair_id","day"],validate="many_to_one")
data["course_group"] = data.provider.astype(str) + "+" + data.base_course_token.astype(str)
data["lightgbm"] = data.base_model.eq("lightgbm").astype(int)
assert data.course_group.nunique() >= 20 and len(data) >= 100

source_features = ["G_S", "source_alpha", "source_joint_beta", "source_raw_risk_mean",
                   "source_raw_risk_variance", "source_cal_n", "risk_wasserstein",
                   "mean_abs_SMD", "max_abs_SMD", "action_JS_divergence",
                   "observation_gap_days", "day", "lightgbm"]
X = data[source_features].to_numpy(dtype=float)
y = data.A.gt(.001).astype(int).to_numpy()
groups = data.course_group.to_numpy()
logo = LeaveOneGroupOut()
preds = {"logistic": np.full(len(y), np.nan), "shallow_tree": np.full(len(y), np.nan)}
for train, test in logo.split(X, y, groups):
    models = {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=1.,max_iter=1000,class_weight="balanced")),
        "shallow_tree": make_pipeline(SimpleImputer(strategy="median"),
                                      DecisionTreeClassifier(max_depth=2,min_samples_leaf=10,
                                                             class_weight="balanced",random_state=17)),
    }
    for name, model in models.items():
        model.fit(X[train],y[train])
        preds[name][test] = model.predict_proba(X[test])[:,1]
assert all(np.isfinite(x).all() for x in preds.values())
meta_pred = data[["pair_id","day","base_model","course_group","A"]].copy()
meta_pred["A_gt_0_001"] = y
for name,p in preds.items():
    meta_pred[name+"_prob"] = p
meta_pred.to_csv(OUT / "loco_transfer_predictions.csv",index=False)
scores = []
for name,p in preds.items():
    scores.append({"model":name,"n_units":len(y),"base_courses":len(np.unique(groups)),
                   "positive_fraction":float(y.mean()),"AUROC":float(roc_auc_score(y,p)),
                   "AP":float(average_precision_score(y,p)),"Brier":float(brier_score_loss(y,p))})
pd.DataFrame(scores).to_csv(OUT / "loco_transfer_summary.csv",index=False)

def design(df,retrospective):
    names = ["intercept","day_per_7","lightgbm","risk_W1_per_0_1","gap_per_100_days","source_risk_mean"]
    arrays = [np.ones(len(df)),df.day.to_numpy()/7,df.lightgbm.to_numpy(),
              df.risk_wasserstein.to_numpy()/.1,df.observation_gap_days.to_numpy()/100,
              df.source_raw_risk_mean.to_numpy()]
    if retrospective:
        names.append("prevalence_shift")
        arrays.append(df.prevalence_shift.to_numpy())
    return names,np.column_stack(arrays)

coef_rows = []
for retrospective in (False,True):
    names,Z = design(data,retrospective)
    beta = np.linalg.lstsq(Z,data.A.to_numpy(),rcond=None)[0]
    fitted = Z@beta
    clusters = [g for _,g in data.groupby("course_group")]
    rng = np.random.default_rng(20261001 + int(retrospective))
    draws = np.empty((2000,len(names)))
    for b in range(2000):
        sample = pd.concat([clusters[i] for i in rng.integers(0,len(clusters),size=len(clusters))],ignore_index=True)
        _,zz = design(sample,retrospective)
        draws[b] = np.linalg.lstsq(zz,sample.A.to_numpy(),rcond=None)[0]
    for j,name in enumerate(names):
        coef_rows.append({"model":"retrospective_with_target_prevalence" if retrospective else "source_observable",
                          "term":name,"coefficient":float(beta[j]),
                          "course_cluster_ci_2_5":float(np.quantile(draws[:,j],.025)),
                          "course_cluster_ci_97_5":float(np.quantile(draws[:,j],.975)),
                          "in_sample_R2_descriptive":float(r2_score(data.A,fitted)),
                          "n_units":len(data),"base_courses":len(clusters)})
pd.DataFrame(coef_rows).to_csv(OUT / "clustered_descriptive_meta_model.csv",index=False)
print(json.dumps({"loco":scores,"units":len(data),"courses":data.course_group.nunique()},indent=2))
