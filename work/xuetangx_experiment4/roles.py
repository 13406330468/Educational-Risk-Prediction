"""Prespecify username-grouped source roles and pre-model unit support."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
OUT = ROOT / "outputs" / "xuetangx_experiment4"
WORK = ROOT / "work" / "xuetangx_experiment4"
PAIRS = pd.read_csv(OUT / "eligible_pairs.csv")
ENROLL = pd.read_csv(ROOT / "work" / "xuetangx_audit" / "enrollment_index.csv",
                     dtype={"enroll_id": str, "username": str, "course_id": str})
BY_COURSE = {k: g for k, g in ENROLL.groupby("course_id")}
SEEDS = (17, 29, 43, 59, 71)
rows = []
support = []

for pair in PAIRS.itertuples():
    source = BY_COURSE[pair.source_course_id].reset_index()
    target = BY_COURSE[pair.target_course_id].reset_index()
    assert source.username.is_unique and target.username.is_unique
    source_users = set(source.username)
    target_unseen = target.loc[~target.username.isin(source_users)]
    for seed in SEEDS:
        split_seed = seed + int(pair.pair_id[1:]) * 1009
        train_idx, held_idx = train_test_split(
            np.arange(len(source)), test_size=0.4, random_state=split_seed,
            stratify=source.truth.to_numpy(),
        )
        valid_idx, cal_idx = train_test_split(
            held_idx, test_size=0.5, random_state=split_seed + 104729,
            stratify=source.iloc[held_idx].truth.to_numpy(),
        )
        role = np.empty(len(source), dtype=object)
        role[train_idx] = "train"
        role[valid_idx] = "validation"
        role[cal_idx] = "calibration"
        for e, u, r in zip(source.enroll_id, source.username, role):
            rows.append((pair.pair_id, seed, e, u, r))
        for day in (7, 14, 21, 28):
            source_active = source[pd.to_datetime(source.first_event).le(pd.Timestamp(pair.source_first_event) + pd.Timedelta(days=day))]
            source_active_roles = role[source_active.index.to_numpy()]
            t = target_unseen[pd.to_datetime(target_unseen.first_event).le(pd.Timestamp(pair.target_first_event) + pd.Timedelta(days=day))]
            rs = {}
            for name in ("train", "validation", "calibration"):
                g = source_active.loc[source_active_roles == name]
                rs[f"source_{name}_n"] = len(g)
                rs[f"source_{name}_minority"] = int(g.truth.value_counts().min()) if g.truth.nunique() == 2 else 0
            target_minority = int(t.truth.value_counts().min()) if t.truth.nunique() == 2 else 0
            admissible = (
                rs["source_train_minority"] >= 5
                and rs["source_validation_minority"] >= 2
                and rs["source_calibration_n"] >= 30
                and rs["source_calibration_minority"] >= 10
                and len(t) >= 50 and target_minority >= 10
            )
            support.append({"pair_id": pair.pair_id, "seed": seed, "day": day,
                            **rs, "target_unseen_n": len(t),
                            "target_unseen_minority": target_minority,
                            "admissible": admissible})

role_df = pd.DataFrame(rows, columns=["pair_id", "seed", "enroll_id", "username", "role"])
role_df.to_parquet(WORK / "source_roles.parquet", index=False)
supp = pd.DataFrame(support)
supp.to_csv(OUT / "unit_support.csv", index=False)
digest = hashlib.sha256((WORK / "source_roles.parquet").read_bytes()).hexdigest()
protocol_path = OUT / "experiment4_protocol.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol["role_assignment_sha256"] = digest
protocol["unit_admissibility"] = "pre-model: source active train minority>=5, validation minority>=2, calibration N>=30 and minority>=10; target unseen active N>=50 and minority>=10; no pair-list changes"
protocol["unit_support_summary"] = {
    str(day): {"admissible_pair_seeds": int(g.admissible.sum()), "total_pair_seeds": len(g)}
    for day, g in supp.groupby("day")
}
protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(protocol["unit_support_summary"], indent=2), flush=True)
