"""Compare regenerated OULAD aggregate results to the frozen reference tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--regenerated", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
key = ["season", "course", "day", "base", "seed"]
results = {}
for name, keys in [("decomposition_by_seed.csv", key), ("matched_size_intercept.csv", key+["repeat"]),
                   ("adaptation_by_cell.csv", key[:-1]), ("matched_size_by_cell.csv", key[:-1])]:
    a, b = pd.read_csv(args.reference/name), pd.read_csv(args.regenerated/name)
    assert len(a) == len(b), f"Incomplete table: {name}"
    merged = a.merge(b, on=keys, validate="one_to_one", suffixes=("_reference", "_regenerated"))
    assert len(merged) == len(a)
    diffs = {}
    for col in b.select_dtypes("number"):
        if col in keys:
            continue
        x, y = merged[col+"_reference"], merged[col+"_regenerated"]
        assert x.isna().equals(y.isna()), (name, col, "NaN mismatch")
        differences = np.abs(x-y)
        diffs[col] = float(differences.max()) if differences.notna().any() else 0.
    assert max(diffs.values()) <= 1e-10, (name, diffs)
    for col in ["mechanism_label", "ratio_eligible"]:
        if col in b:
            assert merged[col+"_reference"].equals(merged[col+"_regenerated"])
    results[name] = {"rows": len(a), "numeric_columns_checked": len(diffs),
                     "maximum_absolute_difference": max(diffs.values()), "differences_by_column": diffs}
report = {"status": "PASS", "tolerance": 1e-10, "full_model_jobs": 80,
          "tables": results,
          "scope": "Regenerated from all four prepared OULAD landmark feature tables; preparation is separately validated."}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
