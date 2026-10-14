"""Complete five-seed default-parameter DFMC sensitivity summary."""
import json
import pandas as pd
from primary import OUT

files = sorted(OUT.glob("dfmc_seed*_day*.csv"))
assert len(files) == 20, len(files)
frame = pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
expected = len(pd.read_csv(OUT/"primary_by_seed.csv"))
assert len(frame) == expected, (len(frame),expected)
assert not frame[["pair_id","day","base_model","seed"]].duplicated().any()
frame.to_csv(OUT/"dfmc_by_seed.csv",index=False)
pair = frame.groupby(["pair_id","day","base_model"],as_index=False).mean(numeric_only=True)
pair.to_csv(OUT/"dfmc_pair_mean.csv",index=False)
rows = []
for (day,base),g in pair.groupby(["day","base_model"]):
    rows.append({"day":day,"base_model":base,"pairs":len(g),
                 "G_S_DFMC_macro":float(g.G_S_DFMC.mean()),
                 "G_T_DFMC_macro":float(g.G_T_DFMC.mean()),
                 "G_T_star_DFMC_macro":float(g.G_T_star_DFMC.mean()),
                 "A_DFMC_macro":float(g.A_DFMC.mean()),
                 "A_DFMC_positive_pair_fraction":float(g.A_DFMC.gt(0).mean()),
                 "source_deployed_out_of_range_fraction_mean":float(g.source_deployed_out_of_range_fraction.mean()),
                 "target_oof_out_of_range_fraction_mean":float(g.target_oof_out_of_range_fraction.mean())})
summary = pd.DataFrame(rows).sort_values(["day","base_model"])
summary.to_csv(OUT/"dfmc_time_model_summary.csv",index=False)
rob = pd.read_csv(OUT/"robustness_family_summary.csv")
add = pd.DataFrame([{"family":"DFMC_default_fixed_groups","day":r.day,"base_model":r.base_model,
                     "pairs":r.pairs,"G_T_macro":r.G_T_DFMC_macro,
                     "G_T_star_macro":r.G_T_star_DFMC_macro,"A_macro":r.A_DFMC_macro,
                     "A_positive_pair_fraction":r.A_DFMC_positive_pair_fraction}
                    for r in summary.itertuples()])
pd.concat([rob,add],ignore_index=True).to_csv(OUT/"robustness_family_summary.csv",index=False)
print(json.dumps({"dfmc_jobs":len(frame),"pair_units":len(pair),"summary_rows":len(summary)},indent=2))
