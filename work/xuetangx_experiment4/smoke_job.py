"""One real, eligible source/target pair; no frozen predictions are reused."""
import json
import numpy as np
import pandas as pd
import primary as p

day=7;seed=17;base='logistic'
ok=p.SUPPORT[p.SUPPORT.day.eq(day)&p.SUPPORT.seed.eq(seed)&p.SUPPORT.admissible]
if ok.empty: raise ValueError('No eligible Day 7 seed 17 real-data smoke pair')
pair_id=ok.pair_id.iloc[0];pair=next(p.PAIRS[p.PAIRS.pair_id.eq(pair_id)].itertuples())
source,target=p.prep_pair(pair,day)
roles=p.ROLES[p.ROLES.pair_id.eq(pair_id)&p.ROLES.seed.eq(seed)]
role_map=dict(zip(roles.enroll_id,roles.role))
# Force a fresh fit into its own smoke directory, independently of other cached jobs.
p.PRED=p.WORK/'smoke_predictions';p.PRED.mkdir(exist_ok=True)
if list(p.PRED.glob('*.parquet')): raise RuntimeError('Use a fresh smoke run directory')
result=p.one_job(pair,day,seed,base,source,target,role_map)
assert abs(result['A']-(result['G_T_star']-result['G_T']))<1e-12
pd.DataFrame([result]).to_csv(p.OUT/'smoke_summary.csv',index=False)
print(json.dumps({'status':'passed','dataset':'XuetangX','pair':pair_id,'day':day,'seed':seed,'base_model':base,'target_n':result['target_n'],'G_S':result['G_S'],'G_T':result['G_T'],'G_T_star':result['G_T_star'],'A':result['A'],'fresh_fit':True},indent=2))
