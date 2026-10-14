"""Compare independently rebuilt raw-derived features with prior preparation."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--fresh',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
rows=[]
def compare_frames(old,new,keys,label):
    assert set(old.columns)==set(new.columns),(label,'columns')
    old=old.sort_values(keys).reset_index(drop=True);new=new.sort_values(keys).reset_index(drop=True)
    assert len(old)==len(new),(label,'row count')
    assert old[keys].equals(new[keys]),(label,'keys')
    max_error=0.; differing=[]
    for col in old:
        if pd.api.types.is_numeric_dtype(old[col]):
            ov,nv=old[col].to_numpy(dtype=float),new[col].to_numpy(dtype=float)
            assert np.array_equal(np.isnan(ov),np.isnan(nv)),(label,col,'missingness')
            err=float(np.nanmax(np.abs(ov-nv))) if len(ov) else 0.
            max_error=max(err,max_error)
            if err:differing.append(col)
            assert np.allclose(ov,nv,rtol=1e-10,atol=1e-10,equal_nan=True),(label,col,err)
        else:assert old[col].fillna('').equals(new[col].fillna('')),(label,col)
    rows.append({'table':label,'rows':len(old),'numeric_max_abs_error':max_error,'floating_difference_columns':differing,'passed':True})
for day in (7,14,21,28):
    rel=f'work/xuetangx_experiment4/features_day{day}.parquet'
    compare_frames(pd.read_parquet(a.previous/rel),pd.read_parquet(a.fresh/rel),['course_id','enroll_id'],f'features_day{day}')
for rel,keys,label in [('work/xuetangx_audit/enrollment_index.csv',['course_id','enroll_id'],'enrollment_index'),('outputs/xuetangx_data_audit/course_inventory.csv',['course_id'],'course_inventory')]:
    compare_frames(pd.read_csv(a.previous/rel),pd.read_csv(a.fresh/rel),keys,label)
compare_frames(pd.read_parquet(a.previous/'work/xuetangx_experiment4/review_extension/future_all_presentations.parquet'),
               pd.read_parquet(a.fresh/'work/review_round3/baseline/future_all_presentations.parquet'),['course_id','enroll_id'],'future_endpoint')
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'raw_preparation_validation.json').write_text(json.dumps({'tables':rows,'scope':'Raw ZIP preparation independently executed; same algorithms, no previous feature/cache reuse'},indent=2))
print(json.dumps(rows,indent=2))
