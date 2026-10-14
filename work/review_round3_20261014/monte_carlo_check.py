"""Post-analysis Monte Carlo precision check for all four nonzero family96 signals.
Not an additional fitted experiment and not used to choose a preferred policy.
Original 100000-draw results remain the prespecified analysis.
"""
import argparse,os
from pathlib import Path
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();rows=[]
keys=['day','base_model','fraction','strategy']
for mode in ['recent','joint']:
    ci=pd.read_csv(a.output/mode/'conservative/update_intervals.csv')
    pm=pd.read_csv(a.output/mode/'conservative/update_triple_means.csv')
    for row in ci[(ci.family96_low>0)|(ci.family96_high<0)].to_dict('records'):
        g=pm.copy()
        for key in keys:g=g[g[key].eq(row[key])]
        cl=g.groupby(['provider','base_course_token']).advantage.agg(['sum','size']);n=len(cl)
        rng=np.random.default_rng(20261014);boot=[]
        for _ in range(100):
            w=rng.multinomial(n,np.repeat(1/n,n),size=10000)
            boot.append((w@cl['sum'].to_numpy())/(w@cl['size'].to_numpy()))
        b=np.concatenate(boot);lo,hi=np.quantile(b,[.05/192,1-.05/192])
        rows.append({'mode':mode,**{k:row[k] for k in keys},'mean_U':row['mean'],
         'original_100k_family96_low':row['family96_low'],'original_100k_family96_high':row['family96_high'],
         'one_million_family96_low':lo,'one_million_family96_high':hi,'bootstrap_draws':1000000,'bootstrap_seed':20261014,
         'same_nonzero_direction':bool((lo>0 and row['family96_low']>0)or(hi<0 and row['family96_high']<0))})
pd.DataFrame(rows).to_csv(a.output/'summary/monte_carlo_signal_check.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
