"""Integrated R3 entry point: aggregate regeneration and distinct raw-data refits."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('action',choices=['quick','raw']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True);a=ap.parse_args()
root=a.root.resolve();repo=a.repo.resolve();out=root/'outputs/review_round3_20261014';scripts=root/'work/review_round3_20261014'
os.environ['CALTRANSFER_ROOT']=str(root)
def run(script,*args):subprocess.run([sys.executable,str(scripts/script),*map(str,args)],cwd=root,check=True)
package=json.loads((repo/'frozen/aggregate_results.json').read_text(encoding='utf-8'))
reference=root/'r3_reference_outputs'
for key,item in package['tables'].items():
    if key.startswith(('review_quality_20261011/','review_round2_20261011/','review_round3_20261014/')):
        assert hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256'],key
        p=reference/key;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text'],encoding='utf-8')
if a.action=='raw':
    cfg=json.loads((root/'run_config.json').read_text());assert cfg['mode']=='full','Fresh full preparation required'
    assert (root/'input_audit.json').exists()
    for mode in ('baseline','recent','joint'):run('run_study.py','--runtime',root,'--output',out,'--mode',mode)
    run('audit_boundaries.py','--runtime',root,'--output',out)
else:
    sys.path.insert(0,str(root/'work/xuetangx_experiment4'))
    import conservative_updates as c
    for mode in ('baseline','recent','joint'):
        c.OUT=out/mode/'conservative';c.PREV=out/mode;c.summarize()
run('analyze_results.py','--output',out,'--previous',reference)
checks=[]
for mode in ('baseline','recent','joint'):
    old=pd.read_csv(reference/'review_round3_20261014'/mode/'conservative/update_intervals.csv');new=pd.read_csv(out/mode/'conservative/update_intervals.csv')
    keys=['day','base_model','fraction','strategy'];old=old.sort_values(keys).reset_index(drop=True);new=new.sort_values(keys).reset_index(drop=True)
    assert old[keys].equals(new[keys]);cols=['mean','point_low','point_high','family96_low','family96_high','family6_low','family6_high']
    assert np.array_equal(old[cols].isna(),new[cols].isna())
    delta=float(np.nanmax(abs(old[cols].to_numpy()-new[cols].to_numpy())))
    assert delta < (1e-10 if a.action=='quick' else 1e-5),(mode,delta)
    checks.append({'population':mode,'comparisons':96,'maximum_interval_difference':delta})
for name,keys,cols in [('pair_intervals',['mode','day','base_model','metric'],['mean','point_low','point_high','family_low','family_high']),('full_budget_absolute_table',['mode','day','base_model'],['raw_Brier','historical_Brier','full_Brier','intercept_Brier','shrink_Brier','full_U','full_U_family6_low','full_U_family6_high'])]:
    old=pd.read_csv(reference/'review_round3_20261014/summary'/f'{name}.csv').sort_values(keys).reset_index(drop=True)
    new=pd.read_csv(out/'summary'/f'{name}.csv').sort_values(keys).reset_index(drop=True)
    assert old[keys].equals(new[keys]);delta=float(np.nanmax(abs(old[cols].to_numpy()-new[cols].to_numpy())))
    assert delta < (1e-10 if a.action=='quick' else 1e-5),(name,delta)
    checks.append({'table':name,'rows':len(old),'maximum_numeric_difference':delta})
# The 1M tail check is independently available and not repeated in the fast aggregate path.
if a.action=='raw':run('monte_carlo_check.py','--output',out)
run('make_figures.py','--output',out)
paper=root/'outputs/manuscript_r3';(paper/'figures').mkdir(parents=True,exist_ok=True)
run('build_evidence.py','--evidence',out,'--paper',paper,'--registry',paper/'evidence_registry.json')
summary={'status':'passed','mode':a.action,'scope':'Frozen aggregate reaggregation, 100000-draw uncertainty and numeric table/figure regeneration; no model fitting' if a.action=='quick' else 'Fresh raw-prepared two/three-presentation pipeline execution; same algorithms, not independent implementation or team','checks':checks,'bootstrap_draws':100000,'populations':3,'update_comparisons':288}
(root/'r3_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
