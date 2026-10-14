"""Portable entry point. All private data and predictions stay inside --run-dir."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
SEEDS = (17, 29, 43, 59, 71)
DAYS = (7, 14, 21, 28)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def source_manifest():
    paths=[REPO/'reproduce.py',REPO/'requirements.txt']
    paths += [p for p in (REPO/'work').rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    files={p.relative_to(REPO).as_posix():digest(p) for p in sorted(paths)}
    encoded=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
    return files,hashlib.sha256(encoded).hexdigest()
def init(args):
    root=args.run_dir.resolve()
    if root==REPO or REPO in root.parents:
        raise ValueError('Use a fresh private --run-dir outside the repository')
    cfgpath=root/'run_config.json'
    cfg=json.loads(cfgpath.read_text()) if cfgpath.exists() else {}
    mode='quick' if args.command=='quick' else ('smoke' if getattr(args,'smoke',False) else 'full')
    files,fingerprint=source_manifest()
    if cfg:
        if cfg.get('config_version')!=2:
            raise ValueError('Legacy runtime lacks content fingerprints; use a fresh --run-dir')
        if cfg.get('mode')!=mode:
            raise ValueError('Quick, full and smoke runtimes are mutually exclusive; use a fresh --run-dir')
        if cfg.get('source_manifest_sha256')!=fingerprint:
            raise ValueError('Distributed source/config changed; use a fresh --run-dir')
        for relative,expected in cfg['source_files'].items():
            if relative.startswith('work/'):
                actual=root/relative
                if not actual.exists() or digest(actual)!=expected:
                    raise ValueError('Runtime source/config changed: '+relative+'; use a fresh --run-dir')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Existing nonempty directory has no verified configuration; use a fresh --run-dir')
    inputs=dict(cfg.get('inputs',{}))
    for key in ('oulad_zip','xuetangx_zip'):
        value=getattr(args,key,None)
        if value and key in cfg and Path(cfg[key]).resolve()!=value.resolve():
            raise ValueError(f'{key} path changed; use a fresh --run-dir')
        current=value.resolve() if value else (Path(cfg[key]) if key in cfg else None)
        if current is not None:
            record={'path':str(current),'sha256':digest(current),'bytes':current.stat().st_size}
            if key in inputs and record!=inputs[key]:
                raise ValueError(f'{key} contents changed (including replacement at the same path); use a fresh --run-dir')
            inputs[key]=record
    if 'prepared_xuetangx_zip' in cfg:
        if digest(cfg['prepared_xuetangx_zip'])!=cfg['prepared_xuetangx_sha256']:
            raise ValueError('Prepared smoke subset changed; use a fresh --run-dir')
    frozen_sha=digest(REPO/'frozen/aggregate_results.json') if mode=='quick' else None
    if cfg and mode=='quick' and cfg.get('frozen_evidence_sha256')!=frozen_sha:
        raise ValueError('Frozen aggregate evidence changed; use a fresh --run-dir')
    # No runtime file has been created or overwritten before all checks above pass.
    root.mkdir(parents=True,exist_ok=True)
    for p in (REPO/'work').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json'):
            q=root/p.relative_to(REPO)
            if not q.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for rel in ('outputs','work/xuetangx_audit','work/xuetangx_experiment4/raw','outputs/xuetangx_experiment4','outputs/xuetangx_gap_supplement'):
        (root/rel).mkdir(parents=True,exist_ok=True)
    for key in ('oulad_zip','xuetangx_zip'):
        value=getattr(args,key,None)
        if value: cfg[key]=str(value.resolve())
    cfg.update(config_version=2,mode=mode,smoke=mode=='smoke',inputs=inputs,
               source_manifest_sha256=fingerprint,source_files=files,frozen_evidence_sha256=frozen_sha)
    cfg.update(threads=args.threads,memory=args.memory)
    cfgpath.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    env=os.environ.copy();env['CALTRANSFER_ROOT']=str(root)
    for key in ('oulad_zip','xuetangx_zip'):
        if key in cfg: env['CALTRANSFER_'+key.upper()]=cfg[key]
    if 'prepared_xuetangx_zip' in cfg:env['CALTRANSFER_XUETANGX_ZIP']=cfg['prepared_xuetangx_zip']
    env['CALTRANSFER_THREADS']=str(cfg['threads']);env['CALTRANSFER_MEMORY']=cfg['memory']
    if cfg.get('smoke'): env['CALTRANSFER_SMOKE']='1'
    return root,cfg,env
def execute(root,env,relative,*args):
    command=[sys.executable,str(root/relative),*map(str,args)]
    print('RUN',relative,*args,flush=True)
    subprocess.run(command,cwd=root,env=env,check=True)
def archive_check(path,members):
    with zipfile.ZipFile(path) as z:
        for name in members:
            if name not in z.namelist(): raise ValueError(f'Missing required archive member: {name}')
            if z.getinfo(name).is_dir(): raise ValueError(name)
    return {'sha256':digest(path),'bytes':Path(path).stat().st_size,'required_members':members}
def extract_xuetangx(path,dest,smoke):
    import pandas as pd
    # Only exact known members are opened; arbitrary archive paths are never extracted.
    courses={'course-v1:MITx+6_00_1x+2015_T1','course-v1:MITx+6_00_1x+2015_T2'}
    for split in ('train','test'):
        target=dest/f'{split}.csv'
        with zipfile.ZipFile(path) as z, z.open(f'{split}/{split}.csv') as src:
            if not smoke:
                with target.open('wb') as out: shutil.copyfileobj(src,out,8*1024*1024)
            else:
                first=True
                for chunk in pd.read_csv(src,chunksize=400000,dtype=str,keep_default_na=False):
                    selected=chunk[chunk.course_id.isin(courses)]
                    selected.to_csv(target,mode='w' if first else 'a',header=first,index=False)
                    first=False
    if smoke:
        subset=dest/'real_course_smoke.zip'
        with zipfile.ZipFile(subset,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for split in ('train','test'): z.write(dest/f'{split}.csv',f'{split}/{split}.csv')
        return subset
    return Path(path)
def prepare(args):
    root,cfg,env=init(args);records={}
    if args.dataset in ('oulad','both'):
        p=cfg.get('oulad_zip')
        if not p: raise ValueError('--oulad-zip is required')
        records['oulad']=archive_check(p,['OULAD.names','studentInfo.csv','studentRegistration.csv','courses.csv','assessments.csv','studentAssessment.csv','studentVle.csv','vle.csv'])
        # All four landmarks are necessary even for one-job smoke: original ID roles use their union.
        execute(root,env,'work/oulad_experiment/prepare.py','--zip',p,'--output',root/'work/oulad_experiment/data')
    if args.dataset in ('xuetangx','both'):
        p=cfg.get('xuetangx_zip')
        if not p: raise ValueError('--xuetangx-zip is required')
        records['xuetangx']=archive_check(p,['train/train.csv','test/test.csv'])
        used=extract_xuetangx(p,root/'work/xuetangx_experiment4/raw',cfg.get('smoke',False))
        if cfg.get('smoke'):
            cfg['prepared_xuetangx_zip']=str(used);cfg['prepared_xuetangx_sha256']=digest(used)
            (root/'run_config.json').write_text(json.dumps(cfg,indent=2))
            env['CALTRANSFER_XUETANGX_ZIP']=str(used)
        for script in ('audit','analyze_courses','p0_freeze','features','roles'):
            execute(root,env,f'work/xuetangx_experiment4/{script}.py')
    (root/'input_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
def run_oulad(root,env,cfg,smoke=False):
    runner=root/'work/oulad_experiment/run_primary.py'
    if not runner.exists(): raise FileNotFoundError('OULAD run_primary.py has not been integrated')
    argv=['--code-dir',root/'work/oulad_experiment','--work-dir',root/'work/oulad_experiment','--oulad-zip',cfg['oulad_zip'],'--results-dir',root/'outputs/oulad_experiment3']
    if smoke: argv+=['--smoke']
    execute(root,env,'work/oulad_experiment/run_primary.py',*argv)
    if not smoke:
        execute(root,env,'work/oulad_experiment/audit_and_summarize.py','--oulad-zip',cfg['oulad_zip'],'--experiment-root',root/'work/oulad_experiment','--experiment3-results',root/'outputs/oulad_experiment3','--output',root/'outputs/revision_review/oulad')
def smoke_xuetangx(root,env):
    execute(root,env,'work/xuetangx_experiment4/smoke_job.py')
def run(args):
    root,cfg,env=init(args)
    if cfg.get('smoke') and not args.smoke: raise ValueError('This is a smoke runtime; create a fresh full run directory')
    if args.dataset in ('oulad','both'): run_oulad(root,env,cfg,args.smoke)
    if args.dataset in ('xuetangx','both'):
        if args.smoke: smoke_xuetangx(root,env);return
        stages=['primary','analyze_primary']
        for script in stages: execute(root,env,f'work/xuetangx_experiment4/{script}.py')
        for day in DAYS:
            execute(root,env,'work/xuetangx_experiment4/secondary_calibration.py',day)
            for seed in SEEDS: execute(root,env,'work/xuetangx_experiment4/dfmc_robustness.py',day,seed)
        for script in ('history_controls','earliest_latest','zero_fill','summarize_robustness','summarize_dfmc','mechanisms','meta','common_pair_sensitivity','unique_user_course_sensitivity','label_window_audit','identity_refit','future_endpoint_refit','analyze_gap_supplement'):
            execute(root,env,f'work/xuetangx_experiment4/{script}.py')
        review=root/'work/xuetangx_experiment4/review_evidence.py'
        if review.exists():
            execute(root,env,'work/xuetangx_experiment4/review_evidence.py','--primary-dir',root/'outputs/xuetangx_experiment4','--gap-dir',root/'outputs/xuetangx_gap_supplement','--prediction-dir',root/'work/xuetangx_experiment4/predictions','--out',root/'outputs/revision_review/xuetangx')
        for mode in ('future','rollout'):
            execute(root,env,'work/xuetangx_experiment4/review_extension.py',mode)
        execute(root,env,'work/xuetangx_experiment4/review_diagnostics.py')
        execute(root,env,'work/xuetangx_experiment4/review_reporting.py')
        for mode in ('fit','summarize'):
            execute(root,env,'work/xuetangx_experiment4/conservative_updates.py',mode)
        execute(root,env,'work/xuetangx_experiment4/conservative_reporting.py')
        execute(root,env,'work/manuscript/make_figures.py')
def quick(args):
    import numpy as np
    import pandas as pd
    root,_,env=init(args)
    package=json.loads((REPO/'frozen/aggregate_results.json').read_text(encoding='utf-8'))
    written=[]
    for key,item in package['tables'].items():
        assert hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256'],key
        p=root/'outputs'/key;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text'],encoding='utf-8');written.append(key)
    data=root/'outputs/xuetangx_experiment4'
    raw=pd.read_csv(data/'primary_by_seed.csv');ref=pd.read_csv(data/'primary_time_model_summary.csv')
    assert np.allclose(raw.A,raw.G_T_star-raw.G_T,atol=1e-12)
    pair=raw.groupby(['pair_id','day','base_model'],as_index=False)[['G_S','G_T','G_T_star','A']].mean()
    summary=pair.groupby(['day','base_model'],as_index=False)[['G_S','G_T','G_T_star','A']].mean()
    errors={}
    check=summary.merge(ref,on=['day','base_model'])
    for c in ['G_S','G_T','G_T_star','A']:
        errors[c]=float(np.max(abs(check[c]-check[c+'_macro'])));assert errors[c]<1e-10
    summary.to_csv(root/'recomputed_primary_summary.csv',index=False)
    o=pd.read_csv(root/'outputs/oulad_experiment3/decomposition_by_seed.csv')
    assert np.allclose(o.A_Platt,o.G_T_star_Platt-o.G_T_Platt,atol=1e-12)
    oc=o.groupby(['season','course','day','base'],as_index=False)[['G_S_Platt','G_T_Platt','G_T_star_Platt','A_Platt']].mean()
    oc.to_csv(root/'recomputed_oulad_cells.csv',index=False)
    review=root/'outputs/revision_review/xuetangx'
    if (review/'target_prevalence_baseline_by_job.csv').exists():
        null=pd.read_csv(review/'target_prevalence_baseline_by_job.csv')
        assert np.allclose(null.Platt_benefit_over_null,null.target_null_Brier-null.target_Platt_Brier,atol=1e-12)
        nullpairs=null.groupby(['pair_id','day','base_model'],as_index=False)[['target_null_Brier','target_Platt_Brier','Platt_benefit_over_null']].mean()
        nullsummary=nullpairs.groupby(['day','base_model'],as_index=False)[['target_null_Brier','target_Platt_Brier','Platt_benefit_over_null']].mean()
        nullsummary.to_csv(root/'recomputed_prevalence_baseline.csv',index=False)
        frozen=pd.read_csv(review/'target_prevalence_baseline_summary.csv')
        joined=nullsummary.merge(frozen,on=['day','base_model'],suffixes=('_computed','_frozen'))
        assert np.allclose(joined.Platt_benefit_over_null_computed,joined.Platt_benefit_over_null_frozen,atol=1e-12)
    new=root/'outputs/review_quality_20261011'
    if (new/'outcome_blind_future_by_job.csv').exists():
        future=pd.read_csv(new/'outcome_blind_future_by_job.csv')
        assert np.allclose(future.A,future.G_T_star-future.G_T,atol=1e-12)
        roll=pd.read_csv(new/'rollout_by_job.csv');v=roll[roll.budget_supported]
        assert np.allclose(v.update_advantage,v.historical_Brier-v.updated_Brier,atol=1e-12)
        placebo=pd.read_csv(new/'placebo_by_repeat.csv');v=placebo[placebo.supported]
        assert np.allclose(v.temporal_minus_random,v.temporal_gap_matched-v.random_within_presentation_gap,atol=1e-12)
        execute(root,env,'work/xuetangx_experiment4/review_reporting.py')
    round2=root/'outputs/review_round2_20261011'
    update_check=None
    if (round2/'conservative_by_job.csv').exists():
        reference=pd.read_csv(round2/'update_intervals.csv').sort_values(['day','base_model','fraction','strategy']).reset_index(drop=True)
        execute(root,env,'work/xuetangx_experiment4/conservative_updates.py','summarize')
        regenerated=pd.read_csv(round2/'update_intervals.csv').sort_values(['day','base_model','fraction','strategy']).reset_index(drop=True)
        assert len(reference)==len(regenerated)==96
        numerical=['mean','point_low','point_high','family96_low','family96_high','family6_low','family6_high']
        assert np.allclose(reference[numerical],regenerated[numerical],atol=1e-10,equal_nan=True)
        update_check={'comparisons':96,'bootstrap_draws_each':100000,'maximum_absolute_interval_error':float(np.nanmax(abs(reference[numerical].to_numpy()-regenerated[numerical].to_numpy()))),'scope':'Aggregate-score reaggregation and resampling, not calibrator fitting'}
        execute(root,env,'work/xuetangx_experiment4/conservative_reporting.py')
    report={'status':'passed','meaning':'Frozen aggregate arithmetic and table regeneration only; no model training','tables_exported':len(written),'max_absolute_primary_summary_errors':errors,'oulad_cells':len(oc),'all_landmark_jobs_including_day21':len(raw),'primary_three_landmark_jobs':int(raw.day.isin([7,14,28]).sum()),'conservative_update_check':update_check}
    (root/'quick_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    execute(root,env,'work/manuscript/make_figures.py')
    execute(root,env,'work/review_round3_20261014/reproduce_r3.py','quick','--root',root,'--repo',REPO)
    report['round3']=json.loads((root/'r3_verification.json').read_text())
    (root/'quick_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

def review3(args):
    root,cfg,env=init(args)
    execute(root,env,'work/review_round3_20261014/reproduce_r3.py','raw','--root',root,'--repo',REPO)

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for cmd in ('prepare','run','quick','review3'):
        q=s.add_parser(cmd);q.add_argument('--run-dir',type=Path,required=True);q.add_argument('--threads',type=int,default=4);q.add_argument('--memory',default='4GB')
        if cmd!='quick':
            q.add_argument('--dataset',choices=['oulad','xuetangx','both'],default='both');q.add_argument('--oulad-zip',type=Path);q.add_argument('--xuetangx-zip',type=Path);q.add_argument('--smoke',action='store_true')
    args=p.parse_args();globals()[args.command](args)
if __name__=='__main__':main()
