"""OULAD DFMC pilot: historical calibration and untouched cohort evaluation."""
from __future__ import annotations
import argparse, hashlib, json, os, time, zipfile, warnings
import itertools
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from dfmc import DFMC

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
RESULTS = ROOT/'results_verified'
EPS = float(P['probability_clip'])
PROGRESS = ['no_due', 'due_complete', 'due_missing']

def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)

def loss(y,p):
    return float(np.mean((np.asarray(y)-clip(p))**2))

def metrics(y, raw):
    y=np.asarray(y); raw=np.asarray(raw); p=clip(raw)
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.sum(bins==b)/len(y)*abs(np.mean(y[bins==b])-np.mean(p[bins==b])) for b in range(10) if np.any(bins==b))
    both=len(np.unique(y))==2
    return dict(n=len(y), positives=int(y.sum()), negatives=int(len(y)-y.sum()),
      prevalence=float(y.mean()), Brier=loss(y,p), Log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
      ROC_AUC=float(roc_auc_score(y,p)) if both else np.nan,
      AP=float(average_precision_score(y,p)) if both else np.nan,
      ECE_10_equal_width=float(ece), signed_calibration_bias=float(np.mean(p-y)),
      raw_out_of_range_fraction=float(np.mean((raw<0)|(raw>1))),
      unstable_group=bool(len(y)<50 or min(y.sum(),len(y)-y.sum())<10))

class ParametricCalibrator:
    def __init__(self,kind): self.kind=kind
    def design(self,p):
        p=clip(p)
        return np.column_stack([logit(p),np.ones(len(p))]) if self.kind=='Platt' else np.column_stack([np.log(p),-np.log1p(-p),np.ones(len(p))])
    def fit(self,p,y):
        X=self.design(p); y=np.asarray(y)
        init=np.array([1.,0.]) if self.kind=='Platt' else np.array([1.,1.,0.])
        def objective(w):
            z=X@w
            reg=1e-5*np.sum((w-init)**2)
            return np.mean(np.logaddexp(0,z)-y*z)+reg, X.T@(expit(z)-y)/len(y)+2e-5*(w-init)
        result=minimize(objective,init,jac=True,method='L-BFGS-B',bounds=[(0.,30.)]*(len(init)-1)+[(-30.,30.)],options={'maxiter':1000,'ftol':1e-12})
        if not np.isfinite(result.fun) or not result.success: raise ValueError('Parametric calibrator did not converge: '+str(result.message))
        self.w=result.x; self.converged=bool(result.success)
        return self
    def predict(self,p): return expit(self.design(p)@self.w)

def grouped_ids(frame):
    # A student with multiple enrollments is kept intact; first course defines the split stratum only.
    z=frame.sort_values(['id_student','code_module','t']).drop_duplicates('id_student')
    return z.id_student.to_numpy(), (z.code_module.astype(str)+'_'+z.y.astype(str)).to_numpy()

def stratified_split(ids,strata,fraction,seed):
    return train_test_split(ids,test_size=fraction,random_state=seed,stratify=strata)

def nested_order(ids,strata,seed):
    rng=np.random.default_rng(seed); groups={}
    for s in sorted(set(strata)):
        x=ids[strata==s].copy(); rng.shuffle(x); groups[s]=list(x)
    sizes={s:len(v) for s,v in groups.items()}; used={s:0 for s in groups}; total=len(ids); order=[]
    for k in range(total):
        available=[s for s in groups if used[s]<sizes[s]]
        chosen=max(available,key=lambda s:((k+1)*sizes[s]/total-used[s],-used[s]))
        order.append(int(groups[chosen][used[chosen]])); used[chosen]+=1
    return order

def get_season_data(season):
    data=pd.read_csv(ROOT/'data/features.csv.gz')
    if 'course' not in data: data['course']=data.code_module
    spec=P['exploration' if season=='J' else 'confirmation']; modules=spec['modules']
    source=data[data.code_presentation.eq(spec['source']) & data.code_module.isin(modules)].copy()
    target=data[data.code_presentation.eq(spec['target']) & data.code_module.isin(modules)].copy()
    with zipfile.ZipFile(__import__("os").environ["CALTRANSFER_OULAD_ZIP"]) as z:
        info=pd.read_csv(z.open('studentInfo.csv'),usecols=['id_student','code_module','code_presentation'])
    seen=set(info.loc[info.code_presentation.eq(spec['source']),'id_student'])
    exclusion={'target_before_id_exclusion':int(len(target)),'source_full_ids':len(seen)}
    target=target[~target.id_student.isin(seen)].copy()
    exclusion['target_after_source_exclusion']=int(len(target))
    if season=='B':
        exploration_ids=set(json.loads((RESULTS/'J/exploration_ids.json').read_text()))
        target=target[~target.id_student.isin(exploration_ids)].copy()
        exclusion['exploration_ids_excluded']=len(exploration_ids)
    exclusion['target_final']=int(len(target))
    exclusion['target_by_t']={str(t):len(target[target.t.eq(t)]) for t in P['times']}
    return source,target,exclusion

def make_splits(source,seed,out):
    ids,strata=grouped_ids(source)
    train,rest=stratified_split(ids,strata,.4,seed)
    mapping=dict(zip(ids,strata)); reststrata=np.array([mapping[i] for i in rest])
    val,cal=stratified_split(rest,reststrata,.5,seed+1)
    order=nested_order(cal,np.array([mapping[i] for i in cal]),seed+2)
    split={'train':[int(i) for i in train],'base_validation':[int(i) for i in val],'calibration_order':order}
    assert not(set(train)&set(val) or set(train)&set(cal) or set(val)&set(cal))
    (out/f'split_seed{seed}.json').write_text(json.dumps(split),encoding='utf-8')
    return split

def groups(frame,modules,p_only=False):
    if p_only: return np.zeros((len(frame),0))
    a=[(frame.code_module==c).to_numpy(dtype=float) for c in modules]
    a += [(frame.progress_state==g).to_numpy(dtype=float) for g in PROGRESS]
    a += [((frame.code_module==c)&(frame.progress_state==g)).to_numpy(dtype=float) for c in modules for g in PROGRESS]
    return np.column_stack(a)

def matrices(train,other,feature_columns,modules,base):
    def build(d):
        x=d[feature_columns].to_numpy(dtype=float)
        if base=='logistic': x=np.sign(x)*np.log1p(np.abs(x))
        # These same course/progress indicators are also available to the base predictor.
        return np.column_stack([x,groups(d,modules)])
    x=build(train); others=[build(z) for z in other]
    if base=='logistic':
        scaler=StandardScaler().fit(x); x=scaler.transform(x); others=[scaler.transform(z) for z in others]
    return x,others

def train_base(train,val,cal,test,feature_columns,modules,base,seed):
    x,(xv,xc,xt)=matrices(train,[val,cal,test],feature_columns,modules,base)
    candidates=[]
    for config in ([{'C':.1},{'C':1.}] if base=='logistic' else [{'num_leaves':7},{'num_leaves':15}]):
        if base=='logistic': model=LogisticRegression(**config,max_iter=1500,solver='lbfgs',random_state=seed)
        else: model=LGBMClassifier(**config,n_estimators=350,learning_rate=.03,min_child_samples=40,reg_lambda=5.,verbosity=-1,n_jobs=2,random_state=seed,deterministic=True,force_col_wise=True)
        model.fit(x,train.y.to_numpy()); score=loss(val.y,model.predict_proba(xv)[:,1]); candidates.append((score,model,config))
    score,model,config=min(candidates,key=lambda item:item[0])
    return clip(model.predict_proba(xc)[:,1]),clip(model.predict_proba(xt)[:,1]),{'source_validation_Brier':score,'chosen':config,'candidate_scores':[{'config':c,'Brier':s} for s,m,c in candidates]}

def calibration_partition(cal,seed):
    ids,strata=grouped_ids(cal)
    # With tiny course/label strata, preserve labels where possible rather than inventing samples.
    counts=pd.Series(strata).value_counts()
    if counts.min()<2 or int(np.ceil(len(ids)*.3))<len(counts):
        first=cal.sort_values(['id_student','code_module']).drop_duplicates('id_student').set_index('id_student')
        strata=first.loc[ids,'y'].to_numpy()
    fit,valid=stratified_split(ids,strata,.3,seed+301)
    return cal.id_student.isin(fit).to_numpy(),cal.id_student.isin(valid).to_numpy()

def fit_dfmc(pc,yc,gc,pt,gt,fit,valid,seed,**params):
    defaults=dict(P['dfmc']); defaults.pop('objective')
    defaults['num_boost_round']=defaults.pop('rounds');defaults['early_stopping_rounds']=defaults.pop('early_stopping')
    defaults.update(params)
    model=DFMC(seed=seed,n_jobs=2,deterministic=True,force_col_wise=True,**defaults).fit(pc[fit],yc[fit],gc[fit],pc[valid],yc[valid],gc[valid])
    return model.predict(pt,gt),model.predict(pc[valid],gc[valid]),model

def evaluate_rows(pred,context,cal,methods):
    masks=[('all','all',np.ones(len(pred),dtype=bool))]
    for column,kind in [('code_module','course'),('progress_state','progress')]:
        masks.extend((kind,str(v),pred[column].eq(v).to_numpy()) for v in sorted(pred[column].unique()))
    for (course,progress),frame in pred.groupby(['code_module','progress_state'],sort=True):
        masks.append(('course_progress',f'{course}|{progress}',(pred.code_module.eq(course)&pred.progress_state.eq(progress)).to_numpy()))
    rows=[]
    for kind,name,mask in masks:
        if kind=='all': cg=cal
        elif kind=='course': cg=cal[cal.code_module.eq(name)]
        elif kind=='progress': cg=cal[cal.progress_state.eq(name)]
        else:
            c,g=name.split('|'); cg=cal[cal.code_module.eq(c)&cal.progress_state.eq(g)]
        for method in methods:
            rows.append(dict(**context,group_type=kind,group=name,method=method,cal_group_n=len(cg),cal_group_students=int(cg.id_student.nunique()),cal_group_positives=int(cg.y.sum()),cal_group_fit_n=int(cg.inner_fit.sum()),cal_group_valid_n=int((~cg.inner_fit).sum()),**metrics(pred.loc[mask,'y'],pred.loc[mask,method])))
    return rows

def one_job(season,seed,t,base,source,target,split,features,modules,improvement=False):
    begin=time.monotonic(); out=RESULTS/season; out.mkdir(parents=True,exist_ok=True)
    stem=f'{base}_s{seed}_t{t}'; done=out/(stem+('_improved' if improvement else '')+'_metrics.csv')
    if done.exists(): return {'cached':stem}
    src=source[source.t.eq(t)].copy(); test=target[target.t.eq(t)].copy().reset_index(drop=True)
    train=src[src.id_student.isin(split['train'])]; val=src[src.id_student.isin(split['base_validation'])]
    cal=src[src.id_student.isin(split['calibration_order'])].copy().reset_index(drop=True)
    assert not(set(src.id_student)&set(test.id_student))
    cache=out/(stem+'_base.npz'); metadata=out/(stem+'_base.json')
    if cache.exists():
        cached=np.load(cache); pc=cached['pc']; pt=cached['pt']; info=json.loads(metadata.read_text())
        assert np.array_equal(cached['cal_keys'],cal[['code_module','code_presentation','id_student','t']].astype(str).to_numpy()) and np.array_equal(cached['test_keys'],test[['code_module','code_presentation','id_student','t']].astype(str).to_numpy())
    else:
        pc,pt,info=train_base(train,val,cal,test,features,modules,base,seed)
        np.savez_compressed(cache,pc=pc,pt=pt,cal_keys=cal[['code_module','code_presentation','id_student','t']].astype(str).to_numpy(dtype=str),test_keys=test[['code_module','code_presentation','id_student','t']].astype(str).to_numpy(dtype=str))
        info.update(n_train=len(train),n_validation=len(val),n_calibration=len(cal),n_test=len(test))
        metadata.write_text(json.dumps(info,indent=2),encoding='utf-8')
    gt=groups(test,modules); rows=[]; tuning=[]
    for budget in P['calibration_budgets']:
        ids=split['calibration_order'] if budget=='full' else split['calibration_order'][:budget]
        selected=cal.id_student.isin(ids).to_numpy(); ca=cal.loc[selected].copy().reset_index(drop=True)
        pp=pc[selected]; yy=ca.y.to_numpy(); gg=groups(ca,modules); fit,valid=calibration_partition(ca,seed)
        ca['inner_fit']=fit
        context=dict(season=season,seed=seed,t=t,base=base,budget=str(budget),cal_n=len(ca),cal_students=int(ca.id_student.nunique()),cal_fit_n=int(fit.sum()),cal_valid_n=int(valid.sum()))
        pred=test[['code_module','code_presentation','id_student','t','progress_state','y']].copy()
        pred['uncalibrated']=pt
        parametric_details={}
        for kind in ['Platt','Beta']:
            model=ParametricCalibrator(kind).fit(pp,yy); pred[kind]=model.predict(pt)
            parametric_details[kind]={'coefficients':model.w.tolist(),'converged':model.converged}
        iso=IsotonicRegression(out_of_bounds='clip').fit(pp,yy); pred['isotonic']=iso.predict(pt)
        raw,vp,dm=fit_dfmc(pp,yy,gg,pt,gt,fit,valid,seed)
        pred['DFMC']=raw
        defaults={k:P['dfmc'][k] for k in ['learning_rate','bagging_fraction']}
        candidates=[(loss(yy[valid],vp),raw,vp,dm,defaults)]
        for lr,bf in itertools.product(P['dfmc_tuning_grid']['learning_rate'],P['dfmc_tuning_grid']['bagging_fraction']):
            if lr==defaults['learning_rate'] and bf==defaults['bagging_fraction']: continue
            raw2,vp2,m2=fit_dfmc(pp,yy,gg,pt,gt,fit,valid,seed,learning_rate=lr,bagging_fraction=bf)
            candidates.append((loss(yy[valid],vp2),raw2,vp2,m2,{'learning_rate':lr,'bagging_fraction':bf}))
        score,best,bestval,bestmodel,params=min(candidates,key=lambda item:item[0])
        pred['DFMC_tuned']=best
        raw0,_,model0=fit_dfmc(pp,yy,groups(ca,modules,True),pt,groups(test,modules,True),fit,valid,seed)
        pred['DFMC_p_only']=raw0
        tuning.append(dict(**context,parametric=parametric_details,default_iteration=int(dm.best_iteration_),tuned_iteration=int(bestmodel.best_iteration_),tuned_params=params,tuned_validation_Brier=score,grid=[{'params':par,'Brier':s,'iteration':int(m.best_iteration_)} for s,r,v,m,par in candidates]))
        if improvement:
            from improvement import improved_predictions
            extra,details=improved_predictions(pp,yy,gg,pt,gt,fit,valid,seed,ca,test,candidates)
            for name,values in extra.items(): pred[name]=values
            tuning[-1]['improvement']=details
        methods=list(pred.columns[6:]); rows.extend(evaluate_rows(pred,context,ca,methods))
        pred['season']=season;pred['seed']=seed;pred['base']=base;pred['budget']=str(budget)
        suffix='_improved' if improvement else ''
        pred.to_parquet(out/f'{stem}_n{budget}{suffix}_predictions.parquet',index=False)
        # Reproducible effective IDs, including inner split; labels remain in private local artifacts.
        assignments=ca[['id_student','code_module','t','progress_state','y']].copy();assignments['inner_role']=np.where(fit,'fit','validation')
        assignments.to_csv(out/f'{stem}_n{budget}_calibration.csv.gz',index=False,compression='gzip')
    pd.DataFrame(rows).to_csv(done,index=False)
    (out/(stem+('_improved' if improvement else '')+'_tuning.json')).write_text(json.dumps(tuning,indent=2),encoding='utf-8')
    return {'completed':stem,'seconds':round(time.monotonic()-begin,2),'n_test':len(test)}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def fingerprint(improvement=False):
    names=['protocol.json','prepare.py','dfmc.py','run_experiment.py','data/features.csv.gz','data/data_audit.json']
    if improvement: names+=['improvement.py']
    return {n:sha(ROOT/n) for n in names}

def check_manifest(out,improvement,season):
    fp=fingerprint(improvement)
    if season=='B':
        decision=json.loads((ROOT/'improvement_decision.json').read_text(encoding='utf-8'))
        expected=decision['frozen_fingerprint']
        for key,value in expected.items():
            if sha(ROOT/key)!=value: raise RuntimeError('Frozen decision fingerprint mismatch: '+key)
    manifest=out/('manifest_improved.json' if improvement else 'manifest.json')
    if manifest.exists():
        if json.loads(manifest.read_text())['fingerprint']!=fp: raise RuntimeError('Stale result cache; use a fresh run directory')
    else:
        manifest.write_text(json.dumps({'fingerprint':fp,'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'versions':{m:__import__(m).__version__ for m in ['numpy','pandas','scipy','sklearn','lightgbm']}},indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--season',choices=['J','B'],default='J');ap.add_argument('--jobs',type=int,default=2);ap.add_argument('--improvement',action='store_true');ap.add_argument('--seeds',type=int,nargs='*');ap.add_argument('--times',type=int,nargs='*');args=ap.parse_args()
    if args.season=='B' and not (ROOT/'improvement_decision.json').exists(): raise RuntimeError('Freeze improvement decision before independent-season evaluation')
    source,target,exclusion=get_season_data(args.season); out=RESULTS/args.season;out.mkdir(parents=True,exist_ok=True)
    check_manifest(out,args.improvement,args.season)
    if args.season=='J':
        # Conservative exclusion union of all actually available J source/target landmark IDs.
        ids=sorted(set(source.id_student)|set(target.id_student))
        (out/'exploration_ids.json').write_text(json.dumps([int(i) for i in ids]),encoding='utf-8')
    (out/'cohort_audit.json').write_text(json.dumps(exclusion,indent=2),encoding='utf-8')
    audit=json.loads((ROOT/'data/data_audit.json').read_text(encoding='utf-8'))
    features=audit['feature_columns']; modules=P['exploration' if args.season=='J' else 'confirmation']['modules']
    tasks=[]
    for seed in args.seeds or P['seeds']:
        split=make_splits(source,seed,out)
        for t in args.times or P['times']:
            for base in P['base_models']: tasks.append((args.season,seed,t,base,source,target,split,features,modules,args.improvement))
    with threadpool_limits(limits=2):
        for result in Parallel(n_jobs=args.jobs,return_as='generator_unordered')(delayed(one_job)(*task) for task in tasks): print(json.dumps(result),flush=True)
    pattern='*_improved_metrics.csv' if args.improvement else '*_metrics.csv'
    files=[f for f in out.glob(pattern) if args.improvement or '_improved_' not in f.name]
    pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv(out/('metrics_improved.csv' if args.improvement else 'metrics.csv'),index=False)
    print('DONE '+str(out),flush=True)

if __name__=='__main__': main()
