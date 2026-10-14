"""Post-review checks. Run only after full XuetangX data preparation.

Original truth is never used for the new candidate pool, outcome, or role split.
All fitted jobs and learner records stay in the private runtime.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from types import SimpleNamespace
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import primary as p

ROOT = Path(os.environ.get('CALTRANSFER_ROOT', str(Path(__file__).resolve().parents[2])))
OUT = ROOT/'outputs/review_quality_20261011'
WORK = ROOT/'work/xuetangx_experiment4/review_extension'
DAYS=(7,14,28)
SEEDS=(17,29,43,59,71)
MODELS=('logistic','lightgbm')

def initialize():
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
    cols=['provider','base_course_token','source_course_id','target_course_id',
          'source_first_event','source_last_event','target_first_event','target_last_event',
          'observation_gap_days','source_enrollments','target_enrollments']
    candidates=pd.read_csv(ROOT/'outputs/xuetangx_data_audit/consecutive_same_base_candidates.csv',usecols=cols)
    candidates['pair_id']=[f'P{i:03}' for i in range(len(candidates))]
    candidates['structural_eligible']=(candidates.observation_gap_days.ge(7)&candidates.source_enrollments.ge(300)&candidates.target_enrollments.ge(300))
    candidates.to_csv(OUT/'outcome_blind_candidate_audit.csv',index=False)
    pairs=candidates[candidates.structural_eligible].copy()
    endpoint=WORK/'future_all_presentations.parquet'
    if not endpoint.exists():
        con=duckdb.connect(str(ROOT/'work/xuetangx_experiment4/events.duckdb'),read_only=True)
        expressions=','.join(f'max(case when day_exact>{d} and day_exact<={d+7} then 1 else 0 end)::INTEGER as future_event_{d}' for d in DAYS)
        f=con.execute(f'select enroll_id,course_id,{expressions} from events group by enroll_id,course_id').df()
        f.to_parquet(endpoint,index=False);con.close()
    f=pd.read_parquet(endpoint)
    e=pd.read_csv(ROOT/'work/xuetangx_audit/enrollment_index.csv',usecols=['enroll_id','username','course_id'],dtype={'enroll_id':'int64','username':'int64','course_id':str})
    e=e.merge(f,on=['enroll_id','course_id'],validate='one_to_one'); assert len(e)==len(p.ENROLL)
    protocol={'version':1,'status':'post-review exploratory','days':DAYS,'seeds':SEEDS,'models':MODELS,
      'candidate_pool':'all adjacent provider/base-course tokens; >=300 enrollments in both; observed gap >=7 days; no truth filter',
      'endpoint':'no logged event in (anchor+day,anchor+day+7]; observed spans required through endpoint',
      'roles':'60/20/20 source split using new outcome, original fixed seeds; refit all components',
      'user_boundary':'target excludes all source-presentation users; rollout T2 excludes S union T1 users',
      'rollout':'frozen S base predictor scores both T1 and T2; source Platt versus T1 Platt; no T2 labels in fits',
      'budget':'nested random T1 permutations; fractions .1,.2,.5,1; no outcome stratification; failed class support reported',
      'sample_support':'original source role thresholds; target N>=50/minority>=10; T1 rollout N>=50/minority>=10; budget N>=10/two classes',
      'inference':'pair/triple equal after averaging seeds; 2000 base-course cluster resamples; shared-user limitations remain',
      'selection_caveat':'structural pair selection excludes truth; final estimation still conditions on NEW endpoint class support',
      'provenance_caveat':'observed logs define timing; complete provider logging and actual production arrival time unverified'}
    path=OUT/'review_extension_protocol.json'
    encoded=json.dumps(protocol,indent=2)
    if path.exists(): assert json.loads(path.read_text())==json.loads(encoded)
    else: path.write_text(encoded)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path(p.__file__).read_bytes()).hexdigest()
    fp=WORK/'source_fingerprint.txt'
    if fp.exists(): assert fp.read_text()==fingerprint,'Code changed: use a fresh private runtime/cache'
    else: fp.write_text(fingerprint)
    return pairs,e

def set_day(e,day):
    current=e[['enroll_id','username','course_id']].copy();current['y']=1-e[f'future_event_{day}']
    p.BY_COURSE={k:g for k,g in current.groupby('course_id')}

def roles_and_support(pair,day,seed):
    s,t=p.prep_pair(pair,day)
    full=p.BY_COURSE[pair.source_course_id].reset_index(drop=True)
    out={'pair_id':pair.pair_id,'day':day,'seed':seed,'admissible':False,'reason':'source_split_support'}
    if full.y.nunique()!=2 or full.y.value_counts().min()<5:return None,s,t,out
    sd=seed+int(pair.pair_id[1:])*1009
    tr,h=train_test_split(np.arange(len(full)),test_size=.4,random_state=sd,stratify=full.y)
    va,ca=train_test_split(h,test_size=.5,random_state=sd+104729,stratify=full.iloc[h].y)
    roles=np.empty(len(full),dtype=object);roles[tr]='train';roles[va]='validation';roles[ca]='calibration'
    rm=dict(zip(full.enroll_id,roles)); ar=s.enroll_id.map(rm)
    for role in ('train','validation','calibration'):
        g=s[ar.eq(role)];out['source_'+role+'_n']=len(g);out['source_'+role+'_minority']=int(g.y.value_counts().min()) if g.y.nunique()==2 else 0
    tgt=t[t.active&t.unseen];out['target_n']=len(tgt);out['target_minority']=int(tgt.y.value_counts().min()) if tgt.y.nunique()==2 else 0
    span=all((pd.Timestamp(getattr(pair,f'{which}_last_event'))-pd.Timestamp(getattr(pair,f'{which}_first_event'))).total_seconds()/86400>=day+7 for which in ('source','target'))
    out['admissible']=bool(span and out['source_train_minority']>=5 and out['source_validation_minority']>=2 and out['source_calibration_n']>=30 and out['source_calibration_minority']>=10 and out['target_n']>=50 and out['target_minority']>=10)
    out['reason']='supported' if out['admissible'] else ('span_below_endpoint' if not span else 'below_new_outcome_support')
    return rm,s,t,out

def intervals(df,unit,values,group=('day','base_model')):
    keys=list(group)+['provider','base_course_token',unit]
    pm=df.groupby(keys,as_index=False)[values].mean()
    rows=[]
    for key,g in pm.groupby(list(group)):
        clusters=list(g.groupby(['provider','base_course_token'],sort=True).indices.values())
        rng=np.random.default_rng(20261011); draws=rng.integers(0,len(clusters),(2000,len(clusters)))
        a=g[values].to_numpy(); boot=np.array([a[np.concatenate([clusters[i] for i in pick])].mean(axis=0) for pick in draws])
        for j,v in enumerate(values):rows.append(dict(zip(group,key if isinstance(key,tuple) else (key,)))|{'metric':v,'mean':a[:,j].mean(),'ci_low':np.quantile(boot[:,j],.025),'ci_high':np.quantile(boot[:,j],.975),'units':len(g),'clusters':len(clusters),'jobs':len(df.merge(g[keys],on=keys))})
    return pm,pd.DataFrame(rows)

def future(pairs,e):
    p.PRED=WORK/'future_predictions';p.PRED.mkdir(exist_ok=True)
    rows=[];support=[];expected=0
    for day in DAYS:
        set_day(e,day)
        for pair in pairs.itertuples():
            for seed in SEEDS:
                rm,s,t,z=roles_and_support(pair,day,seed);support.append(z)
                if not z['admissible']:continue
                for model in MODELS:
                    expected+=1; rows.append(p.one_job(pair,day,seed,model,s,t,rm))
                    if expected%100==0:print('FUTURE',expected,flush=True)
        pd.DataFrame(support).to_csv(OUT/'outcome_blind_future_support.csv',index=False)
    d=pd.DataFrame(rows);assert len(d)==expected and not d[['pair_id','day','seed','base_model']].duplicated().any()
    d.to_csv(OUT/'outcome_blind_future_by_job.csv',index=False)
    pm,ci=intervals(d,'pair_id',['G_S','G_T','G_T_star','A']);pm.to_csv(OUT/'outcome_blind_future_pair_mean.csv',index=False);ci.to_csv(OUT/'outcome_blind_future_intervals.csv',index=False)
    print('FUTURE_COMPLETE',len(d),flush=True)

def rollout(pairs,e):
    triples=pairs.merge(pairs,left_on='target_course_id',right_on='source_course_id',suffixes=('_a','_b'))
    triples['triple_id']=[f'R{i:03}' for i in range(len(triples))];triples.to_csv(OUT/'rollout_candidates.csv',index=False)
    rows=[];support=[];pred=WORK/'rollout_predictions';pred.mkdir(exist_ok=True)
    for day in DAYS:
        set_day(e,day)
        for q in triples.itertuples():
            pair=SimpleNamespace(**{c:getattr(q,c+'_a') for c in pairs.columns if c+'_a' in triples.columns})
            for seed in SEEDS:
                rm,s,t1,z=roles_and_support(pair,day,seed)
                z['triple_id']=q.triple_id
                # Prepare T2 using S only, then exclude ALL users from both earlier presentations.
                pair2=SimpleNamespace(source_course_id=q.source_course_id_a,target_course_id=q.target_course_id_b)
                _,t2=p.prep_pair(pair2,day)
                earlier=set(p.BY_COURSE[q.source_course_id_a].username)|set(p.BY_COURSE[q.target_course_id_a].username)
                t1=t1[t1.active&t1.unseen].copy();t2=t2[t2.active&~t2.username.isin(earlier)].copy()
                span=(pd.Timestamp(q.target_last_event_b)-pd.Timestamp(q.target_first_event_b)).total_seconds()/86400
                lead=(pd.Timestamp(q.target_first_event_b)+pd.Timedelta(days=day)-(pd.Timestamp(q.target_first_event_a)+pd.Timedelta(days=day+7))).total_seconds()/86400
                source_lead=(pd.Timestamp(q.target_first_event_a)+pd.Timedelta(days=day)-(pd.Timestamp(q.source_first_event_a)+pd.Timedelta(days=day+7))).total_seconds()/86400
                z.update({'t1_n':len(t1),'t2_n':len(t2),'t1_outcome_to_t2_prediction_days':lead,'s_outcome_to_t1_prediction_days':source_lead})
                z['admissible']=bool(z['admissible'] and len(t2)>=50 and t2.y.nunique()==2 and t2.y.value_counts().min()>=10 and span>=day+7 and lead>0 and source_lead>0)
                z['reason']='supported' if z['admissible'] else 'source_t1_t2_support_or_timing';support.append(z)
                if not z['admissible']:continue
                assert not set(t1.username)&set(p.BY_COURSE[q.source_course_id_a].username)
                assert not set(t2.username)&earlier
                role=s.enroll_id.map(rm);vocab=p.source_vocabulary(s,role);src=p.apply_vocabulary(s,vocab)
                targets=p.apply_vocabulary(pd.concat([t1,t2],ignore_index=True),vocab)
                for model in MODELS:
                    cache=pred/f'{q.triple_id}_d{day}_{model}_s{seed}.json'
                    if cache.exists():rows.extend(json.loads(cache.read_text()));continue
                    cal,pc,pt,_=p.fit_base(src,targets,role,model,seed)
                    p1,p2=pt[:len(t1)],pt[len(t1):];ys=t2.y.to_numpy()
                    hist=p.apply_cal(p2,p.fit_cal(pc,cal.y,'Platt'));raw=p.brier(ys,p2);bs_hist=p.brier(ys,hist)
                    perm=np.random.default_rng(20261011+seed+int(q.triple_id[1:])*1009+day).permutation(len(t1))
                    job=[];saved=pd.DataFrame({'enroll_id':t2.enroll_id,'username':t2.username,'y':ys,'raw':p2,'historical':hist})
                    for frac in (.1,.2,.5,1.):
                        idx=perm[:max(1,int(np.floor(frac*len(t1))))]; ok=len(idx)>=10 and t1.iloc[idx].y.nunique()==2
                        row={'triple_id':q.triple_id,'provider':q.provider_a,'base_course_token':q.base_course_token_a,'day':day,'seed':seed,'base_model':model,'fraction':frac,'labels':len(idx),'budget_supported':ok,'source_cal_n':len(cal),'t1_n':len(t1),'t2_n':len(t2),'t2_prevalence':float(ys.mean()),'maturity_lead_days':lead,'raw_Brier':raw,'historical_Brier':bs_hist}
                        if ok:
                            updated=p.apply_cal(p2,p.fit_cal(p1[idx],t1.iloc[idx].y,'Platt'));null=np.full(len(t2),t1.iloc[idx].y.mean());bs=p.brier(ys,updated)
                            row.update({'updated_Brier':bs,'recent_null_Brier':p.brier(ys,null),'update_advantage':bs_hist-bs,'update_gain_raw':raw-bs,'update_advantage_null':p.brier(ys,null)-bs})
                            saved[f'updated_{frac}']=updated
                        job.append(row)
                    cache.with_suffix('.parquet').parent.mkdir(exist_ok=True);saved.to_parquet(cache.with_suffix('.parquet'),index=False)
                    cache.write_text(json.dumps(job));rows.extend(job)
            print('ROLLOUT',day,q.triple_id,len(rows),flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'rollout_by_job.csv',index=False);pd.DataFrame(support).to_csv(OUT/'rollout_support.csv',index=False)
    assert not d[['triple_id','day','seed','base_model','fraction']].duplicated().any()
    valid=d[d.budget_supported];pm,ci=intervals(valid,'triple_id',['update_advantage','update_gain_raw','update_advantage_null','raw_Brier','historical_Brier','updated_Brier','recent_null_Brier'],group=('day','base_model','fraction'))
    pm.to_csv(OUT/'rollout_triple_mean.csv',index=False);ci.to_csv(OUT/'rollout_intervals.csv',index=False)
    print('ROLLOUT_COMPLETE',len(d),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['future','rollout']);args=a.parse_args()
    pairs,enroll=initialize()
    (future if args.mode=='future' else rollout)(pairs,enroll)
