"""Aggregate supplementary experiments without reading individual learner data."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);a=ap.parse_args()
OUT=a.output.resolve();PREV=a.previous.resolve();SUM=OUT/'summary';SUM.mkdir(exist_ok=True)
PAIRKEY=['pair_id','day','seed','base_model'];ROLLKEY=['triple_id','day','seed','base_model']
GROUP=['day','base_model'];UGROUP=GROUP+['fraction','strategy'];CL=['provider','base_course_token']
MODES=['baseline','recent','joint'];NBOOT=100000

def summarize(df,unit,metrics,group,family=None):
    keys=list(group)+CL+[unit]
    means=df.groupby(keys,as_index=False)[metrics].mean()
    counts=df.groupby(keys).size().rename('supported_seeds').reset_index()
    means=means.merge(counts,on=keys,validate='one_to_one')
    rows=[]
    for key,g in means.groupby(group,sort=True):
        sums=g.groupby(CL)[metrics].sum();sizes=g.groupby(CL).size()
        n=len(sums);rng=np.random.default_rng(20261012)
        weights=rng.multinomial(n,np.repeat(1/n,n),size=NBOOT)
        boot=(weights@sums.to_numpy())/(weights@sizes.to_numpy())[:,None]
        for j,m in enumerate(metrics):
            lo,hi=np.quantile(boot[:,j],[.025,.975])
            row=dict(zip(group,key if isinstance(key,tuple) else (key,)))|{'metric':m,'mean':g[m].mean(),'point_low':lo,'point_high':hi,
                 'units':len(g),'clusters':n,'supported_jobs':int(g.supported_seeds.sum()),'min_seeds_per_unit':g.supported_seeds.min(),
                 'max_seeds_per_unit':g.supported_seeds.max()}
            if family and m=='A':
                low,high=np.quantile(boot[:,j],[.05/(2*family),1-.05/(2*family)])
                row.update({'family_size':family,'family_low':low,'family_high':high})
            rows.append(row)
    return means,pd.DataFrame(rows)

def pair_data(mode):
    f=pd.read_csv(OUT/mode/'outcome_blind_future_by_job.csv')
    f['historical_Brier']=f.target_raw_Brier-f.G_T
    f['oof_Brier']=f.target_raw_Brier-f.G_T_star
    return f

def update_data(mode):
    d=pd.read_csv(OUT/mode/'conservative/conservative_by_job.csv')
    ref=pd.read_csv(OUT/mode/'rollout_by_job.csv')
    keep=['raw_Brier','t2_prevalence','recent_null_Brier','maturity_lead_days']
    d=d.merge(ref[ROLLKEY+['fraction']+keep],on=ROLLKEY+['fraction'],validate='many_to_one')
    assert np.allclose(d.loc[d.supported,'advantage'],(d.historical_Brier-d.updated_Brier)[d.supported])
    return d

all_a=[];all_abs=[];all_support=[];all_course=[];all_loco=[];all_impact=[];all_common=[]
for mode in MODES:
    assert (OUT/mode/'completion.json').exists(),f'{mode} incomplete'
    d=pair_data(mode)
    vals=['G_S','G_T','G_T_star','A','target_prevalence','target_raw_Brier','historical_Brier','oof_Brier','target_n','source_cal_n']
    pm,ci=summarize(d,'pair_id',vals,GROUP,12 if mode!='baseline' else None)
    pm.to_csv(SUM/f'{mode}_pair_means.csv',index=False)
    ci.insert(0,'mode',mode);all_a.append(ci)
    u=update_data(mode);uv=u[u.supported].copy()
    um,uci=summarize(uv,'triple_id',['advantage','raw_Brier','historical_Brier','updated_Brier','recent_null_Brier','t2_prevalence','t2_n','labels'],UGROUP)
    um.to_csv(SUM/f'{mode}_triple_means.csv',index=False)
    uci.insert(0,'mode',mode);all_abs.append(uci)
    support=pd.read_csv(OUT/mode/'outcome_blind_future_support.csv')
    roll=pd.read_csv(OUT/mode/'rollout_support.csv')
    for day in (7,14,28):
        psub=support[support.day.eq(day)];rsub=roll[roll.day.eq(day)]
        pyes=psub[psub.admissible];ryes=rsub[rsub.admissible]
        pp=d[d.day.eq(day)].drop_duplicates('pair_id')
        uu=uv[uv.day.eq(day)].drop_duplicates('triple_id')
        all_support.append({'mode':mode,'day':day,'candidate_pairs':psub.pair_id.nunique(),'supported_pairs':pyes.pair_id.nunique(),
          'pair_clusters':len(pp[CL].drop_duplicates()),'supported_pair_seed_roles':len(pyes),'pair_model_fits':len(d[d.day.eq(day)]),
          'candidate_triples':rsub.triple_id.nunique(),'supported_triples':ryes.triple_id.nunique(),'triple_clusters':len(uu[CL].drop_duplicates()),
          'rollout_model_fits':len(ryes)*2,'supported_updates':int(u[u.day.eq(day)].supported.sum()),'attempted_updates':len(u[u.day.eq(day)]),
          'target_n_macro':pm[pm.day.eq(day)].target_n.mean(),'t2_n_macro':um[um.day.eq(day)&um.fraction.eq(1)&um.strategy.eq('full')].t2_n.mean()})
    for key,g in um.groupby(UGROUP):
        point=g.advantage.mean()
        courses=g.groupby(CL).agg(course_mean_U=('advantage','mean'),triple_count=('triple_id','size')).reset_index()
        for row in courses.to_dict('records'):
            all_course.append({'mode':mode,**dict(zip(UGROUP,key)),**row})
            keep=~((g.provider==row['provider'])&(g.base_course_token==row['base_course_token']))
            loo=g.loc[keep,'advantage'].mean()
            all_loco.append({'mode':mode,**dict(zip(UGROUP,key)),**{c:row[c] for c in CL},'omitted_triples':row['triple_count'],
                             'full_mean_U':point,'leave_one_course_out_U':loo,'change':loo-point,'sign_changed':bool((loo>0)!=(point>0))})
        cv=courses.course_mean_U
        loc=pd.DataFrame([row for row in all_loco if row['mode']==mode and all(row[k]==v for k,v in zip(UGROUP,key))])
        all_impact.append({'mode':mode,**dict(zip(UGROUP,key)),'mean_U':point,'clusters':len(courses),'positive_course_fraction':float(cv.gt(0).mean()),
          'course_U_min':cv.min(),'course_U_median':cv.median(),'course_U_max':cv.max(),
          'loco_U_min':loc.leave_one_course_out_U.min(),'loco_U_max':loc.leave_one_course_out_U.max(),
          'largest_abs_loco_change':loc.change.abs().max(),'any_loco_sign_change':bool(loc.sign_changed.any())})
    print('SUMMARIZED',mode,flush=True)

for mode in ['recent','joint']:
    b=pair_data('baseline');s=pair_data(mode)
    m=b.merge(s,on=PAIRKEY,suffixes=('_baseline','_sensitivity'),validate='one_to_one').copy()
    for col in CL:m[col]=m[col+'_baseline'];assert (m[col+'_baseline']==m[col+'_sensitivity']).all()
    m['A_difference']=m.A_sensitivity-m.A_baseline
    _,ci=summarize(m,'pair_id',['A_baseline','A_sensitivity','A_difference'],GROUP,12)
    ci.insert(0,'mode',mode);ci.insert(1,'comparison','common_supported_pair_jobs');all_common.append(ci)
    # Same-supported jobs do not mean identical learners; this remains a population sensitivity.
    b=update_data('baseline');s=update_data(mode)
    b=b[b.supported];s=s[s.supported]
    m=b.merge(s,on=ROLLKEY+['fraction','strategy'],suffixes=('_baseline','_sensitivity'),validate='one_to_one')
    for col in CL:m[col]=m[col+'_baseline']
    m['U_baseline']=m.advantage_baseline;m['U_sensitivity']=m.advantage_sensitivity;m['U_difference']=m.U_sensitivity-m.U_baseline
    _,ci=summarize(m,'triple_id',['U_baseline','U_sensitivity','U_difference'],UGROUP)
    ci.insert(0,'mode',mode);ci.insert(1,'comparison','common_supported_rollout_jobs');all_common.append(ci)

for name,frames in [('pair_intervals',all_a),('rollout_absolute_intervals',all_abs),('common_support_intervals',all_common)]:
    pd.concat(frames,ignore_index=True).to_csv(SUM/f'{name}.csv',index=False)
for name,rows in [('support_summary',all_support),('course_update_distribution',all_course),('leave_one_course_out',all_loco),('course_influence_summary',all_impact)]:
    pd.DataFrame(rows).to_csv(SUM/f'{name}.csv',index=False)

# Original absolute main-table candidate: one row per day/model, all three updates.
abs_ci=pd.concat(all_abs);table=[]
for (mode,day,model),g in abs_ci[abs_ci.fraction.eq(1)].groupby(['mode','day','base_model']):
    row={'mode':mode,'day':day,'base_model':model,'triples':int(g.units.iloc[0]),'clusters':int(g.clusters.iloc[0])}
    for metric in ['t2_prevalence','raw_Brier','historical_Brier','recent_null_Brier']:
        v=g[(g.strategy=='full')&(g.metric==metric)].iloc[0];row[metric]=v['mean'];row[metric+'_low']=v.point_low;row[metric+'_high']=v.point_high
    for method in ['full','intercept','shrink']:
        v=g[(g.strategy==method)&(g.metric=='updated_Brier')].iloc[0];row[method+'_Brier']=v['mean'];row[method+'_Brier_low']=v.point_low;row[method+'_Brier_high']=v.point_high
        v=g[(g.strategy==method)&(g.metric=='advantage')].iloc[0];row[method+'_U']=v['mean'];row[method+'_U_low']=v.point_low;row[method+'_U_high']=v.point_high
    mult=pd.read_csv(OUT/mode/'conservative/multiplicity_six_full_budget.csv');v=mult[(mult.day==day)&(mult.base_model==model)].iloc[0]
    row['full_U_family6_low']=v.family6_low;row['full_U_family6_high']=v.family6_high
    table.append(row)
pd.DataFrame(table).to_csv(SUM/'full_budget_absolute_table.csv',index=False)

# Reproduction comparisons: data and support keys must match before numeric comparison.
checks=[]
for label,oldrel,newrel,keys in [
 ('pair_jobs','review_quality_20261011/outcome_blind_future_by_job.csv','baseline/outcome_blind_future_by_job.csv',PAIRKEY),
 ('rollout_jobs','review_quality_20261011/rollout_by_job.csv','baseline/rollout_by_job.csv',ROLLKEY+['fraction']),
 ('conservative_jobs','review_round2_20261011/conservative_by_job.csv','baseline/conservative/conservative_by_job.csv',ROLLKEY+['fraction','strategy']),
 ('matched_jobs','review_round2_20261011/matched_updates_by_job.csv','baseline/conservative/matched_updates_by_job.csv',ROLLKEY+['fraction','strategy']),
 ('pair_support','review_quality_20261011/outcome_blind_future_support.csv','baseline/outcome_blind_future_support.csv',['pair_id','day','seed']),
 ('rollout_support','review_quality_20261011/rollout_support.csv','baseline/rollout_support.csv',['triple_id','day','seed']),
 ('update_intervals','review_round2_20261011/update_intervals.csv','baseline/conservative/update_intervals.csv',UGROUP)]:
    old=pd.read_csv(PREV/oldrel).sort_values(keys).reset_index(drop=True)
    new=pd.read_csv(OUT/newrel).sort_values(keys).reset_index(drop=True)
    assert old[keys].equals(new[keys]),(label,'key mismatch')
    assert set(old.columns)==set(new.columns),(label,'schema mismatch')
    errs={};nonnum=[]
    for col in old:
        if pd.api.types.is_numeric_dtype(old[col]) and old[col].dtype!=bool:
            ov,nv=old[col].to_numpy(float),new[col].to_numpy(float)
            assert np.array_equal(np.isnan(ov),np.isnan(nv)),(label,col,'missingness')
            err=float(np.nanmax(abs(ov-nv))) if (~np.isnan(ov)).any() else 0.
            if err>0:errs[col]=err
        else:
            if not old[col].fillna('').equals(new[col].fillna('')):nonnum.append(col)
    checks.append({'table':label,'rows':len(old),'max_abs_numeric_error':max(errs.values(),default=0.),'column_errors':errs,'non_numeric_different':nonnum})
(SUM/'reproduction_comparison.json').write_text(json.dumps({'checks':checks,'meaning':'Fresh execution of published algorithms from original ZIP; not independent implementation'},indent=2))
print('ANALYSIS_COMPLETE',flush=True)
