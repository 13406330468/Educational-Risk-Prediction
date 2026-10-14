"""Fixed-prediction post-review diagnostics; no training-variance claim."""
from __future__ import annotations
import os,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import primary as p
from review_extension import intervals
ROOT=Path(os.environ.get('CALTRANSFER_ROOT',str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'outputs/review_quality_20261011';OUT.mkdir(exist_ok=True,parents=True)
WORK=ROOT/'work/xuetangx_experiment4/review_diagnostics';WORK.mkdir(exist_ok=True,parents=True)

def main():
    baseline=pd.read_csv(ROOT/'outputs/revision_review/xuetangx/target_prevalence_baseline_pair_mean.csv')
    rows=[]
    for (day,model),g in baseline.groupby(['day','base_model']):
        clusters=list(g.groupby(['provider','base_course_token']).indices.values());rng=np.random.default_rng(20261011)
        vals=g[['target_Platt_Brier','target_null_Brier','raw_Brier']].to_numpy()
        bs=[]
        for _ in range(2000):
            mean=vals[np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))])].mean(axis=0)
            bs.append(1-mean[0]/mean[1])
        mean=vals.mean(axis=0)
        rows.append({'day':day,'base_model':model,'pairs':len(g),'BSS_null':1-mean[0]/mean[1],'ci_low':np.quantile(bs,.025),'ci_high':np.quantile(bs,.975),'null_fraction_of_raw_to_Platt_gain':(mean[2]-mean[1])/(mean[2]-mean[0])})
    pd.DataFrame(rows).to_csv(OUT/'brier_skill_summary.csv',index=False)
    primary=pd.read_csv(ROOT/'outputs/xuetangx_experiment4/primary_by_seed.csv');primary=primary[primary.day.isin([7,14,28])]
    keys=['pair_id','provider','base_course_token','day','base_model']
    pm=primary.groupby(keys,as_index=False).mean(numeric_only=True)
    influence=[];support=[];association=[]
    for (day,model),g in pm.groupby(['day','base_model']):
        for (provider,token),sub in g.groupby(['provider','base_course_token']):
            keep=g[~(g.provider.eq(provider)&g.base_course_token.eq(token))]
            influence.append({'day':day,'base_model':model,'omitted_provider':provider,'omitted_base_course':token,'A_without_cluster':keep.A.mean(),'A_full':g.A.mean(),'removed_pairs':len(sub)})
        for c in ['source_cal_n','target_n','source_prevalence','target_prevalence','source_cal_1','target_1']:
            support.append({'day':day,'base_model':model,'quantity':c,'min':g[c].min(),'q1':g[c].quantile(.25),'median':g[c].median(),'q3':g[c].quantile(.75),'max':g[c].max()})
        for c in ['prevalence_shift','alpha_drift','beta_drift']:
            rho=spearmanr(g.A,g[c].abs()).statistic
            association.append({'day':day,'base_model':model,'quantity':'abs_'+c,'spearman_rho':rho,'pairs':len(g)})
    pd.DataFrame(influence).to_csv(OUT/'leave_one_course_out.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'sample_support_distribution.csv',index=False)
    pd.DataFrame(association).to_csv(OUT/'descriptive_drift_associations.csv',index=False)
    placebo=[];diag=[];loss_rows=[]
    for i,q in enumerate(primary.itertuples()):
        path=p.WORK/'predictions'/f'{q.pair_id}_d{q.day}_{q.base_model}_s{q.seed}.parquet'
        f=pd.read_parquet(path);t=f[f.population.eq('target_primary')].reset_index(drop=True);s=f[f.population.eq('source_calibration')].reset_index(drop=True)
        assert len(t)==q.target_n and t.username.is_unique
        meta={'pair_id':q.pair_id,'provider':q.provider,'base_course_token':q.base_course_token,'day':q.day,'base_model':q.base_model,'seed':q.seed}
        u=t[['username']].copy();u['delta']=(t.y-t.source_Platt)**2-(t.y-t.oof_Platt)**2
        for k,v in meta.items():u[k]=v
        loss_rows.append(u)
        cache=WORK/(path.stem+'.json')
        if cache.exists():
            item=json.loads(cache.read_text());placebo+=item['placebo'];diag+=item['diagnostics'];continue
        ds=[];ps=[]
        for col in ['raw','source_Platt','oof_Platt']:
            alpha=p.fit_cal(t[col],t.y,'intercept')[1];beta=p.fit_cal(t[col],t.y,'Platt')[0]
            ds.append(meta|{'method':col,'calibration_intercept':alpha,'calibration_slope':beta,'ECE10':p.ece_quantile(t.y.to_numpy(),t[col].to_numpy())})
        m=min(len(s),len(t)//3)
        for rep in range(5):
            rng=np.random.default_rng(20261011+q.seed*13+q.day*37+int(q.pair_id[1:])*1009+rep*104729)
            z=rng.permutation(len(t));h=t.iloc[z[:m]];c=t.iloc[z[m:2*m]];e=t.iloc[z[2*m:]];sc=s.iloc[rng.permutation(len(s))[:m]]
            supported=m>=30 and all(a.y.nunique()==2 and a.y.value_counts().min()>=5 for a in [h,c,sc]) and len(e)>=30
            row=meta|{'repeat':rep,'labels_per_calibrator':m,'evaluation_n':len(e),'supported':supported}
            if supported:
                predictions=[p.apply_cal(e.raw,p.fit_cal(a.raw,a.y,'Platt')) for a in (sc,h,c)]
                losses=[p.brier(e.y,a) for a in predictions]
                row.update({'temporal_gap_matched':losses[0]-losses[2],'random_within_presentation_gap':losses[1]-losses[2],'temporal_minus_random':losses[0]-losses[1]})
            ps.append(row)
        cache.write_text(json.dumps({'placebo':ps,'diagnostics':ds}));placebo+=ps;diag+=ds
        if i%100==0:print('DIAGNOSTICS',i,len(primary),flush=True)
    pd.DataFrame(diag).to_csv(OUT/'calibrated_diagnostics_by_job.csv',index=False)
    d=pd.DataFrame(diag).groupby(keys+['method'],as_index=False)[['calibration_intercept','calibration_slope','ECE10']].mean()
    dd=d.groupby(['day','base_model','method']).agg(intercept_median=('calibration_intercept','median'),slope_median=('calibration_slope','median'),ECE10_median=('ECE10','median')).reset_index();dd.to_csv(OUT/'calibrated_diagnostics_summary.csv',index=False)
    d=pd.DataFrame(placebo);d.to_csv(OUT/'placebo_by_repeat.csv',index=False)
    # Average random repetitions within seed before averaging seeds within pair.
    valid=d[d.supported].groupby(keys+['seed'],as_index=False)[['temporal_gap_matched','random_within_presentation_gap','temporal_minus_random']].mean()
    ppm,pci=intervals(valid,'pair_id',['temporal_gap_matched','random_within_presentation_gap','temporal_minus_random']);ppm.to_csv(OUT/'placebo_pair_mean.csv',index=False);pci.to_csv(OUT/'placebo_intervals.csv',index=False)
    losses=pd.concat(loss_rows,ignore_index=True).groupby(keys+['username'],as_index=False).delta.mean()
    crossed=[]
    for (day,model),g in losses.groupby(['day','base_model']):
        g=g.reset_index(drop=True);ui,users=pd.factorize(g.username,sort=True);pi,pairs=pd.factorize(g.pair_id,sort=True)
        lookup=g.drop_duplicates('pair_id').set_index('pair_id').loc[pairs];ci,clusters=pd.factorize(lookup.provider+'|'+lookup.base_course_token,sort=True)
        delta=g.delta.to_numpy();counts=np.bincount(pi);point=np.mean(np.bincount(pi,weights=delta)/counts)
        expected=pm[(pm.day==day)&(pm.base_model==model)].A.mean();assert abs(point-expected)<1e-12
        rng=np.random.default_rng(20261011);boot=[]
        for _ in range(2000):
            uw=rng.exponential(size=len(users));cw=rng.exponential(size=len(clusters));w=uw[ui]
            pairvalue=np.bincount(pi,weights=w*delta)/np.bincount(pi,weights=w)
            boot.append(np.average(pairvalue,weights=cw[ci]))
        crossed.append({'day':day,'base_model':model,'A':point,'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'pairs':len(pairs),'base_courses':len(clusters),'users':len(users),'replicates':2000,'method':'independent exponential user and base-course weights; fixed predictions; normalize users within pair then course-weighted pair mean'})
    pd.DataFrame(crossed).to_csv(OUT/'crossed_user_course_intervals.csv',index=False)
    # Curves use outcome-independent largest target N, median target N, smallest target N at day 7 (LR, seed 17).
    elig=primary[(primary.day==7)&(primary.base_model=='logistic')&(primary.seed==17)].sort_values(['target_n','pair_id'])
    selections=elig.iloc[[0,len(elig)//2,len(elig)-1]];bins=[]
    for q in selections.itertuples():
        for model in ['logistic','lightgbm']:
            f=pd.read_parquet(p.WORK/'predictions'/f'{q.pair_id}_d7_{model}_s17.parquet');t=f[f.population=='target_primary']
            for col in ['raw','source_Platt','oof_Platt']:
                # Fixed probability bins, ties stay together; no confidence claim for graphical descriptions.
                idx=np.minimum((t[col].to_numpy()*10).astype(int),9)
                for b in range(10):
                    g=t.iloc[np.where(idx==b)[0]]
                    if len(g):bins.append({'pair_id':q.pair_id,'base_model':model,'target_n':len(t),'method':col,'bin':b,'bin_n':len(g),'mean_probability':g[col].mean(),'observed_rate':g.y.mean()})
    pd.DataFrame(bins).to_csv(OUT/'reliability_curve_bins.csv',index=False)
    print('DIAGNOSTICS_COMPLETE',len(primary),flush=True)

if __name__=='__main__':main()
