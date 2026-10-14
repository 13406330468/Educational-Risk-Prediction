"""Post-review conservative updating on frozen S -> T1 -> T2 predictions.

fit requires the private caches produced by review_extension.py future/rollout.
summarize uses only aggregate scores and can run in quick mode without raw data.
No T2 outcome is accepted by any parameter-fitting function.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

ROOT = Path(os.environ['CALTRANSFER_ROOT'])
PREV = ROOT / 'outputs/review_quality_20261011'
OUT = ROOT / 'outputs/review_round2_20261011'
PRIVATE = ROOT / 'work/xuetangx_experiment4/review_extension'
FRACTIONS = (.1, .2, .5, 1.)
METHODS = ('full', 'intercept', 'shrink')
KEYS = ['triple_id', 'day', 'seed', 'base_model']
GROUP = ['day', 'base_model', 'fraction', 'strategy']
BOOTSTRAPS = 100000
BOOT_SEED = 20261012
PROTOCOL = {
    'status': 'second post-review exploratory extension; not preregistered',
    'frozen_before_added_strategy_evaluation': True,
    'population': 'same supported triples, seeds, frozen source predictor and T2 learners as the preceding rollout',
    'budgets': list(FRACTIONS),
    'full': 'existing T1 full Platt update, retained as reference',
    'intercept': 'hold historical slope; minimize T1 log loss over intercept b in [-30,30] with 1e-5*(b-bS)^2 penalty',
    'shrink': '(1-w)*thetaS+w*thetaT1, w=nT1/(nT1+nS_cal); convex combination of slope and intercept; count-weighted heuristic, not fitted precision',
    'selection': 'no tuning of strategy, weight or budget using T2 labels; no winning strategy selected',
    'support': 'retain original N>=10/two-class update support for all three strategies',
    'matched': 'full Platt S versus full Platt T1, m=min(nS_cal,nT1_budget), independent nested unstratified S permutation and original T1 permutation; each m>=10 and two classes; no redraw',
    'aggregation': 'average seeds within triple, then equal weight to triples; resample provider/base-course clusters',
    'bootstrap_draws': BOOTSTRAPS,
    'bootstrap_seed': BOOT_SEED,
    'multiplicity': '6-test original full-budget family; separate 96-contrast extension sensitivity (3 strategies x 3 days x 2 models x 4 budgets, plus 24 matched-budget contrasts); percentile tails alpha/(2K)',
    'uncertainty': 'Bonferroni-expanded bootstrap sensitivity, approximate coverage with 23-24 clusters; no full-fitting or shared-user resampling',
    'same_budget_limits': 'matched analysis isolates label count, not unmeasured presentation differences; strategy comparison population differs if matched support fails'
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'conservative_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == PROTOCOL, 'Protocol changed; use a fresh runtime'
    else:
        path.write_text(json.dumps(PROTOCOL, indent=2))

def alternative_params(pc, yc, p1, y1, fit_cal):
    """Inputs include S calibration and T1 only, never T2 probabilities/outcomes."""
    hist = fit_cal(pc, yc, 'Platt')
    full = fit_cal(p1, y1, 'Platt')
    z = logit(np.clip(p1, 1e-6, 1-1e-6))
    yy = np.asarray(y1, dtype=float)
    a, b0 = hist
    def objective(b):
        eta = a*z+b
        return np.mean(np.logaddexp(0, eta)-yy*eta)+1e-5*(b-b0)**2
    opt = minimize_scalar(objective, bounds=(-30, 30), method='bounded', options={'xatol': 1e-9})
    assert opt.success and np.isfinite(opt.fun)
    intercept = np.array([a, opt.x])
    weight = len(y1)/(len(y1)+len(yc))
    shrink = (1-weight)*hist + weight*full
    return hist, {'full': full, 'intercept': intercept, 'shrink': shrink}, weight

def fit():
    freeze_protocol()
    import primary as p
    old = pd.read_csv(PREV / 'rollout_by_job.csv')
    support = pd.read_csv(PREV / 'rollout_support.csv')
    triples = pd.read_csv(PREV / 'rollout_candidates.csv').set_index('triple_id')
    expected = {(r.triple_id, r.day, r.seed, m) for r in support[support.admissible].itertuples() for m in ('logistic','lightgbm')}
    actual = set(old[KEYS].itertuples(index=False, name=None))
    assert expected == actual and len(expected) == len(old)/4
    rows, matched, checks, input_hashes = [], [], [], []
    deviations = {'historical_predictions': 0., 'full_update_predictions': 0., 'full_update_Brier': 0.}
    for j, (key, reference) in enumerate(old.groupby(KEYS, sort=True)):
        triple, day, seed, model = key
        q = triples.loc[triple]
        stem = f'{triple}_d{day}_{model}_s{seed}'
        srcfile = PRIVATE/'future_predictions'/f'{q.pair_id_a}_d{day}_{model}_s{seed}.parquet'
        testfile = PRIVATE/'rollout_predictions'/f'{stem}.parquet'
        f = pd.read_parquet(srcfile); t2 = pd.read_parquet(testfile)
        cal = f[f.population.eq('source_calibration')].reset_index(drop=True)
        t1 = f[f.population.eq('target_primary')].reset_index(drop=True)
        assert len(cal)==reference.source_cal_n.iloc[0] and len(t1)==reference.t1_n.iloc[0] and len(t2)==reference.t2_n.iloc[0]
        assert cal.enroll_id.is_unique and t1.enroll_id.is_unique and t2.enroll_id.is_unique
        # S-cal/T1-cache compatibility with the independently saved rollout is
        # checked by reproducing every historical and full-update prediction.
        pc, p1, p2 = cal.raw.to_numpy(), t1.raw.to_numpy(), t2.raw.to_numpy()
        yc, y1, y2 = cal.y.to_numpy(), t1.y.to_numpy(), t2.y.to_numpy()
        hist = p.fit_cal(pc, yc, 'Platt')
        hp = p.apply_cal(p2, hist)
        dh = float(np.max(abs(hp-t2.historical.to_numpy())))
        assert dh < 1e-10
        deviations['historical_predictions'] = max(deviations['historical_predictions'], dh)
        bs_hist = p.brier(y2, hp)
        assert abs(bs_hist-reference.historical_Brier.iloc[0]) < 1e-12
        assert not set(cal.username)&set(t1.username)
        assert not set(cal.username)&set(t2.username) and not set(t1.username)&set(t2.username)
        perm = np.random.default_rng(20261011+seed+int(triple[1:])*1009+day).permutation(len(t1))
        sper = np.random.default_rng(2026101100+seed+int(triple[1:])*1009+day).permutation(len(cal))
        base = dict(zip(KEYS,key)) | {'provider':q.provider_a, 'base_course_token':q.base_course_token_a,
              'source_cal_n':len(cal),'t1_n':len(t1),'t2_n':len(t2),'historical_Brier':bs_hist}
        for frac in FRACTIONS:
            idx = perm[:max(1,int(np.floor(frac*len(t1))))]
            supported = len(idx)>=10 and len(np.unique(y1[idx]))==2
            oldrow = reference[reference.fraction.eq(frac)].iloc[0]
            assert supported == bool(oldrow.budget_supported)
            params, w = {}, np.nan
            if supported:
                _, params, w = alternative_params(pc, yc, p1[idx], y1[idx], p.fit_cal)
                full_pred = p.apply_cal(p2, params['full'])
                dp = float(np.max(abs(full_pred-t2[f'updated_{frac}'].to_numpy())))
                db = abs(p.brier(y2,full_pred)-oldrow.updated_Brier)
                assert dp < 1e-10 and db < 1e-12
                deviations['full_update_predictions'] = max(deviations['full_update_predictions'], dp)
                deviations['full_update_Brier'] = max(deviations['full_update_Brier'], db)
            for method in METHODS:
                row = base | {'fraction':frac,'strategy':method,'labels':len(idx),'supported':supported,'shrink_weight':w}
                if supported:
                    prediction = p.apply_cal(p2,params[method])
                    bs = p.brier(y2,prediction)
                    row.update({'updated_Brier':bs, 'advantage':bs_hist-bs,
                                'slope':params[method][0], 'intercept':params[method][1]})
                rows.append(row)
            m = min(len(cal),len(idx)); si,ti = sper[:m],perm[:m]
            ok = m>=10 and len(np.unique(yc[si]))==2 and len(np.unique(y1[ti]))==2
            row = base | {'fraction':frac,'strategy':'matched','labels_each':m,'supported':ok}
            if ok:
                historical = p.apply_cal(p2,p.fit_cal(pc[si],yc[si],'Platt'))
                recent = p.apply_cal(p2,p.fit_cal(p1[ti],y1[ti],'Platt'))
                sbs, tbs = p.brier(y2,historical), p.brier(y2,recent)
                row.update({'matched_source_Brier':sbs,'matched_recent_Brier':tbs,'advantage':sbs-tbs})
            matched.append(row)
        checks.append(base | {'historical_max_prediction_error':dh,'earlier_test_role_overlap':0})
        input_hashes.append({'job':stem,'source_cache_sha256':sha(srcfile),'test_cache_sha256':sha(testfile)})
        if (j+1)%100==0: print('CONSERVATIVE',j+1,'/',len(expected),flush=True)
    d, m = pd.DataFrame(rows), pd.DataFrame(matched)
    assert len(d)==len(expected)*12 and len(m)==len(expected)*4
    assert not d[KEYS+['fraction','strategy']].duplicated().any()
    assert not m[KEYS+['fraction']].duplicated().any()
    d.to_csv(OUT/'conservative_by_job.csv',index=False); m.to_csv(OUT/'matched_updates_by_job.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'update_cache_checks.csv',index=False)
    audit = {'jobs':len(expected),'strategy_evaluations':len(d),'matched_evaluations':len(m),
       'supported_strategy_evaluations':int(d.supported.sum()),'supported_matched_evaluations':int(m.supported.sum()),
       'compatibility_max_errors':deviations,'training_scope':'Reused frozen base-model predictions; fitted new calibrators. No new base-model fits.',
       'T2_fitting_access':'no T2 labels in fitting function inputs; same T2 records for every within-job strategy',
       'script_sha256':sha(__file__),'primary_sha256':sha(p.__file__),'input_caches':input_hashes}
    (OUT/'conservative_validation.json').write_text(json.dumps(audit,indent=2))
    print({k:v for k,v in audit.items() if k!='input_caches'},flush=True)

def summarize():
    freeze_protocol()
    d = pd.read_csv(OUT/'conservative_by_job.csv')
    m = pd.read_csv(OUT/'matched_updates_by_job.csv')
    dv,mv = d[d.supported],m[m.supported]
    assert np.isfinite(dv[['historical_Brier','updated_Brier','advantage']]).all().all()
    assert np.allclose(dv.advantage,dv.historical_Brier-dv.updated_Brier,atol=1e-12)
    assert np.allclose(mv.advantage,mv.matched_source_Brier-mv.matched_recent_Brier,atol=1e-12)
    pm = pd.concat([dv,mv]).groupby(GROUP+['provider','base_course_token','triple_id'],as_index=False).advantage.mean()
    pm.to_csv(OUT/'update_triple_means.csv',index=False)
    rows=[]
    for key,g in pm.groupby(GROUP,sort=True):
        cl = g.groupby(['provider','base_course_token']).advantage.agg(['sum','size'])
        rng=np.random.default_rng(BOOT_SEED)
        # Multinomial course weights are exactly resampling whole base-course
        # clusters with replacement; triple counts provide the macro denominator.
        weights=rng.multinomial(len(cl),np.repeat(1/len(cl),len(cl)),size=BOOTSTRAPS)
        boot=(weights@cl['sum'].to_numpy())/(weights@cl['size'].to_numpy())
        point=np.quantile(boot,[.025,.975]); adjusted=np.quantile(boot,[.05/(2*96),1-.05/(2*96)])
        row=dict(zip(GROUP,key))|{'mean':g.advantage.mean(),'point_low':point[0],'point_high':point[1],
             'family96_low':adjusted[0],'family96_high':adjusted[1],'triples':len(g),'clusters':len(cl),'bootstrap_draws':BOOTSTRAPS}
        if key[2]==1. and key[3]=='full':
            focused=np.quantile(boot,[.05/(2*6),1-.05/(2*6)])
            row.update({'family6_low':focused[0],'family6_high':focused[1]})
        rows.append(row)
    result=pd.DataFrame(rows)
    assert len(result)==96
    result.to_csv(OUT/'update_intervals.csv',index=False)
    full=result[(result.fraction==1)&(result.strategy=='full')]
    full.to_csv(OUT/'multiplicity_six_full_budget.csv',index=False)
    support = pd.concat([d,m]).groupby(GROUP,as_index=False).agg(attempted=('supported','size'),supported=('supported','sum'))
    support.to_csv(OUT/'update_support_summary.csv',index=False)
    print(result[result.fraction.eq(1)][['day','base_model','strategy','mean','point_low','point_high','family96_low','family96_high']].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fit','summarize']);args=ap.parse_args()
    (fit if args.mode=='fit' else summarize)()
