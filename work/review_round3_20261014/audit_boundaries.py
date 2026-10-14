"""Audit saved predictions against complete original user rosters and raw membership."""
import argparse,json
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
meta=pd.read_csv(a.runtime/'work/xuetangx_audit/enrollment_index.csv',usecols=['course_id','username','enroll_id'])
rosters={k:set(g.username) for k,g in meta.groupby('course_id')}
recent={}
for day in [7,14,28]:
    f=pd.read_parquet(a.runtime/f'work/xuetangx_experiment4/features_day{day}.parquet',columns=['enroll_id','last_activity_lag'])
    recent[day]=set(f.loc[f.last_activity_lag.gt(day-7),'enroll_id'])
checks=[]
for mode in ['baseline','recent','joint']:
    private=a.runtime/'work/review_round3'/mode
    pairs=pd.read_csv(a.output/mode/'outcome_blind_future_by_job.csv')
    triples=pd.read_csv(a.output/mode/'rollout_candidates.csv').set_index('triple_id')
    jobs=pd.read_csv(a.output/mode/'rollout_by_job.csv').drop_duplicates(['triple_id','day','seed','base_model'])
    assigned={}
    if mode=='joint':
        assigned=pd.read_parquet(private/'user_assignment.parquet').set_index('username').base_course_key.to_dict()
    for row in pairs.itertuples():
        f=pd.read_parquet(private/'future_predictions'/f'{row.pair_id}_d{row.day}_{row.base_model}_s{row.seed}.parquet')
        cal=f[f.population.eq('source_calibration')];target=f[f.population.eq('target_primary')]
        assert len(cal)==row.source_cal_n and len(target)==row.target_n
        assert not set(target.username)&rosters[row.source_course_id]
        assert not set(target.username)&set(cal.username)
        if mode=='recent':assert set(cal.enroll_id)|set(target.enroll_id)<=recent[row.day]
        if mode=='joint':
            expected=row.provider+'|'+row.base_course_token
            assert all(assigned[u]==expected for u in set(f.username))
    for row in jobs.itertuples():
        q=triples.loc[row.triple_id]
        t2=pd.read_parquet(private/'rollout_predictions'/f'{row.triple_id}_d{row.day}_{row.base_model}_s{row.seed}.parquet')
        earlier=rosters[q.source_course_id_a]|rosters[q.target_course_id_a]
        assert len(t2)==row.t2_n
        assert not set(t2.username)&earlier
        if mode=='recent':assert set(t2.enroll_id)<=recent[row.day]
        if mode=='joint':
            expected=q.provider_a+'|'+q.base_course_token_a
            assert all(assigned[u]==expected for u in set(t2.username))
    checks.append({'mode':mode,'pair_prediction_files_checked':len(pairs),'rollout_prediction_files_checked':len(jobs),
                   'earlier_presentation_user_overlap':0,'recent_membership_violations':0 if mode=='recent' else None,
                   'assigned_base_violations':0 if mode=='joint' else None})
    print('BOUNDARY_AUDIT',mode,len(pairs),len(jobs),flush=True)
(a.output/'identity_and_riskset_validation.json').write_text(json.dumps({'checks':checks,'scope':'Saved cal/T1/T2 predictions checked against ALL original earlier-presentation users, not just fitted roles'},indent=2))
