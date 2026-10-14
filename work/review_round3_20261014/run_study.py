"""Post-review full-pipeline experiments on a freshly prepared private runtime.

Requires the published portable repository's `reproduce.py prepare` first.
No old predictions/features are read. All learner-level files remain private.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--runtime',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--mode',choices=['baseline','recent','joint'],required=True)
args=ap.parse_args()
root=args.runtime.resolve(); out=args.output.resolve()/args.mode
private=root/'work/review_round3'/args.mode
os.environ['CALTRANSFER_ROOT']=str(root)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
sys.path.insert(0,str(root/'work/xuetangx_experiment4'))
import duckdb
import numpy as np
import pandas as pd
import primary as p
import review_extension as r
import conservative_updates as c

out.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
protocol_path=Path(__file__).with_name('protocol.json')
inputs=[Path(__file__),protocol_path,Path(p.__file__),Path(r.__file__),Path(c.__file__),root/'run_config.json',root/'input_audit.json']
fingerprint={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
fp=private/'fingerprint.json'
if fp.exists(): assert json.loads(fp.read_text())==fingerprint,'Changed code/protocol/runtime; use a fresh result cache'
else: fp.write_text(json.dumps(fingerprint,indent=2))
(out/'source_fingerprint.json').write_text(json.dumps(fingerprint,indent=2))
start=time.time()
r.OUT=out;r.WORK=private
pairs,e=r.initialize()
all_courses=list(e.course_id.unique())
original_rosters={key:set(g.username) for key,g in e.groupby('course_id')}

if args.mode=='recent':
    # Verify the feature-based membership predicate against a direct raw event query.
    con=duckdb.connect(str(root/'work/xuetangx_experiment4/events.duckdb'),read_only=True)
    checks=[]
    for day in r.DAYS:
        raw=set(con.execute('select distinct enroll_id from events where day_exact>? and day_exact<=?', [day-7,day]).fetchnumpy()['enroll_id'])
        f=p.DAY_FRAMES[day]
        derived=set(f.loc[f.last_activity_lag.gt(day-7),'enroll_id'])
        assert raw==derived, 'Recent activity feature predicate differs from raw events'
        checks.append({'day':day,'raw_recent_enrollments':len(raw),'predicate_symmetric_difference':len(raw^derived)})
    con.close()
    pd.DataFrame(checks).to_csv(out/'recent_membership_validation.csv',index=False)

if args.mode=='joint':
    cmap=pd.concat([pairs[['provider','base_course_token','source_course_id']].rename(columns={'source_course_id':'course_id'}),
                    pairs[['provider','base_course_token','target_course_id']].rename(columns={'target_course_id':'course_id'})]).drop_duplicates()
    cmap['base_course_key']=cmap.provider+'|'+cmap.base_course_token
    assert not cmap.course_id.duplicated().any()
    meta=pd.read_csv(root/'work/xuetangx_audit/enrollment_index.csv',usecols=['enroll_id','username','course_id','first_event'])
    eligible=meta.merge(cmap,on='course_id',validate='many_to_one')
    eligible['first_event']=pd.to_datetime(eligible.first_event,errors='raise')
    perbase=eligible.groupby(['username','base_course_key'],as_index=False).first_event.min()
    assignment=perbase.sort_values(['username','first_event','base_course_key'],kind='stable').drop_duplicates('username')
    retained=eligible.merge(assignment[['username','base_course_key']],on=['username','base_course_key'],validate='many_to_one')
    assert retained.groupby('username').base_course_key.nunique().max()==1
    assignment.to_parquet(private/'user_assignment.parquet',index=False)
    original=eligible.groupby(['provider','base_course_token','course_id']).size().rename('original_enrollments')
    kept=retained.groupby(['provider','base_course_token','course_id']).size().rename('retained_enrollments')
    counts=pd.concat([original,kept],axis=1).fillna(0).reset_index()
    counts.to_csv(out/'assignment_by_presentation.csv',index=False)
    audit={'eligible_pairs':len(pairs),'eligible_presentations':cmap.course_id.nunique(),'eligible_base_courses':cmap.base_course_key.nunique(),
           'eligible_users':eligible.username.nunique(),'cross_base_users_before':int((perbase.groupby('username').size()>1).sum()),
           'original_enrollments':len(eligible),'retained_enrollments':len(retained),'users_with_multiple_bases_after':0,
           'rule':'earliest observed first event; lexical base-course tie; no label input; applied before all fitting'}
    (out/'assignment_validation.json').write_text(json.dumps(audit,indent=2))
    e=e[e.enroll_id.isin(retained.enroll_id)].copy()

base_set_day=r.set_day
def set_day(enroll,day):
    base_set_day(enroll,day)
    # Preserve empty structurally eligible presentations for explicit support failures.
    empty=pd.DataFrame({'enroll_id':pd.Series(dtype='int64'),'username':pd.Series(dtype='int64'),'course_id':pd.Series(dtype=str),'y':pd.Series(dtype='int64')})
    for course in all_courses:p.BY_COURSE.setdefault(course,empty.copy())
r.set_day=set_day
original_prep=p.prep_pair
def restricted_prep(pair,day):
    source_course,target_course=pair.source_course_id,pair.target_course_id
    source=p.BY_COURSE[source_course].merge(p.DAY_BY_COURSE[day][source_course],on=['enroll_id','course_id'],how='inner',validate='one_to_one')
    target=p.BY_COURSE[target_course].merge(p.DAY_BY_COURSE[day][target_course],on=['enroll_id','course_id'],how='left',validate='one_to_one')
    target['unseen']=~target.username.isin(original_rosters[source_course])
    target['active']=target.event_count.notna()
    if args.mode=='recent':
        source=source[source.last_activity_lag.gt(day-7)].copy()
        target['active']=target.active & target.last_activity_lag.gt(day-7)
    for frame in (source,target):
        for col in p.NON_ACTION+p.ACTION_COUNT_COLS:
            frame[col]=frame[col].fillna(day if col in ('first_activity_lag','recency_to_landmark') else 0)
        for action,col in zip(p.ACTION_NAMES,p.ACTION_COUNT_COLS):
            frame[f'action_{action}_fraction']=np.where(frame.event_count.gt(0),frame[col]/frame.event_count,0)
    return source,target
if args.mode!='baseline':p.prep_pair=restricted_prep

print('START',args.mode,len(pairs),'pairs',len(e),'enrollments',flush=True)
r.future(pairs,e)
r.rollout(pairs,e)
c.PREV=out;c.OUT=out/'conservative';c.PRIVATE=private
c.fit();c.summarize()
completion={'mode':args.mode,'elapsed_seconds':time.time()-start,
            'pair_model_fits':len(pd.read_csv(out/'outcome_blind_future_by_job.csv')),
            'rollout_model_fits':len(pd.read_csv(out/'rollout_by_job.csv').drop_duplicates(c.KEYS)),
            'reused_old_fitted_predictions':False,'raw_to_prepared_data':str(root/'input_audit.json'),
            'protocol_version':json.loads(protocol_path.read_text())['version']}
(out/'completion.json').write_text(json.dumps(completion,indent=2))
print('COMPLETE',completion,flush=True)
