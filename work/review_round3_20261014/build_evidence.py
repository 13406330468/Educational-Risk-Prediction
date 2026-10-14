"""Single numeric source for the R3 manuscript: frozen, aggregate CSVs only."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--paper',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);a=p.parse_args()
E=a.evidence;P=a.paper;T=P/'review_tables';T.mkdir(parents=True,exist_ok=True)
models={'logistic':'LR','lightgbm':'LightGBM'};modes={'baseline':'Baseline','recent':'Recent-active','joint':'Joint-assigned'}
frames={};sources={}
for path in sorted((E/'summary').glob('*.csv')):
    rel=path.relative_to(E).as_posix();frames[rel]=pd.read_csv(path);sources[rel]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':json.loads(frames[rel].to_json(orient='records',double_precision=15))}
for mode in modes:
    rel=f'{mode}/conservative/update_intervals.csv';frames[rel]=pd.read_csv(E/rel);sources[rel]={'sha256':hashlib.sha256((E/rel).read_bytes()).hexdigest(),'records':json.loads(frames[rel].to_json(orient='records',double_precision=15))}
pair=frames['summary/pair_intervals.csv'];absolute=frames['summary/full_budget_absolute_table.csv'];support=frames['summary/support_summary.csv'];signal=frames['summary/monte_carlo_signal_check.csv']
assert len(pair.query("metric == 'A'"))==18
assert (pair.query("metric == 'A' and mode != 'baseline'").family_low>0).all()
assert ((absolute.full_U_family6_low<0)&(absolute.full_U_family6_high>0)).all()
def order(f):
    return f.assign(_mode=f['mode'].map({k:i for i,k in enumerate(modes)}) if 'mode' in f else 0,_model=f.base_model.map({'logistic':0,'lightgbm':1}) if 'base_model' in f else 0).sort_values([c for c in ['_mode','day','_model','fraction'] if c in f or c.startswith('_')])
def f(v):return f'{v:.4f}'
def interval(lo,hi):return f'[{f(lo)}, {f(hi)}]'
def table(name,caption,label,columns,heads,rows,size='footnotesize'):
    s='\\begin{table*}[!t]\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\centering\\'+size+'\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
    s+=' & '.join(heads)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    (T/(name+'.tex')).write_text(s,encoding='utf-8')

rows=[]
for day in (7,14,28):
    for model in models:
        row=[day,models[model]]
        for mode in ('recent','joint'):
            v=pair[(pair['mode']==mode)&(pair.day==day)&(pair.base_model==model)&(pair.metric=='A')].iloc[0]
            row += [f'{int(v.units)}/{int(v.clusters)}',f(v['mean'])+' '+interval(v.family_low,v.family_high)]
        rows.append(row)
table('population_main',r'Additional time-defined outcome populations (RQ2). P/C denotes supported pairs/base-course clusters; A intervals use 100,000 cluster resamples and a single 12-comparison family spanning the two added populations, three days and two models. Recent-active Day 7 repeats the baseline on supported jobs and is not independent confirmation. Joint assignment covers the eligible structural course pool only.', 'tab:population','llrlrl',['Day','Model','Recent P/C',r'Recent-active \(A\) [adjusted interval]','Joint P/C',r'Joint-assigned \(A\) [adjusted interval]'],rows)
rows=[]
for r in order(absolute[absolute['mode']=='baseline']).itertuples():
    rows.append([r.day,models[r.base_model],f'{r.triples}/{r.clusters}',f(r.t2_prevalence),f(r.raw_Brier),f(r.historical_Brier),f(r.full_Brier),f(r.intercept_Brier),f(r.shrink_Brier)])
table('future_rollout_main',r'Baseline three-presentation absolute T2 performance (RQ3), using the full T1 label budget. Values average seeds within sequence and sequences equally. N/C denotes sequences/base-course clusters. T2 event rate is descriptive and never used for updating. All six full-Platt U intervals cross zero under the six-comparison sensitivity (100,000 resamples); Supplementary Tables S25--S26 report all populations and uncertainty.', 'tab:future_rollout','llrrrrrrr',['Day','Model','N/C','Event rate','Raw BS','Historical BS','Full BS','Intercept BS','Shrink BS'],rows)
rows=[]
for r in order(support).itertuples():
    rows.append([modes[r.mode],r.day,f'{r.supported_pairs}/{r.pair_clusters}',r.pair_model_fits,f'{r.supported_triples}/{r.triple_clusters}',r.rollout_model_fits,f'{r.supported_updates}/{r.attempted_updates}'])
table('r3_support',r'R3 full-pipeline support by population and landmark. Pairs and sequences use the explicitly timed inactivity outcome. Update evaluations count three strategies and four budgets, not independent cohorts. Day 7 recent-active jobs duplicate baseline support.', 'tab:r3support','llrrrrr',['Population','Day','Pairs/clusters','Pair fits','Sequences/clusters','Rollout fits','Supported/attempted updates'],rows)
rows=[]
for r in order(pair[(pair['mode']!='baseline')&(pair.metric=='A')]).itertuples():
    g=pair[(pair['mode']==r.mode)&(pair.day==r.day)&(pair.base_model==r.base_model)].set_index('metric')['mean']
    rows.append([modes[r.mode],r.day,models[r.base_model],f'{r.units}/{r.clusters}',f(g.G_S),f(g.G_T),f(g.G_T_star),f(r.mean),interval(r.family_low,r.family_high)])
table('r3_pairs',r'Full gain decomposition for the two added populations; 100,000 course-cluster resamples. The 12-comparison adjustment applies only to A across these rows; support conditions and population overlap preclude interpreting the rows as independent replications.', 'tab:r3pairs','lllrrrrrl',['Population','Day','Model','Pairs/C',r'\(G_S\)',r'\(G_T\)',r'\(G_T^*\)',r'\(A\)','Adjusted interval'],rows)
rows=[]
for r in order(absolute).itertuples():
    rows.append([modes[r.mode],r.day,models[r.base_model],f'{r.triples}/{r.clusters}',f(r.t2_prevalence),f(r.raw_Brier),f(r.historical_Brier),f(r.recent_null_Brier),f(r.full_Brier),f(r.intercept_Brier),f(r.shrink_Brier)])
table('r3_absolute',r'All 18 full-budget absolute T2 performance conditions. Equal sequence weights follow within-sequence seed averaging. The constant uses T1 labels; T2 prevalence is only descriptive. Pointwise 100,000-draw intervals for every absolute score are supplied in the machine-readable registry, with paired U intervals in Table S26.', 'tab:r3absolute','lllrrrrrrrr',['Population','Day','Model','N/C','T2 rate','Raw','Historical','T1 constant','Full','Intercept','Shrink'],rows)
rows=[]
for r in order(absolute).itertuples():
    rows.append([modes[r.mode],r.day,models[r.base_model],r.clusters,f(r.full_U),interval(r.full_U_low,r.full_U_high),interval(r.full_U_family6_low,r.full_U_family6_high)])
table('r3_uncertainty',r'Full-budget full-Platt future update advantage U. Intervals use 100,000 base-course resamples. Six-comparison adjustments are applied separately within each population, not jointly to all 18 rows or all paper analyses. Every adjusted interval crosses zero.', 'tab:r3uncertainty','lllrrll',['Population','Day','Model','C',r'\(U\)','Pointwise 95\% interval','Population-wise six-comparison interval'],rows)
rows=[]
for r in signal.itertuples():
    u=frames[f'{r.mode}/conservative/update_intervals.csv'];v=u[(u.day==r.day)&(u.base_model==r.base_model)&(u.fraction==r.fraction)&(u.strategy==r.strategy)].iloc[0]
    # Column names from frozen uncertainty table are explicitly preserved in registry.
    c=int(v['base_course_clusters'] if 'base_course_clusters' in v else v['clusters'])
    rows.append([modes[r.mode],r.day,models[r.base_model],f'{int(r.fraction*100)}\%',r.strategy,c,f(r.mean_U),interval(r.original_100k_family96_low,r.original_100k_family96_high),interval(r.one_million_family96_low,r.one_million_family96_high)])
table('r3_signals',r'Four exploratory tail signals. The 96-comparison family is defined separately within each population; these are not a global correction across all analyses. Primary endpoints use 100,000 draws. One million draws only check Monte Carlo stability of the four selected tails. Small cluster counts and near-zero lower bounds limit positive-strategy claims.', 'tab:r3signals','lllrlrlll',['Population','Day','Model','Budget','Strategy','C',r'\(U\)','100k adjusted interval','1M stability interval'],rows)
impact=frames['summary/course_influence_summary.csv'];rows=[]
for r in order(impact[(impact.fraction==1)&(impact.strategy=='full')]).itertuples():
    rows.append([modes[r.mode],r.day,models[r.base_model],r.clusters,f(r.mean_U),f'{100*r.positive_course_fraction:.1f}\%',interval(r.course_U_min,r.course_U_max),interval(r.loco_U_min,r.loco_U_max)])
table('r3_influence',r'Full-budget full-Platt course heterogeneity and leave-one-course-out (LOCO) means. Within-course means are used only for the distribution; overall U remains sequence weighted. LOCO ranges are influence summaries, not confidence intervals or post-hoc exclusion rules.', 'tab:r3influence','lllrrrll',['Population','Day','Model','C',r'Overall \(U\)','Positive courses',r'Course \(U\) range','LOCO mean range'],rows)

# Replace former future-outcome displays with the single final 100k source.
rows=[]
for r in order(pair[(pair['mode']=='baseline')&(pair.metric=='A')]).itertuples():
    g=pair[(pair['mode']=='baseline')&(pair.day==r.day)&(pair.base_model==r.base_model)].set_index('metric')['mean']
    rows.append([r.day,models[r.base_model],f'{r.units}/{r.clusters}',f(g.G_S),f(g.G_T),f(g.G_T_star),f(r.mean),interval(r.point_low,r.point_high)])
table('future_supp',r'Baseline structural-pool inactivity gains, with 100,000-draw pointwise course-cluster intervals for A. P/C denotes supported pairs/clusters. These intervals are not the 12-comparison expansion for the two added populations.', 'tab:s15','llrrrrrl',['Day','Model','P/C',r'\(G_S\)',r'\(G_T\)',r'\(G_T^*\)',r'\(A\)','Pointwise interval'],rows)
ui=frames['baseline/conservative/update_intervals.csv']
def update_cell(day,model,strategy,frac):
    v=ui[(ui.day==day)&(ui.base_model==model)&(ui.strategy==strategy)&(ui.fraction==frac)].iloc[0]
    return f(v['mean'])+' '+interval(v.point_low,v.point_high)
rows=[]
for day in (7,14,28):
    for model in models:
        v=ui[(ui.day==day)&(ui.base_model==model)&(ui.strategy=='full')].sort_values('fraction')
        rows.append([day,models[model],'/'.join(str(int(x)) for x in v.clusters),*[update_cell(day,model,'full',b) for b in (.1,.2,.5,1)]])
table('rollout_supp',r'Baseline third-presentation full-Platt advantage on each budget\textquotesingle s actual support. All pointwise intervals use 100,000 course resamples. C lists clusters at 10/20/50/100 percent; Day 7 sequences are 29/30/30/30, Days 14 and 28 are 32 at every budget. Figure 5 instead uses common support.', 'tab:s16','llrllll',['Day','Model','C by budget',r'10\%',r'20\%',r'50\%',r'100\%'],rows)
rows=[]
for day in (7,14,28):
    for model in models:
        for strategy in ('full','intercept','shrink'):
            rows.append([day,models[model],strategy,*[update_cell(day,model,strategy,b) for b in (.1,.2,.5,1)]])
table('conservative_supp',r'Baseline update advantage: mean [pointwise 95\% interval] from 100,000 cluster resamples. All three strategies share actual per-budget support (Table S16). All 72 baseline strategy contrasts have zero-crossing intervals under the baseline 96-comparison family; this statement does not apply to the added populations.', 'tab:s20','lllrrrr',['Day','Model','Strategy',r'10\%',r'20\%',r'50\%',r'100\%'],rows)
matched=[x for x in ui.strategy.unique() if x not in ('full','intercept','shrink')];assert len(matched)==1
rows=[]
for day in (7,14,28):
    for model in models:rows.append([day,models[model],*[update_cell(day,model,matched[0],b) for b in (.1,.2,.5,1)]])
table('matched_updates_supp',r'Baseline equal-label-budget S/T1 recency comparison: matched-S Brier minus matched-T1 Brier, with 100,000-draw pointwise intervals. Support follows Table S20 except one extra source split per model fails at Day 28, 10\%. All full-budget pointwise and all 24 baseline 96-family intervals cross zero.', 'tab:s21','llrrrr',['Day','Model',r'10\%',r'20\%',r'50\%',r'100\%'],rows)
rows=[]
for day in (7,14,28):
    for model in models:
        v=ui[(ui.day==day)&(ui.base_model==model)&(ui.strategy=='full')&(ui.fraction==1)].iloc[0]
        rows.append([day,models[model],int(v.clusters),f(v['mean']),interval(v.point_low,v.point_high),interval(v.family6_low,v.family6_high),interval(v.family96_low,v.family96_high)])
table('multiplicity_supp',r'Baseline full-budget full-Platt U under pointwise, six-comparison and 96-comparison percentile expansions, all from 100,000 course resamples. Families concern this baseline population only. Table S26 extends the six-comparison display to the other populations.', 'tab:s22','llrrlll',['Day','Model','C',r'\(U\)','Pointwise','Six-comparison','96-comparison'],rows)

# Current main budget figure: common support across ALL four budgets and ALL three strategies.
u=pd.read_csv(E/'baseline/conservative/conservative_by_job.csv');keys=['triple_id','day','seed','base_model'];cl=['provider','base_course_token'];v=u[u.supported].copy()
counts=v.groupby(keys).size();good=counts[counts==12].reset_index()[keys];v=v.merge(good,on=keys,validate='many_to_one')
fig,axs=plt.subplots(2,3,figsize=(10.2,5.4),sharey='row');summ=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
for i,model in enumerate(models):
    for j,day in enumerate((7,14,28)):
        ax=axs[i,j];ax.axhline(0,color='#a3aab3',ls='--',lw=.8)
        for strategy,color,marker,off in [('full','#2467A9','o',-.08),('intercept','#D86808','s',0),('shrink','#64748B','^',.08)]:
            vals=[]
            for frac in (.1,.2,.5,1.):
                g=v[(v.base_model==model)&(v.day==day)&(v.strategy==strategy)&(v.fraction==frac)].groupby(cl+['triple_id'],as_index=False).advantage.mean()
                s=g.groupby(cl).advantage.sum();n=g.groupby(cl).size();c=len(s);w=np.random.default_rng(20261012).multinomial(c,np.repeat(1/c,c),size=100000)
                boot=(w@s.to_numpy())/(w@n.to_numpy());lo,hi=np.quantile(boot,[.025,.975]);mean=g.advantage.mean();vals.append([mean,lo,hi])
                summ.append(dict(day=day,base_model=model,strategy=strategy,fraction=frac,mean=mean,point_low=lo,point_high=hi,triples=len(g),clusters=c,supported_jobs=len(v[(v.base_model==model)&(v.day==day)&(v.strategy==strategy)&(v.fraction==frac)])))
            z=np.array(vals);ax.errorbar(np.arange(4)+off,z[:,0],yerr=np.array([z[:,0]-z[:,1],z[:,2]-z[:,0]]),color=color,fmt=marker+'-',ms=3.5,lw=.9,capsize=2,label={'full':'Full Platt','intercept':'Historical-slope intercept','shrink':'Count-weighted shrinkage'}[strategy])
        ax.set_title(f'{models[model]} | Day {day}');ax.set_xticks(range(4),['10','20','50','100']);ax.grid(axis='y',alpha=.16)
        if i==1:ax.set_xlabel('T1 label budget (%)')
        if j==0:ax.set_ylabel('Brier update advantage U')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015));fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(P/'figures/figure5_rollout.pdf',bbox_inches='tight');fig.savefig(P/'figures/figure5_rollout.png',dpi=180,bbox_inches='tight');plt.close(fig)
derived=pd.DataFrame(summ);derived.to_csv(a.registry.parent/'common_budget_100k.csv',index=False)
registry={'version':'2026.10.14-r3','numerical_policy':{'new_bootstrap_draws':100000,'seed':20261012,'historical_archival_draws':2000,'new_A_family':12,'U_full_family_per_population':6,'U_extended_family_per_population':96,'conditional_on_fitted_models':True,'independent_replication':False},'sources':sources,'derived_common_budget_panel':summ,'checks':{'new_A_adjusted_positive':12,'full_budget_U_adjusted_crossing_zero':18,'total_base_model_fits':7610}}
a.registry.write_text(json.dumps(registry,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'registry':str(a.registry),'tables':10,'main_budget_panel_rows':len(derived),'main_budget_support':derived[['day','triples','clusters']].drop_duplicates().to_dict('records')}))
