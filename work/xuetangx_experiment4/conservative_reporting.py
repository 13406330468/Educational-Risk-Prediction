"""Aggregate-only conservative-update plot, table inputs and descriptive contrasts."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(os.environ['CALTRANSFER_ROOT'])
OUT=ROOT/'outputs/review_round2_20261011'
KEYS=['triple_id','day','seed','base_model']
LABELS={'logistic':'LR','lightgbm':'LightGBM','full':'Full Platt','intercept':'Intercept','shrink':'Shrinkage'}

def table(caption,label,columns,rows,width):
    return '\n'.join([r'\begin{table*}[!t]',r'\caption{'+caption+'}',r'\label{'+label+'}',
      r'\centering\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}',
      r'\begin{tabular}{'+width+'}',r'\toprule',' & '.join(columns)+r' \\',r'\midrule',
      *(' & '.join(row)+r' \\' for row in rows),r'\bottomrule\end{tabular}\end{table*}'])+'\n'

def ci(row,lo='point_low',hi='point_high'):
    return f"{row['mean']:.4f} [{row[lo]:.4f}, {row[hi]:.4f}]"

def main():
    d=pd.read_csv(OUT/'conservative_by_job.csv'); intervals=pd.read_csv(OUT/'update_intervals.csv')
    good=d.groupby(KEYS).supported.all(); common=d.merge(good[good].reset_index()[KEYS],on=KEYS)
    pm=common.groupby(['day','base_model','strategy','fraction','provider','base_course_token','triple_id'],as_index=False).advantage.mean()
    rows=[]
    for (day,model,strategy),g in pm.groupby(['day','base_model','strategy']):
        wide=g.pivot(index=['provider','base_course_token','triple_id'],columns='fraction',values='advantage').reset_index()
        clusters=list(wide.groupby(['provider','base_course_token']).indices.values());a=wide[[.1,.2,.5,1.]].to_numpy()
        rng=np.random.default_rng(20261011)
        boot=np.array([a[np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))])].mean(axis=0) for _ in range(2000)])
        for j,f in enumerate([.1,.2,.5,1.]):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);rows.append({'day':day,'base_model':model,'strategy':strategy,'fraction':f,'mean':a[:,j].mean(),'ci_low':lo,'ci_high':hi,'triples':len(wide),'clusters':len(clusters)})
    curve=pd.DataFrame(rows);curve.to_csv(OUT/'conservative_common_budget_intervals.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.2),sharex=True,sharey=True)
    for r,model in enumerate(['logistic','lightgbm']):
        for c,day in enumerate([7,14,28]):
            ax=axs[r,c];ax.axhline(0,color='#A5ADB6',ls='--',lw=.8)
            for method,color,marker,offset in [('full','#2467A9','o',-.12),('intercept','#D86808','s',0),('shrink','#64748B','^',.12)]:
                g=curve[(curve.day==day)&(curve.base_model==model)&(curve.strategy==method)].sort_values('fraction');y=g['mean'].to_numpy()
                ax.errorbar(np.arange(4)+offset,y,yerr=[y-g.ci_low,g.ci_high-y],color=color,marker=marker,lw=.9,ms=3,capsize=2,label=LABELS[method])
            ax.set_title(LABELS[model]+f' | Day {day}',fontweight='bold',fontsize=9)
            ax.set_xticks(range(4),['10','20','50','100']);ax.grid(axis='y',color='#E8EDF2',lw=.5);ax.set_axisbelow(True)
            if c==0:ax.set_ylabel('Brier advantage over S map')
            if r==1:ax.set_xlabel('T1 labels used (%)')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1),w_pad=.8,h_pad=.8)
    fig.savefig(OUT/'figure5_conservative_updates.pdf',bbox_inches='tight');fig.savefig(OUT/'figure5_conservative_updates.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # All point means are from per-budget support, not the common-budget plot.
    summary=[];supp=[];matched=[];mult=[]
    for day in [7,14,28]:
        for model in ['logistic','lightgbm']:
            g=intervals[(intervals.day==day)&(intervals.base_model==model)]; full=g[(g.strategy=='full')&(g.fraction==1)].iloc[0]
            other=lambda method:g[(g.strategy==method)&(g.fraction==1)].iloc[0]
            summary.append([str(day),LABELS[model],str(full.clusters),ci(full,'family6_low','family6_high'),f"{other('intercept')['mean']:.4f}",f"{other('shrink')['mean']:.4f}",f"{other('matched')['mean']:.4f}"])
            for method in ['full','intercept','shrink']:
                z=g[g.strategy==method].set_index('fraction')
                supp.append([str(day),LABELS[model],LABELS[method]]+[ci(z.loc[f]) for f in [.1,.2,.5,1.]])
            z=g[g.strategy=='matched'].set_index('fraction');matched.append([str(day),LABELS[model]]+[ci(z.loc[f]) for f in [.1,.2,.5,1.]])
            mult.append([str(day),LABELS[model],f"{full['mean']:.4f}",f'[{full.point_low:.4f}, {full.point_high:.4f}]',f'[{full.family6_low:.4f}, {full.family6_high:.4f}]',f'[{full.family96_low:.4f}, {full.family96_high:.4f}]'])
    tables={
      'conservative_main.tex':table('Full-budget updating and multiplicity sensitivity on the time-defined inactivity outcome. C is the number of base-course clusters, not model fits. Full-Platt intervals expand percentile tails for its six comparisons. Other columns are means, with pointwise and 96-comparison sensitivities in Supplementary Section S21; every expanded interval crosses zero. Matched compares equally sized S/T1 full-Platt fits, whereas the other columns retain the full historical S map.', 'tab:conservative', ['Day','Model','C','Full Platt U [six-comparison interval]','Intercept U','Shrinkage U','Matched contrast'],summary,'llrlrrr'),
      'conservative_supp.tex':table('Updating advantage over the unchanged full historical map: mean [pointwise 95\\% interval] from 100,000 cluster resamples. Full, historical-slope intercept and count-weighted shrinkage updates share the same T2 observations and per-budget support. Days 7/14/28 have 141/152/155 source splits per model; the 10\\% Day 7 budget supports 136 (29 triples, 22 clusters), while other budgets use all splits (30/32/32 triples and 23/24/24 clusters). None of the 72 expanded strategy intervals excludes zero in the 96-contrast sensitivity.', 'tab:s20',['Day','Model','Update','10\\%','20\\%','50\\%','100\\%'],supp,'lllrrrr'),
      'matched_updates_supp.tex':table('Equal-label-budget recency comparison: matched-S Brier minus matched-T1 Brier on the same T2 learners, mean [pointwise 95\\% interval]. Both full-Platt fits use m=min(source calibration N, available T1 budget N). No failed draw is replaced. Support matches Table S20 except for one additional Day 28 source split per model at 10\\% (154 supported); there are 3,572 supported evaluations out of 3,584. Every full-budget pointwise interval and all 24 multiplicity-expanded intervals cross zero.', 'tab:s21',['Day','Model','10\\%','20\\%','50\\%','100\\%'],matched,'llrrrr'),
      'multiplicity_supp.tex':table('Multiplicity sensitivity for the six full-budget full-Platt comparisons. All columns use the same 100,000 course-cluster bootstrap draws. The six-comparison interval uses 0.05/(2x6) tails; the 96-comparison sensitivity uses 0.05/(2x96) tails. Finite-cluster bootstrap coverage remains approximate. These are sensitivity analyses of exploratory results, not exact simultaneous-coverage guarantees.', 'tab:s22',['Day','Model','U','Pointwise 95\\%','Six comparisons','96 comparisons'],mult,'llrlll')
    }
    for name,value in tables.items():(OUT/name).write_text(value)
    print('Conservative update figure and four table inputs generated')

if __name__=='__main__':main()
