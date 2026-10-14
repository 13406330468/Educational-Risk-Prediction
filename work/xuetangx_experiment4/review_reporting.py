"""Generate review figures and common-budget summaries from aggregate evidence only."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(os.environ.get('CALTRANSFER_ROOT',str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'outputs/review_quality_20261011'

def bootstrap_common(d):
    keys=['triple_id','day','base_model','seed']
    good=d.groupby(keys).budget_supported.all()
    valid=d.merge(good[good].reset_index()[keys],on=keys)
    pm=valid.groupby(['triple_id','provider','base_course_token','day','base_model','fraction'],as_index=False).update_advantage.mean()
    rows=[]
    for (day,model),g in pm.groupby(['day','base_model']):
        wide=g.pivot(index=['provider','base_course_token','triple_id'],columns='fraction',values='update_advantage').reset_index()
        clusters=list(wide.groupby(['provider','base_course_token']).indices.values());a=wide[[.1,.2,.5,1.]].to_numpy();rng=np.random.default_rng(20261011)
        boot=np.array([a[np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))])].mean(axis=0) for _ in range(2000)])
        for j,f in enumerate([.1,.2,.5,1.]):
            rows.append({'day':day,'base_model':model,'fraction':f,'mean':a[:,j].mean(),'ci_low':np.quantile(boot[:,j],.025),'ci_high':np.quantile(boot[:,j],.975),'triples':len(wide),'clusters':len(clusters)})
    result=pd.DataFrame(rows);result.to_csv(OUT/'rollout_common_budget_intervals.csv',index=False)
    return result

def main():
    d=bootstrap_common(pd.read_csv(OUT/'rollout_by_job.csv'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.55),sharey=True)
    for ax,day in zip(axs,[7,14,28]):
        ax.axhline(0,c='#738297',ls='--',lw=.8)
        for model,color,marker,offset,label in [('logistic','#2467A9','o',-.06,'LR'),('lightgbm','#D86808','s',.06,'LightGBM')]:
            g=d[(d.day==day)&(d.base_model==model)].sort_values('fraction')
            y=g['mean'].to_numpy();ax.errorbar(np.arange(4)+offset,y,yerr=np.array([y-g.ci_low.to_numpy(),g.ci_high.to_numpy()-y]),color=color,marker=marker,lw=1,capsize=2,ms=4,label=label)
        ax.set_title(f'Day {day}',fontweight='bold');ax.set_xticks(range(4),['10','20','50','100']);ax.set_xlabel('T1 labels used (%)');ax.grid(axis='y',color='#E8EDF2',lw=.5);ax.set_axisbelow(True)
    axs[0].set_ylabel('Brier advantage over S mapping');axs[-1].legend(frameon=False,loc='upper left',fontsize=8)
    fig.tight_layout(w_pad=.8);fig.savefig(OUT/'figure5_rollout.pdf',bbox_inches='tight');fig.savefig(OUT/'figure5_rollout.png',dpi=200,bbox_inches='tight');plt.close(fig)
    bins=pd.read_csv(OUT/'reliability_curve_bins.csv');selected=bins[['pair_id','target_n']].drop_duplicates().sort_values('target_n')
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.8),sharex=True,sharey=True)
    for col,q in enumerate(selected.itertuples()):
        for row,model in enumerate(['logistic','lightgbm']):
            ax=axs[row,col];ax.plot([0,1],[0,1],ls='--',lw=.7,color='#AAB3BD')
            for method,color,marker,label in [('raw','#748398','o','Raw'),('source_Platt','#D86808','s','Historical'),('oof_Platt','#2467A9','^','Target OOF')]:
                g=bins[(bins.pair_id==q.pair_id)&(bins.base_model==model)&(bins.method==method)].sort_values('bin')
                ax.plot(g.mean_probability,g.observed_rate,color=color,marker=marker,lw=.85,ms=3,label=label)
            ax.set_title(f'{q.pair_id}, n={q.target_n}\n'+('LR' if model=='logistic' else 'LightGBM'),fontsize=9)
            ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03));ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
            if col==0:ax.set_ylabel('Observed event rate')
            if row==1:ax.set_xlabel('Mean predicted probability')
    axs[0,0].legend(frameon=False,fontsize=6,loc='upper left');fig.tight_layout();fig.savefig(OUT/'figureS1_reliability.pdf',bbox_inches='tight');fig.savefig(OUT/'figureS1_reliability.png',dpi=200,bbox_inches='tight');plt.close(fig)
    print('Review figures generated from aggregate evidence')

if __name__=='__main__':main()
