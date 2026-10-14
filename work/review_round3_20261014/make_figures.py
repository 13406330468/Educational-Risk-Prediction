"""Publication-ready static plots from aggregate experiment tables only."""
import argparse,os
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
out=a.output/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.labelcolor':'#173453','axes.titlecolor':'#173453','text.color':'#173453','pdf.fonttype':42,'ps.fonttype':42})
colors={'baseline':'#2467A9','recent':'#D86808','joint':'#64748B'}
labels={'baseline':'Baseline rerun','recent':'Recent-active refit','joint':'Joint assigned-user refit'}
def save(fig,name):
    fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)

f=pd.read_csv(a.output/'summary/pair_intervals.csv');f=f[f.metric.eq('A')]
fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,model,title in zip(axs,['logistic','lightgbm'],['Logistic regression','LightGBM']):
    ax.axvline(0,color='#A5ADB6',ls='--',lw=.9)
    for offset,mode,marker in zip([-.22,0,.22],['baseline','recent','joint'],['o','s','^']):
        g=f[(f['mode']==mode)&(f.base_model==model)].sort_values('day')
        means=g['mean'].to_numpy();low=g.point_low.to_numpy();high=g.point_high.to_numpy()
        ax.errorbar(means,np.arange(3)+offset,xerr=np.vstack([means-low,high-means]),color=colors[mode],fmt=marker,ms=5,capsize=3,lw=1.2,label=labels[mode])
    ax.set_title(title,fontweight='bold');ax.set_yticks(range(3),['Day 7','Day 14','Day 28']);ax.set_ylim(2.6,-.6)
    ax.set_xlim(-.002,.048);ax.set_xlabel('Retrospective adaptation gap A (Brier)');ax.grid(axis='x',alpha=.18)
handles,leg=axs[0].get_legend_handles_labels();fig.legend(handles,leg,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
fig.tight_layout(rect=[0,.08,1,1]);save(fig,'01_adaptation_sensitivity')

h=pd.read_csv(a.output/'summary/course_update_distribution.csv');h=h[h['mode'].eq('baseline')&h.fraction.eq(1)&h.strategy.eq('full')]
impact=pd.read_csv(a.output/'summary/course_influence_summary.csv')
fig,axs=plt.subplots(2,3,figsize=(10,5.6),sharey=True)
for row,model in enumerate(['logistic','lightgbm']):
    for col,day in enumerate([7,14,28]):
        ax=axs[row,col];g=h[(h.base_model==model)&(h.day==day)].sort_values('course_mean_U')
        ax.axhline(0,c='#A5ADB6',ls='--',lw=.8)
        v=impact[(impact['mode']=='baseline')&(impact.base_model==model)&(impact.day==day)&impact.fraction.eq(1)&impact.strategy.eq('full')].iloc[0]
        ax.axhline(v.mean_U,c='#173453',lw=.8,label='Triple-weighted overall mean')
        ax.scatter(np.arange(1,len(g)+1),g.course_mean_U,c=np.where(g.course_mean_U.ge(0),'#2467A9','#D86808'),s=23,zorder=3)
        ax.set_title(f'{"LR" if row==0 else "LightGBM"} | Day {day} | {len(g)} courses',fontsize=10)
        ax.set_xticks([1,len(g)]);ax.grid(axis='y',alpha=.16)
        if row==1:ax.set_xlabel('Base course, ordered by mean U')
        if col==0:ax.set_ylabel('Full-budget update advantage U')
fig.tight_layout();save(fig,'02_course_heterogeneity')

f=pd.read_csv(a.output/'recent/conservative/update_intervals.csv')
fig,axs=plt.subplots(2,3,figsize=(10,5.8),sharey='row')
for row,model in enumerate(['logistic','lightgbm']):
    for col,day in enumerate([7,14,28]):
        ax=axs[row,col];ax.axhline(0,c='#A5ADB6',ls='--',lw=.8)
        for method,color,marker,off in [('full','#2467A9','o',-.12),('intercept','#D86808','s',0),('shrink','#64748B','^',.12)]:
            g=f[(f.base_model==model)&(f.day==day)&(f.strategy==method)].sort_values('fraction')
            mean=g['mean'].to_numpy();ax.errorbar(np.arange(4)+off,mean,yerr=np.vstack([mean-g.point_low.to_numpy(),g.point_high.to_numpy()-mean]),
                  color=color,fmt=marker+'-',ms=3.5,lw=.85,capsize=2,label={'full':'Full Platt','intercept':'Intercept update','shrink':'Parameter shrinkage'}[method])
            for j,r in enumerate(g.itertuples()):
                if r.family96_low>0 or r.family96_high<0:ax.plot(j+off,r.mean,marker='*',c='#173453',ms=10,zorder=6)
        ax.set_title(f'{"LR" if row==0 else "LightGBM"} | Day {day}',fontsize=10)
        ax.set_xticks(range(4),['10','20','50','100']);ax.grid(axis='y',alpha=.15)
        if row==1:ax.set_xlabel('T1 label budget (%)')
        if col==0:ax.set_ylabel('Future update advantage U')
handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
fig.tight_layout(rect=[0,.05,1,1]);save(fig,'03_recent_budget_strategies')
print('FIGURES_COMPLETE',flush=True)
