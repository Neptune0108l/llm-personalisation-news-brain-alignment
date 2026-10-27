"""Plot frozen summaries; no confidence intervals or significance are recomputed.

These compact redraws preserve data, not exact manuscript typography.
Current publication exports are in figures/. All new exports go to outputs/.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#7B818A','#B68132','#256B95']
CONDITIONS=['article_only','shuffled_history','true_history']
LABELS=['Article only','Shuffled history','True history']
plt.rcParams.update({'font.family':'sans-serif','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def point(ax,mean,lo,hi,y,color,marker='o'):
    ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt=marker,color=color,ms=4,capsize=2)

def behaviour(which):
    if which=='02':
        frame=pd.read_csv(ROOT/'data/figure_02/summary_full_precision.csv')
        metrics=['ndcg_at_5','map','auc'];titles=['nDCG@5','MAP','AUC'];models=['qwen25_7b','deepseek_r1_8b']
    else:
        import json
        frame=pd.read_csv(ROOT/'data/figure_03/figure3_summary_from_participants.csv')
        metrics=['pearson_r','mae'];titles=['Pearson r','MAE'];models=json.loads((ROOT/'config/laboratory_analysis.json').read_text(encoding='utf-8'))['models']
    fig,axes=plt.subplots(1,len(metrics),figsize=(8,2+len(models)*.42),sharey=True,squeeze=False)
    for ax,metric,title in zip(axes[0],metrics,titles):
        for i,model in enumerate(models):
            for j,condition in enumerate(CONDITIONS):
                row=frame[frame.model.eq(model)&frame.condition.eq(condition)&frame.metric.eq(metric)].iloc[0]
                point(ax,row['mean'],row.ci95_low,row.ci95_high,i+(j-1)*.22,COLORS[j],['o','^','D'][j])
        ax.set(yticks=range(len(models)),yticklabels=models,xlabel=title,ylim=(len(models)-.5,-.5))
        ax.axvline(0,lw=.5,color='grey')
    handles=[plt.Line2D([],[],color=c,marker=m,ls='',label=l) for c,m,l in zip(COLORS,['o','^','D'],LABELS)]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.92));return fig

def activity():
    d=pd.read_csv(ROOT/'data/figure_04/panel_b_roi_group_summary_BH9_checked.csv').sort_values('activation_behavior_r_mean')
    fig,ax=plt.subplots(figsize=(6,4))
    colors=np.where(d.activation_behavior_r_p_bh9<.05,'#A26F4A','#8D959B')
    ax.barh(d.roi,d.activation_behavior_r_mean,xerr=d.activation_behavior_r_sem,color=colors,capsize=2)
    ax.set_xlabel('Mean activity–rating Pearson r (±1 SEM)');ax.axvline(0,color='grey',lw=.5)
    fig.tight_layout();return fig

def s3():
    d=pd.read_csv(ROOT/'data/figure_03/paired_participant_differences.csv')
    models=d.model.drop_duplicates().tolist();fig,axes=plt.subplots(2,2,figsize=(9,7),sharey=True)
    rng=np.random.default_rng(0) # display jitter only
    for ax,(metric,comp) in zip(axes.flat,[(m,c) for m in ['pearson_r','mae'] for c in ['shuffled_history','article_only']]):
        for i,model in enumerate(models):
            v=d[d.model.eq(model)&d.metric.eq(metric)&d.comparator.eq(comp)].benefit.to_numpy()
            ax.scatter(v,i+rng.uniform(-.15,.15,len(v)),color='#888888',s=5,alpha=.6)
            ax.scatter(v.mean(),i,marker='D',color='#A26F4A',s=20)
        ax.set(yticks=range(len(models)),yticklabels=models,title=f'{metric}: true-history benefit over {comp}',ylim=(len(models)-.5,-.5))
        ax.axvline(0,color='grey',ls='--',lw=.5)
    fig.tight_layout();return fig

def s5():
    d=pd.read_csv(ROOT/'data/figure_S05/four_settings_summary.csv')
    rois=['self_referential','mentalizing','value'];variants=['raw','semantic_only','gap1','voxel_centered']
    labels=['Raw','Semantic control','Exclude adjacent trial pairs','Voxel centring across trials']
    fig,axes=plt.subplots(1,3,figsize=(8,3),sharey=True)
    for ax,roi in zip(axes,rois):
        for y,variant in enumerate(variants):
            row=d[d.roi.eq(roi)&d.variant.eq(variant)].iloc[0]
            point(ax,row['mean'],row.ci_low,row.ci_high,y,'#A26F4A')
        ax.set(title=roi,yticks=range(4),yticklabels=labels,ylim=(3.5,-.5),xlabel='T − S ΔRSA')
        ax.axvline(0,color='grey',ls='--',lw=.5)
    fig.tight_layout();return fig

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('figure',choices=['02','03','04b','S03','S05']);a=p.parse_args()
    fig=behaviour(a.figure) if a.figure in {'02','03'} else {'04b':activity,'S03':s3,'S05':s5}[a.figure]()
    out=ROOT/'outputs/summary_plots';out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'Figure_{a.figure}.{ext}',dpi=600)
    plt.close(fig);print(out)

if __name__=='__main__':main()
