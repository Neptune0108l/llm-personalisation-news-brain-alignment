"""Six-panel observed RSA gain curves, paired pointwise intervals, Arial at 180 mm."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/figure_S04'
for child in ['figures','qa']:(OUT/child).mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
data=pd.read_csv(ROOT/'data/figure_S04/layer_significance_Holm342.csv')
fig,axes=plt.subplots(2,3,figsize=(180/25.4,145/25.4),sharex=True,sharey='col')
fig.subplots_adjust(left=.11,right=.98,bottom=.145,top=.90,wspace=.37,hspace=.48)
rois=['self_referential','mentalizing','value'];names=['Self-referential','Mentalizing','Value']
colors={'T-A':'#7B3294','T-S':'#008577'};marks={'T-A':'^','T-S':'v'}
fig.text(.11,.965,'Last token',fontsize=9,fontweight='bold')
fig.text(.11,.505,'Target-token mean',fontsize=9,fontweight='bold')
for pi,pool in enumerate(['last_token','target_mean']):
    for ri,(roi,name) in enumerate(zip(rois,names)):
        ax=axes[pi,ri];letter=chr(97+pi*3+ri);ax.set_label(letter)
        ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(0,9),textcoords='offset points',fontsize=8,fontweight='bold',va='baseline')
        ax.annotate(name,xy=(0,1),xycoords='axes fraction',xytext=(14,9),textcoords='offset points',fontsize=7,va='baseline')
        for contrast in ['T-A','T-S']:
            d=data[(data.pooling==pool)&(data.roi==roi)&(data.contrast==contrast)].sort_values('layer')
            ax.fill_between(d.layer,d.ci_low,d.ci_high,color=colors[contrast],alpha=.16,linewidth=0)
            ax.plot(d.layer,d['mean'],color=colors[contrast],lw=1,marker=marks[contrast],ms=3,markevery=[i for i,x in enumerate(d.layer) if x in [7,14,21,28]])
        ax.axhline(0,color='#92979C',lw=.6,ls=(0,(3,2)),zorder=0)
        ax.set_xlim(0,28);ax.set_xticks([0,7,14,21,28]);ax.set_xlabel('Layer');ax.tick_params(length=2.5,pad=2,labelbottom=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if ri==0:ax.set_ylabel('Model–brain RSA gain (Δρ)')
for ri,roi in enumerate(rois):
    d=data[data.roi==roi];lo=min(float(d.ci_low.min()),0);hi=max(float(d.ci_high.max()),0);pad=(hi-lo)*.08
    axes[0,ri].set_ylim(lo-pad,hi+(hi-lo)*.24)
# Dedicated annotation lanes above every uncertainty band; each dash marks one tested layer.
annotation_rows=[]
for pi,pool in enumerate(['last_token','target_mean']):
    for ri,roi in enumerate(rois):
        ax=axes[pi,ri]
        for contrast,lane in [('T-A',.955),('T-S',.915)]:
            selected=data[(data.pooling==pool)&(data.roi==roi)&(data.contrast==contrast)&data.significant]
            for row in selected.itertuples():
                x=float(row.layer)
                ax.plot([x-.32,x+.32],[lane,lane],transform=ax.get_xaxis_transform(),color=colors[contrast],lw=1.35,solid_capstyle='butt',clip_on=False)
                annotation_rows.append({'panel':chr(97+3*pi+ri),'pooling':pool,'roi':roi,'contrast':contrast,'layer':int(x),'p_holm342':row.p_holm342,'mean':row.mean,'lane_axes_fraction':lane})
pd.DataFrame(annotation_rows).to_csv(OUT/'qa/plotted_significance_marks.csv',index=False)
handles=[Line2D([],[],color=colors[c],marker=marks[c],lw=1,ms=3,label=t) for c,t in [('T-A','T−A: True history − Article only'),('T-S','T−S: True history − Shuffled history')]]
fig.canvas.draw();renderer=fig.canvas.get_renderer()
bottom=min(ax.get_tightbbox(renderer).y0 for ax in axes[1])
legend_y=(bottom-8*fig.dpi/72)/(fig.get_figheight()*fig.dpi)
center=(axes[1,0].get_position().x0+axes[1,2].get_position().x1)/2
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(center,legend_y),ncol=2,frameon=False,borderaxespad=0,columnspacing=2,handlelength=1.6)
fig.canvas.draw()
dest=OUT/'figures/Supplementary_Figure_S4_layer_significance'
fig.savefig(dest.with_suffix('.pdf'))
fig.savefig(dest.with_suffix('.svg'))
fig.savefig(dest.with_suffix('.png'),dpi=600)
fig.savefig(OUT/'figures/Supplementary_Figure_S4_layer_significance_preview.png',dpi=300)
plt.close(fig)
print(dest)
