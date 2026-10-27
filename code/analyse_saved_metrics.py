"""Recompute adopted group tests from released metrics, writing only to outputs/.

This release entry point adapts the saved analysis scripts to the public layout.
It does not run LLMs, fit fMRI models, or overwrite the released reference data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from core.rsa import holm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'

def sign_test(values, signs, tie_tolerance=0):
    values=np.asarray(values,float)
    if values.ndim==1: values=values[:,None]
    if not np.isfinite(values).all(): raise ValueError('Non-finite paired input')
    threshold=np.maximum(0,np.abs(values.mean(0))*(1-tie_tolerance))
    count=np.zeros(values.shape[1],dtype=np.int64)
    for start in range(0,len(signs),2048):
        null=signs[start:start+2048].astype(float)@values/len(values)
        count+=(np.abs(null)>=threshold).sum(0)
    return (count+1)/(len(signs)+1)

def intervals(values,indices):
    return np.percentile(np.asarray(values)[indices].mean(1),[2.5,97.5],axis=0)

def platform():
    frame=pd.read_csv(DATA/'figure_02/user_metrics_full_precision.csv',dtype={'user_id':str},float_precision='round_trip')
    saved=pd.read_csv(DATA/'figure_02/statistics_holm18.csv',float_precision='round_trip')
    rows=[]
    for model in ['qwen25_7b','deepseek_r1_8b']:
        for metric in ['ndcg_at_5','map','auc']:
            wide=frame[frame.model.eq(model)].pivot(index='user_id',columns='condition',values=metric).sort_index()
            for first,second in [('true_history','article_only'),('true_history','shuffled_history'),('shuffled_history','article_only')]:
                x,y=wide[first].to_numpy(),wide[second].to_numpy()
                t,p=stats.ttest_rel(x,y,alternative='two-sided')
                contrast=f'{first}_minus_{second}'
                ci=saved[saved.model.eq(model)&saved.metric.eq(metric)&saved.contrast.eq(contrast)].iloc[0]
                rows.append(dict(model=model,metric=metric,contrast=contrast,mean_difference=(x-y).mean(),t_statistic=t,p_raw=p,ci_low=ci.ci95_low,ci_high=ci.ci95_high,interval_source='released bootstrap interval; not recalculated by this command'))
    out=pd.DataFrame(rows);out['p_holm']=holm(out.p_raw);return out

def laboratory():
    import json
    cfg=json.loads((ROOT/'config/laboratory_analysis.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(DATA/'figure_03/Figure3.csv',float_precision='round_trip')
    rng=np.random.default_rng(cfg['bootstrap_seed']);rows=[]
    for model in cfg['models']:
        g=frame[frame.model.eq(model)].sort_values('sub')
        for metric in ['pearson_r','mae']:
            v=g[[f'{metric}_{c}' for c in cfg['conditions']]].to_numpy(float)
            v=v[np.isfinite(v).all(1)]
            ix=rng.integers(0,len(v),size=(cfg['n_bootstrap'],len(v)))
            boots=v[ix].mean(1)
            for j,comparator in enumerate(cfg['conditions'][:2]):
                sign=1 if metric=='pearson_r' else -1
                diff=sign*(v[:,2]-v[:,j]);lo,hi=np.percentile(sign*(boots[:,2]-boots[:,j]),[2.5,97.5])
                t,p=stats.ttest_rel(sign*v[:,2],sign*v[:,j],alternative='two-sided')
                rows.append(dict(model=model,metric=metric,comparator=comparator,mean_benefit=diff.mean(),ci_low=lo,ci_high=hi,t_statistic=t,p_raw=p))
    out=pd.DataFrame(rows);out['p_holm']=holm(out.p_raw);return out

def roi():
    frame=pd.read_csv(DATA/'figure_06/roi_rsa_participants.csv',float_precision='round_trip')
    with np.load(DATA/'resampling/roi_rsa_indices.npz',allow_pickle=False) as z:
        subjects,signs,ix=z['subjects'],z['signs'],z['bootstrap_000']
    rows=[]
    for model in ['qwen25_7b','deepseek_r1_8b']:
        for region in ['self_referential','mentalizing','value']:
            w=frame[frame.model.eq(model)&frame.roi.eq(region)].pivot(index='sub',columns='condition',values='rho_mean').loc[subjects]
            for ref,label in [('article_only','true_minus_article'),('shuffled','true_minus_shuffled')]:
                v=(w.true_history-w[ref]).to_numpy();lo,hi=intervals(v,ix)
                rows.append(dict(model=model,roi=region,contrast=label,mean_delta=v.mean(),ci_low=lo,ci_high=hi,p_raw=sign_test(v,signs)[0]))
    out=pd.DataFrame(rows);out['p_holm']=holm(out.p_raw);return out

def layers():
    frame=pd.read_csv(DATA/'figure_S04/participant_layer_gains.csv',float_precision='round_trip')
    ref=pd.read_csv(DATA/'figure_S04/layer_gain_summary.csv',float_precision='round_trip')
    with np.load(DATA/'resampling/roi_rsa_indices.npz',allow_pickle=False) as z:subjects,signs=z['subjects'],z['signs']
    keys=['pooling','roi','layer','contrast']
    wide=frame.pivot(index='sub',columns=keys,values='delta_rho').loc[subjects]
    values=wide.loc[:,pd.MultiIndex.from_frame(ref[keys])].to_numpy()
    assert values.shape==(95,342)
    out=ref.copy();out['p_raw']=sign_test(values,signs,tie_tolerance=1e-12)
    out['p_holm342']=holm(out.p_raw);out['significant']=out.p_holm342<.05
    return out

def readout():
    with np.load(DATA/'figure_S04/raw_layer_RSA.npz',allow_pickle=False) as z:
        arr,subjects,pools,rois,conditions=z['rho'],z['subjects'],z['poolings'],z['rois'],z['conditions'].tolist()
    with np.load(DATA/'resampling/roi_rsa_indices.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(subjects,z['subjects']);ix,signs=z['bootstrap_000'],z['signs']
    si=[i for i,c in enumerate(conditions) if c.startswith('shuffled_history_draw_')]
    gain=arr[:,:,:,conditions.index('true_history'),:]-arr[:,:,:,si,:].mean(3)
    v=gain[:,list(pools).index('last_token'),1:,:].mean(1)-gain[:,list(pools).index('target_mean'),1:,:].mean(1)
    lo,hi=intervals(v,ix);p=sign_test(v,signs)
    return pd.DataFrame(dict(roi=rois,mean_difference=v.mean(0),ci_low=lo,ci_high=hi,p_raw=p,p_holm=holm(p)))

def semantic():
    frame=pd.read_csv(DATA/'figure_06/MainFigure2_d_subject_contrasts.csv',float_precision='round_trip')
    rois=['value','self_referential','mentalizing'];arrays=[]
    for region in rois:
        w=frame[frame.roi.eq(region)].pivot(index='sub',columns='variant',values='delta_rho').sort_index()
        arrays.append((w.semantic_only-w.raw).to_numpy())
    v=np.array(arrays).T
    with np.load(DATA/'resampling/semantic_control_signflips.npz',allow_pickle=False) as z:signs=z['signs']
    ix=np.random.default_rng(2026102304).integers(0,95,size=(10000,95))
    lo,hi=intervals(v,ix);p=sign_test(v,signs)
    return pd.DataFrame(dict(roi=rois,mean_delta=v.mean(0),ci_low=lo,ci_high=hi,p_raw=p,p_holm=holm(p)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis',choices=['platform','laboratory','roi','layers','readout','semantic'])
    args=parser.parse_args()
    out=ROOT/'outputs/statistics';out.mkdir(parents=True,exist_ok=True)
    result=globals()[args.analysis]()
    destination=out/(args.analysis+'.csv');result.to_csv(destination,index=False)
    print(destination)

if __name__=='__main__':main()
