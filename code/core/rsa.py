"""RSA and rank-residualisation kernels from the analysis source."""
import numpy as np
from scipy.stats import rankdata
TOL = 64*np.finfo(np.float64).eps

def ranked_rows(x):
    x=np.asarray(x,dtype=float)
    valid=np.isfinite(x).all(1)&(np.ptp(x,axis=1)>TOL)
    r=np.zeros_like(x)
    r[valid]=rankdata(x[valid],method='average',axis=1)
    r-=r.mean(1,keepdims=True)
    return r,valid


def correlate_rows(a,b,valid_a,valid_b):
    na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1)
    valid=valid_a[:,None]&valid_b[None,:]&(na[:,None]>0)&(nb[None,:]>0)
    out=np.full((len(a),len(b)),np.nan)
    np.divide(a@b.T,na[:,None]*nb[None,:],out=out,where=valid)
    return np.clip(out,-1,1)


def spearman_matrix(a,b):
    ar,va=ranked_rows(a);br,vb=ranked_rows(b)
    return correlate_rows(ar,br,va,vb)


def partial_matrix(a,b,controls):
    ar,va=ranked_rows(a);br,vb=ranked_rows(b)
    c=np.asarray(controls,dtype=float)
    if c.ndim==1:c=c[:,None]
    if not np.isfinite(c).all():
        return np.full((len(a),len(b)),np.nan),{'rank':-1,'n_control_columns':int(c.shape[1]),'reason':'nonfinite_control'}
    cols=[]
    for v in c.T:
        if np.ptp(v)==0:continue
        q=rankdata(v,method='average');q-=q.mean();q/=np.linalg.norm(q);cols.append(q)
    c=np.column_stack([np.ones(len(ar[0]))]+cols)
    u,s,_=np.linalg.svd(c,full_matrices=False);keep=s>s[0]*1e-12;basis=u[:,keep]
    ae=ar-(ar@basis)@basis.T;be=br-(br@basis)@basis.T
    an=np.linalg.norm(ar,axis=1);bn=np.linalg.norm(br,axis=1)
    arn=np.linalg.norm(ae,axis=1);brn=np.linalg.norm(be,axis=1)
    va&=arn>an*1e-10;vb&=brn>bn*1e-10
    ra=np.divide(arn,an,out=np.zeros_like(an),where=an>0)
    rb=np.divide(brn,bn,out=np.zeros_like(bn),where=bn>0)
    return correlate_rows(ae,be,va,vb),dict(rank=int(keep.sum()),n_control_columns=len(cols),
        condition_number=float(s[0]/s[keep][-1]),min_model_residual_ratio=float(ra.min()),
        min_brain_residual_ratio=float(rb.min()),n_invalid_model=int((~va).sum()),n_invalid_brain=int((~vb).sum()),reason='ok')


def aggregate_layers(x,condition_indices=None):
    """Input [nonembedding layers,conditions,ROI], never silently omit NA."""
    if condition_indices is None:condition_indices=np.arange(x.shape[1])
    x=x[:,condition_indices]
    good=np.isfinite(x).all(axis=(0,1))
    out=x.mean(0);out[:,~good]=np.nan
    return out,good


def holm(p):
    p=np.asarray(p,float);v=np.where(np.isfinite(p),p,1.);order=np.argsort(v,kind='stable')
    adj=np.minimum(1,np.maximum.accumulate(v[order]*np.arange(len(v),0,-1)))
    out=np.empty_like(v);out[order]=adj;out[~np.isfinite(p)]=np.nan
    return out
