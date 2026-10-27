"""Fixed-neighborhood searchlight RSA and joint two-map max-|t| inference."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='8'
from pathlib import Path
import sys,json,time,hashlib
sys.dont_write_bytecode=True
REPOSITORY=Path(__file__).resolve().parents[2]
OUT=REPOSITORY/'outputs/searchlight_reanalysis'
OUT.mkdir(parents=True,exist_ok=True)
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse,stats
from scipy.spatial import cKDTree
from scipy.ndimage import label
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)
# Explicit upstream inputs; these arrays are not part of the compact release.
F=Path(os.environ['NEWS_BRAIN_ARRAYS'])
H=Path(os.environ['NEWS_MODEL_GRAMS'])
R=Path(os.environ['NEWS_RSA_PREPARED'])
for d in ['data/subjects','maps','qa','figures']:(OUT/d).mkdir(exist_ok=True,parents=True)
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
config=dict(radius_mm=6,min_voxels=27,full_sphere_voxels=33,coverage_rule='>=27 of 33 full-sphere lattice voxels in the common finite mask',
    model='qwen25_7b',pooling='last_token',layers=list(range(1,29)),shuffled_draws=10,
    variants=['raw','semantic_only'],conditions=['A','S','T'],contrast='T-S',n_signflips=9999,seed=20261023,
    inference='two-sided participant sign-flip max absolute t across both variants and all tested centers',alpha=.05,
    participant_unit=True,exploratory=True,spatial_smoothing=False,center_mask='intersection of finite voxels of all 95 participants',
    study_scope='retrospective, fixed articles, no new model inference; local multivoxel RSA, not single-voxel encoding')
cp=OUT/'analysis_config.json'
if cp.exists():assert json.loads(cp.read_text(encoding='utf-8'))==config
else:dump(cp,config)
SUBS=json.loads((H/'participants.json').read_text(encoding='utf-8'))
mask_img=nib.load(F/'analysis_mask.nii.gz');mask=mask_img.get_fdata()>0
assert mask.sum()==61825 and len(SUBS)==95
finite=np.stack([np.load(F/f'{s}_predictors.npz')['finite'] for s in SUBS])
common=finite.all(0); coords=np.argwhere(mask)[common];world=nib.affines.apply_affine(mask_img.affine,coords)
tree=cKDTree(world);neighbors=tree.query_ball_point(world,6.000001)
centers=np.flatnonzero(np.array([len(v) for v in neighbors])>=27)
neighbors=[np.asarray(neighbors[i],dtype=int) for i in centers];nv=len(centers)
sizes=np.array([len(v) for v in neighbors])
W=sparse.csr_matrix((np.ones(sizes.sum()),np.concatenate(neighbors),np.r_[0,np.cumsum(sizes)]),shape=(nv,len(coords)))
def nii(values,name,indices=centers,fill=np.nan):
    array=np.full(mask.shape,fill,dtype=np.float32);array[tuple(coords[indices].T)]=values
    img=nib.Nifti1Image(array,mask_img.affine);img.header.set_xyzt_units('mm');nib.save(img,OUT/'maps'/f'{name}.nii.gz')
nii(np.ones(len(coords)),'common_finite_mask',np.arange(len(coords)),fill=0)
nii(np.ones(nv),'searchlight_tested_centers',fill=0);nii(sizes,'searchlight_neighbor_count')
np.savez_compressed(OUT/'data/searchlight_geometry.npz',common_mask_vector=common,voxel_indices=coords,world_coordinates=world,
                    center_indices=centers,neighbor_indices=W.indices,neighbor_indptr=W.indptr)
dump(OUT/'qa/geometry.json',dict(original_voxels=int(mask.sum()),common_voxels=len(coords),tested_centers=nv,
                               excluded_common_centers=len(coords)-nv,min_neighbors=int(sizes.min()),max_neighbors=int(sizes.max())))
print(f'Geometry: {len(coords)} common voxels, {nv} tested centers; neighborhoods {sizes.min()}–{sizes.max()}.',flush=True)
def unit_rows(x):
    x=x-x.mean(1,keepdims=True);norm=np.linalg.norm(x,axis=1)
    if not (norm>1e-12).all():raise ValueError('Undefined RDM: no silent exclusions allowed')
    return x/norm[:,None]
all_results=[];checks=[];provenance={};started=time.time()
for si,s in enumerate(SUBS):
    target=OUT/'data/subjects'/f'{s}.npz'
    inputs=[F/f'{s}_betas.npy',F/f'{s}_predictors.npz',H/f'qwen25_7b/{s}_grams.npy',R/f'data/prepared/qwen25_7b/{s}.npz']
    digests={str(p):sha(p) for p in inputs};provenance.update(digests)
    if target.exists():
        saved=np.load(target);assert json.loads(str(saved['source_hashes']))==digests
        result=saved['rho'];assert result.shape==(2,3,nv);all_results.append(result)
        checks.extend(json.loads(str(saved['independent_checks'])))
        print(f'{s}: reused verified checkpoint {si+1}/95',flush=True);continue
    pred=np.load(inputs[1]);prep=np.load(inputs[3]);beta=np.load(inputs[0])
    assert np.array_equal(pred['article_ids'],prep['article_ids']) and np.array_equal(pred['trial_index'],prep['trial_indices'])
    x=beta[:,common[pred['finite']]].astype(float);n=len(x);ij=np.triu_indices(n,1);npair=len(ij[0])
    assert x.shape==(n,len(coords)) and np.isfinite(x).all()
    # Local Pearson distances through exact neighborhood sums.
    sums=W@x.T;sq=W@(x*x).T
    energy=sq-sums*sums/sizes[:,None]
    assert (energy>1e-16).all()
    products=x[ij[0]]*x[ij[1]]
    cross=W@products.T
    cov=cross-sums[:,ij[0]]*sums[:,ij[1]]/sizes[:,None]
    corr=np.clip(cov/np.sqrt(energy[:,ij[0]]*energy[:,ij[1]]),-1,1)
    neural=1-corr;rankbrain=stats.rankdata(neural,axis=1,method='average')
    bn=unit_rows(rankbrain)
    semantic=unit_rows(stats.rankdata(prep['semantic_rdm'])[None,:])[0]
    centered=rankbrain-rankbrain.mean(1,keepdims=True)
    bpartial=unit_rows(centered-(centered@semantic)[:,None]*semantic)
    grams=np.load(inputs[2],mmap_mode='r')
    assert grams.shape[:3]==(2,29,12)
    distances=np.stack([1-np.clip(grams[0,layer][:,ij[0],ij[1]],-1,1) for layer in range(1,29)])
    ranks=stats.rankdata(distances.reshape(28*12,npair),axis=1,method='average')
    mr=unit_rows(ranks)
    centered_m=ranks-ranks.mean(1,keepdims=True)
    mp=unit_rows(centered_m-(centered_m@semantic)[:,None]*semantic)
    weights=[]
    for r in [mr,mp]:
        r=r.reshape(28,12,npair)
        # Linearity: average normalized rank vectors, then dot with a brain RDM.
        # Exactly equals mean of separately computed layer/draw RSA, not mean hidden vectors/RDM.
        weights.append(np.stack([r[:,0].mean(0),r[:,1:11].mean((0,1)),r[:,11].mean(0)]))
    result=np.stack([weights[0]@bn.T,weights[1]@bpartial.T])
    assert np.isfinite(result).all()
    independent=[]
    for ci in sorted(set([0,nv//3,nv//2,nv-1])):
        local=x[:,neighbors[ci]];direct=(1-np.corrcoef(local))[ij]
        np.testing.assert_allclose(neural[ci],direct,atol=2e-12,rtol=1e-10)
        # Independent Spearman / residualized rank correlations on selected local patterns.
        control=np.column_stack([np.ones(npair),stats.rankdata(prep['semantic_rdm'])])
        br=stats.rankdata(direct);be=br-control@np.linalg.lstsq(control,br,rcond=None)[0]
        vals=np.empty((2,28,12))
        for l in range(28):
            for c in range(12):
                dm=distances[l,c];rr=stats.rankdata(dm)
                vals[0,l,c]=stats.spearmanr(dm,direct).statistic
                er=rr-control@np.linalg.lstsq(control,rr,rcond=None)[0]
                vals[1,l,c]=np.corrcoef(er,be)[0,1]
        ref=np.stack([vals[:,:,0].mean(1),vals[:,:,1:11].mean((1,2)),vals[:,:,11].mean(1)],axis=1)
        np.testing.assert_allclose(result[:,:,ci],ref,atol=5e-10,rtol=1e-8)
        independent.append(dict(sub=s,center=int(ci),max_abs_error=float(np.max(abs(result[:,:,ci]-ref)))))
    np.savez_compressed(target,rho=result,source_hashes=json.dumps(digests),independent_checks=json.dumps(independent))
    all_results.append(result);checks.extend(independent)
    print(f'{s}: {si+1}/95, elapsed {time.time()-started:.1f}s',flush=True)

rho=np.stack(all_results);delta=rho[:,:,2]-rho[:,:,1]
np.save(OUT/'data/participant_T_minus_S.npy',delta)
dump(OUT/'qa/independent_RSA_checks.json',dict(n_checks=len(checks),maximum_error=max(c['max_abs_error'] for c in checks),checks=checks))
dump(OUT/'qa/input_hashes.json',provenance)
# Common sign vectors preserve spatial and cross-variant dependence under the null.
flat=delta.reshape(95,-1);ss=(flat*flat).sum(0);observed_sum=flat.sum(0)
def t_from_sum(total):
    denom=np.maximum(95*ss-total*total,0)
    assert (denom>0).all()
    return total*np.sqrt(94)/np.sqrt(denom)
tobs=t_from_sum(observed_sum)
np.testing.assert_allclose(tobs,stats.ttest_1samp(flat,0,axis=0).statistic,atol=1e-10)
rng=np.random.default_rng(config['seed']);signs=rng.choice(np.array([-1,1],dtype=np.int8),size=(config['n_signflips'],95))
np.save(OUT/'data/signflip_signs.npy',signs)
maxima=[];pstart=time.time()
for start in range(0,len(signs),128):
    totals=signs[start:start+128].astype(float)@flat
    ts=t_from_sum(totals);maxima.extend(np.max(np.abs(ts),axis=1))
    if start%1280==0:print(f'Joint max-|t| permutations: {min(start+128,len(signs))}/{len(signs)}; {time.time()-pstart:.1f}s',flush=True)
maxima=np.asarray(maxima);np.save(OUT/'data/joint_max_abs_t_null.npy',maxima)
p_fwer=(1+len(maxima)-np.searchsorted(np.sort(maxima),np.abs(tobs),side='left'))/(len(maxima)+1)
p_fwer=p_fwer.reshape(2,nv);tobs=tobs.reshape(2,nv)
summary=[];clusters=[]
for vi,v in enumerate(config['variants']):
    for ci,c in enumerate(config['conditions']):nii(rho[:,vi,ci].mean(0),f'{v}_{c}_mean_RSA')
    mean=delta[:,vi].mean(0);significant=p_fwer[vi]<.05
    nii(mean,f'{v}_T_minus_S_mean_delta_RSA');nii(tobs[vi],f'{v}_T_minus_S_t')
    nii(p_fwer[vi],f'{v}_T_minus_S_p_FWER_joint');nii(significant.astype(float),f'{v}_T_minus_S_significant_FWER',fill=0)
    nii(np.where(significant,mean,np.nan),f'{v}_T_minus_S_FWER_delta_RSA')
    for direction in [1,-1]:
        binary=np.zeros(mask.shape,dtype=bool);binary[tuple(coords[centers].T)]=significant & (mean*direction>0)
        labs,nlab=label(binary,structure=np.ones((3,3,3)))
        for lab in range(1,nlab+1):
            loc=labs[tuple(coords[centers].T)]==lab;ix=np.flatnonzero(loc);peak=ix[np.argmax(np.abs(tobs[vi,ix]))]
            clusters.append(dict(variant=v,sign='positive' if direction==1 else 'negative',component=lab,n_centers=len(ix),
                                 peak_x_mm=world[centers[peak],0],peak_y_mm=world[centers[peak],1],peak_z_mm=world[centers[peak],2],
                                 peak_t=tobs[vi,peak],peak_delta_RSA=mean[peak],peak_p_FWER=p_fwer[vi,peak],
                                 note='descriptive connected component of voxelwise FWER mask; not cluster-level inference'))
    summary.append(dict(variant=v,n_subjects=95,tested_centers=nv,total_family_tests=2*nv,n_positive_FWER=int((significant&(mean>0)).sum()),
        n_negative_FWER=int((significant&(mean<0)).sum()),min_joint_p=float(p_fwer[vi].min()),max_abs_t=float(abs(tobs[vi]).max()),
        min_mean_delta=float(mean.min()),max_mean_delta=float(mean.max()),mean_over_centers=float(mean.mean())))
    pd.DataFrame(dict(x_mm=world[centers,0],y_mm=world[centers,1],z_mm=world[centers,2],n_neighbors=sizes,
        mean_delta_RSA=mean,t=tobs[vi],p_FWER_joint=p_fwer[vi],significant_FWER=significant)).to_csv(OUT/'data'/f'{v}_voxel_results.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'data/searchlight_summary.csv',index=False)
pd.DataFrame(clusters).to_csv(OUT/'data/significant_components.csv',index=False)
dump(OUT/'qa/analysis_complete.json',dict(status='passed',seconds=time.time()-started,n_subjects=95,n_centers=nv,n_joint_tests=nv*2,
    n_permutations=len(signs),min_p_resolution=1/(len(signs)+1),summary=summary,config_sha256=sha(cp)))
print(json.dumps(summary,indent=2),flush=True)
