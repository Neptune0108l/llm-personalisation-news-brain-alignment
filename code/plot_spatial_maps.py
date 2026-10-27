"""Display saved group searchlight maps; this does not rerun spatial inference."""
from pathlib import Path
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contrast',choices=['T_minus_S','T_minus_A'],default='T_minus_S')
    p.add_argument('--setting',choices=['raw','semantic_only'],default='raw')
    p.add_argument('--background',type=Path,required=True,help='Local MNI anatomical template; not bundled')
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from nilearn import plotting
    import nibabel as nib
    import numpy as np
    root=Path(__file__).resolve().parents[1];stem=f'{a.setting}_{a.contrast}'
    img=nib.load(root/'data/searchlight/maps'/f'{stem}_mean_delta_RSA.nii.gz')
    mask=nib.load(root/'data/searchlight/maps'/f'{stem}_significant_FWER.nii.gz')
    values=np.where(mask.get_fdata()>0,img.get_fdata(),0)
    view=nib.Nifti1Image(values,img.affine,img.header)
    out=root/'outputs/spatial_maps';out.mkdir(parents=True,exist_ok=True)
    display=plotting.plot_stat_map(view,bg_img=a.background,display_mode='z',cut_coords=12,threshold=1e-12,colorbar=True,title=f'{a.setting}: {a.contrast}, joint FWER < 0.05')
    display.savefig(out/f'{stem}.png',dpi=300);display.close();print(out)

if __name__=='__main__':main()
