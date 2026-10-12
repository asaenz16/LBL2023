"""Exact-vs-Monte-Carlo validation on contrast-rich 3x3 EMNIST training patches."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from ising_algorithm1 import ising_model,gibbs_ising
from ising_monte_carlo import signed_image,sample_gibbs

ROOT=Path(__file__).resolve().parents[1]


def validate():
    root=ROOT/'results'/'emnist-ising';images=np.load(root/'subset.npz')['images']
    cases=json.loads((root/'manifest.json').read_text())['cases'];results=[]
    for idx in [0,12,24,36]:
        image=images[idx]
        # Choose the highest-variance 3x3 patch without reference segmentation.
        windows=np.lib.stride_tricks.sliding_window_view(image,(3,3))
        r,c=np.unravel_index(np.argmax(windows.var(axis=(-2,-1))),windows.shape[:2])
        patch=windows[r,c].copy()
        for alpha,bs in [(.001,[16.,20.,32.]),(.003,[4.,8.,16.]),(.03,[1.,2.,4.])]:
            exact=ising_model(patch,-alpha)
            mc=sample_gibbs(signed_image(patch,alpha),bs,seeds=[0,4,8],burn=1024,samples=4096,
                            rng_seed=8823+idx+int(alpha*1e6))
            truth=np.array([gibbs_ising(exact,b*9).gamma[[0,4,8]] for b in bs])
            error=abs(mc['gamma']-truth);se=mc['se']
            results.append(dict(case=cases[idx]['name'],patch_origin=[int(r),int(c)],image=patch.tolist(),
                alpha=alpha,bs=bs,max_abs_gamma_error=float(error.max()),
                rmse=float(np.sqrt(np.mean(error**2))),
                max_error_over_se=float(np.max(error/np.maximum(se,1e-12))),
                outside_6se_plus002=int(np.sum(error>6*se+.002)),
                max_rhat=float(mc['rhat_observables'].max())))
    summary=dict(method='Exact enumeration of 512 states vs independent Monte Carlo on 4 EMNIST patches',
        patch_count=4,parameter_points=36,checks=results,
        max_error=max(r['max_abs_gamma_error'] for r in results),
        outside_6se_plus002=sum(r['outside_6se_plus002'] for r in results))
    assert summary['outside_6se_plus002']==0
    assert summary['max_error']<.015
    (ROOT/'emnist-ising-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':validate()
