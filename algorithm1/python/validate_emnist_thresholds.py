"""Independent exact Gibbs checks of all threshold rules on EMNIST patches."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
from ising_algorithm1 import ising_model, gibbs_ising
from ising_monte_carlo import signed_image, sample_gibbs
from ising_thresholds import RULES, K_GRID, ALPHA, rule_name, normalized_model
from run_emnist_thresholds import ROOT, OUTPUT


def main():
    images=np.load(OUTPUT/'subset.npz')['images'];rows=[]
    for idx in (0,23,46):
        image=images[idx]
        windows=np.lib.stride_tricks.sliding_window_view(image,(3,3))
        r,c=np.unravel_index(windows.var(axis=(-2,-1)).argmax(),windows.shape[:2])
        patch=windows[r,c].copy()
        for j,rule in enumerate(RULES):
            full,bs=normalized_model(image,rule,K_GRID)
            model=signed_image(patch,ALPHA,threshold=full.threshold)
            exact=ising_model(patch,-ALPHA,threshold=full.threshold)
            result=sample_gibbs(model,bs,seeds=[0,4,8],burn=1024,samples=4096,rng_seed=998721+idx*100+j)
            truth=np.array([gibbs_ising(exact,9*b).gamma[[0,4,8]] for b in bs])
            error=abs(result['gamma']-truth)
            rows.append(dict(class_id=idx,patch_origin=[int(r),int(c)],image=patch.tolist(),
                rule=rule_name(rule),tau=full.threshold,bs=bs.tolist(),
                max_error=float(error.max()),outside_6se_plus002=int(np.sum(error>6*result['se']+.002)),
                max_rhat=float(result['rhat_observables'].max())))
    summary=dict(description='512-state exact enumeration vs Monte Carlo; thresholds and temperatures from full training image',
        patches=3,parameter_points=len(rows)*len(K_GRID),max_error=max(r['max_error'] for r in rows),
        outside_6se_plus002=sum(r['outside_6se_plus002'] for r in rows),checks=rows)
    assert summary['max_error']<.015 and summary['outside_6se_plus002']==0
    (ROOT/'emnist-thresholds-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
