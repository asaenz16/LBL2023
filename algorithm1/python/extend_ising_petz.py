"""Longer independent finite-sweep checks on the 2x64 Ising circuit channel."""
import os
for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from ising_petz import build_ising_petz,alternating_image,ladder_gibbs
from ising_petz_populations import population_trajectories
from run_ising_petz import OUTPUT,code_hashes,write_json


def run(k):
    output=OUTPUT/'long';output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();model=build_ising_petz(alternating_image(64),k=k)
    checkpoints=[0,50,100,200,500,1000,2000];shots=131072 if k==.8 else 65536
    r=population_trajectories(model,checkpoints,shots=shots,seeds=[0,2],rng_seed=9932100+int(100*k))
    target=ladder_gibbs(model,[0,2]);truth=model.graph.image.ravel()>0;align=np.where(truth,-1,1)
    rows=[]
    for i,s in enumerate(checkpoints):
        g=r['gamma'][i,0];pred=g<=0
        rows.append(dict(sweep=s,gamma_linf=float(np.max(abs(r['gamma'][i]-target))),
            far_aligned=float(-g[-1]),far_se=float(r['se'][i,0,-1]),
            iou=float(np.sum(pred&truth)/max(1,np.sum(pred|truth))),
            accuracy=float(np.mean(pred==truth)),
            correct_resolved_fraction=float(np.mean(g*align>r['simultaneous_radius']))))
    np.savez_compressed(output/f'L64-K{k:g}.npz',gamma_gibbs=target,**r)
    result=dict(k=k,b=model.b,width=64,shots=shots,initial='plus_x',records=rows,
        exact_gibbs_far=float(-target[0,-1]),radius=r['simultaneous_radius'],
        code_hashes=code_hashes(),seconds=time.perf_counter()-start)
    write_json(output/f'L64-K{k:g}.json',result);return result


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run,[.8,1.2]):print(json.dumps(result,indent=2),flush=True)
