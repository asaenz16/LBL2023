"""Frozen-split Gibbs calibration and finite-sweep local-circuit benchmark."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import itertools
import json
import time
import numpy as np
from ising_petz import build_ising_petz
from ising_petz_populations import population_trajectories
from ising_grid_transfer import exact_grid_gibbs
from weizmann_petz_data import ROOT,OUT,seed_for,write_json
from weizmann_petz_metrics import (segmentation,mask_metrics,classical_baselines,
    coupling_diagnostics,correlation_metrics)

DEFAULT=dict(threshold=100.,alpha=.001,k=.8)


def hashes():
    names=['ising_petz.py','ising_petz_populations.py','ising_grid_transfer.py',
           'weizmann_petz_data.py','weizmann_petz_metrics.py','run_weizmann_petz.py']
    return {n:hashlib.sha256((ROOT/'python'/n).read_bytes()).hexdigest() for n in names}


def manifest():return json.loads((ROOT/'weizmann-petz-manifest.json').read_text())


def read_case(case):
    return np.load(OUT/'cases'/f'{case["case_id"]}.npz')


def calibration_job(spec):
    case,params=spec;f=read_case(case);model=build_ising_petz(f['image'],**params)
    ref=exact_grid_gibbs(model,f['seeds']);metrics=correlation_metrics(ref,ref,f['truth'],f['seeds'])
    return dict(case_id=case['case_id'],source_id=case['source_id'],params=params,**metrics)


def calibrate(workers):
    cases=[c for c in manifest()['cases'] if c['split']=='calibration' and c['family']=='natural-square']
    candidates=[dict(threshold=float(t),alpha=a,k=k) for t,a,k in itertools.product(
        (10,25,50,75,100),(.001,.01),(.4,.8,1.2))]
    rows=[];specs=[(c,p) for p in candidates for c in cases]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(calibration_job,s) for s in specs]
        for i,f in enumerate(as_completed(futures),1):
            rows.append(f.result())
            if i%100==0:print(f'Calibration {i}/{len(specs)}',flush=True)
    scores=[]
    for p in candidates:
        selected=[r for r in rows if r['params']==p]
        assert len(selected)==50
        scores.append(dict(params=p,mean_iou=float(np.mean([r['iou'] for r in selected])),
            mean_numerically_resolved=float(np.mean([r['resolved_fraction'] for r in selected]))))
    scores.sort(key=lambda r:(-r['mean_iou'],r['params']!=DEFAULT,r['params']['k'],r['params']['alpha'],r['params']['threshold']))
    result=dict(selected=scores[0]['params'],objective='Exact Gibbs primary-seed IoU; calibration squares only',
        source_ids=manifest()['calibration_ids'],scores=scores,records=sorted(rows,key=lambda r:(str(r['params']),r['case_id'])),code_hashes=hashes())
    write_json(ROOT/'weizmann-petz-calibration.json',result)
    print(json.dumps(dict(selected=result['selected'],best=scores[0],original=next(s for s in scores if s['params']==DEFAULT))),flush=True)


def benchmark_job(spec):
    case,name,params,shots,checkpoints,initial,folder=spec
    path=OUT/folder/f'{case["case_id"]}--{name}--{initial}.json'
    signature=dict(case_id=case['case_id'],parameter_name=name,params=params,shots=shots,
        checkpoints=checkpoints,initial=initial,rng_seed=seed_for(case['case_id']+name+initial+folder),code_hashes=hashes(),
        input_sha256=hashlib.sha256((OUT/'cases'/f'{case["case_id"]}.npz').read_bytes()).hexdigest())
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['signature']!=signature:raise ValueError('Changed cached run: '+str(path))
        return cached
    start=time.perf_counter();f=read_case(case);image,truth,seeds=f['image'],f['truth'],f['seeds']
    model=build_ising_petz(image,**params);ref=exact_grid_gibbs(model,seeds)
    r=population_trajectories(model,checkpoints,shots=shots,seeds=seeds,
        rng_seed=signature['rng_seed'],initial=initial)
    # Unlike the older one-seed report, this bound covers BOTH seeds at each time.
    radius=float(np.sqrt(np.log(2*model.n*len(seeds)/.05)/(8*shots)))
    records=[]
    for t,sweep in enumerate(checkpoints):
        record=correlation_metrics(r['gamma'][t],ref,truth,seeds,radius)
        record['sweep']=sweep;records.append(record)
    baselines=classical_baselines(image,truth,seeds)
    exact=correlation_metrics(ref,ref,truth,seeds)
    prediction=segmentation(ref[0],truth.ravel()[seeds[0]])
    exact['human_iou_each']=[mask_metrics(prediction,m)['iou'] for m in f['masks']]
    result=dict(signature=signature,case=case,b=model.b,beta_paper=model.b*model.n,
        gibbs=exact,baselines={k:mask_metrics(p,truth) for k,p in baselines.items()},
        diagnostics=coupling_diagnostics(model,truth),simultaneous_radius=radius,records=records,
        seconds=time.perf_counter()-start,engine='Independent populations of the gate-validated Ising Petz channel')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),image=image,truth=truth,seeds=seeds,
        gamma_gibbs=ref,gamma=r['gamma'],se=r['se'],sweeps=checkpoints,
        otsu=baselines['otsu'],random_walker=baselines['random_walker'])
    write_json(path,result);return result


def study(workers,limit=None):
    calibrated=json.loads((ROOT/'weizmann-petz-calibration.json').read_text())
    if calibrated['code_hashes']!=hashes():raise ValueError('Calibration source changed.')
    choices=[('original',DEFAULT),('calibrated',calibrated['selected'])]
    specs=[]
    for case in manifest()['cases']:
        control=case['family']=='natural-strip' and case['category']!='boundary'
        options=choices[:1] if control else choices
        for name,params in options:
            specs.append((case,name,params,2048 if control else 4096,
                [0,1,5,20,50] if control else [0,1,5,20,50,100,200,500],'plus_x','study'))
    if limit:specs=specs[:limit]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(benchmark_job,s) for s in specs]
        for i,f in enumerate(as_completed(futures),1):
            row=f.result()
            if i%25==0 or i==len(specs):
                print(json.dumps(dict(done=i,total=len(specs),case=row['case']['case_id'],
                    parameters=row['signature']['parameter_name'],final=row['records'][-1])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibrate','study'])
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.action=='calibrate':calibrate(args.workers)
    else:study(args.workers,args.limit)
