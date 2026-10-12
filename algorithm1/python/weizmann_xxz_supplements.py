"""Secondary raw-IoU selection and exact native-crop calibration.

These supplementary analyses separate numerical sign resolution from the
underlying Gibbs target. They do not replace the frozen primary selection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import run_weizmann_xxz_v2 as study
from xxz_image_gibbs import parameterize,evaluate
from xxz_glauber import graph,exact_gibbs
from weizmann_petz_data import write_json


def raw_job(spec):
    study.SETTINGS['raw']=study.SETTINGS['test']
    return study.run_job(spec)


def raw_rules():
    selected=json.loads((study.ROOT/'weizmann-xxz-v2-selection.json').read_text())
    return {m:max([r for r in selected['refined_scores'] if r['family'].startswith(m+'/')],
        key=lambda r:(r['mean_iou'],r['mean_lower_iou'])) for m in ('Ising','XXZ')}


def exact_case(case,rules):
    f=np.load(study.INPUT/'cases'/f'{case["case_id"]}.npz');image=f['image'];seeds=f['seeds'];truth=f['truth']
    basis=np.arange(256);z=1-2*((basis[:,None]>>np.arange(7,-1,-1))&1)
    spectra={};rows=[]
    for rule in rules:
        m,p=parameterize(image,rule);key=tuple(rule[k] for k in ['threshold','sensitivity','transverse','strength'])
        if key not in spectra:
            h=np.zeros((256,256))
            for (i,j),delta in zip(m.edges,m.delta):
                h[basis,basis]-=delta*z[:,i]*z[:,j]/4
                chosen=basis[z[:,i]!=z[:,j]]
                h[chosen^(1<<(7-i))^(1<<(7-j)),chosen]-=p['g']/2
            if p['g']==0:energies=np.diag(h);probability_basis=np.eye(256)
            else:
                energies,vectors=np.linalg.eigh(h);probability_basis=vectors**2
            spectra[key]=(energies,probability_basis)
        energies,probability_basis=spectra[key]
        weights=np.exp(-p['b']*(energies-energies.min()));weights/=weights.sum()
        prob=probability_basis@weights;gamma=np.array([(prob*z[:,s])@z/4 for s in seeds])
        rows.append(dict(case=case,rule=rule,rule_id=study.identifier(rule),parameters=p,
            metrics=evaluate(gamma,0.,truth,seeds)))
    return rows


def small_exact(workers):
    rules=study.candidates();data=study.manifest()
    cases=[c for c in data['cases'] if c['family']=='natural-strip' and c['width']==4 and c['category']=='boundary']
    training=[c for c in cases if c['split']=='calibration'];records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,rows in enumerate(pool.map(exact_case,training,[rules]*len(training)),1):
            records.extend(rows);print('Exact crop calibration',i,'/',len(training),flush=True)
    grouped=defaultdict(list)
    for r in records:grouped[r['rule_id']].append(r)
    scores=[dict(rule_id=rid,rule=items[0]['rule'],mean_iou=float(np.mean([r['metrics']['iou'] for r in items])))
        for rid,items in grouped.items()]
    scores.sort(key=lambda r:(-r['mean_iou'],r['rule_id']))
    selected={m:next(r for r in scores if (r['rule']['strength']==0)==(m=='Ising')) for m in ('Ising','XXZ')}
    testing=[c for c in cases if c['split']=='test'];test=[]
    # Include the strongest exact-calibration rule at each relative g, with
    # all other hyperparameters selected only on the 25 calibration sources.
    strength_winners=[]
    for strength in (0.,.25,.5,1.,2.):
        strength_winners.append(next(r for r in scores if r['rule']['transverse']=='relative' and r['rule']['strength']==strength))
    chosen={r['rule_id']:r['rule'] for r in list(selected.values())+strength_winners}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(exact_case,testing,[list(chosen.values())]*len(testing)):test.extend(rows)
    result=dict(status='complete',calibration_sources=25,test_sources=75,calibration_evaluations=len(records),
        test_evaluations=len(test),selection=selected,strength_winners=strength_winners,scores=scores,test=test,
        method='Full 256-dimensional Gibbs diagonalization; eigenvectors reused across beta. Ising uses exact diagonal energies.',
        selection_objective='Raw exact-Gibbs foreground IoU on native 2x4 calibration crops; supplementary domain-specific calibration.',
        code_hashes=study.hashes(),supplement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(study.ROOT/'weizmann-xxz-v2-exact-crops.json',result)
    print(json.dumps(selected,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['raw','exact-crops'])
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    metadata=dict(date='2026-10-08',status='specified before supplementary runs',
        purpose='Compare raw-IoU and uncertainty-aware calibration; obtain an exact domain-specific crop benchmark without Monte Carlo sign error.',
        raw_selection='Highest mean raw IoU among the already-refined calibration candidates for each model. No test-driven tuning.',
        crop_selection='All 360 original candidate rules on the 25 native 2x4 calibration crops, exact Gibbs; frozen winners evaluated on 75 comparison crops.',
        code_hashes=study.hashes(),supplement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=study.ROOT/'weizmann-xxz-v2-supplement-protocol.json'
    if path.exists() and json.loads(path.read_text())!=metadata:raise ValueError('Supplement protocol changed.')
    write_json(path,metadata)
    if args.action=='exact-crops':small_exact(args.workers);return
    selected=raw_rules();write_json(study.ROOT/'weizmann-xxz-v2-raw-selection.json',selected)
    cases=[c for c in study.manifest()['cases'] if c['split']=='test' and c['family']=='natural-square']
    specs=[('raw',c,r['rule']) for c in cases for r in selected.values()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(raw_job,s) for s in specs]
        for i,f in enumerate(as_completed(futures),1):
            row=f.result()
            if i%25==0 or i==len(specs):print('Raw-IoU secondary comparison',i,'/',len(specs),flush=True)


if __name__=='__main__':main()
