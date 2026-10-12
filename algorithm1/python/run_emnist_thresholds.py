"""Calibrate global thresholds, freeze selection, then sample fresh EMNIST tests.

Commands: prepare, train, freeze, test. Each has an immutable cached protocol.
Prior training images are reused; all previously inspected test images are
excluded by official index and content hash. No test outcomes enter selection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'): os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from fetch_emnist import load_idx
from run_emnist_ising import choose_seeds, proxy_metrics, image_features
from ising_monte_carlo import sample_gibbs
from ising_thresholds import ALPHA, K_REFERENCE, K_GRID, RULES, rule_name, normalized_model

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results'/'emnist-thresholds'


def write_once(path, value):
    value=json.loads(json.dumps(value,allow_nan=False))
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError(f'Conflicting saved protocol: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def hashes():
    names=['ising_thresholds.py','ising_monte_carlo.py','ising_algorithm1.py',
           'run_emnist_thresholds.py','run_emnist_ising.py','fetch_emnist.py']
    return {n:hashlib.sha256((ROOT/'python'/n).read_bytes()).hexdigest() for n in names}


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():
        if not (output/'subset.npz').exists(): raise ValueError('Missing subset.')
        return
    old=ROOT/'results'/'emnist-ising';data=ROOT/'results'/'emnist-data'
    manifest=json.loads((old/'manifest.json').read_text())
    oldimages=np.load(old/'subset.npz')['images']
    cases=[c for c in manifest['cases'] if c['split']=='train']
    images=[im for c,im in zip(manifest['cases'],oldimages) if c['split']=='train']
    excluded_indices={c['idx_index'] for c in manifest['cases'] if c['split']=='test'}
    excluded_hashes={c['image_sha256'] for c in manifest['cases']}
    x=load_idx(data/'emnist-balanced-test-images-idx3-ubyte.gz')
    y=load_idx(data/'emnist-balanced-test-labels-idx1-ubyte.gz')
    mapping={int(a):chr(int(b)) for a,b in np.loadtxt(data/'emnist-balanced-mapping.txt',dtype=int)}
    rng=np.random.default_rng(20261009)
    for label in range(47):
        for index in rng.permutation(np.flatnonzero(y==label)):
            image=x[index]; sha=hashlib.sha256(image.tobytes()).hexdigest()
            if int(index) in excluded_indices or sha in excluded_hashes: continue
            excluded_hashes.add(sha)
            cases.append(dict(name=f'fresh-test-class{label:02d}',split='test',class_id=label,
                character=mapping[label],idx_index=int(index),image_sha256=sha,
                seeds=choose_seeds(image).tolist(),foreground_fraction=float(np.mean(image>=128))))
            images.append(image);break
        else: raise ValueError('No unseen image available.')
    np.savez_compressed(output/'subset.npz',images=np.asarray(images))
    write_once(output/'manifest.json',dict(source=manifest['source'],cases=cases,
        training='Same 47 training images as prior Ising calibration',test_seed=20261009,
        test_exclusion='All 47 previously inspected test indices and all prior 94 image hashes',
        previous_manifest_sha256=hashlib.sha256((old/'manifest.json').read_bytes()).hexdigest()))


def job(spec):
    case,image,rule,ks,options,path=spec
    model,bs=normalized_model(image,rule,ks)
    signature=dict(rule=rule,ks=ks,alpha=ALPHA,tau=model.threshold,bs=bs.tolist(),
        options=options,image_sha256=case['image_sha256'],seeds=case['seeds'],code_sha256=hashes())
    meta_path=path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['signature']!=signature: raise ValueError(f'Conflicting cache: {path}')
        return meta
    start=time.perf_counter()
    result=sample_gibbs(model,bs,seeds=case['seeds'],**options)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k:v.astype(np.float32) if k in ('blocks','series','energy_blocks') else v
                                for k,v in result.items()})
    features=image_features(image,model)
    positive=model.distances>0
    features['active_margin_median']=float(np.median(abs(model.threshold-model.distances[positive]))) if positive.any() else model.threshold
    truth=image.ravel()>=128
    within_ink=truth[model.edges[:,0]]&truth[model.edges[:,1]]
    features['within_ink_negative_fraction']=float(np.mean(model.delta[within_ink]<0)) if within_ink.any() else 0.
    records=[]
    for t,(k,b) in enumerate(zip(ks,bs)):
        gamma,se=result['gamma'][t],result['se'][t]
        metrics=proxy_metrics(gamma[0],se[0],image,case['seeds'][0])
        ink=proxy_metrics(gamma[1],se[1],image,case['seeds'][1])
        rh=result['rhat_gamma'][t]
        # Primary foreground mask = Gamma(background,p)<=0.
        bg_prediction=gamma[0]<=0; ink_prediction=gamma[1]>0
        records.append(dict(name=case['name'],class_id=case['class_id'],split=case['split'],
            rule=rule_name(rule),tau=model.threshold,k=k,alpha=ALPHA,b=float(b),beta_paper=float(b*image.size),
            **metrics,ink_seed_iou=ink['iou'],ink_seed_resolved_fraction=ink['resolved_fraction'],
            seed_prediction_agreement=float(np.mean(bg_prediction==ink_prediction)),
            iou_threshold96=proxy_metrics(gamma[0],se[0],image,case['seeds'][0],96)['iou'],
            iou_threshold160=proxy_metrics(gamma[0],se[0],image,case['seeds'][0],160)['iou'],
            rhat_observable_max=float(result['rhat_observables'][t].max()),
            rhat_gamma_p99=float(np.quantile(rh[0],.99)),rhat_gamma_max=float(rh[0].max()),
            ess_energy=float(result['ess_observables'][t,0]),ess_mean_gamma=float(result['ess_observables'][t,1]),
            chain_gamma_linf=float(result['chain_linf'][t,0])))
    meta=dict(signature=signature,case=case,features=features,records=records,
        seconds=time.perf_counter()-start,ladder_size=len(result['ladder']),
        swap_min=float(result['swap_acceptance'].min()),round_trips=int(result['round_trips'].sum()))
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def load_runs(output,stage):
    return [json.loads(p.read_text()) for p in sorted((output/stage).glob('*.json'))]


def freeze(output):
    runs=load_runs(output,'train');table=[r for m in runs for r in m['records']]
    cases=json.loads((output/'manifest.json').read_text())['cases']
    expected={c['name'] for c in cases if c['split']=='train'}
    scores=[]
    for rule in RULES:
        for k in K_GRID:
            rows=[r for r in table if r['rule']==rule_name(rule) and r['k']==k]
            if len(rows)!=len(expected) or {r['name'] for r in rows}!=expected:
                raise ValueError('Complete every training rule and temperature before selection.')
            scores.append(dict(rule=rule,k=k,mean_iou=float(np.mean([r['iou'] for r in rows])),
                mean_resolved=float(np.mean([r['resolved_fraction'] for r in rows])),
                max_rhat=max(r['rhat_observable_max'] for r in rows),
                rhat_over1_05=sum(r['rhat_observable_max']>1.05 for r in rows)))
    selections={}
    for name,filter_rule in [('baseline',lambda r:r==dict(kind='fixed',value=125.)),
                             ('global',lambda r:r['kind']=='fixed'),
                             ('quantile',lambda r:r['kind']=='quantile'),
                             ('otsu',lambda r:r['kind']=='otsu')]:
        eligible=[s for s in scores if filter_rule(s['rule'])]
        # All candidates retained; diagnostics explicitly reported, never hidden.
        selections[name]=max(eligible,key=lambda s:s['mean_iou'])
    frozen=dict(selection_metric='Mean training foreground IoU against intensity>=128 proxy',
        n_train=len(expected),alpha=ALPHA,k_reference=float(K_REFERENCE),k_grid=K_GRID,
        candidates=scores,selections=selections,
        training_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((output/'train').glob('*.json'))},
        test_policy='Use only these preselected rules and K values on fresh test images; no retuning',
        code_sha256=hashes())
    write_once(output/'frozen-selection.json',frozen)
    print(json.dumps(selections,indent=2),flush=True)


def run(args):
    prepare(args.output)
    if args.stage=='prepare': return
    if args.stage=='freeze': freeze(args.output);return
    stage=args.stage
    if stage=='train':
        rules_and_ks=[(r,K_GRID) for r in RULES]
        options=dict(burn=512,samples=1024,block_size=64,chains=2)
    else:
        frozen=json.loads((args.output/'frozen-selection.json').read_text())
        if frozen['code_sha256']!=hashes(): raise ValueError('Sampler changed after selection.')
        choices={}
        for selected in frozen['selections'].values():
            name=rule_name(selected['rule'])
            if name not in choices: choices[name]=(selected['rule'],set())
            choices[name][1].add(selected['k'])
        # Preserve the original thermal scale as an additional baseline check.
        choices['fixed-125'][1].add(float(K_REFERENCE))
        rules_and_ks=[(r,sorted(ks)) for r,ks in choices.values()]
        options=dict(burn=1024,samples=2048,block_size=64,chains=4)
    write_once(args.output/f'{stage}-protocol.json',dict(stage=stage,rules_and_ks=rules_and_ks,
        **options,alpha=ALPHA,model='H=-sum atan(alpha*(tau-d))*Zi*Zj/4; open grid',
        beta='b=beta_paper/N=4*K/median(abs(Delta))',mask='intensity>=128 evaluation proxy; sensitivity96,160',
        seed='darkest perimeter; brightest pixel; neither pinned',code_sha256=hashes()))
    manifest=json.loads((args.output/'manifest.json').read_text())
    images=np.load(args.output/'subset.npz')['images'];jobs=[]
    for i,(case,image) in enumerate(zip(manifest['cases'],images)):
        if case['split']!=('train' if stage=='train' else 'test'):continue
        if args.classes is not None and case['class_id'] not in args.classes:continue
        for rule,ks in rules_and_ks:
            key=rule_name(rule)
            seed=20271007+10000*i+int(hashlib.sha256(key.encode()).hexdigest()[:6],16)
            path=args.output/stage/f'{case["name"]}-{key}.npz'
            jobs.append((case,image,rule,ks,dict(options,rng_seed=seed),path))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,spec) for spec in jobs]
        for i,f in enumerate(as_completed(futures),1):
            meta=f.result()
            if i%10==0 or i in (1,len(jobs)):
                print(json.dumps(dict(done=i,total=len(jobs),elapsed=round(time.perf_counter()-start,1),
                    last=meta['case']['name'],rule=rule_name(meta['signature']['rule']),
                    seconds=round(meta['seconds'],1),max_rhat=max(r['rhat_observable_max'] for r in meta['records']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','freeze','test'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--classes',type=int,nargs='+')
    run(p.parse_args())
