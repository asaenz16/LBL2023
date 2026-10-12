"""Weizmann one-seed versus max-correlation two-seed Ising Gibbs study."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import argparse
import hashlib
import io
import itertools
import json
import time
import zipfile
import numpy as np
from PIL import Image
from ising_seed_sampling import (FUNCTIONS,DECISIONS,parameterize,setup_sampler,exact_correlations,
    sample_spins,sufficient_counts,evaluate_counts,exact_decision,segmentation_metrics)
from weizmann_petz_data import ROOT,OUT as OLD,ARCHIVE_HASH,seed_for,write_json,decode_annotation,resize_image,resize_mask,seed_pixels
from weizmann_petz_metrics import classical_baselines,mask_metrics

OUT=ROOT/'results'/'weizmann-ising-seeds'
BUDGETS=[1536,6144,24576]
PRIMARY=24576
METRICS=['iou','dice','accuracy','unresolved','correct_resolved','wrong_resolved','iou_lower','iou_upper','unknown_iou','seed_accuracy']


def identifier(rule):return hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()[:12]


def hashes():
    names=['ising_seed_sampling.py','run_weizmann_ising_seeds.py','ising_thresholds.py','ising_algorithm1.py','weizmann_petz_data.py','weizmann_petz_metrics.py']
    return {n:hashlib.sha256((ROOT/'python'/n).read_bytes()).hexdigest() for n in names}


def candidates():
    thresholds=[dict(kind='fixed',value=float(t)) for t in (25,50,75,100,125)]
    thresholds += [dict(kind='quantile',value=q) for q in (.5,.75,.9)]
    thresholds += [dict(kind='otsu',positive_only=False)]
    profiles=[(f,a) for f in FUNCTIONS for a in (['adaptive'] if f=='linear' else ['a001','a01','adaptive'])]
    return [dict(function=f,sensitivity=a,threshold=t,K=k) for (f,a),t,k in itertools.product(profiles,thresholds,(.4,.8,1.6))]


def manifest():return json.loads((ROOT/'weizmann-ising-seeds-manifest.json').read_text())


def read_case(case):
    p=OUT/'cases'/f'{case["case_id"]}.npz'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=case['input_sha256']:raise ValueError('Input hash changed.')
    return np.load(p)


def prepare():
    if (ROOT/'weizmann-ising-seeds-manifest.json').exists():
        for c in manifest()['cases']:read_case(c).close()
        print('Existing input hashes verified.',flush=True);return
    old=json.loads((ROOT/'weizmann-petz-manifest.json').read_text());archive=OLD/'data'/'Weizmann_Seg_DB_1obj.ZIP'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_HASH:raise ValueError('Unexpected archive.')
    cases=[];excluded=[]
    with zipfile.ZipFile(archive) as z:
        for source in old['sources']:
            image=np.array(Image.open(io.BytesIO(z.read(source['image_file']))).convert('L'))
            masks=np.stack([decode_annotation(np.array(Image.open(io.BytesIO(z.read(a))).convert('RGB'))) for a in source['annotation_files']])
            for side in (8,16):
                name=f'{source["source_id"]}--square-{side}';small=resize_image(image,side);small_masks=np.stack([resize_mask(m,side) for m in masks])
                truth=small_masks.sum(axis=0)>=2;seeds=seed_pixels(truth)
                c=dict(case_id=name,source_id=source['source_id'],split=source['split'],side=side,seeds=seeds)
                if len(seeds)!=2:excluded.append(dict(**c,reason='Only one class after resizing.'));continue
                path=OUT/'cases'/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,image=small,truth=truth,masks=small_masks,seeds=seeds)
                cases.append(dict(**c,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(ROOT/'weizmann-ising-seeds-manifest.json',dict(date='2026-10-08',archive_sha256=ARCHIVE_HASH,
        calibration_ids=old['calibration_ids'],test_ids=old['test_ids'],cases=cases,excluded=excluded,
        preprocessing='Full-frame grayscale BOX resize, individual NEAREST masks, three-annotation majority; oracle background and foreground interior seeds.',
        split='Existing 25/75 source split reused. Internal exploratory holdout, not fresh external validation.'))
    print('Prepared',len(cases),'cases; exclusions:',len(excluded),flush=True)


def initialize():
    protocol=dict(date='2026-10-08',model='H=-sum_e J_e ZiZj/4; rho=exp(-bH)/Z; paper beta=N*b. No clamped spins or seed labels.',
        functions='x=alpha*(tau-d), delta=atan(x). Functions: delta; delta+delta*abs(delta); delta+4*delta^3; x; x+x^3. All odd and sign-preserving.',
        parameters='tau=25,50,75,100,125, positive-edge q50/q75/q90, edge-Otsu. alpha=.001,.01,1/S with S=max(q90(d)-q10(d),1). Pure linear has only alpha=1/S because median normalization makes alpha redundant. K=.4,.8,1.6; b=4K/median|J|, near-zero median fallback max(mean,.001).',
        decisions='Primary one-seed: foreground iff background ZZ correlation <0. Two-seed: foreground iff foreground ZZ correlation > background ZZ correlation; signed values, no absolute value. Foreground-only sign rule is a secondary diagnostic. Ties -> background.',
        sampler='Exact logarithmic frontier conditional probabilities, then independent full-grid Gibbs samples. Shorter side <=16. O(N 2^width) preprocessing/storage and O(N B) sampling. Exact seed correlations from positive-seed conditioning and global spin-flip symmetry.',
        calibration='All rules on 25 8x8 calibration images, three independent 24576-copy replicates. Primary winners per coupling-function and decision maximize mean sampled IoU, then lower IoU bound, exact IoU, then fixed ID. Two additional robustness winners maximize lower IoU bound across all functions. Legacy arctan edge-Otsu/.01/K=.8 retained as fixed control.',
        test='All selected distinct Hamiltonians on every eligible 8x8 and16x16 test image, eight independent sampling replicates with nested budget prefixes. No 16x16 retuning. Every Hamiltonian evaluated with all three decisions using identical samples.',
        uncertainty='Exact Clopper-Pearson binomial intervals with Bonferroni 95 percent coverage across three decisions, all pixels and three budgets per Hamiltonian/replicate. Two-seed uses conditional sf*zi samples when sf!=sb, since Cf-Cb=2*P(sf!=sb)*E[sf*zi | sf!=sb]. No discord samples -> unresolved. Confidence concerns sign/winner, not segmentation correctness.',
        metrics='Primary full-mask foreground IoU, unresolved fraction among pixels excluding both seeds, correct-resolved and wrong-resolved fractions, uncertain-label IoU bounds, unseeded IoU, exact Gibbs IoU. Known seed labels are not imposed.',
        budgets=BUDGETS,primary_budget=PRIMARY,test_replicates=8,calibration_replicates=3,candidates=candidates(),
        code_hashes=hashes(),manifest_sha256=hashlib.sha256((ROOT/'weizmann-ising-seeds-manifest.json').read_bytes()).hexdigest())
    p=ROOT/'weizmann-ising-seeds-protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Frozen protocol changed.')
    write_json(p,protocol);print('Frozen',len(candidates()),'parameter rules.',flush=True)


def calibration_cases():return [c for c in manifest()['cases'] if c['split']=='calibration' and c['side']==8]


def calibrate_case(case):
    path=OUT/'calibration'/f'{case["case_id"]}.npz';signature=dict(case=case,code_hashes=hashes(),replicates=3,budget=PRIMARY)
    if path.exists():
        if json.loads(path.with_suffix('.json').read_text())['signature']!=signature:raise ValueError('Stale calibration.')
        return case['case_id']
    t=time.perf_counter();f=read_case(case);image,truth,seeds=f['image'],f['truth'],f['seeds'];rules=candidates()
    exact=np.zeros((len(rules),3,len(METRICS)));sampled=np.zeros_like(exact);corr=np.zeros((len(rules),2,image.size));parameters=[]
    for ri,rule in enumerate(rules):
        edges,j,p=parameterize(image,rule);parameters.append(p);sampler=setup_sampler(image.shape,edges,j,p['b']);c=exact_correlations(sampler,seeds);corr[ri]=c
        for di,decision in enumerate(DECISIONS):
            m=segmentation_metrics(*exact_decision(c,decision),truth,seeds);exact[ri,di]=[m[k] for k in METRICS]
        for rep in range(3):
            seed=seed_for('ising-seeds-cal:'+case['case_id']+identifier(rule)+str(rep))
            samples=sample_spins(sampler,PRIMARY,np.random.default_rng(seed));counts=sufficient_counts(samples,seeds,[PRIMARY])[0]
            values=evaluate_counts(counts,PRIMARY,truth,seeds)
            for di,decision in enumerate(DECISIONS):sampled[ri,di]+=[values[decision][k]/3 for k in METRICS]
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,exact_metrics=exact,sample_metrics=sampled,correlations=corr)
    write_json(path.with_suffix('.json'),dict(signature=signature,parameters=parameters,seconds=time.perf_counter()-t))
    return case['case_id']


def select():
    ex=[];sm=[]
    cases=calibration_cases()
    if len(cases)!=25:raise ValueError('Expected 25 calibration sources.')
    for c in cases:
        p=OUT/'calibration'/f'{c["case_id"]}.npz';meta=json.loads(p.with_suffix('.json').read_text())
        if meta['signature']!=dict(case=c,code_hashes=hashes(),replicates=3,budget=PRIMARY):raise ValueError('Stale calibration selection.')
        f=np.load(p);ex.append(f['exact_metrics']);sm.append(f['sample_metrics'])
    ex=np.mean(ex,axis=0);sm=np.mean(sm,axis=0);records=[]
    for ri,r in enumerate(candidates()):
        for di,decision in enumerate(DECISIONS):
            records.append(dict(rule=r,rule_id=identifier(r),decision=decision,
                exact=dict(zip(METRICS,ex[ri,di].tolist())),sample=dict(zip(METRICS,sm[ri,di].tolist()))))
    selected={}
    def choose(rows,robust=False):
        return sorted(rows,key=lambda x:(-x['sample']['iou_lower'] if robust else -x['sample']['iou'],
            -x['sample']['iou_lower'],-x['exact']['iou'],x['rule_id']))[0]
    for fn in FUNCTIONS:
        for decision in DECISIONS[:2]:selected[fn+'__'+decision]=choose([r for r in records if r['rule']['function']==fn and r['decision']==decision])
    for decision in DECISIONS[:2]:selected['robust__'+decision]=choose([r for r in records if r['decision']==decision],True)
    legacy=dict(function='arctan',sensitivity='a01',threshold=dict(kind='otsu',positive_only=False),K=.8)
    selected['legacy_arctan']=next(r for r in records if r['rule']==legacy and r['decision']=='one_background')
    write_json(ROOT/'weizmann-ising-seeds-selection.json',dict(winners=selected,code_hashes=hashes()))
    write_json(ROOT/'weizmann-ising-seeds-calibration.json',dict(scores=records,code_hashes=hashes()))
    for label,row in selected.items():print(label,row['rule'],'IoU',round(row['sample']['iou'],4),'unresolved',round(row['sample']['unresolved'],4),flush=True)


def chosen_rules():
    selected=json.loads((ROOT/'weizmann-ising-seeds-selection.json').read_text())['winners'];groups=defaultdict(list)
    for label,row in selected.items():groups[row['rule_id']].append(label)
    return [(next(r['rule'] for r in selected.values() if r['rule_id']==key),labels) for key,labels in groups.items()]


def test_case(case):
    path=OUT/'test'/f'{case["case_id"]}.json';chosen=chosen_rules()
    signature=dict(case=case,code_hashes=hashes(),selected=chosen,budgets=BUDGETS,replicates=8)
    # JSON converts tuple pairs to lists.
    signature=json.loads(json.dumps(signature))
    if path.exists():
        if json.loads(path.read_text())['signature']!=signature:raise ValueError('Stale test case.')
        return case['case_id']
    t=time.perf_counter();f=read_case(case);image,truth,seeds=f['image'],f['truth'],f['seeds'];records=[];arrays={}
    for rule,labels in chosen:
        rid=identifier(rule);edges,j,p=parameterize(image,rule);sampler=setup_sampler(image.shape,edges,j,p['b']);c=exact_correlations(sampler,seeds)
        exact={d:segmentation_metrics(*exact_decision(c,d),truth,seeds) for d in DECISIONS}
        counts=[];values=[]
        for rep in range(8):
            seed=seed_for('ising-seeds-test:'+case['case_id']+rid+str(rep))
            z=sample_spins(sampler,max(BUDGETS),np.random.default_rng(seed));v=sufficient_counts(z,seeds,BUDGETS);counts.append(v)
            values.append([evaluate_counts(v[bi],b,truth,seeds) for bi,b in enumerate(BUDGETS)])
        result={}
        for d in DECISIONS:
            result[d]=dict(exact=exact[d],budgets={str(b):dict(mean={k:float(np.mean([v[bi][d][k] for v in values])) for k in METRICS},
                sd={k:float(np.std([v[bi][d][k] for v in values],ddof=1)) for k in METRICS}) for bi,b in enumerate(BUDGETS)})
        counts=np.asarray(counts);arrays[rid+'_counts']=counts;arrays[rid+'_correlations']=c
        records.append(dict(rule=rule,rule_id=rid,selected_as=labels,parameters=p,decisions=result,
            exact_seed_disagreement=float((1-c[0,seeds[1]])/2),sample_seed_disagreement=float(counts[:,-1,3,0].mean()/PRIMARY)))
    baseline={k:mask_metrics(v,truth) for k,v in classical_baselines(image,truth,seeds).items()}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    write_json(path,dict(signature=signature,case=case,records=records,baselines=baseline,seconds=time.perf_counter()-t))
    return case['case_id']


def dispatch(cases,fn,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fn,c) for c in cases]
        for i,f in enumerate(as_completed(futures),1):print(json.dumps(dict(done=i,total=len(cases),case=f.result())),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','init','calibrate','select','test'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='init':initialize();return
    protocol=json.loads((ROOT/'weizmann-ising-seeds-protocol.json').read_text())
    if protocol['code_hashes']!=hashes():raise ValueError('Frozen experiment code changed.')
    if protocol['manifest_sha256']!=hashlib.sha256((ROOT/'weizmann-ising-seeds-manifest.json').read_bytes()).hexdigest():raise ValueError('Frozen input manifest changed.')
    if args.stage=='select':select();return
    if args.stage=='calibrate':dispatch(calibration_cases(),calibrate_case,args.workers);return
    dispatch([c for c in manifest()['cases'] if c['split']=='test'],test_case,args.workers)


if __name__=='__main__':main()
