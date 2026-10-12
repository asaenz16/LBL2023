"""Frozen image-only rule comparison for Ising and XXZ Gibbs segmentation."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from xxz_image_gibbs import evaluate,parameterize
from xxz_image_gibbs_v2 import sample
from weizmann_petz_data import ROOT,OUT as INPUT,seed_for,write_json
from weizmann_petz_metrics import classical_baselines,mask_metrics,coupling_diagnostics
from types import SimpleNamespace
from ising_grid_transfer import exact_grid_gibbs

OUT=ROOT/'results'/'weizmann-xxz-v2'
SETTINGS={
    'screen':dict(burn=300,samples=512,thin=1,chains=2),
    'refine':dict(burn=1500,samples=2048,thin=2,chains=4),
    'test':dict(burn=2000,samples=4096,thin=2,chains=4),
    'small':dict(burn=1500,samples=4096,thin=2,chains=4),
    'scan':dict(burn=2000,samples=4096,thin=2,chains=4),
    'confirm':dict(burn=8000,samples=16384,thin=2,chains=4)}


def identifier(rule):return hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()[:12]


def candidates():
    choices=[]
    for threshold,sensitivity,k in itertools.product(
            ('fixed50','fixed100','q75','edge_otsu'),
            ('fixed001','fixed01','inverse_spread'),(.4,.8,1.2)):
        base=dict(threshold=threshold,sensitivity=sensitivity,K=k)
        choices.append(dict(**base,transverse='relative',strength=0.,temperature='longitudinal'))
        for strength,temp in itertools.product((.25,.5,1.,2.),('longitudinal','combined')):
            choices.append(dict(**base,transverse='relative',strength=strength,temperature=temp))
        choices.append(dict(**base,transverse='absolute',strength=1.,temperature='combined'))
    return choices


def manifest():return json.loads((ROOT/'weizmann-petz-manifest.json').read_text())


def hashes():
    names=['xxz_image_gibbs_v2.py','xxz_sse_cluster.py','xxz_image_gibbs.py','xxz_sse_fast.py','xxz_glauber.py','ising_grid_transfer.py',
           'ising_monte_carlo.py','ising_thresholds.py','weizmann_petz_metrics.py','run_weizmann_xxz_v2.py']
    return {n:hashlib.sha256((ROOT/'python'/n).read_bytes()).hexdigest() for n in names}


def family(rule):return ('Ising' if rule['strength']==0 else 'XXZ')+'/'+rule['threshold']


def protocol():
    return dict(date='2026-10-08',version=2,revision_reason='Pilot directed-loop-only chains mixed poorly for weak exchange. Add a validated conditional worldline-cluster update, include g/D=0.25 in calibration, and rerun every comparison in a separate output directory. Original pilot records are retained. This revision follows sampling diagnostics and is exploratory.',hamiltonian='H=-sum[g(XX+YY)+atan(alpha*(tau-d))ZZ]/4',
        data='Existing verified Weizmann single-object archive and masks; fixed source-level 25 calibration / 75 comparison split',
        reuse_limit='These 75 images appeared in prior Ising development. This is internal held-out comparison within the present protocol, not fresh external validation.',
        seeds='Same mask-derived background/foreground seeds, no spin clamp; primary background-seed Algorithm 1 signs',
        candidates=candidates(),sensitivity_rule='inverse_spread: alpha=1/max(q90(d)-q10(d),1)',
        exchange_rule='Relative g=r*median(abs(Delta)); absolute g=1 is a separate comparison',
        temperature_rule='longitudinal b=4K/median(abs(Delta)); combined b=4K/(median(abs(Delta))+g); beta_paper=N*b',
        screen='All 360 rules on 25 calibration images at 8x8. Two chains, short screen, never a final Gibbs claim.',
        refine='Within each model/threshold family retain the best two by mean uncertainty-lower IoU plus best by raw IoU; repeat on all 25 calibration 8x8 images with independent longer sampling.',
        selection='Highest mean uncertainty-lower IoU in each of eight model/threshold families, with failed Rhat cases scored zero; ties by raw IoU, then lexical rule ID. Also retain best overall Ising and XXZ.',
        objective='Lower IoU over all assignments of signs whose approximate simultaneous MCMC interval includes zero; not a rigorous finite-time MCMC confidence bound.',
        comparison='Each family winner on all 75 comparison images at 8x8 and 16x16; report every case including convergence failures. Classical baselines and exact Ising targets retained.',
        small='Overall winners on all 100 native 2x4 boundary crops; dense XXZ and exact Ising reference checks.',
        mechanism='Overall XXZ winner: g/median(abs(Delta))=0,.25,.5,1,2 and absolute g=1 with combined temperature normalization on all 75 8x8 comparison images, holding its other rules fixed. Exact classical controls at matched b: remove exchange, neutralize intermediate bonds, and match isolated-bond ZZ correlations.',
        confirmation='Re-run overall-winner comparison cases flagged by Rhat>=1.05 or undefined, or ESS<200, at the confirm budget; record separately, without changing rules. Also confirm the first five comparison IDs at 16x16 for both overall winners independent of scores.',
        uncertainty='Four diverse chain starts for final runs; block/chain SE, approximate Bonferroni-t sign intervals, unranked split Rhat, ESS estimate. Image bootstrap uses original source IDs. Weak signs and convergence failures must remain visible.',
        phase_interpretation='Delta>g, abs(Delta)<=g, Delta<-g are bond-level proxies for uniform-chain regimes, not phase identification on finite disordered 2D images.',
        sampling_settings=SETTINGS,numerical_method='Sign-free SSE diagonal heat bath, eight complete directed loops, and one conditional worldline-cluster update per sweep; fixed expansion cutoff tail < exp(-50). Signed-cluster plus heat-bath Ising sampling; exact row-transfer references through width16.',
        code_hashes=hashes())


def initialize():
    OUT.mkdir(parents=True,exist_ok=True);p=ROOT/'weizmann-xxz-v2-protocol.json';value=protocol()
    if p.exists():
        if json.loads(p.read_text())!=value:raise ValueError('Protocol changed; preserve existing study.')
    else:write_json(p,value)
    data=manifest()
    assert len(data['calibration_ids'])==25 and len(data['test_ids'])==75
    assert not set(data['calibration_ids'])&set(data['test_ids'])
    print('Frozen protocol:',len(value['candidates']),'candidate rules.',flush=True)


def run_job(spec):
    stage,case,rule=spec;rid=identifier(rule)
    input_file=INPUT/'cases'/f'{case["case_id"]}.npz'
    path=OUT/stage/f'{case["case_id"]}--{rid}.json'
    signature=dict(stage=stage,case_id=case['case_id'],rule=rule,settings=SETTINGS[stage],
        code_hashes=hashes(),input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
        rng_seed=seed_for('xxz-v2:'+stage+case['case_id']+rid))
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['signature']!=signature:raise ValueError('Changed cached input or source: '+str(path))
        return cached
    started=time.perf_counter();f=np.load(input_file);image,truth,seeds=f['image'],f['truth'],f['seeds']
    r=sample(image,rule,seeds,rng_seed=signature['rng_seed'],**SETTINGS[stage])
    metrics=evaluate(r['gamma'],r['radius'],truth,seeds)
    exact=evaluate(r['reference'],0.,truth,seeds) if r['reference'] is not None else None
    model,_=parameterize(image,rule)
    diag=coupling_diagnostics(SimpleNamespace(graph=model),truth)
    cross=truth.ravel()[model.edges[:,0]]!=truth.ravel()[model.edges[:,1]]
    intermediate=abs(model.delta)<=r['params']['g']
    diag['boundary_intermediate_fraction']=float(intermediate[cross].mean()) if cross.any() else None
    diag['interior_intermediate_fraction']=float(intermediate[~cross].mean()) if (~cross).any() else None
    baselines=classical_baselines(image,truth,seeds)
    result=dict(signature=signature,rule_id=rid,family=family(rule),case=case,
        parameters=r['params'],metrics=metrics,exact=exact,sampling=r['diagnostics'],bonds=diag,
        baselines={k:mask_metrics(p,truth) for k,p in baselines.items()},seconds=time.perf_counter()-started)
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:r[k] for k in ('gamma','se','radius','batches','chain_means','scalar_series')}
    if r['reference'] is not None:arrays['reference']=r['reference']
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    write_json(path,result);return result


def dispatch(specs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_job,s) for s in specs]
        for i,future in enumerate(as_completed(futures),1):
            r=future.result()
            if i%25==0 or i==len(specs):
                print(json.dumps(dict(done=i,total=len(specs),stage=r['signature']['stage'],
                    case=r['case']['case_id'],family=r['family'],iou=r['metrics']['iou'],
                    lower=r['metrics']['iou_lower'],rhat=r['sampling']['max_correlation_rhat'])),flush=True)


def calibration_cases():return [c for c in manifest()['cases'] if c['split']=='calibration' and c['family']=='natural-square' and c['shape']==[8,8]]


def load_stage(stage):return [json.loads(p.read_text()) for p in sorted((OUT/stage).glob('*.json'))]


def scores(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['rule_id']].append(r)
    result=[]
    for rid,items in groups.items():
        if len(items)!=25:raise ValueError('Incomplete calibration for '+rid)
        result.append(dict(rule_id=rid,rule=items[0]['signature']['rule'],family=items[0]['family'],
            mean_iou=float(np.mean([r['metrics']['iou'] for r in items])),
            mean_lower_iou=float(np.mean([r['metrics']['iou_lower'] if r['sampling']['converged'] else 0. for r in items])),
            mean_unfiltered_lower_iou=float(np.mean([r['metrics']['iou_lower'] for r in items])),
            mean_resolved=float(np.mean([r['metrics']['resolved_fraction'] for r in items])),
            convergence_fraction=float(np.mean([r['sampling']['converged'] for r in items]))))
    return sorted(result,key=lambda r:(-r['mean_lower_iou'],-r['mean_iou'],r['rule_id']))


def shortlist():
    rows=load_stage('screen')
    if len(rows)!=25*len(candidates()):raise ValueError('Screen incomplete.')
    ranked=scores(rows);families=sorted(set(r['family'] for r in ranked));chosen={}
    for fam in families:
        options=[r for r in ranked if r['family']==fam]
        for r in options[:2]+[max(options,key=lambda r:(r['mean_iou'],r['mean_lower_iou']))]:chosen[r['rule_id']]=r
    result=dict(screen_scores=ranked,shortlist=list(chosen.values()),code_hashes=hashes())
    write_json(ROOT/'weizmann-xxz-v2-shortlist.json',result);return result


def select():
    short=json.loads((ROOT/'weizmann-xxz-v2-shortlist.json').read_text())
    rows=load_stage('refine')
    if len(rows)!=25*len(short['shortlist']):raise ValueError('Refinement incomplete.')
    ranked=scores(rows);winners={}
    for r in ranked:
        if r['family'] not in winners:winners[r['family']]=r
    overall={m:next(r for r in ranked if r['family'].startswith(m+'/')) for m in ('Ising','XXZ')}
    result=dict(winners=winners,overall=overall,refined_scores=ranked,code_hashes=hashes())
    write_json(ROOT/'weizmann-xxz-v2-selection.json',result)
    print(json.dumps(dict(overall=overall,winner_count=len(winners)),indent=2),flush=True);return result


def confirmation_specs():
    selected=json.loads((ROOT/'weizmann-xxz-v2-selection.json').read_text())
    ids={r['rule_id'] for r in selected['overall'].values()};first=set(manifest()['test_ids'][:5]);chosen=[]
    for r in load_stage('test'):
        if r['rule_id'] not in ids:continue
        d=r['sampling'];flag=not d['converged'] or d['minimum_variable_correlation_ess']<200
        if flag or (r['case']['shape']==[16,16] and r['case']['source_id'] in first):
            chosen.append(('confirm',r['case'],r['signature']['rule']))
    return chosen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','screen','shortlist','refine','select','test','small','scan','confirm'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();stage=args.stage
    if stage=='init':initialize();return
    if json.loads((ROOT/'weizmann-xxz-v2-protocol.json').read_text())['code_hashes']!=hashes():raise ValueError('Frozen numerical source changed.')
    if stage=='shortlist':print(json.dumps(shortlist()['shortlist'],indent=2));return
    if stage=='select':select();return
    if stage=='screen':specs=[(stage,c,r) for c in calibration_cases() for r in candidates()]
    elif stage=='refine':
        short=json.loads((ROOT/'weizmann-xxz-v2-shortlist.json').read_text())
        specs=[(stage,c,r['rule']) for c in calibration_cases() for r in short['shortlist']]
    elif stage=='confirm':specs=confirmation_specs()
    else:
        selected=json.loads((ROOT/'weizmann-xxz-v2-selection.json').read_text());data=manifest()
        if stage=='test':
            cases=[c for c in data['cases'] if c['split']=='test' and c['family']=='natural-square']
            specs=[(stage,c,r['rule']) for c in cases for r in selected['winners'].values()]
        elif stage=='small':
            cases=[c for c in data['cases'] if c['family']=='natural-strip' and c['width']==4 and c['category']=='boundary']
            specs=[(stage,c,r['rule']) for c in cases for r in selected['overall'].values()]
        else:
            base=selected['overall']['XXZ']['rule'];rules=[]
            for g in (0.,.25,.5,1.,2.):rules.append(dict(base,transverse='relative',strength=g))
            rules.append(dict(base,transverse='absolute',strength=1.,temperature='combined'))
            cases=[c for c in data['cases'] if c['split']=='test' and c['family']=='natural-square' and c['shape']==[8,8]]
            specs=[(stage,c,r) for c in cases for r in rules]
    dispatch(specs,args.workers)


if __name__=='__main__':main()
