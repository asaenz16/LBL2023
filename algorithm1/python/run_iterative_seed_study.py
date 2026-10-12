"""Exhaustive finite calibration grid for adaptive seed/anti-seed Algorithm 1."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import skimage
from run_xxz_two_seed_study import ROOT,INPUT,cases,digest,hashes as prior_hashes,identifier
from weizmann_petz_data import seed_for,write_json
from iterative_seed_gibbs import iter_rounds
from iterative_seed_readout import POLICIES,BASES,METRICS,REASONS,MAX_ROUNDS,RequiredRounds,evaluate_grid

OUT=ROOT/'results/iterative-seeds'
SETTINGS=dict(calibrate=dict(rounds=10,samples=2048,burn=1500,thin=2,chains=4),
    test=dict(rounds=10,samples=2048,burn=2000,thin=2,chains=4),
    confirm=dict(rounds=10,samples=8192,burn=6000,thin=2,chains=4))
REPEATS=dict(calibrate=3,test=10,confirm=10)


def rules():
    return [dict(threshold=t,sensitivity=a,K=k,transverse='relative',strength=g,temperature='longitudinal')
        for t,a,k,g in itertools.product(('edge_otsu','q75','q90','fixed125'),('fixed01','inverse_spread'),(.4,.8,1.2),(0.,.1,.25,.5))]


def policy_id(p):return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def hashes():
    return dict(**prior_hashes(),**{n:digest(ROOT/'python'/n) for n in
        ('iterative_seed_gibbs.py','iterative_seed_readout.py','run_iterative_seed_study.py')})


def protocol():
    return dict(date='2026-10-09',hamiltonian='H=-sum[g(XX+YY)+atan(alpha*(tau-d))ZZ]/4; unchanged between rounds; no clamping or seed-dependent fields.',
        score='mean_{s in F}<Zs Zi> - mean_{a in B}<Za Zi>; either empty average=0. Pauli normalization [-1,1] per correlation, contrast may reach [-2,2]. Ties to background.',
        initial='Compare one known foreground seed with one known background anti-seed. Ground truth used only for that initial pixel and evaluation, never promotions or stopping.',
        promotions='At most one foreground and one background per round, uniform random among eligible pixels not already in either set. Seed sets grow monotonically. No automatic correctness or resolution is assigned to promoted pixels.',
        candidate_gates='confidence: score lower bound > positive cutoff / upper bound < negative cutoff; point_resolved: point exceeds cutoff and interval excludes zero. No candidates on failed Rhat check.',
        stopping='All non-initial pixels resolved, no eligible candidates, unreliable sampling, configured cap, or configured count stabilization. same_count: two consecutive equal-count transitions. no_improvement: two rounds without strict best-count decrease. tolerance_1pct: decrease must exceed max(1,ceil(.01*(N-1))) to reset two-round patience.',
        grid=dict(hamiltonians=rules(),policies=POLICIES,hamiltonian_count=len(rules()),policy_count=len(POLICIES),full_parameter_combinations=len(rules())*len(POLICIES),
            scope='Full Cartesian grid, not exhaustive over continuous parameters. Includes asymmetric seed/anti-seed cutoffs and both starting conventions. 25 calibration sources at both8 and16; reused development split.',
            random_candidate_repeats=REPEATS),
        settings=SETTINGS,measurement='Full diagonal Z-basis snapshots. All ZZ observables commute; reuse the same snapshots across policies, but use separately seeded and initialized fresh ensembles each round. Stop generating future rounds only after all requested policies terminate. Unused future draws never influence past decisions.',
        uncertainty='Four chains, 16 blocks/chain, maximum batch/between-chain SE, Bonferroni-t nominal95% over N pixels and10 rounds per trajectory. Full-series score split-Rhat and scalar Rhat<1.05. Approximate MCMC coverage, not rigorous or simultaneous across the parameter grid.',
        pooling='No pooling past samples for adaptive scores. Fixed-initial-seed and oracle-two-seed controls pool exactly the same number of rounds as the adaptive trajectory; also record first-round and fresh-last-round one-seed controls.',
        selection='Choose maximum mean mixing-aware lower IoU separately for Ising/XXZ and foreground/background starts; secondary maximum raw-IoU selections. Average repeats then both resolutions then25 sources. Ties IoU, fewer rounds, rule ID, policy ID. Never use comparison images for selection.',
        test='75 comparison sources at both8 and16 for unique selected Hamiltonians. Evaluate selected policies and their opposite-start counterpart at fixed parameters, ten random promotion trajectories each. Include all controls and BPM.',
        confirmation='Separate longer Gibbs bank for every test model/image case with a failed final-round convergence flag or minimum score ESS<100 in any selected trajectory, plus first five test source IDs at16. Replace whole bank including all trajectories/controls without retuning or cherry-picking.',
        metrics='Unresolved and unseeded IoU exclude ONLY the originally supplied pixel, not promoted pixels. Full-mask IoU does not overwrite seed labels. Report promotion error against truth after selection, seed counts, stopping reasons, and BPM radii0,1,2 with established Zhang skeleton convention.',
        code_hashes=hashes(),skimage=skimage.__version__,manifest_sha256=digest(ROOT/'weizmann-petz-manifest.json'))


def check_protocol():
    path=ROOT/'iterative-seeds-protocol.json';value=protocol()
    if path.exists():assert json.loads(path.read_text())==value,'Frozen protocol or source changed.'
    else:write_json(path,value)


def selected_jobs():
    d=json.loads((ROOT/'iterative-seeds-selection.json').read_text());assert d['code_hashes']==hashes()
    groups={}
    for label,row in d['winners'].items():
        rid=row['rule_id'];groups.setdefault(rid,dict(rule=row['rule'],policies={}))
        for policy in (row['policy'],dict(row['policy'],start='background' if row['policy']['start']=='foreground' else 'foreground')):
            groups[rid]['policies'][policy_id(policy)]=policy
    return [dict(rule=v['rule'],policies=list(v['policies'].values())) for v in groups.values()]


def job(spec):
    stage,c,rule,policies=spec;rid=identifier(rule);path=OUT/stage/f'{c["case_id"]}--{rid}.json'
    source=INPUT/'cases'/f'{c["case_id"]}.npz';rng_key='iterative-seeds:'+stage+':'+c['case_id']+':'+rid
    signature=dict(stage=stage,case=c,rule=rule,policy_sha256=hashlib.sha256(json.dumps(policies,sort_keys=True).encode()).hexdigest(),settings=SETTINGS[stage],repeats=REPEATS[stage],
        input_sha256=digest(source),code_hashes=hashes(),gibbs_seed=seed_for(rng_key+':gibbs'),promotion_key=rng_key+':promote')
    if path.exists():
        row=json.loads(path.read_text());assert row['signature']==signature
        return c['case_id'],row['rounds_generated'],row['seconds']
    start=time.perf_counter();data=np.load(source);path.parent.mkdir(parents=True,exist_ok=True)
    bank_path=path.with_suffix('.bank.npz');checkpoint=path.with_suffix('.bank.json')
    if bank_path.exists() and checkpoint.exists():
        saved=json.loads(checkpoint.read_text());assert saved['signature']==signature
        packed=np.load(bank_path)['packed'];param=saved['parameters'];diag=saved['sampling']
    else:
        guard=RequiredRounds(data['seeds'],policies,REPEATS[stage],signature['promotion_key'],data['image'].size)
        packed=[];diag=[]
        for z,param,d in iter_rounds(data['image'],rule,**SETTINGS[stage],rng_seed=signature['gibbs_seed']):
            packed.append(z);diag.append(d)
            if not guard.consume(z,d):break
        packed=np.stack(packed)
        np.savez_compressed(bank_path,packed=packed)
        write_json(checkpoint,dict(signature=signature,parameters=param,sampling=diag))
    table,reasons,traces=evaluate_grid(packed,diag,data['truth'],data['seeds'],REPEATS[stage],signature['promotion_key'],
        policies=policies,save_traces=stage!='calibrate')
    np.savez_compressed(path.with_suffix('.npz'),metrics=table,reasons=reasons)
    row=dict(signature=signature,case=c,rule=rule,rule_id=rid,parameters=param,policies=policies if stage!='calibrate' else None,
        metrics=list(METRICS),reasons=list(REASONS),sampling=diag,rounds_generated=len(packed),
        seconds=time.perf_counter()-start,bank_sha256=digest(bank_path),traces=traces)
    write_json(path,row);return c['case_id'],len(packed),row['seconds']


def dispatch(specs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,s) for s in specs]
        for i,f in enumerate(as_completed(futures),1):
            case,rounds,seconds=f.result()
            if i%10==0 or i==len(specs):print(json.dumps(dict(done=i,total=len(specs),last=case,rounds=rounds,seconds=round(seconds,2))),flush=True)


def select():
    scores=[]
    for rule in rules():
        rid=identifier(rule);arrays=[];files=[]
        for c in cases('calibration'):
            path=OUT/'calibrate'/f'{c["case_id"]}--{rid}.json'
            row=json.loads(path.read_text());assert row['signature']['code_hashes']==hashes()
            arrays.append(np.load(path.with_suffix('.npz'))['metrics'].mean(axis=1));files.append(path)
        mean=np.stack(arrays).mean(axis=0)
        for i,p in enumerate(POLICIES):
            scores.append(dict(rule=rule,rule_id=rid,policy=p,policy_id=policy_id(p),metrics=dict(zip(METRICS,mean[i].tolist()))))
    winners={}
    for model,start in itertools.product(('Ising','XXZ'),('foreground','background')):
        group=[r for r in scores if (r['rule']['strength']==0)==(model=='Ising') and r['policy']['start']==start]
        for objective in ('mixing_lower','iou'):
            best=min(group,key=lambda r:(-r['metrics'][objective],-r['metrics']['iou'],r['metrics']['rounds'],r['rule_id'],r['policy_id']))
            label=f'{model}/{start}/{objective}';winners[label]=best
            print(label,best['rule'],best['policy'],{k:round(best['metrics'][k],4) for k in ('iou','unresolved','mixing_lower','rounds')},flush=True)
    result=dict(winners=winners,grid_rows=len(scores),code_hashes=hashes(),
        calibration_hashes={p.name:digest(p) for p in sorted((OUT/'calibrate').glob('*.json'))})
    path=ROOT/'iterative-seeds-selection.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write_json(path,result)
    write_json(OUT/'calibration-scores.json',scores)


def confirmation_specs():
    fixed=set(sorted({c['source_id'] for c in cases('test')})[:5]);specs=[]
    for c in cases('test'):
        for v in selected_jobs():
            path=OUT/'test'/f'{c["case_id"]}--{identifier(v["rule"])}.json';row=json.loads(path.read_text())
            values=np.load(path.with_suffix('.npz'))['metrics']
            bad=np.any(values[:,:,METRICS.index('converged')]<1) or np.any(values[:,:,METRICS.index('minimum_ess')]<100)
            if bad or (c['shape']==[16,16] and c['source_id'] in fixed):specs.append(('confirm',c,v['rule'],v['policies']))
    return specs


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','calibrate','select','test','confirm']);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    check_protocol()
    if args.stage=='init':print(len(rules()),'Hamiltonians;',len(POLICIES),'policies;',len(rules())*len(POLICIES),'combinations;',len(cases('calibration')),'calibration cases');return
    if args.stage=='select':select();return
    if args.stage=='calibrate':specs=[('calibrate',c,r,POLICIES) for c in cases('calibration') for r in rules()]
    elif args.stage=='test':specs=[('test',c,v['rule'],v['policies']) for c in cases('test') for v in selected_jobs()]
    else:specs=confirmation_specs()
    dispatch(specs,args.workers)


if __name__=='__main__':main()
