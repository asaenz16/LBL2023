"""Frozen finite ablation experiment; reuses and extends independent Gibbs banks."""
import os
for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import hashlib
import json
import time
import numpy as np
from run_iterative_seed_study import ROOT,INPUT,cases,digest,identifier,SETTINGS,REPEATS,hashes as old_hashes,OUT as OLD
from weizmann_petz_data import write_json,seed_for
from iterative_seed_readout import base_of
from promotion_reliability_bank import iter_tail
from promotion_reliability import POLICIES,CONFIGS,METRICS,REASONS,evaluate
OUT=ROOT/'results/promotion-reliability'
FAMILIES=('baseline','spatial','agreement','remove','weight','combined','any')


def pid(p):return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]


@lru_cache(None)
def anchors():
    old=json.loads((ROOT/'iterative-seeds-selection.json').read_text());assert old['code_hashes']==old_hashes()
    return {k:dict(rule=v['rule'],base=base_of(v['policy']),historical=v['policy']) for k,v in old['winners'].items()}


def family(p,f):
    s=p['space']!='none';a=p['agreement']!='none';m=p['maintenance']!='keep'
    return {'baseline':not(s or a or m) and p['cut']=='base','spatial':s and not(a or m),
        'agreement':a and not(s or m),'remove':p['maintenance'].startswith('remove') and not(s or a),
        'weight':p['maintenance'].startswith('soft') and not(s or a),'combined':s and a and m,'any':s or a or m}[f]


def neutral(p):return dict(p,space='none',agreement='none',maintenance='keep')


def historical(a):
    p=a['historical'];stop=p['stopping']
    if stop=='no_improvement':assert p['cap']==3;stop='none'
    return dict(space='none',agreement='none',maintenance='keep',cut='base',cap=p['cap'],stopping=stop)


@lru_cache(None)
def hashes():
    return dict(**old_hashes(),**{n:digest(ROOT/'python'/n) for n in ('iterative_seed_extended_loops.py','promotion_reliability.py','promotion_reliability_bank.py','run_promotion_reliability.py')})


def protocol():
    return dict(date='2026-10-09',anchors=anchors(),code_hashes=hashes(),old_selection_sha256=digest(ROOT/'iterative-seeds-selection.json'),
        manifest_sha256=digest(ROOT/'weizmann-petz-manifest.json'),settings=SETTINGS,repeats=REPEATS,policies=POLICIES,
        scope='Finite full 5 spatial x5 agreement x5 maintenance x2 cutoff x3 caps x3 stops grid; 2250 policies for each of four primary fixed-H anchors. 25 calibration sources and75 comparison sources, both8 and16 pixels/side; reused development split.',
        spatial='Euclidean exclusion distance strictly greater than 1/8 or1/4 shorter image side; active same-class or all-class seeds. Simultaneous new opposite-role candidates also obey all-class exclusion.',
        agreement='75% or100% of active seed signed correlations confidently agree; abstentions count against quota. Pairwise Bonferroni-t over N^2*10, split Rhat<1.05. Temporal2 additionally requires eligibility in consecutive fresh rounds; first-round warmup allowed.',
        maintenance='Trusted initial seed protected. Leave-one-out signed contrast removes self correlation. Remove after1 or2 consecutive confident contradictions, permanently retire. Soft target=clip((signed LOO score-radius)/.5,.1,1), next weight=.5old+.5target when difference>=.05, rounded4digits. Relative denominator=sum weights; absolute denominator=number active in class. Weights only change next fresh round.',
        score='Original fixed Gibbs Hamiltonian and Pauli ZZ measurements; no clamping. Weighted mean foreground minus weighted mean background, empty term0. Cutoffs inherited or(.75,.75); inherited candidate gate. At most one uniformly random new seed per role per round. Role selection order randomized with shared per-round uniforms across policies.',
        truth='Only original label and evaluation/calibration selection use truth. No promotion, maintenance or stopping uses truth. Only original pixel excluded from unresolved and unknown IoU.',
        stops='Ordinary all-resolved/no-candidates/failure/cap; same_count requires2 equal-count transitions. audit3 requires >=3 fresh rounds and applies pending maintenance before stopping, subject to cap and failure. Fully resolved rounds never promote. Temporal warmup and second removal strike may defer no-candidates.',
        samples='Old raw bank prefixes SHA-verified, extended to10 independent rounds using identical SeedSequence and extended completed-loop SSE kernel. Prefix never overwritten. Sample sets shared across policies; Gibbs state fixed. Nominal uncertainty approximate for MCMC and not simultaneous across grid.',
        selection='Mean over repeats, resolutions, then25 sources: maximize mixing_lower (primary) or raw IoU (secondary), ties IoU/fewer rounds/policyID. Separately choose each mechanism family per anchor. Each selected rule gets neutral matched cutoff/cap/stop control plus historical baseline. Transfer primary any-family modifier to four secondary Hamiltonian/cutoff anchors without retuning.',
        confirmation='Any final selected or matched-control trajectory failing convergence or minESS<100 triggers8192 samples/chain; also first5 lexically sorted comparison source IDs at16. Replace entire case/H bank including all policies/controls. No selection from confirmation. Report unresolved failures.',
        controls='First-round, last-round, equal-budget pooled original seed, equal-budget oracle true foreground plus background. Same budget equals actual adaptive endpoint rounds, not maximum bank length.',
        reliability='Active and ever wrong promotions, weighted wrong mass, retirement precision/recall, promotion counts and no-promotion frequency. Zero promotions has undefined error ratio, never called perfect precision.',
        uncertainty_analysis='Paired bootstrap over75 source IDs, average random-promotion repeats within source. Pointwise95% intervals exploratory, not simultaneous over policy families.')


def check_protocol():
    p=ROOT/'promotion-reliability-protocol.json';v=protocol()
    if p.exists():assert json.loads(p.read_text())==v,'Frozen protocol/source changed'
    else:write_json(p,v)


def bank(stage,c,rule):
    rid=identifier(rule);stem=f'{c["case_id"]}--{rid}';out=OUT/'banks'/stage/stem;out.parent.mkdir(parents=True,exist_ok=True)
    source=INPUT/'cases'/f'{c["case_id"]}.npz';data=np.load(source)
    key='iterative-seeds:'+stage+':'+c['case_id']+':'+rid
    sig=dict(stage=stage,case=c,rule=rule,settings=SETTINGS[stage],gibbs_seed=seed_for(key+':gibbs'),input_sha256=digest(source),code_hashes=hashes())
    if out.with_suffix('.json').exists():
        meta=json.loads(out.with_suffix('.json').read_text());assert meta['signature']==sig
        assert meta['bank_sha256']==digest(out.with_suffix('.npz'))
        return np.load(out.with_suffix('.npz'))['packed'],meta,data
    prefix=OLD/stage/stem;packed=[];diag=[];provenance=None;param=None
    if prefix.with_suffix('.json').exists():
        row=json.loads(prefix.with_suffix('.json').read_text());p=prefix.with_suffix('.bank.npz')
        assert row['bank_sha256']==digest(p)
        for field in ('settings','gibbs_seed','input_sha256'):assert row['signature'][field]==sig[field]
        assert row['signature']['code_hashes']==old_hashes()
        packed=list(np.load(p)['packed']);diag=row['sampling'];param=row['parameters']
        provenance=dict(path=str(p),sha256=digest(p),rounds=len(packed),metadata_sha256=digest(prefix.with_suffix('.json')))
    for z,param,d in iter_tail(data['image'],rule,**SETTINGS[stage],rng_seed=sig['gibbs_seed'],start_round=len(packed)):
        packed.append(z);diag.append(d)
    packed=np.stack(packed);np.savez_compressed(out.with_suffix('.npz'),packed=packed)
    meta=dict(signature=sig,prefix=provenance,parameters=param,sampling=diag,bank_sha256=digest(out.with_suffix('.npz')))
    write_json(out.with_suffix('.json'),meta);return packed,meta,data


def selection():return json.loads((ROOT/'promotion-reliability-selection.json').read_text())


def test_profiles():
    profiles={}
    for anchor,groups in selection()['winners'].items():
        rows={}
        for name,v in groups.items():rows[name]=v['policy'];rows[name+'/matched']=neutral(v['policy'])
        rows['historical']=historical(anchors()[anchor]);profiles[anchor]=rows
    for anchor in [k for k in anchors() if k.endswith('/iou')]:
        primary=anchor.rsplit('/',1)[0]+'/mixing_lower';p=selection()['winners'][primary]['any/mixing_lower']['policy']
        profiles[anchor]={'transfer':p,'transfer/matched':neutral(p),'historical':historical(anchors()[anchor])}
    return profiles


def grouped(stage):
    profiles={k:{str(i):p for i,p in enumerate(POLICIES)} for k in anchors() if k.endswith('/mixing_lower')} if stage=='calibrate' else test_profiles()
    groups={}
    for anchor,rows in profiles.items():
        a=anchors()[anchor];rid=identifier(a['rule']);groups.setdefault(rid,dict(rule=a['rule'],profiles={}))
        groups[rid]['profiles'][anchor]=rows
    return list(groups.values())


def job(spec):
    stage,c,group=spec;rule=group['rule'];rid=identifier(rule);path=OUT/stage/f'{c["case_id"]}--{rid}.json';path.parent.mkdir(parents=True,exist_ok=True)
    sig=dict(stage=stage,case=c,group=group,code_hashes=hashes(),repeats=REPEATS[stage])
    if path.exists():
        row=json.loads(path.read_text());assert row['signature']==sig;return c['case_id'],row['seconds']
    start=time.perf_counter();packed,meta,data=bank(stage,c,rule);tables={};reason_arrays={};rows={}
    for j,(anchor,labels) in enumerate(group['profiles'].items()):
        policies=list({pid(p):p for p in labels.values()}.values());indices={name:policies.index(p) for name,p in labels.items()}
        key='promotion-reliability:'+stage+':'+c['case_id']+':'+anchor
        table,reasons,traces=evaluate(packed,meta['sampling'],data['truth'],data['seeds'],anchors()[anchor]['base'],policies,REPEATS[stage],key,traces=stage!='calibrate')
        tables[f'metrics{j}']=table;reason_arrays[f'reasons{j}']=reasons
        rows[anchor]=dict(array_index=j,policies=policies,labels=indices,traces=traces,promotion_key=key)
    np.savez_compressed(path.with_suffix('.npz'),**tables,**reason_arrays)
    row=dict(signature=sig,case=c,rule=rule,profiles=rows,metrics=METRICS,reasons=REASONS,bank_sha256=meta['bank_sha256'],seconds=time.perf_counter()-start)
    write_json(path,row);return c['case_id'],row['seconds']


def dispatch(specs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(job,s) for s in specs]
        for i,f in enumerate(as_completed(fs),1):
            c,sec=f.result()
            if i%5==0 or i==len(fs):print(json.dumps(dict(done=i,total=len(fs),last=c,seconds=round(sec,2))),flush=True)


def select():
    winners={};scores={}
    for anchor,a in anchors().items():
        if not anchor.endswith('/mixing_lower'):continue
        vals=[]
        for c in cases('calibration'):
            p=OUT/'calibrate'/f'{c["case_id"]}--{identifier(a["rule"])}.json';r=json.loads(p.read_text());assert r['signature']['code_hashes']==hashes()
            v=r['profiles'][anchor];assert v['policies']==POLICIES
            vals.append(np.load(p.with_suffix('.npz'))[f'metrics{v["array_index"]}'].mean(1))
        mean=np.mean(vals,axis=0);scores[anchor]=mean.tolist();winners[anchor]={}
        for f in FAMILIES:
            eligible=[i for i,p in enumerate(POLICIES) if family(p,f)]
            for objective in ('mixing_lower','iou'):
                best=min(eligible,key=lambda i:(-mean[i,METRICS.index(objective)],-mean[i,0],mean[i,METRICS.index('rounds')],pid(POLICIES[i])))
                winners[anchor][f+'/'+objective]=dict(policy=POLICIES[best],metrics=dict(zip(METRICS,mean[best].tolist())))
        print(anchor,winners[anchor]['any/mixing_lower'],flush=True)
    result=dict(code_hashes=hashes(),winners=winners,calibration_sha256={p.name:digest(p) for p in sorted((OUT/'calibrate').glob('*'))})
    p=ROOT/'promotion-reliability-selection.json'
    if p.exists():assert json.loads(p.read_text())==result
    else:write_json(p,result)
    write_json(OUT/'calibration-scores.json',dict(metrics=METRICS,policies=POLICIES,means=scores))


def confirmation_specs():
    specs=[];fixed=set(sorted({c['source_id'] for c in cases('test')})[:5])
    for c in cases('test'):
        for group in grouped('test'):
            p=OUT/'test'/f'{c["case_id"]}--{identifier(group["rule"])}.json';r=json.loads(p.read_text());v=np.load(p.with_suffix('.npz'))
            bad=any(np.any(v[f'metrics{a["array_index"]}'][:,:,METRICS.index('converged')]<1) or np.any(v[f'metrics{a["array_index"]}'][:,:,METRICS.index('minimum_ess')]<100) for a in r['profiles'].values())
            if bad or (c['shape']==[16,16] and c['source_id'] in fixed):specs.append(('confirm',c,group))
    return specs


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('init','pilot','calibrate','select','test','confirm'));p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.stage=='pilot':
        # Numerical pilot is calibration-only; no comparison outcomes inspected.
        for c in cases('calibration')[:2]:print(job(('calibrate',c,grouped('calibrate')[0])),flush=True)
        return
    check_protocol()
    if args.stage=='init':print('2250 policies/anchor; 1,350,000 calibration endpoints');return
    if args.stage=='select':select();return
    specs=confirmation_specs() if args.stage=='confirm' else [(args.stage,c,g) for c in cases('calibration' if args.stage=='calibrate' else 'test') for g in grouped(args.stage)]
    dispatch(specs,args.workers)


if __name__=='__main__':main()
