"""Reproducible fixed-H higher-order seed readout parameter experiment."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from run_iterative_seed_study import ROOT, INPUT, cases, identifier, digest
from weizmann_petz_data import seed_for, write_json
from higher_seed_readout import CONFIGS, POLICIES, FAMILIES, METRICS, REASONS, evaluate

OUT = ROOT/'results/higher-seeds'
BANKS = ROOT/'results/promotion-reliability/banks'


def anchors():
    old = json.loads((ROOT/'promotion-reliability-protocol.json').read_text())['anchors']
    return {k.rsplit('/',1)[0]:v for k,v in old.items() if k.endswith('/mixing_lower')}


def pid(p): return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]


def code_hashes():
    return {f:digest(ROOT/'python'/f) for f in
            ('higher_seed_readout.py','run_higher_seed_study.py','segmentation_bpm.py')}


def init():
    OUT.mkdir(parents=True, exist_ok=True)
    p = dict(date='2026-10-09', anchors=anchors(), policies=POLICIES,
        config_count=len(CONFIGS), policy_count=len(POLICIES), code_hashes=code_hashes(),
        manifest_sha256=digest(ROOT/'weizmann-petz-manifest.json'),
        dataset='25 calibration and75 comparison source images,8x8 and16x16. Previously reused development split, exploratory not external validation.',
        state='Fixed zero-field Ising/XXZ Hamiltonians inherited from primary promotion study. No fields or clamping. Reuse SHA-verified independent-round Z snapshot banks.',
        baseline='Mean foreground ZZ minus mean background ZZ, divided by number of nonempty roles to normalize [-1,1]. All promoted pixels scored leave-one-out; original trusted label retained. This is a controlled new baseline, not an identical historical replay.',
        groups='One random triple per round; uniform from active seeds or constrained to include original anchor. Shared pre-sample random priorities across methods. For any promoted target, exclude it and select again. With fewer than3 other seeds fall back to pairwise.',
        majority='Samplewise majority of3 signed seed spins, then multiply target spin and average. Blend weights .25,.5,1 for score use.',
        soft='Product weights with eta=.25,.5,.75,.9 for promoted seeds and1 for trusted anchor. Exact global-flip symmetrization: A=sum(v)+prod(v), B=1+sum(pair products); score E[Zi A]/E[B]. Numerator/denominator paired delta-method covariance retained. Kish weight ESS>=100 required; not equivalent to MCMC ESS.',
        gates='Keep pairwise segmentation and confidence-cutoff promotion; require extra signed higher-score confidence lower bound>0 or.15. No extra gate until>=3 active seeds. Score variants use blended score for both segmentation and promotion.',
        promotion='At most1 foreground and1 background uniformly among eligible candidates. Only initial pixel label supplied; truth never used for inference,promotion,triplet selection or stopping.',
        stopping='All resolved,no candidates,failed inference,cap3/6/10,or2 consecutive unresolved-count equalities. No forced bootstrap past all-resolved/no-candidate stopping. Activation frequency reported.',
        confidence='4 chains x16 blocks; max batch and between-chain SE; Bonferroni-t over pixels*10 rounds. Split-Rhat of influence observations and inherited scalar guard<1.05. Approximate MCMC/delta-method intervals, not simultaneous across parameter grid.',
        budgets='Main2048 retained configurations/chain/round (8192 total). Frozen selected parameters also replayed using512 and1024 prefixes. All methods paired on same bank and budget; actual consumed rounds recorded. Bank tails unused by algorithm not charged.',
        calibration='Full finite readout grid on25 images atboth sizes,2 common random promotion repeats. Select separately per model/start/family by mixing_lower,IoU,BPM,and minimum wrong-promotion ratio subject to IoU>=family best-.01, mixing U<=family min+.03, mean promoted>=1. No comparison-based retuning.',
        comparison='75 images atboth sizes,5 common promotion repeats. Calibrated selections plus cutoff/margin/stop/cap matched pair controls and matched-triple linear controls. Primary balanced selections also tested at512/1024 samples/chain.',
        uncertainty_analysis='Paired bootstrap over75 source IDs,average promotion repeats within source; pointwise exploratory95% intervals. Report ratio of total wrong to total promotions; zero promotions undefined.',
        metrics='Full tentative-mask IoU and inherited BPM skeleton proximity at radii0,1,2. Unresolved excludes ONLY trusted initial pixel. Mixing-aware unresolved=1 on failed score convergence. Retained configurations,rounds,score ESS,weight ESS,offline runtime.',
        limitations='No new Hamiltonian/temperature optimization. Ising exact iid conditional-frontier draws; XXZ approximate SSE. Prefix budget scalar diagnostics inherited from full bank and direct readout diagnostics recomputed on actual prefix.')
    path=ROOT/'higher-seeds-protocol.json'
    if path.exists(): assert json.loads(path.read_text())==p, 'Frozen protocol changed'
    else: write_json(path,p)


def load_bank(stage, case, anchor):
    stem = f'{case["case_id"]}--{identifier(anchors()[anchor]["rule"])}'
    path = BANKS/('calibrate' if stage=='calibrate' else 'test')/f'{stem}.json'
    meta=json.loads(path.read_text()); bank=path.with_suffix('.npz')
    assert digest(bank)==meta['bank_sha256']
    source=INPUT/'cases'/f'{case["case_id"]}.npz'
    assert digest(source)==meta['signature']['input_sha256']
    return np.load(bank)['packed'], meta, np.load(source), str(bank)


def selection(): return json.loads((ROOT/'higher-seeds-selection.json').read_text())


def profiles(anchor, budget):
    chosen=selection()['winners'][anchor]; out={}
    for label,v in chosen.items():
        if budget!=2048 and not label.endswith('/balanced'): continue
        p=v['policy'];out[label]=p
        if p['family']!='pair':
            out[label+'/matched_pair']=dict(p,family='pair',group='anchor',eta=0.,blend=0.,gate_margin=0.)
            # Same nonlinear groups, but replace nonlinear numerator by linear majority input.
            out[label+'/matched_linear']=dict(p,family='linear_'+p['family'].split('_')[1],eta=0.)
    return out


def job(spec):
    stage,case,anchor,budget=spec
    path=OUT/stage/f'{case["case_id"]}--{anchor.replace("/","-")}--{budget}.json'
    labels={str(i):p for i,p in enumerate(POLICIES)} if stage=='calibrate' else profiles(anchor,budget)
    policies=list({pid(p):p for p in labels.values()}.values())
    repeats=2 if stage=='calibrate' else 5
    signature=dict(case=case,anchor=anchor,samples=budget,policies=policies,repeats=repeats,code_hashes=code_hashes())
    if path.exists():
        old=json.loads(path.read_text());assert old['signature']==signature
        return {'cached':True,'case':case['case_id'],'anchor':anchor}
    start=time.perf_counter(); packed,meta,data,bank=load_bank(stage,case,anchor)
    initial=int(data['seeds'][1 if anchor.endswith('foreground') else 0])
    table,reasons,traces=evaluate(packed,meta['sampling'],data['truth'],initial,anchor.split('/')[1],
        policies,repeats,seed_for('higher-seeds:'+stage+':'+case['case_id']+':'+anchor),
        samples=budget,traces=stage!='calibrate')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),metrics=table,reasons=reasons)
    row=dict(signature=signature,labels={k:policies.index(p) for k,p in labels.items()},metrics=METRICS,
             reasons=REASONS,bank=bank,bank_sha256=meta['bank_sha256'],bank_settings=meta['signature']['settings'],
             seconds=time.perf_counter()-start,traces=traces)
    write_json(path,row)
    return dict(case=case['case_id'],anchor=anchor,budget=budget,seconds=round(row['seconds'],2))


def select():
    winners={};scores={};mi={m:i for i,m in enumerate(METRICS)}
    for anchor in anchors():
        rows=[]
        for case in cases('calibration'):
            path=OUT/'calibrate'/f'{case["case_id"]}--{anchor.replace("/","-")}--2048.json'
            meta=json.loads(path.read_text()); assert meta['signature']['policies']==POLICIES
            rows.append(np.load(path.with_suffix('.npz'))['metrics'].mean(1))
        mean=np.mean(rows,axis=0);scores[anchor]=mean.tolist();winners[anchor]={}
        for family in FAMILIES:
            eligible=[i for i,p in enumerate(POLICIES) if p['family']==family]
            for objective,metric in [('balanced','mixing_lower'),('iou','iou'),('boundary','bpm_iou_r1'),('reliability',None)]:
                pool=eligible
                if objective=='reliability':
                    best=max(mean[i,mi['iou']] for i in eligible);least=min(mean[i,mi['mixing_unresolved']] for i in eligible)
                    pool=[i for i in eligible if mean[i,mi['iou']]>=best-.01 and
                          mean[i,mi['mixing_unresolved']]<=least+.03 and mean[i,mi['promoted_count']]>=1]
                    if not pool: pool=eligible
                    def order(i):
                        den=mean[i,mi['promoted_count']]
                        return (mean[i,mi['promoted_wrong']]/den if den else 2.,-mean[i,mi['mixing_lower']],mean[i,mi['shots']],pid(POLICIES[i]))
                else:
                    def order(i):return (-mean[i,mi[metric]],-mean[i,mi['mixing_lower']],mean[i,mi['shots']],pid(POLICIES[i]))
                i=min(pool,key=order)
                winners[anchor][family+'/'+objective]=dict(policy=POLICIES[i],calibration=dict(zip(METRICS,mean[i].tolist())))
        print(anchor, {k:round(v['calibration']['iou'],4) for k,v in winners[anchor].items() if k.endswith('/balanced')},flush=True)
    write_json(ROOT/'higher-seeds-selection.json',dict(winners=winners,code_hashes=code_hashes(),
        calibration_sha256={p.name:digest(p) for p in sorted((OUT/'calibrate').glob('*'))}))
    write_json(OUT/'calibration-scores.json',dict(metrics=METRICS,policies=POLICIES,means=scores))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('init','pilot','calibrate','select','test','budget'))
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.stage=='init':init();print(len(CONFIGS),'configurations;',len(POLICIES),'policies/anchor');return
    if args.stage=='pilot':print(job(('calibrate',cases('calibration')[0],next(iter(anchors())),2048)));return
    if args.stage=='select':select();return
    stage='calibrate' if args.stage=='calibrate' else 'test'
    budgets=(512,1024) if args.stage=='budget' else (2048,)
    specs=[(stage,c,a,b) for c in cases('calibration' if stage=='calibrate' else 'test') for a in anchors() for b in budgets]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,s) for s in specs]
        for i,f in enumerate(as_completed(futures),1):
            result=f.result()
            if i%5==0 or i==len(futures):print(json.dumps(dict(done=i,total=len(futures),last=result)),flush=True)


if __name__=='__main__': main()
