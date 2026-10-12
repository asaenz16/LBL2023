"""Frozen reanalysis and new moderate-exchange/temperature experiment."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import skimage
from run_weizmann_xxz_v2 import ROOT,INPUT,OUT as LEGACY,manifest,identifier,hashes as legacy_hashes
from weizmann_petz_data import seed_for,write_json
from weizmann_petz_metrics import classical_baselines,mask_metrics
from xxz_image_gibbs import rhat
from xxz_two_seed import DECISIONS,scores,sample,evaluate_arrays
from segmentation_bpm import bpm_metrics

OUT=ROOT/'results/xxz-two-seed'
SETTINGS=dict(calibrate=dict(burn=1500,samples=2048,thin=2,chains=4),
    test=dict(burn=2000,samples=4096,thin=2,chains=4),
    confirm=dict(burn=8000,samples=16384,thin=2,chains=4))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def hashes():
    return dict(**legacy_hashes(),**{name:digest(ROOT/'python'/name) for name in
        ('run_xxz_two_seed_study.py','xxz_two_seed.py','segmentation_bpm.py','ising_seed_sampling.py','weizmann_petz_data.py')})


def cases(split):
    return [c for c in manifest()['cases'] if c['split']==split and c['family']=='natural-square']


def candidates():
    return [dict(threshold=threshold,sensitivity=sensitivity,K=k,transverse='relative',strength=g,temperature='longitudinal')
        for (threshold,sensitivity),k,g in itertools.product(
            [('edge_otsu','fixed01'),('q90','inverse_spread')],(.4,.8,1.2,1.6),(0.,.1,.25,.5,.75))]


def family(rule):return ('Ising' if rule['strength']==0 else 'XXZ')+'/'+rule['threshold']


def protocol():
    return dict(date='2026-10-08',hamiltonian='H=-sum[g(XX+YY)+atan(alpha*(tau-d))ZZ]/4; Gamma=<ZZ>/4',
        candidates=candidates(),settings=SETTINGS,code_hashes=hashes(),skimage=skimage.__version__,
        manifest_sha256=digest(ROOT/'weizmann-petz-manifest.json'),
        data='Verified Weizmann archive; 25 calibration sources and 75 comparison sources; both 8x8 and 16x16; oracle interior seeds. Reused development split, not external validation.',
        reanalysis='All original test, scan and small runs; original confirm substitutes only for matching main test rows. Paired batch differences retain covariance. Point masks use max signed seed correlation. Source hashes retained.',
        selection='All 40 candidates on 25 calibration sources at BOTH sizes. Equal weight per source and per size. Highest mean mixing-aware lower IoU per model/threshold family for two_max; also overall one_background winners and secondary raw-IoU two_max winners. Ties sampled IoU then rule ID. No BPM tuning. Additional g=0 controls match each selected two_max XXZ state parameters.',
        comparison='Selected rules on all 75 sources at both sizes; all three readouts from same samples. Exact iid Gibbs for Ising, numerical SSE Gibbs for XXZ. Equal retained configuration counts; unequal effective sample sizes and cost are reported.',
        confirmation='All selected-rule test cases with max Rhat>=1.05/undefined or minimum paired-score ESS<200 rerun at fixed larger budget. Also first five lexically sorted test source IDs at16 for each rule independent of outcomes. Substitute confirmations without changing rules or dropping failures.',
        uncertainty='Approximate Bonferroni-t intervals over 3 readouts times all pixels. SE is max of paired batch-mean and paired between-chain SE. Old cached runs retain 16 blocks, new runs32. Not iid binomial coverage or rigorous finite-time MCMC coverage. Old direct-score Rhat only available on batch series; guard at1.1 plus original diagnostics. New direct-score and scalar/raw Rhat checked at1.05.',
        unresolved='Exclude both known seeds; report statistical unresolved and mixing-aware unresolved (all pixels unresolved on failed cases). IoU/BPM use tentative full masks, including unresolved pixels.',
        bpm=dict(source='https://arxiv.org/pdf/2410.10713v1',primary_radius=1,radii=[0,1,2],
            skeletonization='skimage.morphology.skeletonize(method=zhang)',distance='Euclidean disk on evaluated pixel grid; outside image false',
            counts='TP=predicted skeleton within r of truth skeleton; FP=unmatched predicted skeleton; FN=unmatched truth skeleton; IoU=TP/(TP+FP+FN); F1=2TP/(2TP+FP+FN)',
            ambiguity='Paper equations print negative FP/FN masks then l1 norm and an inconsistent unsigned sum. Nonnegative unmatched counts implement the l1 interpretation. Radius and skeletonization algorithm unspecified in paper. Not verified author-code reproduction.',
            limitations='Skeleton metric is a centerline proximity measure, not outline accuracy for filled objects; asymmetric for unequal skeleton lengths. Report symmetric precision/recall F1 separately. r=0 is skeleton IoU, not ordinary IoU. Both masks empty score1, one empty score0.'),
        dataset_comparison='Search for original 32x32 crack test patches and author evaluation code; no claim of direct replication without verified access, exact split and metric choices.')


def check_protocol():
    path=ROOT/'xxz-two-seed-protocol.json';v=protocol()
    if path.exists():assert json.loads(path.read_text())==v,'Frozen protocol changed.'
    else:write_json(path,v)


def dispatch(specs,func,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(func,s) for s in specs]
        for i,f in enumerate(as_completed(jobs),1):
            result=f.result()
            if i%20==0 or i==len(jobs):print(json.dumps(dict(done=i,total=len(jobs),last=result)),flush=True)


def old_job(path_string):
    p=Path(path_string);old=json.loads(p.read_text());c=old['case'];stage=old['signature']['stage']
    source=INPUT/'cases'/f'{c["case_id"]}.npz'
    assert old['signature']['code_hashes']==legacy_hashes()
    assert old['signature']['input_sha256']==digest(source)
    f=np.load(source);arrays=np.load(p.with_suffix('.npz'))
    cr=float(np.max(rhat(scores(arrays['batches']))));converged=old['sampling']['converged'] and cr<1.1
    metrics,exact,inf=evaluate_arrays(arrays,f['truth'],f['seeds'],converged)
    baselines={k:dict(**mask_metrics(v,f['truth']),**bpm_metrics(v.reshape(f['truth'].shape),f['truth']))
        for k,v in classical_baselines(f['image'],f['truth'],f['seeds']).items()}
    row=dict(case=c,rule=old['signature']['rule'],rule_id=old['rule_id'],family=old['family'],
        stage=stage,decisions=metrics,exact=exact,parameters=old['parameters'],
        original_converged=old['sampling']['converged'],converged=converged,
        contrast_batch_rhat=cr if np.isfinite(cr) else None,baselines=baselines,
        source=str(p.relative_to(ROOT)),source_sha256=digest(p),array_sha256=digest(p.with_suffix('.npz')),
        code_hashes=hashes(),interval_multiplier=inf['multiplier'])
    write_json(OUT/'reanalysis'/stage/p.name,row)
    return c['case_id']


def job(spec):
    stage,c,rule=spec;rid=identifier(rule);path=OUT/stage/f'{c["case_id"]}--{rid}.json'
    input_path=INPUT/'cases'/f'{c["case_id"]}.npz'
    signature=dict(stage=stage,case=c,rule=rule,settings=SETTINGS[stage],code_hashes=hashes(),
        input_sha256=digest(input_path),rng_seed=seed_for('xxz-two-seed:'+stage+c['case_id']+rid))
    if path.exists():
        assert json.loads(path.read_text())['signature']==signature,'Cached source changed.'
        return c['case_id']
    start=time.perf_counter();f=np.load(input_path)
    arrays,param,diag=sample(f['image'],rule,f['seeds'],**SETTINGS[stage],rng_seed=signature['rng_seed'])
    metrics,exact,inf=evaluate_arrays(arrays,f['truth'],f['seeds'],diag['converged'])
    arrays.update(score=inf['score'],score_se=inf['se'],score_radius=inf['radius'])
    row=dict(signature=signature,case=c,rule=rule,rule_id=rid,family=family(rule),parameters=param,
        decisions=metrics,exact=exact,sampling=diag,seconds=time.perf_counter()-start,interval_multiplier=inf['multiplier'])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    write_json(path,row);return c['case_id']


def calibration_rows():
    return [json.loads((OUT/'calibrate'/f'{c["case_id"]}--{identifier(rule)}.json').read_text()) for c in cases('calibration') for rule in candidates()]


def select():
    rows=calibration_rows();scores_table=[]
    for rule in candidates():
        rid=identifier(rule);group=[r for r in rows if r['rule_id']==rid]
        assert len(group)==50
        for decision in DECISIONS[:2]:
            mean={k:float(np.mean([r['decisions'][decision][k] for r in group])) for k in group[0]['decisions'][decision]}
            scores_table.append(dict(rule=rule,rule_id=rid,family=family(rule),decision=decision,metrics=mean,
                converged=sum(r['sampling']['converged'] for r in group),cases=len(group)))
    def best(group,key='mixing_lower'):
        return min(group,key=lambda r:(-r['metrics'][key],-r['metrics']['iou'],r['rule_id']))
    winners={}
    for fam in sorted({r['family'] for r in scores_table}):
        winners[fam+'__two_max']=best([r for r in scores_table if r['family']==fam and r['decision']=='two_max'])
    for model in ('Ising','XXZ'):
        winners[model+'__one_background']=best([r for r in scores_table if r['family'].startswith(model+'/') and r['decision']=='one_background'])
        winners[model+'__accuracy_two_max']=best([r for r in scores_table if r['family'].startswith(model+'/') and r['decision']=='two_max'],'iou')
    for label,row in list(winners.items()):
        if label.startswith('XXZ') and row['decision']=='two_max':
            rule=dict(row['rule'],strength=0.);rid=identifier(rule)
            winners['matched_Ising_for_'+label]=next(r for r in scores_table if r['rule_id']==rid and r['decision']=='two_max')
    selected=dict(winners=winners,scores=scores_table,code_hashes=hashes(),
        calibration_hashes={p.name:digest(p) for p in sorted((OUT/'calibrate').glob('*.json'))})
    p=ROOT/'xxz-two-seed-selection.json'
    if p.exists():assert json.loads(p.read_text())==selected
    else:write_json(p,selected)
    for label,r in winners.items():print(label,r['rule'],'IoU',r['metrics']['iou'],'lower',r['metrics']['mixing_lower'],flush=True)


def selected_rules():
    d=json.loads((ROOT/'xxz-two-seed-selection.json').read_text());assert d['code_hashes']==hashes()
    return list({r['rule_id']:r['rule'] for r in d['winners'].values()}.values())


def confirmation_specs():
    fixed=set(sorted(manifest()['test_ids'])[:5]);specs=[]
    for c in cases('test'):
        for rule in selected_rules():
            p=OUT/'test'/f'{c["case_id"]}--{identifier(rule)}.json';r=json.loads(p.read_text());d=r['sampling']
            if not d['converged'] or d['minimum_score_ess']<200 or (c['source_id'] in fixed and c['shape']==[16,16]):
                specs.append(('confirm',c,rule))
    return specs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','reanalyse','calibrate','select','test','confirm']);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    check_protocol()
    if args.stage=='init':print('Frozen:',len(candidates()),'rules,',len(cases('calibration')),'calibration cases');return
    if args.stage=='reanalyse':
        paths=[str(p) for stage in ('test','scan','small','confirm') for p in sorted((LEGACY/stage).glob('*.json'))]
        dispatch(paths,old_job,args.workers);return
    if args.stage=='select':select();return
    if args.stage=='calibrate':specs=[('calibrate',c,r) for c in cases('calibration') for r in candidates()]
    elif args.stage=='test':specs=[('test',c,r) for c in cases('test') for r in selected_rules()]
    else:specs=confirmation_specs()
    dispatch(specs,job,args.workers)


if __name__=='__main__':main()
