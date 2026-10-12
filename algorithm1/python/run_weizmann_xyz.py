"""Frozen small-system XYZ/alignment experiment on Weizmann source images."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from xyz_alignment import (parameterize,thermal_distributions,correlations,draw_correlations,
    metrics,uncertainty_radius,DECISIONS,decision_score)
from weizmann_petz_data import (ROOT,OUT as OLD,ARCHIVE_HASH,decode_annotation,seed_pixels,seed_for,write_json)
from weizmann_petz_metrics import classical_baselines,mask_metrics

OUT=ROOT/'results'/'weizmann-xyz'
BUDGETS=[6144,24576,98304]
PRIMARY_BUDGET=24576
METRICS=['iou','dice','accuracy','iou_lower','iou_upper','resolved','correct_resolved',
    'wrong_resolved','predicted_foreground','unknown_iou','minimum_margin','seed_accuracy']
OBSERVABLES=['alignment','zz']


def identifier(rule):return hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()[:12]


def coupling_identifier(rule):return identifier({k:v for k,v in rule.items() if k not in ('K','normalization')})


def hashes():
    names=['xyz_alignment.py','run_weizmann_xyz.py','weizmann_petz_data.py','weizmann_petz_metrics.py','ising_thresholds.py','ising_algorithm1.py']
    return {n:hashlib.sha256((ROOT/'python'/n).read_bytes()).hexdigest() for n in names}


def candidates():
    rules=[]
    for threshold,sensitivity,k in itertools.product(('fixed50','edge_otsu','q75'),('fixed001','fixed01','inverse_spread'),(.4,.8,1.6)):
        rules.append(dict(model='Ising',threshold=threshold,width=0.,sensitivity=sensitivity,xy_scale=0.,K=k,normalization='sum'))
    bundles=[(t,w) for t in ('fixed50','edge_otsu','q75') for w in (0.,.25,.5)]
    bundles += [(t,0.) for t in ('quantiles2575','quantiles5090','fixed_triple')]
    for (threshold,width),sensitivity,xy,k,norm in itertools.product(bundles,('fixed001','fixed01','inverse_spread'),(.25,1.),(.4,.8,1.6),('sum','max')):
        rules.append(dict(model='XYZ',threshold=threshold,width=width,sensitivity=sensitivity,xy_scale=xy,K=k,normalization=norm))
    return rules


def tied(rule):return rule['width']==0 and rule['threshold'] in ('fixed50','edge_otsu','q75')


def manifest():return json.loads((ROOT/'weizmann-xyz-manifest.json').read_text())


def prepare():
    if (ROOT/'weizmann-xyz-manifest.json').exists():
        for case in manifest()['cases']:source_case(case).close()
        print('Existing inputs verified; no files rewritten.',flush=True);return
    old=json.loads((ROOT/'weizmann-petz-manifest.json').read_text())
    archive=OLD/'data'/'Weizmann_Seg_DB_1obj.ZIP'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_HASH:raise ValueError('Archive changed.')
    cases=[];excluded=[]
    with zipfile.ZipFile(archive) as z:
        for s in old['sources']:
            image=np.array(Image.open(io.BytesIO(z.read(s['image_file']))).convert('L'))
            masks=np.stack([decode_annotation(np.array(Image.open(io.BytesIO(z.read(a))).convert('RGB'))) for a in s['annotation_files']])
            majority=masks.sum(axis=0)>=2
            for shape,domain in itertools.product(((3,3),(3,4)),('native','full')):
                h,w=shape;name=f'{s["source_id"]}--{domain}-{h}x{w}'
                meta=dict(case_id=name,source_id=s['source_id'],split=s['split'],shape=list(shape),domain=domain)
                if domain=='native':
                    counts=convolve2d(majority.astype(int),np.ones(shape,int),mode='valid')
                    locations=np.argwhere((counts>=np.ceil(h*w*.25))&(counts<=np.floor(h*w*.75)))
                    if not len(locations):
                        excluded.append(dict(**meta,reason='No 25-75 percent foreground native patch'));continue
                    rng=np.random.default_rng(seed_for('xyz-crop:'+name));r,c=locations[rng.integers(len(locations))]
                    small=image[r:r+h,c:c+w].astype(float);small_masks=masks[:,r:r+h,c:c+w]
                    meta.update(top_left=[int(r),int(c)],eligible_windows=len(locations))
                else:
                    small=np.array(Image.fromarray(image.astype(np.float32)).resize((w,h),Image.Resampling.BOX),float)
                    small_masks=np.stack([np.array(Image.fromarray(a.astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST),bool) for a in masks])
                truth=small_masks.sum(axis=0)>=2;seeds=seed_pixels(truth)
                if len(seeds)!=2:
                    excluded.append(dict(**meta,reason='Resized majority mask contains only one class'));continue
                path=OUT/'cases'/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,image=small,truth=truth,masks=small_masks,seeds=seeds)
                meta.update(seeds=seeds,foreground_fraction=float(truth.mean()),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                cases.append(meta)
    result=dict(date='2026-10-08',archive_sha256=ARCHIVE_HASH,calibration_ids=old['calibration_ids'],test_ids=old['test_ids'],
        cases=cases,excluded=excluded,source_split='Same original 25/75 source split; all resolutions and domains stay with source.',
        selection_note='Native windows selected from masks before any Hamiltonian evaluation. Full-frame one-class masks excluded before outcomes and listed explicitly.')
    write_json(ROOT/'weizmann-xyz-manifest.json',result)
    print('Prepared',len(cases),'cases;',len(excluded),'predeclared one-class/window exclusions.',flush=True)


def initialize():
    value=dict(date='2026-10-08',model='H=-sum_e [Jx XX+Jy YY+Jz ZZ]/4; Ja=wa*atan(alpha*(tau_a-d)); wx=wy=xy_scale,wz=1',
        alignment='A=<XX+YY+ZZ>; internal mean a=A/3 for equal-copy comparison, with self a=1 and distinct-pair a in [-1,1/3].',
        threshold_rules='Central tau is fixed50, edge-Otsu or positive-edge q75. tau_a=tau+[-width,0,width]*S, S=max(q90(d)-q10(d),1). No clipping. Independent triplets are quantiles(25,50,75), quantiles(50,75,90) of all neighbor differences, or (25,75,125).',
        coupling_rules='alpha=.001,.01,1/S; wx=wy=.25 or1; K=.4,.8,1.6; b=4K/median(sum_a |Ja|) or4K/median(max_a |Ja|). Median-zero fallback max(mean,.001). beta_paper=N*b.',
        decisions='Background: -a_bg > t, t=-.1,0,.1. Two-seed contrast: a_fg-a_bg >0. Same rules also evaluated using ZZ only. Ties predict background; seed labels not clamped.',
        calibration='All 459 rules on native and full-frame 3x3 calibration cases. Average domains within each original source, then average sources. Three independent measurement replicates at 24576 Gibbs copies.',
        primary_selection='Per family maximize mean worst-case IoU over unresolved labels, then exact IoU, then sampled IoU, then lexical IDs. Report raw-exact-IoU winners separately. Never tune on test outcomes.',
        families=['Ising_zero','Ising_flexible','XYZ_tied_zero','XYZ_tied_flexible','XYZ_staggered_zero','XYZ_staggered_flexible','XYZ_staggered_ZZ','Ising_exact_best','XYZ_exact_best'],
        test='All family winners on eligible 3x3 test cases. Extend Ising_zero, Ising_flexible, XYZ_tied_flexible, XYZ_staggered_zero and XYZ_staggered_flexible to eligible 3x4 test cases.',
        controls='For XYZ_staggered_flexible: same Hamiltonian ZZ and alignment readouts; also Jz-only at the same b, and collapsed threshold triple at its arithmetic mean at both unchanged b and matched median total interaction scale.',
        samples='Independent global Pauli-basis projective samples from exact Gibbs distributions; no MCMC sign problem, mixing error, tensor truncation or circuit implementation. Alignment splits total copies equally across X/Y/Z; ZZ spends all copies on Z. Both seeds estimated from each outcome.',
        uncertainty='Hoeffding union bound, 95 percent jointly over both observables, background/contrast scores, all pixels and three budgets per model/replicate. Contrast radius twice background radius. Mean-IoU ranges allow every unresolved label assignment. Resolution excludes both known seeds; confidently incorrect labels reported.',
        budgets=BUDGETS,test_replicates=16,primary_budget=PRIMARY_BUDGET,
        exact='All parity sectors, dense exact spectra through12 spins. These experiments do not establish scalability to previous64/256-spin benchmarks. No claim of quantum speedup.',
        reuse='Sources used in previous development; internal held-out exploratory comparison, not fresh external validation.',
        candidates=candidates(),code_hashes=hashes(),manifest_sha256=hashlib.sha256((ROOT/'weizmann-xyz-manifest.json').read_bytes()).hexdigest())
    path=ROOT/'weizmann-xyz-protocol.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen protocol changed.')
    write_json(path,value);print('Frozen',len(value['candidates']),'Hamiltonian rules.',flush=True)


def source_case(case):
    path=OUT/'cases'/f'{case["case_id"]}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=case['input_sha256']:raise ValueError('Input changed.')
    return np.load(path)


def calibrate_case(case):
    path=OUT/'calibration'/f'{case["case_id"]}.npz';meta_path=path.with_suffix('.json')
    signature=dict(case=case,code_hashes=hashes(),budget=PRIMARY_BUDGET,replicates=3)
    if path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text())['signature']!=signature:raise ValueError('Stale calibration.')
        return case['case_id']
    start=time.perf_counter();f=source_case(case);image,truth,seeds=f['image'],f['truth'],f['seeds']
    rules=candidates();groups=defaultdict(list)
    for i,r in enumerate(rules):groups[coupling_identifier(r)].append(i)
    exact_metrics=np.zeros((len(rules),2,4,len(METRICS)));sample_metrics=np.zeros_like(exact_metrics)
    corr=np.zeros((len(rules),3,2,image.size));residual=0.;params=[]
    for r in rules:params.append(parameterize(image,r)[2])
    for indices in groups.values():
        edge,j,_=parameterize(image,rules[indices[0]])
        p,diag=thermal_distributions(image.size,edge,j,[params[i]['b'] for i in indices]);residual=max(residual,diag['eigen_residual'])
        for pi,i in enumerate(indices):
            corr[i]=correlations(p[pi],seeds)
            for oi,obs in enumerate(OBSERVABLES):
                target=corr[i].mean(axis=0) if obs=='alignment' else corr[i,2]
                for di,decision in enumerate(DECISIONS):
                    m=metrics(target,truth,seeds,decision);exact_metrics[i,oi,di]=[m[k] for k in METRICS]
                for replicate in range(3):
                    rng=np.random.default_rng(seed_for('xyz-cal:'+case['case_id']+identifier(rules[i])+obs+str(replicate)))
                    measured=draw_correlations(p[pi],seeds,PRIMARY_BUDGET,rng,obs)
                    for di,decision in enumerate(DECISIONS):
                        m=metrics(measured,truth,seeds,decision,uncertainty_radius(image.size,PRIMARY_BUDGET,decision))
                        sample_metrics[i,oi,di]+=[m[k]/3 for k in METRICS]
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,exact_metrics=exact_metrics,sample_metrics=sample_metrics,correlations=corr)
    write_json(meta_path,dict(signature=signature,parameters=params,eigen_residual=residual,seconds=time.perf_counter()-start))
    return case['case_id']


def calibration_cases():return [c for c in manifest()['cases'] if c['split']=='calibration' and c['shape']==[3,3]]


def select():
    groups=defaultdict(list)
    for c in calibration_cases():groups[c['source_id']].append(c)
    sampled=[];exact=[]
    for items in groups.values():
        for c in items:
            path=OUT/'calibration'/f'{c["case_id"]}.npz'
            expected=dict(case=c,code_hashes=hashes(),budget=PRIMARY_BUDGET,replicates=3)
            if json.loads(path.with_suffix('.json').read_text())['signature']!=expected:raise ValueError('Stale calibration selection.')
    if len(groups)!=25:raise ValueError('Expected all 25 calibration sources.')
    # Average domains inside each source before weighting sources equally.
    for items in groups.values():
        files=[np.load(OUT/'calibration'/f'{c["case_id"]}.npz') for c in items]
        sampled.append(np.mean([f['sample_metrics'] for f in files],axis=0))
        exact.append(np.mean([f['exact_metrics'] for f in files],axis=0))
    sampled=np.mean(sampled,axis=0);exact=np.mean(exact,axis=0);rules=candidates();records=[]
    for ri,r in enumerate(rules):
        for oi,obs in enumerate(OBSERVABLES):
            for di,decision in enumerate(DECISIONS):
                records.append(dict(rule=r,rule_id=identifier(r),observable=obs,decision=decision,decision_id=di,
                    exact=dict(zip(METRICS,exact[ri,oi,di].tolist())),sample=dict(zip(METRICS,sampled[ri,oi,di].tolist()))))
    def eligible(row,family):
        r=row['rule'];obs=row['observable'];di=row['decision_id']
        if family.startswith('Ising'):return r['model']=='Ising' and obs=='zz' and (family!='Ising_zero' or di==1)
        if r['model']!='XYZ':return False
        if family=='XYZ_exact_best':return not tied(r) and obs=='alignment'
        is_tied='tied' in family and 'staggered' not in family
        return tied(r)==is_tied and obs==('zz' if family.endswith('_ZZ') else 'alignment') and (not family.endswith('_zero') or di==1)
    families=json.loads((ROOT/'weizmann-xyz-protocol.json').read_text())['families'];winners={}
    for family in families:
        rows=[r for r in records if eligible(r,family)]
        rows.sort(key=lambda r:(-r['exact']['iou'] if family.endswith('exact_best') else -r['sample']['iou_lower'],
            -r['exact']['iou'],-r['sample']['iou'],r['rule_id'],r['decision_id']))
        winners[family]=rows[0]
    write_json(ROOT/'weizmann-xyz-selection.json',dict(winners=winners,calibration_sources=len(groups),calibration_cases=len(calibration_cases()),code_hashes=hashes()))
    write_json(ROOT/'weizmann-xyz-calibration-scores.json',dict(scores=records,code_hashes=hashes()))
    print(json.dumps(winners,indent=2),flush=True)


def entries(extension=False):
    winners=json.loads((ROOT/'weizmann-xyz-selection.json').read_text())['winners']
    families=['Ising_zero','Ising_flexible','XYZ_tied_flexible','XYZ_staggered_zero','XYZ_staggered_flexible'] if extension else list(winners)
    return {family:winners[family] for family in families}


def test_case(case):
    path=OUT/'test'/f'{case["case_id"]}.json';selected=entries(case['shape']==[3,4])
    signature=dict(case=case,selection=selected,code_hashes=hashes(),budgets=BUDGETS,replicates=16)
    if path.exists():
        if json.loads(path.read_text())['signature']!=signature:raise ValueError('Stale comparison.')
        return case['case_id']
    start=time.perf_counter();f=source_case(case);image,truth,seeds=f['image'],f['truth'],f['seeds']
    groups=defaultdict(list)
    for label,entry in selected.items():groups[coupling_identifier(entry['rule'])].append((label,entry))
    results=[];arrays={};residual=0.
    for rows in groups.values():
        edge,j,_=parameterize(image,rows[0][1]['rule']);params=[parameterize(image,r['rule'])[2] for _,r in rows]
        p,diag=thermal_distributions(image.size,edge,j,[r['b'] for r in params]);residual=max(residual,diag['eigen_residual'])
        for i,(label,entry) in enumerate(rows):
            results.append(evaluate_entry(case,p[i],truth,seeds,label,entry,params[i],arrays))
            if label=='XYZ_staggered_flexible':primary_p=p[i].copy()
    # Mechanism controls matched to the primary staggered/flexible winner.
    entry=selected['XYZ_staggered_flexible'];edge,j,param=parameterize(image,entry['rule']);b=param['b']
    control_j=j.copy();control_j[:,:2]=0
    p,_=thermal_distributions(image.size,edge,control_j,[b])
    results.append(evaluate_entry(case,p[0],truth,seeds,'matched_Jz_only',dict(entry,observable='zz'),
        dict(param,control='Set Jx=Jy=0; retain original Jz and b; other metadata describe source XYZ rule.'),arrays))
    tau=float(np.mean(param['thresholds']));d=abs(image.ravel()[edge[:,0]]-image.ravel()[edge[:,1]])
    collapsed=np.repeat(np.arctan(param['alpha']*(tau-d))[:,None],3,axis=1)
    collapsed[:,:2]*=entry['rule']['xy_scale']
    scale=np.median(abs(collapsed).sum(axis=1));scale=max(float(abs(collapsed).sum(axis=1).mean()),.001) if scale<1e-8 else float(scale)
    matched_b=b*param['scales']['sum']/scale
    p,diag=thermal_distributions(image.size,edge,collapsed,[b,matched_b]);residual=max(residual,diag['eigen_residual'])
    for i,label in enumerate(['collapsed_same_b','collapsed_matched_scale']):
        control_b=b if i==0 else matched_b
        results.append(evaluate_entry(case,p[i],truth,seeds,label,entry,
            dict(param,thresholds=[tau]*3,b=control_b,beta_paper=image.size*control_b,distinct_thresholds=1,
                 control='Collapse thresholds to their mean; other diagnostics describe source XYZ rule.',collapsed_sum_scale=scale),arrays))
    # Same XYZ state with the same decision but ZZ readout: no extra diagonalization.
    results.append(evaluate_entry(case,primary_p,truth,seeds,'matched_XYZ_ZZ',dict(entry,observable='zz'),param,arrays))
    baseline={k:mask_metrics(v,truth) for k,v in classical_baselines(image,truth,seeds).items()}
    pred=np.ones(image.size,bool);pred[seeds[0]]=False;baseline['foreground_except_bg']=mask_metrics(pred,truth)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    write_json(path,dict(signature=signature,case=case,results=results,baselines=baseline,
        eigen_residual=residual,seconds=time.perf_counter()-start))
    return case['case_id']


def evaluate_entry(case,p,truth,seeds,label,entry,params,arrays):
    axes=correlations(p,seeds);obs=entry['observable'];decision=entry['decision']
    target=axes.mean(axis=0) if obs=='alignment' else axes[2]
    reference=metrics(target,truth,seeds,decision);budgets={};sampled=np.zeros((3,16,2,truth.size))
    unknown=np.ones(truth.size,bool);unknown[seeds]=False
    for bi,budget in enumerate(BUDGETS):
        values=[];coverage=[];wrong_sign=[]
        for rep in range(16):
            rng=np.random.default_rng(seed_for('xyz-test:'+case['case_id']+label+str(budget)+':'+str(rep)))
            measured=draw_correlations(p,seeds,budget,rng,obs);sampled[bi,rep]=measured
            radius=uncertainty_radius(truth.size,budget,decision)
            values.append(metrics(measured,truth,seeds,decision,radius))
            err=abs(decision_score(measured,decision)-decision_score(target,decision))
            coverage.append(bool(np.all(err<=radius+1e-12)))
            wrong_sign.append(float(np.mean(((decision_score(measured,decision)>decision['threshold']+1e-12)!=(decision_score(target,decision)>decision['threshold']+1e-12))[unknown])))
        budgets[str(budget)]=dict(mean={k:float(np.mean([v[k] for v in values])) for k in METRICS},
            sd={k:float(np.std([v[k] for v in values],ddof=1)) for k in METRICS},
            simultaneous_coverage_fraction=float(np.mean(coverage)),label_disagreement_with_exact=float(np.mean(wrong_sign)))
    arrays[label+'_axes']=axes;arrays[label+'_samples']=sampled
    return dict(label=label,entry=entry,parameters=params,exact=reference,budgets=budgets)


def dispatch(cases,fn,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fn,c) for c in cases]
        for done,f in enumerate(as_completed(futures),1):print(json.dumps(dict(done=done,total=len(cases),case=f.result())),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','init','calibrate','select','test','extend'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='init':initialize();return
    protocol=json.loads((ROOT/'weizmann-xyz-protocol.json').read_text())
    if protocol['code_hashes']!=hashes():raise ValueError('Frozen code changed.')
    if protocol['manifest_sha256']!=hashlib.sha256((ROOT/'weizmann-xyz-manifest.json').read_bytes()).hexdigest():raise ValueError('Frozen manifest changed.')
    if args.stage=='select':select();return
    if args.stage=='calibrate':dispatch(calibration_cases(),calibrate_case,args.workers);return
    shape=[3,3] if args.stage=='test' else [3,4]
    cases=[c for c in manifest()['cases'] if c['split']=='test' and c['shape']==shape]
    dispatch(cases,test_case,args.workers)


if __name__=='__main__':main()
