"""Label-blind larger-budget check of frozen primary XXZ predictions."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import numpy as np
import torch
from correlation_nn_data import ROOT,OUT,INPUT,init,moments,digest
from promotion_reliability_bank import iter_tail
from correlation_nn import Decoder,probabilities
from run_correlation_nn import dataset
from weizmann_petz_data import seed_for,write_json


def confirm_job(item):
    case,rid,rule=item;target=OUT/'confirmation'/rid/f'{case["case_id"]}.npz'
    if target.exists() and target.with_suffix('.json').exists():return rid,case['case_id'],'cached'
    start=time.perf_counter();data=np.load(INPUT/f'{case["case_id"]}.npz')
    packed,param,diag=next(iter_tail(data['image'],rule,rounds=1,samples=8192,burn=6000,thin=2,chains=4,
        rng_seed=seed_for('correlation-nn:confirmation:'+case['case_id']+':'+rid)))
    c,se,rh,ess=moments(packed,256,(8192,));off=~np.eye(256,dtype=bool)
    converged=bool((rh[off]<1.05).all() and diag['scalar_rhat'] is not None and diag['scalar_rhat']<1.05)
    target.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(target,corr=c[0],se=se,rhat=rh,ess=ess,packed=packed)
    write_json(target.with_suffix('.json'),dict(case_id=case['case_id'],rule=rule,parameters=param,sampling=diag,
        converged=converged,failed_pair_fraction=float(np.mean(rh[off]>=1.05)),max_pair_rhat=float(rh[off].max()) if np.isfinite(rh[off]).all() else None,
        seconds=time.perf_counter()-start,sha256=digest(target),samples_per_chain=8192,burn=6000))
    return rid,case['case_id'],round(time.perf_counter()-start,2)


def plan(scope='primary'):
    p=init();s=json.loads((ROOT/'correlation-nn-selection.json').read_text());ids=[c['source_id'] for c in p['cases']];jobs={}
    for name,v in s.items():
        prefix='XXZ/' if scope=='primary' else 'XXZ-unrestricted/'
        if not name.startswith(prefix):continue
        j=v['job'];rid=j['rid'];d=json.loads((OUT/f'diagnostics-{rid}.json').read_text())
        for sid in p['folds'][j['fold']]['test']:
            i=ids.index(sid)
            if not d[i]['converged']:jobs[rid+'/'+sid]=(p['cases'][i],rid,p['rules'][rid])
    filename='correlation-nn-confirmation-protocol.json' if scope=='primary' else 'correlation-nn-unrestricted-confirmation-protocol.json'
    write_json(ROOT/filename,dict(jobs=list(jobs),scope=scope,criterion='Selected XXZ in this scope, either seed mode, any failed full-pair/scalar convergence. Test labels and prediction errors not used.',settings=dict(chains=4,samples=8192,burn=6000,thin=2)))
    return list(jobs.values())


def inference():
    from analyze_correlation_nn import ensemble_predictions,ious,aggregate,metric_arrays
    p=init();s=json.loads((ROOT/'correlation-nn-final-selection.json').read_text());im,y,seed,_,ids=dataset()
    secondary=ROOT/'correlation-nn-unrestricted-confirmation-protocol.json'
    if secondary.exists():
        for job in json.loads(secondary.read_text())['jobs']:
            rid,sid=job.split('/')
            assert (OUT/'confirmation'/rid/f'{sid}--square-16.json').exists(),'Unrestricted confirmation sampling is unfinished.'
    original=ensemble_predictions(s);results={}
    groups=['XXZ/none','XXZ/foreground']+(['XXZ-unrestricted/none','XXZ-unrestricted/foreground'] if secondary.exists() else [])
    for group in groups:
        mode=group.split('/')[1];pred=original[group]['prob'].copy();changed=[];remaining=[];bank_hashes={}
        for fold in range(5):
            selection=s[f'{group}/{fold}'];j=selection['job'];rid=j['rid']
            indices=[ids.index(i) for i in p['folds'][fold]['test']]
            use=[i for i in indices if (OUT/'confirmation'/rid/f'{p["cases"][i]["case_id"]}.npz').exists()]
            if not use:continue
            arrays=[]
            for i in use:
                path=OUT/'confirmation'/rid/f'{p["cases"][i]["case_id"]}.npz';m=json.loads(path.with_suffix('.json').read_text())
                assert digest(path)==m['sha256'];bank_hashes[str(path)]=m['sha256']
                arrays.append(np.load(path)['corr']);remaining.append(not m['converged'])
            c=torch.tensor(np.stack(arrays));image=torch.tensor(im[use]).flatten(1);seeds=torch.tensor(seed[use]);reps=[]
            for rep in range(3):
                model=Decoder(16,j['config']['width'],j['config']['architecture'],j['input_kind'],mode=='foreground')
                path=OUT/'final'/f'{group.replace("/","--")}--{fold}--r{rep}.pt'
                model.load_state_dict(torch.load(path,weights_only=True));model.eval()
                with torch.no_grad():reps.append(probabilities(model,c,image,seeds).numpy())
            pred[use]=np.mean(reps,0);changed+=use
        threshold=original[group]['threshold'];old=ious(original[group]['pred'],y);new=ious(pred>threshold[:,None],y)
        prior=ious(original['prior/'+mode]['pred'],y)
        measured=metric_arrays(pred>threshold[:,None],y,pred,threshold,seed if mode=='foreground' else None)
        results[group]=dict(changed_images=len(changed),remaining_failed=sum(remaining),iou=aggregate(new),
            metrics={key:aggregate(value) for key,value in measured.items()},
            difference_vs_original=aggregate(new-old),difference_vs_prior=aggregate(new-prior),
            difference_vs_ising=aggregate(new-ious(original['Ising/'+mode]['pred'],y)),
            difference_vs_primary_xxz=aggregate(new-ious(original['XXZ/'+mode]['pred'],y)),
            mean_retained_configurations=float(8192+len(changed)/100*(32768-8192)),
            mean_total_configurations_spent=float(8192+len(changed)/100*32768),bank_sha256=bank_hashes)
        suffix=mode if group.startswith('XXZ/') else 'unrestricted-'+mode
        np.savez_compressed(OUT/f'confirmation-predictions-{suffix}.npz',probabilities=pred,prediction=pred>threshold[:,None],iou=new)
    write_json(ROOT/'correlation-nn-confirmation-summary.json',results)
    report=ROOT/'correlation-nn.md';heading='## Higher-budget XXZ confirmation'
    if report.exists():
        lines=['',heading,'',
            'This check changes measurement inputs only. Networks, thresholds, folds, and Hamiltonian choices remain frozen. Cases are selected using failed sampling diagnostics, not segmentation errors. No case is dropped.','',
            '| Seed information | Resampled images | Still failing | IoU [95% CI] | Boundary proximity | Change in IoU | Mean configurations in final input |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name,v in results.items():
            q=v['iou'];delta=v['difference_vs_original']['mean']
            lines.append(f'| {name} | {v["changed_images"]} | {v["remaining_failed"]} | {q["mean"]:.4f} [{q["ci95"][0]:.4f}, {q["ci95"][1]:.4f}] | {v["metrics"]["bpm"]["mean"]:.4f} | {delta:+.4f} | {v["mean_retained_configurations"]:.0f} |')
        if not any(v['changed_images'] for k,v in results.items() if k.startswith('XXZ/')):
            lines+=['','All selected primary XXZ test cases passed the initial diagnostics and retain their original banks. No independent extra bank was generated for these primary cases.']
        if any(v['changed_images'] for v in results.values()):
            lines+=['','Resampled cases use four chains × 8,192 retained configurations, replacing the original four × 2,048 bank. The original bank was still generated: the JSON report separately records total configurations spent, including discarded inputs. Remaining failures continue to limit a Gibbs-state interpretation.']
        if secondary.exists():
            lines+=['','The unrestricted rows are secondary diagnostics. Their networks were trained on the original, sometimes unconverged banks; replacing only flagged test inputs can change the input distribution. These rows do not replace the primary reliably sampled comparison or constitute retraining on converged Gibbs measurements.']
        report.write_text(report.read_text().split('\n'+heading)[0].rstrip()+'\n'+'\n'.join(lines)+'\n')
    print('confirmation inference',results,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['sample','infer']);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--scope',choices=['primary','unrestricted','both'],default='primary');a=ap.parse_args()
    if a.stage=='sample':
        jobs=plan('primary')+plan('unrestricted') if a.scope=='both' else plan(a.scope)
        jobs=list({(rid,case['case_id']):(case,rid,rule) for case,rid,rule in jobs}.values())
        print('confirmation measurement jobs',len(jobs),flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(confirm_job,j) for j in jobs]):print(f.result(),flush=True)
    else:inference()
