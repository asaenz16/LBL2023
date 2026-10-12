"""Nested image-level training, validation, and locked outer-test predictions."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed,wait,FIRST_COMPLETED
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import torch
from correlation_nn_data import ROOT,OUT,INPUT,init,rule_id,digest
from correlation_nn import train,probabilities,iou_values
from weizmann_petz_data import write_json,seed_for


@lru_cache(maxsize=1)
def dataset():
    p=init();values=[np.load(INPUT/f'{c["case_id"]}.npz') for c in p['cases']]
    image=np.stack([v['image']/255 for v in values]).astype(np.float32)
    truth=np.stack([v['truth'] for v in values]);seeds=np.stack([v['seeds'] for v in values])
    seed=np.zeros((100,256),np.float32);seed[np.arange(100),seeds[:,1]]=1
    ids=[c['source_id'] for c in p['cases']]
    return image,truth,seed,seeds,ids


@lru_cache(maxsize=3)
def correlations(rid):
    if rid is None:return np.zeros((100,256,256),np.float32)
    cached=OUT/'training-cache'/f'{rid}.npy'
    if cached.exists():return np.load(cached)
    path=OUT/f'correlations-{rid}.npy'
    if path.exists():return np.asarray(np.load(path,mmap_mode='r')[:,-1]).copy()
    p=init();return np.stack([np.load(OUT/'measurements'/rid/f'{c["case_id"]}.npz')['corr'][-1] for c in p['cases']])


def descriptor(fold,rid,mode,config,kind='full'):
    return dict(fold=fold,rid=rid,seed_mode=mode,config=config,input_kind=kind)


def job_id(job):return rule_id(job)


def fit_job(job):
    tag=job_id(job);path=OUT/'fits'/f'{tag}.json'
    if path.exists():return tag,'cached'
    p=init();fold=p['folds'][job['fold']];im,y,seed,_,ids=dataset();c=correlations(job['rid'])
    tr=[ids.index(s) for s in fold['train']];va=[ids.index(s) for s in fold['validation']]
    rng=seed_for(f'correlation-nn:screen:{job["fold"]}:{job["seed_mode"]}')
    model,result,pred=train(c,im,seed,y,tr,va,job['config'],input_kind=job['input_kind'],seeded=job['seed_mode']=='foreground',rng_seed=rng)
    path.parent.mkdir(exist_ok=True,parents=True)
    torch.save(model.state_dict(),path.with_suffix('.pt'))
    np.savez_compressed(path.with_suffix('.npz'),validation_probabilities=pred,validation_indices=va)
    write_json(path,dict(job=job,result=result,train_ids=fold['train'],validation_ids=fold['validation'],test_ids_never_read=fold['test'],
        source_sha256={n:digest(Path(__file__).parent/n) for n in ('correlation_nn.py','run_correlation_nn.py')},checkpoint_sha256=digest(path.with_suffix('.pt'))))
    return tag,round(result['seconds'],2),round(result['validation_iou'],4)


def run_jobs(jobs,workers,stage):
    # All jobs are specified before reading outer-test outcomes.
    unique={job_id(j):j for j in jobs};write_json(OUT/f'{stage}-jobs.json',unique)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k,f in enumerate(as_completed([pool.submit(fit_job,j) for j in unique.values()])):
            result=f.result()
            if (k+1)%5==0:print(stage,k+1,'/',len(unique),result,flush=True)


def screen_jobs(p):
    return [descriptor(f,rid,m,p['network_screen']) for f in range(5) for rid in p['rules'] for m in p['seed_modes']]


def screen_ready(p,workers):
    """Overlap independent model fitting with measurement generation."""
    jobs=screen_jobs(p);write_json(OUT/'screen-jobs.json',{job_id(j):j for j in jobs})
    pending=list(jobs);ready=set();running={};finished=0;start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            for rid in p['rules']:
                if rid not in ready and all((OUT/'measurements'/rid/f'{c["case_id"]}.json').exists() for c in p['cases']):
                    cached=OUT/'training-cache'/f'{rid}.npy';cached.parent.mkdir(exist_ok=True,parents=True)
                    if not cached.exists():np.save(cached,np.stack([np.load(OUT/'measurements'/rid/f'{c["case_id"]}.npz')['corr'][-1] for c in p['cases']]))
                    ready.add(rid)
            for job in list(pending):
                if len(running)>=2*workers:break
                if job['rid'] in ready:
                    running[pool.submit(fit_job,job)]=job;pending.remove(job)
            if running:
                done,_=wait(running,timeout=3,return_when=FIRST_COMPLETED)
                for f in done:
                    value=f.result();running.pop(f);finished+=1
                    if finished%5==0:print('screen',finished,'/',len(jobs),'ready',len(ready),value,flush=True)
            else:time.sleep(3)
            if time.time()-start>12*3600:raise TimeoutError('Measurement/training pipeline exceeded 12 hours.')
            log=ROOT/'results/correlation-nn-sampling.log'
            if len(ready)<len(p['rules']) and log.exists() and 'Traceback (most recent call last)' in log.read_text():
                raise RuntimeError('Measurement generation failed; inspect sampling log.')


def available_fits():
    return [json.loads(path.read_text()) for path in sorted((OUT/'fits').glob('*.json'))]


def historical_rule(p,model):
    target=dict(threshold='q90',sensitivity='inverse_spread' if model=='Ising' else 'fixed01',K=1.2 if model=='Ising' else .8,
        transverse='relative',strength=0. if model=='Ising' else .1,temperature='longitudinal')
    rid=rule_id(target);assert rid in p['rules'];return rid


def reliable_rules(p,fold):
    """Sampling-quality eligibility uses only this fold's training/validation IDs."""
    ids=[c['source_id'] for c in p['cases']];f=p['folds'][fold]
    indices=[ids.index(s) for s in f['train']+f['validation']]
    result=[]
    for rid in p['rules']:
        path=OUT/f'diagnostics-{rid}.json'
        d=json.loads(path.read_text()) if path.exists() else [json.loads((OUT/'measurements'/rid/f'{c["case_id"]}.json').read_text())['diagnostics'] for c in p['cases']]
        if np.mean([d[i]['converged'] for i in indices])>=.95:result.append(rid)
    return result


def refine_jobs(p):
    fitted=available_fits();jobs=[]
    for f in range(5):
        reliable=set(reliable_rules(p,f))
        for mode in p['seed_modes']:
            for model in ('Ising','XXZ'):
                eligible=[v for v in fitted if v['job']['fold']==f and v['job']['seed_mode']==mode and v['job']['input_kind']=='full'
                    and v['job']['config']==p['network_screen']
                    and (p['rules'][v['job']['rid']]['strength']==0)==(model=='Ising')]
                eligible.sort(key=lambda v:(-v['result']['validation_iou'],job_id(v['job'])))
                shortlist=list(dict.fromkeys([v['job']['rid'] for v in eligible[:2]]+
                    [v['job']['rid'] for v in eligible if v['job']['rid'] in reliable][:2]+[historical_rule(p,model)]))
                for rid in shortlist:
                    for config in p['network_refine']:jobs.append(descriptor(f,rid,mode,config))
            for kind in ('prior','image'):
                for config in [p['network_screen']]+p['network_refine']:
                    jobs.append(descriptor(f,None,mode,config,kind))
    return jobs


def choose(p):
    fits=available_fits();selected={}
    def best(eligible):
        if not eligible:raise ValueError('Missing training/validation fits')
        return min(eligible,key=lambda v:(-v['result']['validation_iou'],v['result']['parameters'],job_id(v['job'])))
    for f in range(5):
        reliable=set(reliable_rules(p,f))
        for mode in p['seed_modes']:
            local=[v for v in fits if v['job']['fold']==f and v['job']['seed_mode']==mode]
            for model in ('Ising','XXZ'):
                a=[v for v in local if v['job']['input_kind']=='full' and (p['rules'][v['job']['rid']]['strength']==0)==(model=='Ising')]
                selected[f'{model}/{mode}/{f}']=best([v for v in a if v['job']['rid'] in reliable])
                if model=='XXZ':selected[f'XXZ-unrestricted/{mode}/{f}']=best(a)
                selected[f'{model}-historical/{mode}/{f}']=best([v for v in a if v['job']['rid']==historical_rule(p,model)])
            for kind in ('prior','image'):
                selected[f'{kind}/{mode}/{f}']=best([v for v in local if v['job']['input_kind']==kind])
    path=ROOT/'correlation-nn-selection.json'
    if path.exists():assert json.loads(path.read_text())==selected,'Selection changed after locking.'
    else:write_json(path,selected)
    return selected


def column_jobs(p,selected):
    jobs=[]
    for f in range(5):
        for model in ('Ising','XXZ'):
            rid=selected[f'{model}/foreground/{f}']['job']['rid']
            for config in [p['network_screen']]+p['network_refine']:
                jobs.append(descriptor(f,rid,'foreground',config,'column'))
    return jobs


def final_job(args):
    name,selection,rep=args
    target=OUT/'final'/f'{name.replace("/","--")}--r{rep}.json'
    if target.exists():return name,rep,'cached'
    p=init();job=selection['job'];fold=p['folds'][job['fold']]
    im,y,seed,_,ids=dataset();c=correlations(job['rid'])
    tr=[ids.index(s) for s in fold['train']+fold['validation']];test=[ids.index(s) for s in fold['test']]
    rng=seed_for(f'correlation-nn:refit:{job["fold"]}:{rep}:{job["seed_mode"]}')
    model,result,_=train(c,im,seed,y,tr,[],job['config'],input_kind=job['input_kind'],seeded=job['seed_mode']=='foreground',
        rng_seed=rng,epochs=selection['result']['epoch'])
    # Outer-test data enter only this evaluation after every choice is locked.
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32)
    with torch.no_grad():
        prediction=probabilities(model,tensor(c[test]),tensor(im[test]).flatten(1),tensor(seed[test])).numpy()
        noise=[]
        for budget in (0,1):
            cb=np.asarray(np.load(OUT/f'correlations-{job["rid"]}.npy',mmap_mode='r')[test,budget]) if job['rid'] is not None else c[test]
            noise.append(probabilities(model,tensor(cb),tensor(im[test]).flatten(1),tensor(seed[test])).numpy())
    target.parent.mkdir(exist_ok=True,parents=True);torch.save(model.state_dict(),target.with_suffix('.pt'))
    np.savez_compressed(target.with_suffix('.npz'),probabilities=prediction,prefix_probabilities=np.asarray(noise),test_indices=test)
    write_json(target,dict(name=name,selection_job=job,selection_id=job_id(job),selected_threshold=selection['result']['threshold'],
        result=result,replicate=rep,train_ids=fold['train']+fold['validation'],test_ids=fold['test'],checkpoint_sha256=digest(target.with_suffix('.pt'))))
    return name,rep,round(result['seconds'],2)


def finals(p,selected,workers):
    fits=available_fits()
    for f in range(5):
        for model in ('Ising','XXZ'):
            rid=selected[f'{model}/foreground/{f}']['job']['rid']
            eligible=[v for v in fits if v['job']['fold']==f and v['job']['input_kind']=='column' and v['job']['rid']==rid]
            selected[f'{model}-column/foreground/{f}']=min(eligible,key=lambda v:(-v['result']['validation_iou'],job_id(v['job'])))
    write_json(ROOT/'correlation-nn-final-selection.json',selected)
    jobs=[(k,v,r) for k,v in selected.items() for r in range(3)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k,f in enumerate(as_completed([pool.submit(final_job,j) for j in jobs])):
            result=f.result()
            if (k+1)%5==0:print('final',k+1,'/',len(jobs),result,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','screen-ready','refine','select','columns','final','all']);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();p=init();OUT.mkdir(exist_ok=True,parents=True)
    if a.stage in ('screen','all'):run_jobs(screen_jobs(p),a.workers,'screen')
    if a.stage=='screen-ready':screen_ready(p,a.workers)
    if a.stage in ('refine','all'):run_jobs(refine_jobs(p),a.workers,'refine')
    if a.stage in ('select','all'):choose(p)
    if a.stage in ('columns','all'):
        selected=json.loads((ROOT/'correlation-nn-selection.json').read_text());run_jobs(column_jobs(p,selected),a.workers,'columns')
    if a.stage in ('final','all'):
        selected=json.loads((ROOT/'correlation-nn-selection.json').read_text());finals(p,selected,a.workers)


if __name__=='__main__':main()
