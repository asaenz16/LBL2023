"""Label-independent all-pair ZZ measurements for the learned-decoder study."""
import os
for _k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from iterative_seed_gibbs import unpack
from promotion_reliability_bank import iter_tail
from weizmann_petz_data import seed_for,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/correlation-nn'
INPUT=ROOT/'results/weizmann-petz/cases'
PROTOCOL=ROOT/'correlation-nn-protocol.json'
BUDGETS=(512,1024,2048)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rule_id(rule):
    return hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()[:12]


def init():
    if PROTOCOL.exists():return json.loads(PROTOCOL.read_text())
    manifest=json.loads((ROOT/'weizmann-petz-manifest.json').read_text())
    cases=sorted([c for c in manifest['cases'] if c['family']=='natural-square' and c['shape']==[16,16]],key=lambda c:c['source_id'])
    assert len(cases)==100
    rules=[dict(threshold=t,sensitivity=a,K=k,transverse='relative',strength=g,temperature='longitudinal')
           for t,a,k,g in itertools.product(('edge_otsu','q90'),('fixed01','inverse_spread'),(.4,.8,1.2),(0.,.1,.5))]
    ids=[c['source_id'] for c in cases]
    ordered=sorted(ids,key=lambda s:seed_for('correlation-nn:outer:'+s))
    folds=[]
    for fold in range(5):
        test=ordered[20*fold:20*(fold+1)];rest=[s for s in ids if s not in test]
        inner=sorted(rest,key=lambda s:seed_for(f'correlation-nn:inner:{fold}:'+s))
        folds.append(dict(fold=fold,train=inner[16:],validation=inner[:16],test=test))
    p=dict(version=1,date='2026-10-12',cases=cases,rules={rule_id(r):r for r in rules},folds=folds,
        observable='C_ij = <Z_i Z_j>, all pairs, Pauli normalization; no X/Y correlators',
        hamiltonian='H=-sum_e [g(XX+YY)+atan(alpha*(tau-d))ZZ]/4; rho=exp(-b H)/Z',
        scales='D=median(abs(Delta)); g=r D; b=4K/D; beta_paper=N*b. Same fallback as previous studies.',
        sampling=dict(chains=4,samples=2048,burn=2000,thin=2,rounds=1),budgets=list(BUDGETS),
        split='Five outer image-level folds, each 64 training /16 inner validation /20 outer test. All parameter selection uses inner validation only.',
        reuse='Existing development dataset, not untouched external validation. Existing state banks reused only as label-independent measurements.',
        seed_modes=['none','foreground'],
        seed_note='The foreground seed is the existing mask-derived interior pixel, shared by seeded neural and classical controls. No seed supplied in the seedless arm.',
        network_screen=dict(architecture='volume',width=24,lr=.003,weight_decay=.001,epochs=100),
        network_refine=[dict(architecture=a,width=w,lr=lr,weight_decay=wd,epochs=140)
            for a,w,lr,wd in [('volume',24,.001,.001),('volume',40,.003,.01),('graph',24,.003,.001)]],
        refinement='Refine the top two Hamiltonians per model, seed mode, and outer fold; select checkpoint and output threshold on inner validation.',
        final='Refit on all 80 outer-training sources for the selected epoch count, with three fixed training seeds; average probabilities. Keep the inner-selected output threshold.',
        controls=['single foreground seed sign','two-seed max (extra known background label)','seed/position only neural','single-seed correlation neural','raw grayscale image neural'],
        validation='Same-image pairwise comparisons and source bootstrap; neither pixels nor sample configurations are independent evaluation images.',
        code_sha256={name:digest(Path(__file__).parent/name) for name in ['correlation_nn_data.py','promotion_reliability_bank.py','iterative_seed_extended_loops.py','ising_seed_sampling.py']})
    write_json(PROTOCOL,p);return p


def moments(packed,n,budgets=BUDGETS):
    """All products from the same commuting Z measurements, without N² shots."""
    z=unpack(packed,n).astype(np.float32)
    chains,count,_=z.shape
    corr=[]
    for b in budgets:
        q=z[:,:b].reshape(chains,16,b//16,n)
        batch=np.matmul(q.swapaxes(-1,-2),q)/(b//16)
        corr.append(batch.mean(axis=(0,1)))
    mean=corr[-1];cm=batch.mean(1)
    se=np.maximum(batch.reshape(chains*16,n,n).std(0,ddof=1)/np.sqrt(chains*16),cm.std(0,ddof=1)/np.sqrt(chains))
    half=batch.reshape(chains,2,8,n,n).mean(2).reshape(chains*2,n,n)
    m=budgets[-1]//2
    within=((1-half**2)*m/(m-1)).mean(0)
    between=m*half.var(0,ddof=1)
    rhat=np.ones_like(mean);v=within>1e-12
    rhat[v]=np.sqrt((m-1)/m+between[v]/(m*within[v]))
    rhat[(~v)&(between>1e-12)]=np.inf
    ess=np.full_like(mean,float(chains*budgets[-1]))
    total=chains*budgets[-1]
    np.divide((1-mean**2)*total/(total-1),se**2,out=ess,where=se>1e-12)
    return np.asarray(corr),se,rhat,np.minimum(ess,chains*budgets[-1])


def reusable(case,rule):
    path=ROOT/'results/iterative-seeds/test'/f'{case["case_id"]}--{rule_id(rule)}.bank.npz'
    if not path.exists():return None
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['signature']['rule']!=rule or meta['signature']['case']['case_id']!=case['case_id']:raise ValueError('Bank identity mismatch')
    return path,meta


def job(args):
    case,rid,rule=args;start=time.perf_counter()
    target=OUT/'measurements'/rid/f'{case["case_id"]}.npz';jpath=target.with_suffix('.json')
    if target.exists() and jpath.exists():return rid,case['case_id'],'cached'
    data=np.load(INPUT/f'{case["case_id"]}.npz');old=reusable(case,rule)
    if old:
        src,meta=old;packed=np.load(src)['packed'][0]
        params=meta['parameters'];sampling=meta['sampling'][0]
        provenance=dict(source=str(src),sha256=digest(src),round=0)
    else:
        packed,params,sampling=next(iter_tail(data['image'],rule,rounds=1,samples=2048,burn=2000,thin=2,chains=4,
            rng_seed=seed_for('correlation-nn:measure:'+case['case_id']+':'+rid)))
        provenance=dict(source='new',rng_seed=seed_for('correlation-nn:measure:'+case['case_id']+':'+rid))
    c,se,rh,ess=moments(packed,data['image'].size)
    off=~np.eye(data['image'].size,dtype=bool);finite=np.isfinite(rh)
    assert np.max(abs(c-c.swapaxes(-1,-2)))<1e-6 and np.all(np.diagonal(c,axis1=-2,axis2=-1)==1)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,corr=c,se=se,rhat=rh,ess=ess,packed=packed)
    diagnostics=dict(max_pair_rhat=float(rh[off].max()) if finite[off].all() else None,
        failed_pair_fraction=float(np.mean(rh[off]>=1.05)),median_pair_ess=float(np.median(ess[off])),
        minimum_pair_ess=float(ess[off].min()),max_se=float(se[off].max()),
        converged=bool((rh[off]<1.05).all() and sampling['scalar_rhat'] is not None and sampling['scalar_rhat']<1.05))
    write_json(jpath,dict(case_id=case['case_id'],rule=rule,parameters=params,sampling=sampling,diagnostics=diagnostics,
        provenance=provenance,ess_ddof=1,input_sha256=digest(INPUT/f'{case["case_id"]}.npz'),measurement_sha256=digest(target),seconds=time.perf_counter()-start))
    return rid,case['case_id'],round(time.perf_counter()-start,2)


def consolidate(p):
    OUT.mkdir(exist_ok=True,parents=True)
    truth=[];images=[];seeds=[]
    for case in p['cases']:
        d=np.load(INPUT/f'{case["case_id"]}.npz');truth.append(d['truth']);images.append(d['image']/255);seeds.append(d['seeds'])
    np.savez_compressed(OUT/'labels.npz',truth=truth,images=images,seeds=seeds)
    for rid in p['rules']:
        cs=[];ses=[];diagnostics=[]
        for case in p['cases']:
            path=OUT/'measurements'/rid/f'{case["case_id"]}.npz'
            d=np.load(path);cs.append(d['corr']);ses.append(d['se'])
            diagnostics.append(json.loads(path.with_suffix('.json').read_text())['diagnostics'])
        # Uncompressed arrays support fast repeated training reads.
        np.save(OUT/f'correlations-{rid}.npy',np.stack(cs))
        np.save(OUT/f'standard-errors-{rid}.npy',np.stack(ses))
        write_json(OUT/f'diagnostics-{rid}.json',diagnostics)
    print('consolidated',len(p['rules']),'Hamiltonians ×',len(p['cases']),'images',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['init','sample','consolidate']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();p=init()
    if a.stage=='sample':
        jobs=[(c,rid,r) for rid,r in p['rules'].items() for c in p['cases']]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for k,f in enumerate(as_completed([pool.submit(job,j) for j in jobs])):
                value=f.result()
                if (k+1)%25==0:print(k+1,'/',len(jobs),value,flush=True)
        consolidate(p)
    elif a.stage=='consolidate':consolidate(p)
    else:print('protocol initialized:',len(p['rules']),'Hamiltonians',flush=True)


if __name__=='__main__':main()
