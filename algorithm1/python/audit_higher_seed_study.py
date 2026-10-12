"""Trajectory audit and equal-consumed-budget original-seed controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
from run_higher_seed_study import ROOT,OUT,INPUT,digest,code_hashes,anchors
from higher_seed_readout import METRICS,metrics,evaluate,select_triple
from iterative_seed_gibbs import unpack,infer_batches
from weizmann_petz_data import seed_for,write_json


def control(path):
    dest=OUT/'pooled-controls'/path.name
    if dest.exists():return
    meta=json.loads(path.read_text());spec=meta['signature']
    assert spec['code_hashes']==code_hashes()
    data=np.load(INPUT/'cases'/f'{spec["case"]["case_id"]}.npz')
    bankmeta=json.loads(Path(meta['bank']).with_suffix('.json').read_text())
    assert digest(meta['bank'])==meta['bank_sha256']
    packed=np.load(meta['bank'])['packed'];n=data['truth'].size
    initial=int(data['seeds'][1 if spec['anchor'].endswith('foreground') else 0]);y=1 if spec['anchor'].endswith('foreground') else -1
    batch=[];scalars=[];infs={};cache={};rows={};triplets={};triplet_cache={}
    rng=np.random.default_rng(seed_for('higher-seeds:test:'+spec['case']['case_id']+':'+spec['anchor']))
    rng.random((spec['repeats'],10,2));priorities=rng.random((spec['repeats'],10,n))
    for p,d in zip(packed,bankmeta['sampling']):
        z=unpack(p[:,:spec['samples']],n)
        obs=z*z[...,initial,None]*y
        batch.append(obs.reshape(4,16,spec['samples']//16,n).mean(2))
        scalars.append(d['scalar_rhat'] if d['scalar_rhat'] is not None else float('inf'))
    for label,pi in meta['labels'].items():
        if not label.endswith('/balanced'):continue
        p=spec['policies'][pi];values=[];observations=[]
        for trace in meta['traces']:
            if trace['policy_index']!=pi:continue
            k=len(trace['history'])
            if k not in infs:
                infs[k]=infer_batches(np.concatenate(batch[:k]),np.array([initial]),np.array([y]),None,
                                      spec['samples'],max(scalars[:k]),10)
            row=dict(inference=infs[k],active={initial:y},round=k,higher_rounds=0,weighted_ess=k*4*spec['samples'],
                     gate_rejections=0,score_evaluations=n)
            values.append(metrics(row,data['truth'],initial,p['margin'],spec['samples'],cache))
            if p['family']!='pair':
                for h in trace['history']:
                    active={int(k):v for k,v in h['active'].items()};t=h['round']-1
                    triple=select_triple(sorted(active),initial,p['group'],priorities[trace['repeat'],t])
                    if not triple:continue
                    key=(t,tuple((j,active[j]) for j in triple))
                    if key not in triplet_cache:
                        u=np.stack([(2*((packed[t,:,:spec['samples'],j//8]>>(j%8))&1).astype(np.int8)-1)*active[j] for j in triple],-1)
                        agree=float(((u[...,0]==u[...,1])&(u[...,1]==u[...,2])).mean())
                        wrong=sum(bool(data['truth'].ravel()[j])!=(active[j]==1) for j in triple)
                        triplet_cache[key]=dict(agreement_probability=agree,majority_wrong_labels=float(wrong>=2),
                                               majority_linear_difference_bound=2*(1-agree)/3)
                    observations.append(dict(repeat=trace['repeat'],round=t+1,**triplet_cache[key]))
        rows[label]=values
        triplets[label]=observations
    dest.parent.mkdir(exist_ok=True);write_json(dest,dict(case=spec['case'],anchor=spec['anchor'],samples=spec['samples'],controls=rows,triplets=triplets))


def audit():
    counts=0;files=0;errors=[];tables=0;replays=[];selected={}
    calibration=list((OUT/'calibrate').glob('*.json'));assert len(calibration)==200
    calibration_endpoints=0
    for path in calibration:
        meta=json.loads(path.read_text());assert meta['signature']['code_hashes']==code_hashes()
        v=np.load(path.with_suffix('.npz'))['metrics'];assert np.isfinite(v).all()
        assert v.shape==(1836,2,len(METRICS));calibration_endpoints+=v.shape[0]*v.shape[1]
    for path in sorted((OUT/'test').glob('*.json')):
        meta=json.loads(path.read_text());s=meta['signature'];assert s['code_hashes']==code_hashes()
        provenance=json.loads(Path(meta['bank']).with_suffix('.json').read_text())
        assert provenance['signature']['rule']==anchors()[s['anchor']]['rule']
        value=np.load(path.with_suffix('.npz'))['metrics'];data=np.load(INPUT/'cases'/f'{s["case"]["case_id"]}.npz')
        assert np.isfinite(value).all()
        truth=data['truth'].ravel();initial=int(data['seeds'][1 if s['anchor'].endswith('foreground') else 0])
        assert bool(truth[initial])==s['anchor'].endswith('foreground')
        assert len(meta['traces'])==value.shape[0]*value.shape[1]
        assert len({(t['policy_index'],t['repeat']) for t in meta['traces']})==len(meta['traces'])
        for trace in meta['traces']:
            counts+=1;pi=trace['policy_index'];rep=trace['repeat'];p=s['policies'][pi];h=trace['history'];row=value[pi,rep]
            assert len(h)<=p['cap']
            for a,b in zip(h,h[1:]):
                old={int(k):v for k,v in a['active'].items()};new={int(k):v for k,v in b['active'].items()}
                added={k:v for k,v in new.items() if k not in old}
                assert all(new[k]==v for k,v in old.items())
                assert sum(v==1 for v in added.values())<=1 and sum(v==-1 for v in added.values())<=1
                assert set(added)=={k for k in a['added'] if k>=0}
            end={int(k):v for k,v in h[-1]['active'].items() if int(k)!=initial}
            assert row[METRICS.index('promoted_count')]==len(end)
            assert row[METRICS.index('promoted_wrong')]==sum(bool(truth[k])!=(v==1) for k,v in end.items())
            assert row[METRICS.index('shots')]==4*s['samples']*len(h)
            pred=np.unpackbits(np.array(trace['prediction'],np.uint8),count=len(truth)).astype(bool)
            union=(pred|truth).sum();iou=(pred&truth).sum()/union if union else 1.
            assert abs(iou-row[METRICS.index('iou')])<1e-12
        for m in ('iou','unresolved','mixing_unresolved','bpm_iou_r1','activated'):
            assert np.all((value[:,:,METRICS.index(m)]>=0)&(value[:,:,METRICS.index(m)]<=1))
        tables+=value.shape[0]*value.shape[1];files+=1
        key=(s['anchor'],s['case']['shape'][0],s['samples'])
        selected.setdefault(key,path)
    assert files==1800,files
    print('trajectory audit passed:',files,'files,',counts,'trajectories; starting bank replays',flush=True)
    # Recompute one complete case per model/start/size/budget directly from bank.
    for key,path in selected.items():
        meta=json.loads(path.read_text());s=meta['signature'];data=np.load(INPUT/'cases'/f'{s["case"]["case_id"]}.npz')
        bankmeta=json.loads(Path(meta['bank']).with_suffix('.json').read_text())
        p=np.load(meta['bank'])['packed'];initial=int(data['seeds'][1 if s['anchor'].endswith('foreground') else 0])
        v,r,_=evaluate(p,bankmeta['sampling'],data['truth'],initial,s['anchor'].split('/')[1],s['policies'],s['repeats'],
                      seed_for('higher-seeds:test:'+s['case']['case_id']+':'+s['anchor']),s['samples'])
        old=np.load(path.with_suffix('.npz'));diff=float(np.max(abs(v-old['metrics'])))
        assert diff<1e-9 and np.array_equal(r,old['reasons'])
        replays.append(dict(file=path.name,maximum_metric_discrepancy=diff))
        print('replay',len(replays),'/',len(selected),path.name,'maximum discrepancy',diff,flush=True)
    write_json(ROOT/'higher-seeds-validation.json',dict(passed=True,files=files,trajectories=counts,endpoints=tables,
        calibration_endpoints=calibration_endpoints,full_bank_replays=replays,code_hashes=code_hashes()))
    print(files,'files,',counts,'trajectories;',len(replays),'bank replays passed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('controls','audit'));p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage=='audit':audit();return
    paths=[p for p in (OUT/'test').glob('*.json') if json.loads(p.read_text())['signature']['samples']==2048]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(control,p) for p in paths]
        for i,f in enumerate(as_completed(fs),1):
            f.result()
            if i%50==0 or i==len(fs):print(i,'/',len(fs),flush=True)


if __name__=='__main__':main()
