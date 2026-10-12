"""Higher-budget XXZ validation with frozen balanced policies and common banks."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json
from itertools import islice
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import numpy as np
from run_higher_seed_study import ROOT,OUT,INPUT,BANKS,anchors,identifier,cases,profiles,pid,code_hashes,digest
from higher_seed_readout import METRICS,REASONS
from run_higher_seed_bounded import bounded_evaluate
from promotion_reliability_bank import iter_tail
from weizmann_petz_data import seed_for,write_json


def specs():
    rule=anchors()['XXZ/foreground']['rule'];rid=identifier(rule)
    available={json.loads(p.read_text())['signature']['case']['case_id']:str(p)
               for p in (BANKS/'confirm').glob('*--'+rid+'.json')}
    flags=set()
    for path in (OUT/'test').glob('*--XXZ-*--2048.json'):
        m=json.loads(path.read_text());v=np.load(path.with_suffix('.npz'))['metrics']
        ix=[i for k,i in m['labels'].items() if k.endswith('/balanced')]
        if np.any(v[ix,:,METRICS.index('converged')]<1) or np.any(v[ix,:,METRICS.index('minimum_ess')]<100):
            flags.add(m['signature']['case']['case_id'])
    cs={c['case_id']:c for c in cases('test')}
    return [(cs[k],available.get(k)) for k in sorted(set(available)|flags)],sorted(flags)


def job(spec):
    case,existing=spec;rule=anchors()['XXZ/foreground']['rule'];rid=identifier(rule)
    source=INPUT/'cases'/f'{case["case_id"]}.npz';data=np.load(source)
    if existing:
        from pathlib import Path
        path=Path(existing)
    else:
        path=OUT/'confirmation-banks'/f'{case["case_id"]}--{rid}.json'
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            settings=dict(rounds=10,samples=8192,burn=6000,thin=2,chains=4)
            seed=seed_for('higher-seeds:confirm:'+case['case_id']+':'+rid)
            rows=list(islice(iter_tail(data['image'],rule,**settings,rng_seed=seed),3))
            np.savez_compressed(path.with_suffix('.npz'),packed=np.stack([r[0] for r in rows]))
            write_json(path,dict(signature=dict(case=case,rule=rule,settings=settings,gibbs_seed=seed,
                input_sha256=digest(source)),sampling=[r[2] for r in rows],parameters=rows[0][1],
                bank_sha256=digest(path.with_suffix('.npz')),note='Three used rounds; ten-round seed spawning and confidence correction preserved.'))
    meta=json.loads(path.read_text());assert digest(path.with_suffix('.npz'))==meta['bank_sha256']
    assert digest(source)==meta['signature']['input_sha256'] and meta['signature']['rule']==rule
    packed=np.load(path.with_suffix('.npz'))['packed'][:3]
    assert packed.shape[1:3]==(4,8192)
    for anchor in ('XXZ/foreground','XXZ/background'):
        dest=OUT/'confirm'/f'{case["case_id"]}--{anchor.replace("/","-")}--8192.json'
        if dest.exists():continue
        start=time.perf_counter();labels=profiles(anchor,8192);policies=list({pid(p):p for p in labels.values()}.values())
        assert max(p['cap'] for p in policies)==3
        initial=int(data['seeds'][1 if anchor.endswith('foreground') else 0])
        table,reasons,traces=bounded_evaluate(packed,meta['sampling'][:3],data['truth'],initial,anchor.split('/')[1],
            policies,5,seed_for('higher-seeds:test:'+case['case_id']+':'+anchor),samples=8192,traces=True)
        dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),metrics=table,reasons=reasons)
        write_json(dest,dict(signature=dict(case=case,anchor=anchor,samples=8192,policies=policies,repeats=5,code_hashes=code_hashes()),
            labels={k:policies.index(p) for k,p in labels.items()},metrics=METRICS,reasons=REASONS,bank=str(path.with_suffix('.npz')),
            bank_sha256=meta['bank_sha256'],bank_settings=meta['signature']['settings'],seconds=time.perf_counter()-start,traces=traces))
    return case['case_id']


def main():
    tasks,flags=specs()
    protocol=dict(cases=[c['case_id'] for c,p in tasks],flagged=flags,new_banks=[c['case_id'] for c,p in tasks if p is None],
        samples_per_chain=8192,chains=4,used_rounds=3,seed_spawning_rounds=10,burn=6000,thin=2,
        policies='Frozen balanced selections and matched pair/linear controls only; no retuning.',
        scope='All available prior primary-XXZ confirmation banks plus every new standard-budget balanced trajectory with failed final score convergence or ESS<100. Replace the whole case bank for both starts/all methods. Report the original fixed-budget results separately and a common-bank confirmed 75-source comparison.',
        selection_sha256=digest(ROOT/'higher-seeds-selection.json'),script_sha256=digest(__file__))
    p=ROOT/'higher-seeds-confirmation-protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:write_json(p,protocol)
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(job,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):print(i,'/',len(tasks),f.result(),flush=True)


if __name__=='__main__':main()
