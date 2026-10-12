"""Post-hoc 2x2 threshold/temperature diagnostic; does not retune main results."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from run_weizmann_ising_seeds import ROOT,OUT,METRICS,DECISIONS,PRIMARY,hashes,manifest,read_case,dispatch
from ising_seed_sampling import (parameterize,setup_sampler,exact_correlations,sample_spins,sufficient_counts,
    evaluate_counts,exact_decision,segmentation_metrics)
from weizmann_petz_data import write_json,seed_for

RULES=[dict(function='arctan_cubic',sensitivity='adaptive',threshold=dict(kind='quantile',value=.9),K=.4),
       dict(function='arctan_cubic',sensitivity='adaptive',threshold=dict(kind='fixed',value=75.),K=.8)]
LABELS=['q90_K04','fixed75_K08']


def job(case):
    path=OUT/'threshold-diagnostic'/f'{case["case_id"]}.json'
    signature=dict(case=case,rules=RULES,code_hashes=hashes(),diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),budget=PRIMARY,replicates=8)
    if path.exists():
        assert json.loads(path.read_text())['signature']==signature;return case['case_id']
    t=time.perf_counter();f=read_case(case);image,truth,seeds=f['image'],f['truth'],f['seeds'];rows=[];arrays={}
    for label,rule in zip(LABELS,RULES):
        edges,j,param=parameterize(image,rule);sampler=setup_sampler(image.shape,edges,j,param['b']);corr=exact_correlations(sampler,seeds)
        exact={d:segmentation_metrics(*exact_decision(corr,d),truth,seeds) for d in DECISIONS};counts=[];metrics=[]
        for rep in range(8):
            seed=seed_for('ising-threshold-diagnostic:'+case['case_id']+label+str(rep))
            z=sample_spins(sampler,PRIMARY,np.random.default_rng(seed));c=sufficient_counts(z,seeds,[PRIMARY])[0]
            counts.append(c);metrics.append(evaluate_counts(c,PRIMARY,truth,seeds))
        decisions={d:dict(exact=exact[d],sample={m:float(np.mean([x[d][m] for x in metrics])) for m in METRICS}) for d in DECISIONS}
        rows.append(dict(label=label,rule=rule,parameters=param,decisions=decisions));arrays[label+'_counts']=np.array(counts);arrays[label+'_correlations']=corr
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    write_json(path,dict(signature=signature,case=case,records=rows,seconds=time.perf_counter()-t));return case['case_id']


def main():
    protocol=dict(date='2026-10-08',status='post-hoc mechanism diagnostic',
        rationale='Main test revealed poor 16x16 transfer of the fixed75/K=.4 cubic recipe. Threshold and temperature both differ from the q90/K=.8 control. Add the two missing factorial corners to separate their effects. Main selection and results remain frozen; this is not an independent confirmation or a new parameter selection.',
        fixed='J=Delta+4Delta^3, alpha=1/S; same source split and sample budget.',rules=RULES,
        code_hashes=hashes(),diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    p=ROOT/'weizmann-ising-seeds-threshold-diagnostic-protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Diagnostic changed.')
    write_json(p,protocol);dispatch([c for c in manifest()['cases'] if c['split']=='test'],job,4)


if __name__=='__main__':main()
