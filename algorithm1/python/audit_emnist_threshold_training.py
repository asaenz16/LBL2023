"""Longer checks of every calibration job with observable split R-hat >1.05.

This is a diagnostic audit. Frozen test selections remain unchanged. Report
whether replacing flagged training estimates would alter the selection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import numpy as np
from run_emnist_thresholds import ROOT,OUTPUT,job,load_runs,write_once
from ising_thresholds import rule_name


def main():
    train=load_runs(OUTPUT,'train')
    flagged=[m for m in train if any(r['rhat_observable_max']>1.05 for r in m['records'])]
    cases=json.loads((OUTPUT/'manifest.json').read_text())['cases']
    images={c['name']:im for c,im in zip(cases,np.load(OUTPUT/'subset.npz')['images'])}
    write_once(OUTPUT/'training-audit-protocol.json',dict(
        trigger='Any observable split R-hat >1.05 in original training sweep',
        cases=[dict(name=m['case']['name'],rule=m['signature']['rule']) for m in flagged],
        burn=2048,samples=4096,chains=4,seed_offset=610000,
        policy='Report effects on calibration; do not alter frozen test selections'))
    jobs=[]
    for m in flagged:
        sig=m['signature'];case=m['case'];options=dict(sig['options'],burn=2048,samples=4096,chains=4,
            rng_seed=sig['options']['rng_seed']+610000)
        jobs.append((case,images[case['name']],sig['rule'],sig['ks'],options,
            OUTPUT/'training-audit'/f'{case["name"]}-{rule_name(sig["rule"])}.npz'))
    audited=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,spec) for spec in jobs]
        for f in as_completed(futures):
            m=f.result();audited.append(m);print(m['case']['name'],m['signature']['rule'],flush=True)
    adjusted={(r['name'],r['rule'],r['k']):r for m in train for r in m['records']}
    differences=[]
    for m in audited:
        for r in m['records']:
            key=(r['name'],r['rule'],r['k']);old=adjusted[key]
            differences.append(dict(name=r['name'],rule=r['rule'],k=r['k'],old_iou=old['iou'],new_iou=r['iou'],
                old_rhat=old['rhat_observable_max'],new_rhat=r['rhat_observable_max']))
            adjusted[key]=r
    frozen=json.loads((OUTPUT/'frozen-selection.json').read_text());scores=[]
    for candidate in frozen['candidates']:
        rows=[r for r in adjusted.values() if r['rule']==rule_name(candidate['rule']) and r['k']==candidate['k']]
        scores.append(dict(rule=candidate['rule'],k=candidate['k'],mean_iou=float(np.mean([r['iou'] for r in rows]))))
    unchanged={};winners={}
    for name,selected in frozen['selections'].items():
        if name=='baseline':eligible=[s for s in scores if s['rule']==selected['rule']]
        else:eligible=[s for s in scores if s['rule']['kind']==selected['rule']['kind']]
        best=max(eligible,key=lambda s:s['mean_iou']);winners[name]=best
        unchanged[name]=(best['rule']==selected['rule'] and best['k']==selected['k'])
    result=dict(flagged_jobs=len(flagged),comparisons=differences,adjusted_training_winners=winners,
        frozen_selections_unchanged=unchanged,max_audited_rhat=max(r['new_rhat'] for r in differences) if differences else None)
    (ROOT/'emnist-thresholds-training-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
