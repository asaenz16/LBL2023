"""Preselected longer independent checks on fresh test classes 0,12,24,36.

Double warmup and measurement length; use different random seeds. These checks
diagnose sampling stability and do not change the frozen parameter selection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import numpy as np
from run_emnist_thresholds import ROOT, OUTPUT, job, write_once

CLASSES=[0,12,24,36]


def main(workers):
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    images=np.load(OUTPUT/'subset.npz')['images']
    imdict={c['name']:im for c,im in zip(manifest['cases'],images)}
    jobs=[];sources=[]
    for path in sorted((OUTPUT/'test').glob('*.json')):
        m=json.loads(path.read_text());case=m['case']
        if case['class_id'] not in CLASSES:continue
        sig=m['signature'];options=dict(sig['options'],burn=2048,samples=4096,rng_seed=sig['options']['rng_seed']+310000)
        jobs.append((case,imdict[case['name']],sig['rule'],sig['ks'],options,OUTPUT/'confirmation'/path.with_suffix('.npz').name))
        sources.append(path)
    expected=json.loads((OUTPUT/'test-protocol.json').read_text())['rules_and_ks']
    assert len(jobs)==len(CLASSES)*len(expected), 'Finish every preselected test case first.'
    write_once(OUTPUT/'confirmation-protocol.json',dict(classes=CLASSES,burn=2048,samples=4096,chains=4,
        seed_offset=310000,purpose='Independent longer run, no reselection',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,spec) for spec in jobs]
        for i,f in enumerate(as_completed(futures),1):
            m=f.result();print(json.dumps(dict(done=i,total=len(jobs),elapsed=round(time.perf_counter()-start,1),
                rule=m['signature']['rule'],class_id=m['case']['class_id'])),flush=True)
    rows=[]
    for path in sources:
        m=json.loads(path.read_text());long=json.loads((OUTPUT/'confirmation'/path.name).read_text())
        shortdata=np.load(path.with_suffix('.npz'));longdata=np.load(OUTPUT/'confirmation'/path.with_suffix('.npz').name)
        for i,(a,b) in enumerate(zip(m['records'],long['records'])):
            error=abs(shortdata['gamma'][i,0]-longdata['gamma'][i,0])
            combined=np.sqrt(shortdata['se'][i,0]**2+longdata['se'][i,0]**2)
            rows.append(dict(name=a['name'],rule=a['rule'],k=a['k'],short_iou=a['iou'],long_iou=b['iou'],
                gamma_linf=float(error.max()),gamma_rmse=float(np.sqrt(np.mean(error**2))),
                outside_6_combined_se_plus002=int(np.sum(error>6*combined+.002)),
                short_rhat=a['rhat_observable_max'],long_rhat=b['rhat_observable_max'],
                long_ess_mean_gamma=b['ess_mean_gamma'],long_resolved_fraction=b['resolved_fraction']))
    result=dict(protocol=json.loads((OUTPUT/'confirmation-protocol.json').read_text()),records=rows,
        max_gamma_linf=max(r['gamma_linf'] for r in rows),median_gamma_linf=float(np.median([r['gamma_linf'] for r in rows])),
        max_rhat=max(r['long_rhat'] for r in rows),min_ess_mean_gamma=min(r['long_ess_mean_gamma'] for r in rows),
        outside_6_combined_se_plus002=sum(r['outside_6_combined_se_plus002'] for r in rows),
        max_iou_change=max(abs(r['short_iou']-r['long_iou']) for r in rows),
        mean_absolute_iou_change=float(np.mean([abs(r['short_iou']-r['long_iou']) for r in rows])))
    (ROOT/'emnist-thresholds-confirmation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','protocol']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);args=p.parse_args();main(args.workers)
