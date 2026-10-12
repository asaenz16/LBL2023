"""Supplementary calibration-only selection explicitly targeting seed reliability.

Added after the segmentation-objective calibration, before inspecting comparison
outcomes. The original frozen selection and experiment are left unchanged.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import run_promotion_reliability as main
from promotion_reliability import evaluate,METRICS,REASONS
from weizmann_petz_data import write_json
OUT=main.OUT/'reliability-targeted'
SELECT=main.ROOT/'promotion-reliability-targeted-selection.json'


def select():
    j=json.loads((main.OUT/'calibration-scores.json').read_text());winners={}
    for anchor,values in j['means'].items():
        a=np.asarray(values);policies=j['policies'];original=main.historical(main.anchors()[anchor]);base=a[policies.index(original)]
        n=METRICS.index('promoted_count');err=METRICS.index('promoted_wrong');iou=METRICS.index('iou');u=METRICS.index('mixing_unresolved')
        good=[i for i,p in enumerate(policies) if main.family(p,'any') and a[i,n]>=.5*base[n] and a[i,iou]>=base[iou]-.01 and a[i,u]<=base[u]+.02 and a[i,n]>0]
        assert good
        best=min(good,key=lambda i:(a[i,err]/a[i,n],-a[i,iou],a[i,METRICS.index('rounds')],main.pid(policies[i])))
        winners[anchor]=dict(policy=policies[best],metrics=dict(zip(METRICS,a[best].tolist())),historical_metrics=dict(zip(METRICS,base.tolist())),eligible_policies=len(good))
    result=dict(date='2026-10-09',status='frozen_before_comparison_analysis',selection='Minimum pooled active wrong-seed fraction on25 calibration sources; require at least half historical promotion count, mean IoU no more than.01 lower and mixing U no more than.02 higher. Ties IoU, rounds, policyID. Only modified policies. All constraints use calibration data only. Supplement added after primary calibration; no comparison outcomes used.',
        confirmation='Use primary experiment confirmations wherever available; also confirm any supplementary endpoint failing convergence or ESS<100. Whole supplementary case bank replaced, never cherry-picked.',
        code_hashes=main.hashes(),source_sha256=main.digest(__file__),calibration_scores_sha256=main.digest(main.OUT/'calibration-scores.json'),winners=winners)
    if SELECT.exists():assert json.loads(SELECT.read_text())==result
    else:write_json(SELECT,result)
    for a,r in winners.items():print(a,r['policy'],r['metrics']['promoted_wrong']/r['metrics']['promoted_count'],flush=True)


def groups():
    chosen=json.loads(SELECT.read_text());assert chosen['source_sha256']==main.digest(__file__);out={}
    for anchor,v in chosen['winners'].items():
        rule=main.anchors()[anchor]['rule'];rid=main.identifier(rule);out.setdefault(rid,dict(rule=rule,profiles={}))
        out[rid]['profiles'][anchor]={'reliability':v['policy'],'reliability/matched':main.neutral(v['policy']),'historical':main.historical(main.anchors()[anchor])}
    return list(out.values())


def job(spec):
    c,group=spec;rid=main.identifier(group['rule']);stem=f'{c["case_id"]}--{rid}';path=OUT/f'{stem}.json';path.parent.mkdir(parents=True,exist_ok=True)
    sig=dict(case=c,group=group,selection_sha256=main.digest(SELECT))
    if path.exists():assert json.loads(path.read_text())['signature']==sig;return stem
    stage='confirm' if (main.OUT/'confirm'/f'{stem}.json').exists() else 'test'
    while True:
        packed,meta,data=main.bank(stage,c,group['rule']);profiles={};arrays={};bad=False
        for j,(anchor,labels) in enumerate(group['profiles'].items()):
            policies=list({main.pid(p):p for p in labels.values()}.values());key='promotion-reliability:'+stage+':'+c['case_id']+':'+anchor
            table,reasons,traces=evaluate(packed,meta['sampling'],data['truth'],data['seeds'],main.anchors()[anchor]['base'],policies,10,key,True)
            bad|=bool(np.any(table[:,:,METRICS.index('converged')]<1) or np.any(table[:,:,METRICS.index('minimum_ess')]<100))
            arrays[f'metrics{j}']=table;arrays[f'reasons{j}']=reasons;profiles[anchor]=dict(array_index=j,policies=policies,labels={label:policies.index(p) for label,p in labels.items()},traces=traces,promotion_key=key)
        if bad and stage=='test':stage='confirm';continue
        break
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    write_json(path,dict(signature=sig,case=c,rule=group['rule'],profiles=profiles,metrics=METRICS,reasons=REASONS,stage=stage,bank_sha256=meta['bank_sha256']));return stem


def run():
    # Main confirmation files are stable once the main analysis has started.
    while not (main.ROOT/'promotion-reliability-summary.json').exists():time.sleep(10)
    specs=[(c,g) for c in main.cases('test') for g in groups()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(job,x) for x in specs]
        for i,f in enumerate(as_completed(fs),1):
            stem=f.result()
            if i%25==0 or i==len(fs):print(i,len(fs),stem,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','run']);a=p.parse_args()
    select() if a.stage=='select' else run()
