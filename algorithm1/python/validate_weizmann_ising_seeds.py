"""Independent artifact, numerical, and old-baseline audits of the seed study."""
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from run_weizmann_ising_seeds import (ROOT,OUT,METRICS,DECISIONS,BUDGETS,PRIMARY,manifest,hashes,
    read_case,calibration_cases,candidates,chosen_rules)
from ising_seed_sampling import evaluate_counts,exact_decision,segmentation_metrics
from weizmann_petz_data import write_json


def main():
    protocol=json.loads((ROOT/'weizmann-ising-seeds-protocol.json').read_text());m=manifest()
    assert protocol['code_hashes']==hashes()
    assert protocol['manifest_sha256']==hashlib.sha256((ROOT/'weizmann-ising-seeds-manifest.json').read_bytes()).hexdigest()
    controls=json.loads((ROOT/'weizmann-ising-seeds-controls-protocol.json').read_text())
    assert controls['expanded_selection']==json.loads((ROOT/'weizmann-ising-seeds-selection.json').read_text())
    assert controls['extension_code_sha256']==hashlib.sha256((ROOT/'python/add_ising_seed_controls.py').read_bytes()).hexdigest()
    assert len(m['calibration_ids'])==25 and len(m['test_ids'])==75
    assert set(m['calibration_ids']).isdisjoint(m['test_ids'])
    for c in m['cases']:
        f=read_case(c);assert f['image'].shape==(c['side'],c['side'])
        assert f['truth'].ravel()[f['seeds']].tolist()==[False,True]
    max_strength=0.
    for c in calibration_cases():
        p=OUT/'calibration'/f'{c["case_id"]}.npz';f=np.load(p);meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['signature']==dict(case=c,code_hashes=hashes(),replicates=3,budget=PRIMARY)
        assert f['exact_metrics'].shape==(351,3,len(METRICS))
        assert np.isfinite(f['exact_metrics']).all() and np.isfinite(f['sample_metrics']).all()
        max_strength=max(max_strength,max(x['max_abs_theta'] for x in meta['parameters']))
    prior_path=ROOT/'weizmann-xxz-v2-case-results.json';prior=json.loads(prior_path.read_text())
    old={r['case']['case_id']:r['exact']['iou'] for r in prior if r.get('family')=='Ising/edge_otsu' and r['signature']['stage']=='test'}
    tests=[c for c in m['cases'] if c['split']=='test'];chosen=json.loads(json.dumps(chosen_rules()))
    error=0.;self_error=0.;reciprocity=0.;prior_errors=[];model_cases=0;sampling_decisions=0;result_hashes={};sample_bound_excess=0.
    for index,c in enumerate(tests,1):
        path=OUT/'test'/f'{c["case_id"]}.json';data=json.loads(path.read_text());f=np.load(path.with_suffix('.npz'));source=read_case(c)
        assert data['signature']==dict(case=c,code_hashes=hashes(),selected=chosen,budgets=BUDGETS,replicates=8)
        truth,seeds=source['truth'],source['seeds'];n=truth.size
        for record in data['records']:
            rid=record['rule_id'];corr=f[rid+'_correlations'];counts=f[rid+'_counts']
            assert corr.shape==(2,n) and counts.shape==(8,3,4,n) and np.isfinite(corr).all()
            assert abs(corr).max()<=1+1e-10
            self_error=max(self_error,float(abs(corr[np.arange(2),seeds]-1).max()))
            reciprocity=max(reciprocity,float(abs(corr[0,seeds[1]]-corr[1,seeds[0]])))
            for decision in DECISIONS:
                expected=segmentation_metrics(*exact_decision(corr,decision),truth,seeds)
                error=max(error,max(abs(expected[k]-record['decisions'][decision]['exact'][k]) for k in METRICS))
            for bi,b in enumerate(BUDGETS):
                assert counts[:,bi].min()>=0 and counts[:,bi].max()<=b
                assert np.all(counts[:,bi,1]<=counts[:,bi,3])
                vals=[evaluate_counts(counts[rep,bi],b,truth,seeds) for rep in range(8)]
                for decision in DECISIONS:
                    for metric in METRICS:
                        actual=record['decisions'][decision]['budgets'][str(b)]['mean'][metric]
                        expected=float(np.mean([r[decision][metric] for r in vals]));error=max(error,abs(actual-expected))
                    for r in vals:
                        assert r[decision]['iou_lower']<=r[decision]['iou']+1e-12<=r[decision]['iou_upper']+1e-12
                    sampling_decisions+=8
                measured=np.stack([1-2*counts[:,bi,0]/b,2*counts[:,bi,2]/b-1],axis=1)
                # Very conservative 0.1% global Hoeffding audit, not the reported winner interval.
                radius=np.sqrt(2*np.log(2*len(tests)*len(chosen)*8*3*2*256/.001)/b)
                sample_bound_excess=max(sample_bound_excess,float(np.max(abs(measured-corr))-radius))
            if 'legacy_arctan' in record['selected_as']:
                prior_errors.append(record['decisions']['one_background']['exact']['iou']-old[c['case_id']])
            max_strength=max(max_strength,record['parameters']['max_abs_theta']);model_cases+=1
        result_hashes[c['case_id']]={suffix:hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest() for suffix in ('.json','.npz')}
        if index%25==0:print('Audited',index,'/',len(tests),flush=True)
    xml=ET.parse(OUT/'tests.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    test_count=sum(int(x.attrib['tests']) for x in suites);failures=sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)
    assert failures==0 and test_count>=37 and error<1e-12 and self_error<1e-9 and reciprocity<1e-8 and sample_bound_excess<=0
    report=dict(date='2026-10-08',status='passed',tests=test_count,failures=failures,prepared_cases=len(m['cases']),
        calibration_cases=len(calibration_cases()),test_cases=len(tests),model_case_results=model_cases,
        sampling_decisions_recomputed=sampling_decisions,max_metric_recompute_error=error,max_self_error=self_error,
        max_correlation_reciprocity_error=reciprocity,global_sampling_bound_excess=sample_bound_excess,
        maximum_dimensionless_bond=max_strength,legacy_exact_iou_max_difference=float(max(abs(np.asarray(prior_errors)))),
        legacy_exact_iou_nonidentical_cases=int(np.count_nonzero(abs(np.asarray(prior_errors))>1e-12)),
        legacy_reference_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),code_hashes=hashes(),result_hashes=result_hashes)
    write_json(ROOT/'weizmann-ising-seeds-validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='result_hashes'},indent=2))


if __name__=='__main__':main()
