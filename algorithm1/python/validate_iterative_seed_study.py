"""Completeness, source, promotion-causality and saved-array audits."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from run_iterative_seed_study import ROOT,OUT,INPUT,check_protocol,hashes,digest,cases,rules,selected_jobs,identifier,confirmation_specs
from iterative_seed_readout import METRICS,POLICIES,REASONS,evaluate_grid
from weizmann_petz_metrics import mask_metrics
from segmentation_bpm import bpm_metrics
from weizmann_petz_data import write_json,seed_for


def result_files(stage):return [p for p in sorted((OUT/stage).glob('*.json')) if not p.name.endswith('.bank.json')]


def main():
    check_protocol();records=0;endpoints=0;trace_count=0;max_error=0.;sampling_flags=0;stage_counts={};audited={};amended=0
    amendment_path=ROOT/'iterative-seeds-numerical-amendment.json'
    amendment=json.loads(amendment_path.read_text())
    assert amendment['base_code_hashes']==hashes()
    assert amendment['extended_sampler_sha256']==digest(ROOT/'python/iterative_seed_extended_loops.py')
    for stage in ('calibrate','test','confirm'):
        paths=result_files(stage);stage_counts[stage]=len(paths)
        for path in paths:
            row=json.loads(path.read_text());sig=row['signature'];src=INPUT/'cases'/f'{row["case"]["case_id"]}.npz'
            assert sig['code_hashes']==hashes() and sig['input_sha256']==digest(src)
            assert row['bank_sha256']==digest(path.with_suffix('.bank.npz'))
            expected_amendment=stage!='calibrate' or path.stem==amendment['failed_case']
            assert ('execution_amendment' in row)==expected_amendment
            if expected_amendment:
                assert row['execution_amendment']==dict(amendment_sha256=digest(amendment_path),extended_sampler_sha256=amendment['extended_sampler_sha256'])
                amended+=1
            values=np.load(path.with_suffix('.npz'));table=values['metrics'];reason=values['reasons']
            policies=POLICIES if stage=='calibrate' else row['policies'];shape=(len(policies),sig['repeats'],len(METRICS))
            assert table.shape==shape and reason.shape==shape[:2] and np.isfinite(table).all()
            assert np.isin(reason,np.arange(len(REASONS))).all()
            for m in ('iou','unknown_iou','unresolved','mixing_unresolved','mixing_lower','correct_resolved','wrong_resolved','bpm_iou_r0','bpm_iou_r1','bpm_iou_r2','promoted_error','converged'):
                assert np.all((table[:,:,METRICS.index(m)]>=-1e-12)&(table[:,:,METRICS.index(m)]<=1+1e-12))
            assert np.all(table[:,:,METRICS.index('iou_lower')]<=table[:,:,METRICS.index('iou')]+1e-12)
            assert np.all(table[:,:,METRICS.index('iou')]<=table[:,:,METRICS.index('iou_upper')]+1e-12)
            assert np.all(table[:,:,METRICS.index('promoted_count')]<=2*(table[:,:,METRICS.index('rounds')]-1))
            assert np.all(table[:,:,METRICS.index('bpm_iou_r0')]<=table[:,:,METRICS.index('bpm_iou_r1')]+1e-12)
            assert np.all(table[:,:,METRICS.index('bpm_iou_r1')]<=table[:,:,METRICS.index('bpm_iou_r2')]+1e-12)
            packed=np.load(path.with_suffix('.bank.npz'))['packed'];n=int(np.prod(row['case']['shape']))
            assert packed.shape==(row['rounds_generated'],sig['settings']['chains'],sig['settings']['samples'],(n+7)//8)
            assert all(d['maximum_order']<=d['capacity'] for d in row['sampling'])
            if stage!='calibrate':
                data=np.load(src);truth=data['truth'];seeds=data['seeds']
                for tr in row['traces']:
                    pi=tr['policy_index'];rep=tr['repeat'];policy=policies[pi];h=tr['history']
                    initial=int(seeds[1 if policy['start']=='foreground' else 0]);f0=[initial] if policy['start']=='foreground' else [];b0=[] if f0 else [initial]
                    assert h[0]['foreground']==f0 and h[0]['background']==b0
                    assert len(h)<=policy['cap'] and len(h)<=row['rounds_generated']
                    for t,step in enumerate(h):
                        assert step['round']==t+1
                        assert not set(step['foreground'])&set(step['background'])
                        assert len(step['foreground'])+len(step['background'])<=2*t+1
                        fp=[p for p in step['foreground'] if p!=initial];bp=[p for p in step['background'] if p!=initial]
                        wrong=int(np.sum(~truth.ravel()[fp]))+int(np.sum(truth.ravel()[bp]))
                        assert step['metrics']['promoted_wrong']==wrong
                        if t+1<len(h):
                            af,ab=step['added'];assert h[t+1]['foreground']==step['foreground']+([af] if af>=0 else [])
                            assert h[t+1]['background']==step['background']+([ab] if ab>=0 else [])
                        else:assert step['added']==[-1,-1]
                    max_error=max(max_error,float(np.max(abs(table[pi,rep]-np.array([h[-1]['metrics'][m] for m in METRICS])))))
                    pred=np.unpackbits(np.array(tr['prediction'],np.uint8),count=n).astype(bool)
                    resolved=np.unpackbits(np.array(tr['resolved'],np.uint8),count=n).astype(bool)
                    actual=dict(**mask_metrics(pred,truth),**bpm_metrics(pred.reshape(truth.shape),truth))
                    for m,v in actual.items():max_error=max(max_error,abs(v-h[-1]['metrics'][m]))
                    unknown=np.arange(n)!=initial
                    max_error=max(max_error,abs(np.mean(~resolved[unknown])-h[-1]['metrics']['unresolved']))
                    assert REASONS[reason[pi,rep]]==tr['stop'];trace_count+=1
            endpoints+=table.shape[0]*table.shape[1];records+=1
            audited[str(path.relative_to(OUT))]=dict(json=digest(path),arrays=digest(path.with_suffix('.npz')),bank=row['bank_sha256'])
            if records%300==0:print('Audited',records,flush=True)
    assert stage_counts['calibrate']==len(rules())*len(cases('calibration'))==4800
    assert stage_counts['test']==len(selected_jobs())*len(cases('test'))
    expected={f'{c["case_id"]}--{identifier(r)}.json' for _,c,r,p in confirmation_specs()}
    assert expected=={p.name for p in result_files('confirm')}
    # Replay a deterministic hash-selected subset from packed measurements.
    allpaths=result_files('calibrate')+result_files('test')+result_files('confirm')
    picks=sorted(allpaths,key=lambda p:seed_for('iterative-audit:'+str(p)))[:24]
    for path in picks:
        row=json.loads(path.read_text());sig=row['signature'];src=np.load(INPUT/'cases'/f'{row["case"]["case_id"]}.npz')
        packed=np.load(path.with_suffix('.bank.npz'))['packed'];expected=np.load(path.with_suffix('.npz'))
        policies=POLICIES if sig['stage']=='calibrate' else row['policies']
        values,reasons,_=evaluate_grid(packed,row['sampling'],src['truth'],src['seeds'],sig['repeats'],sig['promotion_key'],policies)
        max_error=max(max_error,float(np.max(abs(values-expected['metrics']))));np.testing.assert_array_equal(reasons,expected['reasons'])
    suites=[]
    for filename in ('tests.xml','amendment-tests.xml'):
        xml=ET.parse(OUT/filename).getroot();suites.extend(list(xml) if xml.tag=='testsuites' else [xml])
    tests=sum(int(x.attrib['tests']) for x in suites);failed=sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)
    assert max_error<1e-10 and failed==0
    result=dict(status='passed',tests=tests,test_failures=failed,model_image_records=records,stage_counts=stage_counts,
        parameter_random_endpoints=endpoints,test_confirmation_trajectories=trace_count,replayed_banks=len(picks),
        maximum_metric_discrepancy=max_error,code_hashes=hashes(),result_hashes=audited,
        numerical_amendment=amendment,amended_sampler_records=amended,
        scope='All saved records checked for signatures, completeness, bounds and trace causality. All test/confirmation final masks and promotion errors recomputed.24 deterministic banks replayed. This audit is not a convergence guarantee.')
    write_json(ROOT/'iterative-seeds-validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('code_hashes','result_hashes')},indent=2))


if __name__=='__main__':main()
