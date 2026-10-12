"""Audit cached XYZ study artifacts without changing frozen experiment code."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/xyz-matplotlib')
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from run_weizmann_xyz import ROOT,OUT,manifest,hashes,calibration_cases,source_case,candidates,entries,BUDGETS,PRIMARY_BUDGET,METRICS
from xyz_alignment import metrics,uncertainty_radius
from weizmann_petz_data import write_json


def main():
    protocol=json.loads((ROOT/'weizmann-xyz-protocol.json').read_text())
    assert protocol['code_hashes']==hashes()
    assert protocol['manifest_sha256']==hashlib.sha256((ROOT/'weizmann-xyz-manifest.json').read_bytes()).hexdigest()
    m=manifest();cal=set(m['calibration_ids']);test=set(m['test_ids'])
    assert len(cal)==25 and len(test)==75 and cal.isdisjoint(test)
    for c in m['cases']:
        assert c['source_id'] in (cal if c['split']=='calibration' else test)
        f=source_case(c);truth=f['truth'];seeds=f['seeds']
        assert truth.shape==tuple(c['shape']) and list(truth.ravel()[seeds])==[False,True]
    calibration_residual=[]
    for c in calibration_cases():
        p=OUT/'calibration'/f'{c["case_id"]}.npz';f=np.load(p);meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['signature']==dict(case=c,code_hashes=hashes(),budget=PRIMARY_BUDGET,replicates=3)
        assert f['exact_metrics'].shape==(len(candidates()),2,4,len(METRICS))
        assert np.isfinite(f['sample_metrics']).all() and np.isfinite(f['exact_metrics']).all()
        calibration_residual.append(meta['eigen_residual'])
    test_cases=[c for c in m['cases'] if c['split']=='test'];max_metric_error=0.;max_self_error=0.;max_isotropic_error=0.
    records=0;replicates=0;residual=[];result_hashes={}
    for c in test_cases:
        name=c['case_id'];p=OUT/'test'/f'{name}.json';data=json.loads(p.read_text());f=np.load(p.with_suffix('.npz'));inputs=source_case(c)
        assert data['signature']==dict(case=c,selection=entries(c['shape']==[3,4]),code_hashes=hashes(),budgets=BUDGETS,replicates=16)
        truth=inputs['truth'];seeds=inputs['seeds'];n=truth.size
        for row in data['results']:
            label=row['label'];axes=f[label+'_axes'];draws=f[label+'_samples'];obs=row['entry']['observable'];decision=row['entry']['decision']
            assert axes.shape==(3,2,n) and draws.shape==(3,16,2,n)
            assert np.isfinite(axes).all() and np.isfinite(draws).all()
            assert abs(axes).max()<1+1e-11 and abs(draws).max()<1+1e-11
            for k,seed in enumerate(seeds):
                max_self_error=max(max_self_error,float(abs(axes[:,k,seed]-1).max()))
                other=np.arange(n)!=seed
                assert axes.mean(axis=0)[k,other].max()<=1/3+1e-11
            if label=='XYZ_tied_flexible' and row['entry']['rule']['xy_scale']==1:
                max_isotropic_error=max(max_isotropic_error,float(abs(axes-axes[0]).max()))
            target=axes.mean(axis=0) if obs=='alignment' else axes[2]
            expected=metrics(target,truth,seeds,decision)
            for key in METRICS:max_metric_error=max(max_metric_error,abs(expected[key]-row['exact'][key]))
            # Recompute every reported mean from stored sampling replicates.
            for bi,b in enumerate(BUDGETS):
                radius=uncertainty_radius(n,b,decision)
                values=[metrics(draw,truth,seeds,decision,radius) for draw in draws[bi]]
                for key in METRICS:
                    error=abs(float(np.mean([v[key] for v in values]))-row['budgets'][str(b)]['mean'][key])
                    max_metric_error=max(max_metric_error,error)
                replicates+=len(values)
            records+=1
        residual.append(data['eigen_residual'])
        result_hashes[name]={suffix:hashlib.sha256(p.with_suffix(suffix).read_bytes()).hexdigest() for suffix in ('.json','.npz')}
    xml=ET.parse(OUT/'tests.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    tests=sum(int(s.attrib['tests']) for s in suites);failures=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    assert failures==0 and tests>=22
    assert max_metric_error<1e-12 and max_self_error<1e-10 and max_isotropic_error<1e-10 and max(residual+calibration_residual)<1e-10
    report=dict(date='2026-10-08',status='passed',tests=tests,failures=failures,prepared_cases=len(m['cases']),exclusions=len(m['excluded']),
        calibration_cases=len(calibration_cases()),test_cases=len(test_cases),model_case_records=records,sampling_replicates_recomputed=replicates,
        max_stored_metric_error=max_metric_error,max_self_correlation_error=max_self_error,max_isotropic_axis_difference=max_isotropic_error,
        max_eigen_residual=max(residual+calibration_residual),code_hashes=hashes(),result_hashes=result_hashes)
    write_json(ROOT/'weizmann-xyz-validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='result_hashes'},indent=2))


if __name__=='__main__':main()
