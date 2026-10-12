"""Artifact audits for paired XXZ scores and BPM; never changes simulations."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize,disk
from run_xxz_two_seed_study import ROOT,OUT,INPUT,hashes,check_protocol,cases,selected_rules,identifier,confirmation_specs,digest
from xxz_two_seed import evaluate_arrays,parameterize,DECISIONS,scores
from segmentation_bpm import boundary_proximity
from weizmann_petz_data import write_json


def main():
    check_protocol();rng=np.random.default_rng(9107);morphology_checks=0
    for size in (8,16,32):
        for trial in range(12):
            a=rng.random((size,size))<.3;b=rng.random((size,size))<.3
            p=skeletonize(a,method='zhang');g=skeletonize(b,method='zhang');last=-1.
            for radius in (0,1,2):
                dp=binary_dilation(p,structure=disk(radius));dg=binary_dilation(g,structure=disk(radius))
                tp=int(np.sum(p&dg));fp=int(np.sum(p&~dg));fn=int(np.sum(g&~dp))
                actual=boundary_proximity(a,b,radius)
                assert (tp,fp,fn)==(actual['tp'],actual['fp'],actual['fn'])
                assert actual['iou']>=last;last=actual['iou'];morphology_checks+=1
    errors=0.;correlation_error=0.;records=0;decisions=0;result_hashes={};confirmed=set()
    for stage in ('calibrate','test','confirm'):
        for path in sorted((OUT/stage).glob('*.json')):
            row=json.loads(path.read_text());c=row['case'];src=INPUT/'cases'/f'{c["case_id"]}.npz'
            assert row['signature']['code_hashes']==hashes()
            assert row['signature']['input_sha256']==digest(src)
            source=np.load(src);a=np.load(path.with_suffix('.npz'))
            metrics,exact,inference=evaluate_arrays(a,source['truth'],source['seeds'],row['sampling']['converged'])
            for d in DECISIONS:
                for k,v in metrics[d].items():errors=max(errors,abs(v-row['decisions'][d][k]))
                assert metrics[d]['iou_lower']<=metrics[d]['iou']+1e-12<=metrics[d]['iou_upper']+1e-12
                decisions+=1
            if exact is not None:
                for d in DECISIONS:
                    for k,v in exact[d].items():errors=max(errors,abs(v-row['exact'][d][k]))
            for k in ('score','score_se','score_radius'):
                other={'score':'score','score_se':'se','score_radius':'radius'}[k]
                errors=max(errors,float(abs(a[k]-inference[other]).max()))
            assert np.isfinite(a['gamma']).all() and abs(a['gamma']).max()<=.25+1e-10
            correlation_error=max(correlation_error,float(abs(a['gamma'][[0,1],source['seeds']]-.25).max()))
            model,p=parameterize(source['image'],row['rule'])
            for k in ('threshold','alpha','g','b'):errors=max(errors,abs(p[k]-row['parameters'][k]))
            assert row['sampling']['maximum_order']<=row['sampling']['capacity']
            if stage=='confirm':confirmed.add(path.name)
            result_hashes[str(path.relative_to(OUT))]=dict(json=digest(path),npz=digest(path.with_suffix('.npz')))
            records+=1
            if records%200==0:print('Audited',records,flush=True)
    requested={f'{c["case_id"]}--{identifier(rule)}.json' for _,c,rule in confirmation_specs()}
    assert requested==confirmed
    assert len(list((OUT/'calibrate').glob('*.json')))==2000
    assert len(list((OUT/'test').glob('*.json')))==150*len(selected_rules())
    # Check old masks/paired errors and source hashes, including cached longer runs.
    old_count=0;legacy_iou_error=0.
    for path in sorted((OUT/'reanalysis').glob('*/*.json')):
        row=json.loads(path.read_text());old=ROOT/row['source'];original=json.loads(old.read_text())
        assert row['source_sha256']==digest(old) and row['array_sha256']==digest(old.with_suffix('.npz'))
        src=np.load(INPUT/'cases'/f'{row["case"]["case_id"]}.npz');a=np.load(old.with_suffix('.npz'))
        metrics,_,_=evaluate_arrays(a,src['truth'],src['seeds'],row['converged'])
        for d in DECISIONS:
            for k,v in metrics[d].items():errors=max(errors,abs(v-row['decisions'][d][k]))
        legacy_iou_error=max(legacy_iou_error,abs(metrics['one_background']['iou']-original['metrics']['iou']))
        old_count+=1
    xml=ET.parse(OUT/'tests.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    tests=sum(int(x.attrib['tests']) for x in suites);failures=sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)
    assert errors<1e-12 and correlation_error<1e-9 and legacy_iou_error<1e-12 and failures==0
    result=dict(status='passed',tests=tests,failures=failures,new_model_cases=records,new_readouts_recomputed=decisions,
        old_model_cases=old_count,max_metric_error=errors,max_self_correlation_error=correlation_error,
        legacy_one_seed_iou_difference=legacy_iou_error,independent_morphology_checks=morphology_checks,
        required_confirmations=len(requested),code_hashes=hashes(),result_hashes=result_hashes)
    write_json(ROOT/'xxz-two-seed-validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('result_hashes','code_hashes')},indent=2))


if __name__=='__main__':main()
