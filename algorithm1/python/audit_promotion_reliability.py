"""Provenance, transition, metric and deterministic replay audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from run_promotion_reliability import *
from promotion_reliability import evaluate,spatial_mask


def audit_trace(truth,seeds,base,p,tr):
    initial=int(seeds[1 if base['start']=='foreground' else 0]);last=None;n=truth.size
    assert not truth.ravel()[int(seeds[0])] and truth.ravel()[int(seeds[1])]
    for t,h in enumerate(tr['history']):
        f=h['foreground'];b=h['background'];active={x:1 for x in f}|{x:-1 for x in b};weights={int(x):y for x,y in h['weights'].items()};retired={int(x):y for x,y in h['retired'].items()}
        assert len(set(f+b))==len(f+b) and initial in active and weights[initial]==1 and not set(active)&set(retired)
        assert active[initial]==(1 if base['start']=='foreground' else -1)
        assert len(h['added'])==2 and h['added'][0]!=h['added'][1] or h['added']==[-1,-1]
        assert h['round']==t+1 and t<p['cap']
        promoted={x:y for x,y in active.items() if x!=initial};wrong=[x for x,y in promoted.items() if bool(truth.ravel()[x])!=(y==1)]
        m=h['metrics'];assert m['promoted_count']==len(promoted) and m['promoted_wrong']==len(wrong)
        np.testing.assert_allclose(m['wrong_weight'],sum(weights[x] for x in wrong),atol=1e-12)
        ever={int(x):y for x,y in h['ever'].items()}
        assert ever==promoted|retired and initial not in ever
        retired_wrong=sum(bool(truth.ravel()[x])!=(y==1) for x,y in retired.items())
        assert m['retired_count']==len(retired) and m['retired_wrong']==retired_wrong
        assert m['ever_promoted']==len(ever) and m['ever_wrong']==len(wrong)+retired_wrong
        if last:
            before={x:1 for x in last['foreground']}|{x:-1 for x in last['background']}
            after=before.copy()
            for x in last['removed']:assert x!=initial;after.pop(x)
            for role,x in zip((1,-1),last['added']):
                if x>=0:
                    assert x not in before and x not in last['retired'] and str(x) not in last['retired']
                    assert spatial_mask(truth.shape,list(before),list(before.values()),role,p['space'])[x]
                    after[x]=role
            assert after==active
            expected={int(x):y for x,y in last['next_weights'].items()}
            expected.update({x:1. for x in last['added'] if x>=0});assert expected==weights
            if p['space'].startswith('all') and min(last['added'])>=0:
                assert spatial_mask(truth.shape,[last['added'][0]],[1],-1,p['space'])[last['added'][1]]
        if m['unresolved']==0:assert h['added']==[-1,-1]
        last=h
    assert last['added']==[-1,-1] and last['removed']==[]
    pred=np.unpackbits(np.array(tr['prediction'],np.uint8),count=n).astype(bool);resolved=np.unpackbits(np.array(tr['resolved'],np.uint8),count=n).astype(bool);target=truth.ravel()
    union=np.sum(pred|target);iou=np.sum(pred&target)/union if union else 1
    np.testing.assert_allclose(iou,last['metrics']['iou'],atol=1e-12)
    np.testing.assert_allclose((~resolved[np.arange(n)!=initial]).mean(),last['metrics']['unresolved'],atol=1e-12)


def main():
    check_protocol()
    assert len(list((OUT/'calibrate').glob('*.json')))==150
    assert len(list((OUT/'test').glob('*.json')))==750
    expected=confirmation_specs()
    assert len(list((OUT/'confirm').glob('*.json')))==len(expected)
    for stage,c,g in expected:assert (OUT/stage/f'{c["case_id"]}--{identifier(g["rule"])}.json').exists()
    count=dict(banks=0,prefixes=0,calibration_endpoints=0,comparison_endpoints=0,traces=0,replayed_banks=0);replay=[]
    for stage in ('calibrate','test','confirm'):
        for path in sorted((OUT/stage).glob('*.json')):
            row=json.loads(path.read_text());assert row['signature']['code_hashes']==hashes();data=np.load(INPUT/'cases'/f'{row["case"]["case_id"]}.npz');arrays=np.load(path.with_suffix('.npz'))
            bp=OUT/'banks'/stage/path.name;meta=json.loads(bp.read_text());assert meta['bank_sha256']==row['bank_sha256']==digest(bp.with_suffix('.npz'));packed=np.load(bp.with_suffix('.npz'))['packed'];assert 1<=len(packed)<=10;count['banks']+=1
            if len(packed)<10:
                amendment=json.loads((ROOT/'promotion-reliability-execution-amendment.json').read_text())
                assert stage=='confirm' and len(packed)>=amendment['maximum_requested_caps'][identifier(row['rule'])]
                assert meta['execution_amendment']['sha256']==digest(ROOT/'promotion-reliability-execution-amendment.json')
            if meta['prefix']:
                old=meta['prefix'];assert digest(old['path'])==old['sha256'];np.testing.assert_array_equal(packed[:old['rounds']],np.load(old['path'])['packed']);count['prefixes']+=1
            for anchor,profile in row['profiles'].items():
                a=arrays[f'metrics{profile["array_index"]}'];assert np.isfinite(a).all()
                count['calibration_endpoints' if stage=='calibrate' else 'comparison_endpoints']+=a.shape[0]*a.shape[1]
                for tr in profile['traces']:
                    p=profile['policies'][tr['policy_index']];audit_trace(data['truth'],data['seeds'],anchors()[anchor]['base'],p,tr);count['traces']+=1
                    np.testing.assert_allclose([tr['history'][-1]['metrics'][m] for m in METRICS],a[tr['policy_index'],tr['repeat']],atol=1e-12)
            # Deterministic coverage: several low/high dimension banks per stage.
            if stage!='calibrate' and len([r for r in replay if r[0]==stage])<8:replay.append((stage,path))
    maxdiff=0.
    for stage,path in replay:
        row=json.loads(path.read_text());data=np.load(INPUT/'cases'/f'{row["case"]["case_id"]}.npz');bp=OUT/'banks'/stage/path.name;meta=json.loads(bp.read_text());packed=np.load(bp.with_suffix('.npz'))['packed'];arrays=np.load(path.with_suffix('.npz'))
        for anchor,profile in row['profiles'].items():
            a,r,tr=evaluate(packed,meta['sampling'],data['truth'],data['seeds'],anchors()[anchor]['base'],profile['policies'],REPEATS[stage],profile['promotion_key'],True)
            old=arrays[f'metrics{profile["array_index"]}'];maxdiff=max(maxdiff,float(np.max(abs(a-old))));np.testing.assert_array_equal(a,old)
            np.testing.assert_array_equal(r,arrays[f'reasons{profile["array_index"]}'])
            # JSON normalizes integer dictionary keys; compare that representation.
            assert json.loads(json.dumps(tr))==profile['traces']
        count['replayed_banks']+=1
    assert count['calibration_endpoints']==1350000
    result=dict(status='passed',counts=count,max_metric_replay_difference=maxdiff,code_hashes=hashes(),protocol_sha256=digest(ROOT/'promotion-reliability-protocol.json'))
    write_json(ROOT/'promotion-reliability-validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
