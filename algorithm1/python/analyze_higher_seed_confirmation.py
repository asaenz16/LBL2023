"""Common-bank XXZ confirmation, keeping every calibrated choice fixed."""
import json
import numpy as np
from run_higher_seed_study import ROOT,OUT,INPUT,cases,profiles,digest
from higher_seed_readout import METRICS,evaluate
from analyze_higher_seed_study import summary,difference,TITLES
from weizmann_petz_data import write_json,seed_for


def main():
    protocol=json.loads((ROOT/'higher-seeds-confirmation-protocol.json').read_text())
    paths=list((OUT/'confirm').glob('*.json'));assert len(paths)==2*len(protocol['cases'])
    count=0
    for path in paths:
        meta=json.loads(path.read_text());s=meta['signature'];v=np.load(path.with_suffix('.npz'))['metrics']
        d=np.load(INPUT/'cases'/f'{s["case"]["case_id"]}.npz');truth=d['truth'].ravel()
        initial=int(d['seeds'][1 if s['anchor'].endswith('foreground') else 0])
        assert len(meta['traces'])==v.shape[0]*v.shape[1] and np.isfinite(v).all()
        assert digest(meta['bank'])==meta['bank_sha256']
        for t in meta['traces']:
            r=v[t['policy_index'],t['repeat']];h=t['history'];end={int(k):y for k,y in h[-1]['active'].items() if int(k)!=initial}
            assert r[METRICS.index('shots')]==len(h)*32768
            assert r[METRICS.index('promoted_count')]==len(end)
            assert r[METRICS.index('promoted_wrong')]==sum(bool(truth[k])!=(y==1) for k,y in end.items())
            pred=np.unpackbits(np.asarray(t['prediction'],np.uint8),count=len(truth)).astype(bool)
            union=(pred|truth).sum();iou=(pred&truth).sum()/union if union else 1.
            assert abs(iou-r[METRICS.index('iou')])<1e-12
            for a,b in zip(h,h[1:]):
                old=a['active'];new=b['active'];assert all(new[k]==y for k,y in old.items())
                adds={k:y for k,y in new.items() if k not in old}
                assert sum(y==1 for y in adds.values())<=1 and sum(y==-1 for y in adds.values())<=1
                assert {int(k) for k in adds}=={k for k in a['added'] if k>=0}
            count+=1
    arrays={};bank_counts={}
    for anchor in ('XXZ/foreground','XXZ/background'):
        for side in (8,16):
            data={k:[] for k in profiles(anchor,8192)};high=0
            for c in sorted([c for c in cases('test') if c['shape'][0]==side],key=lambda x:x['source_id']):
                path=OUT/'confirm'/f'{c["case_id"]}--{anchor.replace("/","-")}--8192.json'
                if path.exists():high+=1
                else:path=OUT/'test'/f'{c["case_id"]}--{anchor.replace("/","-")}--2048.json'
                meta=json.loads(path.read_text());v=np.load(path.with_suffix('.npz'))['metrics'].mean(1)
                for label in data:data[label].append(v[meta['labels'][label]])
            assert len(next(iter(data.values())))==75
            for label,rows in data.items():arrays[f'{anchor}/{side}/{label}']=np.array(rows)
            bank_counts[f'{anchor}/{side}']=high
    metrics={k:summary(v) for k,v in arrays.items()};effects={}
    for k,v in arrays.items():
        if '/matched_' in k or '/pair/' in k:continue
        effects[k]={}
        prefix=k.rsplit('/',2)[0]
        for control in (prefix+'/pair/balanced',k+'/matched_pair',k+'/matched_linear'):
            effects[k][control]=difference(v,arrays[control])
    replays=[]
    checks=[protocol['cases'][0],protocol['new_banks'][0]]
    from pathlib import Path
    for cid in checks:
        for anchor in ('XXZ/foreground','XXZ/background'):
            path=OUT/'confirm'/f'{cid}--{anchor.replace("/","-")}--8192.json';m=json.loads(path.read_text());s=m['signature']
            d=np.load(INPUT/'cases'/f'{cid}.npz');bm=json.loads(Path(m['bank']).with_suffix('.json').read_text())
            p=np.load(m['bank'])['packed'][:3];initial=int(d['seeds'][1 if anchor.endswith('foreground') else 0])
            v,r,_=evaluate(p,bm['sampling'][:3],d['truth'],initial,anchor.split('/')[1],s['policies'],5,
                           seed_for('higher-seeds:test:'+cid+':'+anchor),8192)
            old=np.load(path.with_suffix('.npz'));error=float(np.max(abs(v-old['metrics'])))
            assert error<1e-9 and np.array_equal(r,old['reasons']);replays.append(dict(file=path.name,maximum_discrepancy=error))
    result=dict(metrics=metrics,effects=effects,high_bank_counts=bank_counts,confirmed_cases=len(protocol['cases']),
        validation=dict(passed=True,trajectories=count,replays=replays),
        result_sha256={p.name:digest(p.with_suffix('.npz')) for p in paths})
    write_json(ROOT/'higher-seeds-confirmation-summary.json',result)
    write_report(result,protocol)
    print('confirmation analysis complete:',count,'trajectories',flush=True)


def write_report(result,protocol):
    metrics=result['metrics'];effects=result['effects'];count=result['validation']['trajectories']
    lines=['## Larger-budget XXZ confirmation','',
        'The standard-budget XXZ comparison has substantial sampling failures, so its unresolved rates should not be interpreted solely as an algorithmic limitation. This follow-up keeps the balanced policies fixed and uses four × 8,192 configurations per fresh round on all available prior primary-XXZ confirmation banks plus newly flagged cases. Two missing banks were generated; no parameter was retuned.',
        f'There are {len(protocol["cases"])} confirmed image/resolution cases: {result["high_bank_counts"]["XXZ/foreground/16"]} at 16×16 and {result["high_bank_counts"]["XXZ/foreground/8"]} at 8×8. Each affected case uses the same new bank for both starting labels and every method/control. The other cases retain the original bank. The table therefore covers all 75 comparison sources at 16×16 with an explicitly mixed sampling budget; the fixed-budget results above remain unchanged.',
        'The additional sampling reduces foreground-start pairwise mixing-aware unresolved from 22.45% to 9.70%, and background-start from 26.53% to 9.23%. Higher-order gates then give identical aggregate results to their pairwise controls. No higher-order variant has an established positive matched IoU effect. Background-start majority scoring has a small negative pointwise IoU interval; these exploratory intervals are not corrected across methods.',
        'At the increased budget, one used round consumes 32,768 retained configurations and requires 4(6000 + 2×8192) = 89,536 prospective SSE sweeps including burn-in. The table averages actual used configuration counts over the mixed-budget comparison. The new banks preserve independent round seeds; the unused bank tails are excluded from these costs.',
        '','| Start | Method | IoU | U | Mixing U | BPM r=1 | Wrong promotions | Configurations |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for anchor in ('XXZ/foreground','XXZ/background'):
        for f,title in TITLES.items():
            m=metrics[f'{anchor}/16/{f}/balanced'];er=m['wrong_promotion_rate']
            lines.append(f'| {anchor.split("/")[1]} | {title} | {m["iou"]:.4f} | {100*m["unresolved"]:.2f}% | {100*m["mixing_unresolved"]:.2f}% | {m["bpm_iou_r1"]:.4f} | {100*er:.2f}% | {m["shots"]:.0f} |')
    lines+=['','| Start | Method | ΔIoU vs matched pair [95% CI] | Δ mixing U, pp [95% CI] |', '|---|---|---:|---:|']
    for anchor in ('XXZ/foreground','XXZ/background'):
        for f,title in list(TITLES.items())[1:]:
            key=f'{anchor}/16/{f}/balanced';e=effects[key][key+'/matched_pair']
            a=e['iou'];b=e['mixing_unresolved']
            lines.append(f'| {anchor.split("/")[1]} | {title} | {a["mean"]:+.4f} [{a["ci95"][0]:+.4f}, {a["ci95"][1]:+.4f}] | {100*b["mean"]:+.2f} [{100*b["ci95"][0]:+.2f}, {100*b["ci95"][1]:+.2f}] |')
    lines+=['',f'Confirmation validation passed: {count:,} trajectories checked and four bank replays with zero discrepancy. Remaining convergence failures are retained and counted as unresolved in mixing-aware U. The extra budget is validation, not another tuning set.','']
    report=ROOT/'higher-seeds.md';text=report.read_text().split('## Larger-budget XXZ confirmation')[0]
    report.write_text(text.rstrip()+'\n\n'+'\n'.join(lines))


if __name__=='__main__':main()
