"""Supplementary reliability objective: paired validation and summaries."""
import os
for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json
from collections import defaultdict
import numpy as np
from promotion_reliability_targeted import OUT,SELECT,groups
from run_promotion_reliability import ROOT,INPUT,cases,identifier,anchors,hashes,digest,REPEATS
from analyze_promotion_reliability import compact,summarize,difference,save,plt
from audit_promotion_reliability import audit_trace
from weizmann_petz_data import write_json
from promotion_reliability import evaluate


def plot_targeted(summary,chosen):
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,anchor in zip(axes.ravel(),chosen['winners']):
        for x,(label,color) in enumerate([('historical','#777777'),('reliability/matched','#cd9753'),('reliability','#28788e')]):
            r=summary[f'{anchor}|{label}|16'];ax.scatter(r['metrics']['promoted_count']['mean'],r['reliability']['active_error']['mean'],s=140 if label.endswith('/matched') else 70,facecolors='none' if label.endswith('/matched') else color,edgecolors=color,linewidths=2,label=label)
            ax.annotate(f"IoU {r['metrics']['iou']['mean']:.3f}",(r['metrics']['promoted_count']['mean'],r['reliability']['active_error']['mean']),xytext=(-5 if r['metrics']['promoted_count']['mean']>7 else 5, -18 if label=='reliability' else 10),ha='right' if r['metrics']['promoted_count']['mean']>7 else 'left',textcoords='offset points',fontsize=8)
        ax.set_title(anchor.replace('/mixing_lower',''));ax.set_xlabel('Mean active promoted seeds');ax.set_ylabel('Wrong active seed fraction');ax.grid(alpha=.2);ax.set_xlim(0,12);ax.set_ylim(.20,.38)
    axes[0,0].legend(fontsize=8);fig.suptitle('Reliability-targeted calibration, evaluated on 75 images at 16×16\nPoint estimates; paired confidence intervals are reported in the tables');save(fig,'reliability-targeted-16')


def main():
    assert len([p for p in OUT.glob('*.json') if p.name!='case-summary.json'])==450
    collected=defaultdict(list);checked=0;confirmed=0;records=[];replayed=0
    for c in cases('test'):
        data=np.load(INPUT/'cases'/f'{c["case_id"]}.npz')
        for group in groups():
            path=OUT/f'{c["case_id"]}--{identifier(group["rule"])}.json';row=json.loads(path.read_text());arr=np.load(path.with_suffix('.npz'));confirmed+=row['stage']=='confirm'
            bp=OUT.parent/'banks'/row['stage']/path.name;meta=json.loads(bp.read_text())
            assert row['bank_sha256']==meta['bank_sha256']==digest(bp.with_suffix('.npz'))
            do_replay=c['source_id']==cases('test')[0]['source_id']
            if do_replay:replayed+=1
            for anchor,profile in row['profiles'].items():
                table=arr[f'metrics{profile["array_index"]}'];local={}
                if do_replay:
                    value,reason,traces=evaluate(np.load(bp.with_suffix('.npz'))['packed'],meta['sampling'],data['truth'],data['seeds'],anchors()[anchor]['base'],profile['policies'],10,profile['promotion_key'],True)
                    np.testing.assert_array_equal(value,table);np.testing.assert_array_equal(reason,arr[f'reasons{profile["array_index"]}'])
                    assert json.loads(json.dumps(traces))==profile['traces']
                for tr in profile['traces']:
                    audit_trace(data['truth'],data['seeds'],anchors()[anchor]['base'],profile['policies'][tr['policy_index']],tr);checked+=1
                for label,pi in profile['labels'].items():
                    if pi not in local:local[pi]=compact(c,anchor,profile,table,pi,row['stage']=='confirm');records.append(local[pi])
                    collected[f'{anchor}|{label}|{c["shape"][0]}'].append(local[pi])
    summary={k:summarize(v,k) for k,v in collected.items()};paired={}
    chosen=json.loads(SELECT.read_text())
    for anchor in chosen['winners']:
        for size in (8,16):
            key=f'{anchor}|reliability|{size}';paired[key]={label:difference(collected[key],collected[f'{anchor}|{label}|{size}'],key+label) for label in ('historical','reliability/matched')}
    result=dict(status='completed',selection=chosen,groups=summary,paired=paired,banks=450,confirmed_banks=confirmed,validated_traces=checked,replayed_banks=replayed,max_metric_replay_difference=0.,code_hashes=hashes())
    write_json(ROOT/'promotion-reliability-targeted-summary.json',result);write_json(OUT/'case-summary.json',records)
    plot_targeted(summary,chosen)
    for anchor in chosen['winners']:
        key=f'{anchor}|reliability|16';r=summary[key]
        print(anchor,{m:r['metrics'][m]['mean'] for m in ('iou','unresolved','promoted_count')},r['reliability']['active_error'],paired[key]['historical']['iou'],flush=True)


if __name__=='__main__':main()
