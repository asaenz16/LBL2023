"""Audit and summarize the explicitly post-hoc threshold/temperature factorial."""
import hashlib
import json
from pathlib import Path
import numpy as np
from run_ising_seed_threshold_diagnostic import RULES,LABELS
from run_weizmann_ising_seeds import ROOT,OUT,PRIMARY,METRICS,DECISIONS,hashes,read_case
from analyze_weizmann_ising_seeds import load_results,record,ci,save
from ising_seed_sampling import evaluate_counts,exact_decision,segmentation_metrics
from weizmann_petz_data import write_json
import matplotlib.pyplot as plt


def main():
    records=load_results();rows=[];max_error=0.;audited=0;result_hashes={}
    script=Path(__file__).with_name('run_ising_seed_threshold_diagnostic.py')
    code_sha256=hashlib.sha256(script.read_bytes()).hexdigest()
    protocol=json.loads((ROOT/'weizmann-ising-seeds-threshold-diagnostic-protocol.json').read_text())
    assert protocol['code_hashes']==hashes() and protocol['diagnostic_sha256']==code_sha256
    for index,data in enumerate(records,1):
        case=data['case'];source=read_case(case);truth,seeds=source['truth'],source['seeds'];n=truth.size
        path=OUT/'threshold-diagnostic'/f'{case["case_id"]}.json';diag=json.loads(path.read_text())
        assert diag['signature']==dict(case=case,rules=RULES,code_hashes=hashes(),diagnostic_sha256=code_sha256,budget=PRIMARY,replicates=8)
        result_hashes[case['case_id']]={suffix:hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest() for suffix in ('.json','.npz')}
        corners={}
        for key,label in [('fixed75_K04','arctan_cubic__two_max'),('q90_K08','matched__arctan_cubic')]:
            r=record(data,label);corners[key]={d:dict(sample=r['decisions'][d]['budgets'][str(PRIMARY)]['mean'],exact=r['decisions'][d]['exact']) for d in DECISIONS}
        with np.load(path.with_suffix('.npz')) as arrays:
            for label,rule,r in zip(LABELS,RULES,diag['records']):
                assert r['label']==label and r['rule']==rule
                counts=arrays[label+'_counts'];corr=arrays[label+'_correlations']
                assert counts.shape==(8,4,n) and corr.shape==(2,n)
                assert np.isfinite(corr).all() and np.max(abs(corr))<=1+1e-9
                assert counts.min()>=0 and counts.max()<=PRIMARY and np.all(counts[:,1]<=counts[:,3])
                assert np.max(abs(corr[np.arange(2),seeds]-1))<1e-9
                values=[evaluate_counts(c,PRIMARY,truth,seeds) for c in counts]
                for decision in DECISIONS:
                    exact=segmentation_metrics(*exact_decision(corr,decision),truth,seeds)
                    for m in METRICS:
                        expected=float(np.mean([v[decision][m] for v in values]))
                        max_error=max(max_error,abs(expected-r['decisions'][decision]['sample'][m]),abs(exact[m]-r['decisions'][decision]['exact'][m]))
                    audited+=8
                corners[label]=r['decisions']
        rows.append(dict(case=case,corners=corners))
        if index%25==0:print('Audited diagnostic',index,'/',len(records),flush=True)
    assert max_error<1e-12
    summary=dict(status='post-hoc mechanism diagnostic; no new main-model selection',
        function='Delta + 4 Delta^3',sensitivity='1/max(q90(d)-q10(d),1)',sizes={},
        validation=dict(status='passed',test_cases=len(records),additional_model_cases=2*len(records),
            sampling_decisions_recomputed=audited,max_metric_recompute_error=max_error,code_hashes=hashes(),
            diagnostic_sha256=code_sha256,result_hashes=result_hashes))
    keys=['fixed75_K04','q90_K04','fixed75_K08','q90_K08']
    for side in (8,16):
        group=[r for r in rows if r['case']['side']==side]
        result=dict(sources=len(group),corners={},paired={})
        for key in keys:
            result['corners'][key]={d:{kind:{m:float(np.mean([r['corners'][key][d][kind][m] for r in group])) for m in METRICS} for kind in ('sample','exact')} for d in DECISIONS}
        for tag,a,b in [('threshold_at_K04','q90_K04','fixed75_K04'),('threshold_at_K08','q90_K08','fixed75_K08'),
            ('K_at_fixed75','fixed75_K08','fixed75_K04'),('K_at_q90','q90_K08','q90_K04')]:
            result['paired'][tag]={m:ci([r['corners'][a]['two_max']['sample'][m]-r['corners'][b]['two_max']['sample'][m] for r in group],str(side)+tag+m) for m in ('iou','unresolved')}
        summary['sizes'][str(side)]=result
    write_json(ROOT/'weizmann-ising-seeds-threshold-diagnostic-summary.json',summary)
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for col,side in enumerate((8,16)):
        for row,metric in enumerate(('iou','unresolved')):
            matrix=np.array([[summary['sizes'][str(side)]['corners'][threshold+'_K'+k]['two_max']['sample'][metric] for threshold in ('fixed75','q90')] for k in ('04','08')])
            ax=axes[row,col];im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis' if row==0 else 'magma')
            ax.set_xticks([0,1],['Fixed 75','Positive-difference q90']);ax.set_yticks([0,1],['K = 0.4','K = 0.8'])
            ax.set_title(f'{side}×{side}: '+('IoU' if row==0 else 'unresolved fraction'))
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{matrix[i,j]:.3f}',color='white',ha='center',va='center',fontsize=16)
            fig.colorbar(im,ax=ax,fraction=.045,pad=.04)
    fig.suptitle('Post-hoc threshold / temperature diagnostic — two seeds\nCubic enhancement; adaptive sensitivity; main selections unchanged',fontsize=12)
    fig.tight_layout();save(fig,'threshold-temperature-diagnostic')
    print(json.dumps({side:{k:v['two_max']['sample'] for k,v in s['corners'].items()} for side,s in summary['sizes'].items()},indent=2))


if __name__=='__main__':main()
