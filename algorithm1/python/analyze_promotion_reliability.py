"""Paired source-image summaries, reliability denominators, and ablation figures."""
import os
for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/promotion-reliability-mpl')
import json
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_promotion_reliability import ROOT,OUT,INPUT,cases,grouped,identifier,hashes,selection,METRICS,REASONS,pid
from weizmann_petz_data import write_json,seed_for

DISPLAY=('iou','unresolved','mixing_unresolved','mixing_lower','wrong_resolved','bpm_iou_r1','rounds','promoted_count','promoted_wrong','ever_promoted','ever_wrong','retired_count','retired_wrong','wrong_weight','promoted_weight','weight_reductions','converged')
COMPARE=('iou','unresolved','mixing_unresolved','bpm_iou_r1','rounds','promoted_count','promoted_wrong')


def estimate(x,key):
    x=np.asarray(x,float);ix=np.random.default_rng(seed_for('promotion-bootstrap:'+key)).integers(len(x),size=(5000,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[ix].mean(1),[.025,.975]).tolist())


def ratio(num,den,key):
    a=np.asarray(num,float);b=np.asarray(den,float)
    if not b.sum():return dict(mean=None,ci95=None)
    ix=np.random.default_rng(seed_for('promotion-ratio:'+key)).integers(len(a),size=(5000,len(a)));ds=b[ix].sum(1);ns=a[ix].sum(1)
    return dict(mean=float(a.sum()/b.sum()),ci95=np.quantile(ns[ds>0]/ds[ds>0],[.025,.975]).tolist())


def compact(case,anchor,profile,table,pi,confirmed):
    traces=[t for t in profile['traces'] if t['policy_index']==pi];a=table[pi];metrics=dict(zip(METRICS,a.mean(0).tolist()))
    controls={name:{m:float(np.mean([t['controls'][name][m] for t in traces])) for m in ('iou','unresolved','mixing_unresolved','bpm_iou_r1')} for name in traces[0]['controls']}
    trajectory=[]
    for t in range(10):
        rows=[tr['history'][min(t,len(tr['history'])-1)]['metrics'] for tr in traces]
        trajectory.append({m:float(np.mean([r[m] for r in rows])) for m in ('iou','unresolved','promoted_wrong','promoted_count','wrong_weight','retired_wrong')})
    return dict(case=case,anchor=anchor,policy=profile['policies'][pi],metrics=metrics,controls=controls,trajectory=trajectory,
        no_promotion=float(np.mean(a[:,METRICS.index('ever_promoted')]==0)),failed=bool(np.any(a[:,METRICS.index('converged')]<1)),
        low_ess=bool(np.any(a[:,METRICS.index('minimum_ess')]<100)),confirmed=confirmed,
        stops={reason:sum(t['stop']==reason for t in traces)/len(traces) for reason in REASONS},
        random_iou_sd=float(a[:,0].std(ddof=1)),predictions=[t['prediction'] for t in traces])


def summarize(rows,key):
    metrics={m:estimate([r['metrics'][m] for r in rows],key+m) for m in DISPLAY}
    ratios={}
    for name,num,den in [('active_error','promoted_wrong','promoted_count'),('ever_error','ever_wrong','ever_promoted'),('weighted_error','wrong_weight','promoted_weight'),('retirement_precision','retired_wrong','retired_count'),('wrong_seed_retirement_recall','retired_wrong','ever_wrong')]:
        ratios[name]=ratio([r['metrics'][num] for r in rows],[r['metrics'][den] for r in rows],key+name)
    return dict(sources=len(rows),metrics=metrics,reliability=ratios,
        no_promotion=float(np.mean([r['no_promotion'] for r in rows])),failed_cases=sum(r['failed'] for r in rows),low_ess_cases=sum(r['low_ess'] for r in rows),confirmed_cases=sum(r['confirmed'] for r in rows),
        controls={c:{m:estimate([r['controls'][c][m] for r in rows],key+c+m) for m in ('iou','unresolved','mixing_unresolved','bpm_iou_r1')} for c in rows[0]['controls']},
        versus_pooled={m:estimate([r['metrics'][m]-r['controls']['pooled_one_seed'][m] for r in rows],key+'pooled'+m) for m in ('iou','unresolved','mixing_unresolved','bpm_iou_r1')},
        stops={reason:float(np.mean([r['stops'][reason] for r in rows])) for reason in REASONS},
        trajectory=[{m:float(np.mean([r['trajectory'][t][m] for r in rows])) for m in rows[0]['trajectory'][t]} for t in range(10)])


def difference(a,b,key):
    lookup={r['case']['case_id']:r for r in b}
    out={m:estimate([r['metrics'][m]-lookup[r['case']['case_id']]['metrics'][m] for r in a],key+m) for m in COMPARE}
    changes=[]
    for r in a:
        other=lookup[r['case']['case_id']];n=int(np.prod(r['case']['shape']))
        x=np.unpackbits(np.array(r['predictions'],np.uint8),axis=1,count=n);y=np.unpackbits(np.array(other['predictions'],np.uint8),axis=1,count=n)
        changes.append(float((x!=y).mean()))
    out['pixel_disagreement']=estimate(changes,key+'pixel_disagreement')
    # Ratio difference keeps promotion counts in the denominator; no-promotion
    # images contribute zero counts, not a fictitious perfect-precision score.
    for label,num,den in [('active_error','promoted_wrong','promoted_count'),('weighted_error','wrong_weight','promoted_weight')]:
        an=np.array([r['metrics'][num] for r in a]);ad=np.array([r['metrics'][den] for r in a]);bn=np.array([lookup[r['case']['case_id']]['metrics'][num] for r in a]);bd=np.array([lookup[r['case']['case_id']]['metrics'][den] for r in a])
        if min(ad.sum(),bd.sum())>0:
            ix=np.random.default_rng(seed_for(key+label)).integers(len(a),size=(5000,len(a)));aa=ad[ix].sum(1);bb=bd[ix].sum(1);ok=(aa>0)&(bb>0)
            d=an[ix].sum(1)[ok]/aa[ok]-bn[ix].sum(1)[ok]/bb[ok]
            out[label]=dict(mean=float(an.sum()/ad.sum()-bn.sum()/bd.sum()),ci95=np.quantile(d,[.025,.975]).tolist())
        else:out[label]=dict(mean=None,ci95=None)
    return out


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(folder/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def plot(summary):
    colors=['#236b8e','#58a1ae','#e6943a','#ad563e','#8169a9','#56974a','#292929']
    families=['historical','spatial/mixing_lower','agreement/mixing_lower','remove/mixing_lower','weight/mixing_lower','combined/mixing_lower','any/mixing_lower']
    labels=['Original','Spatial','Agreement','Removal','Weights','Combined','Selected']
    fig,axes=plt.subplots(4,3,figsize=(13,13),layout='constrained')
    for row,anchor in enumerate([k for k in selection()['winners']]):
        for col,metric in enumerate(('iou','unresolved','active_error')):
            ax=axes[row,col]
            for x,(family,color) in enumerate(zip(families,colors)):
                s=summary[f'{anchor}|{family}|16'];v=s['reliability'][metric] if metric=='active_error' else s['metrics'][metric]
                if v['mean'] is None:continue
                lo,hi=v['ci95'];y=v['mean'];ax.bar(x,y,color=color,yerr=[[max(0,y-lo)],[max(0,hi-y)]],capsize=2)
            ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right',fontsize=8);ax.set_ylim(0,1 if col==0 else .2 if col==1 else .65)
            ax.set_title(anchor.replace('/mixing_lower','').replace('/',' · ')+' | '+{'iou':'IoU','unresolved':'Unresolved','active_error':'Wrong active seeds'}[metric],fontsize=10)
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('Promotion reliability on 75 comparison images at 16×16\nRules chosen on 25 calibration sources; pointwise source-bootstrap intervals')
    save(fig,'ablation-16')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,anchor in zip(axes.ravel(),selection()['winners']):
        for family,color in zip(families,colors):
            s=summary[f'{anchor}|{family}|16'];ax.scatter(s['metrics']['unresolved']['mean'],s['metrics']['iou']['mean'],color=color,s=65,label=labels[families.index(family)])
        ax.set_title(anchor.replace('/mixing_lower',''));ax.set_xlabel('Unresolved fraction');ax.set_ylabel('IoU');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Accuracy–resolution tradeoff (16×16)');save(fig,'tradeoff-16')


def plot_paired(result):
    families=('spatial','agreement','remove','weight','combined','any');names=('Spatial','Agreement','Removal','Weights','Combined','Selected')
    fig,axes=plt.subplots(4,2,figsize=(11,11),layout='constrained')
    colors=('#58a1ae','#e6943a','#ad563e','#8169a9','#56974a','#292929')
    for row,anchor in enumerate(result['selection']):
        for col,metric in enumerate(('iou','active_error')):
            ax=axes[row,col]
            for y,(family,color) in enumerate(zip(families,colors)):
                d=result['paired'][f'{anchor}|{family}/mixing_lower|16']['matched'][metric]
                if d['mean'] is None:continue
                value=100*d['mean'];lo,hi=np.array(d['ci95'])*100
                ax.errorbar(value,y,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',color=color,capsize=3)
            ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(6),names);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
            ax.set_title(anchor.replace('/mixing_lower',''),fontsize=10)
            ax.set_xlabel(('Δ IoU' if col==0 else 'Δ wrong-seed fraction')+' (percentage points)')
            ax.set_xlim((-1.6,.4) if col==0 else (-2.5,9.))
    fig.suptitle('Matched-control effects at 16×16 (75 source images)\nSame cutoffs, cap and stopping; pointwise paired 95% intervals')
    save(fig,'matched-effects-16')


def calibration_diagnostics():
    j=json.loads((OUT/'calibration-scores.json').read_text());metrics=j['metrics'];result={}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'keep':'#555555','remove1':'#bc6c25','remove2':'#dda15e','soft_relative':'#457b9d','soft_absolute':'#7b2cbf'}
    for ax,(anchor,values) in zip(axes.ravel(),j['means'].items()):
        a=np.asarray(values);count=a[:,metrics.index('promoted_count')];wrong=a[:,metrics.index('promoted_wrong')];error=np.divide(wrong,count,out=np.full_like(wrong,np.nan),where=count>0)
        retired=a[:,metrics.index('retired_count')];removal=np.array([p['maintenance'].startswith('remove') for p in j['policies']])
        result[anchor]=dict(removal_policies=int(removal.sum()),removal_policies_with_any_retirement=int(np.sum(removal&(retired>0))),maximum_retired_per_image=float(retired.max()),
            policies_with_weight_updates=int(np.sum(a[:,metrics.index('weight_reductions')]>0)))
        for kind,color in colors.items():
            ix=np.array([p['maintenance']==kind for p in j['policies']]);ax.scatter(error[ix],a[ix,metrics.index('mixing_lower')],s=8,alpha=.3,color=color,label=kind)
        ax.set_title(anchor.replace('/mixing_lower',''));ax.set_xlabel('Wrong active seed fraction');ax.set_ylabel('Mixing-aware lower IoU');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Full calibration grid: seed correctness and segmentation quality are distinct\nOverlapping points include different caps and stopping rules; 25 source images')
    save(fig,'calibration-reliability');write_json(OUT/'calibration-diagnostics.json',result)
    return result


def main():
    groups=defaultdict(list);records=[];counts=Counter()
    for c in cases('test'):
        for group in grouped('test'):
            stem=f'{c["case_id"]}--{identifier(group["rule"])}';p=OUT/'confirm'/f'{stem}.json';confirmed=p.exists()
            if not confirmed:p=OUT/'test'/f'{stem}.json'
            row=json.loads(p.read_text());assert row['signature']['code_hashes']==hashes();arrays=np.load(p.with_suffix('.npz'));counts[row['signature']['stage']]+=1
            for anchor,profile in row['profiles'].items():
                table=arrays[f'metrics{profile["array_index"]}'];local={}
                for label,pi in profile['labels'].items():
                    if pi not in local:local[pi]=compact(c,anchor,profile,table,pi,confirmed);records.append(local[pi])
                    groups[f'{anchor}|{label}|{c["shape"][0]}'].append(local[pi])
    summary={k:summarize(v,k) for k,v in groups.items()};diff={}
    for k,rows in groups.items():
        anchor,label,size=k.split('|')
        if label.endswith('/matched') or label=='historical':continue
        matched=f'{anchor}|{label}/matched|{size}'
        if matched in groups:diff[k]=dict(matched=difference(rows,groups[matched],k+'matched'),historical=difference(rows,groups[f'{anchor}|historical|{size}'],k+'historical'))
    output=dict(status='completed',execution_amendment=json.loads((ROOT/'promotion-reliability-execution-amendment.json').read_text()),code_hashes=hashes(),selection=selection()['winners'],groups=summary,paired=diff,
        calibration_diagnostics=calibration_diagnostics(),sources_per_size=75,calibration_sources=25,calibration_endpoints=1350000,calibration_policies=9000,comparison_banks=750,confirmation_banks=counts['confirm'],
        interpretation='Reused development split; finite policy grid, not continuous exhaustive search. Pointwise paired source-bootstrap intervals, no multiplicity correction. Nominal MCMC intervals; mixing failures retained. Promotion replicates are not independent images.')
    write_json(ROOT/'promotion-reliability-summary.json',output);write_json(OUT/'case-summary.json',records);plot(summary);plot_paired(output)
    for key,v in summary.items():
        if '|any/mixing_lower|16' in key:print(key,{m:round(v['metrics'][m]['mean'],5) for m in ('iou','unresolved','promoted_count','promoted_wrong','retired_wrong')},v['reliability'],flush=True)


if __name__=='__main__':main()
