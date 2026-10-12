"""Complete-study analysis of the Ising/XXZ Weizmann Gibbs comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quantum-segmentation-matplotlib')
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_weizmann_xxz import (ROOT,OUT,INPUT,load_stage,hashes,manifest,identifier,
    confirmation_specs,SETTINGS)
from run_weizmann_xxz_scan import scan_rules
from weizmann_petz_data import write_json,seed_for
from weizmann_petz_metrics import segmentation


def estimate(rows,getter):
    sources=defaultdict(list)
    for r in rows:sources[r['case']['source_id']].append(float(getter(r)))
    values=np.array([np.mean(v) for _,v in sorted(sources.items())])
    rng=np.random.default_rng(281071)
    samples=values[rng.integers(len(values),size=(3000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),sources=len(values))


def aggregate(rows):
    keys=['iou','iou_lower','iou_upper','dice','accuracy','resolved_fraction',
          'correct_resolved_fraction','wrong_resolved_fraction','seed_agreement','secondary_iou']
    result=dict(cases=len(rows),metrics={k:estimate(rows,lambda r,k=k:r['metrics'][k]) for k in keys},
        baselines={k:estimate(rows,lambda r,k=k:r['baselines'][k]['iou']) for k in ('otsu','random_walker')},
        passed=sum(r['sampling']['converged'] for r in rows),
        ess_below_200=sum(r['sampling']['minimum_variable_correlation_ess']<200 for r in rows),
        median_minimum_ess=float(np.median([r['sampling']['minimum_variable_correlation_ess'] for r in rows])),
        mean_intermediate_fraction=float(np.mean([r['parameters']['intermediate_fraction'] for r in rows])),
        mean_boundary_intermediate_fraction=float(np.mean([r['bonds']['boundary_intermediate_fraction'] for r in rows])),
        mean_interior_intermediate_fraction=float(np.mean([r['bonds']['interior_intermediate_fraction'] for r in rows])),
        mean_tau=float(np.mean([r['parameters']['threshold'] for r in rows])),
        tau_range=[min(r['parameters']['threshold'] for r in rows),max(r['parameters']['threshold'] for r in rows)],
        mean_alpha=float(np.mean([r['parameters']['alpha'] for r in rows])),
        g_range=[min(r['parameters']['g'] for r in rows),max(r['parameters']['g'] for r in rows)],
        b_range=[min(r['parameters']['b'] for r in rows),max(r['parameters']['b'] for r in rows)])
    exact=[r for r in rows if r['exact'] is not None]
    if exact:
        result['exact_iou']=estimate(exact,lambda r:r['exact']['iou'])
        result['mean_gamma_linf_vs_exact']=float(np.mean([r['sampling']['gamma_linf_vs_exact'] for r in exact]))
        result['maximum_gamma_linf_vs_exact']=max(r['sampling']['gamma_linf_vs_exact'] for r in exact)
        result['mean_exact_outside_interval_fraction']=float(np.mean([r['sampling']['exact_outside_interval_fraction'] for r in exact]))
    return result


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def array_for(row):
    return np.load(OUT/row['signature']['stage']/f'{row["case"]["case_id"]}--{row["rule_id"]}.npz')


def figures(summary,test,scan,small,controls,confirmed):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fams=['fixed50','fixed100','q75','edge_otsu'];labels=['Fixed 50','Fixed 100','Edge quantile 75%','Edge Otsu']
    colors={'Ising':'#24678b','XXZ':'#bd5934'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,side in enumerate((8,16)):
        for model in ('Ising','XXZ'):
            vals=[summary['test_groups'][f'{model}/{f}/{side}'] for f in fams]
            means=np.array([v['metrics']['iou']['mean'] for v in vals]);ci=np.array([v['metrics']['iou']['ci95'] for v in vals]).T
            x=np.arange(4)+(-.14 if model=='Ising' else .14)
            axes[i,0].errorbar(x,means,yerr=np.maximum(0,np.stack([means-ci[0],ci[1]-means])),
                fmt='o',capsize=3,color=colors[model],label=model)
            low=[v['metrics']['iou_lower']['mean'] for v in vals];high=[v['metrics']['iou_upper']['mean'] for v in vals]
            axes[i,1].vlines(x,low,high,color=colors[model],lw=5,alpha=.55)
            axes[i,1].scatter(x,means,color=colors[model],s=28,label=model)
        for j,ax in enumerate(axes[i]):
            ax.set_xticks(range(4),labels,rotation=15);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
            ax.set_ylabel(f'{side} x {side}: foreground IoU')
        baseline=summary['test_groups'][f'Ising/fixed50/{side}']['baselines']
        axes[i,0].axhline(baseline['otsu']['mean'],ls='--',color='#7d7d7d',label='Seeded Otsu')
        axes[i,0].axhline(baseline['random_walker']['mean'],ls=':',color='#477646',label='Random walker')
    axes[0,0].set_title('Mean scores and image-bootstrap intervals');axes[0,1].set_title('Range from unresolved MCMC signs')
    axes[0,0].legend(ncols=2,fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Ising versus XXZ: separately calibrated rules, 75 comparison images\n'
                 'Primary runs; all cases retained, including flagged chains',fontsize=13)
    save(fig,'model-rule-comparison')

    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    scan_ids=[identifier(r) for r in scan_rules()];x=np.arange(6)
    scanlabels=['g=0','g=0.25D','g=0.5D','g=D','g=2D','g=1\ncombined T']
    vals=[summary['scan'][rid] for rid in scan_ids]
    mean=np.array([v['metrics']['iou']['mean'] for v in vals])
    axes[0].plot(x,mean,'o-',color=colors['XXZ'])
    axes[0].fill_between(x,[v['metrics']['iou_lower']['mean'] for v in vals],
        [v['metrics']['iou_upper']['mean'] for v in vals],alpha=.2,color=colors['XXZ'],label='Unresolved-sign range')
    axes[0].set_ylabel('Foreground IoU');axes[0].legend(fontsize=8)
    axes[1].plot(x,[v['metrics']['resolved_fraction']['mean'] for v in vals],'o-',color=colors['Ising'])
    axes[1].set_ylabel('Resolved correlation signs')
    axes[2].plot(x,[v['mean_intermediate_fraction'] for v in vals],'o-',label='All bonds')
    axes[2].plot(x,[v['mean_boundary_intermediate_fraction'] for v in vals],'s--',label='Human boundary bonds')
    axes[2].set_ylabel('Fraction with |Delta| <= g');axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,scanlabels,rotation=30);ax.set_ylim(0,1);ax.grid(alpha=.2)
    fig.suptitle('Controlled exchange scan on 75 8 x 8 images\n'
                 'Selected XXZ image rule; D = median |Delta|; relative-g arms hold b fixed',fontsize=12)
    save(fig,'exchange-scan')

    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for ax,shape in zip(axes,('strip-4','square-8','square-16')):
        group=summary['controls'][shape]
        labels2=['Best Ising','XXZ','Remove XX/YY','Neutral band','Isolated-bond\nmatching']
        values=[group[k]['mean'] for k in ['selected_ising','selected_xxz','remove_exchange','neutral_band','isolated_bond']]
        ax.bar(range(5),values,color=[colors['Ising'],colors['XXZ'],'#69889c','#c09c57','#6e9371'])
        ax.set_xticks(range(5),labels2,rotation=40,ha='right');ax.set_ylim(0,1);ax.set_title(shape)
        for j,v in enumerate(values):ax.text(j,v+.015,f'{v:.3f}',ha='center',fontsize=8)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Mean foreground IoU')
    fig.suptitle('Mechanism controls: classical targets at the selected XXZ temperature\n'
                 '2 x 4 XXZ is exact; square XXZ uses primary QMC; best Ising uses its own calibrated rule',fontsize=12)
    save(fig,'mechanism-controls')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    delta=np.linspace(-1.5,1.5,600);b=8.;g=.7
    gamma_ising=np.tanh(b*delta/4)/4
    gamma_xxz=np.tanh(b*delta/4-.5*(np.logaddexp(b*g/2,-b*g/2)-np.log(2)))/4
    axes[0].axvspan(-g,g,color='#e9c46a',alpha=.25,label='Intermediate bond proxy')
    axes[0].plot(delta,gamma_ising,label='Ising');axes[0].plot(delta,gamma_xxz,label='XXZ')
    axes[0].axhline(0,color='gray',lw=.7);axes[0].set_xlabel('Delta');axes[0].set_ylabel('<ZZ>/4')
    axes[0].set_title('Exact isolated bond: b=8, g=0.7');axes[0].legend(fontsize=8)
    bg=np.linspace(.001,12,500);shift=2/bg*(np.logaddexp(bg/2,-bg/2)-np.log(2))
    axes[1].plot(bg,shift,color=colors['XXZ']);axes[1].axhline(1,color='gray',ls=':')
    axes[1].set_xlabel('b g');axes[1].set_ylabel('Delta at sign change / g');axes[1].set_ylim(0,1.05)
    axes[1].set_title('The ZZ sign threshold shifts with temperature')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('The intermediate XXZ regime does not act as a neutral-bond window',fontsize=13)
    save(fig,'isolated-bond-mechanism')

    selection=summary['selection'];best={m:r['rule_id'] for m,r in selection['overall'].items()}
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
    for ax,side in zip(axes,(8,16)):
        ir={r['case']['case_id']:r for r in test if r['rule_id']==best['Ising'] and r['case']['shape']==[side,side]}
        qr=[r for r in test if r['rule_id']==best['XXZ'] and r['case']['shape']==[side,side]]
        x=[ir[r['case']['case_id']]['exact']['iou'] for r in qr];y=[r['metrics']['iou'] for r in qr]
        color=[r['metrics']['resolved_fraction'] for r in qr]
        im=ax.scatter(x,y,c=color,vmin=0,vmax=1,cmap='viridis',s=35)
        ax.plot([0,1],[0,1],'k--',lw=.8);ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02)
        ax.set_xlabel('Selected Ising: exact Gibbs IoU');ax.set_ylabel('Selected XXZ: QMC IoU');ax.set_title(f'{side} x {side}')
    fig.colorbar(im,ax=axes,label='XXZ resolved fraction',shrink=.8)
    fig.suptitle('Paired image comparison; color exposes uncertain signs',fontsize=13)
    save(fig,'paired-images')

    sources=sorted(manifest()['test_ids'],key=lambda s:seed_for('xxz-display:'+s))[:4]
    fig,axes=plt.subplots(4,5,figsize=(11,9),layout='constrained')
    for i,source in enumerate(sources):
        cid=source+'--square-16';f=np.load(INPUT/'cases'/f'{cid}.npz');truth=f['truth']
        ir=next(r for r in test if r['case']['case_id']==cid and r['rule_id']==best['Ising'])
        qr=next(r for r in test if r['case']['case_id']==cid and r['rule_id']==best['XXZ'])
        ia,qa=array_for(ir),array_for(qr)
        pi=segmentation(ia['reference'][0],False).reshape(truth.shape)
        pq=segmentation(qa['gamma'][0],False).reshape(truth.shape)
        marked=np.repeat(pq[:,:,None].astype(float),3,axis=2)
        marked[(abs(qa['gamma'][0])<=qa['radius'][0]).reshape(truth.shape)]=[.94,.66,.12]
        panels=[f['image'],truth,pi,pq,marked]
        for j,panel in enumerate(panels):
            ax=axes[i,j];ax.imshow(panel,cmap='gray',vmin=0,vmax=255 if j==0 else 1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
        axes[i,0].set_ylabel(source,fontsize=8)
        axes[i,2].set_xlabel(f'IoU {ir["exact"]["iou"]:.3f}');axes[i,3].set_xlabel(f'IoU {qr["metrics"]["iou"]:.3f}')
    for ax,title in zip(axes[0],('Input','Human majority','Ising: exact Gibbs','XXZ: QMC','Gold = unresolved')):ax.set_title(title,fontsize=10)
    fig.suptitle('16 x 16 examples selected by source-ID hash, without inspecting scores',fontsize=12)
    save(fig,'segmentation-examples')


def main():
    selection=json.loads((ROOT/'weizmann-xxz-selection.json').read_text())
    stages={s:load_stage(s) for s in ('screen','refine','test','small','scan','confirm')}
    short=json.loads((ROOT/'weizmann-xxz-shortlist.json').read_text())
    expected={'screen':7200,'refine':25*len(short['shortlist']),'test':1200,'small':200,'scan':450,'confirm':len(confirmation_specs())}
    for stage,rows in stages.items():
        if len(rows)!=expected[stage]:raise ValueError(f'Incomplete {stage}: {len(rows)}/{expected[stage]}')
        for r in rows:
            if r['signature']['code_hashes']!=hashes():raise ValueError('Numerical source changed.')
    test=stages['test'];small=stages['small'];scan=stages['scan'];confirmed=stages['confirm']
    casehash={}
    for r in test+small+scan+confirmed:
        cid=r['case']['case_id']
        if cid not in casehash:casehash[cid]=hashlib.sha256((INPUT/'cases'/f'{cid}.npz').read_bytes()).hexdigest()
        if r['signature']['input_sha256']!=casehash[cid]:raise ValueError('Input changed.')
    grouped=defaultdict(list)
    for r in test:grouped[r['family']+'/'+str(r['case']['shape'][0])].append(r)
    groups={k:aggregate(v) for k,v in sorted(grouped.items())}
    scan_groups={identifier(rule):aggregate([r for r in scan if r['rule_id']==identifier(rule)]) for rule in scan_rules()}
    small_groups={m:aggregate([r for r in small if r['family'].startswith(m+'/') and r['case']['split']=='test']) for m in ('Ising','XXZ')}
    best={m:r['rule_id'] for m,r in selection['overall'].items()};paired={};confirmed_groups={}
    revised={(r['case']['case_id'],r['rule_id']):r for r in confirmed}
    for side in (8,16):
        ir={r['case']['case_id']:r for r in test if r['rule_id']==best['Ising'] and r['case']['shape']==[side,side]}
        qr=[r for r in test if r['rule_id']==best['XXZ'] and r['case']['shape']==[side,side]]
        paired[str(side)]=dict(
            sampled=estimate(qr,lambda r:r['metrics']['iou']-ir[r['case']['case_id']]['metrics']['iou']),
            versus_exact_ising=estimate(qr,lambda r:r['metrics']['iou']-ir[r['case']['case_id']]['exact']['iou']),
            conservative_gap_lower=estimate(qr,lambda r:r['metrics']['iou_lower']-ir[r['case']['case_id']]['exact']['iou']),
            conservative_gap_upper=estimate(qr,lambda r:r['metrics']['iou_upper']-ir[r['case']['case_id']]['exact']['iou']))
        for m,rid in best.items():
            rows=[revised.get((r['case']['case_id'],rid),r) for r in test if r['rule_id']==rid and r['case']['shape']==[side,side]]
            confirmed_groups[f'{m}/{side}']=aggregate(rows)
    control_file=json.loads((ROOT/'weizmann-xxz-controls.json').read_text());control_summary={}
    if len(control_file['rows'])!=225:raise ValueError('Incomplete classical controls.')
    for name,shape in [('strip-4',[2,4]),('square-8',[8,8]),('square-16',[16,16])]:
        rows=[r for r in control_file['rows'] if r['case']['shape']==shape]
        group={k:estimate(rows,lambda r,k=k:r['metrics'][k]['iou']) for k in rows[0]['metrics']}
        source=small if name=='strip-4' else test
        for m,rid in best.items():
            matching=[r for r in source if r['rule_id']==rid and r['case']['shape']==shape and r['case']['split']=='test']
            group['selected_'+m.lower()]=estimate(matching,lambda r:r['exact']['iou'] if r['exact'] is not None else r['metrics']['iou'])
        control_summary[name]=group
    summary=dict(protocol=json.loads((ROOT/'weizmann-xxz-protocol.json').read_text()),
        amendment=json.loads((ROOT/'weizmann-xxz-amendment.json').read_text()),selection=selection,
        counts={k:len(v) for k,v in stages.items()},test_groups=groups,scan=scan_groups,small=small_groups,
        paired=paired,controls=control_summary,overall_after_confirmation=confirmed_groups,code_hashes=hashes(),
        interval_note='Image bootstrap intervals are pointwise and conditional on estimated labels. MCMC sign ranges are approximate and require equilibration; Rhat does not prove convergence.')
    write_json(ROOT/'weizmann-xxz-summary.json',summary)
    write_json(ROOT/'weizmann-xxz-case-results.json',test+small+scan+confirmed)
    figures(summary,test,scan,small,control_file,confirmed)
    for name,group in groups.items():print(name,'IoU',round(group['metrics']['iou']['mean'],4),
        'range',[round(group['metrics'][key]['mean'],4) for key in ['iou_lower','iou_upper']],
        'resolved',round(group['metrics']['resolved_fraction']['mean'],4),'passed',group['passed'],flush=True)
    print('Paired:',json.dumps(paired),flush=True)
    print('All complete-stage comparisons and figures saved.',flush=True)


if __name__=='__main__':main()
