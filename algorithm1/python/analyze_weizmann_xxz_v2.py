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
from run_weizmann_xxz_v2 import (ROOT,OUT,INPUT,load_stage,hashes,manifest,identifier,
    confirmation_specs,SETTINGS,candidates,calibration_cases)
from weizmann_xxz_analysis_helpers import scan_rules
from weizmann_petz_data import write_json,seed_for
from weizmann_petz_metrics import segmentation


def sign_bound(row,upper=False):
    if not row['sampling']['converged']:return float(upper)
    return row['metrics']['iou_upper' if upper else 'iou_lower']


def paired_to_ising(quantum,ising):
    """Pair by source/case, using the exact Ising target when available."""
    lookup={r['case']['case_id']:r for r in ising}
    def reference(r):
        other=lookup[r['case']['case_id']]
        return other['exact']['iou'] if other.get('exact') is not None else other['metrics']['iou']
    return dict(versus_exact_ising=estimate(quantum,lambda r:r['metrics']['iou']-reference(r)),
        conservative_gap_lower=estimate(quantum,lambda r:sign_bound(r)-reference(r)),
        conservative_gap_upper=estimate(quantum,lambda r:sign_bound(r,True)-reference(r)))


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
        seed_complement_baseline=estimate(rows,lambda r:r['case']['foreground_fraction']*np.prod(r['case']['shape'])/(np.prod(r['case']['shape'])-1)),
        equilibration_aware_lower=estimate(rows,lambda r:r['metrics']['iou_lower'] if r['sampling']['converged'] else 0.),
        equilibration_aware_upper=estimate(rows,lambda r:r['metrics']['iou_upper'] if r['sampling']['converged'] else 1.),
        mean_magnetization_squared=float(np.mean([r['sampling']['magnetization_squared'] for r in rows])),
        mean_global_offdiagonal_gamma=float(np.mean([(np.prod(r['case']['shape'])*r['sampling']['magnetization_squared']-1)/(4*(np.prod(r['case']['shape'])-1)) for r in rows])),
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
            low=[v['equilibration_aware_lower']['mean'] for v in vals];high=[v['equilibration_aware_upper']['mean'] for v in vals]
            axes[i,1].vlines(x,low,high,color=colors[model],lw=5,alpha=.55)
            axes[i,1].scatter(x,means,color=colors[model],s=28,label=model)
        for j,ax in enumerate(axes[i]):
            ax.set_xticks(range(4),labels,rotation=15);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
            ax.set_ylabel(f'{side} x {side}: foreground IoU')
        baseline=summary['test_groups'][f'Ising/fixed50/{side}']['baselines']
        axes[i,0].axhline(baseline['otsu']['mean'],ls='--',color='#7d7d7d',label='Seeded Otsu')
        axes[i,0].axhline(baseline['random_walker']['mean'],ls=':',color='#477646',label='Random walker')
    axes[0,0].set_title('Mean scores and image-bootstrap intervals');axes[0,1].set_title('Unresolved signs; flagged cases allowed [0,1]')
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
    axes[0].fill_between(x,[v['equilibration_aware_lower']['mean'] for v in vals],
        [v['equilibration_aware_upper']['mean'] for v in vals],alpha=.2,color=colors['XXZ'],label='Sign/mixing uncertainty')
    axes[0].axhline(vals[0]['seed_complement_baseline']['mean'],ls=':',color='gray',label='Foreground except seed')
    axes[0].set_ylabel('Foreground IoU');axes[0].legend(fontsize=8)
    axes[1].plot(x,[v['metrics']['resolved_fraction']['mean'] for v in vals],'o-',color=colors['Ising'])
    axes[1].set_ylabel('Resolved correlation signs')
    axes[2].plot(x,[v['mean_intermediate_fraction'] for v in vals],'o-',label='All bonds')
    axes[2].plot(x,[v['mean_boundary_intermediate_fraction'] for v in vals],'s--',label='Human boundary bonds')
    axes[2].set_ylabel('Fraction with |Delta| <= g');axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,scanlabels,rotation=30);ax.set_ylim(0,1);ax.grid(alpha=.2)
        ax.axvline(4.5,color='gray',ls=':',lw=.8)
    fig.suptitle('Controlled exchange scan on 75 8 x 8 images\n'
                 'Selected XXZ image rule; D = median |Delta|; relative-g arms hold b fixed',fontsize=12)
    save(fig,'exchange-scan')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].plot(x,[v['mean_magnetization_squared'] for v in vals],'o-',color=colors['XXZ'])
    axes[0].set_ylabel('Mean <(sum Z / N)^2>')
    axes[1].plot(x,[v['mean_global_offdiagonal_gamma'] for v in vals],'o-',color=colors['Ising'])
    axes[1].axhline(0,color='gray',lw=.7);axes[1].axhline(-1/(4*63),ls=':',color='gray',label='Total Z = 0 limit')
    axes[1].set_ylabel('Mean Gamma over all distinct pixel pairs');axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,scanlabels,rotation=30);ax.grid(alpha=.2);ax.axvline(4.5,color='gray',ls=':',lw=.8)
    fig.suptitle('Magnetization diagnostic for the 8 x 8 exchange scan\n'
                 'All magnetization sectors are sampled; pair average inferred exactly from <(sum Z)^2>',fontsize=12)
    save(fig,'magnetization-diagnostic')

    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for ax,shape in zip(axes,('strip-4','square-8','square-16')):
        group=summary['controls'][shape]
        labels2=['Selected Ising','XXZ','Remove XX/YY','Neutral band','Isolated-bond\nmatching']
        values=[group[k]['mean'] for k in ['selected_ising','selected_xxz','remove_exchange','neutral_band','isolated_bond']]
        ax.bar(range(5),values,color=[colors['Ising'],colors['XXZ'],'#69889c','#c09c57','#6e9371'])
        ax.set_xticks(range(5),labels2,rotation=40,ha='right');ax.set_ylim(0,1);ax.set_title(shape)
        for j,v in enumerate(values):ax.text(j,v+.015,f'{v:.3f}',ha='center',fontsize=8)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Mean foreground IoU')
    fig.suptitle('Mechanism controls: classical targets at the selected XXZ temperature\n'
                 '2 x 4 XXZ is exact; square XXZ uses primary QMC; selected Ising uses its own calibrated rule',fontsize=12)
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


def supplementary_figures(summary):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    crop=summary['exact_crops'];rules=crop['strength_winners']
    x=np.arange(len(rules));values=[crop['test'][r['rule_id']] for r in rules]
    means=np.array([v['mean'] for v in values]);ci=np.array([v['ci95'] for v in values]).T
    axes[0].errorbar(x,means,yerr=[means-ci[0],ci[1]-means],fmt='o-',capsize=3,color='#825198')
    axes[0].set_xticks(x,[str(r['rule']['strength']) for r in rules])
    axes[0].set_xlabel('g / median |Delta|');axes[0].set_ylabel('Exact Gibbs foreground IoU')
    axes[0].set_title('Native 2 x 4 crops: separately tuned per g')
    axes[0].annotate('Calibration-selected XXZ',xy=(4,means[-1]),xytext=(2,.88),
        arrowprops={'arrowstyle':'->'},fontsize=9)
    for j,side in enumerate((8,16)):
        for model,color,offset in [('Ising','#24678b',-.13),('XXZ','#bd5934',.13)]:
            a=summary['raw_groups'][f'{model}/{side}'];v=a.get('exact_iou',a['metrics']['iou']);m=v['mean']
            axes[1].errorbar(j+offset,m,yerr=[[m-v['ci95'][0]],[v['ci95'][1]-m]],fmt='o',capsize=3,
                color=color,label=model if j==0 else None)
            if model=='XXZ':axes[1].vlines(j+offset+.055,a['equilibration_aware_lower']['mean'],
                a['equilibration_aware_upper']['mean'],color=color,alpha=.35,lw=5)
    axes[1].set_xticks([0,1],['8 x 8','16 x 16']);axes[1].set_ylabel('Foreground IoU')
    axes[1].set_title('Alternative calibration: maximize raw IoU');axes[1].legend()
    for ax in axes:ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Sensitivity to calibration domain and selection objective\n'
                 '75 sources; error bars = pointwise image bootstrap; pale ranges = XXZ sign/mixing uncertainty',fontsize=12)
    save(fig,'calibration-sensitivity')


def main():
    selection=json.loads((ROOT/'weizmann-xxz-v2-selection.json').read_text())
    stages={s:load_stage(s) for s in ('screen','refine','test','small','scan','confirm','raw')}
    short=json.loads((ROOT/'weizmann-xxz-v2-shortlist.json').read_text())
    expected={'screen':9000,'refine':25*len(short['shortlist']),'test':1200,'small':200,'scan':450,'confirm':len(confirmation_specs()),'raw':300}
    data=manifest();calibration=set(data['calibration_ids']);testing=set(data['test_ids'])
    if calibration & testing:raise ValueError('Source split overlaps.')
    raw_selection=json.loads((ROOT/'weizmann-xxz-v2-raw-selection.json').read_text())
    squares=[c for c in data['cases'] if c['split']=='test' and c['family']=='natural-square']
    crops=[c for c in data['cases'] if c['family']=='natural-strip' and c['width']==4 and c['category']=='boundary']
    def pairs(cases,rules):return {(c['case_id'],identifier(r)) for c in cases for r in rules}
    required={
        'screen':pairs(calibration_cases(),candidates()),
        'refine':pairs(calibration_cases(),[r['rule'] for r in short['shortlist']]),
        'test':pairs(squares,[r['rule'] for r in selection['winners'].values()]),
        'small':pairs(crops,[r['rule'] for r in selection['overall'].values()]),
        'scan':pairs([c for c in squares if c['shape']==[8,8]],scan_rules()),
        'confirm':{(c['case_id'],identifier(r)) for _,c,r in confirmation_specs()},
        'raw':pairs(squares,[r['rule'] for r in raw_selection.values()])}
    for stage,rows in stages.items():
        if len(rows)!=expected[stage]:raise ValueError(f'Incomplete {stage}: {len(rows)}/{expected[stage]}')
        keys={(r['case']['case_id'],r['rule_id']) for r in rows}
        if len(keys)!=len(rows):raise ValueError('Duplicate case-rule pair.')
        if keys!=required[stage]:raise ValueError('Incorrect case-rule coverage in '+stage)
        for r in rows:
            if r['signature']['code_hashes']!=hashes():raise ValueError('Numerical source changed.')
            if identifier(r['signature']['rule'])!=r['rule_id']:raise ValueError('Invalid rule ID.')
            if r['signature']['settings']!=SETTINGS['test' if stage=='raw' else stage]:raise ValueError('Incorrect sampling budget.')
            allowed=calibration if stage in ('screen','refine') else (calibration|testing if stage=='small' else testing)
            if r['case']['source_id'] not in allowed:raise ValueError('Wrong source split.')
    test=stages['test'];small=stages['small'];scan=stages['scan'];confirmed=stages['confirm']
    casehash={}
    for r in test+small+scan+confirmed+stages['raw']:
        cid=r['case']['case_id']
        if cid not in casehash:casehash[cid]=hashlib.sha256((INPUT/'cases'/f'{cid}.npz').read_bytes()).hexdigest()
        if r['signature']['input_sha256']!=casehash[cid]:raise ValueError('Input changed.')
    grouped=defaultdict(list)
    for r in test:grouped[r['family']+'/'+str(r['case']['shape'][0])].append(r)
    groups={k:aggregate(v) for k,v in sorted(grouped.items())}
    scan_groups={identifier(rule):aggregate([r for r in scan if r['rule_id']==identifier(rule)]) for rule in scan_rules()}
    scan_ising=[r for r in scan if r['rule_id']==identifier(scan_rules()[0])]
    scan_paired={identifier(rule):paired_to_ising([r for r in scan if r['rule_id']==identifier(rule)],scan_ising) for rule in scan_rules()[1:]}
    small_groups={m:aggregate([r for r in small if r['family'].startswith(m+'/') and r['case']['split']=='test']) for m in ('Ising','XXZ')}
    best={m:r['rule_id'] for m,r in selection['overall'].items()};paired={};confirmed_groups={}
    matched={};matched_confirmed={};confirmed_paired={}
    revised={(r['case']['case_id'],r['rule_id']):r for r in confirmed}
    for side in (8,16):
        ir={r['case']['case_id']:r for r in test if r['rule_id']==best['Ising'] and r['case']['shape']==[side,side]}
        qr=[r for r in test if r['rule_id']==best['XXZ'] and r['case']['shape']==[side,side]]
        paired[str(side)]=paired_to_ising(qr,list(ir.values()))
        matched[str(side)]=paired_to_ising(
            [r for r in test if r['family']=='XXZ/edge_otsu' and r['case']['shape']==[side,side]],
            [r for r in test if r['family']=='Ising/edge_otsu' and r['case']['shape']==[side,side]])
        confirm_rows={}
        for m,rid in best.items():
            rows=[revised.get((r['case']['case_id'],rid),r) for r in test if r['rule_id']==rid and r['case']['shape']==[side,side]]
            confirmed_groups[f'{m}/{side}']=aggregate(rows)
            confirm_rows[m]=rows
        confirmed_paired[str(side)]=paired_to_ising(confirm_rows['XXZ'],confirm_rows['Ising'])
        matched_confirmed[str(side)]=paired_to_ising(confirm_rows['XXZ'],
            [r for r in test if r['family']=='Ising/edge_otsu' and r['case']['shape']==[side,side]])
    raw_groups={};raw_paired={}
    for side in (8,16):
        raw_by_model={}
        for model,r in raw_selection.items():
            rows=[x for x in stages['raw'] if x['rule_id']==r['rule_id'] and x['case']['shape']==[side,side]]
            if len(rows)!=75:raise ValueError('Incomplete raw-selected comparison.')
            raw_by_model[model]=rows;raw_groups[f'{model}/{side}']=aggregate(rows)
        raw_paired[str(side)]=paired_to_ising(raw_by_model['XXZ'],raw_by_model['Ising'])
    crop=json.loads((ROOT/'weizmann-xxz-v2-exact-crops.json').read_text())
    if crop['code_hashes']!=hashes() or len(crop['test'])!=375 or crop['calibration_evaluations']!=9000:
        raise ValueError('Incomplete or changed exact crop comparison.')
    crop_groups={r['rule_id']:estimate([x for x in crop['test'] if x['rule_id']==r['rule_id']],lambda x:x['metrics']['iou']) for r in crop['strength_winners']}
    crop_i={r['case']['case_id']:r for r in crop['test'] if r['rule_id']==crop['selection']['Ising']['rule_id']}
    crop_q=[r for r in crop['test'] if r['rule_id']==crop['selection']['XXZ']['rule_id']]
    crop_difference=estimate(crop_q,lambda r:r['metrics']['iou']-crop_i[r['case']['case_id']]['metrics']['iou'])
    crop_strength_paired={r['rule_id']:estimate([x for x in crop['test'] if x['rule_id']==r['rule_id']],
        lambda x:x['metrics']['iou']-crop_i[x['case']['case_id']]['metrics']['iou']) for r in crop['strength_winners']}
    control_file=json.loads((ROOT/'weizmann-xxz-v2-controls.json').read_text());control_summary={}
    if len(control_file['rows'])!=225:raise ValueError('Incomplete classical controls.')
    if control_file['code_hashes']!=hashes():raise ValueError('Classical control source changed.')
    if control_file['control_code_sha256']!=hashlib.sha256((ROOT/'python'/'weizmann_xxz_controls_v2.py').read_bytes()).hexdigest():
        raise ValueError('Classical control script changed.')
    if crop['supplement_script_sha256']!=hashlib.sha256((ROOT/'python'/'weizmann_xxz_supplements.py').read_bytes()).hexdigest():
        raise ValueError('Supplement source changed.')
    for name,shape in [('strip-4',[2,4]),('square-8',[8,8]),('square-16',[16,16])]:
        rows=[r for r in control_file['rows'] if r['case']['shape']==shape]
        group={k:estimate(rows,lambda r,k=k:r['metrics'][k]['iou']) for k in rows[0]['metrics']}
        source=small if name=='strip-4' else test
        for m,rid in best.items():
            matching=[r for r in source if r['rule_id']==rid and r['case']['shape']==shape and r['case']['split']=='test']
            group['selected_'+m.lower()]=estimate(matching,lambda r:r['exact']['iou'] if r['exact'] is not None else r['metrics']['iou'])
        control_summary[name]=group
    summary=dict(protocol=json.loads((ROOT/'weizmann-xxz-v2-protocol.json').read_text()),
        selection=selection,
        counts={k:len(v) for k,v in stages.items()},test_groups=groups,scan=scan_groups,scan_paired=scan_paired,small=small_groups,
        paired=paired,matched_edge_otsu=matched,controls=control_summary,overall_after_confirmation=confirmed_groups,
        paired_after_confirmation=confirmed_paired,matched_edge_otsu_after_confirmation=matched_confirmed,
        raw_selection=raw_selection,raw_groups=raw_groups,raw_paired=raw_paired,
        exact_crops=dict(selection=crop['selection'],strength_winners=crop['strength_winners'],test=crop_groups,paired=crop_difference,strength_paired=crop_strength_paired),
        code_hashes=hashes(),
        interval_note='Image bootstrap intervals are pointwise and conditional on estimated labels. MCMC sign ranges are approximate and require equilibration; Rhat does not prove convergence.')
    write_json(ROOT/'weizmann-xxz-v2-summary.json',summary)
    write_json(ROOT/'weizmann-xxz-v2-case-results.json',test+small+scan+confirmed+stages['raw'])
    figures(summary,test,scan,small,control_file,confirmed)
    supplementary_figures(summary)
    for name,group in groups.items():print(name,'IoU',round(group['metrics']['iou']['mean'],4),
        'range',[round(group['metrics'][key]['mean'],4) for key in ['iou_lower','iou_upper']],
        'resolved',round(group['metrics']['resolved_fraction']['mean'],4),'passed',group['passed'],flush=True)
    print('Paired:',json.dumps(paired),flush=True)
    print('All complete-stage comparisons and figures saved.',flush=True)


if __name__=='__main__':main()
