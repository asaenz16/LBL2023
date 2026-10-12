"""Aggregate only a complete benchmark; bootstrap by original source image."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quantum-segmentation-matplotlib')
from collections import defaultdict
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from weizmann_petz_data import ROOT,OUT,write_json,seed_for
from run_weizmann_petz import hashes
from weizmann_petz_metrics import segmentation


def estimate(rows,getter):
    by_source=defaultdict(list)
    for row in rows:by_source[row['case']['source_id']].append(float(getter(row)))
    values=np.array([np.mean(v) for _,v in sorted(by_source.items())])
    rng=np.random.default_rng(710932)
    boots=values[rng.integers(0,len(values),(3000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),images=len(values))


def aggregate(rows):
    keys=('iou','dice','accuracy','gamma_linf','resolved_fraction','correct_resolved_fraction',
          'resolved_error_fraction','primary_resolved_fraction','secondary_iou','seed_agreement')
    result=dict(cases=len(rows),sources=len(set(r['case']['source_id'] for r in rows)),
        gibbs=estimate(rows,lambda r:r['gibbs']['iou']),
        bath=estimate(rows,lambda r:r['records'][-1]['iou']),
        otsu=estimate(rows,lambda r:r['baselines']['otsu']['iou']),
        random_walker=estimate(rows,lambda r:r['baselines']['random_walker']['iou']),
        last={key:estimate(rows,lambda r,k=key:r['records'][-1][k]) for key in keys},
        all_ferromagnetic_fraction=float(np.mean([r['diagnostics']['all_ferromagnetic'] for r in rows])),
        negative_bond_fraction=float(np.mean([r['diagnostics']['negative_bond_fraction'] for r in rows])),
        frustrated_plaquette_fraction=float(np.mean([r['diagnostics']['frustrated_plaquette_fraction'] for r in rows])),
        boundary_antiferromagnetic_fraction=float(np.mean([r['diagnostics']['boundary_antiferromagnetic_fraction']
            for r in rows if r['diagnostics']['boundary_antiferromagnetic_fraction'] is not None]))
            if any(r['diagnostics']['boundary_antiferromagnetic_fraction'] is not None for r in rows) else None,
        interior_antiferromagnetic_fraction=float(np.mean([r['diagnostics']['interior_antiferromagnetic_fraction']
            for r in rows if r['diagnostics']['interior_antiferromagnetic_fraction'] is not None]))
            if any(r['diagnostics']['interior_antiferromagnetic_fraction'] is not None for r in rows) else None,
        mean_gibbs_numerical_resolution=float(np.mean([r['gibbs']['resolved_fraction'] for r in rows])),
        gibbs_iou_below_half=sum(r['gibbs']['iou']<.5 for r in rows),
        significant_bath_bias=sum(r['records'][-1]['gamma_linf']>r['simultaneous_radius'] for r in rows),
        mean_bias_lower_bound=float(np.mean([max(0,r['records'][-1]['gamma_linf']-r['simultaneous_radius']) for r in rows])),
        good_segmentation_far_from_gibbs=sum(r['records'][-1]['iou']>=.9 and
            r['records'][-1]['correct_resolved_fraction']>=.9 and
            r['records'][-1]['gamma_linf']-r['simultaneous_radius']>.02 for r in rows))
    result['history']=[]
    for i,record in enumerate(rows[0]['records']):
        result['history'].append(dict(sweep=record['sweep'],**{key:estimate(rows,lambda r,k=key:r['records'][i][k]) for key in keys}))
    if all(r['case']['family']=='synthetic' for r in rows):
        # The designed suite includes intensity-inverted pairs, not iid draws
        # from a population of natural images. Report descriptive means only.
        def clear_intervals(value):
            if isinstance(value,dict):
                if 'ci95' in value:value['ci95']=None
                for v in value.values():clear_intervals(v)
            elif isinstance(value,list):
                for v in value:clear_intervals(v)
        clear_intervals(result)
    return result


def group_name(row):
    c=row['case'];name=row['signature']['parameter_name']
    if c['family']=='natural-square':return f"{c['split']}/square-{c['shape'][0]}/{name}"
    if c['family']=='natural-strip':return f"{c['split']}/strip-{c['width']}-{c['category']}/{name}"
    return f"synthetic/{c['condition']}/{name}"


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=190,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def figures(rows,groups,calibration,long):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    categories=['strip-4-boundary','strip-16-boundary','strip-64-boundary','square-8','square-16']
    labels=['2 x 4 crop','2 x 16 crop','2 x 64 crop','8 x 8 image','16 x 16 image']
    methods=[('original','gibbs','Original Gibbs','#8db3ce'),('original','bath','Original bath','#235a82'),
        ('calibrated','gibbs','Recalibrated Gibbs','#f0bd91'),('calibrated','bath','Recalibrated bath','#b65329'),
        ('original','otsu','Seeded Otsu','#8a8584'),('original','random_walker','Random walker','#397652')]
    fig,ax=plt.subplots(figsize=(13,5.5),layout='constrained');x=np.arange(5);width=.13
    for i,(p,m,label,color) in enumerate(methods):
        vals=[groups[f'test/{c}/{p}'][m] for c in categories]
        mean=np.array([v['mean'] for v in vals]);ci=np.array([v['ci95'] for v in vals]).T
        ax.bar(x+(i-2.5)*width,mean,width,color=color,label=label,
               yerr=np.maximum(0,np.stack([mean-ci[0],ci[1]-mean])),capsize=2,error_kw={'lw':.7})
    ax.set_xticks(x,labels);ax.set_ylim(0,1.02);ax.set_ylabel('Mean foreground IoU')
    ax.legend(ncols=3,loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.15)
    ax.set_title('75 held-out Weizmann images: exact Gibbs, 500-sweep bath, and classical baselines\n'
                 'Bars: means; intervals: 95% bootstrap over original images')
    save(fig,'heldout-comparison')

    fig,axes=plt.subplots(2,3,figsize=(13.5,7.8),layout='constrained')
    colors=plt.get_cmap('viridis')(np.linspace(.05,.87,5))
    for r,param in enumerate(('original','calibrated')):
        for c,label,color in zip(categories,labels,colors):
            history=groups[f'test/{c}/{param}']['history'];t=[v['sweep'] for v in history]
            for j,key in enumerate(('gamma_linf','iou','correct_resolved_fraction')):
                axes[r,j].plot(t,[v[key]['mean'] for v in history],'.-',color=color,label=label)
        for j,ax in enumerate(axes[r]):
            ax.set_xscale('symlog',linthresh=1);ax.set_xlim(0,550)
            ax.set_xticks([0,1,10,100,500],['0','1','10','100','500']);ax.set_xlabel('Sweeps');ax.grid(alpha=.2)
        axes[r,0].set_yscale('log');axes[r,0].set_ylim(.001,.3)
        axes[r,0].set_ylabel(param+'\nCorrelation infinity-norm error')
        axes[r,1].set_ylabel('Foreground IoU');axes[r,1].set_ylim(-.02,1.02)
        axes[r,2].set_ylabel('Correct and resolved fraction');axes[r,2].set_ylim(-.02,1.02)
    axes[0,0].set_title('Distance from exact Gibbs correlations')
    axes[0,1].set_title('Segmentation against human masks')
    axes[0,2].set_title('Joint sign bounds across both seeds')
    axes[0,1].legend(fontsize=8,ncols=2)
    fig.suptitle('Held-out circuit-channel dynamics: 4,096 independent trajectories per case\n'
                 'Initial state: all +x; no quantum spin is pinned',fontsize=13)
    save(fig,'bath-convergence')

    conditions=['clean','noise20','noise60','contrast80','blur'];geometries=['disk','ring','islands','bridge','path']
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for i,param in enumerate(('original','calibrated')):
        for j,method in enumerate(('gibbs','bath')):
            values=np.zeros((len(geometries),len(conditions)))
            for a,geometry in enumerate(geometries):
                for b,condition in enumerate(conditions):
                    selected=[r for r in rows if r['case']['family']=='synthetic' and r['signature']['parameter_name']==param
                              and r['case']['category']==geometry and r['case']['condition']==condition]
                    values[a,b]=np.mean([r['gibbs']['iou'] if method=='gibbs' else r['records'][-1]['iou'] for r in selected])
            ax=axes[i,j];im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
            ax.set_xticks(range(5),conditions,rotation=20);ax.set_yticks(range(5),geometries)
            ax.set_title(f'{param}: {"exact Gibbs" if method=="gibbs" else "500-sweep bath"}')
            for a in range(5):
                for b in range(5):ax.text(b,a,f'{values[a,b]:.2f}',ha='center',va='center',color='white' if values[a,b]<.65 else 'black')
    fig.colorbar(im,ax=axes,label='Foreground IoU',shrink=.75)
    fig.suptitle('Controlled synthetic tests: geometry, contrast, noise, and blur\n'
                 'Each cell averages two sizes and two fixed replicates',fontsize=13)
    save(fig,'synthetic-tests')

    # Reproducible display selection independent of segmentation scores.
    selected_ids=sorted({r['case']['source_id'] for r in rows if r['case']['split']=='test'},key=lambda s:seed_for('figure:'+s))[:4]
    fig,axes=plt.subplots(4,6,figsize=(13.5,9),layout='constrained')
    for a,source in enumerate(selected_ids):
        row=next(r for r in rows if r['case']['case_id']==source+'--square-16' and r['signature']['parameter_name']=='calibrated')
        stem=f'{source}--square-16--calibrated--plus_x'
        data=np.load(OUT/'study'/f'{stem}.npz');truth=data['truth'];seed=int(data['seeds'][0]);fg=truth.ravel()[seed]
        gamma=data['gamma'][-1,0];pred=segmentation(gamma,fg).reshape(truth.shape)
        rgb=np.repeat(pred[:,:,None].astype(float),3,axis=2);rgb[(abs(gamma)<=row['simultaneous_radius']).reshape(truth.shape)]=[.94,.65,.12]
        panels=[data['image'],truth,segmentation(data['gamma_gibbs'][0],fg).reshape(truth.shape),pred,rgb,data['random_walker']]
        for b,panel in enumerate(panels):
            axes[a,b].imshow(panel,cmap='gray',vmin=0,vmax=255 if b==0 else 1,interpolation='nearest')
            axes[a,b].set_xticks([]);axes[a,b].set_yticks([])
        axes[a,0].set_ylabel(source,fontsize=8)
        axes[a,2].set_xlabel(f'IoU {row["gibbs"]["iou"]:.3f}')
        axes[a,3].set_xlabel(f'IoU {row["records"][-1]["iou"]:.3f}')
        axes[a,5].set_xlabel(f'IoU {row["baselines"]["random_walker"]["iou"]:.3f}')
    for ax,title in zip(axes[0],['Input','Human majority','Exact Gibbs','Bath: 500 sweeps','Gold = unresolved','Random walker']):ax.set_title(title,fontsize=10)
    fig.suptitle('16 x 16 held-out examples, using the frozen recalibrated parameters\n'
                 'Examples selected by source-name hash, without inspecting scores',fontsize=13)
    save(fig,'heldout-examples')

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for col,alpha in enumerate((.001,.01)):
        values=np.array([[next(r['mean_iou'] for r in calibration['scores'] if r['params']==dict(threshold=t,alpha=alpha,k=k))
                          for k in (.4,.8,1.2)] for t in (10.,25.,50.,75.,100.)])
        im=axes[col].imshow(values,vmin=0,vmax=.4,aspect='auto',cmap='viridis')
        axes[col].set_xticks(range(3),[.4,.8,1.2]);axes[col].set_yticks(range(5),[10,25,50,75,100])
        axes[col].set_xlabel('K');axes[col].set_ylabel('Fixed threshold');axes[col].set_title(f'alpha = {alpha}')
        for i in range(5):
            for j in range(3):axes[col].text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]<.25 else 'black')
    fig.colorbar(im,ax=axes,label='Mean exact-Gibbs foreground IoU')
    fig.suptitle('Calibration only: 25 original images, both square resolutions')
    save(fig,'calibration')

    case_ids=sorted(set(r['case']['case_id'] for r in long))
    fig,axes=plt.subplots(len(case_ids),2,figsize=(11,3*len(case_ids)),layout='constrained')
    for i,case_id in enumerate(case_ids):
        for initial,color in [('plus_x','#21678c'),('all_up','#b3572c')]:
            r=next(x for x in long if x['case']['case_id']==case_id and x['signature']['initial']==initial)
            t=[v['sweep'] for v in r['records']]
            axes[i,0].plot(t,[v['gamma_linf'] for v in r['records']],'o-',color=color,label=initial)
            axes[i,1].plot(t,[v['iou'] for v in r['records']],'o-',color=color,label=initial)
        axes[i,0].axhline(r['simultaneous_radius'],ls=':',color='gray',label='Shot bound')
        axes[i,1].axhline(r['gibbs']['iou'],ls='--',color='black',label='Gibbs IoU')
        axes[i,0].set_yscale('log');axes[i,0].set_ylim(.001,.6);axes[i,1].set_ylim(-.02,1.02)
        axes[i,0].set_title(case_id,fontsize=9)
        for j,ax in enumerate(axes[i]):
            ax.set_xscale('symlog',linthresh=20);ax.set_xlim(0,2100);ax.set_xlabel('Sweeps');ax.grid(alpha=.2)
            ax.set_ylabel('Correlation infinity-norm error' if j==0 else 'Foreground IoU')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Independent 2,000-sweep checks with 16,384 trajectories\n'
                 'Two calibration images and two prespecified synthetic cases; parameters frozen',fontsize=13)
    save(fig,'long-confirmation')


def main():
    data=json.loads((ROOT/'weizmann-petz-manifest.json').read_text())
    expected=sum(1 if c['family']=='natural-strip' and c['category']!='boundary' else 2 for c in data['cases'])
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'study').glob('*.json'))]
    long=[json.loads(p.read_text()) for p in sorted((OUT/'long').glob('*.json'))]
    if len(rows)!=expected or len(long)!=8:raise ValueError(f'Incomplete runs: primary {len(rows)}/{expected}, confirmation {len(long)}/8')
    for r in rows+long:
        if r['signature']['code_hashes']!=hashes():raise ValueError('Numerical source changed.')
    grouped=defaultdict(list)
    for r in rows:grouped[group_name(r)].append(r)
    groups={name:aggregate(items) for name,items in sorted(grouped.items())}
    calibration=json.loads((ROOT/'weizmann-petz-calibration.json').read_text())
    expected_ids={(c['case_id'],p) for c in data['cases'] for p in
        (['original'] if c['family']=='natural-strip' and c['category']!='boundary' else ['original','calibrated'])}
    actual_ids={(r['case']['case_id'],r['signature']['parameter_name']) for r in rows}
    if actual_ids!=expected_ids:raise ValueError('Unexpected or missing case/parameter combinations.')
    input_hashes={c['case_id']:hashlib.sha256((OUT/'cases'/f'{c["case_id"]}.npz').read_bytes()).hexdigest()
                  for c in data['cases']}
    for r in rows+long:
        if r['signature']['input_sha256']!=input_hashes[r['case']['case_id']]:raise ValueError('Numerical input changed.')
        if [v['sweep'] for v in r['records']]!=r['signature']['checkpoints']:raise ValueError('Incomplete sweep record.')
    calibration_sources=set(data['calibration_ids']);test_sources=set(data['test_ids'])
    if calibration_sources&test_sources:raise ValueError('Source split leakage.')
    for c in data['cases']:
        if c['family'].startswith('natural'):
            if c['source_id'] not in (calibration_sources if c['split']=='calibration' else test_sources):
                raise ValueError('Derivative in the wrong source split.')
    if calibration['code_hashes']!=hashes():raise ValueError('Calibration source changed.')
    paired={};baseline_comparisons={}
    for category in ['square-8','square-16','strip-4-boundary','strip-16-boundary','strip-64-boundary']:
        original={r['case']['source_id']:r for r in grouped[f'test/{category}/original']}
        selected=grouped[f'test/{category}/calibrated']
        paired[category]={method:estimate(selected,lambda r,m=method:r[m]['iou']-original[r['case']['source_id']][m]['iou'])
                          for method in ['gibbs']}
        paired[category]['bath']=estimate(selected,lambda r:r['records'][-1]['iou']-original[r['case']['source_id']]['records'][-1]['iou'])
        baseline_comparisons[category]={method:estimate(selected,lambda r,m=method:
            r['records'][-1]['iou']-r['baselines'][m]['iou']) for method in ('otsu','random_walker')}
    summary=dict(protocol=json.loads((ROOT/'weizmann-petz-protocol.json').read_text()),
        protocol_sha256=hashlib.sha256((ROOT/'weizmann-petz-protocol.json').read_bytes()).hexdigest(),
        primary_runs=len(rows),confirmation_runs=len(long),unique_cases=len(data['cases']),
        source_images=100,calibration_images=25,test_images=75,selected_parameters=calibration['selected'],
        groups=groups,paired_improvements=paired,paired_calibrated_bath_minus_baseline=baseline_comparisons,long=long,
        resources=json.loads((OUT/'resources.json').read_text()),
        matlab_validation=json.loads((ROOT/'weizmann-petz-matlab-validation.json').read_text()),
        code_hashes=hashes(),maximum_mask_decode_sensitivity=max(max(s['annotation_decode_64_vs_192_fraction']) for s in data['sources']))
    write_json(ROOT/'weizmann-petz-summary.json',summary)
    compact=[dict(case=r['case'],parameter_name=r['signature']['parameter_name'],gibbs=r['gibbs'],
                  final=r['records'][-1],diagnostics=r['diagnostics'],baselines=r['baselines']) for r in rows]
    write_json(ROOT/'weizmann-petz-case-results.json',compact)
    figures(rows,groups,calibration,long)
    for name,g in groups.items():
        if name.startswith('test/') and ('square' in name or 'boundary' in name):
            print(name,{k:round(g[k]['mean'],4) for k in ('gibbs','bath','otsu','random_walker')},
                  'linf',round(g['last']['gamma_linf']['mean'],4),'resolved',round(g['last']['resolved_fraction']['mean'],4),flush=True)
    print('Complete results, image-bootstrap intervals and figures saved.',flush=True)


if __name__=='__main__':main()
