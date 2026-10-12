"""Longer independent checks and binary-input controls at the frozen image rule.

Also exports a native-image result in MATLAB-compatible form. Confirmation
does not retune the rule. Classes are fixed in advance by stride four.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import savemat
from analyze_emnist_ising import choose_grid_rule,continuous_rule,bootstrap,save
from run_emnist_ising import ROOT,job,choose_seeds
from ising_monte_carlo import signed_image


def confirm(args):
    root=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
    rule=json.loads((root/'frozen-rule.json').read_text())
    cases=json.loads((root/'manifest.json').read_text())['cases']
    images=np.load(root/'subset.npz')['images'];specs=[]
    point_mode=args.continuous or args.fixed_reference
    rng_offset=213000019 if args.fixed_reference else 93000001 if args.continuous else 173000009
    protocol=dict(classes=args.classes,split='test',burn=args.burn,samples=args.samples,chains=4,
        continuous=args.continuous,
        rng_offset=rng_offset,rule=rule['adaptive'],binary_control_classes=[0,12,24,36])
    if args.fixed_reference:protocol['fixed_reference']=rule['fixed']
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Confirmation output already has a different protocol.')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    for ci,(case,image) in enumerate(zip(cases,images)):
        if case['split']!='test' or case['class_id'] not in args.classes:continue
        a,b,_=choose_grid_rule(image,rule['adaptive']['k'],rule['adaptive']['alpha_coefficient'],rule['available_pairs'])
        options=dict(burn=args.burn,samples=args.samples,chains=4,block_size=128,rng_seed=rng_offset+1000*ci)
        if point_mode:
            if args.fixed_reference:
                a,b=rule['fixed']['alpha'],rule['fixed']['b'];suffix='-fixed-reference'
            else:
                a,b,_=continuous_rule(image,rule['adaptive']['alpha_coefficient'],rule['adaptive']['k'])
                suffix='-continuous'
            case=dict(case,name=case['name']+suffix)
            specs.append((case,image,a,[b],options,out/'samples'/f'{case["name"]}-a{a:g}.npz'))
        else:
            cold=max(y for x,y in rule['available_pairs'] if x==a)
            specs.append((case,image,a,sorted(set([b,cold])),options,out/'samples'/f'{case["name"]}-a{a:g}.npz'))
        if not point_mode and case['class_id'] in [0,12,24,36]:
            binary=(image>=128).astype(np.uint8)*255
            control=dict(case,name=case['name']+'-binary',image_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),
                         seeds=choose_seeds(binary).tolist())
            specs.append((control,binary,a,[b],options,out/'samples'/f'{control["name"]}-a{a:g}.npz'))
    start=time.perf_counter();metas=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,spec) for spec in specs]
        for i,f in enumerate(as_completed(futures),1):
            m=f.result();metas.append(m)
            print(json.dumps(dict(done=i,total=len(specs),elapsed=round(time.perf_counter()-start,1),
                name=m['case']['name'],worst_rhat=max(r['rhat_observable_max'] for r in m['records']))),flush=True)
    comparison=[];controls=[]
    if point_mode:
        records=[r for meta in metas for r in meta['records']]
        summary=dict(protocol=protocol,records=records,aggregate={key:bootstrap([r[key] for r in records])
            for key in ['iou','balanced_accuracy','resolved_fraction','confident_wrong_fraction',
                        'iou_threshold96','iou_threshold160','ink_seed_iou']},
            rhat_max=max(r['rhat_observable_max'] for r in records),
            ess_mean_gamma_min=min(r['ess_mean_gamma'] for r in records),seconds=time.perf_counter()-start)
        tag='fixed-reference' if args.fixed_reference else 'continuous'
        (ROOT/f'emnist-ising-{tag}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2));return
    for meta in sorted(metas,key=lambda m:m['case']['name']):
        case=meta['case'];a=meta['signature']['alpha'];name=case['name']
        if name.endswith('-binary'):
            controls+=meta['records'];continue
        image=images[next(i for i,c in enumerate(cases) if c['name']==name)]
        longer=np.load(out/'samples'/f'{name}-a{a:g}.npz')
        selected_b=choose_grid_rule(image,rule['adaptive']['k'],rule['adaptive']['alpha_coefficient'],rule['available_pairs'])[1]
        for r in meta['records']:
            b=r['b'];path=root/'samples'/f'{name}-a{a:g}.npz'
            if not path.exists() or b not in np.load(path)['bs']:
                path=Path(rule['extension'])/'samples'/f'{name}-a{a:g}.npz'
            original=np.load(path)
            i=int(np.flatnonzero(original['bs']==b)[0]);j=int(np.flatnonzero(longer['bs']==b)[0])
            diff=abs(longer['gamma'][j,0]-original['gamma'][i,0])
            se=np.hypot(longer['se'][j,0],original['se'][i,0])
            comparison.append(dict(**r,is_selected_rule=b==selected_b,
                gamma_linf_change=float(diff.max()),gamma_rms_change=float(np.sqrt(np.mean(diff**2))),
                fraction_changes_over3_combined_se=float(np.mean(diff>np.maximum(3*se,1e-12))),
                sign_change_fraction=float(np.mean((longer['gamma'][j,0]>0)!=(original['gamma'][i,0]>0))),
                min_swap=meta['swap_min'],round_trips=meta['round_trips']))
        if case['class_id']==0:
            model=signed_image(image,a)
            savemat(out/'emnist-gibbs-reference.mat',dict(image=image,alpha=a,edges_zero_based=model.edges,
                delta=model.delta,beta_effective=longer['bs'],beta_paper=longer['beta_paper'],
                seeds_zero_based=longer['seeds'],gamma_seed=longer['gamma'],gamma_se=longer['se']))
    summary=dict(protocol=protocol,comparisons=comparison,binary_controls=controls,
        seconds=time.perf_counter()-start)
    (ROOT/'emnist-ising-confirmation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_checks(out,summary)
    print(json.dumps(summary,indent=2))


def plot_checks(out,summary):
    import matplotlib.pyplot as plt
    chosen=[r for r in summary['comparisons'] if r['is_selected_rule']]
    # Display a fixed case and the case with largest independent-run discrepancy.
    examples=[chosen[0],max(chosen,key=lambda r:r['gamma_linf_change'])]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,record in enumerate(examples):
        data=np.load(out/'samples'/f"{record['name']}-a{record['alpha']:g}.npz")
        i=int(np.flatnonzero(data['bs']==record['b'])[0]);blocks=data['blocks'][:,:,i,0]
        final=blocks.mean(axis=(0,1));time_axis=np.arange(1,blocks.shape[1]+1)*int(data['block_size'])
        for c in range(len(blocks)):
            running=np.cumsum(blocks[c],axis=0)/np.arange(1,blocks.shape[1]+1)[:,None]
            axes[row,0].plot(time_axis,np.max(abs(running-final),axis=1),label=f'Chain {c+1}')
            axes[row,1].plot(time_axis,blocks[c].mean(axis=1),alpha=.7,label=f'Chain {c+1}')
        axes[row,0].set_ylabel(r'$\|\widehat\Gamma_{1:t}-\widehat\Gamma_{final}\|_\infty$')
        axes[row,1].set_ylabel('Mean seed correlation Γ')
        for ax in axes[row]:ax.set_xlabel('Measurement sweeps after burn-in');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[row,0].set_title(record['name']+f"; α={record['alpha']:g}, b={record['b']:g}")
        axes[row,1].set_title(f"Independent run: R-hat {record['rhat_observable_max']:.4f}")
    fig.suptitle('Longer-run diagnostics: final pooled estimate is a reference, not exact Gibbs truth')
    save(fig,out,'emnist-convergence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results'/'emnist-ising')
    p.add_argument('--output',type=Path,default=ROOT/'results'/'emnist-confirmation')
    p.add_argument('--classes',nargs='+',type=int,default=list(range(0,47,4)))
    p.add_argument('--continuous',action='store_true')
    p.add_argument('--fixed-reference',action='store_true')
    p.add_argument('--all-classes',action='store_true')
    p.add_argument('--burn',type=int,default=2048);p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.all_classes:args.classes=list(range(47))
    if args.continuous and args.fixed_reference:p.error('Choose continuous or fixed-reference, not both.')
    confirm(args)
