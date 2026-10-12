"""Train-only rule selection, held-out evaluation, scientific figures and summary."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quantum-segmentation-matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_emnist_ising import ALPHAS, BS, ROOT, proxy_metrics
from ising_monte_carlo import signed_image,image_parameter_rule


def load_runs(root):
    return [json.loads(p.read_text()) for p in sorted((root/'samples').glob('*.json'))]


def collect(root, extension=None):
    runs=(load_runs(extension) if extension is not None else [])+load_runs(root)
    # The primary sweep takes precedence for the deliberately repeated (.003,8) point.
    table={(r['name'],r['alpha'],r['b']):r for run in runs for r in run['records']}
    return runs,table


def continuous_rule(image, coefficient, k):
    result=image_parameter_rule(image,coefficient,k)
    return result['alpha'],result['b'],dict(margin_median=result['margin_median'],
        target_alpha=result['alpha'],target_b=result['b'])


def choose_grid_rule(image, k, coefficient=1., pairs=None):
    target_alpha,_,details=continuous_rule(image,coefficient,k)
    pairs=pairs or [(a,b) for a in ALPHAS for b in BS]
    alphas=sorted(set(a for a,b in pairs))
    alpha=float(alphas[np.argmin(abs(np.log(np.asarray(alphas)/target_alpha)))])
    scale=float(np.median(abs(signed_image(image,alpha).delta)))/4
    target_b=k/max(scale,1e-10)
    bs=sorted(b for a,b in pairs if a==alpha)
    b=float(bs[np.argmin(abs(np.log(np.asarray(bs)/target_b)))])
    return alpha,b,dict(details,target_b_after_alpha_rounding=target_b)


def bootstrap(values, seed=3107):
    values=np.sort(np.asarray(values,float)); rng=np.random.default_rng(seed)
    samples=rng.choice(values,(5000,len(values)),replace=True).mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),n=len(values))


def fit(root, extension=None):
    allruns,_=collect(root,extension)
    runs=[r for r in allruns if r['case']['split']=='train']
    cases=json.loads((root/'manifest.json').read_text())['cases']
    images=np.load(root/'subset.npz')['images']; images={c['name']:im for c,im in zip(cases,images)}
    table={(r['name'],r['alpha'],r['b']):r for run in runs for r in run['records']}
    names=sorted(c['name'] for c in cases if c['split']=='train')
    pairs=sorted(set((a,b) for n,a,b in table))
    assert all((n,a,b) in table for n in names for a,b in pairs), 'Finish the training sweep first.'
    fixed=[]
    for a,b in pairs:
        rows=[table[n,a,b] for n in names]
        fixed.append(dict(alpha=a,b=b,mean_iou=float(np.mean([r['iou'] for r in rows])),
            mean_resolved=float(np.mean([r['resolved_fraction'] for r in rows])),
            observable_rhat_over1_05=sum(r['rhat_observable_max']>1.05 for r in rows)))
    best=max(fixed,key=lambda r:r['mean_iou'])
    q_ref=float(np.median([continuous_rule(images[n],1.,1.)[2]['margin_median'] for n in names]))
    coefficient=best['alpha']*q_ref
    k_ref=float(np.median([best['b']*np.median(abs(signed_image(images[n],best['alpha']).delta))/4 for n in names]))
    selected=[table[(n,*choose_grid_rule(images[n],k_ref,coefficient,pairs)[:2])] for n in names]
    chosen=dict(k=k_ref,alpha_coefficient=coefficient,training_margin_median=q_ref,
        mean_iou=float(np.mean([r['iou'] for r in selected])),
        mean_resolved=float(np.mean([r['resolved_fraction'] for r in selected])))
    rule=dict(training_images=len(names),selection_metric='mean foreground IoU at proxy threshold 128',
        fixed=best,adaptive=chosen,fixed_candidates=fixed,available_pairs=pairs,
        extension=str(extension) if extension else None,
        formula='q=median(|125-d| over d>0), clipped below at 1; alpha nearest log-grid to c/q; '
                'b nearest log-grid to 4*k/median(|Delta(alpha)|); beta_paper=N*b',
        continuous_formula='alpha=c/q; b=4*k/median(|Delta(alpha)|), beta_paper=N*b; '
                           'c=best training alpha*median training q, k=median training b*median(|Delta|)/4',
        warning='Exploratory rounded rule on this fixed grid, not a universal optimum or confidence guarantee.',
        test_used_for_selection=False)
    path=root/'frozen-rule.json'
    if path.exists() and json.loads(path.read_text())!=rule:
        raise ValueError('Rule already frozen with different contents; do not silently refit on held-out results.')
    path.write_text(json.dumps(rule,indent=2)+'\n')
    print(json.dumps({k:rule[k] for k in ('fixed','adaptive')},indent=2))
    return rule


def otsu(image):
    hist=np.bincount(image.ravel(),minlength=256).astype(float); hist/=hist.sum()
    p=np.cumsum(hist); mu=np.cumsum(hist*np.arange(256)); total=mu[-1]
    var=np.divide((total*p-mu)**2,p*(1-p),out=np.zeros(256),where=(p*(1-p)>0))
    threshold=int(np.argmax(var))
    return image>threshold,threshold


def segmentation_iou(prediction, truth):
    return float(np.sum(prediction & truth)/max(1,np.sum(prediction | truth)))


def evaluate(root, output, extension=None):
    rule=json.loads((root/'frozen-rule.json').read_text()); runs,table=collect(root,extension)
    manifest=json.loads((root/'manifest.json').read_text()); cases=manifest['cases']
    images=np.load(root/'subset.npz')['images']; imdict={c['name']:im for c,im in zip(cases,images)}
    feature_table={r['case']['name']:dict(r['features'],evaluated_alpha=r['signature']['alpha'])
        for r in sorted(runs,key=lambda x:x['signature']['alpha'],reverse=True)}
    groups=[]; selections=[]; bysplit={}
    for split in ['train','test']:
        names=sorted(c['name'] for c in cases if c['split']==split)
        assert all((n,a,b) in table for n in names for a,b in rule['available_pairs']), f'{split} incomplete'
        for a,b in rule['available_pairs']:
                rows=[table[n,a,b] for n in names]
                groups.append(dict(split=split,alpha=a,b=b,
                    **{key:bootstrap([r[key] for r in rows]) for key in
                       ['iou','balanced_accuracy','resolved_fraction','confident_wrong_fraction']},
                    observable_rhat_over1_05=sum(r['rhat_observable_max']>1.05 for r in rows)))
        fixed=[]; adaptive=[]; oracle=[]; baseline=[]
        for n in names:
            a,b,details=choose_grid_rule(imdict[n],rule['adaptive']['k'],rule['adaptive']['alpha_coefficient'],rule['available_pairs'])
            f=table[n,rule['fixed']['alpha'],rule['fixed']['b']]; ad=table[n,a,b]
            best=max((table[n,x,y] for x,y in rule['available_pairs']),key=lambda r:r['iou'])
            pred,t=otsu(imdict[n]); base=segmentation_iou(pred,imdict[n]>=128)
            fixed.append(f); adaptive.append(ad); oracle.append(best); baseline.append(base)
            selections.append(dict(name=n,split=split,fixed=f,adaptive=ad,oracle=best,
                                   rule_details=details,otsu_threshold=t,otsu_iou=base,features=feature_table[n]))
        bysplit[split]={}
        for label,rows in [('fixed',fixed),('adaptive',adaptive),('oracle',oracle)]:
            bysplit[split][label]={key:bootstrap([r[key] for r in rows]) for key in
                ['iou','balanced_accuracy','accuracy','foreground_recall','foreground_precision',
                 'resolved_fraction','confident_wrong_fraction','ink_seed_iou','iou_threshold96','iou_threshold160']}
            bysplit[split][label]['rhat_max']=max(r['rhat_observable_max'] for r in rows)
            bysplit[split][label]['ess_mean_gamma_min']=min(r['ess_mean_gamma'] for r in rows)
        bysplit[split]['otsu_iou']=bootstrap(baseline)
        bysplit[split]['adaptive_minus_fixed_iou']=bootstrap([a['iou']-f['iou'] for a,f in zip(adaptive,fixed)])
    summary=dict(protocol=json.loads((root/'protocol.json').read_text()),manifest=manifest,rule=rule,
        results=bysplit,groups=groups,selections=selections,
        sampler=dict(total_image_alpha_jobs=len(runs),total_grid_points=len(table),
            sum_job_wall_seconds=sum(r['seconds'] for r in runs),
            minimum_swap_acceptance=min(r['swap_min'] for r in runs),
            jobs_with_no_round_trip=sum(r['round_trips']==0 for r in runs),
            completed_round_trips=sum(r['round_trips'] for r in runs),
            grid_points_observable_rhat_above1_05=sum(r['rhat_observable_max']>1.05 for r in table.values()),
            observable_rhat_max=max(r['rhat_observable_max'] for r in table.values())))
    continuous_path=ROOT/'emnist-ising-continuous-summary.json'
    if root.resolve()==(ROOT/'results'/'emnist-ising').resolve() and continuous_path.exists():
        continuous=json.loads(continuous_path.read_text())
        fixed_map={e['name']:e['fixed']['iou'] for e in selections if e['split']=='test'}
        summary['continuous_rule']=dict(aggregate=continuous['aggregate'],
            rhat_max=continuous['rhat_max'],ess_mean_gamma_min=continuous['ess_mean_gamma_min'],
            alpha_range=[min(r['alpha'] for r in continuous['records']),max(r['alpha'] for r in continuous['records'])],
            b_range=[min(r['b'] for r in continuous['records']),max(r['b'] for r in continuous['records'])],
            minus_fixed_iou=bootstrap([r['iou']-fixed_map[r['name'].removesuffix('-continuous')] for r in continuous['records']]),
            caveat='Continuous runs have four times as many measurement sweeps as the exploratory fixed grid; '
                   'this is not a matched-compute comparison of adaptation.')
        fixed_path=ROOT/'emnist-ising-fixed-reference-summary.json'
        if fixed_path.exists():
            reference=json.loads(fixed_path.read_text())
            fixed_long={r['name'].removesuffix('-fixed-reference'):r['iou'] for r in reference['records']}
            summary['fixed_reference']=dict(aggregate=reference['aggregate'],rhat_max=reference['rhat_max'],
                ess_mean_gamma_min=reference['ess_mean_gamma_min'])
            summary['continuous_rule']['minus_matched_fixed_iou']=bootstrap([
                r['iou']-fixed_long[r['name'].removesuffix('-continuous')] for r in continuous['records']])
    audit(summary,root)
    output.write_text(json.dumps(summary,indent=2)+'\n')
    plot(root,summary,imdict,extension)
    print(json.dumps(bysplit,indent=2))
    return summary


def audit(summary,root):
    """Attach reproducible source/environment metadata and completed checks."""
    import scipy,numba
    summary['environment']=dict(python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,numba=numba.__version__)
    summary['source_sha256']={name:hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest()
        for name in ['fetch_emnist.py','ising_monte_carlo.py','run_emnist_ising.py',
                     'analyze_emnist_ising.py','confirm_emnist_ising.py','validate_emnist_ising.py']}
    if root.resolve()!=(ROOT/'results'/'emnist-ising').resolve():return
    summary['validation']={}
    validation_path=ROOT/'emnist-ising-validation.json'
    if validation_path.exists():
        v=json.loads(validation_path.read_text())
        summary['validation'].update(exact_patch_max_gamma_error=v['max_error'],
            exact_patch_conditions=v['parameter_points'],outside_6se_plus002=v['outside_6se_plus002'])
    confirmation_path=ROOT/'emnist-ising-confirmation-summary.json'
    if confirmation_path.exists():
        v=json.loads(confirmation_path.read_text());chosen=[r for r in v['comparisons'] if r['is_selected_rule']]
        summary['validation'].update(long_confirmation_images=len(chosen),
            long_max_observable_rhat=max(r['rhat_observable_max'] for r in v['comparisons']),
            long_max_gamma_linf_change=max(r['gamma_linf_change'] for r in chosen),
            binary_controls_perfect=sum(r['iou']==1 for r in v['binary_controls']))
    mat_path=ROOT/'results'/'emnist-confirmation'/'emnist-gibbs-reference.mat'
    if mat_path.exists():
        from scipy.io import loadmat
        m=loadmat(mat_path);n=m['image'].size;model=signed_image(m['image'],m['alpha'].item())
        assert m['gamma_seed'].shape[-1]==n and np.max(abs(m['gamma_seed']))<=.25
        assert np.allclose(m['beta_paper'],n*m['beta_effective'])
        assert np.allclose(m['delta'].ravel(),model.delta)
        summary['validation'].update(mat_file_python_roundtrip_verified=True,matlab_executed=False)


def save(fig,root,name):
    for ext in ('png','svg'): fig.savefig(root/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot(root,s,images,extension=None):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    for ax,key,title in zip(axes,['iou','resolved_fraction','confident_wrong_fraction'],
        ['Foreground IoU','Fraction with resolved correlation sign','Confidently wrong pixels']):
        mat=np.array([[next(g[key]['mean'] for g in s['groups'] if g['split']=='test' and g['alpha']==a and g['b']==b)
                       for b in BS] for a in ALPHAS])
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis' if key!='confident_wrong_fraction' else 'magma')
        ax.set_xticks(range(len(BS)),[f'{b:g}' for b in BS]);ax.set_yticks(range(len(ALPHAS)),ALPHAS)
        ax.set_xlabel(r'$b=\beta_{paper}/N$');ax.set_ylabel(r'Positive sensitivity $\alpha$');ax.set_title(title)
        for i in range(len(ALPHAS)):
            for j in range(len(BS)):
                flagged=next(g['observable_rhat_over1_05'] for g in s['groups'] if g['split']=='test' and g['alpha']==ALPHAS[i] and g['b']==BS[j])
                ax.text(j,i,f'{mat[i,j]:.2f}'+('*' if flagged else ''),ha='center',va='center',color='white' if mat[i,j]<.65 else 'black',fontsize=8)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('EMNIST: held-out 28×28 images from all 47 classes\n* At least one image has R-hat >1.05; small-b sign estimates can be unresolved',fontsize=12)
    save(fig,root,'emnist-grid')
    if extension is not None:
        ext=json.loads((extension/'protocol.json').read_text());ea=ext['alphas'];eb=ext['bs']
        fig,axes=plt.subplots(1,2,figsize=(11,3.2),layout='constrained')
        for ax,key,title in zip(axes,['iou','resolved_fraction'],['Foreground IoU','Resolved correlation signs']):
            mat=np.array([[next(g[key]['mean'] for g in s['groups'] if g['split']=='test' and g['alpha']==a and g['b']==b) for b in eb] for a in ea])
            im=ax.imshow(mat,vmin=0,vmax=1,aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(eb)),eb);ax.set_yticks(range(len(ea)),ea)
            ax.set_xlabel(r'$b=\beta_{paper}/N$');ax.set_ylabel(r'$\alpha$');ax.set_title(title)
            for i in range(len(ea)):
                for j in range(len(eb)):
                    flag=next(g['observable_rhat_over1_05'] for g in s['groups'] if g['split']=='test' and g['alpha']==ea[i] and g['b']==eb[j])
                    ax.text(j,i,f'{mat[i,j]:.3f}'+('*' if flag else ''),ha='center',va='center',color='white' if mat[i,j]<.65 else 'black')
            fig.colorbar(im,ax=ax)
        fig.suptitle('Extended held-out sweep (* includes an image with R-hat >1.05)')
        save(fig,root,'emnist-extension')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for a in sorted(set(g['alpha'] for g in s['groups'])):
        rows=sorted([g for g in s['groups'] if g['split']=='test' and g['alpha']==a],key=lambda r:r['b'])
        bs=[r['b'] for r in rows]
        y=[r['iou']['mean'] for r in rows]
        axes[0].plot(bs,y,'o-',label=f'α={a:g}')
        axes[1].plot(np.asarray(bs)*np.arctan(125*a)/4,y,'o-',label=f'α={a:g}')
    axes[0].set_xlabel(r'$b=\beta_{paper}/N$');axes[1].set_xlabel(r'Background coupling $K=b\arctan(125\alpha)/4$')
    for ax in axes: ax.set_ylabel('Mean foreground IoU');ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_title('Temperature sensitivity');axes[1].set_title('Coupling-normalized comparison')
    fig.suptitle('Exploratory mean IoU; see grid figures for convergence flags')
    save(fig,root,'emnist-scaling')
    test=sorted([r for r in s['selections'] if r['split']=='test'],key=lambda r:r['adaptive']['iou'])
    example=[test[i] for i in [0,len(test)//4,len(test)//2,3*len(test)//4,len(test)-1]]
    fig,axes=plt.subplots(len(example),5,figsize=(10,9.5),layout='constrained')
    for row,entry in enumerate(example):
        n=entry['name'];image=images[n];r=entry['adaptive'];a=r['alpha'];b=r['b']
        path=root/'samples'/f'{n}-a{a:g}.npz'
        if not path.exists() or b not in np.load(path)['bs']:
            path=extension/'samples'/f'{n}-a{a:g}.npz'
        data=np.load(path);idx=int(np.flatnonzero(data['bs']==b)[0])
        g=data['gamma'][idx,0].reshape(image.shape);se=data['se'][idx,0].reshape(image.shape)
        arrays=[image,image>=128,g,g<=0,abs(g)<=np.maximum(3*se,1e-12)]
        for col,array in enumerate(arrays):
            kwargs=dict(cmap='coolwarm',vmin=-.25,vmax=.25) if col==2 else dict(cmap='gray',vmin=0,vmax=255 if col==0 else 1)
            axes[row,col].imshow(array,**kwargs);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(f"class {entry['adaptive']['class_id']}\nIoU {r['iou']:.2f}\nα={a:g}, b={b:g}")
    for ax,title in zip(axes[0],['EMNIST input','Proxy mask ≥128','Γ: red +, blue −','Algorithm 1 mask','Unresolved signs']):ax.set_title(title,fontsize=10)
    axes[-1,2].set_xlabel('Shared color range: −0.25 to +0.25')
    fig.suptitle('Grid-rounded rule: worst, quartiles, and best held-out result',fontsize=13)
    save(fig,root,'emnist-examples')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    x=np.array([e['features']['boundary_antiferromagnetic_fraction'] for e in test]);y=np.array([e['adaptive']['iou'] for e in test])
    axes[0].scatter(x,y,c=[e['features']['frustrated_plaquette_fraction'] for e in test],cmap='magma')
    axes[0].set_xlabel('Proxy boundary edges with d>125');axes[0].set_ylabel('Adaptive-rule IoU');axes[0].set_title('Does the Hamiltonian encode the boundary?')
    labels=['Fixed','Image rule','Per-image oracle','Otsu']
    vals=[[e['fixed']['iou'] for e in test],[e['adaptive']['iou'] for e in test],
          [e['oracle']['iou'] for e in test],[e['otsu_iou'] for e in test]]
    if 'fixed_reference' in s:
        ref=json.loads((ROOT/'emnist-ising-fixed-reference-summary.json').read_text())
        cont=json.loads((ROOT/'emnist-ising-continuous-summary.json').read_text())
        labels=['Fixed (long)','Image rule','Grid oracle','Otsu']
        vals[:2]=[[r['iou'] for r in ref['records']],[r['iou'] for r in cont['records']]]
    axes[1].boxplot(vals,tick_labels=labels,showmeans=True);axes[1].set_ylabel('Held-out foreground IoU');axes[1].set_ylim(0,1.03)
    axes[1].set_title('Oracle uses test masks; Otsu is a proxy comparison')
    save(fig,root,'emnist-rule-comparison')
    if extension is not None:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        edges=np.arange(0,57,7);centers=(edges[:-1]+edges[1:])/2
        names=[c['name'] for c in s['manifest']['cases'] if c['split']=='test']
        seedmap={c['name']:c['seeds'][0] for c in s['manifest']['cases']}
        for b in [8.,16.,20.,24.,32.]:
            curves=[[],[]]
            for n in names:
                image=images[n];truth=image.ravel()>=128;seed=seedmap[n]
                rr,cc=np.indices(image.shape);sr,sc=divmod(seed,image.shape[1])
                distance=(abs(rr-sr)+abs(cc-sc)).ravel()
                data=np.load(extension/'samples'/f'{n}-a0.001.npz')
                idx=int(np.flatnonzero(data['bs']==b)[0]);gamma=data['gamma'][idx,0]
                for kind in [0,1]:
                    values=[]
                    for low,high in zip(edges[:-1],edges[1:]):
                        keep=(distance>low)&(distance<=high)&(truth==bool(kind))
                        values.append(float(np.mean(gamma[keep])) if keep.any() else np.nan)
                    curves[kind].append(values)
            for kind,ax in enumerate(axes):
                values=np.array(curves[kind]);counts=np.sum(np.isfinite(values),axis=0)
                good=counts>0;mean=np.nansum(values,axis=0)/np.maximum(counts,1)
                ax.plot(centers[good],mean[good],'o-',label=f'b={b:g}')
        for ax in axes:ax.axhline(0,color='black',lw=.8);ax.set_xlabel('Manhattan distance from background seed');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylabel('Mean seed correlation Γ');axes[0].set_title('Background pixels: desired sign positive')
        axes[1].set_ylabel('Mean seed correlation Γ');axes[1].set_title('Foreground pixels: desired sign negative')
        fig.suptitle('Long-range correlations, α=0.001; image means averaged within each nonempty distance bin')
        save(fig,root,'emnist-correlations-distance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results'/'emnist-ising')
    p.add_argument('--extension',type=Path)
    p.add_argument('--fit',action='store_true');p.add_argument('--summary',type=Path,default=ROOT/'emnist-ising-summary.json')
    args=p.parse_args()
    if args.fit:fit(args.input,args.extension)
    else:evaluate(args.input,args.summary,args.extension)
