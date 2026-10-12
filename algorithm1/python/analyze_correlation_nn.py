"""Locked out-of-fold evaluation and comparisons with earlier Algorithm 1."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_key]='1'
import json
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import t as student_t
from correlation_nn_data import ROOT,OUT,init
from run_correlation_nn import dataset,reliable_rules
from weizmann_petz_data import write_json
from segmentation_bpm import boundary_proximity


def ious(pred,truth):
    p=np.asarray(pred,bool).reshape(len(pred),-1);y=np.asarray(truth,bool).reshape(len(truth),-1)
    return (p&y).sum(1)/np.maximum((p|y).sum(1),1)


def metric_arrays(pred,truth,prob=None,thresholds=None,seed=None):
    p=pred.reshape(-1,16,16);y=truth.reshape(-1,16,16)
    iou=ious(p,y);inter=(p&y).sum((1,2));den=p.sum((1,2))+y.sum((1,2))
    out=dict(iou=iou,dice=2*inter/np.maximum(den,1),pixel_accuracy=(p==y).mean((1,2)),
        bpm=np.array([boundary_proximity(a,b,1)['iou'] for a,b in zip(p,y)]))
    if prob is not None:
        unresolved=abs(prob-thresholds[:,None])<=.1
        known=np.zeros_like(unresolved) if seed is None else seed.astype(bool)
        out['score_band_unresolved']=(unresolved&~known).sum(1)/(~known).sum(1)
    return out


def aggregate(values):
    rng=np.random.default_rng(718625);draw=rng.integers(0,len(values),(10000,len(values)))
    return dict(mean=float(np.mean(values)),ci95=np.quantile(values[draw].mean(1),[.025,.975]).tolist())


def ensemble_predictions(selection):
    grouped=defaultdict(list)
    for key in selection:grouped['/'.join(key.split('/')[:-1])].append(key)
    output={}
    for group,names in grouped.items():
        prob=np.empty((100,256));prefix=np.empty((2,100,256));threshold=np.empty(100);rep=np.empty((3,100,256));covered=[]
        for name in names:
            paths=[OUT/'final'/f'{name.replace("/","--")}--r{r}.npz' for r in range(3)]
            data=[np.load(p) for p in paths];indices=data[0]['test_indices'];covered+=indices.tolist()
            rep[:,indices]=np.stack([d['probabilities'] for d in data])
            prob[indices]=rep[:,indices].mean(0)
            prefix[:,indices]=np.stack([d['prefix_probabilities'] for d in data]).mean(0)
            threshold[indices]=selection[name]['result']['threshold']
        assert sorted(covered)==list(range(100))
        output[group]=dict(prob=prob,prefix=prefix,threshold=threshold,replicates=rep,pred=prob>threshold[:,None])
    return output


def classical(p,selection):
    im,y,seed,seeds,ids=dataset();truth=y.reshape(100,-1)
    corr={rid:np.load(OUT/f'correlations-{rid}.npy',mmap_mode='r')[:,-1] for rid in p['rules']}
    output={};choices={};cost={}
    for model in ('Ising','XXZ'):
        candidates=[rid for rid,r in p['rules'].items() if (r['strength']==0)==(model=='Ising')]
        for method in ('one-sign','one-threshold','two-max','same-H-sign','one-sign-80','one-threshold-80','two-max-80'):
            decision=method.removesuffix('-80')
            pred=np.zeros((100,256),bool);picked=[];resolved=np.ones((100,256),bool)
            for fold in p['folds']:
                fitting=fold['train']+fold['validation'] if method.endswith('-80') else fold['validation']
                f=fold['fold'];va=[ids.index(s) for s in fitting];te=[ids.index(s) for s in fold['test']]
                eligible=[r for r in candidates if r in reliable_rules(p,f)]
                thresholds=np.round(np.arange(-.8,.801,.1),2) if decision=='one-threshold' else [0.]
                if decision=='same-H-sign':eligible=[selection[f'{model}/foreground/{f}']['job']['rid']]
                winner=None
                for rid in eligible:
                    value=corr[rid][np.arange(100),:,seeds[:,1]]
                    if decision=='two-max':value=value-corr[rid][np.arange(100),:,seeds[:,0]]
                    for t in thresholds:
                        candidate=value[va]>t
                        candidate[np.arange(len(va)),seeds[va,1]]=True
                        if decision=='two-max':candidate[np.arange(len(va)),seeds[va,0]]=False
                        score=float(ious(candidate,truth[va]).mean())
                        rank=(-score,rid,float(t))
                        if winner is None or rank<winner[0]:winner=(rank,rid,float(t),value)
                _,rid,t,value=winner;pred[te]=value[te]>t
                pred[te,seeds[te,1]]=True
                if decision=='two-max':pred[te,seeds[te,0]]=False
                picked.append(dict(fold=f,rule_id=rid,threshold=t,selection_iou=-winner[0][0],fitting_sources=len(va)))
                se=np.load(OUT/f'standard-errors-{rid}.npy',mmap_mode='r')[te]
                # For two seeds, conservative sum of individual SE; no false independence assumption.
                width=se[np.arange(20),:,seeds[te,1]]
                if decision=='two-max':width=width+se[np.arange(20),:,seeds[te,0]]
                mult=student_t.ppf(1-.05/(2*256),63)
                resolved[te]=abs(value[te]-t)>mult*width
            unknown=np.ones_like(resolved);unknown[np.arange(100),seeds[:,1]]=False
            if decision=='two-max':unknown[np.arange(100),seeds[:,0]]=False
            name=f'{model}/{method}';output[name]=dict(pred=pred,statistical_unresolved=((~resolved)&unknown).sum(1)/unknown.sum(1));choices[name]=picked;cost[name]=8192
    return output,choices,cost


def historical(p):
    """Per-source previous balanced adaptive baseline, matching old report."""
    from analyze_higher_seed_study import load,MI
    arrays,*_=load();out={};all_ids=[c['source_id'] for c in p['cases']]
    old=json.loads((ROOT/'weizmann-petz-manifest.json').read_text())['test_ids'];indices=[all_ids.index(s) for s in sorted(old)]
    out['Ising']=dict(indices=indices,iou=arrays['Ising/foreground',16,2048,'pair/balanced'][:,MI['iou']])
    values=[]
    for sid in sorted(old):
        cid=sid+'--square-16';name=cid+'--XXZ-foreground'
        path=ROOT/'results/higher-seeds/confirm'/f'{name}--8192.json'
        if not path.exists():path=ROOT/'results/higher-seeds/test'/f'{name}--2048.json'
        m=json.loads(path.read_text());v=np.load(path.with_suffix('.npz'))['metrics']
        values.append(float(v[m['labels']['pair/balanced'],:,MI['iou']].mean()))
    out['XXZ']=dict(indices=indices,iou=np.asarray(values))
    two=json.loads((ROOT/'xxz-two-seed-summary.json').read_text())
    for model in ('Ising','XXZ'):
        label=two['primary_labels'][model];rid=two['selection'][label]['rule_id'];values=[]
        for sid in sorted(old):
            name=f'{sid}--square-16--{rid}.json';path=ROOT/'results/xxz-two-seed/confirm'/name
            if not path.exists():path=ROOT/'results/xxz-two-seed/test'/name
            values.append(json.loads(path.read_text())['decisions']['two_max']['iou'])
        out[model]['two_seed_iou']=np.asarray(values)
        expected=two['groups'][label+'/16']['decisions']['two_max']['iou']['mean']
        assert abs(np.mean(values)-expected)<1e-10
    return out


def main():
    p=init();s=json.loads((ROOT/'correlation-nn-final-selection.json').read_text());im,y,seed,seeds,ids=dataset()
    predictions=ensemble_predictions(s);baselines,choices,cost=classical(p,s)
    values={};summary={}
    for name,v in predictions.items():
        values[name]=metric_arrays(v['pred'],y,v['prob'],v['threshold'],seed if name.endswith('/foreground') else None)
        summary[name]={k:aggregate(a) for k,a in values[name].items()}
        summary[name]['individual_training_seed_iou']=[float(ious(a>v['threshold'][:,None],y).mean()) for a in v['replicates']]
        summary[name]['prefix_iou']={str(b):float(ious(v['prefix'][i]>v['threshold'][:,None],y).mean()) for i,b in enumerate((2048,4096))}
        summary[name]['retained_configurations']=0 if name.startswith(('prior/','image/')) else 8192
    for name,v in baselines.items():
        values[name]=metric_arrays(v['pred'],y);summary[name]={k:aggregate(a) for k,a in values[name].items()}
        summary[name]['statistical_unresolved']=float(v['statistical_unresolved'].mean());summary[name]['retained_configurations']=cost[name]
    pairing={}
    for name in ('Ising/none','XXZ/none'):
        wrong_pairing=np.empty(100)
        for fold in p['folds']:
            ix=np.array([ids.index(sid) for sid in fold['test']]);pred=predictions[name]['pred'][ix]
            truth=y[ix].reshape(20,-1)
            # Seedless forward ignores intensities and seeds. A donor's output
            # in this same fold is exactly the frozen decoder on donor C.
            inter=(truth[:,None,:]&pred[None,:,:]).sum(-1)
            union=(truth[:,None,:]|pred[None,:,:]).sum(-1)
            matrix=inter/np.maximum(union,1)
            np.testing.assert_allclose(np.diag(matrix),values[name]['iou'][ix],rtol=0,atol=1e-12)
            wrong_pairing[ix]=(matrix.sum(1)-np.diag(matrix))/19
        pairing[name]=dict(matched=aggregate(values[name]['iou']),mismatched=aggregate(wrong_pairing),
            difference=aggregate(values[name]['iou']-wrong_pairing),per_image_expected_mismatch_iou=wrong_pairing.tolist(),
            definition='Expected IoU when each source receives C from a uniformly chosen different outer-test source in the same fold; exactly average over all 19 donors; frozen model and threshold.')
    effects={}
    for model in ('Ising','XXZ'):
        for mode in ('none','foreground'):
            target=f'{model}/{mode}';controls=[f'prior/{mode}',f'image/{mode}']
            if mode=='foreground':controls += [f'{model}/one-sign',f'{model}/one-threshold',f'{model}/one-threshold-80',f'{model}/two-max-80',f'{model}/same-H-sign',f'{model}-column/foreground']
            effects[target]={control:{k:aggregate(values[target][k]-values[control][k]) for k in ('iou','bpm')} for control in controls}
            historical_target=f'{model}-historical/{mode}'
            effects[historical_target]={control:{k:aggregate(values[historical_target][k]-values[control][k]) for k in ('iou','bpm')}
                for control in (f'prior/{mode}',f'image/{mode}')}
    old=historical(p);comparisons={};two_seed_comparisons={}
    for model,v in old.items():
        indices=v['indices'];comparison={}
        for key in (f'{model}/foreground',f'{model}-historical/foreground',f'{model}/none'):
            new=values[key]['iou'][indices]
            comparison[key]=dict(new=aggregate(new),old=aggregate(v['iou']),difference=aggregate(new-v['iou']))
        comparisons[model]=comparison
        two_seed_comparisons[model]={key:dict(new=aggregate(values[key]['iou'][indices]),old=aggregate(v['two_seed_iou']),
            difference=aggregate(values[key]['iou'][indices]-v['two_seed_iou'])) for key in (f'{model}/foreground',f'{model}/none')}
    parameters={}
    quality={}
    for group in predictions:
        configs=[v for k,v in s.items() if '/'.join(k.split('/')[:-1])==group]
        parameters[group]=dict(hamiltonians=dict(Counter(v['job']['rid'] for v in configs)),
            architecture=dict(Counter(v['job']['config']['architecture'] for v in configs)),
            epochs=[v['result']['epoch'] for v in configs],thresholds=[v['result']['threshold'] for v in configs],
            mean_inner_train_iou=float(np.mean([v['result']['train_iou'] for v in configs])),mean_inner_validation_iou=float(np.mean([v['result']['validation_iou'] for v in configs])))
        diagnostics=[]
        for key,v in s.items():
            if '/'.join(key.split('/')[:-1])!=group or v['job']['rid'] is None:continue
            fold=p['folds'][v['job']['fold']];d=json.loads((OUT/f'diagnostics-{v["job"]["rid"]}.json').read_text())
            diagnostics += [d[ids.index(sid)] for sid in fold['test']]
        if diagnostics:quality[group]=dict(converged_images=sum(d['converged'] for d in diagnostics),images=len(diagnostics),
            mean_failed_pair_fraction=float(np.mean([d['failed_pair_fraction'] for d in diagnostics])),
            saturated_pair_fraction=float(np.mean([d['saturated_pair_fraction'] for d in diagnostics])),
            participation_rank=float(np.mean([d['participation_rank'] for d in diagnostics])))
    fits=[json.loads(x.read_text()) for x in (OUT/'fits').glob('*.json')];finals=[json.loads(x.read_text()) for x in (OUT/'final').glob('*.json')]
    result=dict(metrics=summary,effects=effects,correlation_pairing_ablation=pairing,historical_comparisons=comparisons,previous_two_seed_comparisons=two_seed_comparisons,selection_summary=parameters,quality=quality,
        xxz_minus_ising={mode:{metric:aggregate(values[f'XXZ/{mode}'][metric]-values[f'Ising/{mode}'][metric])
            for metric in ('iou','bpm')} for mode in ('none','foreground')},
        representation_diagnostics=json.loads((ROOT/'correlation-nn-measurement-validation.json').read_text())['quality'],
        classical_selections=choices,rules=p['rules'],inner_fit_count=len(fits),refit_count=len(finals),
        summed_training_seconds=sum(x['result']['seconds'] for x in fits+finals),
        evaluation='100 source-level out-of-fold predictions; 3-training-seed probability ensemble; inner-selected threshold; intervals conditional on fitted models, not repeated nested-CV intervals.')
    write_json(ROOT/'correlation-nn-summary.json',result)
    write_json(OUT/'per-image-metrics.json',{name:[dict(source_id=ids[i],**{m:float(a[i]) for m,a in v.items()})
        for i in range(100)] for name,v in values.items()})
    np.savez_compressed(OUT/'evaluation-arrays.npz',**{k.replace('/','--')+'--'+m:a for k,v in values.items() for m,a in v.items()})
    np.savez_compressed(OUT/'predictions.npz',**{k.replace('/','--'):v['pred'] for k,v in {**predictions,**baselines}.items()})
    figures(result,values,predictions,baselines,im,y,ids);write_report(result)
    print('analysis complete',len(summary),'methods',flush=True)


def figures(r,values,predictions,baselines,image,truth,ids):
    cache=OUT/'plot-cache';cache.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    groups=['prior/none','image/none','Ising/none','XXZ/none','prior/foreground','image/foreground','Ising/one-sign-80','XXZ/one-sign-80',
        'Ising/one-threshold-80','XXZ/one-threshold-80','Ising/two-max-80','XXZ/two-max-80',
        'Ising-column/foreground','XXZ-column/foreground','Ising/foreground','XXZ/foreground']
    labels=['Position prior, no seed','Image NN, no seed','Ising all-pair NN, no seed','XXZ all-pair NN, no seed',
        'Position prior, 1 seed','Image NN, 1 seed','Ising sign, 1 seed','XXZ sign, 1 seed',
        'Ising threshold, 1 seed','XXZ threshold, 1 seed','Ising max, 2 seeds','XXZ max, 2 seeds',
        'Ising seed-column NN','XXZ seed-column NN','Ising all-pair NN, 1 seed','XXZ all-pair NN, 1 seed']
    fig,axes=plt.subplots(1,2,figsize=(11,8),layout='constrained')
    for ax,metric in zip(axes,('iou','bpm')):
        for i,name in enumerate(groups):
            a=r['metrics'][name][metric];lo,hi=a['ci95'];color='#b35720' if name.startswith('XXZ') else '#007c91' if name.startswith('Ising') else '#666666'
            ax.plot([lo,hi],[i,i],color=color);ax.plot(a['mean'],i,'o',color=color,ms=5)
        ax.set_yticks(range(len(groups)),labels if metric=='iou' else ['']*len(groups),fontsize=9);ax.invert_yaxis()
        ax.set_title('Foreground IoU' if metric=='iou' else 'Boundary proximity, radius 1');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Full pair-correlation neural decoding | 16×16 Weizmann\n100 out-of-fold images; pointwise source-bootstrap 95% intervals')
    fig.savefig(dest/'decoder-comparison.png',dpi=180);fig.savefig(dest/'decoder-comparison.pdf');plt.close(fig)
    # Fixed IDs, chosen without model outcomes; display every one, not only successes.
    chosen=np.linspace(0,99,6,dtype=int)
    columns=[('Image',image),('Consensus',truth),('Seed/position prior',predictions['prior/foreground']['pred'].reshape(-1,16,16)),
        ('Ising max, 2 seeds',baselines['Ising/two-max-80']['pred'].reshape(-1,16,16)),('Ising NN',predictions['Ising/foreground']['pred'].reshape(-1,16,16)),
        ('XXZ NN',predictions['XXZ/foreground']['pred'].reshape(-1,16,16)),('Image NN',predictions['image/foreground']['pred'].reshape(-1,16,16))]
    seedless=[('Image',image),('Consensus',truth)]+[(title,predictions[key+'/none']['pred'].reshape(-1,16,16))
        for title,key in [('Position prior','prior'),('Ising NN','Ising'),('XXZ NN','XXZ'),('Image NN','image')]]
    for filename,cols,description in [('segmentation-examples.png',columns,'NNs receive one known foreground seed; classical max receives two seeds'),
        ('segmentation-examples-seedless.png',seedless,'Seedless predictions: no test-image pixel labels supplied')]:
        fig,axes=plt.subplots(6,len(cols),figsize=(12,10),layout='constrained')
        for row,i in enumerate(chosen):
            for col,(title,data) in enumerate(cols):
                axes[row,col].imshow(data[i],cmap='gray',vmin=0,vmax=1);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if row==0:axes[row,col].set_title(title,fontsize=10)
                if col==0:axes[row,col].set_ylabel(ids[i],fontsize=7)
        fig.suptitle(description+'\nDeterministically selected outer-test examples; full-frame 16×16 normalization')
        fig.savefig(dest/filename,dpi=180);plt.close(fig)
    # Inner validation only: no use of outer-test data to draw parameter optima.
    scores=defaultdict(list)
    for path in (OUT/'fits').glob('*.json'):
        m=json.loads(path.read_text());j=m['job']
        if j['input_kind']=='full' and j['config']==init()['network_screen']:
            scores[(j['rid'],j['seed_mode'])].append(m['result']['validation_iou'])
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,mode in zip(axes,('none','foreground')):
        for g in (0.,.1,.5):
            xx=[];yy=[]
            for k in (.4,.8,1.2):
                a=[float(np.mean(scores[rid,mode])) for rid,rule in r['rules'].items() if rule['strength']==g and rule['K']==k]
                xx.append(k);yy.append(max(a))
            ax.plot(xx,yy,'-o',label=f'g/D={g:g}')
        ax.set_title('No seed' if mode=='none' else 'Known foreground seed');ax.set_xlabel('K = bD/4');ax.set_ylabel('Mean inner-validation IoU')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Hamiltonian screen: best threshold/sensitivity at each temperature\nExploratory inner-validation summary, not an outer-test parameter choice')
    fig.savefig(dest/'hamiltonian-sweep.png',dpi=180);plt.close(fig)


def write_report(r):
    def fmt(v):return f'{v["mean"]:.4f} [{v["ci95"][0]:.4f}, {v["ci95"][1]:.4f}]'
    def mean(name):return r['metrics'][name]['iou']['mean']
    previous=r['previous_two_seed_comparisons']
    lines=['# Neural decoding of all-pair Ising and XXZ correlations','',
        'This experiment tests whether a learned segmentation rule can extract information missed by Algorithm 1. A positive gain over seed/position-only controls is evidence of usable correlation information. A negative result cannot establish an information-theoretic limit.','',
        '## Main findings','',
        f'On 100 source-level out-of-fold predictions, the primary all-pair networks achieve mean IoU {mean("Ising/none"):.4f} (Ising) and {mean("XXZ/none"):.4f} (XXZ) without any test-image seed. With one known foreground seed, the scores are {mean("Ising/foreground"):.4f} and {mean("XXZ/foreground"):.4f}. These are the choices selected with the sampling-quality requirement.',
        f'The seedless position-only prior scores {mean("prior/none"):.4f}, and the separately trained raw-image network scores {mean("image/none"):.4f}. Their seeded counterparts score {mean("prior/foreground"):.4f} and {mean("image/foreground"):.4f}. The raw-image control is a small network trained from scratch, not a pretrained segmentation benchmark.',
        f'On the common 75-image historical subset, the foreground-seeded all-pair networks score {previous["Ising"]["Ising/foreground"]["new"]["mean"]:.4f} and {previous["XXZ"]["XXZ/foreground"]["new"]["mean"]:.4f}, compared with {previous["Ising"]["Ising/foreground"]["old"]["mean"]:.4f} and {previous["XXZ"]["XXZ/foreground"]["old"]["mean"]:.4f} for the earlier two-seed Ising and XXZ rules. The neural methods have additional supervised training; matched controls and paired intervals below qualify the comparison.',
        f'Holding the earlier foreground-start Ising Hamiltonian fixed still gives NN IoU {r["historical_comparisons"]["Ising"]["Ising-historical/foreground"]["new"]["mean"]:.4f} on those same 75 sources, versus {r["historical_comparisons"]["Ising"]["Ising-historical/foreground"]["old"]["mean"]:.4f} for the earlier adaptive pairwise readout. Thus improvement does not require changing that Hamiltonian: learned decoding of the full pair matrix already finds useful information.',
        f'At the newly selected Hamiltonians, using only the foreground-seed correlation column gives NN IoU {mean("Ising-column/foreground"):.4f} and {mean("XXZ-column/foreground"):.4f}. Mismatching matrices with other images lowers seedless IoU to {r["correlation_pairing_ablation"]["Ising/none"]["mismatched"]["mean"]:.4f} and {r["correlation_pairing_ablation"]["XXZ/none"]["mismatched"]["mean"]:.4f}. These controls support the contribution of image-specific correlations beyond average object shape or seed position.',
        f'The XXZ-minus-Ising IoU difference is {fmt(r["xxz_minus_ising"]["none"]["iou"])} without seeds and {fmt(r["xxz_minus_ising"]["foreground"]["iou"])} with a foreground seed. These compare complete selected pipelines and do not isolate exchange as the cause.',
        'All conclusions concern this internal 16×16 benchmark. The final section reports independent larger-bank checks of the unrestricted XXZ selections separately.','',
        '## What the network receives','',
        'For each image, estimate C[i,j] = mean_s z_i^(s) z_j^(s) from Z-basis spin snapshots. The network maps the complete C matrix and fixed lattice coordinates to one foreground probability per pixel. In the seeded arm it also receives one known foreground pixel, which is forced to remain foreground. An inner-validation-selected probability threshold converts the output into a binary segmentation. Every measured pair is available to the decoder; the full-C model never receives image intensities or the Hamiltonian parameter values.',
        'C uses Pauli eigenvalues ±1 and equals four times the spin-1/2 Gamma values used in the paper code. Pixel indices use NumPy row-major order. The diagonal is exactly one and is not an additional measured feature. All pairs commute in the Z basis, so one snapshot supplies every pair product. No X/X or Y/Y correlation is included, even for XXZ.',
        'The volume decoder has 29,785 parameters at width 24 or 75,241 at width 40; the graph decoder has 21,289 at width 24. The latter normalizes positive and negative correlation-weighted messages separately. Both also receive local correlations and row summaries derived from the same matrix.','',
        '## Design','',
        '- All 100 Weizmann single-object source images, grayscale and 16×16 full-frame normalization; majority vote of three annotations. These images were used in earlier development, so this is internal cross-validation, not external validation.',
        '- Five fixed outer folds: 64 training, 16 inner-validation and 20 outer-test sources. No pixel-level split. Hamiltonian, architecture, checkpoint epoch, and output threshold are chosen on the inner validation set. Refit on the 80 outer-training sources with three fixed random seeds and average probabilities. Each source is evaluated once.',
        '- Input is the full 256×256 Pauli ZZ correlation matrix, with known pixel coordinates and optional one known foreground seed. The full-correlation models cannot access grayscale intensities. The foreground seed is the same mask-derived interior pixel as earlier studies; its geometry contains privileged information, measured by the seed/position-only control.',
        '- 36 Hamiltonians: threshold edge-Otsu/q90; sensitivity alpha=0.01 or inverse q90−q10 edge spread; K=0.4/0.8/1.2; g/D=0/0.1/0.5. D=median absolute Delta, b=4K/D, Delta=atan(alpha(tau−d)). The Ising branch has g=0. No fields or spin clamping.',
        '- A correlation-volume residual CNN and a graph message-passing network both receive every pair correlation. Widths 24/40, learning rates 0.001/0.003 and weight decay 0.001/0.01 are explored in the documented screening/refinement scheme. Dihedral augmentations transform both correlation axes and labels together.',
        '- Networks are trained from scratch with AdamW, batch size 16, equal-weight binary cross-entropy and soft-IoU loss, gradient norm clipping at 5, and cosine learning-rate decay. Screening runs 100 epochs; refinement runs 140. Checkpoints and mask thresholds (0.25 to 0.75 in steps of 0.05) are selected every five epochs using validation images. Refits preserve the selected learning-rate schedule and stop at the selected epoch.',
        '- Primary Hamiltonian choices require >=95% of the 80 outer-training images to pass full-pair/scalar convergence checks; no test labels or test convergence flags enter selection. Unrestricted XXZ choices are retained separately. No test image is dropped.',
        '- Four chains × 2,048 retained configurations per image/setting. Ising uses exact independent frontier samples; XXZ uses the existing complete-loop SSE/worldline-cluster sampler. Prefix tests use 512 and 1,024 samples per chain without retraining. Existing matching banks are reused; all correlations of an image share the same samples.',
        '- The raw-image CNN, spatial/seed prior, and single-seed-column CNN are separately trained controls. Classical sign and scalar-threshold rules select their own Hamiltonians on the same validation fold. The two-seed control receives an extra known background pixel.',
        '- The seed-column neural control uses the Hamiltonian chosen for the corresponding full-matrix foreground-seeded model, then tunes its own architecture and training settings. This isolates additional pairs at that Hamiltonian; it is not a separate Hamiltonian search for the best possible column-only decoder.',
        '- Prior/image/column controls retain the constant identity-matrix position encoding and are offered the same convolutional/graph architecture choices. This prevents attributing gains from extra position-coding capacity to measured off-diagonal correlations.',
        '- Additional classical controls suffixed -80 fit their Hamiltonian and scalar decision rule using all 80 outer-training labels. This gives the simpler baselines access to the full training-label budget; the original inner-validation-only controls are also retained. Neither version reads outer-test labels.',
        '- IoU is mean foreground IoU per source, not pooled pixels. Boundary proximity uses the existing Zhang skeleton convention at radius 1: the documented operational interpretation inherited from the crack-paper comparison, not a conventional contour-boundary IoU. Neural probability-band abstention is an uncalibrated diagnostic; it is not comparable to a Gibbs confidence interval. Hard neural masks assign every pixel.',
        f'- Completed {r["inner_fit_count"]} inner fits and {r["refit_count"]} refits; {r["summed_training_seconds"]/3600:.2f} summed worker-hours. Pointwise source-bootstrap intervals condition on the fitted models and do not include all variation from repeating nested cross-validation.','',
        '## Outer-test results','',
        '| Method / seed information | Mean IoU [95% CI] | Boundary proximity [95% CI] | Configurations per image |',
        '|---|---:|---:|---:|']
    for name,m in r['metrics'].items():lines.append(f'| {name} | {fmt(m["iou"])} | {fmt(m["bpm"])} | {m["retained_configurations"]} |')
    lines+=['','![Decoder comparison](results/correlation-nn/figures/decoder-comparison.png)','',
        '## Paired differences','', '| Decoder | Control | ΔIoU [95% CI] | Δ boundary [95% CI] |','|---|---|---:|---:|']
    for target,controls in r['effects'].items():
        for control,v in controls.items():lines.append(f'| {target} | {control} | {fmt(v["iou"])} | {fmt(v["bpm"])} |')
    lines+=['','| Seed information | XXZ minus Ising IoU [95% CI] | XXZ minus Ising boundary score [95% CI] |','|---|---:|---:|']
    for mode,v in r['xxz_minus_ising'].items():lines.append(f'| {mode} | {fmt(v["iou"])} | {fmt(v["bpm"])} |')
    lines+=['','Each model selects its own Hamiltonian and decoder; these are comparisons of the complete pipelines, not isolated causal effects of exchange.']
    lines+=['','### Does the decoder use image-specific correlation information?','',
        'For the seedless models, replace each image’s matrix with that of another held-out image in the same fold, keeping the trained network and threshold fixed. Since these models read neither intensities nor seeds, this exactly permutes their existing output masks. The table averages over all 19 alternative donors for each source; it preserves the decoder and its marginal output distribution while breaking the image–correlation pairing. This is an input ablation, not retraining or a formal information bound.','',
        '| Seedless model | Correct pairing IoU | Mismatched pairing IoU | Paired difference [95% CI] |','|---|---:|---:|---:|']
    for name,v in r['correlation_pairing_ablation'].items():
        lines.append(f'| {name} | {v["matched"]["mean"]:.4f} | {v["mismatched"]["mean"]:.4f} | {fmt(v["difference"])} |')
    lines+=['','## Comparison on the previous 75-image evaluation subset','',
        'These compare the same source images. The neural models were trained out of fold on up to 80 other sources; previous methods were tuned on the old 25-image split. Supervised training is additional information and computation. Historical NN rows retain the old foreground-start Hamiltonian, separating decoder changes from Hamiltonian tuning. The old adaptive baselines consume multiple rounds; the new networks consume one 8,192-configuration bank.','',
        '| Decoder | NN IoU | Earlier adaptive pairwise IoU | Paired difference [95% CI] |','|---|---:|---:|---:|']
    for model,v in r['historical_comparisons'].items():
        for name,m in v.items():lines.append(f'| {name} | {m["new"]["mean"]:.4f} | {m["old"]["mean"]:.4f} | {fmt(m["difference"])} |')
    lines+=['','Earlier two-seed tests are stronger benchmarks than the recent adaptive one-seed methods: primary Ising reached 0.4988 and primary XXZ 0.4973, with a secondary XXZ rule at 0.5010. The paired comparison below uses those primary methods on the same 75 sources. They receive both foreground and background seed labels; the neural rows receive one or neither. See [the earlier two-seed report](xxz-two-seed-bpm.md).','',
        '| Decoder | NN IoU | Earlier two-seed IoU | Paired difference [95% CI] |','|---|---:|---:|---:|']
    for model,v in r['previous_two_seed_comparisons'].items():
        for name,m in v.items():lines.append(f'| {name} | {m["new"]["mean"]:.4f} | {m["old"]["mean"]:.4f} | {fmt(m["difference"])} |')
    lines+=['','## Selected settings and sampling diagnostics','',
        '| Decoder | Architectures over five folds | Selected Hamiltonian IDs | Mean inner training IoU | Mean inner validation IoU | Converged test images |',
        '|---|---|---|---:|---:|---:|']
    for name,v in r['selection_summary'].items():
        q=r['quality'].get(name);qc=f'{q["converged_images"]}/{q["images"]}' if q else 'n/a'
        lines.append(f'| {name} | {v["architecture"]} | {v["hamiltonians"]} | {v["mean_inner_train_iou"]:.4f} | {v["mean_inner_validation_iou"]:.4f} | {qc} |')
    lines+=['','The complete rules, per-fold selections, thresholds, epochs, checkpoints, and predictions are saved in JSON/NPZ files. No Hamiltonian is selected separately using a test image’s annotation.','',
        '| Primary decoder | Threshold rule | Sensitivity rule | K | g/D | Folds selecting this rule |','|---|---|---|---:|---:|---:|']
    for group in ('Ising/none','XXZ/none','Ising/foreground','XXZ/foreground'):
        for rid,count in r['selection_summary'][group]['hamiltonians'].items():
            rule=r['rules'][rid]
            lines.append(f'| {group} | {rule["threshold"]} | {rule["sensitivity"]} | {rule["K"]} | {rule["strength"]} | {count} |')
    lines+=['','These selections are image-adaptive rules: q90 and edge-Otsu are computed from the current image’s nearest-neighbor intensity distances; inverse-spread sensitivity is 1/max(q90(d)−q10(d),1). K sets b=4K/D using that image’s D. Thus the same selected K does not imply the same numerical beta for every image. The selections optimize inner-validation performance within this grid, not a universal optimum.','',
        '![Hamiltonian sweep](results/correlation-nn/figures/hamiltonian-sweep.png)','',
        'The exploratory sweep figure uses the initial width-24 volume CNN, tested across the complete grid. Final choices may use the subsequent graph-network refinement. The figure includes unrestricted settings, including colder XXZ banks that fail convergence; those curves are not evidence of physical Gibbs-state behavior in those regimes. Primary outer-test selections apply the quality eligibility criterion.','',
        '### Correlation saturation diagnostic','',
        'The following label-independent diagnostic holds the Ising q90/inverse-spread rule fixed and varies temperature. Participation rank is (tr C)^2/tr(C²). It measures matrix concentration, not segmentation information: the infinite-temperature identity matrix has maximal rank but carries no image-dependent signal.','',
        '| K | Mean absolute off-diagonal correlation | Fraction with absolute correlation >0.95 | Mean participation rank |','|---:|---:|---:|---:|']
    for rid,rule in r['rules'].items():
        if rule['strength']==0 and rule['threshold']=='q90' and rule['sensitivity']=='inverse_spread':
            d=r['representation_diagnostics'][rid]
            lines.append(f'| {rule["K"]} | {d["mean_absolute_correlation"]:.4f} | {100*d["saturated_pair_fraction"]:.2f}% | {d["participation_rank"]:.2f} |')
    lines+=['',
        '## Sampling sensitivity','', '| Decoder | 2,048 total configurations | 4,096 | 8,192 |','|---|---:|---:|---:|']
    for name,m in r['metrics'].items():
        if 'prefix_iou' in m and m['retained_configurations']:
            lines.append(f'| {name} | {m["prefix_iou"]["2048"]:.4f} | {m["prefix_iou"]["4096"]:.4f} | {m["iou"]["mean"]:.4f} |')
    lines+=['','The models and thresholds stay fixed during prefix evaluation; these are input-noise sensitivity tests, not separately optimized low-budget models. Configuration counts are not independent sample counts for XXZ.','',
        '## Example segmentations','', '![Seedless examples](results/correlation-nn/figures/segmentation-examples-seedless.png)','',
        '![Seeded examples](results/correlation-nn/figures/segmentation-examples.png)','',
        '## What this can establish','',
        'An out-of-fold gain over Algorithm 1 on the same Hamiltonian demonstrates a decoder limitation. A gain over the spatial/seed prior is evidence that the measured correlations contribute information beyond seed geometry and average mask shape. Comparing all pairs with the single-seed column tests the incremental value of the rest of the matrix at the selected Hamiltonian. A gap to the raw-image network suggests information loss or difficult decoding, but does not by itself prove loss: input representation and training optimization differ.',
        'Conversely, outperforming this raw-image network would demonstrate a useful representation or inductive bias under this training budget, not additional information created by the Hamiltonian. The Hamiltonian and its sampled correlations are generated from the image. This study does not compare with a pretrained segmentation model or establish a new classical benchmark.',
        'For a finite zero-field classical Ising model with known graph and finite couplings, the edge-product statistics form a minimal exponential family. Strict convexity of the log partition function makes exact edge-correlation means identify the dimensionless edge couplings. Thus exact two-point information is not automatically a fundamental bottleneck for identifying that Gibbs distribution. This is an application of the exponential-family mean-parameter correspondence; it does not identify the original image or its semantic mask. Near saturation, inversion may be poorly conditioned, and finite sampling changes the practical question. See [Wainwright (2006), exponential-family background](https://jmlr.org/papers/volume7/wainwright06a/wainwright06a.pdf).',
        'A restricted XXZ extension follows from the relative-entropy argument in [Anshu et al., Proposition 17](https://arxiv.org/pdf/2004.07266). Our application: hold the dimensionless exchange operator A0 fixed, set A(θ)=A0+sum_e θ_e Z_i Z_j, and let ρθ=exp(−A(θ))/Zθ. Writing mθ for the edge ZZ means gives D(ρθ||ρη)+D(ρη||ρθ)=(η−θ)·(mθ−mη). Identical means make both relative entropies zero, hence the states equal. Taking matrix logarithms and using independence of the traceless edge Pauli operators then gives θ=η. In this study bg/4=K(g/D) is fixed within each Hamiltonian rule, satisfying that condition. This is an identifiability argument for exact finite-temperature means, not a practical reconstruction guarantee, and not a claim for arbitrary unknown exchange couplings. It does not undo information lost when converting image intensities to Hamiltonian couplings, determine semantic masks, or justify a zero-temperature limit.',
        'All-pair storage and classical processing scale quadratically in pixel count: 32,640 distinct off-diagonal pairs for 256 pixels. The volume CNN is resolution-specific. These coarse-image tests establish neither full-resolution performance nor quantum computational advantage. An unsuccessful learned decoder on only 100 images would not establish an upper bound.','',
        '## Reproduction','', 'Use the existing virtual environment plus requirements-neural.txt. The underlying dataset archive and cached-bank hashes are inherited and verified by the existing preparation scripts.','',
        '```sh', '.venv/bin/python -m pytest algorithm1/tests/test_correlation_nn.py -q',
        '.venv/bin/python algorithm1/python/correlation_nn_data.py sample --workers 4',
        '.venv/bin/python algorithm1/python/run_correlation_nn.py screen-ready --workers 4',
        '.venv/bin/python algorithm1/python/finish_correlation_nn.py', '```','',
        'The completion driver runs the measurement audit, refinement, selection, column controls, final refits, higher-budget XXZ checks, training audit, analysis, and frozen-model confirmation inference. Completed fit files are reused on restart. The saved protocol and amendments specify the fixed choices.','']
    (ROOT/'correlation-nn.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
