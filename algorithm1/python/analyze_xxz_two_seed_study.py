"""Reanalysis, source-paired inference, and plots for XXZ seed comparisons."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/xxz-two-seed-mpl')
import argparse
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_xxz_two_seed_study import ROOT,OUT,INPUT,DECISIONS,hashes,selected_rules,identifier,cases,confirmation_specs,digest
from weizmann_petz_data import write_json,seed_for
from weizmann_petz_metrics import mask_metrics
from segmentation_bpm import bpm_metrics
from skimage.morphology import skeletonize

METRICS=('iou','dice','unresolved','mixing_unresolved','iou_lower','iou_upper','mixing_lower','mixing_upper',
    'correct_resolved','wrong_resolved','unknown_iou','bpm_iou_r0','bpm_iou_r1','bpm_iou_r2','bpm_f1_r1','bpm_symmetric_f1_r1')


def estimate(rows,getter,key=''):
    source=defaultdict(list)
    for r in rows:source[r['case']['source_id']].append(float(getter(r)))
    x=np.array([np.mean(v) for _,v in sorted(source.items())]);rng=np.random.default_rng(seed_for('xxz2-bootstrap:'+key))
    means=x[rng.integers(len(x),size=(5000,len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),sources=len(x))


def aggregate(rows,key=''):
    out=dict(cases=len(rows),decisions={d:{m:estimate(rows,lambda r,d=d,m=m:r['decisions'][d][m],key+d+m) for m in METRICS} for d in DECISIONS},
        two_minus_one={m:estimate(rows,lambda r,m=m:r['decisions']['two_max'][m]-r['decisions']['one_background'][m],key+'paired'+m) for m in METRICS},
        converged=sum(r.get('converged',r.get('sampling',{}).get('converged',False)) for r in rows))
    if 'original_converged' in rows[0]:
        out['original_converged']=sum(r['original_converged'] for r in rows)
        out['batch_guard_caveat']='The extra maximum-Rhat guard on just16 batch means is a conservative flag, not a calibrated convergence test. Do not interpret its failure as proof of non-equilibration. Original full-series checks are also reported; fresh runs directly check the full paired-score series.'
    if 'sampling' in rows[0]:
        out['sampling_diagnostics']={k:dict(median=float(np.median([r['sampling'][k] for r in rows])),
            minimum=float(min(r['sampling'][k] for r in rows))) for k in ('minimum_score_ess','minimum_correlation_ess')}
        out['sampling_diagnostics']['confirmed_cases']=sum(r['signature']['stage']=='confirm' for r in rows)
        out['sampling_diagnostics']['low_score_ess_cases']=sum(r['sampling']['minimum_score_ess']<200 for r in rows)
        out['sampling_diagnostics']['mean_final_run_seconds']=float(np.mean([r['seconds'] for r in rows]))
    out['mean_bond_regime_fractions']={k:float(np.mean([r['parameters'][k] for r in rows]))
        for k in ('ferromagnetic_fraction','intermediate_fraction','antiferromagnetic_fraction')}
    exact=[r for r in rows if r['exact'] is not None]
    if exact:out['exact']={d:{m:estimate(exact,lambda r,d=d,m=m:r['exact'][d][m],key+'exact'+d+m) for m in ('iou','bpm_iou_r1','unknown_iou')} for d in DECISIONS}
    return out


def paired(a,b,da='two_max',db='two_max',key='',exact_b=False):
    lookup={r['case']['case_id']:r for r in b};assert all(r['case']['case_id'] in lookup for r in a)
    return {m:estimate(a,lambda r,m=m:r['decisions'][da][m]-lookup[r['case']['case_id']]['exact' if exact_b else 'decisions'][db][m],key+m)
        for m in ('iou','bpm_iou_r1','unknown_iou')+(() if exact_b else ('unresolved','mixing_unresolved'))}


def save(fig,name):
    p=OUT/'figures';p.mkdir(parents=True,exist_ok=True)
    # Standalone scientific figures; PNG plus editable vector SVG.
    for suffix in ('png','svg'):fig.savefig(p/f'{name}.{suffix}',dpi=175,bbox_inches='tight')
    plt.close(fig)


def plot_bpm_diagnostic():
    """Reproducible examples of the difference between area and centerlines."""
    crack=np.zeros((16,16),bool);crack[3:13,7]=True
    shifted=np.zeros_like(crack);shifted[3:13,8]=True
    filled=np.zeros_like(crack);filled[3:13,3:13]=True
    thin=skeletonize(filled,method='zhang')
    examples=[('Crack shifted by one pixel',crack,shifted),('Filled object replaced by skeleton',filled,thin)]
    fig,axes=plt.subplots(2,3,figsize=(9,6));records=[]
    for i,(name,truth,prediction) in enumerate(examples):
        metrics=dict(**mask_metrics(prediction,truth),**bpm_metrics(prediction,truth))
        records.append(dict(example=name,metrics=metrics))
        for ax,mask,label in zip(axes[i],(truth,prediction,truth.astype(int)+2*prediction.astype(int)),('Truth','Prediction','Overlap')):
            ax.imshow(mask,cmap='viridis',vmin=0,vmax=3 if label=='Overlap' else 1,interpolation='nearest')
            ax.set_title(label);ax.set_xticks([]);ax.set_yticks([])
        axes[i,0].set_ylabel(name,fontsize=9)
        axes[i,2].set_xlabel(f"IoU={metrics['iou']:.2f}; BPM-IoU(r=1)={metrics['bpm_iou_r1']:.2f}")
    fig.suptitle('BPM measures tolerant skeleton overlap; ordinary IoU measures area')
    fig.tight_layout();save(fig,'bpm-diagnostic')
    write_json(ROOT/'xxz-two-seed-bpm-diagnostic.json',dict(shape=[16,16],examples=records))


def reanalysis():
    root=OUT/'reanalysis';raw=[json.loads(p.read_text()) for p in sorted((root/'test').glob('*.json'))]
    confirms={p.name:json.loads(p.read_text()) for p in (root/'confirm').glob('*.json')}
    confirmed=[confirms.get(f'{r["case"]["case_id"]}--{r["rule_id"]}.json',r) for r in raw]
    assert len(raw)==1200
    groups={}
    for mode,records in [('primary',raw),('confirmed',confirmed)]:
        for side in (8,16):
            for fam in sorted({r['family'] for r in records}):
                rows=[r for r in records if r['case']['shape']==[side,side] and r['family']==fam]
                assert len(rows)==75;key=f'{mode}/{fam}/{side}';groups[key]=aggregate(rows,key)
    comparison={}
    for side in (8,16):
        a=[r for r in confirmed if r['case']['shape']==[side,side] and r['family']=='XXZ/edge_otsu']
        b=[r for r in confirmed if r['case']['shape']==[side,side] and r['family']=='Ising/edge_otsu']
        comparison[str(side)]=dict(sampled=paired(a,b,key='oldXXZIsing'+str(side)),versus_exact_ising=paired(a,b,key='oldexact'+str(side),exact_b=True))
    scan=[json.loads(p.read_text()) for p in sorted((root/'scan').glob('*.json'))];scan_groups={}
    for rid in sorted({r['rule_id'] for r in scan}):
        group=[r for r in scan if r['rule_id']==rid];scan_groups[rid]=dict(rule=group[0]['rule'],**aggregate(group,rid))
    small=[json.loads(p.read_text()) for p in sorted((root/'small').glob('*.json')) if json.loads(p.read_text())['case']['split']=='test']
    small_groups={f:aggregate([r for r in small if r['family']==f],'small'+f) for f in {r['family'] for r in small}}
    summary=dict(status='completed cached-state reanalysis',date='2026-10-08',groups=groups,matched_xxz_minus_ising=comparison,
        scan=scan_groups,small=small_groups,code_hashes=hashes(),substituted_confirmations=sum(r['stage']=='confirm' for r in confirmed),
        caveat='Reanalysis of old Monte Carlo states; approximate paired-batch intervals differ from previous sign-only intervals and new iid Ising intervals. Some model families fail mixing checks. Reported raw IoU is not proof of convergence.')
    write_json(ROOT/'xxz-two-seed-reanalysis-summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for model,color in [('Ising','#24678b'),('XXZ','#bd5934')]:
        for index,decision in enumerate(DECISIONS[:2]):
            for ax,metric in zip(axes,('iou','unresolved','bpm_iou_r1')):
                values=[groups[f'confirmed/{model}/edge_otsu/{side}']['decisions'][decision][metric]['mean'] for side in (8,16)]
                ax.plot([8,16],values,'-o' if index==0 else '--s',color=color,label=f'{model}, '+('one seed' if index==0 else 'two seeds'))
                ax.set_xticks([8,16],['8×8','16×16']);ax.set_ylim(0,1);ax.grid(alpha=.2)
    for ax,title in zip(axes,('IoU','Unresolved fraction','BPM IoU, radius 1')):ax.set_title(title)
    axes[0].legend(fontsize=8);fig.suptitle('Existing edge-Otsu states: same Hamiltonian, paired one/two-seed decisions\nLonger confirmations substituted; 75 sources per size')
    fig.tight_layout();save(fig,'existing-state-comparison')
    for side in (8,16):
        for model in ('Ising','XXZ'):
            g=groups[f'confirmed/{model}/edge_otsu/{side}']
            print(side,model,'passed',g['converged'],{d:{m:round(g['decisions'][d][m]['mean'],4) for m in ('iou','unresolved','bpm_iou_r1')} for d in DECISIONS[:2]},flush=True)


def final_analysis():
    selected=json.loads((ROOT/'xxz-two-seed-selection.json').read_text());rows=[];raw=[]
    for c in cases('test'):
        for rule in selected_rules():
            name=f'{c["case_id"]}--{identifier(rule)}.json';r=json.loads((OUT/'test'/name).read_text());raw.append(r)
            p=OUT/'confirm'/name
            rows.append(json.loads(p.read_text()) if p.exists() else r)
    assert len(rows)==150*len(selected_rules())
    groups={};comparison={}
    for side in (8,16):
        for label,entry in selected['winners'].items():
            group=[r for r in rows if r['case']['shape']==[side,side] and r['rule_id']==entry['rule_id']]
            groups[f'{label}/{side}']=aggregate(group,label+str(side))
        for label,entry in selected['winners'].items():
            control='matched_Ising_for_'+label
            if control not in selected['winners']:continue
            a=[r for r in rows if r['case']['shape']==[side,side] and r['rule_id']==entry['rule_id']]
            b=[r for r in rows if r['case']['shape']==[side,side] and r['rule_id']==selected['winners'][control]['rule_id']]
            comparison[f'{label}/{side}']=dict(sampled=paired(a,b,key=label+str(side)),versus_exact_ising=paired(a,b,key='exact'+label+str(side),exact_b=True))
    selection_primary={m:max([k for k in selected['winners'] if k.startswith(m+'/')],key=lambda k:selected['winners'][k]['metrics']['mixing_lower']) for m in ('Ising','XXZ')}
    overall={};recipe_comparison={}
    for side in (8,16):
        a=[r for r in rows if r['case']['shape']==[side,side] and r['rule_id']==selected['winners'][selection_primary['XXZ']]['rule_id']]
        b=[r for r in rows if r['case']['shape']==[side,side] and r['rule_id']==selected['winners'][selection_primary['Ising']]['rule_id']]
        overall[str(side)]=paired(a,b,key='overall'+str(side))
        for model in ('Ising','XXZ'):
            rid_a=selected['winners'][f'{model}/q90__two_max']['rule_id']
            rid_b=selected['winners'][f'{model}/edge_otsu__two_max']['rule_id']
            aa=[r for r in rows if r['case']['shape']==[side,side] and r['rule_id']==rid_a]
            bb=[r for r in rows if r['case']['shape']==[side,side] and r['rule_id']==rid_b]
            recipe_comparison[f'{model}/{side}']=paired(aa,bb,key='recipe'+model+str(side))
    summary=dict(date='2026-10-08',status='completed',selection=selected['winners'],primary_labels=selection_primary,
        groups=groups,matched_comparison=comparison,selected_xxz_minus_selected_ising=overall,
        selected_q90_minus_selected_edge_otsu=recipe_comparison,
        code_hashes=hashes(),test_model_cases=len(rows),unique_rules=len(selected_rules()),
        confirmation_runs=sum(r['signature']['stage']=='confirm' for r in rows),
        failed_cases=sum(not r['sampling']['converged'] for r in rows),
        low_score_ess_cases=sum(r['sampling']['minimum_score_ess']<200 for r in rows),
        confirmation_effects={label:{str(side):dict(
            cases=sum(r['signature']['stage']=='confirm' for r in rows if r['case']['shape']==[side,side] and r['rule_id']==entry['rule_id']),
            mean_two_seed_iou_change=float(np.mean([r['decisions']['two_max']['iou']-before['decisions']['two_max']['iou']
                for r,before in zip(rows,raw) if r['case']['shape']==[side,side] and r['rule_id']==entry['rule_id']])))
            for side in (8,16)} for label,entry in selected['winners'].items()},
        failed_case_details=[dict(case_id=r['case']['case_id'],rule_id=r['rule_id'],sampling=r['sampling']) for r in rows if not r['sampling']['converged']],
        all_sampling_settings={stage:sum(r['signature']['stage']==stage for r in rows) for stage in ('test','confirm')},
        statistics='5000 paired source-image bootstrap replicates, no multiplicity adjustment; two sizes not independent sources; MCMC uncertainty and image variability separate.',
        hashes={str(p.relative_to(OUT)):digest(p) for folder in ('test','confirm') for p in sorted((OUT/folder).glob('*.json'))})
    write_json(ROOT/'xxz-two-seed-summary.json',summary)
    plot_final(summary,selected,rows)
    plot_bpm_diagnostic()
    plot_examples(summary,selected,rows)
    for side in (8,16):
        for label in selected['winners']:
            g=groups[f'{label}/{side}'];d=selected['winners'][label]['decision']
            print(side,label,'pass',g['converged'],{m:round(g['decisions'][d][m]['mean'],4) for m in ('iou','unresolved','bpm_iou_r1')},flush=True)


def plot_examples(summary,selected,rows):
    """Fixed lexical examples, never selected by accuracy."""
    chosen=sorted([c for c in cases('test') if c['shape']==[16,16]],key=lambda c:c['source_id'])[:4]
    lookup={(r['case']['case_id'],r['rule_id']):r for r in rows}
    ids={m:selected['winners'][summary['primary_labels'][m]]['rule_id'] for m in ('Ising','XXZ')}
    fig,axes=plt.subplots(len(chosen),6,figsize=(14,10))
    for i,c in enumerate(chosen):
        data=np.load(INPUT/'cases'/f'{c["case_id"]}.npz');arrays={};records={}
        for m in ('Ising','XXZ'):
            record=lookup[(c['case_id'],ids[m])];records[m]=record
            arrays[m]=np.load(OUT/record['signature']['stage']/f'{c["case_id"]}--{ids[m]}.npz')
        xx=arrays['XXZ'];ii=arrays['Ising'];shape=(16,16)
        masks=[data['image'],data['truth'],ii['score'][1].reshape(shape)>1e-12,
            xx['score'][0].reshape(shape)>1e-12,xx['score'][1].reshape(shape)>1e-12,
            (abs(xx['score'][1])<=np.maximum(xx['score_radius'][1],1e-12)).reshape(shape)]
        for j,(ax,mask) in enumerate(zip(axes[i],masks)):
            ax.imshow(mask,cmap='gray' if j<5 else 'Reds',interpolation='nearest',vmin=0,vmax=255 if j==0 else 1)
            ax.set_xticks([]);ax.set_yticks([])
        for seed,color,marker in zip(data['seeds'],('cyan','magenta'),('x','+')):
            y,x=divmod(int(seed),16);axes[i,0].scatter([x],[y],c=color,marker=marker,s=45)
        axes[i,0].set_ylabel(c['source_id'],fontsize=8)
        for j,m,d in ((2,'Ising','two_max'),(3,'XXZ','one_background'),(4,'XXZ','two_max')):
            v=records[m]['decisions'][d]
            axes[i,j].set_xlabel(f"IoU {v['iou']:.3f}; BPM {v['bpm_iou_r1']:.3f}",fontsize=8)
        axes[i,5].set_xlabel(f"Unresolved {100*records['XXZ']['decisions']['two_max']['unresolved']:.1f}%",fontsize=8)
    for ax,title in zip(axes[0],('Input + seeds','Truth','Ising: two seeds','XXZ: one seed','XXZ: two seeds','XXZ: unresolved')):ax.set_title(title,fontsize=10)
    fig.suptitle('First four test sources by ID at 16×16; calibration-selected models\nSeed markers: background cyan ×, foreground magenta +; BPM radius 1')
    fig.tight_layout();save(fig,'segmentation-examples')


def plot_final(summary,selected,rows):
    ks=[.4,.8,1.2,1.6];gs=[0.,.1,.25,.5,.75]
    metric_titles={'iou':'IoU','unresolved':'Unresolved fraction','bpm_iou_r1':'BPM IoU (r=1)'}
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,recipe in enumerate(('edge_otsu','q90')):
        for j,metric in enumerate(('iou','unresolved','bpm_iou_r1')):
            mat=np.empty((len(ks),len(gs)))
            for a,k in enumerate(ks):
                for b,g in enumerate(gs):
                    r=next(r for r in selected['scores'] if r['decision']=='two_max' and r['rule']['threshold']==recipe and r['rule']['K']==k and r['rule']['strength']==g)
                    mat[a,b]=r['metrics'][metric]
            ax=axes[i,j];im=ax.imshow(mat,vmin=0,vmax=1,cmap='magma' if metric=='unresolved' else 'viridis',aspect='auto')
            ax.set_xticks(range(len(gs)),gs);ax.set_yticks(range(len(ks)),ks);ax.set_xlabel('g / median |Delta|');ax.set_ylabel('K');ax.set_title(recipe.replace('edge_otsu','Edge Otsu')+': '+metric_titles[metric])
            for a in range(len(ks)):
                for b in range(len(gs)):ax.text(b,a,f'{mat[a,b]:.2f}',color='white',ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    fig.suptitle('Calibration sweep: two-seed ZZ, 25 sources at both resolutions\nBPM is evaluated, not used to select parameters')
    fig.tight_layout();save(fig,'calibration-sweep')
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for i,recipe in enumerate(('edge_otsu','q90')):
        for j,metric in enumerate(('converged','mixing_lower')):
            mat=np.empty((len(ks),len(gs)))
            for a,k in enumerate(ks):
                for b,g in enumerate(gs):
                    r=next(r for r in selected['scores'] if r['decision']=='two_max' and r['rule']['threshold']==recipe and r['rule']['K']==k and r['rule']['strength']==g)
                    mat[a,b]=r['converged']/r['cases'] if metric=='converged' else r['metrics'][metric]
            ax=axes[i,j];im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
            ax.set_xticks(range(len(gs)),gs);ax.set_yticks(range(len(ks)),ks);ax.set_xlabel('g / median |Delta|');ax.set_ylabel('K')
            ax.set_title(recipe.replace('edge_otsu','Edge Otsu')+': '+('convergence pass fraction' if j==0 else 'mixing-aware lower IoU'))
            for a in range(len(ks)):
                for b in range(len(gs)):ax.text(b,a,f'{mat[a,b]:.2f}',color='white' if mat[a,b]<.65 else 'black',ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    fig.suptitle('Calibration reliability: poorer scores at large K mix numerical and physical effects')
    fig.tight_layout();save(fig,'calibration-convergence')
    labels=[summary['primary_labels'][m] for m in ('Ising','XXZ')]
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i,side in enumerate((8,16)):
        for j,metric in enumerate(('iou','unresolved','bpm_iou_r1')):
            ax=axes[i,j];values=[];ticks=[]
            for label in labels:
                for d in DECISIONS[:2]:
                    values.append(summary['groups'][f'{label}/{side}']['decisions'][d][metric]['mean'])
                    ticks.append(label.split('/')[0]+'\n'+('one seed' if d==DECISIONS[0] else 'two seeds'))
            ax.bar(range(4),values,color=['#92b5ce','#24678b','#e5b698','#bd5934']);ax.set_xticks(range(4),ticks,fontsize=8);ax.set_ylim(0,1);ax.set_title(f'{side}×{side}: {metric_titles[metric]}')
            for k,v in enumerate(values):ax.text(k,v+.02,f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Calibration-selected Ising and XXZ models on 75 test sources\nWithin each model, readouts use identical Gibbs estimates')
    fig.tight_layout();save(fig,'selected-models')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,side in zip(axes,(8,16)):
        for label in labels:
            g=summary['groups'][f'{label}/{side}']['decisions']['two_max']
            ax.plot([0,1,2],[g[f'bpm_iou_r{r}']['mean'] for r in (0,1,2)],'-o',label=label.split('/')[0])
        ax.set_xticks([0,1,2]);ax.set_ylim(0,1);ax.set_xlabel('Euclidean skeleton tolerance (pixels)');ax.set_ylabel('BPM IoU');ax.set_title(f'{side}×{side}');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Sensitivity of BPM to tolerance; radius zero is skeleton IoU')
    fig.tight_layout();save(fig,'bpm-tolerance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reanalysis','final']);a=p.parse_args()
    if a.stage=='reanalysis':reanalysis()
    else:final_analysis()
