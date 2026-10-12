"""Analyze the frozen threshold experiment; never change its selected settings."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quantum-segmentation-matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_emnist_ising import bootstrap, otsu, segmentation_iou
from run_emnist_thresholds import ROOT, OUTPUT, load_runs
from ising_thresholds import RULES, K_GRID, K_REFERENCE, rule_name

LABELS={'baseline':'Fixed 125 (tuned K)','original':'Fixed 125 (original K)',
        'global':'Option 1: fixed threshold','quantile':'Option 2: edge quantile',
        'otsu':'Option 2: edge Otsu'}
COLORS={'baseline':'#8a8a8a','original':'#bbbbbb','global':'#0072b2','quantile':'#e69f00','otsu':'#009e73'}


def save(fig,output,name):
    fig.savefig(output/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/f'{name}.svg',bbox_inches='tight');plt.close(fig)


def training_plot(output):
    runs=load_runs(output,'train');rows=[r for m in runs for r in m['records']]
    matrix=np.array([[np.mean([r['iou'] for r in rows if r['rule']==rule_name(rule) and r['k']==k])
                       for k in K_GRID] for rule in RULES])
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_xticks(range(len(K_GRID)),[f'{k:.3f}' for k in K_GRID])
    ax.set_yticks(range(len(RULES)),[rule_name(r) for r in RULES])
    ax.set_xlabel(r'Typical thermal coupling $K=b\,\mathrm{median}|\Delta|/4$')
    ax.set_title('Training only: mean foreground IoU\n47 native 28×28 images; intensity ≥128 proxy')
    for (i,j),value in np.ndenumerate(matrix):
        ax.text(j,i,f'{value:.3f}',ha='center',va='center',color='black' if value>.65 else 'white')
    fig.colorbar(im,ax=ax,label='Mean IoU');fig.tight_layout();save(fig,output,'threshold-training')


def analyze(output):
    frozen=json.loads((output/'frozen-selection.json').read_text())
    manifest=json.loads((output/'manifest.json').read_text());images=np.load(output/'subset.npz')['images']
    imdict={c['name']:im for c,im in zip(manifest['cases'],images)}
    cases={c['name']:c for c in manifest['cases']};names=sorted(n for n,c in cases.items() if c['split']=='test')
    runs=load_runs(output,'test');table={(r['name'],r['rule'],r['k']):r for m in runs for r in m['records']}
    byjob={(m['case']['name'],rule_name(m['signature']['rule'])):m for m in runs}
    selections=dict(frozen['selections'])
    selections['original']=dict(rule=dict(kind='fixed',value=125.),k=float(K_REFERENCE))
    methods={};records={};pairwise={}
    metrics=['iou','balanced_accuracy','accuracy','foreground_recall','foreground_precision',
             'resolved_fraction','confident_wrong_fraction','ink_seed_iou','seed_prediction_agreement',
             'iou_threshold96','iou_threshold160']
    for method,s in selections.items():
        selected=[table[n,rule_name(s['rule']),s['k']] for n in names]
        records[method]=selected
        jobs=[byjob[n,rule_name(s['rule'])] for n in names]
        methods[method]=dict(selection=s,**{key:bootstrap([r[key] for r in selected]) for key in metrics},
            tau_range=[min(r['tau'] for r in selected),max(r['tau'] for r in selected)],
            tau_median=float(np.median([r['tau'] for r in selected])),
            b_range=[min(r['b'] for r in selected),max(r['b'] for r in selected)],
            b_median=float(np.median([r['b'] for r in selected])),
            rhat_max=max(r['rhat_observable_max'] for r in selected),
            gamma_rhat_p99_max=max(r['rhat_gamma_p99'] for r in selected),
            ess_mean_gamma_min=min(r['ess_mean_gamma'] for r in selected),
            observable_rhat_over1_05=sum(r['rhat_observable_max']>1.05 for r in selected),
            iou_at_least_09=sum(r['iou']>=.9 for r in selected),iou_below_05=sum(r['iou']<.5 for r in selected),
            features={k:float(np.mean([m['features'][k] for m in jobs])) for k in
                      ['negative_edge_fraction','frustrated_plaquette_fraction','boundary_antiferromagnetic_fraction',
                       'within_ink_negative_fraction','zero_edge_fraction']},
            worst=sorted(selected,key=lambda r:r['iou'])[:5])
    for method in ['global','quantile','otsu','original']:
        difference=np.array([r['iou'] for r in records[method]])-np.array([r['iou'] for r in records['baseline']])
        pairwise[f'{method}_minus_baseline']=dict(bootstrap(difference),
            wins=int(np.sum(difference>0)),ties=int(np.sum(difference==0)),losses=int(np.sum(difference<0)))
    for a,b in [('quantile','global'),('otsu','global'),('otsu','quantile')]:
        pairwise[f'{a}_minus_{b}']=bootstrap([x['iou']-y['iou'] for x,y in zip(records[a],records[b])])
    pixel_otsu={str(t):bootstrap([segmentation_iou(otsu(imdict[n])[0],imdict[n]>=t) for n in names])
                for t in [96,128,160]}
    train=load_runs(output,'train');trainrows=[r for m in train for r in m['records']]
    sampler=dict(train_jobs=len(train),train_conditions=len(trainrows),test_jobs=len(runs),
        test_conditions=len(table),test_measurements_per_condition=8192,
        train_max_rhat=max(r['rhat_observable_max'] for r in trainrows),
        train_rhat_over1_05=sum(r['rhat_observable_max']>1.05 for r in trainrows),
        test_max_rhat=max(r['rhat_observable_max'] for r in table.values()),
        test_gamma_rhat_max=max(r['rhat_gamma_max'] for r in table.values()),
        test_rhat_over1_05=sum(r['rhat_observable_max']>1.05 for r in table.values()),
        min_swap_acceptance=min(m['swap_min'] for m in runs),round_trips=sum(m['round_trips'] for m in runs),
        jobs_without_round_trip=sum(m['round_trips']==0 for m in runs))
    summary=dict(frozen_selection=frozen,manifest=manifest,methods=methods,pairwise=pairwise,
        pixel_otsu_baseline=pixel_otsu,sampler=sampler,records=records,
        limitations=['EMNIST has character labels, not segmentation annotations; masks are intensity proxies.',
                     '47 fresh test images, one per class; bootstrap intervals reflect this small sample.',
                     'Diagnostics do not prove equilibration or guarantee accurate signs near zero.',
                     'Only alpha=.001 and K=.5,.6217749727,.8 were tested in this threshold experiment.',
                     'No quantum circuit was executed; this study uses classical sampling of a diagonal Hamiltonian.'],
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        test_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output/'test').glob('*.json'))})
    validation=ROOT/'emnist-thresholds-validation.json'
    if validation.exists():summary['exact_validation']=json.loads(validation.read_text())
    for key,name in [('training_audit','emnist-thresholds-training-audit.json'),
                     ('longer_confirmation','emnist-thresholds-confirmation.json')]:
        if (ROOT/name).exists():summary[key]=json.loads((ROOT/name).read_text())
    (ROOT/'emnist-thresholds-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    training_plot(output)
    plot_comparison(output,methods,records,pairwise,pixel_otsu)
    plot_examples(output,names,imdict,cases,records,selections)
    print(json.dumps(dict(methods={m:{k:v for k,v in x.items() if k in ['selection','iou','resolved_fraction','rhat_max','tau_range','b_range']}
                                  for m,x in methods.items()},pairwise=pairwise,sampler=sampler),indent=2))


def plot_comparison(output,methods,records,pairwise,pixel_otsu):
    keys=['original','baseline','global','quantile','otsu']
    if (methods['baseline']['selection']['rule']==methods['original']['selection']['rule'] and
        methods['baseline']['selection']['k']==methods['original']['selection']['k']):
        keys.remove('original')
    fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1.05,1]})
    ax=axes[0]
    for i,k in enumerate(keys):
        m=methods[k]['iou'];low,high=m['ci95']
        ax.errorbar(m['mean'],i,xerr=[[m['mean']-low],[high-m['mean']]],fmt='o',color=COLORS[k],capsize=4)
        ax.text(m['mean']+.012,i+.17,f"{m['mean']:.3f}",fontsize=9)
    m=pixel_otsu['128'];ax.errorbar(m['mean'],len(keys),xerr=[[m['mean']-m['ci95'][0]],[m['ci95'][1]-m['mean']]],fmt='s',color='#333333',capsize=4)
    ax.set_yticks(range(len(keys)+1),[LABELS[k] for k in keys]+['Pixel Otsu baseline'])
    ax.invert_yaxis();ax.set_xlim(0,1.02);ax.set_xlabel('Mean IoU and image-bootstrap 95% interval')
    ax.set_title('Fresh test images: 47 classes, 47 images');ax.grid(axis='x',alpha=.2)
    ax=axes[1];keys=['global','quantile','otsu']
    for i,k in enumerate(keys):
        d=np.array([r['iou'] for r in records[k]])-np.array([r['iou'] for r in records['baseline']])
        jitter=np.linspace(-.16,.16,len(d))
        ax.scatter(d,i+jitter,s=12,alpha=.45,color=COLORS[k])
        m=pairwise[f'{k}_minus_baseline'];lo,hi=m['ci95']
        ax.errorbar(m['mean'],i,xerr=[[m['mean']-lo],[hi-m['mean']]],fmt='D',color='black',capsize=5)
    ax.axvline(0,color='#555555',lw=1);ax.set_yticks(range(3),['Fixed threshold','Edge quantile','Edge Otsu'])
    ax.invert_yaxis();ax.set_xlabel('Paired IoU change from fixed 125, tuned K')
    ax.set_title('Dots: individual images; black: mean and 95% interval');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Threshold calibration with positive sensitivity α=0.001\nIntensity ≥128 is an evaluation proxy, not a human segmentation annotation',fontsize=12)
    fig.tight_layout();save(fig,output,'threshold-comparison')


def plot_examples(output,names,images,cases,records,selections):
    bymethod={m:{r['name']:r for r in rows} for m,rows in records.items()}
    # Transparent illustrative selection: largest gains and losses of the
    # training-selected global threshold relative to 125, not handpicked images.
    ranked=sorted(names,key=lambda n:bymethod['global'][n]['iou']-bymethod['baseline'][n]['iou'])
    selected=ranked[-3:][::-1]+ranked[:3]
    keys=['baseline','global','quantile','otsu']
    fig,axes=plt.subplots(len(selected),6,figsize=(12,12))
    for row,n in enumerate(selected):
        image=images[n];axes[row,0].imshow(image,cmap='gray',vmin=0,vmax=255)
        axes[row,0].set_title(f"Class {cases[n]['class_id']}: {cases[n]['character']}",fontsize=10)
        axes[row,1].imshow(image>=128,cmap='gray',vmin=0,vmax=1);axes[row,1].set_title('Proxy ≥128',fontsize=10)
        for col,key in enumerate(keys,2):
            s=selections[key];name=rule_name(s['rule']);meta=json.loads((output/'test'/f'{n}-{name}.json').read_text())
            t=meta['signature']['ks'].index(s['k']);data=np.load(output/'test'/f'{n}-{name}.npz')
            gamma=data['gamma'][t,0].reshape(image.shape);prediction=gamma<=0
            axes[row,col].imshow(prediction,cmap='gray',vmin=0,vmax=1)
            axes[row,col].set_title(f"{key}: IoU {bymethod[key][n]['iou']:.2f}\nτ={bymethod[key][n]['tau']:g}",fontsize=9)
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Three largest gains (top) and losses (bottom) for the selected global threshold\nAll masks are obtained from the sign of the background-seed Gibbs correlation',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,output,'threshold-examples')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=__import__('pathlib').Path,default=OUTPUT)
    p.add_argument('--training-only',action='store_true');args=p.parse_args()
    if args.training_only:training_plot(args.output)
    else:analyze(args.output)
