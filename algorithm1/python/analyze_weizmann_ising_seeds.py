"""Source-level analysis and figures for exact-sampled one/two-seed Ising."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ising-seeds-matplotlib')
from collections import defaultdict
from functools import lru_cache
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from run_weizmann_ising_seeds import ROOT,OUT,FUNCTIONS,DECISIONS,METRICS,BUDGETS,PRIMARY,manifest,hashes,chosen_rules
from weizmann_petz_data import write_json,seed_for

SHORT={'arctan':'Arctan','arctan_quadratic':'Delta + Delta|Delta|','arctan_cubic':'Delta + 4 Delta³','linear':'Linear','linear_cubic':'x + x³'}


@lru_cache(maxsize=1)
def selection():return json.loads((ROOT/'weizmann-ising-seeds-selection.json').read_text())['winners']


def load_results():
    results=[];selected=json.loads(json.dumps(chosen_rules()))
    for c in manifest()['cases']:
        if c['split']!='test':continue
        data=json.loads((OUT/'test'/f'{c["case_id"]}.json').read_text())
        assert data['signature']==dict(case=c,code_hashes=hashes(),selected=selected,budgets=BUDGETS,replicates=8)
        results.append(data)
    return results


def record(data,label):
    return next(r for r in data['records'] if label in r['selected_as'])


def value(data,label,metric,decision=None,budget=PRIMARY,exact=False):
    d=decision or selection()[label]['decision'];r=record(data,label)['decisions'][d]
    return r['exact'][metric] if exact else r['budgets'][str(budget)]['mean'][metric]


def ci(v,key):
    v=np.asarray(v);rng=np.random.default_rng(seed_for('ising-seeds-bootstrap:'+key))
    means=v[rng.integers(len(v),size=(5000,len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(v))


def aggregate(records,side):
    selected=selection();summary=dict(sources=len(records),models={},matched_readouts={},paired={},baselines={})
    for label,entry in selected.items():
        r=dict(exact={m:float(np.mean([value(d,label,m,exact=True) for d in records])) for m in METRICS},
            budgets={str(b):{m:float(np.mean([value(d,label,m,budget=b) for d in records])) for m in METRICS} for b in BUDGETS})
        r['iou_interval']=ci([value(d,label,'iou') for d in records],str(side)+label)
        r['mean_seed_disagreement']=float(np.mean([record(d,label)['exact_seed_disagreement'] for d in records]))
        summary['models'][label]=r
    for fn in FUNCTIONS:
        label=fn+'__two_max';summary['matched_readouts'][fn]={}
        for decision in DECISIONS:
            summary['matched_readouts'][fn][decision]={m:float(np.mean([value(d,label,m,decision=decision) for d in records])) for m in METRICS}
        comparisons=[('separately_calibrated',label,fn+'__one_background','two_max','one_background'),
            ('same_H_background',label,label,'two_max','one_background'),
            ('same_H_foreground',label,label,'two_max','one_foreground'),
            ('matched_function','matched__'+fn,'matched__arctan','two_max','two_max'),
            ('optimized_two_vs_arctan',label,'arctan__two_max','two_max','two_max')]
        for tag,a,b,da,db in comparisons:
            summary['paired'][fn+'__'+tag]={m:ci([value(d,a,m,da)-value(d,b,m,db) for d in records],str(side)+fn+tag+m)
                for m in ('iou','unresolved','correct_resolved','wrong_resolved','unknown_iou')}
    summary['matched_readouts']['legacy_arctan']={decision:{m:float(np.mean([value(d,'legacy_arctan',m,decision=decision) for d in records])) for m in METRICS} for decision in DECISIONS}
    for tag,a,b,da,db in [
        ('legacy_two_minus_one','legacy_arctan','legacy_arctan','two_max','one_background'),
        ('adaptive_two_minus_legacy_one','arctan__two_max','legacy_arctan','two_max','one_background'),
        ('mild_two_minus_legacy_one','matched__arctan_quadratic','legacy_arctan','two_max','one_background')]:
        summary['paired'][tag]={m:ci([value(d,a,m,da)-value(d,b,m,db) for d in records],str(side)+tag+m)
            for m in ('iou','unresolved','correct_resolved','wrong_resolved','unknown_iou')}
    primary={d:max([k for k in selected if k.endswith('__'+d) and not k.startswith('robust')],key=lambda k:selected[k]['sample']['iou']) for d in DECISIONS[:2]}
    summary['paired']['overall_two_minus_one']={m:ci([value(d,primary['two_max'],m)-value(d,primary['one_background'],m) for d in records],str(side)+'overall'+m)
        for m in ('iou','unresolved','correct_resolved','wrong_resolved','unknown_iou')}
    for name in records[0]['baselines']:
        summary['baselines'][name]={m:float(np.mean([r['baselines'][name][m] for r in records])) for m in ('iou','dice','accuracy')}
    # Sensitivity to using generic Hoeffding bounds instead of exact binomial winner tests.
    summary['uncertainty_comparison']={}
    counts_files={r['case']['case_id']:np.load(OUT/'test'/f'{r["case"]["case_id"]}.npz') for r in records}
    for label in ['arctan__two_max','arctan_cubic__two_max','robust__two_max']:
        rates={d:[] for d in DECISIONS[:2]}
        for r in records:
            rec=record(r,label);n=side*side;counts=counts_files[r['case']['case_id']][rec['rule_id']+'_counts'][:,-1]
            keep=np.ones(n,bool);keep[r['case']['seeds']]=False
            cb=1-2*counts[:,0]/PRIMARY;contrast=2*(2*counts[:,1]-counts[:,3])/PRIMARY
            radius=np.sqrt(2*np.log(2*9*n/.05)/PRIMARY)
            rates['one_background'].append(float(np.mean(abs(cb[:,keep])<=radius)))
            rates['two_max'].append(float(np.mean(abs(contrast[:,keep])<=2*radius)))
        summary['uncertainty_comparison'][label]={d:dict(binomial=float(np.mean([value(r,label,'unresolved',decision=d) for r in records])),
            hoeffding=float(np.mean(rates[d]))) for d in DECISIONS[:2]}
    return summary


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(folder/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plots(summary,records):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey='row')
    names=['Original\none seed','Original\ntwo seeds','Adaptive\ntwo seeds','Adaptive + mild\ntwo seeds']
    for col,side in enumerate((8,16)):
        group=summary['sizes'][str(side)];models=[group['matched_readouts']['legacy_arctan'][d] for d in DECISIONS[:2]]
        models += [group['models'][k]['budgets'][str(PRIMARY)] for k in ('arctan__two_max','matched__arctan_quadratic')]
        for row,metric in enumerate(('iou','unresolved')):
            vals=[m[metric] for m in models];ax=axes[row,col]
            ax.bar(np.arange(4),vals,color=['#6c7786','#518f9c','#157d9a','#be6b28'])
            ax.set_xticks(range(4),names,fontsize=9);ax.set_ylim(0,.6 if row==0 else .08);ax.grid(axis='y',alpha=.2)
            for i,v in enumerate(vals):ax.text(i,v+.009 if row==0 else v+.001,f'{v:.3f}' if row==0 else f'{100*v:.1f}%',ha='center',fontsize=10)
        axes[0,col].set_title(f'{side}×{side} — 75 test sources')
    axes[0,0].set_ylabel('Foreground IoU');axes[1,0].set_ylabel('Unresolved fraction')
    fig.suptitle('Adding a foreground seed and changing the image parameter rule\n24,576 independent Gibbs copies; mean of eight replicates',fontsize=12)
    fig.tight_layout();save(fig,'baseline-comparison')
    x=np.arange(len(FUNCTIONS));width=.36
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey='row')
    for col,side in enumerate((8,16)):
        group=summary['sizes'][str(side)]
        for index,decision in enumerate(DECISIONS[:2]):
            for row,metric in enumerate(('iou','unresolved')):
                vals=[group['models'][fn+'__'+decision]['budgets'][str(PRIMARY)][metric] for fn in FUNCTIONS]
                axes[row,col].bar(x+width*(index-.5),vals,width,label='One background seed' if index==0 else 'Two seeds: max correlation',color=['#6c7786','#157d9a'][index])
        for ax in axes[:,col]:
            ax.set_xticks(x,[SHORT[f] for f in FUNCTIONS],rotation=30,ha='right',fontsize=9);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
        axes[0,col].set_title(f'{side}×{side} full images — 75 test sources')
    axes[0,0].set_ylabel('Foreground IoU');axes[1,0].set_ylabel('Unresolved fraction, excluding seeds');axes[0,1].legend(fontsize=9)
    fig.suptitle('One-seed versus two-seed Ising: separately calibrated parameter rules\n24,576 iid Gibbs samples; mean of eight replicates; simultaneous binomial winner intervals',y=1.01)
    fig.tight_layout();save(fig,'one-versus-two')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey='row')
    for col,side in enumerate((8,16)):
        for fn in FUNCTIONS:
            model=summary['sizes'][str(side)]['models']['matched__'+fn]
            for row,metric in enumerate(('iou','unresolved')):
                axes[row,col].plot(BUDGETS,[model['budgets'][str(b)][metric] for b in BUDGETS],'-o',label=SHORT[fn])
        for ax in axes[:,col]:
            ax.set_xscale('log');ax.set_xticks(BUDGETS,[f'{b:,}' for b in BUDGETS]);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.2)
        axes[0,col].set_ylim(.35,.55);axes[1,col].set_ylim(0,.22)
        axes[0,col].set_title(f'{side}×{side}');axes[1,col].set_xlabel('Independent Gibbs copies')
    axes[0,0].set_ylabel('Foreground IoU');axes[1,0].set_ylabel('Unresolved fraction');axes[0,1].legend(fontsize=9)
    fig.suptitle('Matched coupling-function controls, all using two seeds\nSame threshold and median interaction scale as the arctan calibration winner')
    fig.tight_layout();save(fig,'matched-functions')
    scores=json.loads((ROOT/'weizmann-ising-seeds-calibration.json').read_text())['scores']
    thresholds=[]
    for r in scores:
        key=json.dumps(r['rule']['threshold'],sort_keys=True)
        if key not in thresholds:thresholds.append(key)
    labels=['25','50','75','100','125','q50+','q75+','q90+','Otsu']
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,decision in zip(axes,DECISIONS[:2]):
        matrix=np.zeros((len(FUNCTIONS),len(thresholds)))
        for i,fn in enumerate(FUNCTIONS):
            for j,t in enumerate(thresholds):
                rows=[r for r in scores if r['decision']==decision and r['rule']['function']==fn and json.dumps(r['rule']['threshold'],sort_keys=True)==t]
                matrix[i,j]=max(r['sample']['iou'] for r in rows)
        im=ax.imshow(matrix,vmin=0,vmax=.5,cmap='viridis',aspect='auto')
        ax.set_xticks(range(9),labels);ax.set_yticks(range(5),[SHORT[f] for f in FUNCTIONS]);ax.set_title('One background seed' if decision==DECISIONS[0] else 'Two seeds: max correlation');ax.set_xlabel('Threshold rule')
        for i in range(5):
            for j in range(9):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white',fontsize=8)
    fig.suptitle('Threshold scan on 25 calibration sources at 8×8\nEach cell maximizes over sensitivity and temperature; these are calibration scores')
    fig.subplots_adjust(top=.76,bottom=.15,left=.15,right=.87,wspace=.5)
    fig.colorbar(im,cax=fig.add_axes([.9,.18,.018,.55]),label='Calibration IoU');save(fig,'threshold-calibration')
    xx=np.linspace(-3,3,500);delta=np.arctan(xx)
    funcs=[delta,delta+delta*abs(delta),delta+4*delta**3,xx,xx+xx**3]
    at1=np.arctan(1);norm=[at1,at1+at1**2,at1+4*at1**3,1,2]
    fig,ax=plt.subplots(figsize=(7,4))
    for fn,y,den in zip(FUNCTIONS,funcs,norm):ax.plot(xx,y/den,label=SHORT[fn])
    ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set_yscale('symlog',linthresh=1)
    ax.set(xlabel='x = alpha (threshold − color difference)',ylabel='J(x) / J(1)',title='Coupling shapes: equal scale at x = 1');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();save(fig,'coupling-functions')
    primary=summary['primary_by_calibration']['two_max'];rows=[r for r in records if r['case']['side']==16]
    rows.sort(key=lambda r:value(r,primary,'iou')-value(r,primary,'iou',decision='one_background'))
    examples=[rows[0],rows[len(rows)//2],rows[-1]]
    fig,axes=plt.subplots(3,5,figsize=(11,7))
    from ising_seed_sampling import classify_counts
    for ri,data in enumerate(examples):
        case=data['case'];inputs=np.load(OUT/'cases'/f'{case["case_id"]}.npz');r=record(data,primary)
        counts=np.load(OUT/'test'/f'{case["case_id"]}.npz')[r['rule_id']+'_counts'][0,-1]
        pictures=[inputs['image'],inputs['truth']]
        for decision in DECISIONS[:2]:
            pred,resolved=classify_counts(counts,PRIMARY,decision);pixels=pred.astype(float);pixels[~resolved]=.5;pictures.append(pixels.reshape(16,16))
        c=np.load(OUT/'test'/f'{case["case_id"]}.npz')[r['rule_id']+'_correlations'];pictures.append((c[1]-c[0]).reshape(16,16))
        for col,pixels in enumerate(pictures):
            im=axes[ri,col].imshow(pixels,cmap='coolwarm' if col==4 else 'gray',vmin=-2 if col==4 else 0,vmax=2 if col==4 else (255 if col==0 else 1))
            axes[ri,col].set_xticks([]);axes[ri,col].set_yticks([])
            if ri==0:axes[ri,col].set_title(['Image','Human majority','One seed\nsame Hamiltonian','Two seeds\nsame Hamiltonian','Exact correlation\ndifference'][col],fontsize=9)
            if col==4:fig.colorbar(im,ax=axes[ri,col],fraction=.046,pad=.04)
        axes[ri,0].set_ylabel(case['source_id'],fontsize=8)
        for k,seed in enumerate(inputs['seeds']):
            y,z=divmod(int(seed),16);axes[ri,1].text(z,y,'B' if k==0 else 'F',ha='center',va='center',color='#ed7600',weight='bold',fontsize=8)
    fig.suptitle('Illustrative worst / median / best two-minus-one cases for the primary two-seed model\nFirst sampling replicate; gray masks indicate unresolved pixels. Chosen after testing for illustration only.')
    fig.tight_layout();save(fig,'examples')


def main():
    records=load_results();selected=selection()
    primary={d:max([k for k in selected if k.endswith('__'+d) and not k.startswith('robust')],key=lambda k:selected[k]['sample']['iou']) for d in DECISIONS[:2]}
    summary=dict(date='2026-10-08',selection=selected,primary_by_calibration=primary,
        sizes={str(side):aggregate([r for r in records if r['case']['side']==side],side) for side in (8,16)},
        code_hashes=hashes(),case_count=len(records),test_sources=len({r['case']['source_id'] for r in records}),
        statistics='5000 paired bootstrap resamples of original source images; eight sampling replicates averaged first. Exploratory 95 percent intervals without multiplicity adjustment.',
        uncertainty='Binomial winner intervals differ from previous generic Hoeffding intervals. Both reported as sensitivity analysis; no direct cross-study comparison of unresolved rates.',
        timing={str(side):float(np.median([r['seconds'] for r in records if r['case']['side']==side])) for side in (8,16)})
    write_json(ROOT/'weizmann-ising-seeds-summary.json',summary);plots(summary,records)
    print('Primary choices:',primary)
    for side in (8,16):
        print('SIDE',side)
        for label in [f+'__'+d for f in FUNCTIONS for d in DECISIONS[:2]]+['robust__one_background','robust__two_max']:
            r=summary['sizes'][str(side)]['models'][label];m=r['budgets'][str(PRIMARY)]
            print(label,'IoU',round(m['iou'],4),'exact',round(r['exact']['iou'],4),'unresolved',round(m['unresolved'],4))


if __name__=='__main__':main()
