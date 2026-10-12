"""Aggregate frozen XYZ experiments by original source and make research figures."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/xyz-matplotlib')
from collections import defaultdict
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_weizmann_xyz import ROOT,OUT,manifest,hashes,BUDGETS,PRIMARY_BUDGET,METRICS,entries
from xyz_alignment import decision_score,uncertainty_radius
from weizmann_petz_data import write_json,seed_for

MAIN=['Ising_zero','Ising_flexible','XYZ_tied_flexible','XYZ_staggered_zero','XYZ_staggered_flexible']
LABELS={'Ising_zero':'Ising: sign','Ising_flexible':'Ising: selected rule','XYZ_tied_flexible':'XYZ: tied thresholds',
'XYZ_staggered_zero':'XYZ: staggered, sign','XYZ_staggered_flexible':'XYZ: staggered, selected rule',
'matched_Jz_only':'Only ZZ, same temperature','collapsed_same_b':'Collapsed thresholds, same temperature',
'collapsed_matched_scale':'Collapsed thresholds, matched scale','matched_XYZ_ZZ':'Same XYZ state, ZZ readout'}
COLORS=['#586579','#2171b5','#a98bc7','#e49543','#17856e']


def load_cases():
    expected=[c for c in manifest()['cases'] if c['split']=='test']
    records=[]
    for c in expected:
        p=OUT/'test'/f'{c["case_id"]}.json'
        data=json.loads(p.read_text());s=data['signature']
        assert s['code_hashes']==hashes() and s['selection']==entries(c['shape']==[3,4])
        assert s['case']==c and s['budgets']==BUDGETS
        assert p.with_suffix('.npz').exists()
        records.append(data)
    return records


def case_value(data,label,metric,budget=PRIMARY_BUDGET,exact=False):
    row=next(r for r in data['results'] if r['label']==label)
    return row['exact'][metric] if exact else row['budgets'][str(budget)]['mean'][metric]


def source_average(records,fn):
    grouped=defaultdict(list)
    for r in records:grouped[r['case']['source_id']].append(fn(r))
    return np.array([np.mean(grouped[k],axis=0) for k in sorted(grouped)])


def interval(values,key):
    values=np.asarray(values);rng=np.random.default_rng(seed_for('xyz-bootstrap:'+key))
    means=values[rng.integers(len(values),size=(5000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),sources=len(values))


def aggregate(records,name):
    common=set.intersection(*[set(r['label'] for r in d['results']) for d in records])
    result=dict(cases=len(records),sources=len({r['case']['source_id'] for r in records}),models={},baselines={},paired={})
    for label in sorted(common):
        exact={m:float(source_average(records,lambda r:case_value(r,label,m,exact=True)).mean()) for m in METRICS}
        budget_metrics={}
        for b in BUDGETS:
            budget_metrics[str(b)]={m:float(source_average(records,lambda r:case_value(r,label,m,b)).mean()) for m in METRICS}
        result['models'][label]=dict(exact=exact,budgets=budget_metrics,
            iou_interval=interval(source_average(records,lambda r:case_value(r,label,'iou')),name+label))
    for label in records[0]['baselines']:
        result['baselines'][label]={m:float(source_average(records,lambda r:r['baselines'][label][m]).mean()) for m in ('iou','dice','accuracy')}
    comparisons=[('XYZ_staggered_zero','Ising_zero'),('XYZ_staggered_flexible','Ising_flexible'),
        ('XYZ_staggered_flexible','XYZ_tied_flexible'),('XYZ_staggered_flexible','matched_XYZ_ZZ'),
        ('XYZ_staggered_flexible','matched_Jz_only'),('XYZ_staggered_flexible','collapsed_same_b'),
        ('XYZ_staggered_flexible','collapsed_matched_scale'),('XYZ_staggered_flexible','XYZ_staggered_zero')]
    comparisons += [('XYZ_tied_flexible','Ising_flexible'),('Ising_flexible','Ising_zero')]
    if 'XYZ_exact_best' in common:comparisons.append(('XYZ_exact_best','Ising_exact_best'))
    for a,b in comparisons:
        result['paired'][a+' minus '+b]={}
        for m in ('iou','resolved','correct_resolved','wrong_resolved','unknown_iou'):
            v=source_average(records,lambda r:case_value(r,a,m)-case_value(r,b,m))
            result['paired'][a+' minus '+b][m]=interval(v,name+a+b+m)
    def cancellation(data):
        path=OUT/'test'/f'{data["case"]["case_id"]}.npz'
        axes=np.load(path)['XYZ_staggered_flexible_axes'];score=axes[:,1]-axes[:,0]
        keep=np.ones(score.shape[1],bool);keep[data['case']['seeds']]=False;score=score[:,keep]
        scale=abs(score).sum(axis=0);valid=scale>1e-8
        return [np.mean((score.min(axis=0)<-1e-8)&(score.max(axis=0)>1e-8)),
            np.mean(abs(score.mean(axis=0))<abs(score[2])-1e-8),
            np.mean(abs(score.sum(axis=0)[valid])/scale[valid]) if valid.any() else 1.]
    result['alignment_diagnostics']=dict(zip(['conflicting_axis_contrasts','normalized_alignment_smaller_than_ZZ','mean_cancellation_ratio'],
        source_average(records,cancellation).mean(axis=0).tolist()))
    return result


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def plots(summary,records):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold'})
    cohorts=['native_9','full_9','native_12','full_12']
    titles=['Native 3×3 patches','Full image → 3×3','Native 3×4 patches','Full image → 3×4']
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey='row')
    for col,(key,title) in enumerate(zip(cohorts,titles)):
        data=summary['cohorts'][key]
        for i,label in enumerate(MAIN):
            row=data['models'][label];m=row['budgets'][str(PRIMARY_BUDGET)]
            axes[0,col].bar(i,m['iou'],color=COLORS[i],width=.72)
            axes[0,col].plot(i,row['exact']['iou'],'k_',markersize=10)
            axes[1,col].bar(i,1-m['resolved'],color=COLORS[i],width=.72)
        for ax in axes[:,col]:
            ax.set_xticks(range(5),['Ising\nsign','Ising\n2 seeds','XYZ tied\n2 seeds','XYZ staggered\nsign','XYZ staggered\n2 seeds'],rotation=45,ha='right',fontsize=9);ax.set_ylim(0,1)
            ax.grid(axis='y',alpha=.2)
        axes[0,col].set_title(f'{title}\n{data["sources"]} test sources')
    axes[0,0].set_ylabel('Foreground IoU');axes[1,0].set_ylabel('Unresolved fraction (non-seeds)')
    fig.suptitle('XYZ alignment at equal Gibbs-copy budgets: 24,576 copies per prediction\nBlack ticks: exact Gibbs IoU; bars: mean over 16 independent sampling replicates',y=1.02)
    fig.tight_layout();save(fig,'comparison')
    fig,axes=plt.subplots(1,4,figsize=(15,3.8),sharey=True)
    for ax,key,title in zip(axes,cohorts,titles):
        for i,label in enumerate(MAIN):
            row=summary['cohorts'][key]['models'][label]
            ax.plot(BUDGETS,[1-row['budgets'][str(b)]['resolved'] for b in BUDGETS],'-o',color=COLORS[i],label=LABELS[label])
        ax.set_xscale('log');ax.set_ylim(0,1);ax.set_title(title);ax.set_xlabel('Total Gibbs copies');ax.grid(alpha=.2)
    axes[0].set_ylabel('Unresolved fraction (non-seeds)');axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    fig.suptitle('Resolution versus measurement budget; conservative simultaneous 95% intervals')
    fig.tight_layout();save(fig,'resolution-budgets')
    control_labels=['XYZ_staggered_flexible','matched_XYZ_ZZ','matched_Jz_only','collapsed_same_b','collapsed_matched_scale']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,key,title in zip(axes,['pooled_9','pooled_12'],['9 spins','12 spins']):
        data=summary['cohorts'][key];x=np.arange(len(control_labels));width=.25
        for k,(m,desc) in enumerate([('iou','IoU'),('correct_resolved','Correct and resolved'),('wrong_resolved','Wrong and resolved')]):
            ax.bar(x+width*(k-1),[data['models'][l]['budgets'][str(PRIMARY_BUDGET)][m] for l in control_labels],width,label=desc)
        ax.set_xticks(x,['XYZ alignment','Same XYZ, ZZ','Jz only','Collapsed τ','Collapsed τ\nmatched scale'],rotation=30,ha='right')
        ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Source-weighted mean');axes[1].legend(fontsize=9)
    fig.suptitle('Matched controls for the selected staggered XYZ model\nNative/full domains averaged within original source; 75 sources per size')
    fig.tight_layout();save(fig,'mechanism-controls')
    scores=json.loads((ROOT/'weizmann-xyz-calibration-scores.json').read_text())['scores']
    fig,ax=plt.subplots(figsize=(7,5))
    for model,obs,color in [('XYZ','alignment','#17856e'),('Ising','zz','#2171b5')]:
        rows=[r for r in scores if r['rule']['model']==model and r['observable']==obs]
        ax.scatter([1-r['sample']['resolved'] for r in rows],[r['sample']['iou'] for r in rows],s=12,alpha=.3,color=color,label=model+' '+obs)
    ax.set(xlabel='Unresolved fraction (non-seeds)',ylabel='Calibration foreground IoU',xlim=(-.02,1.02),ylim=(-.02,1.02))
    ax.grid(alpha=.2);ax.legend();ax.set_title('Calibration scan: accuracy and resolution are distinct\n459 Hamiltonian rules × four segmentation decisions')
    save(fig,'calibration-tradeoff')
    # Examples are explicitly chosen after outcomes for illustration, not tuning.
    pool=[r for r in records if r['case']['domain']=='full' and r['case']['shape']==[3,4]]
    pool.sort(key=lambda r:case_value(r,'XYZ_staggered_flexible','iou')-case_value(r,'Ising_flexible','iou'))
    examples=[pool[0],pool[len(pool)//2],pool[-1]]
    fig,axes=plt.subplots(3,6,figsize=(13,7))
    for row,data in enumerate(examples):
        cid=data['case']['case_id'];f=np.load(OUT/'cases'/f'{cid}.npz');saved=np.load(OUT/'test'/f'{cid}.npz')
        plots=[(f['image'],'Image','gray',0,255),(f['truth'],'Human majority','gray',0,1)]
        for label in ['Ising_flexible','XYZ_staggered_zero','XYZ_staggered_flexible']:
            r=next(x for x in data['results'] if x['label']==label);decision=r['entry']['decision']
            c=saved[label+'_samples'][1,0];score=decision_score(c,decision)-decision['threshold']
            pred=(score>1e-12).astype(float);radius=uncertainty_radius(f['truth'].size,PRIMARY_BUDGET,decision)
            pred[abs(score)<=radius]=.5
            plots.append((pred.reshape(f['truth'].shape),LABELS[label],'gray',0,1))
        r=next(x for x in data['results'] if x['label']=='XYZ_staggered_flexible');a=saved['XYZ_staggered_flexible_axes'].mean(axis=0)[0].reshape(f['truth'].shape)
        plots.append((a,'Exact background alignment / 3','coolwarm',-1,1))
        for col,(pixels,title,cmap,vmin,vmax) in enumerate(plots):
            im=axes[row,col].imshow(pixels,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['Image','Human majority','Ising\n2-seed rule','XYZ staggered\nsign rule','XYZ staggered\n2-seed rule','Exact background\nalignment / 3'][col],fontsize=9)
            if col==5:fig.colorbar(im,ax=axes[row,col],fraction=.046,pad=.04,ticks=[-1,0,1])
        for k,seed in enumerate(f['seeds']):
            sy,sx=divmod(int(seed),f['truth'].shape[1])
            axes[row,1].text(sx,sy,'B' if k==0 else 'F',ha='center',va='center',color='#e36b0a',weight='bold')
        axes[row,0].set_ylabel(data['case']['source_id'],fontsize=8)
    fig.suptitle('Illustrative worst / median / best XYZ-minus-Ising full-frame cases\nFirst sampling replicate; gray pixels unresolved. Examples selected after testing, not used for tuning.')
    fig.tight_layout();save(fig,'examples')


def main():
    protocol=json.loads((ROOT/'weizmann-xyz-protocol.json').read_text())
    assert protocol['code_hashes']==hashes()
    assert protocol['manifest_sha256']==hashlib.sha256((ROOT/'weizmann-xyz-manifest.json').read_bytes()).hexdigest()
    records=load_cases();cohorts={}
    for n in (9,12):
        by_size=[r for r in records if np.prod(r['case']['shape'])==n]
        for domain in ('native','full'):
            key=f'{domain}_{n}';cohorts[key]=aggregate([r for r in by_size if r['case']['domain']==domain],key)
        cohorts[f'pooled_{n}']=aggregate(by_size,f'pooled_{n}')
    coverage=[b['simultaneous_coverage_fraction'] for d in records for r in d['results'] for b in r['budgets'].values()]
    summary=dict(date='2026-10-08',cohorts=cohorts,code_hashes=hashes(),
        selection=json.loads((ROOT/'weizmann-xyz-selection.json').read_text())['winners'],
        validation=dict(test_cases=len(records),test_sources=len({r['case']['source_id'] for r in records}),
            max_eigen_residual=max(r['eigen_residual'] for r in records),
            average_simultaneous_coverage=float(np.mean(coverage)),minimum_model_budget_coverage=float(np.min(coverage)),
            summed_case_wall_seconds=float(sum(r['seconds'] for r in records)),
            median_case_wall_seconds_by_spins={str(n):float(np.median([r['seconds'] for r in records if np.prod(r['case']['shape'])==n])) for n in (9,12)}),
        statistics='5000 paired bootstrap resamples of original source images; intervals exploratory and not corrected for multiple comparisons. Sampling replicates are averaged before source bootstrap.',
        limitations='Exact spectra through 12 spins; grayscale; mask-selected boundary crops; very coarse full images; two oracle seed locations; original split reused from earlier development.')
    write_json(ROOT/'weizmann-xyz-summary.json',summary);plots(summary,records)
    print(json.dumps(summary['validation'],indent=2))
    for key in ['native_9','full_9','native_12','full_12']:
        print(key)
        for label in MAIN:
            r=cohorts[key]['models'][label];m=r['budgets'][str(PRIMARY_BUDGET)]
            print(label,'IoU',round(m['iou'],4),'exact',round(r['exact']['iou'],4),'unresolved',round(1-m['resolved'],4),'wrong-resolved',round(m['wrong_resolved'],4))


if __name__=='__main__':main()
