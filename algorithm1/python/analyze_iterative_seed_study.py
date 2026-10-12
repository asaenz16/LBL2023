"""Source-paired analysis for iterative seeds, with sample-budget controls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/iterative-seeds-mpl')
import json
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_iterative_seed_study import ROOT,OUT,INPUT,cases,selected_jobs,identifier,policy_id,hashes,METRICS,REASONS
from weizmann_petz_data import write_json,seed_for

DISPLAY=('iou','unknown_iou','unresolved','mixing_unresolved','mixing_lower','correct_resolved','wrong_resolved',
    'bpm_iou_r0','bpm_iou_r1','bpm_iou_r2','rounds','foreground_seeds','background_seeds','promoted_error',
    'promoted_count','promoted_wrong','converged','minimum_ess','initial_iou','iou_change')


def estimate(values,key):
    x=np.asarray(values,float);rng=np.random.default_rng(seed_for('iterative-bootstrap:'+key))
    samples=x[rng.integers(len(x),size=(5000,len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),sources=len(x))


def summarize(rows,key):
    out=dict(cases=len(rows),metrics={m:estimate([r['metrics'][m] for r in rows],key+m) for m in DISPLAY},
        controls={c:{m:estimate([r['controls'][c][m] for r in rows],key+c+m) for m in ('iou','unknown_iou','unresolved','mixing_unresolved','bpm_iou_r1')}
            for c in ('initial','resample_only','pooled_one_seed','oracle_two_seed')},
        adaptive_minus_control={c:{m:estimate([r['metrics'][m]-r['controls'][c][m] for r in rows],key+'paired'+c+m)
            for m in ('iou','unknown_iou','unresolved','mixing_unresolved','bpm_iou_r1')}
            for c in ('initial','resample_only','pooled_one_seed','oracle_two_seed')},
        stop_fractions={name:float(np.mean([r['stops'].get(name,0.) for r in rows])) for name in REASONS},
        mean_within_image_promotion_iou_sd=float(np.mean([r['random_iou_sd'] for r in rows])),
        cases_with_any_failed_trajectory=sum(r['any_failed'] for r in rows),
        cases_with_any_ess_below100=sum(r['any_low_ess'] for r in rows),
        confirmed_cases=sum(r['confirmed'] for r in rows),
        collapse_fractions={k:float(np.mean([r['collapse'][k] for r in rows])) for k in
            ('all_foreground','all_background','no_foreground_seed','no_background_seed')},
        mean_trajectory=[{m:float(np.mean([r['trajectory'][t][m] for r in rows])) for m in ('iou','unresolved','mixing_unresolved','promoted_count','promoted_wrong','active')}
            for t in range(10)])
    added=sum(r['metrics']['promoted_count'] for r in rows);wrong=sum(r['metrics']['promoted_wrong'] for r in rows)
    out['pooled_promoted_label_error']=wrong/added if added else None
    out['promotion_by_class']={}
    for role in ('foreground','background'):
        total=sum(r['promotion_tally'][role+'_total'] for r in rows);errors=sum(r['promotion_tally'][role+'_wrong'] for r in rows)
        out['promotion_by_class'][role]=dict(mean_added=total/len(rows),error_fraction=errors/total if total else None)
    out['mean_wrong_seed_weight_fraction']=float(np.mean([r['promotion_tally']['wrong_weight_fraction'] for r in rows]))
    return out


def comparison(a,b,key):
    lookup={r['case']['case_id']:r for r in b}
    return {m:estimate([r['metrics'][m]-lookup[r['case']['case_id']]['metrics'][m] for r in a],key+m)
        for m in ('iou','unknown_iou','unresolved','mixing_unresolved','bpm_iou_r1','rounds')}


def compact(row,table,pi,truth,seeds):
    trace=[t for t in row['traces'] if t['policy_index']==pi];arr=table[pi]
    controls={k:{m:float(np.mean([t['controls'][k][m] for t in trace])) for m in trace[0]['controls'][k]}
        for k in trace[0]['controls']}
    trajectory=[]
    for step in range(10):
        values=[t['history'][min(step,len(t['history'])-1)]['metrics'] for t in trace]
        d={m:float(np.mean([v[m] for v in values])) for m in ('iou','unresolved','mixing_unresolved','promoted_count','promoted_wrong')}
        d['active']=float(np.mean([len(t['history'])>step for t in trace]));trajectory.append(d)
    stop=Counter(t['stop'] for t in trace)
    n=int(np.prod(row['case']['shape']))
    predictions=[np.unpackbits(np.asarray(t['prediction'],np.uint8),count=n) for t in trace]
    collapse=dict(all_foreground=float(np.mean([np.all(p) for p in predictions])),
        all_background=float(np.mean([not np.any(p) for p in predictions])),
        no_foreground_seed=float(np.mean([not t['history'][-1]['foreground'] for t in trace])),
        no_background_seed=float(np.mean([not t['history'][-1]['background'] for t in trace])))
    tallies=[];flat=truth.ravel();policy=row['policies'][pi];initial=int(seeds[1 if policy['start']=='foreground' else 0])
    for tr in trace:
        last=tr['history'][-1];tally={};weights=[]
        for role in ('foreground','background'):
            points=last[role];promoted=[p for p in points if p!=initial]
            wrong=np.sum(~flat[promoted]) if role=='foreground' else np.sum(flat[promoted])
            tally[role+'_total']=len(promoted);tally[role+'_wrong']=int(wrong)
            if points:weights.append(wrong/len(points))
        tally['wrong_weight_fraction']=float(np.mean(weights));tallies.append(tally)
    tally={k:float(np.mean([t[k] for t in tallies])) for k in tallies[0]}
    return dict(case=row['case'],rule=row['rule'],rule_id=row['rule_id'],policy=row['policies'][pi],
        metrics=dict(zip(METRICS,arr.mean(axis=0).tolist())),controls=controls,
        stops={k:v/len(trace) for k,v in stop.items()},trajectory=trajectory,collapse=collapse,promotion_tally=tally,
        random_iou_sd=float(arr[:,METRICS.index('iou')].std(ddof=1)),
        any_failed=bool(np.any(arr[:,METRICS.index('converged')]<1)),
        any_low_ess=bool(np.any(arr[:,METRICS.index('minimum_ess')]<100)),
        confirmed=row['signature']['stage']=='confirm')


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(folder/f'{name}.{suffix}',dpi=175,bbox_inches='tight')
    plt.close(fig)


def main():
    selection=json.loads((ROOT/'iterative-seeds-selection.json').read_text());groups=defaultdict(list);all_records=[]
    routes=defaultdict(list)
    for label,w in selection['winners'].items():
        routes[(w['rule_id'],policy_id(w['policy']))].append(label)
        reverse=dict(w['policy'],start='background' if w['policy']['start']=='foreground' else 'foreground')
        routes[(w['rule_id'],policy_id(reverse))].append(label+'/opposite_start')
    for c in cases('test'):
        source=np.load(INPUT/'cases'/f'{c["case_id"]}.npz')
        for entry in selected_jobs():
            name=f'{c["case_id"]}--{identifier(entry["rule"])}';path=OUT/'confirm'/f'{name}.json'
            if not path.exists():path=OUT/'test'/f'{name}.json'
            row=json.loads(path.read_text());table=np.load(path.with_suffix('.npz'))['metrics']
            for pi,policy in enumerate(row['policies']):
                record=compact(row,table,pi,source['truth'],source['seeds']);all_records.append(record)
                for label in routes[(row['rule_id'],policy_id(policy))]:groups[f'{label}/{c["shape"][0]}'].append(record)
    summary={k:summarize(v,k) for k,v in groups.items()};model_differences={};start_differences={}
    for size in (8,16):
        for start in ('foreground','background'):
            a=groups[f'XXZ/{start}/mixing_lower/{size}'];b=groups[f'Ising/{start}/mixing_lower/{size}']
            model_differences[f'{start}/{size}']=comparison(a,b,'model'+start+str(size))
        for label in selection['winners']:
            start_differences[f'{label}/{size}']=comparison(groups[f'{label}/{size}'],groups[f'{label}/opposite_start/{size}'],'start'+label+str(size))
    result=dict(status='completed',date='2026-10-09',selection=selection['winners'],groups=summary,
        numerical_amendment=json.loads((ROOT/'iterative-seeds-numerical-amendment.json').read_text()),
        selected_xxz_minus_selected_ising=model_differences,selected_start_minus_opposite_start=start_differences,
        code_hashes=hashes(),unique_hamiltonians=len(selected_jobs()),test_model_cases=150*len(selected_jobs()),
        confirmation_runs=sum(not p.name.endswith('.bank.json') for p in (OUT/'confirm').glob('*.json')),calibration_model_cases=4800,
        grid_hamiltonians=96,grid_policies=432,grid_combinations=41472,calibration_random_trajectory_endpoints=6220800,
        statistical_unit='75 original source images per resolution; ten promotion replicates averaged within image; 5000 paired source-bootstrap replicates, no multiplicity adjustment. Reused development split, not external validation.',
        plot_trajectory='Stopped trajectories carried forward; active fractions are recorded in mean_trajectory. No post-stopping measurements affect their masks.')
    write_json(ROOT/'iterative-seeds-summary.json',result);write_json(OUT/'case-summary.json',all_records)
    plot_main(result);plot_calibration(selection);plot_policy_slices(selection);plot_examples(selection)
    for size in (8,16):
        for model in ('Ising','XXZ'):
            for start in ('foreground','background'):
                key=f'{model}/{start}/mixing_lower/{size}';a=summary[key]
                print(key,{m:round(a['metrics'][m]['mean'],4) for m in ('iou','unresolved','mixing_unresolved','bpm_iou_r1','rounds')},
                    'vs_pooled',a['adaptive_minus_control']['pooled_one_seed']['iou'],flush=True)


def plot_main(j):
    colors={'Ising/foreground':'#277da8','Ising/background':'#76b5ce','XXZ/foreground':'#b4542c','XXZ/background':'#d99b76'}
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,size in enumerate((8,16)):
        for col,metric in enumerate(('iou','unresolved','bpm_iou_r1')):
            ax=axes[row,col]
            for x,(modelstart,color) in enumerate(colors.items()):
                a=j['groups'][f'{modelstart}/mixing_lower/{size}'];v=a['metrics'][metric]['mean']
                lo,hi=a['metrics'][metric]['ci95']
                ax.bar(x,v,color=color,yerr=np.array([[v-lo],[hi-v]]),capsize=3)
                ax.text(x,hi+.02,f'{v:.3f}',ha='center',fontsize=9)
            ax.set_xticks(range(4),['Ising\nFG start','Ising\nBG start','XXZ\nFG start','XXZ\nBG start'],fontsize=8);ax.set_ylim(0,1)
            ax.set_title(f'{size}×{size}: '+{'iou':'IoU','unresolved':'Unresolved fraction','bpm_iou_r1':'BPM IoU (r=1)'}[metric])
    fig.suptitle('Selected iterative seed models: 75 sources, ten promotion replicates\nBars: source bootstrap 95% intervals; unresolved excludes only the initial pixel')
    fig.tight_layout();save(fig,'selected-results')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,size in enumerate((8,16)):
        for col,metric in enumerate(('iou','mixing_unresolved','promoted_count')):
            for key,color in colors.items():
                a=j['groups'][f'{key}/mixing_lower/{size}'];values=[t[metric] for t in a['mean_trajectory']]
                axes[row,col].plot(range(1,11),values,'-o',markersize=3,color=color,label=key.replace('/',' / '))
            axes[row,col].set_title(f'{size}×{size}: '+{'iou':'IoU','mixing_unresolved':'Unresolved, including mixing flags','promoted_count':'Promoted pixels'}[metric])
            axes[row,col].set_xlabel('Round');axes[row,col].grid(alpha=.2)
            if col<2:axes[row,col].set_ylim(0,1)
    axes[0,0].legend(fontsize=8);fig.suptitle('Evolution of selected procedures; stopped trajectories carried forward')
    fig.tight_layout();save(fig,'round-trajectories')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    labels=['Initial','Fresh last round','Pooled one seed','Iterative seeds','Oracle two seeds']
    for ax,size in zip(axes,(8,16)):
        for i,(key,color) in enumerate(colors.items()):
            a=j['groups'][f'{key}/mixing_lower/{size}']
            values=[a['controls'][m]['iou']['mean'] for m in ('initial','resample_only','pooled_one_seed')]+[a['metrics']['iou']['mean'],a['controls']['oracle_two_seed']['iou']['mean']]
            ax.plot(range(5),values,'-o',label=key.replace('/',' / '),color=color)
        ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('IoU');ax.set_title(f'{size}×{size}');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed-Hamiltonian controls; pooled controls use the same number of rounds\nOracle two-seed control supplies an additional true label')
    fig.tight_layout();save(fig,'sample-budget-controls')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    palette=plt.get_cmap('tab10')
    for ax,size in zip(axes,(8,16)):
        bottom=np.zeros(4)
        for k,reason in enumerate(REASONS):
            values=[j['groups'][f'{key}/mixing_lower/{size}']['stop_fractions'][reason] for key in colors]
            ax.bar(range(4),values,bottom=bottom,label=reason.replace('_',' '),color=palette(k));bottom+=values
        ax.set_xticks(range(4),[key.replace('/','\n') for key in colors],fontsize=9);ax.set_ylim(0,1);ax.set_title(f'{size}×{size}')
    axes[1].legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8);fig.suptitle('Why each selected procedure terminates')
    fig.tight_layout();save(fig,'stopping-reasons')


def plot_calibration(selection):
    rows=json.loads((OUT/'calibration-scores.json').read_text());fig,axes=plt.subplots(2,4,figsize=(14,8))
    reliability=[]
    for g in (0.,.1,.25,.5):
        for k in (.4,.8,1.2):
            pool=[r for r in rows if r['policy']['start']=='foreground' and r['rule']['threshold']=='q90' and r['rule']['sensitivity']=='fixed01' and r['rule']['strength']==g and r['rule']['K']==k]
            reliability.append(min(pool,key=lambda r:(-r['metrics']['mixing_lower'],-r['metrics']['iou'],r['metrics']['rounds'],r['rule_id'],r['policy_id'])))
    write_json(ROOT/'iterative-seeds-calibration-reliability.json',reliability)
    recipes=[(t,a) for t in ('edge_otsu','q75','q90','fixed125') for a in ('fixed01','inverse_spread')]
    for i,start in enumerate(('foreground','background')):
        for col,g in enumerate((0.,.1,.25,.5)):
            mat=np.full((8,3),np.nan)
            for a,recipe in enumerate(recipes):
                for b,k in enumerate((.4,.8,1.2)):
                    values=[r['metrics']['mixing_lower'] for r in rows if r['policy']['start']==start and r['rule']['strength']==g and r['rule']['K']==k and (r['rule']['threshold'],r['rule']['sensitivity'])==recipe]
                    mat[a,b]=max(values)
            ax=axes[i,col];im=ax.imshow(mat,vmin=0,vmax=.6,cmap='viridis',aspect='auto')
            ax.set_xticks(range(3),[.4,.8,1.2]);ax.set_xlabel('K');ax.set_title(f'{start} start; g/D={g:g}')
            ax.set_yticks(range(8),[t.replace('edge_otsu','Otsu')+('/.01' if s=='fixed01' else '/adaptive') for t,s in recipes],fontsize=8)
            for a in range(8):
                for b in range(3):ax.text(b,a,f'{mat[a,b]:.2f}',ha='center',va='center',color='white',fontsize=8)
    fig.suptitle('Best calibration lower IoU across promotion/stopping policies at each Hamiltonian\nTraining envelope only; every displayed cell includes the full policy grid')
    fig.tight_layout();save(fig,'calibration-envelope')


def plot_policy_slices(selection):
    rows=json.loads((OUT/'calibration-scores.json').read_text())
    labels=[f'{model}/{start}/mixing_lower' for model in ('Ising','XXZ') for start in ('foreground','background')]
    fig,axes=plt.subplots(4,3,figsize=(12,13));stopping_tables={}
    for i,label in enumerate(labels):
        winner=selection['winners'][label];policy=winner['policy'];rule_id=winner['rule_id']
        group=[r for r in rows if r['rule_id']==rule_id and r['policy']['start']==policy['start']]
        for col,metric in enumerate(('iou','unresolved','rounds')):
            mat=np.empty((3,3))
            for a,cf in enumerate((.25,.5,.75)):
                for b,cb in enumerate((.25,.5,.75)):
                    target=dict(policy,cut_fg=cf,cut_bg=cb)
                    r=next(x for x in group if x['policy']==target);mat[a,b]=r['metrics'][metric]
            ax=axes[i,col];im=ax.imshow(mat,vmin=0,vmax=10 if metric=='rounds' else 1,cmap='viridis',aspect='auto')
            ax.set_xticks(range(3),[.25,.5,.75]);ax.set_yticks(range(3),[.25,.5,.75]);ax.set_xlabel('Anti-seed cutoff');ax.set_ylabel('Seed cutoff')
            ax.set_title(label.replace('/mixing_lower','').replace('/',' / ')+': '+metric,fontsize=10)
            for a in range(3):
                for b in range(3):ax.text(b,a,f'{mat[a,b]:.2f}',ha='center',va='center',color='white',fontsize=9)
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        stopping_tables[label]=[r for r in group if all(r['policy'][k]==policy[k] for k in ('cut_fg','cut_bg','gate'))]
    fig.suptitle('Calibration cutoff slices: other parameters held at the selected settings\nPositive/negative cutoffs use the unnormalized Pauli-correlation contrast')
    fig.tight_layout();save(fig,'cutoff-slices')
    write_json(ROOT/'iterative-seeds-stopping-slices.json',stopping_tables)
    fig,axes=plt.subplots(2,2,figsize=(11,8));colors=plt.get_cmap('tab10')
    for ax,label in zip(axes.ravel(),labels):
        for i,stop in enumerate(('none','same_count','no_improvement','tolerance_1pct')):
            data=sorted([r for r in stopping_tables[label] if r['policy']['stopping']==stop],key=lambda r:r['policy']['cap'])
            x=[r['metrics']['rounds'] for r in data];y=[r['metrics']['iou'] for r in data]
            ax.plot(x,y,'-',color=colors(i),label=stop.replace('_',' '))
            for r,xx,yy in zip(data,x,y):ax.scatter([xx],[yy],marker={3:'^',6:'s',10:'o'}[r['policy']['cap']],color=colors(i),s=36)
        ax.set_title(label.replace('/mixing_lower','').replace('/',' / '));ax.set_xlabel('Mean rounds used');ax.set_ylabel('Calibration IoU');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Stopping-rule tradeoffs at fixed Hamiltonian and cutoffs; vertical axes are zoomed\nRound caps: triangle = 3, square = 6, circle = 10')
    fig.tight_layout();save(fig,'stopping-tradeoffs')


def plot_examples(selection):
    chosen=sorted([c for c in cases('test') if c['shape']==[16,16]],key=lambda c:c['source_id'])[:4]
    labels=[f'{model}/{start}/mixing_lower' for model in ('Ising','XXZ') for start in ('foreground','background')]
    fig,axes=plt.subplots(4,6,figsize=(14,10))
    for i,c in enumerate(chosen):
        data=np.load(INPUT/'cases'/f'{c["case_id"]}.npz');axes[i,0].imshow(data['image'],cmap='gray',vmin=0,vmax=255)
        axes[i,1].imshow(data['truth'],cmap='gray',vmin=0,vmax=1);axes[i,0].set_ylabel(c['source_id'],fontsize=8)
        for j,label in enumerate(labels,2):
            winner=selection['winners'][label];name=f'{c["case_id"]}--{winner["rule_id"]}.json';path=OUT/'confirm'/name
            if not path.exists():path=OUT/'test'/name
            row=json.loads(path.read_text());pi=row['policies'].index(winner['policy'])
            tr=next(x for x in row['traces'] if x['policy_index']==pi and x['repeat']==0)
            mask=np.unpackbits(np.array(tr['prediction'],np.uint8),count=256).reshape(16,16)
            ax=axes[i,j];ax.imshow(mask,cmap='gray',vmin=0,vmax=1);last=tr['history'][-1]
            for pixels,color,marker in ((last['foreground'],'magenta','+'),(last['background'],'cyan','x')):
                for p in pixels:y,x=divmod(p,16);ax.scatter([x],[y],c=color,marker=marker,s=20)
            initial=int(data['seeds'][1 if winner['policy']['start']=='foreground' else 0]);y,x=divmod(initial,16)
            ax.scatter([x],[y],facecolors='none',edgecolors='yellow',marker='o',s=60,linewidths=.8)
            ax.set_xlabel(f"IoU {last['metrics']['iou']:.3f}; U {100*last['metrics']['unresolved']:.1f}%\nrounds {len(tr['history'])}",fontsize=8)
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Input','Truth','Ising: FG start','Ising: BG start','XXZ: FG start','XXZ: BG start']):ax.set_title(title,fontsize=10)
    fig.suptitle('Fixed first four test sources by ID, first random promotion replicate\nMagenta +: foreground seeds; cyan ×: background seeds; yellow circle: the sole supplied true label')
    fig.tight_layout();save(fig,'segmentation-examples')


if __name__=='__main__':main()
