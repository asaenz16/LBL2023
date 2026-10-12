"""Source-level paired analysis and scientific figures for higher-seed study."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
_plot_cache=Path(__file__).resolve().parents[1]/'results/higher-seeds/plot-cache'
_plot_cache.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(_plot_cache)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from run_higher_seed_study import ROOT, OUT, anchors, selection, METRICS, digest
from weizmann_petz_data import write_json

MI={m:i for i,m in enumerate(METRICS)}
BOOT=np.random.default_rng(52611).integers(0,75,(5000,75))
TITLES={'pair':'Pairwise baseline','majority_gate':'Majority gate','soft_gate':'Soft-conditioning gate',
        'majority_score':'Majority score','soft_score':'Soft-conditioning score'}


def summary(v):
    a=v.mean(0);out=dict(zip(METRICS,a.tolist()))
    den=v[:,MI['promoted_count']].sum()
    out['wrong_promotion_rate']=float(v[:,MI['promoted_wrong']].sum()/den) if den else None
    out['images']=len(v)
    return out


def difference(a,b):
    result={}
    for metric in ('iou','unresolved','mixing_unresolved','mixing_lower','bpm_iou_r1','shots'):
        d=a[:,MI[metric]]-b[:,MI[metric]];draw=d[BOOT].mean(1)
        result[metric]=dict(mean=float(d.mean()),ci95=np.quantile(draw,[.025,.975]).tolist())
    num=MI['promoted_wrong'];den=MI['promoted_count']
    ad=a[BOOT,den].sum(1);bd=b[BOOT,den].sum(1)
    valid=(ad>0)&(bd>0)
    if a[:,den].sum()>0 and b[:,den].sum()>0 and valid.any():
        draw=a[BOOT,num].sum(1)[valid]/ad[valid]-b[BOOT,num].sum(1)[valid]/bd[valid]
        result['wrong_promotion_rate']=dict(mean=float(a[:,num].sum()/a[:,den].sum()-b[:,num].sum()/b[:,den].sum()),
            ci95=np.quantile(draw,[.025,.975]).tolist())
    else:result['wrong_promotion_rate']=None
    return result


def load():
    groups=defaultdict(dict);runtime=defaultdict(float);trace_count=0;inventory={};triplets=defaultdict(list)
    calibration=list((OUT/'calibrate').glob('*.json'));assert len(calibration)==200
    runtime['calibration']=sum(json.loads(p.read_text())['seconds'] for p in calibration)
    for path in sorted((OUT/'test').glob('*.json')):
        meta=json.loads(path.read_text());spec=meta['signature'];side=spec['case']['shape'][0]
        value=np.load(path.with_suffix('.npz'))['metrics'].mean(1)
        for label,i in meta['labels'].items():
            groups[(spec['anchor'],side,spec['samples'],label)][spec['case']['source_id']]=value[i]
        runtime[str(spec['samples'])]+=meta['seconds'];trace_count+=len(meta['traces'])
        inventory[path.name]=digest(path.with_suffix('.npz'))
    for path in sorted((OUT/'pooled-controls').glob('*.json')):
        meta=json.loads(path.read_text());side=meta['case']['shape'][0]
        for label,values in meta['controls'].items():
            v=np.array([[r[m] for m in METRICS] for r in values]).mean(0)
            groups[(meta['anchor'],side,meta['samples'],label+'/pooled_one_seed')][meta['case']['source_id']]=v
        for label,obs in meta.get('triplets',{}).items():
            if obs:
                triplets[f'{meta["anchor"]}/{side}/{label}'].append({k:float(np.mean([x[k] for x in obs])) for k in
                    ('agreement_probability','majority_wrong_labels','majority_linear_difference_bound')})
    arrays={}
    for key,rows in groups.items():
        assert len(rows)==75,(key,len(rows))
        arrays[key]=np.array([rows[s] for s in sorted(rows)])
    diagnostic={k:dict(images_with_triplets=len(v),**{m:float(np.mean([r[m] for r in v])) for m in v[0]}) for k,v in triplets.items()}
    return arrays,runtime,trace_count,inventory,diagnostic


def main():
    arrays,runtime,traces,inventory,diagnostic=load();chosen=selection()['winners'];rows={};effects={}
    for key,a in arrays.items():
        anchor,side,budget,label=key;name=f'{anchor}/{side}/{budget}/{label}'
        rows[name]=summary(a)
        rows[name]['estimated_sse_sweeps']=rows[name]['rounds']*4*(2000+2*budget) if anchor.startswith('XXZ/') else None
        if '/matched_' not in label and '/pooled_one_seed' not in label and not label.startswith('pair/'):
            effects[name]={}
            controls=['pair/balanced',label+'/matched_pair',label+'/matched_linear']
            if (anchor,side,budget,label+'/pooled_one_seed') in arrays:controls.append(label+'/pooled_one_seed')
            for control in controls:
                effects[name][control]=difference(a,arrays[anchor,side,budget,control])
    write_json(ROOT/'higher-seeds-summary.json',dict(metrics=rows,effects=effects,offline_seconds=runtime,
        trajectories=traces,triplet_diagnostics=diagnostic,selection_sha256=digest(ROOT/'higher-seeds-selection.json'),result_sha256=inventory))
    with (OUT/'metrics.csv').open('w',newline='') as f:
        fields=['setting']+list(next(iter(rows.values())).keys());w=csv.DictWriter(f,fields);w.writeheader()
        for setting,v in rows.items():w.writerow(dict(setting=setting,**v))
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    # Effect intervals isolate each mechanism from its exact cutoff/stop/cap control.
    fig,axs=plt.subplots(1,3,figsize=(14,7),layout='constrained')
    entries=[]
    for anchor in anchors():
        for family in ('majority_gate','soft_gate','majority_score','soft_score'):
            label=family+'/balanced';name=f'{anchor}/16/2048/{label}'
            entries.append((anchor+' | '+TITLES[family],effects[name][label+'/matched_pair']))
    for ax,metric,title,scale in zip(axs,('iou','unresolved','wrong_promotion_rate'),
                                    ('IoU change','Unresolved change (pp)','Wrong promotions change (pp)'),(1,100,100)):
        for y,(label,result) in enumerate(entries):
            r=result.get(metric)
            if r is None:continue
            lo,hi=np.array(r['ci95'])*scale;mean=r['mean']*scale
            ax.plot([lo,hi],[y,y],color='#156082');ax.plot(mean,y,'o',color='#156082',ms=4)
        ax.axvline(0,color='gray',ls='--',lw=1);ax.set_title(title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if metric=='iou':ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_yticks(range(len(entries)),[e[0] for e in entries] if ax is axs[0] else ['']*len(entries),fontsize=8)
    fig.suptitle('Three-seed methods vs matched pairwise controls | 16×16 Weizmann\nCalibration-selected balanced policies; paired source-bootstrap 95% intervals')
    fig.savefig(figures/'matched-effects-16.png',dpi=170);fig.savefig(figures/'matched-effects-16.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for col,anchor in enumerate(anchors()):
        for family in TITLES:
            available=[b for b in (512,1024,2048) if (anchor,16,b,family+'/balanced') in arrays]
            v=[rows[f'{anchor}/16/{b}/{family}/balanced'] for b in available]
            axs[0,col].plot([x['shots'] for x in v],[x['iou'] for x in v],'-o',label=TITLES[family],ms=4)
            axs[1,col].plot([x['shots'] for x in v],[100*x['mixing_unresolved'] for x in v],'-o',ms=4)
        axs[0,col].set_title(anchor);axs[1,col].set_xlabel('Mean retained configurations')
        for ax in axs[:,col]:ax.grid(alpha=.2);ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axs[0,0].set_ylabel('IoU');axs[1,0].set_ylabel('Mixing-aware unresolved (%)')
    axs[0,0].legend(fontsize=7);fig.suptitle('Sampling-budget sensitivity at fixed calibrated parameters | 16×16')
    fig.savefig(figures/'sampling-cost-16.png',dpi=170);plt.close(fig)
    calibration=json.loads((OUT/'calibration-scores.json').read_text())
    fig,axs=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for col,anchor in enumerate(anchors()):
        vals=np.array(calibration['means'][anchor]);policies=calibration['policies']
        for row,(family,param,grid) in enumerate((('soft_gate','gate_margin',[0.,.15]),('soft_score','blend',[.25,.5,1.]))):
            a=np.empty((4,len(grid)))
            for yi,eta in enumerate((.25,.5,.75,.9)):
                for xi,x in enumerate(grid):
                    ix=[i for i,p in enumerate(policies) if p['family']==family and p['eta']==eta and p[param]==x]
                    a[yi,xi]=max(vals[i,MI['mixing_lower']] for i in ix)
            ax=axs[row,col];im=ax.imshow(a,vmin=0,vmax=.6,cmap='viridis',aspect='auto')
            ax.set_xticks(range(len(grid)),grid);ax.set_yticks(range(4),[.25,.5,.75,.9]);ax.set_xlabel('Gate margin' if row==0 else 'Blend weight')
            if col==0:ax.set_ylabel(('Soft gate' if row==0 else 'Soft score')+'\nPromoted-seed strength η')
            if row==0:ax.set_title(anchor)
            for yi,xi in np.ndindex(a.shape):ax.text(xi,yi,f'{a[yi,xi]:.3f}',ha='center',va='center',color='white',fontsize=8)
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Calibration mixing-aware lower IoU',shrink=.8)
    fig.suptitle('Soft-conditioning parameter sweep\nEach cell optimizes the remaining settings on calibration images only')
    fig.savefig(figures/'calibration-soft-parameters.png',dpi=170);plt.close(fig)
    write_report(rows,effects,chosen,runtime,traces,diagnostic)


def write_report(rows,effects,chosen,runtime,traces,diagnostic):
    lines=['# Three-seed majority and soft conditioning: numerical Gibbs experiment','',
        'Completed readout and promotion study on the Weizmann Segmentation Evaluation Database. All selections use calibration images; comparison outcomes do not choose parameters. These are reused development images, not an external validation set.','',
        '## Findings','',
        'Neither three-seed majority nor soft conditioning gives a consistent segmentation or promotion-reliability improvement beyond matched pairwise controls in this parameter grid. At 16×16, none of the 16 balanced higher-order methods has a positive matched-control IoU confidence interval excluding zero. Promotion gates are usually redundant; score methods change coverage and cost more than IoU. This is evidence about the tested policies and Hamiltonians, not a general impossibility result for higher correlations.',
        'The apparent Ising/foreground coverage gain against the separately tuned baseline is misleading without matching iteration settings. Majority scoring has 1.31% unresolved pixels versus 0.59% for pairwise scoring with the same settings; soft scoring has 1.06% versus 0.70%. Their matched IoU differences are −0.00065 [−0.00162, +0.00010] and +0.00170 [−0.00266, +0.00708], respectively. Neither improvement is established. The headline soft method also promotes more incorrect labels and consumes more samples than the separately tuned baseline.',
        'Optimizing promotion reliability on calibration data lowers the comparison wrong-promotion rate to 28.02% for Ising/foreground soft scoring and 24.71% for Ising/background majority scoring, at unresolved fractions of 5.97% and 5.36%. The changes relative to controls with the same promotion cutoff and stopping rule remain inconclusive. Stricter promotion settings account for much of the tradeoff.',
        'XXZ favors a small higher-order blend (0.25) in the balanced calibration selections. Increasing the blend often worsens the calibration lower-IoU objective. Ising selects stronger blends, but their comparison gains do not survive matched-control scrutiny. There is no parameter setting established here as uniformly best for IoU, coverage, boundaries, reliability, and cost.',
        'Sampling limits matter: the fixed-budget XXZ estimates have substantial convergence failures. The larger-budget confirmation at the end of this report preserves all selected parameters and uses the same replacement bank for every method on an affected image. Do not interpret statistical abstention alone as total uncertainty.',
        'One plausible explanation for the small higher-order effect is vote redundancy and correlated label mistakes. For example, Ising/foreground soft-score triplets have 84.78% average signed-spin unanimity, yet 25.16% of evaluated triplets contain at least two semantically wrong labels. Agreement among inferred seeds is not independent evidence that the segmentation is correct.','',
        '## Experimental scope','',
        '- Ising and XXZ, each with a known foreground or background starting pixel; fixed primary Hamiltonians inherited from the promotion study. Hamiltonians were selected separately previously, so differences across model/start rows do not isolate only the model or only the starting label.',
        '- 25 calibration sources and 75 comparison sources at both 8×8 and 16×16. Two random promotion replicates per calibration case; five per comparison case.',
        '- 306 readout/promotion configurations × 3 round caps × 2 stopping settings = 1,836 policies per model/start, 7,344 combinations and 734,400 calibration endpoints.',
        '- Promotion cutoffs 0.25/0.5/0.75; abstention margins 0/0.05; anchored or uniform triplets; gate confidence margins 0/0.15; score blend weights 0.25/0.5/1; soft promoted-seed strengths 0.25/0.5/0.75/0.9, with trusted-anchor strength 1.',
        '- Caps 3/6/10; ordinary stopping or two equal-count transitions. Stop immediately on all resolved, no candidates, or failed score convergence. No forced extra labels to make a triplet.',
        '- Main budget: four chains × 2,048 configurations per fresh round. Budget sensitivity: 512 and 1,024 per chain, same frozen parameters and paired prefixes.',
        '- Select separately by mixing-aware lower IoU (balanced), full-mask IoU, BPM, and promotion reliability subject to calibration quality/coverage constraints where feasible.',
        f'- {traces:,} comparison trajectory records; {sum(runtime.values())/3600:.2f} summed worker-hours of offline reanalysis. Existing Gibbs banks are reused; these times are not new state-preparation times.','',
        '## Definitions and controls','',
        'All scores are normalized to [-1,1]. The pairwise control divides the original foreground-minus-background mean by the number of nonempty roles. Every promoted pixel is evaluated with its own label removed; only the initial trusted label is retained. Thus the numerical baseline here differs from historical reports. The same convention applies to every variant.',
        'Three-seed majority evaluates Zi × majority(ya Za,yb Zb,yc Zc) sample by sample. Its mean is (ya Cia+yb Cib+yc Cic−ya yb yc Ciabc)/2 for distinct sites. Raw odd Z correlations vanish by global spin-flip symmetry for both finite zero-field Gibbs states. Soft conditioning uses A=va+vb+vc+va vb vc, B=1+va vb+va vc+vb vc and E[Zi A]/E[B], where vs=eta_s ys Zs. This is exact spin-flip symmetrization of the soft conditional ratio; it does not double the independent sample count. Ratio uncertainty retains numerator/denominator covariance using paired influence observations.',
        'One triple per target/round is selected using random priorities fixed before the round. Anchored groups contain the original seed when eligible. A promoted target is excluded from its own triple; if fewer than three other seeds exist its score falls back to leave-one-out pairwise inference. Before three total seeds, every method uses the same pairwise bootstrap.',
        'Gate variants retain pairwise segmentation and require an extra confident higher-order sign before promotion. Score variants blend the higher-order readout with the pair score for both segmentation and promotion. At most one seed and one anti-seed are selected uniformly per round. Truth is used only for the original label, calibration selection, and evaluation.',
        'Each selected variant is compared with a pair control having the same cutoff, margin, cap and stop; and an unweighted linear signed-triple vote using the same group-selection rule. This controls triplet selection. Soft conditioning also gives the trusted anchor a different strength from promoted seeds, so this unweighted linear control alone cannot attribute a soft-method gain specifically to the four-point term rather than seed weighting. Different methods can stop at different rounds, so actual retained configurations are reported rather than claiming identical consumed budgets.',
        'IoU and BPM evaluate tentative masks, including unresolved pixels. Statistical unresolved excludes only the original trusted pixel; mixing-aware unresolved is 100% for failed final score convergence. BPM is the inherited Zhang skeleton-proximity IoU with Euclidean radius 1, not ordinary object-outline accuracy; radius 0 and 2 are also saved. One pixel represents a different image fraction at the two resolutions, so boundary comparisons should be made within a resolution. Wrong-promotion rate pools wrong promoted labels / all promoted labels; no-promotion runs do not count as perfect precision.',
        'Errors use 16 blocks per chain and the larger of block and between-chain SE; nominal Bonferroni-t correction covers pixels × ten rounds within one trajectory. Direct score/influence split-Rhat and inherited scalar guard must be below 1.05. Soft weighting requires Kish effective weight count ≥100; this does not substitute for autocorrelation ESS. Prefix scalar guards are inherited from full banks; direct readout diagnostics are recomputed. MCMC/delta-method coverage and pointwise bootstrap intervals are exploratory, not simultaneous across the sweep.','',
        '## Balanced selections: primary comparison','']
    for side in (16,8):
        lines += [f'### {side}×{side}','', '| Model / start | Method | IoU | U | Mixing U | BPM r=1 | Wrong promotions | Promoted / image | Configurations | Activated |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for anchor in anchors():
            for family in TITLES:
                r=rows[f'{anchor}/{side}/2048/{family}/balanced'];error='n/a' if r['wrong_promotion_rate'] is None else f'{100*r["wrong_promotion_rate"]:.2f}%'
                lines.append(f'| {anchor} | {TITLES[family]} | {r["iou"]:.4f} | {100*r["unresolved"]:.2f}% | {100*r["mixing_unresolved"]:.2f}% | {r["bpm_iou_r1"]:.4f} | {error} | {r["promoted_count"]:.2f} | {r["shots"]:.0f} | {100*r["activated"]:.1f}% |')
        lines += ['']
    lines += ['![Matched-control effects](results/higher-seeds/figures/matched-effects-16.png)','',
              '### Paired mechanism effects, 16×16','',
              '| Model / start | Method | ΔIoU [95% CI] | ΔU, pp [95% CI] | Δwrong promotions, pp [95% CI] |',
              '|---|---|---:|---:|---:|']
    for anchor in anchors():
        for family in list(TITLES)[1:]:
            label=family+'/balanced';r=effects[f'{anchor}/16/2048/{label}'][label+'/matched_pair']
            def fmt(metric,scale):
                v=r[metric]
                return 'n/a' if v is None else f'{scale*v["mean"]:+.3f} [{scale*v["ci95"][0]:+.3f}, {scale*v["ci95"][1]:+.3f}]'
            lines.append(f'| {anchor} | {TITLES[family]} | {fmt("iou",1)} | {fmt("unresolved",100)} | {fmt("wrong_promotion_rate",100)} |')
    lines += ['','Intervals bootstrap the 75 source images, after averaging promotion replicates within source. They are unadjusted pointwise intervals. Matched linear-control effects are included in the machine-readable summary.','',
              '### Equal-consumed-budget one-seed controls, 16×16','',
              '| Model / start | Method | Adaptive IoU | Pooled one-seed IoU | ΔIoU [95% CI] |',
              '|---|---|---:|---:|---:|']
    for anchor in anchors():
        for family in list(TITLES)[1:]:
            label=family+'/balanced';name=f'{anchor}/16/2048/{label}';key=label+'/pooled_one_seed'
            if key not in effects[name]:continue
            d=effects[name][key]['iou'];a=rows[name];b=rows[name+'/pooled_one_seed']
            lines.append(f'| {anchor} | {TITLES[family]} | {a["iou"]:.4f} | {b["iou"]:.4f} | {d["mean"]:+.3f} [{d["ci95"][0]:+.3f}, {d["ci95"][1]:+.3f}] |')
    lines += ['','The one-seed control pools precisely the independent rounds consumed by each adaptive trajectory; no extra trusted labels are supplied. Both methods have identical retained-configuration counts within each pairing.','',
              '## Alternative calibration objectives, 16×16','',
              '| Model / start | Objective | Selected new method | IoU | U | BPM | Wrong promotions | Configurations |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for anchor in anchors():
        for objective,metric in [('iou','iou'),('boundary','bpm_iou_r1'),('reliability','promoted_wrong')]:
            eligible=[(k,v) for k,v in chosen[anchor].items() if k.endswith('/'+objective) and not k.startswith('pair/')]
            if objective=='reliability':
                def rank(kv):
                    c=kv[1]['calibration'];return c['promoted_wrong']/c['promoted_count'] if c['promoted_count'] else 2.
            else:
                def rank(kv):return -kv[1]['calibration'][metric]
            label,v=min(eligible,key=rank);r=rows[f'{anchor}/16/2048/{label}'];er=r['wrong_promotion_rate']
            lines.append(f'| {anchor} | {objective} | {TITLES[label.split("/")[0]]} | {r["iou"]:.4f} | {100*r["unresolved"]:.2f}% | {r["bpm_iou_r1"]:.4f} | {100*er:.2f}% | {r["shots"]:.0f} |' if er is not None else f'| {anchor} | {objective} | {label} | no promotions |')
    lines += ['','The method in each row is selected from calibration metrics, not chosen for its comparison score. Every family/objective result is retained in metrics.csv and higher-seeds-summary.json.','',
              '![Soft parameter sensitivity](results/higher-seeds/figures/calibration-soft-parameters.png)','',
              '## Sampling cost','',
              '![Sampling budget](results/higher-seeds/figures/sampling-cost-16.png)','',
              'All observables use the same Z-basis configurations; no new measurement basis or Hamiltonian evolution is required. A round costs 4 × samples retained configurations. Ising draws are independent; XXZ configurations are correlated and score ESS is reported separately in the data. For the fixed-budget XXZ banks, a prospective round at budget S requires 4(2000+2S) SSE sweeps including burn-in; the summary also records this estimate. A sweep is not a fixed number of operations across Hamiltonians. Soft conditioning can reduce effective weight support; higher-order readout is not free in statistical precision. Reported configuration counts exclude unused precomputed bank tails and are per segmentation run; the numerical study reuses banks across methods and promotion replicates. The fixed-budget sweep reused existing banks, so it does not newly measure Gibbs-generation wall time. The confirmation adds two banks and uses burn-in 6000. Quantum state-preparation depth is not estimated.','',
              '## Selected balanced parameters','']
    for anchor in anchors():
        for family in TITLES:
            p=chosen[anchor][family+'/balanced']['policy'];lines.append(f'- {anchor} / {TITLES[family]}: `{json.dumps(p,sort_keys=True)}`')
    lines += ['','## Do the three seeds provide different votes?','',
              '| Model / start | Method | Images with triplets | Mean P(all three signed votes agree) | Triplets with at least two wrong labels |',
              '|---|---|---:|---:|---:|']
    for anchor in anchors():
        for family in list(TITLES)[1:]:
            d=diagnostic.get(f'{anchor}/16/{family}/balanced')
            if d:
                lines.append(f'| {anchor} | {TITLES[family]} | {d["images_with_triplets"]} | {100*d["agreement_probability"]:.2f}% | {100*d["majority_wrong_labels"]:.2f}% |')
    lines += ['','This diagnostic uses the shared triplet for unseeded targets, excluding per-seed leave-one-out alternatives. It averages groups within each image, then weights images with any triplet equally. It is descriptive and includes terminal-round evaluations. Semantic label errors are measured only afterward using annotations.',
              'For three signed ±1 votes, majority equals their linear average whenever all three agree. Consequently the absolute difference between their candidate correlation scores is at most (2/3) P(the votes do not all agree). For perfectly aligned signed seeds, soft conditioning reduces to a positive rescaling of their common seed correlation, with unit scale when the trusted hard anchor is included. Strong internal agreement therefore need not add new segmentation information or establish semantic correctness.','',
              '## Reproduction','',
              'Requires the prepared Weizmann case files and the existing independent-round snapshot banks in algorithm1/results/promotion-reliability/banks. Input images and banks are SHA-verified. These are reused data dependencies, not regenerated by the readout runner. Run from the repository root with the existing .venv:',
              '```sh', '.venv/bin/python -m pytest algorithm1/tests/test_higher_seed*.py algorithm1/tests/test_iterative_seed_gibbs.py algorithm1/tests/test_promotion_reliability.py -q',
              '.venv/bin/python algorithm1/python/run_higher_seed_study.py init',
              '.venv/bin/python algorithm1/python/run_higher_seed_study.py calibrate --workers 4',
              '.venv/bin/python algorithm1/python/run_higher_seed_study.py select',
              '.venv/bin/python algorithm1/python/run_higher_seed_bounded.py test --workers 4',
              '.venv/bin/python algorithm1/python/run_higher_seed_bounded.py budget --workers 4',
              '.venv/bin/python algorithm1/python/audit_higher_seed_study.py controls --workers 3',
              '.venv/bin/python algorithm1/python/audit_higher_seed_study.py audit',
              '.venv/bin/python algorithm1/python/analyze_higher_seed_study.py',
              '.venv/bin/python algorithm1/python/run_higher_seed_confirmation.py',
              '.venv/bin/python algorithm1/python/analyze_higher_seed_confirmation.py', '```','',
              'Validation: 46 unit/integration tests passed. The main audit checked 161,250 trajectory records and replayed 24 bank/setting combinations with zero discrepancy. The confirmation has its own audit below. Reproduction of the confirmation also requires the prior primary-XXZ confirmation banks listed in its protocol.','']
    (ROOT/'higher-seeds.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
