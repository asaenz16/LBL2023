"""Populate the research report from completed, audited study outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/iterative-seeds'


def number(v,d=3):return f'{v:.{d}f}'
def interval(e,d=3):return f"{number(e['mean'],d)} [{number(e['ci95'][0],d)}, {number(e['ci95'][1],d)}]"
def percent(v):return f'{100*v:.1f}%'
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def figure(name,caption):return f'![{caption}](results/iterative-seeds/figures/{name}.png)\n\n{caption}'


def main():
    data=json.loads((ROOT/'iterative-seeds-summary.json').read_text())
    audit=json.loads((ROOT/'iterative-seeds-validation.json').read_text())
    assert data['status']=='completed' and audit['status']=='passed'
    groups=data['groups'];winners=data['selection'];sections={}
    primary=[f'{model}/{start}/mixing_lower' for model in ('Ising','XXZ') for start in ('foreground','background')]
    comparisons=[]
    for size in (8,16):
        for label in primary:
            g=groups[f'{label}/{size}'];delta=g['adaptive_minus_control']['pooled_one_seed']['iou']
            relation='positive' if delta['ci95'][0]>0 else ('negative' if delta['ci95'][1]<0 else 'overlaps zero')
            comparisons.append((size,label,delta,relation))
    positive=sum(x[3]=='positive' for x in comparisons)
    negative=sum(x[3]=='negative' for x in comparisons)
    sections['FINDINGS']=f'''## Main findings

The complete finite grid finished: **41,472 parameter combinations**, evaluated on 25 calibration sources at both resolutions with three random promotion replicates (**6,220,800 endpoints**). Selected rules were then compared on **75 different source images**, at 8×8 and 16×16, with ten random promotion replicates. Both known-foreground and known-background starts are included.

Against the **pooled one-seed control with the same sample budget**, {positive} of the eight primary model/start/resolution comparisons have a positive paired-bootstrap IoU interval, {negative} have a negative interval, and {8-positive-negative} overlap zero. These are exploratory, unadjusted intervals; they are not eight independent confirmatory tests.

The tables below distinguish area accuracy, unresolved pixels, incorrect promotions, and convergence. A lower unresolved fraction alone does not establish a better segmentation. The Ising/XXZ headline comparison uses separately selected Hamiltonian rules; it is not an isolated causal effect of adding XX/YY exchange.'''
    main16=[groups[f'{label}/16'] for label in primary]
    errors=[g['pooled_promoted_label_error'] for g in main16]
    sections['FINDINGS']+=f'\n\n**The main obstacle is label reliability, not just unresolved pixels.** At 16×16, {percent(min(errors))}–{percent(max(errors))} of promoted labels are incorrect across the primary procedures. Their statistically resolved but semantically wrong fractions range from {percent(min(g["metrics"]["wrong_resolved"]["mean"] for g in main16))} to {percent(max(g["metrics"]["wrong_resolved"]["mean"] for g in main16))}. Correlation confidence therefore cannot be read as segmentation accuracy.'
    bgi=groups['Ising/background/mixing_lower/16'];bgx=groups['XXZ/background/mixing_lower/16']
    sections['FINDINGS']+=f'\n\nThe background-start procedures also leave **more** unresolved pixels than their pooled one-seed controls at 16×16: +{100*bgi["adaptive_minus_control"]["pooled_one_seed"]["unresolved"]["mean"]:.2f} percentage points for Ising and +{100*bgx["adaptive_minus_control"]["pooled_one_seed"]["unresolved"]["mean"]:.2f} for XXZ; both paired intervals exclude zero. Compared with the first round alone, resolution can improve, but the equal-budget comparison does not support a general resolution benefit from promotion.'
    sections['FINDINGS']+='\n\nStarting-label effects depend on the Hamiltonian rule. At 16×16, the selected Otsu/Ising procedure benefits from a background start relative to its fixed-parameter foreground counterpart; the q90 foreground/background contrasts are not clearly separated. There is no evidence here for one universally preferable starting class or a consistent XXZ advantage.'
    secondary=groups['XXZ/background/iou/16'];gain=secondary['adaptive_minus_control']['pooled_one_seed']['iou']
    loss=groups['XXZ/foreground/iou/16']['adaptive_minus_control']['pooled_one_seed']['iou']
    sections['FINDINGS']+=f'\n\nThe secondary selections contain a small exception: the g/D=0.5 background-start XXZ rule gains {interval(gain,4)} IoU over its pooled one-seed control, with {percent(secondary["metrics"]["unresolved"]["mean"])} unresolved ({percent(secondary["metrics"]["mixing_unresolved"]["mean"])} including mixing failures). The secondary foreground-start XXZ rule instead changes IoU by {interval(loss,4)}. These unadjusted exploratory contrasts do not establish a robust improvement from the modification.'
    sections['FINDINGS']+='\n\nThe next modification worth testing is promotion reliability: spatially separated candidates, agreement checks before promotion, and removal or downweighting of inconsistent seeds. Those are hypotheses for a subsequent experiment, not results demonstrated by this sweep.'
    rows=[]
    for label in primary:
        w=winners[label];r=w['rule'];p=w['policy']
        rows.append([label.replace('/mixing_lower','').replace('/',' / '),r['threshold'],r['sensitivity'],r['K'],r['strength'],p['cut_fg'],p['cut_bg'],p['gate'],p['stopping'],p['cap']])
    sections['SELECTION']='## Selected primary rules\n\n'+table(['Model / start','Threshold','Alpha rule','K','g/D','Seed cut','Anti cut','Gate','Stop','Cap'],rows)
    sections['SELECTION']+='\n\nSelection maximizes the calibration mixing-aware lower IoU. All secondary raw-IoU selections and their exact parameters are retained in the selection JSON.\n\n'+figure('calibration-envelope','Calibration envelope: best promotion/stopping policy at each Hamiltonian. These values are training performance, not comparison-set estimates.')
    rows=[]
    for size in (8,16):
        for label in primary:
            g=groups[f'{label}/{size}'];m=g['metrics']
            rows.append([f'{size}×{size}',label.replace('/mixing_lower','').replace('/',' / '),interval(m['iou']),percent(m['unresolved']['mean']),percent(m['mixing_unresolved']['mean']),number(m['bpm_iou_r1']['mean']),number(m['rounds']['mean'],2)])
    results='## Comparison results\n\n'+table(['Size','Model / start','IoU [95% interval]','Unresolved','Mixing-aware U','BPM-IoU r=1','Rounds'],rows)
    results+='\n\nIntervals resample source images after averaging the ten promotion replicates. “Mixing-aware U” marks every non-initial pixel unresolved in a failed-convergence trajectory.\n\n'+figure('selected-results','Primary selected rules: mean performance and source-bootstrap intervals.')
    rows=[]
    for size in (8,16):
        for start in ('foreground','background'):
            d=data['selected_xxz_minus_selected_ising'][f'{start}/{size}']
            rows.append([f'{size}×{size}',start,interval(d['iou']),interval(d['unresolved']),interval(d['bpm_iou_r1'])])
    results+='\n\n### Selected XXZ versus selected Ising\n\n'+table(['Size','Start','XXZ − Ising IoU [95%]','XXZ − Ising U [95%]','XXZ − Ising BPM [95%]'],rows)
    results+='\n\nThese are source-paired contrasts between separately optimized procedures. Their Hamiltonian rules and promotion/stopping policies can differ, so they compare the selected methods rather than exchange alone.'
    rows=[]
    for size,label,d,relation in comparisons:
        g=groups[f'{label}/{size}'];rows.append([f'{size}×{size}',label.replace('/mixing_lower','').replace('/',' / '),number(g['controls']['initial']['iou']['mean']),number(g['controls']['pooled_one_seed']['iou']['mean']),number(g['metrics']['iou']['mean']),interval(d),number(g['controls']['oracle_two_seed']['iou']['mean'])])
    results+='\n\n### Sample-budget controls\n\n'+table(['Size','Model / start','Initial','Pooled original seed','Iterative','Iterative − pooled [95%]','Oracle two seeds'],rows)
    results+='\n\nThe oracle column has one additional ground-truth label. All columns use the same Hamiltonian within a row; pooled and oracle controls use the adaptive trajectory’s actual number of sampled rounds.\n\n'+figure('sample-budget-controls','IoU for the unchanged original seed, adaptive seed sets, and an additional correctly labeled seed.')
    rows=[]
    for size in (8,16):
        for label in primary:
            g=groups[f'{label}/{size}'];e=g['pooled_promoted_label_error'];f=g['promotion_by_class']['foreground'];b=g['promotion_by_class']['background']
            rows.append([f'{size}×{size}',label.replace('/mixing_lower','').replace('/',' / '),number(g['metrics']['promoted_count']['mean'],2),'n/a' if e is None else percent(e),'n/a' if f['error_fraction'] is None else percent(f['error_fraction']),'n/a' if b['error_fraction'] is None else percent(b['error_fraction']),percent(g['metrics']['wrong_resolved']['mean']),number(g['mean_within_image_promotion_iou_sd'])])
    results+='\n\n### Promotion errors and random selection\n\n'+table(['Size','Model / start','Added pixels','Wrong / all added','Wrong FG additions','Wrong BG additions','Wrong resolved pixels','Mean within-image IoU SD'],rows)
    results+='\n\nPromotion-error ratios pool counts across images and random trajectories; a trajectory with no additions contributes no promoted labels to the ratio. The final column measures variability from the random promotions on a shared Gibbs bank. It does not include all Gibbs-sampling variability.\n\n'+figure('round-trajectories','Round-by-round means. Stopped trajectories are carried forward; this is the evolution of the procedure, not ten fresh observations for every case.')
    rows=[]
    for size in (8,16):
        for label in primary:
            d=data['selected_start_minus_opposite_start'][f'{label}/{size}'];rows.append([f'{size}×{size}',label.replace('/mixing_lower','').replace('/',' / '),interval(d['iou']),interval(d['unresolved'])])
    results+='\n\n### Starting-label comparison at fixed parameters\n\n'+table(['Size','Rule selected for','Selected start − opposite start: IoU','Selected start − opposite start: U'],rows)
    results+='\n\nFor a foreground-selected row the contrast is FG minus BG; for a background-selected row it is BG minus FG. These paired comparisons keep the Hamiltonian, cutoffs, gate and stopping rule fixed. They complement the separately optimized starting conventions above.\n\n'+figure('segmentation-examples','Examples fixed before inspecting outcomes: the first four comparison source IDs, at 16×16, with promotion replicate zero.')
    rows=[]
    for size in (8,16):
        for model in ('Ising','XXZ'):
            for start in ('foreground','background'):
                g=groups[f'{model}/{start}/iou/{size}'];m=g['metrics'];rows.append([f'{size}×{size}',f'{model} / {start}',number(m['iou']['mean']),percent(m['unresolved']['mean']),percent(m['mixing_unresolved']['mean']),number(m['bpm_iou_r1']['mean']),number(m['rounds']['mean'],2)])
    results+='\n\n### Secondary raw-IoU selections\n\n'+table(['Size','Model / start','IoU','Unresolved','Mixing-aware U','BPM-IoU r=1','Rounds'],rows)
    results+='\n\nThese settings were also selected on calibration only. Their objective ignores uncertainty during selection, so they are kept separate from the primary conservative selections.'
    sections['RESULTS']=results
    sections['PARAMETER_EFFECTS']='## Cutoff and stopping tradeoffs\n\n'+figure('cutoff-slices','Calibration cutoff slices hold the selected Hamiltonian, gate, stopping rule and cap fixed while varying positive and negative promotion thresholds independently.')
    sections['PARAMETER_EFFECTS']+='\n\n'+figure('stopping-tradeoffs','Calibration stopping comparisons hold the Hamiltonian, both cutoffs and eligibility gate fixed. Markers denote caps of 3 (triangle), 6 (square), and 10 (circle); horizontal position is actual mean rounds. Vertical axes are zoomed, so inspect the numerical scale when judging the small accuracy differences.')
    stopping=json.loads((ROOT/'iterative-seeds-stopping-slices.json').read_text())
    span=max(max(r['metrics']['iou'] for r in a)-min(r['metrics']['iou'] for r in a) for a in stopping.values())
    round_span=max(max(r['metrics']['rounds'] for r in a)-min(r['metrics']['rounds'] for r in a) for a in stopping.values())
    sections['PARAMETER_EFFECTS']+=f'\n\nAcross these four fixed-parameter slices, the largest spread in calibration IoU among stopping/cap choices is only **{100*span:.3f} percentage points**, while the largest spread in mean rounds is **{round_span:.2f}**. Here stopping has a much larger effect on sampling cost than on area accuracy. This is a calibration observation, not an independently confirmed universal rule.'
    reliability=json.loads((ROOT/'iterative-seeds-calibration-reliability.json').read_text())
    rows=[[r['rule']['strength'],r['rule']['K'],number(r['metrics']['iou']),percent(r['metrics']['unresolved']),percent(r['metrics']['converged']),number(r['metrics']['mixing_lower'])] for r in reliability]
    sections['PARAMETER_EFFECTS']+='\n\n### Temperature and exchange versus sampling reliability\n\n'+table(['g/D','K','Calibration IoU','Unresolved','Passed Rhat','Mixing-aware lower IoU'],rows)
    sections['PARAMETER_EFFECTS']+='\n\nThis slice fixes a foreground start, q90 threshold and alpha=0.01, then selects the best conservative promotion/stopping policy separately within each row. Ising samples are independent; XXZ samples use MCMC. Many colder XXZ trajectories fail the calibration mixing guard. Consequently, the fall in conservative score at K=1.2 is partly a sampling limitation and does not establish that the exact cold Gibbs state is intrinsically worse for segmentation. The primary selections compare procedures at the stated numerical budgets. Caps of three rounds also make the count-stabilization rules equivalent to the cap itself, so a selected tie between those rules should not be interpreted as evidence for a particular stopping mechanism.'
    rows=[]
    for size in (8,16):
        for label in primary:
            g=groups[f'{label}/{size}'];c=g['collapse_fractions'];rows.append([f'{size}×{size}',label.replace('/mixing_lower','').replace('/',' / '),g['confirmed_cases'],g['cases_with_any_failed_trajectory'],g['cases_with_any_ess_below100'],percent(c['all_foreground']),percent(c['all_background']),percent(c['no_foreground_seed']),percent(c['no_background_seed'])])
    sections['RELIABILITY']=f'''## Reliability and degenerate outcomes

The comparison contains **{data['test_model_cases']} Hamiltonian/image cases** across {data['unique_hamiltonians']} unique selected Hamiltonians. **{data['confirmation_runs']} cases** received longer independent confirmation runs and were replaced without retuning. Counts below are out of 75 source images for each row and include final confirmation outcomes. “Any” means at least one of the ten promotion trajectories.

'''+table(['Size','Model / start','Confirmed','Any failed Rhat','Any ESS <100','All FG','All BG','Never F','Never B'],rows)
    sections['RELIABILITY']+='\n\nCollapsed-mask and missing-class entries are trajectory fractions. A fully resolved mask can still be all foreground or all background. Remaining low-ESS or failed-Rhat outcomes are retained, not silently removed.\n\n'+figure('stopping-reasons','Termination reasons for the selected procedures, including failed sampling and lack of eligible promotion candidates.')
    sections['VALIDATION']=f'''## Validation

**{audit['tests']} tests passed**, including exact small-system Gibbs checks, score-moment identities, stopping rules, random-prefix equivalence, truth-independent promotions, and reuse of the established samplers. The final audit checked all **{audit['model_image_records']} saved model/image records**, **{audit['parameter_random_endpoints']:,} policy/replicate endpoints**, and {audit['test_confirmation_trajectories']:,} detailed comparison/confirmation traces. It independently recomputed final masks' metrics and promotion errors and replayed {audit['replayed_banks']} deterministically selected measurement banks. Maximum metric discrepancy: {audit['maximum_metric_discrepancy']:.3g}.

Protocol, source, input and measurement-bank hashes were verified. These checks validate implementation consistency; they do not prove finite-time MCMC convergence or generalization beyond the reused Weizmann development split.'''
    path=ROOT/'iterative-seeds.md';text=path.read_text()
    text=text.replace('Experiment started 2026-10-09. Status: calibration running. Numerical results will be inserted only after the frozen evaluation and confirmation stages finish.','Experiment completed 2026-10-09. Status: full calibration, comparison, confirmation and implementation audit completed.')
    for key,value in sections.items():
        start=f'<!-- {key} -->';end=f'<!-- END {key} -->'
        if end in text:
            a=text.index(start)+len(start);b=text.index(end,a);text=text[:a]+'\n\n'+value+'\n\n'+text[b:]
        else:text=text.replace(start,start+'\n\n'+value+'\n\n'+end)
    path.write_text(text)
    print(path)


if __name__=='__main__':main()
