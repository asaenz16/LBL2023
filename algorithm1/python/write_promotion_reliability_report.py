"""Render the completed experiment with denominators and paired controls."""
import json
from run_promotion_reliability import ROOT,OUT,selection,anchors


def f(x,percent=False):return '—' if x is None else f'{100*x:.2f}%' if percent else f'{x:.4f}'
def ci(v,percent=False):
    if v['mean'] is None:return '—'
    scale=100 if percent else 1
    return f"{v['mean']*scale:+.3f} [{v['ci95'][0]*scale:+.3f}, {v['ci95'][1]*scale:+.3f}]"


def main():
    s=json.loads((ROOT/'promotion-reliability-summary.json').read_text());v=json.loads((ROOT/'promotion-reliability-validation.json').read_text());groups=s['groups'];sel=selection()['winners']
    overview=[
        '**The tested safeguards did not produce a consistent segmentation improvement against matched controls.** Spatial separation increased wrong-promotion rates for foreground-start runs by 2.87 percentage points in Ising and 5.33 points in XXZ at 16×16; the pointwise paired intervals exclude zero. Agreement and weights mostly produced small or negligible mask changes.',
        '',
        'No real seed was retired in either calibration or the selected comparison runs. The leave-one-out check detects deliberately planted contradictions in the numerical tests, but stable model correlations can support incorrect image labels. The result concerns the tested settings and confidence criteria, not every possible consistency rule.',
        '',
    ]
    targeted=ROOT/'promotion-reliability-targeted-summary.json'
    if targeted.exists():
        t=json.loads(targeted.read_text());overview += [
            'A supplementary selection that explicitly minimizes seed error improves the Ising wrong-seed fraction from 32.94% to 28.22% with a foreground start and from 31.42% to 26.78% with a background start at 16×16. It promotes fewer seeds and does not materially improve IoU. For the foreground-start case the matched control is identical: the benefit comes from the stricter cutoff and stopping settings, not agreement or weight maintenance.',
            '',
            'In XXZ, unanimous agreement gives a modest foreground-start seed-error reduction against its matched control (1.29 percentage points, pointwise 95% CI 0.01–2.60), but increases unresolved pixels by 0.91 points and gives no clear IoU gain. This narrow, unadjusted interval is exploratory evidence; the effect does not generalize across the four model/start settings.',
            '',
            f'Validation passed: 33 tests, {v["counts"]["traces"]+t["validated_traces"]:,} trajectory records, and {v["counts"]["replayed_banks"]+t["replayed_banks"]} full-bank replays with zero metric discrepancy.',
            '',
        ]
    lines=['# Promotion reliability: spatial separation, agreement, and seed maintenance','',
        'Completed numerical Gibbs-state experiment on the Weizmann Segmentation Evaluation Database. This study changes promotion and readout while keeping the previously selected image Hamiltonians fixed.','',
        *overview, '## Scope and interpretation','',
        '- 25 calibration source images and 75 comparison source images, each resized to 8×8 and 16×16. The split has been used in earlier development: this is an exploratory comparison, not an independent final benchmark.',
        '- Four primary settings: Ising and XXZ, each with a known foreground seed or known background anti-seed. Four secondary settings test transfer without retuning.',
        '- 2,250 policies per primary setting; 9,000 setting-policy combinations and **1,350,000 calibration endpoints**, including three random-promotion replicates. Comparison uses ten replicates, averaged within each source image.',
        f'- {s["comparison_banks"]} comparison image/Hamiltonian banks and {s["confirmation_banks"]} longer confirmation banks. Ordinary banks contain ten independently initialized rounds. Longer confirmations for the secondary g/D=0.5 setting stop at the maximum requested cap of three; its unused future rounds do not affect any trajectory or control. The original ten-round seed spawning schedule and interval correction are unchanged.',
        '- Ising samples are exact independent Gibbs draws from conditional-frontier sampling. XXZ uses the existing validated stochastic-series-expansion sampler. Each ordinary round has four chains × 2,048 retained configurations; confirmation uses four × 8,192.',
        '- Primary policy selection maximizes calibration mean mixing-aware lower IoU. Secondary selections maximize raw IoU. Error rates below evaluate promotion reliability; policies were not selected by minimizing semantic seed error.',
        '- BPM r=1 uses the inherited Zhang skeleton and Euclidean-distance convention. It measures skeleton proximity, not ordinary outline IoU; one pixel represents a different image scale at the two resolutions.',
        '- IoU uses the full tentative binary mask, including unresolved pixels. Statistical unresolved excludes only the originally provided seed. Mixing-aware unresolved counts every unknown pixel unresolved when convergence checks fail.',
        '- Promotion error is total wrong active promoted seeds divided by total active promoted seeds, pooled over sources and random runs. Cases with no promotion contribute no seeds, not a fictitious 100% precision.',
        '- Paired 95% intervals bootstrap the 75 source images, averaging promotion randomness within image. They are pointwise, exploratory intervals without adjustment for multiple policy families. Neither Gibbs rounds nor promotion replicates count as extra independent images.','',
        '## Comparison results','',
        'The original row replays the previous rule with the same new random uniforms used by the variants. Minor differences from the prior report can arise from these random choices and the broader confirmation trigger. Each mechanism row was selected separately on calibration images.','']
    families=['historical','baseline/mixing_lower','spatial/mixing_lower','agreement/mixing_lower','remove/mixing_lower','weight/mixing_lower','combined/mixing_lower','any/mixing_lower']
    names=['Original','Retuned baseline','Spatial only','Agreement only','Removal only','Downweight only','All three','Selected modification']
    for size in (16,8):
        lines+= [f'### {size}×{size} images','', '| Model / initial seed | Rule | IoU | Statistical U | Mixing U | BPM r=1 | Active seed error | Active promoted / image |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for anchor in sel:
            for family,name in zip(families,names):
                row=groups[f'{anchor}|{family}|{size}'];m=row['metrics'];val=lambda k:m[k]['mean']
                lines.append(f'| {anchor.replace("/mixing_lower","").replace("/"," / ")} | {name} | {f(val("iou"))} | {f(val("unresolved"),True)} | {f(val("mixing_unresolved"),True)} | {f(val("bpm_iou_r1"))} | {f(row["reliability"]["active_error"]["mean"],True)} | {val("promoted_count"):.2f} |')
        lines+=['']
    lines+= ['![Ablation results](results/promotion-reliability/figures/ablation-16.png)','',
        '## Is the selected modification better than matched controls?','',
        'The matched baseline has the same cutoff, stopping rule, round cap, and random uniforms, but disables separation, agreement, and maintenance. This comparison isolates the selected mechanisms more cleanly than comparison with the original policy. Equal-budget one-seed controls pool exactly the samples consumed by each adaptive trajectory.','',
        '| Setting | Size | ΔIoU vs matched [95% CI] | ΔU vs matched, pp [95% CI] | ΔIoU vs pooled one seed [95% CI] |',
        '|---|---:|---|---|---|']
    for anchor in sel:
        for size in (8,16):
            key=f'{anchor}|any/mixing_lower|{size}';p=s['paired'][key]['matched'];r=groups[key]
            lines.append(f'| {anchor.replace("/mixing_lower","")} | {size} | {ci(p["iou"])} | {ci(p["unresolved"],True)} | {ci(r["versus_pooled"]["iou"])} |')
    lines+=['','![Matched mechanism effects](results/promotion-reliability/figures/matched-effects-16.png)','', 'In this plot, positive ΔIoU and negative Δwrong-seed fraction are favorable. Removal is exactly zero against matched controls because no seeds were retired; differences between its aggregate row and the original rule come from the chosen stopping settings.', '', '## What happened to inconsistent seeds?','',
        '| Setting at 16×16 | Rule | Retired / image | Wrong among retired | Fraction of all wrong promotions retired | Weighted seed error | No-promotion runs |',
        '|---|---|---:|---:|---:|---:|---:|']
    for anchor in sel:
        for family in ('remove/mixing_lower','weight/mixing_lower','combined/mixing_lower','any/mixing_lower'):
            r=groups[f'{anchor}|{family}|16'];rel=r['reliability']
            lines.append(f'| {anchor.replace("/mixing_lower","")} | {family.replace("/mixing_lower","")} | {r["metrics"]["retired_count"]["mean"]:.4f} | {f(rel["retirement_precision"]["mean"],True)} | {f(rel["wrong_seed_retirement_recall"]["mean"],True)} | {f(rel["weighted_error"]["mean"],True)} | {f(r["no_promotion"],True)} |')
    lines+=['','An absent retirement precision means no seed was removed. Removing no seeds does not establish that they were correct. A high-confidence seed can agree with the Hamiltonian while disagreeing with the ground-truth object.','',
        'For example, a new foreground seed j is admitted when its signed correlation contrast is positive. Immediately after admission, its leave-one-out contrast is exactly the contrast that admitted it, before sampling noise or other seed changes. A fresh round can detect unstable evidence, but a systematic, stationary model error remains consistent. Independent agreement therefore does not supply independent semantic supervision.','',
        '## Selected policies and Hamiltonians','',
        'Distances are fractions of the shorter image side: 0.125 is one pixel at 8×8 and two pixels at 16×16; 0.25 is two and four pixels respectively. `same` separates seeds of the same role; `all` separates both roles.','']
    for anchor,rows in sel.items():
        a=anchors()[anchor];lines += [f'### {anchor}','',f'- Hamiltonian rule: `{json.dumps(a["rule"],sort_keys=True)}`',f'- Original nomination rule: `{json.dumps(a["base"],sort_keys=True)}`']
        for family in ('spatial','agreement','remove','weight','combined','any'):
            lines.append(f'- {family}: `{json.dumps(rows[family+"/mixing_lower"]["policy"],sort_keys=True)}`')
        lines+=['']
    lines+=['## Rules tested','',
        'The full product comprises five spatial settings, five agreement settings, five maintenance settings, two cutoff choices, three round caps (3, 6, 10), and three stopping rules.','',
        '1. **Spatial separation:** no constraint; same-role exclusion at 1/8 or 1/4 side length; all-role exclusion at those distances. Distance must be strictly larger than the exclusion radius. Opposite-role candidates proposed in the same round also obey all-role separation. Selection is uniform among eligible candidates; role selection order is randomized.',
        '2. **Agreement:** no additional check; at least 75% or 100% of active seed votes confidently share the proposed sign; eligibility in two consecutive independent rounds; or 75% consensus plus two-round persistence. Uncertain votes count against consensus, rather than disappearing from the denominator.',
        '3. **Maintenance:** keep all; remove after one or two consecutive confident leave-one-out contradictions; or damp confidence weights using relative or absolute normalization. The initial trusted seed cannot be removed or downweighted. Removed seeds cannot return.',
        '4. **Cutoffs and stopping:** inherit the prior asymmetric cutoffs or use 0.75 for both roles. Stop at all-resolved, no good candidates, a convergence failure, or the cap; optional two-transition unresolved-count stability. The minimum-three-round audit waits for fresh evidence and pending maintenance, subject to cap and sampling failure. No seeds are added on a fully resolved round.','',
        'The readout is S_i = sum_F(w_s C_si)/D_F − sum_B(w_a C_ai)/D_B, where C_si=⟨Z_s Z_i⟩. Relative normalization uses D=sum(weights); absolute normalization uses D=number of active seeds. Either empty term is zero. Relative normalization cannot attenuate a class containing a single seed, because its weight cancels. Absolute normalization can.',
        'For a promoted seed j, remove its own contribution, recompute both class denominators, and test its signed leave-one-out score. A confident negative value counts as a contradiction. Soft maintenance sets a confidence target clip((signed score − interval radius)/0.5, 0.1, 1); the next weight is halfway between the current weight and target, when the change is at least 0.05. Updates affect only the next fresh round.','',
        'Main score intervals use the original nominal Bonferroni-t construction over pixels and ten rounds. Pairwise consensus uses the more conservative N²-pair correction. All checks require split-Rhat <1.05 and the scalar sampler guard. These are approximate MCMC uncertainty diagnostics, not rigorous simultaneous guarantees over the parameter search.','',
        '## Transfer and secondary selections','',
        '| Secondary setting, 16×16 | IoU | U | Active seed error | ΔIoU vs matched [95% CI] |', '|---|---:|---:|---:|---|']
    for anchor in anchors():
        if anchor.endswith('/iou'):
            key=f'{anchor}|transfer|16';r=groups[key];p=s['paired'][key]['matched']
            lines.append(f'| {anchor} | {f(r["metrics"]["iou"]["mean"])} | {f(r["metrics"]["unresolved"]["mean"],True)} | {f(r["reliability"]["active_error"]["mean"],True)} | {ci(p["iou"])} |')
    lines+=['','The complete JSON also contains raw-IoU-selected policies, all matched controls, both sizes, stopping frequencies, per-round trajectories, and pairing against historical baselines. Transfer changes the Hamiltonian and/or nomination cutoff to the earlier secondary anchor without retuning the selected modifier.','',
        '## Reproducibility and limitations','',
        f'- Validation: {v["counts"]["banks"]} banks checked; {v["counts"]["prefixes"]} reused prefixes verified byte-for-byte; {v["counts"]["traces"]} stored trajectories audited; {v["counts"]["replayed_banks"]} full deterministic bank replays. Maximum replayed metric discrepancy: {v["max_metric_replay_difference"]}.',
        '- Source and protocol SHA-256 hashes are frozen. Old experiment files are unchanged. Any missing independent sampling rounds were added under a new result directory with the identical original round seed schedule.',
        '- Confirmation replaces the entire case/Hamiltonian bank, including controls, whenever any evaluated trajectory has a convergence failure or score ESS below 100; five predetermined source IDs at 16 are also confirmed. Cases still failing are retained and marked, not discarded.',
        '- The results are for 8×8 and 16×16 numerical Gibbs sampling, not native-resolution segmentation, quantum-circuit execution, or an external validation set. Hyperparameters are fixed to the earlier selected anchors; this is not a fresh exhaustive Hamiltonian search.',
        '- Aggregate error ratios can change because different numbers or classes of pixels are promoted. Read them with promotion counts, no-promotion fractions, and IoU. Stopped trajectories are carried forward only for trajectory plots; no later samples change their masks.','',
        'Files: [protocol](promotion-reliability-protocol.json), [execution amendment](promotion-reliability-execution-amendment.json), [selection](promotion-reliability-selection.json), [summary](promotion-reliability-summary.json), [validation](promotion-reliability-validation.json), [core](python/promotion_reliability.py), [runner](python/run_promotion_reliability.py), [analysis](python/analyze_promotion_reliability.py), [tests](tests/test_promotion_reliability.py).','',
        'To reproduce using the project virtual environment, run `run_promotion_reliability.py` stages `init`, `calibrate`, `select`, `test`, and `confirm`; use `promotion_reliability_confirmation.py confirm` for the documented execution amendment, then run `analyze_promotion_reliability.py`, `audit_promotion_reliability.py`, and this report generator. Results resume only when signatures match.','']
    (ROOT/'promotion-reliability.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
