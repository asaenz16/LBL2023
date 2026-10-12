"""Append the supplementary reliability objective to the main experiment report."""
import json
from run_promotion_reliability import ROOT
from write_promotion_reliability_report import f,ci


def main():
    p=ROOT/'promotion-reliability.md';text=p.read_text();text=text.split('\n## Supplement: select directly for promotion reliability')[0]
    s=json.loads((ROOT/'promotion-reliability-targeted-summary.json').read_text());groups=s['groups'];chosen=s['selection']['winners']
    lines=['','## Supplement: select directly for promotion reliability','',
        'This additional selection was frozen after the main calibration sweep and before inspecting the comparison outcomes. It directly minimizes the pooled wrong-active-seed fraction. On calibration data it must retain at least half the original promotion count, lose no more than 0.01 mean IoU, and increase mixing-aware unresolved by no more than 0.02. These are calibration constraints, not guarantees on the comparison images.','',
        'The selected policy, its matched control, and the original rule are rerun together on the same Gibbs bank. Main confirmations are reused, and any newly failing supplementary trajectory triggers a longer bank. All comparisons below are paired within this supplementary experiment.','',
        '| Setting at 16×16 | Rule | IoU | Statistical U | Mixing U | Wrong active seeds | Active promotions/image | Mean rounds |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for a in chosen:
        for label in ('historical','reliability/matched','reliability'):
            r=groups[f'{a}|{label}|16'];m=r['metrics'];value=lambda k:m[k]['mean']
            lines.append(f'| {a.replace("/mixing_lower","")} | {label} | {f(value("iou"))} | {f(value("unresolved"),True)} | {f(value("mixing_unresolved"),True)} | {f(r["reliability"]["active_error"]["mean"],True)} | {value("promoted_count"):.2f} | {value("rounds"):.2f} |')
    lines += ['','| Setting | Size | ΔIoU vs original [95% CI] | Δwrong-seed fraction vs original, pp [95% CI] | ΔIoU vs matched [95% CI] |', '|---|---:|---|---|---|']
    for a in chosen:
        for size in (8,16):
            key=f'{a}|reliability|{size}';d=s['paired'][key]
            lines.append(f'| {a.replace("/mixing_lower","")} | {size} | {ci(d["historical"]["iou"])} | {ci(d["historical"]["active_error"],True)} | {ci(d["reliability/matched"]["iou"])} |')
    r=groups['XXZ/background/mixing_lower|reliability|16']
    matched=s['paired']['XXZ/background/mixing_lower|reliability|16']['reliability/matched']['iou']
    lines+=['',f'The XXZ background-start supplementary variant improves raw IoU over the equal-budget pooled one-seed control by {ci(r["versus_pooled"]["iou"])}. Against the matched adaptive control, however, the difference is {ci(matched)}. This comparison does not establish that the new safeguards caused the gain.','', 'Selected rules:','']
    for a,r in chosen.items():lines.append(f'- {a.replace("/mixing_lower","")}: `{json.dumps(r["policy"],sort_keys=True)}`')
    lines+=['',f'Supplement validation: {s["banks"]} image/Hamiltonian banks; {s["confirmed_banks"]} use longer sampling; {s["validated_traces"]} seed-transition and mask-metric audits passed. Complete results: [targeted summary](promotion-reliability-targeted-summary.json) and [frozen targeted selection](promotion-reliability-targeted-selection.json).',
        '', '![Reliability-targeted comparison](results/promotion-reliability/figures/reliability-targeted-16.png)','',
        '## Calibration reliability landscape','',
        'None of the 3,600 calibration policies that enabled removal retired a seed in the evaluated trajectories. Soft weights did change in some settings, especially XXZ. This is evidence about these settings and cutoffs, not a theorem that the removal procedure can never activate. A planted-inconsistent-seed test verifies that it activates when a confident contradiction is present.',
        '', '![Full calibration grid](results/promotion-reliability/figures/calibration-reliability.png)','',
        'Seed error and segmentation IoU are distinct objectives. More spatial coverage can introduce seeds from different semantic regions; consistency checks can preserve a confidently wrong but stable model pattern. Any proposed improvement should therefore be judged jointly by IoU, unresolved pixels, wrong-seed fraction, number promoted, and measurement cost.','']
    p.write_text(text+'\n'.join(lines))


if __name__=='__main__':main()
