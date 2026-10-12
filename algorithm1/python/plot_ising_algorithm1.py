"""Scientific plots and compact export for the exact Ising alpha sweep."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'quantum-segmentation-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig, root, name):
    fig.savefig(root/f'{name}.png', dpi=180)
    fig.savefig(root/f'{name}.svg'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'ising-algorithm1')
    p.add_argument('--summary', type=Path)
    args = p.parse_args(); root = args.input
    report = json.loads((root/'report.json').read_text()); groups = report['groups']; rows = report['records']
    alphas = report['protocol']['alphas']; bs = report['protocol']['bs']; magnitudes = sorted(a for a in alphas if a > 0)
    main_b = 4.
    def group(name, alpha, b=main_b):
        return next(g for g in groups if g['group'] == name and g['alpha'] == alpha and g['b'] == b)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained', gridspec_kw={'width_ratios': [1.45, 1]})
    labels = ['clean', 'noise20', 'noise50', 'noise80']
    values = np.array([[100*group(name, a)['mean_nonseed_accuracy'] for a in alphas] for name in labels])
    im = axes[0].imshow(values, vmin=0, vmax=100, cmap='viridis', aspect='auto')
    for i in range(len(labels)):
        for j in range(len(alphas)):
            v = values[i,j]; axes[0].text(j, i, f'{v:.0f}' if v == 100 else f'{v:.1f}',
                ha='center', va='center', fontsize=7, color='white' if v < 55 else 'black')
    axes[0].set(xticks=range(len(alphas)), xticklabels=[f'{a:g}' for a in alphas],
        yticks=range(4), yticklabels=['Clean (13)', 'Noise σ=20 (15)', 'Noise σ=50 (15)', 'Noise σ=80 (15)'],
        xlabel='α in the literal proposed formula', title='Exact accuracy at b=β/N=4 (%)')
    axes[0].tick_params(axis='x', rotation=60)
    axes[0].axvline(alphas.index(0)-.5, color='white', lw=1)
    axes[0].axvline(alphas.index(0)+.5, color='white', lw=1)
    fig.colorbar(im, ax=axes[0], shrink=.75)
    for b in bs:
        margins = np.array([group('clean', -a, b)['worst_aligned_gamma'] for a in magnitudes])
        axes[1].loglog(magnitudes, np.where(margins > 1e-12, margins, np.nan), '.-', label=f'b={b:g}')
    axes[1].axhline(.25, color='black', ls=':', lw=.8)
    axes[1].set(xlabel='|α|, with α < 0', ylabel='Minimum correctly aligned Γ across clean images',
                title='Correct signs can still have tiny magnitudes')
    axes[1].grid(alpha=.2, which='both'); axes[1].legend()
    fig.suptitle('Ising Algorithm 1 • Δ=arctan[α(d−125)], H=−Σ Δ ZZ/4\nAccuracy excludes the seed, averages all seed choices and then images; Γ=⟨ZpZq⟩/4', fontsize=11)
    save(fig, root, 'alpha-sweep')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    d = np.linspace(0,255,800)
    for i, a in enumerate([.003, .03, .3]):
        axes[0].plot(d, np.arctan(a*(d-125)), color=f'C{i}', label=f'α=+{a:g}')
        axes[0].plot(d, np.arctan(-a*(d-125)), color=f'C{i}', ls='--', label=f'α=−{a:g}')
    axes[0].axvline(125, color='black', lw=.6); axes[0].axhline(0, color='black', lw=.6)
    axes[0].set(xlabel='Neighbor color distance d (raw channel units)', ylabel='Δ (radians)', title='The sign of α reverses alignment preferences')
    axes[0].legend(fontsize=8, ncol=2)
    for b in bs:
        axes[1].semilogx(magnitudes, [100*group('noise80', -a, b)['mean_nonseed_accuracy'] for a in magnitudes],
                        '.-', label=f'b={b:g}')
    axes[1].set(xlabel='|α|, with α < 0', ylabel='Mean accuracy excluding seed (%)', ylim=(85, 101),
                title='Strong-noise cases: sharper is not always better')
    axes[1].legend(); axes[1].grid(alpha=.2)
    save(fig, root, 'couplings-temperature')

    worst = min([r for r in rows if r['group'] == 'noise80' and r['b'] == 4 and r['alpha'] == -.03],
                key=lambda r: r['seed0_accuracy'])['name']
    examples = ['T4', 'strip8', 'rgb-T4', worst, 'low-contrast4', 'threshold-blocks8']
    fig, axes = plt.subplots(len(examples), 4, figsize=(10.5, 10.5), layout='constrained')
    for ri, name in enumerate(examples):
        with np.load(root/f'{name}.npz') as f:
            image, truth, aa, bb, gammas = (f[k] for k in ['image', 'labels', 'alphas', 'bs', 'gamma'])
        ai_plus = int(np.flatnonzero(aa == .03)[0]); ai_minus = int(np.flatnonzero(aa == -.03)[0]); bi = int(np.flatnonzero(bb == 4)[0])
        gp, gm = gammas[ai_plus, bi, 0].reshape(truth.shape), gammas[ai_minus, bi, 0].reshape(truth.shape)
        same_truth = truth == truth.ravel()[0]; prediction = gm > 0
        axes[ri,0].imshow(image/255 if image.ndim == 3 else image, cmap='gray', vmin=0, vmax=255 if image.ndim == 2 else 1)
        for ax, g in zip(axes[ri,1:3], [gp, gm]):
            cmap = ax.imshow(g, cmap='RdBu', vmin=-.25, vmax=.25)
            ax.set_xlabel(f'Seed-0 accuracy {100*np.mean((g > 0) == same_truth):.1f}%', fontsize=8)
        axes[ri,3].imshow(np.where(prediction, 0., 1.), cmap='gray', vmin=0, vmax=1)
        yy, xx = np.where(prediction != same_truth); axes[ri,3].scatter(xx, yy, marker='x', color='red', s=35)
        axes[ri,0].set_ylabel(name, fontsize=8)
        for ax in axes[ri]: ax.set(xticks=[], yticks=[])
        axes[ri,0].scatter([0], [0], facecolors='none', edgecolors='red', s=55)
    for ax, title in zip(axes[0], ['Input (seed circled)', 'Literal α=+0.03: Γ', 'Control α=−0.03: Γ', 'α=−0.03 segmentation']): ax.set_title(title, fontsize=10)
    fig.colorbar(cmap, ax=axes[:,1:3], shrink=.35, label='Γ(seed,p); positive = same class')
    fig.suptitle('Exact Gibbs, b=β/N=4 • red crosses mark wrong labels\nThe illustrated noisy image is the worst seed-0 case in the σ=80 batch at α=−0.03', fontsize=11)
    save(fig, root, 'segmentation-examples')

    if args.summary:
        selected = [r for r in rows if r['b'] == 4 and r['alpha'] in [-.03, .03]]
        clean_margins = [{**group('clean', -a), 'worst_sufficient_shots_all_pairs_95':
            max(r['sufficient_shots_all_pairs_95'] for r in rows if r['group'] == 'clean' and r['b'] == 4 and r['alpha'] == -a)} for a in magnitudes]
        compact = {k: report[k] for k in ['protocol', 'cases', 'groups', 'max_spin_flip_symmetry_residual',
                                        'max_bipartite_gauge_identity_error', 'runtime_seconds']}
        compact.update(evaluations=len(rows), selected_case_results=selected, clean_margins_at_b4=clean_margins,
                       illustrated_worst_noise_case=worst)
        args.summary.write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
