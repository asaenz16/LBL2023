"""Summarize completed Ising Petz experiments and draw scientific figures."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quantum-segmentation-matplotlib')
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ising-petz'
WIDTHS = (4, 8, 16, 32, 64)


def read(path):
    return json.loads(path.read_text())


def verify_hashes(hashes):
    for name, expected in hashes.items():
        actual = hashlib.sha256((ROOT / 'python' / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Experiment source has changed: {name}')


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=190, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    study = [read(p) for p in sorted((OUT / 'study').glob('*.json'))]
    long = [read(OUT / 'long' / f'L64-K{k:g}.json') for k in (.8, 1.2)]
    if len(study) != 12:
        raise ValueError('Expected ten plus-x and two all-up experiments.')
    for row in study:
        verify_hashes(row['signature']['code_hashes'])
    for row in long:
        verify_hashes(row['code_hashes'])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    colors = dict(zip(WIDTHS, plt.get_cmap('viridis')(np.linspace(.08, .88, 5))))
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), layout='constrained')
    for r, k in enumerate((.8, 1.2)):
        for width in WIDTHS:
            row = next(s for s in study if s['signature']['k'] == k
                       and s['signature']['width'] == width
                       and s['signature']['initial'] == 'plus_x')
            t = np.array([v['sweep'] for v in row['records']])
            far = np.array([v['far_aligned'] for v in row['records']])
            se = np.array([v['far_se'] for v in row['records']])
            color = colors[width]
            axes[r, 0].plot(t, far, '.-', color=color, label=f'2 x {width}')
            axes[r, 0].fill_between(t, far-2*se, far+2*se, color=color, alpha=.1)
            axes[r, 0].axhline(row['exact_gibbs_far'], color=color, ls='--', alpha=.6, lw=.9)
            axes[r, 1].plot(t, [v['gamma_linf'] for v in row['records']], '.-', color=color)
            axes[r, 2].plot(t, [v['correct_resolved_fraction'] for v in row['records']], '.-', color=color)
        axes[r, 0].set_ylabel(f'K = {k}\n'+r'$-\Gamma_{\rm seed,last}$')
        axes[r, 1].set_ylabel(r'$\max_{s,p}|\widehat\Gamma_{sp}-\Gamma^{\rm Gibbs}_{sp}|$')
        axes[r, 1].set_yscale('log')
        axes[r, 1].set_ylim(1e-4, .3)
        axes[r, 2].set_ylabel('Correct and resolved fraction')
        axes[r, 2].set_ylim(-.03, 1.04)
        for ax in axes[r]:
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(0, 220)
            ax.set_xticks([0, 1, 10, 50, 200], ['0', '1', '10', '50', '200'])
            ax.set_xlabel('Local-bath sweeps')
            ax.grid(alpha=.2)
    axes[0, 0].set_title('End-to-end correlation; dashed = exact Gibbs')
    axes[0, 1].set_title('Distance of correlations from Gibbs')
    axes[0, 2].set_title('Segmentation with simultaneous sign bounds')
    axes[0, 0].legend(ncols=2, fontsize=8, loc='lower right')
    fig.suptitle('Immediate-neighbor Ising Petz circuit: size and temperature\n'
                 'Threshold 100; alpha 0.001; initial |+x>; 65,536 independent population trajectories', fontsize=13)
    save(fig, 'strip-study')

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.5), layout='constrained')
    for r, row in enumerate(long):
        t = np.array([v['sweep'] for v in row['records']])
        far = np.array([v['far_aligned'] for v in row['records']])
        se = np.array([v['far_se'] for v in row['records']])
        color = '#176a91' if r == 0 else '#ae4c27'
        axes[r, 0].plot(t, far, 'o-', color=color, label='Circuit-channel estimate')
        axes[r, 0].fill_between(t, far-2*se, far+2*se, color=color, alpha=.2)
        axes[r, 0].axhline(row['exact_gibbs_far'], color='black', ls='--', label='Exact Gibbs')
        axes[r, 0].set_ylabel(f'K = {row["k"]}\n'+r'$-\Gamma_{\rm seed,last}$')
        axes[r, 1].semilogy(t, [v['gamma_linf'] for v in row['records']], 'o-', color=color)
        axes[r, 1].set_ylabel('Correlation infinity-norm error')
        axes[r, 1].set_ylim(5e-4, .3)
        axes[r, 2].plot(t, [v['iou'] for v in row['records']], 's--', color='gray', label='Raw sign IoU')
        axes[r, 2].plot(t, [v['correct_resolved_fraction'] for v in row['records']], 'o-', color=color,
                        label='Correct + resolved fraction')
        axes[r, 2].set_ylim(-.03, 1.04)
        axes[r, 2].set_ylabel('Segmentation score')
        for ax in axes[r]:
            ax.set_xscale('symlog', linthresh=50)
            ax.set_xlim(0, 2100)
            ax.set_xticks([0, 50, 200, 500, 2000], ['0', '50', '200', '500', '2000'])
            ax.set_xlabel('Local-bath sweeps')
            ax.grid(alpha=.2)
    axes[0, 0].set_title('End-to-end correlation (bands: +/- 2 SE)')
    axes[0, 1].set_title('Comparison with exact transfer reference')
    axes[0, 2].set_title('Correct signs can precede Gibbs equilibration')
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8, loc='lower right')
    fig.suptitle('2 x 64 image: 2,000 local-bath sweeps\n'
                 'K=0.8: 131,072 trajectories; K=1.2: 65,536 trajectories; initial |+x>', fontsize=13)
    save(fig, 'long-strip')

    # Never hide inconclusive correlation signs behind a visually perfect mask.
    fig, axes = plt.subplots(5, 2, figsize=(13, 5.6), layout='constrained')
    truth = np.tile((np.arange(64)//2) % 2, (2, 1))
    for col, row in enumerate(long):
        data = np.load(OUT / 'long' / f'L64-K{row["k"]:g}.npz')
        axes[0, col].imshow(truth, cmap='gray', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        axes[0, col].set_title(f'K = {row["k"]}: input image')
        for j, sweep in enumerate((50, 200, 500, 2000), 1):
            idx = next(i for i, v in enumerate(row['records']) if v['sweep'] == sweep)
            gamma = data['gamma'][idx, 0].reshape(2, 64)
            pred = (gamma <= 0).astype(float)
            rgb = np.repeat(pred[:, :, None], 3, axis=2)
            rgb[abs(gamma) <= row['radius']] = [.94, .66, .16]
            axes[j, col].imshow(rgb, aspect='auto', interpolation='nearest')
            axes[j, col].set_title(f'{sweep:,} sweeps; raw IoU {row["records"][idx]["iou"]:.3f}', fontsize=10)
        for ax in axes[:, col]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Algorithm 1 segmentation from a black seed at the top-left\n'
                 'Black/white: resolved labels; gold: sign unresolved by the 95% simultaneous bound', fontsize=13)
    save(fig, 'segmentation')

    summary = {
        'model': {'hamiltonian': '-sum atan(alpha*(threshold-distance))*Zi*Zj/4',
                  'threshold': 100, 'alpha': .001, 'boundary': 'open',
                  'initial': 'plus_x unless explicitly stated',
                  'temperature': 'b=beta_paper/N; K=b*median(abs(Delta))/4',
                  'image': 'two copies of [0,0,255,255,...]'},
        'methods': {'small': 'QCLAB gate-derived full density simulation, 8 system spins',
                    'population_exact': '16 spins, 50 sweeps, cross-validated MATLAB and Python',
                    'large': 'independent population trajectories of the gate-validated channel',
                    'reference': 'exact 4x4 rung transfer contraction',
                    'bound': '95% Hoeffding simultaneous across pixels for primary seed at each checkpoint; not across time',
                    'gamma_linf': 'maximum across both seeds and all pixels; includes sampling error'},
        'resources': read(OUT / 'resources.json'),
        'matlab_validation': read(ROOT / 'ising-petz-matlab-validation.json'),
        'study': study, 'long': long,
        'scope': 'Unfrustrated binary ladders, ideal operations; no hardware noise or generic-image universality test.'}
    installation = ROOT / 'ising-petz-qclab-installation.json'
    if installation.exists():
        summary['qclab_installation'] = read(installation)
    (ROOT / 'ising-petz-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for row in study:
        if row['signature']['initial'] == 'plus_x':
            selected = [v for v in row['records'] if v['sweep'] in (50, 200)]
            print(row['signature']['width'], row['signature']['k'], selected)
    print('Saved validated summary and three figure sets to', OUT)


if __name__ == '__main__':
    main()
