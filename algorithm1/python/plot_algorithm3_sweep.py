"""Plots and auditable tables for screening and independent confirmations."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from importlib.metadata import version
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'quantum-segmentation-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reports(root):
    return [json.loads(p.read_text()) for p in sorted(root.glob('*/report.json'))]


def first_accuracy(record):
    truth = np.array(record['truth_same_as_seed']).ravel()
    prediction = np.zeros(len(truth), bool); prediction[record['seeds']] = True
    first = record['rounds'][0]
    prediction[first['newly_frozen']] = True
    return float(np.mean(prediction == truth))


def export_summary(root, comparison, destination):
    stages = ['exact8', 'screen16', 'screen32', 'capped', 'confirm-baseline',
              'confirm-warm', 'confirm-runner-up', 'confirm-transfer16']
    compact = []
    fields = ['status', 'name', 'backend', 'decision', 'b', 'beta_paper', 'kappa',
              'sigma', 'sigma_ddof', 'seeds', 'replicas', 'bond', 'cutoff', 'random_seed',
              'sweeps_per_round', 'metrics', 'first_round_min_aligned_m',
              'first_round_far_aligned_m', 'uncertain_freeze_count',
              'max_discarded_weight_sum', 'max_probability_sum_error', 'runtime_seconds']
    refinement = []
    for stage in stages:
        for record in reports(root/stage):
            item = {key: record.get(key) for key in fields}; item['stage'] = stage
            if record['status'] == 'complete':
                item['first_round_accuracy'] = first_accuracy(record)
                item['rounds'] = len(record['rounds'])
                item['unresolved_active_count'] = len(record['unresolved_active'])
            else:
                item['config'] = record['config']; item['error'] = record.get('error')
            compact.append(item)
        for path in sorted((root/stage).glob('*/bond-refinement.json')):
            refinement.extend(json.loads(path.read_text()))
    output = {'normalization': 'b=beta_paper/N_original', 'initial_state': 'plus_x',
        'scope': 'Synthetic alternating-block strips; ideal local-Petz channel, strict adaptive freezing, no K cap except capped references.',
        'versions': {name: version(name) for name in ['numpy', 'scipy', 'quimb', 'qiskit', 'matplotlib']},
        'runs': compact, 'confirmation': comparison['confirmation'],
        'first_round_signals': comparison.get('first_round_signals', []), 'bond_refinement': refinement}
    for name in ['confirmation-selection', 'channel-validation']:
        path = root/f'{name}.json'
        if path.exists(): output[name.replace('-', '_')] = json.loads(path.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'algorithm3-sweep')
    parser.add_argument('--summary', type=Path, help='Optional compact numerical summary outside the raw-results directory.')
    args = parser.parse_args(); root = args.input
    screening = reports(root/'screen32')
    jobs = json.loads((root/'screen32'/'protocol.json').read_text())['jobs']
    bs = sorted({job['b'] for job in jobs})
    kappas = sorted({job['kappa'] for job in jobs})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    for ax, key, title, cmap, limits in zip(axes,
        ['accuracy', 'first_round_far_aligned_m', 'first_round_min_aligned_m'],
        ['Final segmentation accuracy (%)', 'Far-column aligned magnetization', 'Weakest aligned magnetization'],
        ['viridis', 'RdBu', 'RdBu'], [(0, 100), (-.5, .5), (-.5, .5)]):
        grid = np.full((len(bs), len(kappas)), np.nan)
        for record in screening:
            if record['status'] != 'complete': continue
            row, col = bs.index(record['config']['b']), kappas.index(record['config']['kappa'])
            grid[row, col] = 100*record['metrics'][key] if key == 'accuracy' else record[key]
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap, vmin=limits[0], vmax=limits[1])
        for i in range(len(bs)):
            for j in range(len(kappas)):
                value = grid[i, j]
                label = '—' if np.isnan(value) else (f'{value:.1f}' if key == 'accuracy' else f'{value:+.3f}')
                white = value < 45 if key == 'accuracy' else abs(value) > .28
                ax.text(j, i, label, ha='center', va='center', fontsize=9, color='white' if white else 'black')
        ax.set(xticks=range(len(kappas)), xticklabels=kappas, yticks=range(len(bs)), yticklabels=bs,
               xlabel=r'$\kappa$', ylabel=r'$b=\beta_{\rm paper}/N$', title=title)
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle('2×16 strip: 50 sweeps/round, 16 trajectories/round, bond 32\nExploratory grid; positive aligned magnetization favors the correct label', fontsize=12)
    fig.savefig(root/'parameter-screen.png', dpi=180)
    fig.savefig(root/'parameter-screen.svg'); plt.close(fig)

    by_size = [reports(root/'exact8'), reports(root/'screen16'), screening]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), layout='constrained')
    for ax, records, label in zip(axes, by_size, ['2×4: exact channel', '2×8: 16 trajectories', '2×16: 16 trajectories']):
        grid = np.full((len(bs), len(kappas)), np.nan)
        for record in records:
            if record['status'] != 'complete': continue
            row, col = bs.index(record['config']['b']), kappas.index(record['config']['kappa'])
            grid[row, col] = 100*record['metrics']['accuracy']
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)
        for i in range(len(bs)):
            for j in range(len(kappas)):
                value = grid[i, j]
                ax.text(j, i, '—' if np.isnan(value) else f'{value:.1f}', ha='center', va='center',
                        fontsize=9, color='white' if value < 45 else 'black')
        ax.set(xticks=range(len(kappas)), xticklabels=kappas, yticks=range(len(bs)), yticklabels=bs,
               xlabel=r'$\kappa$', ylabel=r'$b=\beta_{\rm paper}/N$', title=label)
    fig.colorbar(im, ax=axes, shrink=.7, label='Final segmentation accuracy (%)')
    fig.suptitle('Algorithm 3 with local Petz updates: 50 sweeps per round\nExploratory grid; sampled results require independent confirmation', fontsize=12)
    fig.savefig(root/'size-comparison.png', dpi=180)
    fig.savefig(root/'size-comparison.svg'); plt.close(fig)

    confirmation = []
    for directory in sorted(root.glob('confirm*')):
        confirmation.extend(reports(directory))
    good = [r for r in confirmation if r['status'] == 'complete']
    groups = {}
    for r in good:
        c = r['config']; key = (c['case'], c['b'], c['kappa'], c['decision'], c['replicas'])
        groups.setdefault(key, []).append(r)
    table = []
    for (case, b, kappa, decision, replicas), group in sorted(groups.items()):
        table.append({'case': case, 'b': b, 'beta_paper': group[0]['beta_paper'], 'kappa': kappa,
            'decision': decision, 'replicas_per_round': replicas,
            'seeds': [r['random_seed'] for r in group],
            'accuracies': [r['metrics']['accuracy'] for r in group],
            'first_round_accuracies': [first_accuracy(r) for r in group],
            'mean_accuracy': float(np.mean([r['metrics']['accuracy'] for r in group])),
            'perfect_runs': sum(r['metrics']['all_pixels_correct'] for r in group),
            'uncertain_freezes': [r['uncertain_freeze_count'] for r in group],
            'unresolved_active_counts': [len(r['unresolved_active']) for r in group],
            'max_discarded_weight_sum': max(r['max_discarded_weight_sum'] for r in group)})
    comparison = {'screening': screening, 'screening16': by_size[1],
                  'exact8': by_size[0], 'confirmation': table}
    (root/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    selected = [(key, group) for key, group in sorted(groups.items()) if key[0] == 'strip16' and key[3] == 'strict']
    if not selected:
        if args.summary: export_summary(root, comparison, args.summary)
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    signal_diagnostics = []
    for index, (key, group) in enumerate(selected):
        _, b, kappa, _, replicas = key
        label = f'b={b:g}, κ={kappa:g}'
        values = [r['metrics']['accuracy']*100 for r in group]
        first_values = [first_accuracy(r)*100 for r in group]
        positions = np.full(len(values), index)+np.linspace(-.08, .08, len(values))
        color = f'C{index}'
        for position, start, end in zip(positions, first_values, values):
            axes[0].plot([position, position], [start, end], color=color, alpha=.6)
        axes[0].scatter(positions, first_values, s=70, facecolors='none', edgecolors=color)
        axes[0].scatter(positions, values, s=35, color=color)
        axes[0].plot([index-.2, index+.2], [np.mean(values)]*2, color='black')
        arrays = []
        for r in group:
            config = r['config']; directory = Path(config['output'])
            path = next(p for p in directory.glob('*/report.json') if json.loads(p.read_text())['config'] == config)
            with np.load(path.parent/'round-1-samples.npz') as f: samples = f['magnetization']
            # Average the two rows within each trajectory before estimating SE.
            pixels = np.full((len(samples), 32), -.5)
            pixels[:, r['rounds'][0]['active']] = samples
            truth = np.array(r['truth_same_as_seed']).ravel()
            aligned = pixels*np.where(truth, -1., 1.)
            arrays.append(aligned.reshape(-1, 2, 16).mean(axis=1))
        samples = np.concatenate(arrays)
        mean = samples.mean(axis=0); se = samples.std(axis=0, ddof=1)/np.sqrt(len(samples))
        signal_diagnostics.append({'b': b, 'kappa': kappa, 'trajectories': len(samples),
            'columns': list(range(16)), 'aligned_row_average': mean.tolist(), 'se': se.tolist(),
            'far_column_mean': float(mean[-1]),
            'far_column_pointwise_approximate_95_interval': [float(mean[-1]-1.96*se[-1]), float(mean[-1]+1.96*se[-1])],
            'scope': 'Pooled first-round trajectories only; subsequent adaptive rounds must not be pooled across differing fields.'})
        x = np.arange(16)
        axes[1].plot(x, mean, marker='.', label=label)
        axes[1].fill_between(x, mean-1.96*se, mean+1.96*se, alpha=.15)
    axes[0].set(xticks=range(len(selected)), xticklabels=[f'b={k[1]:g}\nκ={k[2]:g}' for k, _ in selected],
                ylabel='Final segmentation accuracy (%)', ylim=(0, 105), title='Independent confirmation runs')
    axes[0].grid(axis='y', alpha=.2)
    axes[1].axhline(0, color='black', lw=.7)
    axes[1].set(xlabel='Column (seed at column 0)', ylabel='Aligned magnetization (row average)',
                title='First-round signal, pooled fresh trajectories')
    axes[1].legend(fontsize=9)
    fig.suptitle('2×16 strip • 50 sweeps and 64 trajectories per round\nOpen: first-round accuracy; filled: final; bars: means • shading: approximate pointwise 95% sampling intervals', fontsize=11)
    fig.savefig(root/'independent-confirmation.png', dpi=180)
    fig.savefig(root/'independent-confirmation.svg'); plt.close(fig)
    comparison['first_round_signals'] = signal_diagnostics
    (root/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    if args.summary: export_summary(root, comparison, args.summary)


if __name__ == '__main__': main()
