"""Accuracy tables and pixel-level comparisons for Algorithm 3 experiments."""
import os, tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'algorithm3-mpl'))
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_algorithm3 import benchmark, metrics


def read(root, name, backend, decision='strict', replicas=64, bond=32, seed=1200):
    suffix = 'k3' if backend == 'capped' else 's50'
    if backend == 'mps': suffix += f'-r{replicas}-chi{bond}-seed{seed}'
    path = root/f'{name}-{backend}-{decision}-b6-{suffix}'/'report.json'
    return json.loads(path.read_text()) if path.exists() else None


def render(ax, pixels, title, *, seeds=(), wrong=None):
    ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    if seeds:
        coords = np.array([divmod(p, pixels.shape[1]) for p in seeds])
        ax.scatter(coords[:, 1], coords[:, 0], s=60, facecolors='none', edgecolors='#008ba3', linewidths=2)
    if wrong is not None:
        r, c = np.where(wrong); ax.scatter(c, r, marker='x', s=55, color='#e14531', linewidths=2)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def panel(root, names, filename):
    fig, axes = plt.subplots(len(names), 4, figsize=(12, 1.8*len(names)+.7), squeeze=False, constrained_layout=True)
    for axs, name in zip(axes, names):
        image, truth, seeds = benchmark(name)
        backend = 'dense' if name in ('tiny4', 'strip4', 'center9', 'L9') else 'mps'
        local = read(root, name, backend); cap = read(root, name, 'capped')
        render(axs[0], image, f'{name}: input + supplied seeds', seeds=seeds)
        render(axs[1], np.where(truth, 0, 255), 'Known truth')
        if cap is None: axs[2].axis('off'); axs[2].text(.5, .5, 'K=3 reference\nnot computed at this size', ha='center', va='center', transform=axs[2].transAxes)
        else:
            pred = np.array(cap['same_as_seed'])
            render(axs[2], np.where(pred, 0, 255), f"Capped K=3: {100*cap['metrics']['accuracy']:.1f}%", wrong=pred != truth)
        if local is None: raise ValueError(f'Incomplete local run: {name}.')
        pred = np.array(local['same_as_seed'])
        label = 'exact channel' if backend == 'dense' else '64 trajectories/round'
        render(axs[3], np.where(pred, 0, 255), f"Local Petz: {100*local['metrics']['accuracy']:.1f}%\n{label}", wrong=pred != truth)
    fig.suptitle('Algorithm 3 adaptation · b=6 · 50 local sweeps per freezing round\nCyan circles: supplied seeds; red crosses: wrong labels. New synthetic benchmarks.', fontsize=12)
    fig.savefig(root/f'{filename}.png', dpi=180); fig.savefig(root/f'{filename}.svg'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'algorithm3')
    args = parser.parse_args(); root = args.input
    names = ['tiny4', 'center9', 'L9', 'T16', 'T16-noisy', 'O16', 'strip4', 'strip8', 'strip16']
    summary = []
    for name in names:
        image, truth, seeds = benchmark(name)
        backend = 'dense' if name in ('tiny4', 'strip4', 'center9', 'L9') else 'mps'
        local = read(root, name, backend); cap = read(root, name, 'capped'); resolved = read(root, name, 'mps', 'resolved')
        if local is None: raise ValueError(f'Incomplete local run: {name}.')
        first = np.zeros(truth.size, bool); first[seeds] = True
        record = local['rounds'][0]; first[np.array(record['active'])] = np.array(record['magnetization']) < -local['sign_tolerance']
        ambiguous = []
        for record in local['rounds']:
            for pixel, m in zip(record['active'], record['magnetization']):
                if pixel in record['newly_frozen'] and m+record['simultaneous_sampling_radius'] >= -local['sign_tolerance']:
                    ambiguous.append(pixel)
        summary.append({'name': name, 'n_pixels': image.size, 'n_seeds': len(seeds), 'backend': backend,
                        'first_round_metrics': metrics(first.reshape(truth.shape), truth, seeds),
                        'local_metrics': local['metrics'], 'rounds': len(local['rounds']),
                        'uncertain_freezes': ambiguous, 'resolved_metrics': resolved['metrics'] if resolved else None,
                        'resolved_unresolved_pixels': resolved['unresolved_active'] if resolved else None,
                        'k3_metrics': cap['metrics'] if cap else None})
    (root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    panel(root, ['tiny4', 'center9', 'L9', 'T16', 'T16-noisy', 'O16'], 'segmentation-shapes')
    panel(root, ['strip4', 'strip8', 'strip16'], 'segmentation-strips')

    cases = [name for name in ('strip8', 'strip16') if read(root, name, 'mps', 'resolved') is not None]
    if cases:
        fig, axes = plt.subplots(len(cases), 3, figsize=(13, 3*len(cases)), squeeze=False, constrained_layout=True)
        for axs, name in zip(axes, cases):
            image, truth, seeds = benchmark(name)
            for ax, decision in zip(axs[:2], ('strict', 'resolved')):
                result = read(root, name, 'mps', decision)
                pred = np.array(result['same_as_seed'])
                render(ax, np.where(pred, 0, 255), f"{name} · {decision}: {100*result['metrics']['accuracy']:.1f}%", wrong=pred != truth)
                if decision == 'resolved' and result['unresolved_active']:
                    rc = np.array([divmod(p, image.shape[1]) for p in result['unresolved_active']])
                    ax.scatter(rc[:, 1], rc[:, 0], s=25, color='#ce9800', marker='s')
            for decision, color in [('strict', '#177ea1'), ('resolved', '#c96535')]:
                result = read(root, name, 'mps', decision)
                labels = np.zeros(image.size, bool); labels[seeds] = True
                accuracy = [np.mean(labels.reshape(truth.shape) == truth)]
                for record in result['rounds']:
                    labels[record['newly_frozen']] = True; accuracy.append(np.mean(labels.reshape(truth.shape) == truth))
                axs[2].plot(range(len(accuracy)), 100*np.array(accuracy), 'o-', color=color, label=decision)
            axs[2].set_xlabel('Freezing round (0 = seeds only)'); axs[2].set_ylabel('Pixel accuracy (%)')
            axs[2].set_ylim(0, 105); axs[2].grid(alpha=.2); axs[2].legend()
        fig.suptitle('Strict sign freezing versus simultaneous-confidence freezing\nRed crosses: wrong labels; gold squares: unresolved remaining signs (assigned to complement by the algorithm).', fontsize=12)
        fig.savefig(root/'freezing-controls.png', dpi=180); fig.savefig(root/'freezing-controls.svg'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
