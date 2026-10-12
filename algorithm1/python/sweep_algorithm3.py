"""Resumable beta/kappa sweep of the existing Algorithm 3 local-Petz method.

Use b=beta_paper/N_original, not beta_paper, to compare image sizes. Screening
and independent confirmations have separate output directories and RNG seeds.
No ground-truth information enters the segmentation or freezing decisions.
"""
import os
import tempfile
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'quantum-segmentation-numba'))
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback
import numpy as np
from algorithm3 import segment, mps_local_magnetization
from local_petz import image_graph
from run_algorithm3 import benchmark, metrics


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def chunked_estimate(model, sweeps, replicas, bond, cutoff, seed, *, progress=None, chunk=8):
    """Exactly the same RNG stream per trajectory as the original estimator."""
    arrays = []
    diagnostics = []
    # Avoid a final singleton because the base estimator requires replicas >= 2.
    offset = 0
    while offset < replicas:
        count = min(chunk, replicas-offset)
        if replicas-offset-count == 1:
            count += 1
        _, details = mps_local_magnetization(model, sweeps, count, bond, cutoff, seed+offset)
        arrays.append(details['samples']); diagnostics.append(details)
        offset += count
        if progress is not None:
            progress(offset)
    samples = np.concatenate(arrays)
    return samples.mean(axis=0), {
        'samples': samples, 'se': samples.std(axis=0, ddof=1)/np.sqrt(replicas),
        'max_discarded_weight_sum': max(d['max_discarded_weight_sum'] for d in diagnostics),
        'max_probability_sum_error': max(d['max_probability_sum_error'] for d in diagnostics),
        'svd_fallbacks': sum(d['svd_fallbacks'] for d in diagnostics)}


def run_one(config):
    root = Path(config['output'])
    stem = (f"{config['case']}-{config['backend']}-{config['decision']}"
            f"-b{config['b']:g}-kappa{config['kappa']:g}"
            f"-s{config['sweeps']}-r{config['replicas']}-chi{config['bond']}-seed{config['seed']}")
    directory = root/stem; directory.mkdir(parents=True, exist_ok=True)
    report_path = directory/'report.json'
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior['config'] != config:
            raise ValueError(f'Existing run has different configuration: {directory}')
        return prior
    started = time.monotonic(); rounds = []
    image, truth, seeds = benchmark(config['case'])
    graph = image_graph(image, kappa=config['kappa'])
    def estimator(model, sweeps, replicas, bond, cutoff, seed):
        iteration = (seed-config['seed'])//100000+1
        def progress(completed):
            write_json(directory/'status.json', {'config': config, 'round': iteration,
                'active': model.n, 'completed_replicas': completed,
                'runtime_seconds': time.monotonic()-started})
        return chunked_estimate(model, sweeps, replicas, bond, cutoff, seed, progress=progress)
    def callback(record, details):
        rounds.append(record); write_json(directory/'progress.json', rounds)
        if 'samples' in details:
            np.savez_compressed(directory/f"round-{record['round']}-samples.npz",
                                magnetization=details['samples'])
    try:
        result = segment(graph, seeds, b=config['b'], backend=config['backend'],
            sweeps=config['sweeps'], k=3, replicas=config['replicas'], bond=config['bond'],
            random_seed=config['seed'], decision=config['decision'],
            mps_estimator=estimator, callback=callback)
        first = result['rounds'][0]
        active = np.array(first['active'])
        # Positive aligned margin means the sign matches the known benchmark.
        aligned = np.array(first['magnetization'])*np.where(truth.ravel()[active], -1., 1.)
        last_column = active % image.shape[1] == image.shape[1]-1
        result.update({'status': 'complete', 'config': config, 'name': config['case'],
            'image': image.tolist(), 'truth_same_as_seed': truth.tolist(),
            'kappa': config['kappa'], 'sigma': graph.sigma, 'sigma_ddof': graph.sigma_ddof,
            'metrics': metrics(result['same_as_seed'], truth, seeds),
            'first_round_min_aligned_m': float(aligned.min()),
            'first_round_far_aligned_m': float(aligned[last_column].mean()),
            'uncertain_freeze_count': sum(len(r['uncertain_freezes']) for r in result['rounds']),
            'max_discarded_weight_sum': max(r['max_discarded_weight_sum'] for r in result['rounds']),
            'max_probability_sum_error': max(r['max_probability_sum_error'] for r in result['rounds'])})
    except Exception as exc:
        result = {'status': 'failed', 'config': config, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'completed_rounds': rounds}
    result['runtime_seconds'] = time.monotonic()-started
    write_json(report_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['strip16'])
    parser.add_argument('--bs', nargs='+', type=float, default=[2, 4, 6, 8, 12])
    parser.add_argument('--kappas', nargs='+', type=float, default=[1.5, 2, 3, 4])
    parser.add_argument('--pairs', nargs='+', help='Explicit b:kappa pairs instead of the Cartesian grid.')
    parser.add_argument('--backend', choices=['dense', 'mps', 'capped'], default='mps')
    parser.add_argument('--decision', choices=['strict', 'resolved'], default='strict')
    parser.add_argument('--sweeps', type=int, default=50)
    parser.add_argument('--replicas', type=int, default=16)
    parser.add_argument('--bond', type=int, default=32)
    parser.add_argument('--seeds', nargs='+', type=int, default=[10000000])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.replicas < 2 or args.workers < 1:
        parser.error('Require at least two replicas and one worker.')
    if args.pairs is None:
        pairs = [(b, kappa) for b in args.bs for kappa in args.kappas]
    else:
        try:
            pairs = [tuple(map(float, item.split(':'))) for item in args.pairs]
            if any(len(pair) != 2 for pair in pairs): raise ValueError
        except ValueError:
            parser.error('Each --pairs entry must have the form b:kappa.')
    jobs = [{'case': name, 'b': b, 'kappa': kappa, 'backend': args.backend,
             'decision': args.decision, 'sweeps': args.sweeps, 'replicas': args.replicas,
             'bond': args.bond, 'seed': seed, 'output': str(args.output.resolve())}
            for name in args.cases for b, kappa in pairs for seed in args.seeds]
    write_json(args.output/'protocol.json', {'jobs': jobs, 'normalization': 'b=beta_paper/N_original',
        'initial_state': 'plus_x', 'patch': 'nearest-neighbor star', 'seed_pixels': 'benchmark defaults',
        'versions': {name: version(name) for name in ['numpy', 'scipy', 'quimb', 'qiskit']},
        'uncertainty': 'Independent Kraus trajectories, no additional Z-shot noise; MPS error is separate.'})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            write_json(args.output/'summary.json', results)
            config = result['config']
            print(len(results), '/', len(jobs), config['case'], 'b', config['b'],
                  'kappa', config['kappa'], 'seed', config['seed'], result['status'],
                  result.get('metrics', result.get('error')), flush=True)
    if any(r['status'] != 'complete' for r in results):
        raise SystemExit('Some settings failed; see saved reports.')


if __name__ == '__main__':
    main()
