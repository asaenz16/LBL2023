"""Compare extended local-Petz trajectories with their archived shorter runs.

All confidence intervals are pointwise normal approximations across independent
Kraus trajectories. Changes between sweep checkpoints use paired trajectories.
Optionally repeat the four most truncated trajectories at doubled MPS bond.
"""
import os
import tempfile
for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'quantum-segmentation-numba'))

import argparse
import json
import time
from pathlib import Path
import numpy as np


def interval(samples):
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    return {'mean': mean, 'se': se, 'ci95': [mean - 1.96 * se, mean + 1.96 * se]}


def load_run(path):
    report = json.loads(path.read_text())
    with np.load(path.with_suffix('.npz')) as f:
        raw = {key: f[key].copy() for key in f.files}
    assert len(raw['gamma']) == report['replicas'], f'Incomplete run: {path}'
    np.testing.assert_array_equal(raw['sweeps'], report['sweeps'])
    np.testing.assert_allclose(raw['gamma'].mean(axis=0), report['gamma'], atol=1e-14, rtol=0)
    np.testing.assert_allclose(raw['gamma'].std(axis=0, ddof=1) / np.sqrt(report['replicas']),
                               report['gamma_se'], atol=1e-14, rtol=0)
    assert np.isfinite(raw['gamma']).all() and np.max(abs(raw['gamma'])) <= .25 + 1e-10
    assert report['max_probability_sum_error'] < 1e-9
    return report, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'local-petz')
    parser.add_argument('--earlier-sweep', type=int, default=25)
    parser.add_argument('--refine-worst', type=int, default=0)
    args = parser.parse_args()
    out = args.input
    archive = out / f'sweeps{args.earlier_sweep}' / 'trajectories'
    runs = {}
    for path in sorted((out / 'trajectories').glob('*.json')):
        report, raw = load_run(path)
        overlap_error = None
        if (archive / path.name).exists():
            old, oldraw = load_run(archive / path.name)
            for key in ('width', 'b', 'sigma', 'mode', 'initial', 'seed', 'replicas', 'max_bond', 'cutoff', 'seeds'):
                assert report[key] == old[key], f'Protocol changed: {path.name}: {key}'
            indices = [report['sweeps'].index(t) for t in old['sweeps']]
            overlap_error = float(np.max(abs(raw['gamma'][:, indices] - oldraw['gamma'])))
            np.testing.assert_allclose(raw['gamma'][:, indices], oldraw['gamma'], atol=1e-10, rtol=0)
        runs[path.stem] = (report, raw, overlap_error)

    dense = json.loads((out / 'dense' / 'report.json').read_text())
    summary = {'earlier_sweep': args.earlier_sweep, 'runs': {}, 'bond_refinement': {},
               'uncertainty': 'Approximate pointwise 95% intervals across independent trajectories. Changes use paired samples. Profile maxima are point estimates; no simultaneous coverage or stationarity claim.'}
    for name, (report, raw, overlap_error) in runs.items():
        times = report['sweeps']
        assert times[-1] > args.earlier_sweep, f'Run has not been extended: {name}'
        earlier = times.index(args.earlier_sweep)
        far = -raw['gamma'][:, :, 0, 1, -1]
        result = {'width': report['width'], 'replicas': report['replicas'], 'initial': report['initial'],
                  'final_sweep': times[-1], 'archive_max_gamma_difference': overlap_error,
                  'earlier_far': interval(far[:, earlier]), 'final_far': interval(far[:, -1]),
                  'paired_far_change': interval(far[:, -1] - far[:, earlier]),
                  'profile_linf_change': float(np.max(abs(np.array(report['gamma'])[-1] - np.array(report['gamma'])[earlier]))),
                  'max_discarded_weight_sum': report['max_discarded_weight_sum'],
                  'max_probability_sum_error': report['max_probability_sum_error']}
        late = np.array(times) > args.earlier_sweep
        result['late_window_sweeps'] = np.array(times)[late].tolist()
        # Average within each trajectory first: checkpoints are not independent samples.
        result['late_window_far'] = interval(far[:, late].mean(axis=1))
        if report['width'] == 4:
            assert dense['b'] == report['b'] and np.isclose(dense['sigma'], report['sigma'])
            reference = np.array(dense['gibbs']['gamma'])
            records = dense['models']['star']['cases'][f"{report['initial']}-p0"]['records']
            by_time = {r['sweep']: r for r in records}
            exact = np.array([by_time[t]['gamma'] for t in times]).reshape(np.array(report['gamma']).shape)
            residual = abs(np.array(report['gamma']) - exact)
            se = np.array(report['gamma_se'])
            assert np.max(residual[se < 1e-12], initial=0) < 1e-10
            result['max_exact_discrepancy_in_se'] = float(np.max(residual[se >= 1e-12] / se[se >= 1e-12]))
            mask = se[-1] >= 1e-12
            result['final_max_exact_discrepancy_in_se'] = float(np.max(residual[-1][mask] / se[-1][mask]))
            result['exact_final_far'] = by_time[times[-1]]['far_aligned']
        else:
            path = out.parent / 'xxz-glauber' / 'long-run' / f"L{report['width']}-b{report['b']:g}.json"
            ref = json.loads(path.read_text())
            assert ref['b'] == report['b'] and ref['seeds'] == report['seeds'] and np.isclose(ref['sigma'], report['sigma'])
            reference = np.array(ref['reference']['gamma']).reshape(2, 2, report['width'])
        distances = np.max(abs(np.array(report['gamma']) - reference), axis=(1, 2, 3))
        result['earlier_profile_linf_to_gibbs'] = float(distances[earlier])
        result['final_profile_linf_to_gibbs'] = float(distances[-1])
        summary['runs'][name] = result

    for width in (8, 16):
        highreport, high, _ = runs[f'L{width}-plus_x-chi64-r16']
        lowreport, low, _ = runs[f'L{width}-plus_x-chi32-r128']
        for key in ('width', 'b', 'sigma', 'mode', 'initial', 'seed', 'sweeps', 'cutoff', 'seeds'):
            assert highreport[key] == lowreport[key], f'Bond-refinement protocol changed: {key}'
        difference = high['gamma'] - low['gamma'][:highreport['replicas']]
        summary['bond_refinement'][f'L{width}'] = {
            'paired_replicas': highreport['replicas'],
            'max_individual_gamma_difference': float(np.max(abs(difference))),
            'max_mean_gamma_difference': float(np.max(abs(difference.mean(axis=0))))}

    if args.refine_worst:
        from local_petz import image_graph, build_local_petz
        from ladder_tensor import build_ladder
        from local_petz_trajectories import trajectory_plan, initial_trajectory, apply_trajectory_sweep, trajectory_observe
        checks = []
        start = time.time()
        for width in (8, 16):
            report, raw, _ = runs[f'L{width}-plus_x-chi32-r128']
            model = build_local_petz(image_graph(build_ladder(width).image), report['b'], report['mode'])
            plan = trajectory_plan(model)
            times = report['sweeps']
            for replica in np.argsort(raw['diagnostics'][:, 0])[-args.refine_worst:]:
                trajectory = initial_trajectory(model, report['initial'], report['seed'] + int(replica))
                gammas = []
                for sweep in range(times[-1] + 1):
                    if sweep in times:
                        gammas.append(trajectory_observe(trajectory)[0])
                    if sweep < times[-1]:
                        apply_trajectory_sweep(trajectory, plan, 2 * report['max_bond'], report['cutoff'])
                difference = np.array(gammas) - raw['gamma'][replica]
                checks.append({'width': width, 'replica': int(replica), 'sweeps': times,
                               'chi32_discarded_weight': float(raw['diagnostics'][replica, 0]),
                               'chi64_discarded_weight': trajectory.discarded_weight,
                               'maximum_gamma_change': float(np.max(abs(difference))),
                               'final_far_change': float(-difference[-1, 0, 1, -1])})
                (out / 'worst-case-bond-refinement.json').write_text(json.dumps(checks, indent=2) + '\n')
                print('REFINED', width, int(replica), 'max Γ difference', checks[-1]['maximum_gamma_change'],
                      'elapsed', round(time.time() - start, 1), flush=True)
        summary['worst_case_bond_refinement'] = checks
    path = out / f'sweeps{max(r[0]["sweeps"][-1] for r in runs.values())}-comparison.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print('Saved', path, flush=True)
    for name, result in summary['runs'].items():
        print(name, 'final', result['final_far'], 'paired change', result['paired_far_change'], flush=True)


if __name__ == '__main__':
    main()
