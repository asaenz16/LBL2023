"""Exact Gibbs alpha sweep on known-truth images for the diagonal Ising model."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'): os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import time
from importlib.metadata import version
import numpy as np
from scipy.io import savemat
from ising_algorithm1 import ising_model, gibbs_ising

MAGNITUDES = [.001, .003, .01, .03, .1, .3, 1.]
ALPHAS = [-a for a in reversed(MAGNITUDES)] + [0.] + MAGNITUDES


def benchmarks():
    r, c = np.indices((4, 4)); masks = {
        'uniform4': np.ones((4, 4), bool), 'horizontal4': r < 2, 'vertical4': c < 2,
        'diagonal4': r+c < 4, 'checkerboard4': (r+c) % 2 == 0,
        'center3': np.array([[1,1,1],[1,0,1],[1,1,1]], bool),
        'L3': np.array([[1,1,1],[1,0,0],[1,0,1]], bool),
        'T4': ~(((r == 1) & (c < 3)) | ((r == 2) & (c == 1))),
        'O4': ~((r > 0) & (r < 3) & (c > 0) & (c < 3)),
        'islands4': ~(((r == 1) & (c == 1)) | ((r == 3) & (c == 3))),
        'strip4': np.tile((np.arange(4)//2) % 2 == 0, (2, 1)),
        'strip8': np.tile((np.arange(8)//2) % 2 == 0, (2, 1))}
    cases = []
    def add(name, image, truth, group, **extra):
        cases.append(dict(name=name, image=np.asarray(image, float), labels=np.asarray(truth, bool), group=group, **extra))
    for name, mask in masks.items(): add(name, np.where(mask, 0., 255.), mask, 'clean')
    for i, name in enumerate(['vertical4', 'T4', 'O4', 'checkerboard4', 'islands4']):
        for sigma in [20, 50, 80]:
            for replica in range(3):
                rng_seed = 20261007+100*i+replica
                image = np.where(masks[name], 0., 255.)
                noise = np.random.default_rng(rng_seed).normal(size=image.shape)
                add(f'{name}-noise{sigma}-seed{replica}', np.clip(image+sigma*noise, 0, 255),
                    masks[name], f'noise{sigma}', noise_sigma=sigma, noise_seed=rng_seed)
    add('low-contrast4', np.where(masks['vertical4'], 80., 160.), masks['vertical4'], 'stress')
    add('threshold-halves4', np.where(masks['vertical4'], 0., 125.), masks['vertical4'], 'stress')
    add('threshold-blocks8', np.where(masks['strip8'], 0., 125.), masks['strip8'], 'stress')
    add('near-threshold-blocks8', np.where(masks['strip8'], 0., 128.), masks['strip8'], 'stress')
    rgb = np.where(masks['T4'][..., None], np.array([230., 30, 40]), np.array([30., 40, 220]))
    add('rgb-T4', rgb, masks['T4'], 'clean')
    rgb_low = np.where(masks['vertical4'][..., None], np.array([30., 70, 100]), np.array([100., 130, 150]))
    add('rgb-low-contrast4', rgb_low, masks['vertical4'], 'stress')
    return cases


def score(gamma, labels, tolerance=1e-12):
    labels = np.asarray(labels).ravel(); n = len(labels)
    truth = labels[:, None] == labels[None, :]; off = ~np.eye(n, dtype=bool)
    prediction = gamma > 0; aligned = gamma*np.where(truth, 1., -1.)
    margins = aligned[off]; minimum = float(margins.min())
    pairs = n*(n-1)//2
    # Each Z-shot pair observable is +/-1/4. Union bound over unique pairs.
    # Bound guarantees correct estimated signs, not a CI excluding zero.
    sufficient_shots = int(np.floor(np.log(2*pairs/.05)/(8*minimum**2)))+1 if minimum > tolerance else None
    return dict(accuracy_all_seeds=float(np.mean(prediction == truth)),
        nonseed_accuracy_all_seeds=float(np.mean((prediction == truth)[off])),
        seed0_accuracy=float(np.mean(prediction[0] == truth[0])),
        all_seeds_correct=bool(np.all(prediction == truth)),
        all_signs_correct_above_tolerance=bool(minimum > tolerance),
        min_aligned_gamma=minimum, mean_aligned_gamma=float(margins.mean()),
        weak_pair_fraction=float(np.mean(abs(gamma[off]) <= tolerance)),
        sufficient_shots_all_pairs_95=sufficient_shots)


def grouped(rows):
    out = []
    for group in ['clean', 'noise20', 'noise50', 'noise80', 'stress']:
        for b in sorted({x['b'] for x in rows}):
            for alpha in sorted({x['alpha'] for x in rows}):
                subset = [x for x in rows if x['group'] == group and x['b'] == b and x['alpha'] == alpha]
                if not subset: continue
                out.append(dict(group=group, b=b, alpha=alpha, images=len(subset),
                    mean_nonseed_accuracy=float(np.mean([x['nonseed_accuracy_all_seeds'] for x in subset])),
                    perfect_images=sum(x['all_seeds_correct'] for x in subset),
                    images_with_all_signs_resolved=sum(x['all_signs_correct_above_tolerance'] for x in subset),
                    worst_aligned_gamma=min(x['min_aligned_gamma'] for x in subset),
                    mean_weak_pair_fraction=float(np.mean([x['weak_pair_fraction'] for x in subset]))))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alphas', nargs='+', type=float, default=ALPHAS)
    p.add_argument('--bs', nargs='+', type=float, default=[1., 2., 4., 8.])
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'ising-algorithm1')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); records = []; cases = benchmarks(); max_residual = 0.; max_gauge_error = 0.
    protocol = dict(formula='H=-sum_unique_edges arctan(alpha*(d-125))*Zp*Zq/4',
        normalization='b=beta_paper/N_original', distance='Euclidean in raw 0..255 color channels; no sigma',
        alphas=args.alphas, bs=args.bs, method='Exact enumeration of every Z-basis configuration; no sampling',
        threshold=125., numerical_sign_tolerance=1e-12, truth_used_only_for_scoring=True,
        versions={name: version(name) for name in ['numpy', 'scipy', 'qiskit']})
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for case in cases:
        n = case['labels'].size; gamma_grid = np.zeros((len(args.alphas), len(args.bs), n, n))
        for ai, alpha in enumerate(args.alphas):
            model = ising_model(case['image'], alpha)
            edge_truth = case['labels'].ravel()[model.edges[:, 0]] == case['labels'].ravel()[model.edges[:, 1]]
            edge_alignment = model.delta*np.where(edge_truth, 1., -1.)
            for bi, b in enumerate(args.bs):
                result = gibbs_ising(model, beta=b*n); gamma_grid[ai, bi] = result.gamma
                max_residual = max(max_residual, result.numerical_magnetization_residual,
                                   result.numerical_disconnected_correlation_residual)
                records.append(dict(name=case['name'], group=case['group'], n=n, b=b, beta_paper=b*n, alpha=alpha,
                    zero_edges=int(np.sum(model.delta == 0)), wrong_sign_edges=int(np.sum(edge_alignment < 0)),
                    mean_energy=result.mean_energy, **score(result.gamma, case['labels'])))
        parity = (-1.)**np.indices(case['labels'].shape).sum(axis=0).ravel()
        for ai, alpha in enumerate(args.alphas):
            if alpha > 0 and -alpha in args.alphas:
                transformed = gamma_grid[args.alphas.index(-alpha)]*parity[None, :, None]*parity[None, None, :]
                max_gauge_error = max(max_gauge_error, float(np.max(abs(gamma_grid[ai]-transformed))))
        np.savez_compressed(args.output/f"{case['name']}.npz", image=case['image'], labels=case['labels'],
                            alphas=args.alphas, bs=args.bs, gamma=gamma_grid)
        print(case['name'], 'complete', round(time.monotonic()-started, 1), 'seconds', flush=True)
    serial_cases = [{key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in c.items()} for c in cases]
    summary = dict(protocol=protocol, cases=serial_cases, records=records, groups=grouped(records),
        max_spin_flip_symmetry_residual=max_residual, max_bipartite_gauge_identity_error=max_gauge_error,
        runtime_seconds=time.monotonic()-started)
    (args.output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    # Small fixture can also be passed to the original Qiskit purification code.
    for alpha in [-.03, .03]:
        model = ising_model([[0,0],[255,255]], alpha); result = gibbs_ising(model, beta=16.)
        savemat(args.output/f'fixture-alpha{alpha:g}.mat', dict(image=model.image, alpha=alpha,
            beta_paper=16., beta_effective=4., n_pixels=4, edges_zero_based=model.edges,
            distances=model.distances, delta=model.delta, energies=model.energies,
            weights=result.weights, gamma=result.gamma, rho=np.diag(result.weights)))
    print('DONE', len(cases), 'images;', len(records), 'exact Gibbs evaluations;',
          'gauge error', max_gauge_error, 'symmetry residual', max_residual, flush=True)


if __name__ == '__main__': main()
