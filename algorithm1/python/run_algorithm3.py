"""Known-truth segmentation benchmarks for the local-Petz Algorithm 3 adaptation."""
import os, tempfile
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'): os.environ.setdefault(key, '1')
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'quantum-segmentation-numba'))
import argparse, gzip, json, time
from pathlib import Path
import numpy as np
from qiskit import qpy, qasm2
from local_petz import image_graph
from algorithm3 import (segment, build_active_petz, active_sweep_circuit, preparation_circuit,
                        mps_local_magnetization, dense_local_magnetization, verified_field_recovery)


def benchmark(name):
    if name.startswith('strip'):
        width = int(name[5:]); truth = np.tile((np.arange(width)//2) % 2 == 0, (2, 1)); seeds = [0]
    elif name == 'tiny4':
        truth = np.array([[True, True], [False, False]]); seeds = [0]
    elif name == 'center9':
        truth = np.ones((3, 3), bool); truth[1, 1] = False; seeds = [0, 2, 6, 8]
    elif name == 'L9':
        truth = np.ones((3, 3), bool); truth[1, 1:3] = False; truth[2, 1] = False; seeds = [0, 2, 6, 8]
    elif name in ('T16', 'T16-noisy', 'O16'):
        truth = np.ones((4, 4), bool)
        if name == 'O16': truth[1:3, 1:3] = False
        else: truth[1, :3] = False; truth[2, 1] = False
        seeds = [0, 3, 12, 15]
    else: raise ValueError('Unknown benchmark.')
    image = np.where(truth, 0., 255.)
    if name.endswith('-noisy'):
        image = np.clip(image + np.random.default_rng(2026).normal(0, 20, image.shape), 0, 255)
    assert truth.ravel()[seeds].all()
    return image, truth, seeds


def metrics(prediction, truth, seeds):
    prediction = np.array(prediction, bool); truth = np.array(truth, bool)
    mask = np.ones(truth.size, bool); mask[seeds] = False
    return {'accuracy': float(np.mean(prediction == truth)),
            'nonseed_accuracy': float(np.mean((prediction == truth).ravel()[mask])),
            'all_pixels_correct': bool(np.all(prediction == truth)),
            'false_seed_segment': int(np.sum(prediction & ~truth)),
            'missed_seed_segment': int(np.sum(~prediction & truth)),
            'iou_seed_segment': float(np.sum(prediction & truth)/np.sum(prediction | truth)),
            'iou_opposite_segment': float(np.sum(~prediction & ~truth)/np.sum(~prediction | ~truth))}


def write_json(path, data): path.write_text(json.dumps(data, indent=2)+'\n')


def export_rounds(graph, result, directory):
    directory.mkdir(exist_ok=True); manifests = []
    for record in result['rounds']:
        frozen = [np.nan if v is None else v for v in record['frozen_before']]
        model = build_active_petz(graph, frozen, result['b'])
        body = active_sweep_circuit(model)
        circuit = preparation_circuit(model, result['sweeps_per_round'])
        prefix = directory / f"round-{record['round']}"
        with prefix.with_suffix('.qpy').open('wb') as f: qpy.dump(circuit, f)
        with gzip.open(directory/f"round-{record['round']}-one-sweep.qasm.gz", 'wt') as f: f.write(qasm2.dumps(body))
        with prefix.with_suffix('.qpy').open('rb') as f: restored = qpy.load(f)[0]
        assert restored.count_ops() == circuit.count_ops()
        assert len(restored.data) == len(circuit.data)
        for a, c in zip(circuit.data, restored.data):
            if a.operation.name == 'local_petz_sweep':
                assert a.operation.definition == c.operation.definition
        with gzip.open(directory/f"round-{record['round']}-one-sweep.qasm.gz", 'rt') as f: parsed = qasm2.loads(f.read())
        assert body.count_ops() == parsed.count_ops()
        for a, c in zip(body.data, parsed.data):
            assert a.operation.name == c.operation.name
            assert [body.find_bit(q).index for q in a.qubits] == [parsed.find_bit(q).index for q in c.qubits]
            assert np.allclose(a.operation.params, c.operation.params, rtol=0, atol=1e-10)
        operations = dict(body.count_ops())
        synthesis = [verified_field_recovery(len(u.sites), u.edges, u.delta, tuple(model.fields[list(u.sites)]),
                                             model.b, u.local.site)[1] for u in model.updates]
        manifests.append({'round': record['round'], 'active_pixels': list(model.active),
                          'sweeps': result['sweeps_per_round'], 'n_system': model.n, 'n_ancillas': model.n,
                          'sweep_operations': operations, 'sweep_depth': body.depth(),
                          'cx_per_preparation': operations.get('cx', 0)*result['sweeps_per_round'],
                          'recovery_synthesis': synthesis,
                          'qpy_roundtrip': True, 'qasm_all_gates_roundtrip': True})
    write_json(directory/'manifest.json', manifests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['tiny4', 'strip4', 'center9', 'L9'])
    parser.add_argument('--backend', choices=['dense', 'mps', 'capped', 'counts'], default='dense',
                        help='counts simulates ideal Z shots from the exact local-channel density, limited to nine active spins.')
    parser.add_argument('--decision', choices=['strict', 'resolved'], default='strict')
    parser.add_argument('--sweeps', type=int, default=50)
    parser.add_argument('--b', type=float, default=6.)
    parser.add_argument('--kappa', type=float, default=2.)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--replicas', type=int, default=128)
    parser.add_argument('--bond', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1200)
    parser.add_argument('--shots', type=int, default=8192)
    parser.add_argument('--export-circuits', action='store_true')
    parser.add_argument('--reuse-strict', action='store_true', help='Reuse exactly matching saved strict-rule ensembles; simulate new branches normally.')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'algorithm3')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        image, truth, seeds = benchmark(name); graph = image_graph(image, kappa=args.kappa)
        suffix = f'k{args.k}' if args.backend == 'capped' else f's{args.sweeps}'
        if args.backend == 'mps': suffix += f'-r{args.replicas}-chi{args.bond}-seed{args.seed}'
        if args.backend == 'counts': suffix += f'-shots{args.shots}-seed{args.seed}'
        if args.kappa != 2.: suffix += f'-kappa{args.kappa:g}'
        stem = f'{name}-{args.backend}-{args.decision}-b{args.b:g}-{suffix}'
        target = args.output/stem; target.mkdir(exist_ok=True); started = time.time(); progress = []
        source = None
        if args.reuse_strict:
            if args.backend != 'mps' or args.decision != 'resolved': raise ValueError('Reuse is for resolved MPS controls.')
            source = args.output/f'{name}-mps-strict-b{args.b:g}-{suffix}'
            prior = json.loads((source/'report.json').read_text())
            for key, expected in {'image': image.tolist(), 'b': args.b, 'kappa': args.kappa,
                                  'sigma': graph.sigma, 'sigma_ddof': graph.sigma_ddof,
                                  'seeds': seeds, 'sweeps_per_round': args.sweeps, 'replicas': args.replicas,
                                  'bond': args.bond, 'cutoff': 1e-12, 'random_seed': args.seed}.items():
                if prior[key] != expected: raise ValueError(f'Sample reuse protocol mismatch: {key}.')
        def estimator(model, sweeps, replicas, bond, cutoff, seed):
            if source is not None:
                iteration = (seed-args.seed)//100000
                if iteration < len(prior['rounds']):
                    record = prior['rounds'][iteration]
                    frozen = [np.nan if v is None else v for v in record['frozen_before']]
                    oldmodel = build_active_petz(graph, frozen, args.b)
                    if model.active == oldmodel.active and np.array_equal(model.fields, oldmodel.fields):
                        with np.load(source/f"round-{record['round']}-samples.npz") as f: samples = f['magnetization']
                        assert samples.shape == (replicas, model.n) and np.isfinite(samples).all()
                        return samples.mean(axis=0), {'samples': samples, 'se': samples.std(axis=0, ddof=1)/np.sqrt(replicas),
                            'max_discarded_weight_sum': record['max_discarded_weight_sum'],
                            'max_probability_sum_error': record['max_probability_sum_error'], 'reused_samples': True}
            return mps_local_magnetization(model, sweeps, replicas, bond, cutoff, seed)
        def callback(record, details):
            progress.append(record); write_json(target/'progress.json', progress)
            if 'samples' in details:
                np.savez_compressed(target/f"round-{record['round']}-samples.npz", magnetization=details['samples'])
            if 'counts' in details: write_json(target/f"round-{record['round']}-counts.json", details['counts'])
            print(name, args.backend, args.decision, 'round', record['round'], 'active', len(record['active']),
                  'frozen', len(record['newly_frozen']), 'elapsed', round(time.time()-started, 1), flush=True)
        def count_provider(model, sweeps, iteration):
            _, exact = dense_local_magnetization(model, sweeps)
            probabilities = np.maximum(0., exact['density'].diagonal().real); probabilities /= probabilities.sum()
            draws = np.random.default_rng(args.seed+100000*iteration).multinomial(args.shots, probabilities)
            return {format(i, f'0{model.n}b'): int(count) for i, count in enumerate(draws) if count}
        result = segment(graph, seeds, b=args.b, backend=args.backend, sweeps=args.sweeps,
                         k=args.k, replicas=args.replicas, bond=args.bond, random_seed=args.seed,
                         decision=args.decision, callback=callback, mps_estimator=estimator, count_provider=count_provider)
        result.update({'name': name, 'image': image.tolist(), 'truth_same_as_seed': truth.tolist(),
                       'kappa': args.kappa, 'sigma': graph.sigma, 'sigma_ddof': graph.sigma_ddof,
                       'runtime_seconds': time.time()-started,
                       'metrics': metrics(result['same_as_seed'], truth, seeds),
                       'method': 'Sign-consistent Algorithm 3 with capped Gibbs reference' if args.backend == 'capped' else 'Algorithm 3 adaptation with full-active-space local Petz preparation; no global K cap',
                       'seeds_description': 'One top-left seed on strips; same-class corners on square shape examples.',
                       'uncertainty': 'MPS samples are expectation values per independent Kraus trajectory, without additional Z-shot noise. Hoeffding bounds exclude truncation and preparation bias. Adaptive freezing can amplify sampling errors.'})
        if args.backend == 'counts':
            result['uncertainty'] = 'Ideal independent Z shots sampled from exact local-channel density matrices; no hardware or gate noise.'
        write_json(target/'report.json', result)
        print('DONE', stem, result['metrics'], flush=True)
        if args.export_circuits:
            if args.backend == 'capped': raise ValueError('Only local Petz rounds have scalable circuit exports.')
            export_rounds(graph, result, target/'circuits')


if __name__ == '__main__': main()
