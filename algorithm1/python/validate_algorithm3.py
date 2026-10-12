"""Paired bond-dimension refinements along the recorded Algorithm 3 branches."""
import os, tempfile
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'): os.environ.setdefault(key, '1')
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'quantum-segmentation-numba'))
import argparse, json, time
from pathlib import Path
import numpy as np
from local_petz import image_graph
from algorithm3 import build_active_petz, mps_local_magnetization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'algorithm3')
    p.add_argument('--cases', nargs='+', default=['T16', 'T16-noisy', 'O16', 'strip8', 'strip16'])
    p.add_argument('--decision', choices=['strict', 'resolved'], default='strict')
    p.add_argument('--paired', type=int, default=8)
    p.add_argument('--reports', nargs='+', type=Path,
                   help='Explicit report.json files, including arbitrary beta/kappa sweep runs.')
    p.add_argument('--rounds', nargs='+', type=int, help='Only check these one-based rounds; default all.')
    args = p.parse_args(); checks = []; started = time.time()
    paths = args.reports or [args.input/f'{name}-mps-{args.decision}-b6-s50-r64-chi32-seed1200'/'report.json'
                             for name in args.cases]
    for path in paths:
        directory = path.parent
        report = json.loads(path.read_text()); name = report['name']; local_checks = []
        if report['backend'] != 'mps' or not 2 <= args.paired <= report['replicas']:
            raise ValueError('Paired refinement needs an MPS report and 2 <= paired <= saved replicas.')
        graph = image_graph(report['image'], kappa=report['kappa'], sigma=report['sigma'], sigma_ddof=report['sigma_ddof'])
        for record in report['rounds']:
            if args.rounds is not None and record['round'] not in args.rounds: continue
            model = build_active_petz(graph, [np.nan if m is None else m for m in record['frozen_before']], report['b'])
            with np.load(directory/f"round-{record['round']}-samples.npz") as f: low = f['magnetization'][:args.paired]
            mean, high = mps_local_magnetization(model, report['sweeps_per_round'], args.paired, 2*report['bond'],
                                                report['cutoff'], report['random_seed']+100000*(record['round']-1))
            difference = high['samples']-low
            item = {'case': name, 'decision': report['decision'], 'b': report['b'], 'kappa': report['kappa'],
                    'random_seed': report['random_seed'], 'refined_bond': 2*report['bond'],
                    'round': record['round'], 'paired_replicas': args.paired,
                    'max_individual_magnetization_change': float(np.max(abs(difference))),
                    'max_mean_magnetization_change': float(np.max(abs(difference.mean(axis=0)))),
                    'max_discarded_weight_sum_at_refined_bond': high['max_discarded_weight_sum'],
                    'max_probability_sum_error': high['max_probability_sum_error'],
                    'scope': 'Same recorded frozen fields and random seeds; empirical subset check, not a global error bound or a new adaptive run.'}
            checks.append(item); local_checks.append(item)
            np.savez_compressed(directory/f"round-{record['round']}-paired-chi{2*report['bond']}.npz", magnetization=high['samples'])
            (directory/'bond-refinement.json').write_text(json.dumps(local_checks, indent=2)+'\n')
            print(item, 'elapsed', round(time.time()-started, 1), flush=True)


if __name__ == '__main__': main()
