"""Reproducible native-resolution EMNIST parameter sweep (CPU Monte Carlo).

Pixel masks are evaluation proxies, not EMNIST annotations or sampler inputs.
Run training first, freeze a parameter rule, then evaluate the official test split.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quantum-segmentation-matplotlib')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from fetch_emnist import load_idx
from ising_monte_carlo import signed_image, sample_gibbs

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [.003, .01, .03, .1, .3]
BS = [.5, 1., 2., 3., 4., 6., 8.]


def choose_seeds(image):
    boundary = np.zeros(image.shape, bool)
    boundary[[0,-1],:] = True; boundary[:,[0,-1]] = True
    indices = np.flatnonzero(boundary)
    return np.array([indices[np.argmin(image.ravel()[indices])], np.argmax(image)], int)


def select_data(data, output, per_class=1):
    output.mkdir(parents=True, exist_ok=True)
    mapping = {int(a): chr(int(b)) for a,b in np.loadtxt(data/'emnist-balanced-mapping.txt', dtype=int)}
    cases = []; images = []
    for split, seed in [('train',20261007), ('test',20261008)]:
        x = load_idx(data/f'emnist-balanced-{split}-images-idx3-ubyte.gz')
        y = load_idx(data/f'emnist-balanced-{split}-labels-idx1-ubyte.gz')
        rng = np.random.default_rng(seed)
        for label in range(47):
            for rep, index in enumerate(rng.choice(np.flatnonzero(y == label), per_class, replace=False)):
                image = x[index]; name = f'{split}-class{label:02d}-rep{rep}'
                images.append(image)
                cases.append(dict(name=name, split=split, class_id=label, character=mapping[label],
                    idx_index=int(index), image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    seeds=choose_seeds(image).tolist(), foreground_fraction=float(np.mean(image>=128))))
    np.savez_compressed(output/'subset.npz', images=np.array(images))
    manifest = dict(source=json.loads((data/'provenance.json').read_text()), per_class=per_class,
                    selection_seeds={'train':20261007,'test':20261008}, cases=cases)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return cases, np.array(images)


def proxy_metrics(gamma, se, image, seed, threshold=128):
    truth = image.ravel() >= threshold
    same = gamma > 0  # Algorithm 1 includes zero in the opposite class.
    prediction = same if truth[seed] else ~same
    # The orientation uses known dark/bright EMNIST background, not a fitted label.
    tp = np.sum(prediction & truth); tn = np.sum(~prediction & ~truth)
    union = np.sum(prediction | truth)
    aligned = gamma*np.where(truth == truth[seed], 1., -1.)
    resolved = abs(gamma) > np.maximum(3*se, 1e-12)
    return dict(iou=float(tp/max(1,union)), balanced_accuracy=float(.5*(tp/max(1,truth.sum())+
        tn/max(1,(~truth).sum()))), accuracy=float(np.mean(prediction == truth)),
        foreground_recall=float(tp/max(1,truth.sum())), foreground_precision=float(tp/max(1,prediction.sum())),
        resolved_fraction=float(resolved.mean()), confident_wrong_fraction=float(np.mean((aligned<0)&resolved)),
        min_aligned_gamma=float(aligned.min()), mean_aligned_gamma=float(aligned.mean()),
        median_abs_gamma=float(np.median(abs(gamma))))


def image_features(image, model):
    d = model.distances; margin = abs(125-d); active = d>0
    sign = np.sign(model.delta); rows,cols=image.shape
    horizontal = np.zeros((rows,cols-1)); vertical = np.zeros((rows-1,cols))
    for (p,q), s in zip(model.edges,sign):
        r,c = divmod(int(p), cols)
        if q == p+1: horizontal[r,c] = s
        else: vertical[r,c] = s
    loops = horizontal[:-1]*horizontal[1:]*vertical[:,:-1]*vertical[:,1:]
    truth = image.ravel()>=128
    target_same = truth[model.edges[:,0]]==truth[model.edges[:,1]]
    crossing = ~target_same
    return dict(active_margin_median=float(np.median(margin[active])) if active.any() else 125.,
        median_abs_delta=float(np.median(abs(model.delta))),
        negative_edge_fraction=float(np.mean(sign<0)), frustrated_plaquette_fraction=float(np.mean(loops<0)),
        zero_edge_fraction=float(np.mean(sign==0)),
        # Evaluation diagnostics below use the proxy; never used to choose updates.
        boundary_antiferromagnetic_fraction=float(np.mean(sign[crossing]<0)) if crossing.any() else 0.,
        within_class_ferromagnetic_fraction=float(np.mean(sign[~crossing]>0)),
        edge_sign_proxy_agreement=float(np.mean((sign>0)==target_same)))


def job(spec):
    case,image,alpha,bs,options,path = spec
    signature = dict(alpha=alpha, bs=bs, **options, image_sha256=case['image_sha256'], seeds=case['seeds'])
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('signature') != signature: raise ValueError(f'Conflicting cached protocol: {path}')
        return meta
    start=time.perf_counter(); model=signed_image(image,alpha)
    result=sample_gibbs(model,bs,seeds=case['seeds'],**options)
    path.parent.mkdir(parents=True,exist_ok=True)
    packed = {k:(v.astype(np.float32) if k in ('blocks','series','energy_blocks') else v)
              for k,v in result.items()}
    np.savez_compressed(path, **packed)
    records=[]
    for i,b in enumerate(bs):
        metrics=proxy_metrics(result['gamma'][i,0],result['se'][i,0],image,case['seeds'][0])
        ink=proxy_metrics(result['gamma'][i,1],result['se'][i,1],image,case['seeds'][1])
        # Correlation R-hat uses batches; observable R-hat uses individual sweeps.
        rh=result['rhat_gamma'][i,0]
        records.append(dict(name=case['name'],split=case['split'],class_id=case['class_id'],
            alpha=alpha,b=b,beta_paper=b*image.size,**metrics,ink_seed_iou=ink['iou'],
            ink_seed_resolved_fraction=ink['resolved_fraction'],
            iou_threshold96=proxy_metrics(result['gamma'][i,0],result['se'][i,0],image,case['seeds'][0],96)['iou'],
            iou_threshold160=proxy_metrics(result['gamma'][i,0],result['se'][i,0],image,case['seeds'][0],160)['iou'],
            rhat_observable_max=float(result['rhat_observables'][i].max()),
            rhat_gamma_p99=float(np.quantile(rh,.99)), rhat_gamma_max=float(rh.max()),
            ess_energy=float(result['ess_observables'][i,0]),
            ess_mean_gamma=float(result['ess_observables'][i,1]),
            chain_gamma_linf=float(result['chain_linf'][i,0]),
            mean_energy_per_spin=float(result['energy_blocks'][:,:,i].mean())))
    meta=dict(signature=signature,case=case,features=image_features(image,model),
        seconds=time.perf_counter()-start,ladder=result['ladder'].tolist(),
        swap_min=float(result['swap_acceptance'].min()),
        swap_median=float(np.median(result['swap_acceptance'])),
        round_trips=int(result['round_trips'].sum()),records=records)
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def run(args):
    output=args.output; output.mkdir(parents=True,exist_ok=True)
    if not (output/'manifest.json').exists(): select_data(args.data,output,args.per_class)
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['per_class'] != args.per_class: raise ValueError('Subset size differs from cached manifest.')
    images=np.load(output/'subset.npz')['images']; jobs=[]
    for i,case in enumerate(manifest['cases']):
        if case['split'] not in args.splits or (args.classes and case['class_id'] not in args.classes): continue
        for ai,alpha in enumerate(args.alphas):
            seed=20261007+10000*i+int(round(alpha*1e6))
            options=dict(burn=args.burn,samples=args.samples,block_size=64,chains=args.chains,rng_seed=seed)
            path=output/'samples'/f'{case["name"]}-a{alpha:g}.npz'
            jobs.append((case,images[i],alpha,args.bs,options,path))
    protocol=dict(model='H=-sum atan(alpha*(125-d))*Zi*Zj/4; alpha>0; open grid',
        beta='b=beta_paper/N',alphas=args.alphas,bs=args.bs,burn=args.burn,
        samples=args.samples,chains=args.chains,block_size=64,
        mask='Evaluation proxy: intensity>=128; sensitivity at 96,160',
        sampler='systematic heat bath, signed SW cluster, adjacent parallel tempering each sweep',
        primary_seed='first minimum-intensity perimeter pixel',secondary_seed='first maximum-intensity pixel')
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Output directory already has a different protocol.')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.perf_counter(); metas=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,spec) for spec in jobs]
        for completed,future in enumerate(as_completed(futures),1):
            meta=future.result(); metas.append(meta)
            if completed%5==0 or completed==1 or completed==len(jobs):
                print(json.dumps(dict(done=completed,total=len(jobs),elapsed=round(time.perf_counter()-started,1),
                    last=meta['case']['name'],alpha=meta['signature']['alpha'],
                    best_iou=round(max(r['iou'] for r in meta['records']),3),
                    worst_rhat=round(max(r['rhat_observable_max'] for r in meta['records']),3))),flush=True)
    print(f'Finished {len(metas)} image/alpha jobs in {time.perf_counter()-started:.1f} s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=ROOT/'results'/'emnist-data')
    p.add_argument('--output',type=Path,default=ROOT/'results'/'emnist-ising')
    p.add_argument('--splits',nargs='+',choices=['train','test'],default=['train'])
    p.add_argument('--classes',nargs='+',type=int); p.add_argument('--per-class',type=int,default=1)
    p.add_argument('--alphas',nargs='+',type=float,default=ALPHAS)
    p.add_argument('--bs',nargs='+',type=float,default=BS)
    p.add_argument('--burn',type=int,default=512); p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--chains',type=int,default=2); p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
