"""Ground-state and isolated-quench tests on alternating-block 2 x 4N images."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR",os.path.join(tempfile.gettempdir(),"quantum-segmentation-numba"))
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1")
import numpy as np
import quimb.tensor as qtn
from ladder_tensor import (build_ladder,ground_state,energy,seed_profile,rotate,
                           save_mps,load_mps,rung_arrays)


def overlap(left,right):
    env=np.ones((1,1),complex)
    for a,b in zip(rung_arrays(left),rung_arrays(right)):
        env=sum(a[:,:,s].conj().T@env@b[:,:,s] for s in range(4))
    return env[0,0]


def trajectory(ladder,ground,angle,times,dt,bond,pattern="uniform"):
    signs=None
    if pattern == "random":
        signs=np.random.default_rng(1200+ladder.width).choice([-1.,1.],size=(2,ladder.width))
    initial=rotate(ground,angle,signs=signs)
    evolution=qtn.TEBD(initial,ladder.local_ham,dt=dt,
                       split_opts={"max_bond":bond,"cutoff":1e-11},progbar=False)
    records=[]; seeds=[(0,0),(0,ladder.width//4)]
    baseline=[seed_profile(ground,seed)['gamma'] for seed in seeds]
    truth=np.where(ladder.image == 0,1.,-1.)
    for t,state in zip(times,evolution.at_times(times,order=2,progbar=False)):
        entries=[]
        for seed,reference in zip(seeds,baseline):
            stats=seed_profile(state,seed); signed=stats['gamma']*truth*truth[seed]
            mask=np.ones((2,ladder.width),bool); mask[seed]=False
            entries.append({"seed":list(seed),"gamma":stats['gamma'].tolist(),
                            "connected":stats['connected'].tolist(),"means":stats['means'].tolist(),
                            "linf_reference":float(np.max(abs(stats['gamma'][mask]-reference[mask]))),
                            "signed_margin":float(signed[mask].min()),
                            "fraction_signs_correct":float(np.mean(signed[mask]>0))})
        norm=seed_profile(state)['norm']
        fidelity=abs(overlap(ground,state))**2/norm
        records.append({"time":float(t),"seed_profiles":entries,"max_bond":state.max_bond(),
                        "norm":norm,"energy":energy(state,ladder)/norm,
                        "distance_from_ground":float(np.sqrt(max(0.,1-fidelity)))})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widths',type=int,nargs='+',default=[4,8,12,16])
    parser.add_argument('--normalization',choices=['sample','bulk'],default='sample')
    parser.add_argument('--angles',type=float,nargs='+',default=[0.,.15,.3,.6])
    parser.add_argument('--pattern',choices=['uniform','random'],default='uniform')
    parser.add_argument('--tmax',type=float,default=6.)
    parser.add_argument('--dt',type=float,default=.04)
    parser.add_argument('--bond',type=int,default=64)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'ladders')
    args=parser.parse_args()
    if args.tmax <= 0 or args.dt <= 0 or args.bond < 4: parser.error('Require positive times and bond >=4.')
    root=args.output/args.normalization; root.mkdir(parents=True,exist_ok=True)
    times=np.linspace(0,args.tmax,round(args.tmax/.2)+1)
    for width in args.widths:
        ladder=build_ladder(width,normalization=args.normalization)
        ground_file=root/f'ground-L{width}.npz'; info_file=root/f'ground-L{width}.json'
        if ground_file.exists() and info_file.exists():
            ground=load_mps(ground_file); info=json.loads(info_file.read_text())
        else:
            start=time.time()
            ground,info=ground_state(ladder,bond_dims=(16,32,64),cutoff=1e-12,max_sweeps=20,tolerance=1e-9)
            if info['variance'] > 1e-6:
                raise RuntimeError(f'Ground-state variance too large at L={width}: {info["variance"]}')
            info.update({"width":width,"n_spins":2*width,"normalization":args.normalization,
                         "sigma":ladder.sigma,"horizontal_delta":ladder.horizontal_delta.tolist(),
                         "image":ladder.image.tolist(),"runtime_seconds":time.time()-start,
                         "seed_profiles":[{"seed":list(seed),**{k:np.asarray(v).tolist() for k,v in seed_profile(ground,seed).items()}}
                                          for seed in [(0,0),(0,width//4)]]})
            save_mps(ground_file,ground); info_file.write_text(json.dumps(info,indent=2)+'\n')
        print(f'L={width}: ground E={info["energy"]:.10f}, variance={info["variance"]:.2e}',flush=True)
        for angle in args.angles:
            name=f'quench-L{width}-{args.pattern}-theta{angle:g}-dt{args.dt:g}-chi{args.bond}.json'
            path=root/name
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['tmax'] == args.tmax:
                    print('Using existing',name,flush=True); continue
            start=time.time()
            records=trajectory(ladder,ground,angle,times,args.dt,args.bond,args.pattern)
            report={"width":width,"n_spins":2*width,"normalization":args.normalization,
                    "angle":angle,"axis":"X","pattern":args.pattern,"rotation_seed":1200+width,
                    "tmax":args.tmax,"dt":args.dt,"max_bond_setting":args.bond,"cutoff":1e-11,
                    "method":"even-parity DMRG state, one Rx rotation layer, isolated TEBD order 2",
                    "records":records,"runtime_seconds":time.time()-start,
                    "max_energy_drift":max(abs(row['energy']-records[0]['energy']) for row in records),
                    "max_local_magnetization":max(np.max(abs(np.array(prof['means']))) for row in records for prof in row['seed_profiles'])}
            path.write_text(json.dumps(report,indent=2)+'\n')
            print(f'{name}: margin={records[-1]["seed_profiles"][0]["signed_margin"]:.5f}; '
                  f'drift={report["max_energy_drift"]:.2e}; seconds={report["runtime_seconds"]:.1f}',flush=True)


if __name__ == '__main__': main()
