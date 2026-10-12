"""Local-bond bath and sustained-noise scaling study; no global Gibbs guarantee."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import numpy as np
from ladder_tensor import build_ladder,save_mps,load_mps
from ladder_bath import bath_step,mixed_density,plus_x_density,density_profile,density_reduced,density_energy


def summarize(rho,ladder):
    entries=[]; truth=np.where(ladder.image==0,1.,-1.)
    for seed in [(0,0),(0,ladder.width//4)]:
        stats=density_profile(rho,seed); mask=np.ones((2,ladder.width),bool); mask[seed]=False
        signed=stats['gamma']*truth*truth[seed]
        entries.append({'seed':list(seed),'gamma':stats['gamma'].tolist(),
                        'connected':stats['connected'].tolist(),'means':stats['means'].tolist(),
                        'signed_margin':float(signed[mask].min()),
                        'fraction_signs_correct':float(np.mean(signed[mask]>1e-10))})
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--widths',type=int,nargs='+',default=[4,8,12,16])
    p.add_argument('--normalization',choices=['sample','bulk'],default='sample')
    p.add_argument('--cases',nargs='+',default=['Z:0','Z:0.02','X:0.02'])
    p.add_argument('--sweeps',type=int,default=60)
    p.add_argument('--duration',type=float,default=.1)
    p.add_argument('--bond',type=int,default=32)
    p.add_argument('--initial-state',choices=['mixed','plus_x'],default='mixed')
    p.add_argument('--resume-bond',type=int,default=None,help='Continue a saved lower-bond state for convergence checking.')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'ladders')
    args=p.parse_args()
    root=args.output/args.normalization; root.mkdir(parents=True,exist_ok=True)
    for width in args.widths:
        ladder=build_ladder(width,normalization=args.normalization)
        for case in args.cases:
            axis,prob=case.split(':');prob=float(prob)
            if axis not in ['Z','X'] or not 0 <= prob <= .5: p.error('Use X:p or Z:p with p in [0,0.5].')
            # Keep the added-noise rate fixed when halving the bath time step.
            effective_p=.5*(1-(1-2*prob)**(args.duration/.1))
            stem=f'bath-L{width}-{axis}-p{prob:g}-dt{args.duration:g}-chi{args.bond}'
            if args.initial_state!='mixed':stem+=f'-init{args.initial_state}'
            path=root/(stem+'.json')
            if path.exists():
                print('Using existing',stem,flush=True);continue
            start=time.time(); initial_step=0
            if args.resume_bond is None:
                rho=mixed_density(width) if args.initial_state=='mixed' else plus_x_density(width)
            else:
                source=root/f'bath-L{width}-{axis}-p{prob:g}-dt{args.duration:g}-chi{args.resume_bond}'
                if args.initial_state!='mixed':source=Path(str(source)+f'-init{args.initial_state}')
                previous=json.loads(Path(str(source)+'.json').read_text())
                rho=load_mps(Path(str(source)+'.npz'))
                for tensor in rho.tensors:
                    tensor.modify(data=np.real_if_close(tensor.data))
                initial_step=previous['last_sweep']
            def additional_observables():
                if args.initial_state=='mixed':return {}
                return {'energy':density_energy(rho,ladder),
                        'x_magnetization_per_spin':float(np.mean(density_profile(rho,axis='X')['means']))}
            records=[{'sweep':initial_step,'seed_profiles':summarize(rho,ladder),'max_bond':rho.max_bond(),
                      **additional_observables()}]
            drifts=[];stable=0;changes=[]
            for step in range(1,args.sweeps+1):
                drifts.append(bath_step(rho,ladder,b=6.,duration=args.duration,axis=axis,
                                        probability=effective_p,max_bond=args.bond,cutoff=1e-12))
                if step%5==0 or step==args.sweeps:
                    profiles=summarize(rho,ladder)
                    change=max(np.max(abs(np.asarray(now['gamma'])-np.asarray(old['gamma'])))
                               for now,old in zip(profiles,records[-1]['seed_profiles']))
                    changes.append(float(change));stable=stable+1 if change < 1e-7 else 0
                    records.append({'sweep':initial_step+step,'seed_profiles':profiles,
                                    'max_bond':rho.max_bond(),'change_since_previous_record':float(change),
                                    **additional_observables()})
                    if step%20==0:
                        print(stem,'step',initial_step+step,'change',f'{change:.2e}',flush=True)
                    if stable>=3: break
            min_eig=1.; herm_error=0.
            for col in range(width-1):
                reduced=density_reduced(rho,col,2)
                min_eig=min(min_eig,float(np.linalg.eigvalsh((reduced+reduced.conj().T)/2).min()))
                herm_error=max(herm_error,float(np.linalg.norm(reduced-reduced.conj().T)))
            report={'width':width,'n_spins':2*width,'normalization':args.normalization,
                    'sigma':ladder.sigma,'beta_effective':6.,'beta_paper':12*width,
                    'axis':axis,'p_per_reference_step':prob,'actual_p_per_step':effective_p,
                    'dt':args.duration,'max_bond_setting':args.bond,'cutoff':1e-12,
                    'initial_state':args.initial_state,
                    'initial_sweep':initial_step,'last_sweep':initial_step+step,
                    'resumed_from_bond':args.resume_bond,'records':records,
                    'observable_stabilized':stable>=3,'max_trace_renormalization':max(drifts),
                    'last_trace_renormalization':drifts[-1],
                    'minimum_two_rung_eigenvalue':min_eig,'maximum_two_rung_hermiticity_error':herm_error,
                    'runtime_seconds':time.time()-start,
                    'limitations':['Local-bond thermalization is not globally Gibbs preserving.',
                                   'Density-MPS truncation is not positivity preserving; reduced positivity and bond convergence are diagnostics, not a proof of global positivity.',
                                   'Stabilization criterion checks two seed correlation profiles, not the full state.']}
            # Density tensors have local dimension 16; save_mps preserves it.
            save_mps(root/(stem+'.npz'),rho)
            path.write_text(json.dumps(report,indent=2)+'\n')
            print(stem,'finished',initial_step+step,'far',records[-1]['seed_profiles'][0]['gamma'][1][-1],
                  'drift',max(drifts),'min_reduced_eig',min_eig,'seconds',report['runtime_seconds'],flush=True)


if __name__=='__main__': main()
