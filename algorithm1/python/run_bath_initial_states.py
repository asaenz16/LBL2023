"""Compare bath-only evolution from I/d and the pure all-+x spin state."""
import os
import tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
for key in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
from sequential_bath import ExactEdgeDynamics
from ladder_tensor import build_ladder
from run_bath_modifications import dense_profiles,decorate
from thermal_bath import trace_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dt',type=float,default=.05,help='Output interval; no Trotter splitting is used.')
    p.add_argument('--exposure',type=float,default=8.)
    p.add_argument('--tolerance',type=float,default=1e-10)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'bath-plus-x')
    a=p.parse_args()
    if min(a.dt,a.exposure,a.tolerance)<=0 or not np.isfinite(a.dt+a.exposure+a.tolerance):p.error('Require positive finite parameters.')
    a.output.mkdir(parents=True,exist_ok=True)
    start=time.time();ladder=build_ladder(4);d=ExactEdgeDynamics(ladder,sector='all')
    ground=json.loads((Path(__file__).resolve().parents[1]/'results'/'ladders'/'sample'/'ground-L4.json').read_text())
    v=np.column_stack([d.identity,d.pack(np.ones((d.d,d.d))/d.d)])
    zero_mask=np.zeros(len(v),bool);zero_mask[d.zero_charge_indices]=True
    rows=np.arange(d.d);flipped=[rows^(1<<(d.n-1-j)) for j in range(d.n)]
    generator=a.dt*d.diss;generator_trace=generator.diagonal().sum()
    records=[];stable=0
    def measure(step,states):
        matrices=[d.unpack(states[:,j]) for j in range(2)];out={}
        for name,matrix,vector in zip(['mixed','plus_x'],matrices,states.T):
            profiles=decorate(dense_profiles(matrix,4),ladder,ground['seed_profiles'])
            out[name]={'seed_profiles':profiles,'far_aligned':-profiles[0]['gamma'][1][-1],
                       'interior_aligned':-profiles[1]['gamma'][1][3],
                       'energy':float(np.einsum('ij,ji->',matrix,d.h).real),
                       'purity':float(np.einsum('ij,ji->',matrix,matrix).real),
                       'x_magnetization_per_spin':float(np.mean([matrix[rows,indices].sum().real/2 for indices in flipped])),
                       'outside_zero_sector_frobenius_norm':float(np.linalg.norm(vector[~zero_mask]))}
            out[name]['energy_above_ground_per_spin']=(out[name]['energy']-d.evals[0])/d.n
        profile_distance=max(np.max(abs(np.asarray(x['gamma'])-np.asarray(y['gamma'])))
            for x,y in zip(out['mixed']['seed_profiles'],out['plus_x']['seed_profiles']))
        return {'sweep':step,'time':step*a.dt,'states':out,
                'profile_linf_between_initializations':float(profile_distance),
                'trace_distance_between_initializations':trace_distance(*matrices)}
    records.append(measure(0,v))
    for step in range(1,int(np.ceil(a.exposure/a.dt))+1):
        previous=v.copy();v=expm_multiply(generator,v,traceA=generator_trace)
        row=measure(step,v)
        bound=float(np.sqrt(d.d)*max(np.linalg.norm(v[:,j]-previous[:,j]) for j in range(2))/2)
        row['maximum_step_change_trace_distance_upper_bound']=bound
        records.append(row);stable=stable+1 if bound<a.tolerance else 0
        if step%20==0:
            print('u',round(step*a.dt,3),'far mixed/plus_x',row['states']['mixed']['far_aligned'],row['states']['plus_x']['far_aligned'],
                  'D between',row['trace_distance_between_initializations'],'sec',round(time.time()-start,1),flush=True)
        if step*a.dt>=3 and stable>=3:break
    future=expm_multiply(generator,v,traceA=generator_trace);diagnostics={}
    for j,name in enumerate(['mixed','plus_x']):
        matrix=d.unpack(v[:,j]);diagnostics[name]={
            'trace':float(np.trace(matrix).real),'hermiticity_error':float(np.linalg.norm(matrix-matrix.conj().T)),
            'minimum_eigenvalue':float(np.linalg.eigvalsh((matrix+matrix.conj().T)/2).min()),
            'final_cycle_residual':trace_distance(matrix,d.unpack(future[:,j]))}
    report={'width':4,'n_spins':8,'b':6.,'normalization':'sample','dt':a.dt,'coherent_H':False,
            'operator_space_dimension':len(d.sector),'operator_sector':'all','stabilized':stable>=3,
            'convergence_observable':'full-state trace distance upper bound, for both initializations',
            'tolerance':a.tolerance,'records':records,'diagnostics':diagnostics,'runtime_seconds':time.time()-start,
            'integration':'Sparse exponential of the sum of edge Davies generators; no Trotter splitting or density truncation.',
            'initial_states':{'mixed':'I/256','plus_x':'(|+><+|) tensor 8; coherent pure state; no magnetization twirling'}}
    (a.output/'exact-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'exact-comparison.npz',mixed=d.unpack(v[:,0]),plus_x=d.unpack(v[:,1]))
    print('Finished',a.output,'stabilized',stable>=3,'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
