"""Scan finite Hamiltonian-then-edge-bath stages and exposure-matched controls."""
import os
import tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
for key in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sequential_bath import ExactEdgeDynamics
from ladder_tensor import build_ladder
from run_bath_modifications import dense_profiles,decorate
from thermal_bath import trace_distance


def run(dynamics,ladder,ground,t_h,t_b,protocol,output,max_exposure=12.,tolerance=1e-9,minimum_exposure=3.):
    stem=f'{protocol}-L{ladder.width}-H{t_h:g}-B{t_b:g}'
    path=output/(stem+'.json')
    if path.exists():
        print('Existing',stem,flush=True);return
    start=time.time();v=dynamics.identity.copy();unitary=dynamics.unitary(t_h)
    records=[];stable=0;max_energy_error=0.;previous_bath=None
    ground_energy=float(dynamics.evals[0])
    gibbs_weights=np.exp(-6*(dynamics.evals-ground_energy));gibbs_weights/=gibbs_weights.sum()
    gibbs=(dynamics.evecs*gibbs_weights)@dynamics.evecs.conj().T
    def measure(vector,cycle,phase,energy_before=None):
        matrix=dynamics.unpack(vector)
        profiles=decorate(dense_profiles(matrix,ladder.width),ladder,ground['seed_profiles'])
        energy=float(np.einsum('ij,ji->',matrix,dynamics.h).real)
        out={'cycle':cycle,'phase':phase,'bath_exposure':(cycle-(phase=='after_H'))*t_b,
             'hamiltonian_exposure':cycle*t_h,'energy':energy,
             'energy_above_ground_per_spin':(energy-ground_energy)/ladder.n_spins,
             'seed_profiles':profiles,'far_aligned':-profiles[0]['gamma'][1][-1],
             'interior_aligned':-profiles[1]['gamma'][1][3*ladder.width//4]}
        if energy_before is not None:out['energy_change_in_stage']=energy-energy_before
        return out
    initial=measure(v,0,'initial');records.append(initial);last_energy=initial['energy']
    max_cycles=int(np.ceil(max_exposure/t_b))
    for cycle in range(1,max_cycles+1):
        before=v.copy()
        after_h,v=dynamics.cycle(v,t_h,t_b,protocol,unitary)
        energy_before_bath=last_energy
        if after_h is not None:
            row=measure(after_h,cycle,'after_H',last_energy);records.append(row)
            energy_before_bath=row['energy']
            max_energy_error=max(max_energy_error,abs(row['energy_change_in_stage']))
        row=measure(v,cycle,'after_bath' if protocol=='sequential' else 'after_joint',energy_before_bath)
        # ||A||_1 <= sqrt(d)||A||_F: conservative full-state convergence test.
        bound=float(np.sqrt(dynamics.d)*np.linalg.norm(v-before)/2)
        row['cycle_change_trace_distance_upper_bound']=bound
        if previous_bath is not None:
            row['profile_change']=max(np.max(abs(np.asarray(p['gamma'])-np.asarray(q['gamma'])))
                for p,q in zip(row['seed_profiles'],previous_bath['seed_profiles']))
        records.append(row);previous_bath=row;last_energy=row['energy']
        stable=stable+1 if bound<tolerance else 0
        if cycle*t_b>=minimum_exposure and stable>=3:break
    after_h,future=dynamics.cycle(v,t_h,t_b,protocol,unitary)
    final=dynamics.unpack(v)
    diagnostic={'trace':float(np.trace(final).real),'trace_imaginary':float(np.trace(final).imag),
                'hermiticity_error':float(np.linalg.norm(final-final.conj().T)),
                'minimum_eigenvalue':float(np.linalg.eigvalsh((final+final.conj().T)/2).min()),
                'cycle_residual_trace_distance':trace_distance(dynamics.unpack(future),final),
                'distance_gibbs':trace_distance(final,gibbs),
                'max_H_energy_conservation_error':max_energy_error,
                'sector_leakage':dynamics.sector_leakage}
    if after_h is not None:
        diagnostic['within_cycle_trace_distance']=trace_distance(dynamics.unpack(after_h),final)
        diagnostic['within_cycle_profile_change']=max(np.max(abs(np.asarray(p['gamma'])-np.asarray(q['gamma'])))
            for p,q in zip(measure(after_h,cycle+1,'after_H')['seed_profiles'],records[-1]['seed_profiles']))
    report={'protocol':protocol,'width':ladder.width,'n_spins':ladder.n_spins,'t_h':t_h,'t_b':t_b,
            'H_to_bath_ratio':t_h/t_b,'b':6.,'normalization':'sample','initial_state':'maximally mixed',
            'bath_definition':'sum of edge-local Davies generators, same XYZ proposals as ladder_bath.py',
            'integration':'spectral full-H unitary and sparse exponential action of full bath sum; no Trotter splitting',
            'operator_sector_dimension':len(dynamics.sector),'stabilized':stable>=3,'tolerance':tolerance,
            'max_exposure':max_exposure,'minimum_exposure':minimum_exposure,'records':records,
            'diagnostics':diagnostic,'runtime_seconds':time.time()-start,
            'comparison':'Simultaneous cycle exp(t_h*C+t_b*D) matches integrated generator strengths. '
                         'Sequential cycle exp(t_b*D) exp(t_h*C).'}
    np.savez_compressed(output/(stem+'.npz'),rho=final,rho_after_H=dynamics.unpack(after_h) if after_h is not None else final)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(stem,'cycles',cycle,'u',round(cycle*t_b,3),'far',round(records[-1]['far_aligned'],8),
          'E/spin',round(records[-1]['energy_above_ground_per_spin'],6),
          'residual',f"{diagnostic['cycle_residual_trace_distance']:.2e}",
          'stable',stable>=3,'seconds',round(time.time()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t-h',type=float,nargs='+',default=[0.,.05,.2,.5,1.,2.,4.])
    p.add_argument('--t-b',type=float,nargs='+',default=[.05,.2,.5,1.,2.])
    p.add_argument('--protocol',choices=['sequential','simultaneous'],default='sequential')
    p.add_argument('--ratios',type=float,nargs='+',help='For joint controls, use t_b=.25 and these H/bath ratios.')
    p.add_argument('--max-exposure',type=float,default=12.)
    p.add_argument('--tolerance',type=float,default=1e-9)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'sequential-bath')
    a=p.parse_args()
    if min(a.t_h)<0 or min(a.t_b)<=0 or a.max_exposure<=0:p.error('Nonnegative H times and positive bath times/exposure required.')
    if a.ratios is not None and (a.protocol!='simultaneous' or min(a.ratios)<0):p.error('Ratios require simultaneous protocol and nonnegative values.')
    a.output.mkdir(parents=True,exist_ok=True)
    ladder=build_ladder(4);dynamics=ExactEdgeDynamics(ladder)
    ground=json.loads((Path(__file__).resolve().parents[1]/'results'/'ladders'/'sample'/'ground-L4.json').read_text())
    pairs=[(r*.25,.25) for r in a.ratios] if a.ratios is not None else [(h,b) for h in a.t_h for b in a.t_b]
    for h,b in pairs:run(dynamics,ladder,ground,h,b,a.protocol,a.output,a.max_exposure,a.tolerance)


if __name__=='__main__':main()
