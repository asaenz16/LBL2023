"""Local finite-collision approximation error, separate from spatial splitting."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from estimate_bath_circuits import collision_bath
from thermal_bath import channel_matrix,davies_generator,trace_distance
from bath_modifications import plaquette_hamiltonian,plaquette_generator
from ladder_tensor import X,Y,Z,build_ladder


def main():
    delta=float(build_ladder(4).horizontal_delta[1]);rng=np.random.default_rng(1200);rows=[]
    for k,d,name in [(2,2.,'bond_same'),(2,delta,'bond_boundary'),(4,2.,'plaquette_same'),(4,delta,'plaquette_boundary')]:
        h=(-.25*(np.kron(X,X)+np.kron(Y,Y)+d*np.kron(Z,Z)) if k==2 else plaquette_hamiltonian(d))
        weights=[1.]*2 if k==2 else [1.5]*4
        local=collision_bath(h,6.,.1,weights)
        generator=davies_generator(local) if k==2 else plaquette_generator(d,6.,1.5,1.5)
        exact=expm(.1*generator);dim=2**k
        states=[np.eye(dim)/dim]+[np.outer(v,v.conj()) for v in np.eye(dim)]
        for _ in range(8):
            v=rng.normal(size=dim)+1j*rng.normal(size=dim);v/=np.linalg.norm(v);states.append(np.outer(v,v.conj()))
        for subdivisions in [1,4,16,64]:
            approx=np.linalg.matrix_power(channel_matrix(collision_bath(h,6.,.1/subdivisions,weights)),subdivisions)
            errors=[trace_distance((approx@rho.reshape(-1,order='F')).reshape(dim,dim,order='F'),
                                   (exact@rho.reshape(-1,order='F')).reshape(dim,dim,order='F')) for rho in states]
            row={'template':name,'duration':.1,'subdivisions':subdivisions,'sample_states':len(states),
                 'maximum_sampled_trace_error':max(errors),'frobenius_superoperator_error':float(np.linalg.norm(approx-exact))}
            rows.append(row);print(row,flush=True)
    out=Path(__file__).resolve().parents[1]/'results'/'bath-modifications'/'collision-calibration.json'
    out.write_text(json.dumps({'results':rows,'limitations':'Sampled trace distances are not worst-case channel error bounds; local tests do not certify the full preparation circuit.'},indent=2)+'\n')


if __name__=='__main__':main()
