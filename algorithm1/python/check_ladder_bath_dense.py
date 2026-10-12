"""Independent Qiskit density-matrix calibration of the 2x4 local bath."""
import os
import tempfile
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import DensityMatrix,Kraus
from ladder_tensor import build_ladder,I2,X,Z,dense_row_hamiltonian,rung_to_row_indices
from ladder_bath import bond_kraus
from thermal_bath import trace_distance


def evolve_channel(rho,kraus,sites):
    return DensityMatrix(rho).evolve(Kraus(list(kraus)),[7-j for j in reversed(sites)]).data


def step(rho,ladder,axis,p,dt=.1):
    for j in range(4): rho=evolve_channel(rho,bond_kraus(2.,6.,dt/2),(2*j,2*j+1))
    for parity,factor in [(0,.5),(1,1.),(0,.5)]:
        for j in range(parity,3,2):
            for row in range(2):
                rho=evolve_channel(rho,bond_kraus(float(ladder.horizontal_delta[j]),6.,dt*factor),(2*j+row,2*j+row+2))
    for j in range(4): rho=evolve_channel(rho,bond_kraus(2.,6.,dt/2),(2*j,2*j+1))
    op={'X':X,'Z':Z}[axis]
    if p:
        for j in range(8): rho=evolve_channel(rho,[np.sqrt(1-p)*I2,np.sqrt(p)*op],(j,))
    return rho


def profiles(rho):
    spins=((1-2*((np.arange(256)[:,None]>>np.arange(7,-1,-1))&1))/2).reshape(256,4,2).transpose(0,2,1)
    return [np.einsum('i,i,irc->rc',np.diag(rho).real,spins[:,0,col],spins).tolist() for col in [0,1]]


def main():
    out=Path(__file__).resolve().parents[1]/'results'/'ladders'/'sample'
    out.mkdir(parents=True,exist_ok=True)
    ladder=build_ladder(4); perm=rung_to_row_indices(4)
    h=dense_row_hamiltonian(ladder)[np.ix_(perm,perm)]
    gibbs=expm(-6*(h-np.linalg.eigvalsh(h)[0]*np.eye(256))); gibbs/=np.trace(gibbs)
    results=[]
    for axis,p in [('Z',0.),('Z',.02),('X',.02)]:
        rho=np.eye(256)/256
        for sweep in range(60): rho=step(rho,ladder,axis,p)
        results.append({'axis':axis,'p':p,'sweeps':60,'gamma':profiles(rho),
                        'state_distance_to_global_gibbs':trace_distance(rho,gibbs),
                        'full_state_one_cycle_residual':trace_distance(step(rho,ladder,axis,p),rho),
                        'minimum_eigenvalue':float(np.linalg.eigvalsh(rho).min()),
                        'trace':float(np.trace(rho).real)})
        print('dense',axis,p,'D Gibbs',results[-1]['state_distance_to_global_gibbs'],
              'far',results[-1]['gamma'][0][1][-1],flush=True)
    report={'width':4,'beta_effective':6.,'results':results,'gibbs_profiles':profiles(gibbs),
            'gibbs_stationarity_bias_one_local_bath_cycle':trace_distance(step(gibbs,ladder,'Z',0),gibbs)}
    (out/'bath-dense-validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
