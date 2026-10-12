"""Prepare independent MATLAB/QCLAB fixtures for arbitrary natural images."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
from scipy.io import savemat
from qiskit import qasm2
from ising_petz import build_ising_petz,image_circuit,local_kraus,apply_dense_sweep
from ising_petz_populations import exact_population_sweep,population_trajectories
from ising_grid_transfer import exact_grid_gibbs
from ising_algorithm1 import spin_basis
from weizmann_petz_data import ROOT,OUT,write_json


def main():
    manifest=json.loads((ROOT/'weizmann-petz-manifest.json').read_text())
    source=manifest['test_ids'][0];fixture={};resources=[]
    for name,case_id in [('small',source+'--strip-4-boundary'),('square8',source+'--square-8'),
                         ('square16',source+'--square-16')]:
        f=np.load(OUT/'cases'/f'{case_id}.npz');image=f['image'];seeds=f['seeds'];m=build_ising_petz(image)
        fixture['image_'+name]=image;fixture['seeds_'+name]=seeds+1
        fixture['b_'+name]=m.b;fixture['reference_'+name]=exact_grid_gibbs(m,seeds)
        c=image_circuit(m,initial='plus_x',measure=True)
        folder=OUT/'circuits';folder.mkdir(exist_ok=True)
        (folder/f'{name}-one-sweep.qasm').write_text(qasm2.dumps(c)+'\n')
        bare=image_circuit(m)
        resources.append(dict(name=name,source_id=source,n=m.n,qubits=c.num_qubits,
            layers=len(m.layers),depth_per_sweep=bare.depth(),operations=dict(bare.count_ops())))
        if name=='small':
            z=spin_basis(m.n);p=np.ones(2**m.n)/2**m.n;history=[]
            for t in range(21):
                history.append(np.array([(p*z[:,s])@z/4 for s in seeds]))
                if t<20:p=exact_population_sweep(p,m)
            fixture['history_small']=np.array(history);fixture['rho_small']=np.diag(p)
        if name=='square8':
            r=population_trajectories(m,[0,20],shots=32768,seeds=seeds,rng_seed=171711)
            fixture['gamma_square8_20']=r['gamma'][-1];fixture['se_square8_20']=r['se'][-1]
        center=(image.shape[1]+1) if name!='small' else 1
        fixture['center_'+name]=center+1
        fixture['kraus_'+name]=np.stack(local_kraus(m,center),axis=2)
    image=np.array([[0,110,20],[40,80,200],[240,30,90]],float)
    m=build_ising_petz(image);rho=np.ones((512,512))/512
    for _ in range(2):rho=apply_dense_sweep(rho,m)
    fixture.update(image_nine=image,b_nine=m.b,rho_nine=rho,seeds_nine=np.array([1,5]),
                   reference_nine=exact_grid_gibbs(m,[0,4]))
    savemat(OUT/'matlab-fixture.mat',fixture,do_compression=True)
    write_json(OUT/'resources.json',resources)
    print(json.dumps(resources,indent=2),flush=True)


if __name__=='__main__':main()
