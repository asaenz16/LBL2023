"""Build elementary circuits, MATLAB fixtures, and alternating-strip tests."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quantum-segmentation-matplotlib')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import savemat
from qiskit import qasm2,qpy
from ising_algorithm1 import ising_model,gibbs_ising,spin_basis
from ising_petz import (alternating_image,build_ising_petz,image_circuit,local_circuit,
    local_kraus,apply_dense_sweep,ladder_gibbs)
from ising_petz_populations import exact_population_sweep,population_trajectories

ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/'results'/'ising-petz'


def code_hashes():
    return {n:hashlib.sha256((ROOT/'python'/n).read_bytes()).hexdigest() for n in
        ['ising_petz.py','ising_petz_populations.py','ising_monte_carlo.py','run_ising_petz.py']}


def write_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def exact_history(model,sweeps,seeds):
    p=np.ones(2**model.n)/2**model.n;z=spin_basis(model.n);history=[]
    for t in range(sweeps+1):
        history.append(np.array([(p*z[:,seed])@z/4 for seed in seeds]))
        if t<sweeps:p=exact_population_sweep(p,model)
    return np.array(history),p


def artifacts(output):
    folder=output/'circuits';folder.mkdir(parents=True,exist_ok=True);resources=[]
    for width in [4,8,16,32,64]:
        m=build_ising_petz(alternating_image(width));c=image_circuit(m)
        (folder/f'L{width}-one-sweep.qasm').write_text(qasm2.dumps(c)+'\n')
        with (folder/f'L{width}-one-sweep.qpy').open('wb') as f:qpy.dump(c,f)
        parsed=qasm2.loads(qasm2.dumps(c));assert c.count_ops()==parsed.count_ops()
        for a,b in zip(c.data,parsed.data):
            assert a.operation.name==b.operation.name
            assert [c.find_bit(q).index for q in a.qubits]==[parsed.find_bit(q).index for q in b.qubits]
            assert np.allclose(a.operation.params,b.operation.params,rtol=0,atol=1e-14)
        serial=image_circuit(m,parallel=False)
        resources.append(dict(width=width,n_spins=m.n,system_plus_bath_qubits=c.num_qubits,
            layers=m.layers,b=m.b,beta_paper=m.b*m.n,operations=dict(c.count_ops()),depth=c.depth(),
            serial_qubits=serial.num_qubits,serial_depth=serial.depth()))
    m=build_ising_petz(alternating_image(4));seeds=[0,2];sweeps=50
    for parallel in [True,False]:
        c=image_circuit(m,sweeps,parallel=parallel,initial='plus_x',measure=True)
        (folder/f'L4-algorithm1-50-sweeps-{"parallel" if parallel else "serial"}.qasm').write_text(qasm2.dumps(c)+'\n')
    rho0=np.ones((2**m.n,2**m.n))/2**m.n;rho1=apply_dense_sweep(rho0,m)
    history,p=exact_history(m,sweeps,seeds)
    assert np.max(abs(rho1-np.diag(exact_population_sweep(np.diag(rho0),m))))<1e-13
    exact=gibbs_ising(ising_model(m.graph.image,-m.graph.alpha,threshold=m.graph.threshold),m.b*m.n)
    assert np.max(abs(ladder_gibbs(m,seeds)-exact.gamma[seeds]))<1e-13
    kraus=np.empty((1,m.n),object)
    for i in range(m.n):kraus[0,i]=np.stack(local_kraus(m,i),axis=2)
    m8=build_ising_petz(alternating_image(8));history8,p8=exact_history(m8,50,seeds)
    savemat(output/'qclab-fixture.mat',dict(image=m.graph.image,alpha=m.graph.alpha,threshold=m.graph.threshold,
        b=m.b,k=m.k,beta_paper=m.b*m.n,kraus=kraus,seeds_zero_based=np.array(seeds),
        rho_initial=rho0,rho_after_one=rho1,rho_after_50=np.diag(p),gamma_history=history,
        gamma_gibbs=exact.gamma[seeds],gamma_history_L8=history8,population_final_L8=p8,
        gamma_gibbs_L8=ladder_gibbs(m8,seeds)),do_compression=True)
    write_json(output/'resources.json',resources)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=local_circuit(m,1)
    # Circuit Qiskit wires are reversed relative to local row-major spin order.
    fig=c.draw(output='mpl',fold=-1,style={'name':'bw'},scale=.85)
    fig.suptitle('One interior Ising Petz update\nspin 3: center; spins 2, 1, 0: left, right, vertical neighbors',fontsize=12)
    fig.savefig(folder/'local-update.png',dpi=180,bbox_inches='tight');fig.savefig(folder/'local-update.svg',bbox_inches='tight');plt.close(fig)
    (folder/'local-update.txt').write_text(str(c.draw(output='text',fold=-1))+'\n')
    print('Circuit exports and MATLAB fixtures saved.',flush=True)


def study_job(spec):
    width,k,shots,sweeps,initial,output=spec
    name=f'L{width}-K{k:g}-{initial}';path=output/f'{name}.json'
    checkpoints=sorted(set(t for t in [0,1,2,4,8,12,20,30,50,100,200,sweeps] if t<=sweeps))
    seed=6211200+width*1000+int(k*100)+(100000 if initial=='all_up' else 0)
    signature=dict(width=width,k=k,shots=shots,sweeps=sweeps,initial=initial,rng_seed=seed,code_hashes=code_hashes())
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['signature']!=signature:raise ValueError(f'Conflicting cached run: {name}')
        return saved
    start=time.perf_counter();m=build_ising_petz(alternating_image(width),k=k);seeds=[0,2]
    target=ladder_gibbs(m,seeds);r=population_trajectories(m,checkpoints,shots=shots,seeds=seeds,rng_seed=seed,initial=initial)
    np.savez_compressed(output/f'{name}.npz',gamma_gibbs=target,**r)
    truth=m.graph.image.ravel()>0;align=np.where(truth,-1,1);records=[]
    for t,sweep in enumerate(checkpoints):
        g=r['gamma'][t,0];prediction=g<=0;tp=np.sum(prediction&truth);union=np.sum(prediction|truth)
        records.append(dict(sweep=sweep,iou=float(tp/max(1,union)),accuracy=float(np.mean(prediction==truth)),
            resolved_fraction=float(np.mean(abs(g)>r['simultaneous_radius'])),
            correct_resolved_fraction=float(np.mean(g*align>r['simultaneous_radius'])),
            gamma_linf=float(np.max(abs(r['gamma'][t]-target))),far_aligned=float(-g[-1]),
            far_se=float(r['se'][t,0,-1])))
    result=dict(signature=signature,b=m.b,beta_paper=m.b*m.n,threshold=100,alpha=.001,
        exact_gibbs_far=float(-target[0,-1]),exact_min_abs_gamma=float(np.min(abs(target[0]))),
        simultaneous_radius=r['simultaneous_radius'],records=records,seconds=time.perf_counter()-start,
        exact_gibbs_iou=float(np.sum((target[0]<=0)&truth)/np.sum((target[0]<=0)|truth)),
        ideal_equilibrium_shots_for_all_signs=int(np.ceil(np.log(2*m.n/.05)/(8*np.min(abs(target[0]))**2))),
        engine='Independent population trajectories of the validated quantum channel; no hardware or full-state QCLAB execution at this size')
    write_json(path,result);return result


def study(args):
    out=args.output/'study';out.mkdir(parents=True,exist_ok=True)
    jobs=[(l,k,args.shots,args.sweeps,'plus_x',out) for k in [.8,1.2] for l in [4,8,16,32,64]]
    jobs += [(l,.8,args.shots,args.sweeps,'all_up',out) for l in [4,64]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(study_job,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();print(json.dumps(dict(done=i,total=len(jobs),case=r['signature'],
                final=r['records'][-1],seconds=round(r['seconds'],2))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['artifacts','study'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--shots',type=int,default=65536)
    p.add_argument('--sweeps',type=int,default=200);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.action=='artifacts':artifacts(args.output)
    else:study(args)


if __name__=='__main__':main()
