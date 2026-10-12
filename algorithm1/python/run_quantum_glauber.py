"""Compile and verify physical XXZ heat-bath circuits on 4 or 8 system spins."""
import os
for key in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
import tempfile
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "quantum-segmentation-numba"))
import argparse
import gzip
import json
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
from scipy.io import savemat
from qiskit import QuantumCircuit, QuantumRegister, qasm2, qpy
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import partial_trace
from algorithm1 import image_model
from thermal_bath import trace_distance
from quantum_glauber import (build_quantum_heat_bath, apply_update, apply_sweep,
    full_kraus, recovery_circuit, update_circuit, sweep_circuit, compile_circuit,
    elementary_isometry_columns, ensemble_circuit_density, preparation_circuit,
    bit_flip_layer, apply_bit_flips)


def image_hamiltonian(n):
    if n == 4:
        model = image_model([[0,0],[255,255]])
        return model.image, model.hamiltonian, model.sigma
    if n == 8:
        from ladder_tensor import build_ladder, dense_row_hamiltonian
        model = build_ladder(4)
        return model.image, dense_row_hamiltonian(model), model.sigma
    raise ValueError("This experiment uses 4 or 8 spins.")


def initial_density(image, kind, corruption=.1):
    n = image.size; d = 2**n
    if kind == "mixed": return np.eye(d)/d
    if kind == "plus_x": return np.ones((d,d))/d
    z = 1-2*((np.arange(d)[:,None] >> np.arange(n-1,-1,-1)) & 1)
    truth = np.where(image.ravel() == 0, 1, -1)
    if kind == "ordered":
        p = (np.all(z == truth,axis=1).astype(float)+np.all(z == -truth,axis=1))/2
    elif kind == "corrupted":
        p = (np.prod(np.where(z == truth,1-corruption,corruption),axis=1) +
             np.prod(np.where(z == -truth,1-corruption,corruption),axis=1))/2
    else: raise ValueError(kind)
    return np.diag(p)


def observe(rho, bath, image):
    n = bath.n; rows=np.arange(2**n)
    z = 1-2*((rows[:,None] >> np.arange(n-1,-1,-1)) & 1)
    probabilities=np.diag(rho).real
    means=probabilities@z/2
    gamma=np.array([(probabilities*z[:,s])@z/4 for s in [0,1]])
    target=np.array([(np.diag(bath.sigma).real*z[:,s])@z/4 for s in [0,1]])
    truth=np.where(image.ravel()==0,1,-1)
    return {"gamma":gamma.reshape(2,*image.shape).tolist(),
            "connected":(gamma-means[[0,1],None]*means).reshape(2,*image.shape).tolist(),
            "means":means.reshape(image.shape).tolist(),
            "far_aligned":float(gamma[0,-1]*truth[0]*truth[-1]),
            "profile_linf_to_gibbs":float(abs(gamma-target).max()),
            "trace_distance_to_gibbs":trace_distance(rho,bath.sigma),
            "energy":float(np.einsum('ij,ji->',rho,bath.hamiltonian).real),
            "purity":float(np.einsum('ij,ji->',rho,rho).real),
            "x_magnetization_per_spin":float(np.mean([rho[rows,rows^(1<<q)].sum().real/2 for q in range(n)])),
            "trace_error":float(abs(np.trace(rho)-1)),
            "hermiticity_error":float(np.linalg.norm(rho-rho.conj().T)),
            "minimum_eigenvalue":float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min())}


def gate_array(circuit):
    """Portable rows: opcode (1=u3,2=cx), q, target/-1, theta,phi,lambda."""
    rows=[]
    for ins in circuit.data:
        qs=[circuit.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "u3": rows.append([1,qs[0],-1,*map(float,ins.operation.params)])
        elif ins.operation.name == "cx": rows.append([2,*qs,0,0,0])
        else: raise ValueError(ins.operation.name)
    return np.array(rows)


def build_elementary_update(recovery, n, site):
    """Explicit erasure/reset plus a previously compiled recovery circuit."""
    q=n-1-site
    circuit=QuantumCircuit(QuantumRegister(n,"spin"),QuantumRegister(1,"bath"))
    circuit.reset(n)
    circuit.cx(q,n);circuit.cx(n,q);circuit.cx(q,n)  # SWAP
    circuit.reset(n)
    circuit.compose(recovery,qubits=[j for j in range(n) if j!=q]+[q,n],inplace=True)
    circuit.reset(n)
    return circuit


def boxed_sweep(blocks):
    """Box compiled updates for portable QPY, retaining their full definitions.

    Qiskit 2.5.2 cannot reload a directly serialized Isometry constructor.
    Generic instructions with elementary definitions round-trip correctly.
    """
    circuit=blocks[0].copy_empty_like(name="quantum_heat_bath_sweep")
    for site,block in enumerate(blocks):
        circuit.append(block.to_instruction(label=f"Update pixel {site}"),circuit.qubits)
    return circuit


def run(args,n):
    start=time.time();out=args.output/f"n{n}";out.mkdir(parents=True,exist_ok=True)
    image,h,sigma=image_hamiltonian(n)
    bath=build_quantum_heat_bath(h,args.b)
    report={"method":"Quantum conditional heat bath: erase one physical spin and Petz-recover it",
            "n_system":n,"n_ancilla":1,"b":args.b,"beta_paper":args.b*n,
            "sweep_definition":"One deterministic row-major pass over all spin erasure/recovery channels.",
            "image":image.tolist(),"kappa":2.,"sigma":sigma,"sigma_ddof":1,
            "assumption":"Full XXZ Hamiltonian; nonlocal recovery synthesized with exponential dense preprocessing.",
            "bath_note":"Reset ancilla is pure |0>; target temperature is encoded in recovery gates. This is an engineered channel, not a passive thermal reservoir.",
            "gibbs":observe(bath.sigma,bath,image),"trajectories":{},"circuits":[],
            "versions":{key:version(key) for key in ["numpy","scipy","qiskit"]}}
    finals={}
    cases=[(kind,0.) for kind in ["mixed","plus_x","ordered","corrupted"]]
    cases += [("plus_x",p) for p in args.noise if p>0]
    for kind,p in cases:
        rho=initial_density(image,kind);records=[]
        for step in range(args.sweeps+1):
            record=observe(rho,bath,image);record["sweep"]=step;records.append(record)
            if step<args.sweeps:
                rho=apply_bit_flips(apply_sweep(bath,rho),p) if p else apply_sweep(bath,rho)
        residual=trace_distance(apply_bit_flips(apply_sweep(bath,rho),p),rho)
        name=f"{kind}-p{p:g}"
        report["trajectories"][name]={"initial":kind,"bit_flip_probability":p,"records":records,"final_sweep_residual":residual}
        finals[name]=rho
        print(n,name,'final far',records[-1]['far_aligned'],'D',records[-1]['trace_distance_to_gibbs'],'residual',residual,flush=True)
    compiled_sweep=QuantumCircuit(QuantumRegister(n,"spin"),QuantumRegister(1,"bath"),name="quantum_heat_bath_sweep")
    isometries=np.empty(n,dtype=object);kraus=np.empty(n,dtype=object);gates=np.empty(n,dtype=object)
    compiled_channels=[];compiled_blocks=[]
    for update in bath.updates:
        t=time.time()
        compiled=compile_circuit(recovery_circuit(update))
        columns=elementary_isometry_columns(compiled,n-1)
        phase=np.vdot(update.isometry,columns);phase/=abs(phase)
        error=float(np.linalg.norm(columns-phase*update.isometry))
        if error>1e-7:raise RuntimeError(f"Compiled isometry failed at site {update.site}: {error}")
        # Every initialized input column is checked, including complex superpositions.
        # Thus the Stinespring channel is checked on arbitrary inputs, not just |0>.
        from dataclasses import replace
        compiled_channels.append(replace(update,recovery=columns.reshape(2,2**n,2**(n-1))))
        block=build_elementary_update(compiled,n,update.site)
        compiled_blocks.append(block)
        compiled_sweep.compose(block,inplace=True)
        isometries[update.site]=update.isometry
        kraus[update.site]=np.stack(full_kraus(update),axis=2)
        # Small-system MATLAB/QCLAB replay includes actual elementary recovery gates.
        gates[update.site]=gate_array(compiled) if n<=4 else np.empty((0,6))
        report["circuits"].append({"site":update.site,"compiled_recovery_operations":dict(compiled.count_ops()),
            "compiled_recovery_depth":compiled.depth(),"all_input_columns_checked":2**(n-1),
            "compiled_isometry_error_up_to_global_phase":error,
            "isometry_error":update.isometry_error,"gibbs_fixed_point_error":update.gibbs_error,
            "thermal_amplitude_condition_number":float(update.singular_values[0]/update.singular_values[-1]),
            "seconds":time.time()-t})
        print(n,'compiled+verified site',update.site,'CX',compiled.count_ops().get('cx',0),'isometry error',error,'seconds',time.time()-t,flush=True)
    # Channel reconstructed exclusively from all simulated elementary-gate columns.
    probes=[bath.sigma,initial_density(image,'plus_x'),initial_density(image,'mixed')]
    compiled_errors=[]
    for rho in probes:
        direct=apply_sweep(bath,rho);actual=rho.copy()
        for update in compiled_channels:actual=apply_update(update,actual)
        compiled_errors.append(trace_distance(actual,direct))
    report["compiled_sweep_channel_errors"]=compiled_errors
    if max(compiled_errors)>1e-7:raise RuntimeError('Compiled sweep channel does not match the target maps.')
    report["one_sweep_resources"]={"qubits":n+1,"operations":dict(compiled_sweep.count_ops()),
        "depth":compiled_sweep.depth(),"routing":"All-to-all, no hardware noise or device routing",
        "scope":"One complete sweep, with all recovery blocks expanded into u3/cx and all resets retained"}
    with gzip.open(out/'one-sweep.qasm.gz','wt') as handle:handle.write(qasm2.dumps(compiled_sweep))
    logical=boxed_sweep(compiled_blocks)
    with (out/'one-sweep.qpy').open('wb') as handle:qpy.dump(logical,handle)
    with (out/'one-sweep-compiled.qpy').open('wb') as handle:qpy.dump(compiled_sweep,handle)
    (out/'one-update.txt').write_text(str(update_circuit(bath.updates[0]).draw(output='text',fold=-1)))
    if n<=4:
        # Independent Qiskit density-matrix execution of the entire compiled circuit.
        rho=initial_density(image,'plus_x')
        result=ensemble_circuit_density(compiled_sweep,np.kron(np.diag([1.,0]),rho))
        report['qiskit_full_sweep_density_error']=trace_distance(partial_trace(result,[n]).data,apply_sweep(bath,rho))
        shot_circuit=QuantumCircuit(n+1,n)
        shot_circuit.h(range(n));shot_circuit.compose(compiled_sweep,inplace=True)
        shot_circuit.measure(range(n),range(n))
        counts=BasicSimulator().run(shot_circuit,shots=args.shots,seed_simulator=1200).result().get_counts()
        frequencies=np.array([counts.get(f'{i:0{n}b}',0) for i in range(2**n)])/args.shots
        z=1-2*((np.arange(2**n)[:,None]>>np.arange(n-1,-1,-1))&1)
        measured=(frequencies*z[:,0])@z/4
        target=np.asarray(observe(apply_sweep(bath,rho),bath,image)['gamma'])[0].ravel()
        radius=float(np.sqrt(np.log(2*n/.05)/(8*args.shots)))
        report['shot_check']={'shots':args.shots,'counts':counts,'gamma':measured.tolist(),'gamma_exact':target.tolist(),
            'simultaneous_95_hoeffding_radius':radius,'max_gamma_error':float(abs(measured-target).max()),
            'scope':'Actual ideal BasicSimulator execution, one sweep from all +x; not density-matrix resampling or hardware.'}
        if np.max(abs(measured-target))>radius:raise RuntimeError('Shot validation outside simultaneous 95% bound.')
        (out/'plus-x-one-sweep.qasm').write_text(qasm2.dumps(shot_circuit))
    np.savez_compressed(out/'final-states.npz',rho_gibbs=bath.sigma,**finals)
    savemat(out/'quantum-heat-bath.mat',{'image':image,'kappa':2.,'sigma':sigma,'sigma_ddof':1,
        'H':h,'n_pixels':n,'beta':args.b*n,'b':args.b,'rho_gibbs':bath.sigma,
        'sweeps':args.sweeps,'rho_initial':initial_density(image,'plus_x'),'rho_final':finals['plus_x-p0'],
        'kraus':kraus,'recovery_isometries':isometries,'recovery_gates':gates,
        'site_zero_based':np.arange(n),'note':'Recovery gates: [opcode(1=u3,2=cx),qubit,target/-1,theta,phi,lambda]. Recovery input order B low bits, erased site next, bath highest; top two input qubits zero.'},do_compression=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=update_circuit(bath.updates[0]).draw(output='mpl',fold=-1,style={'name':'bw'})
    fig.suptitle(f'One quantum heat-bath update: {n} XXZ spins + 1 reusable reset ancilla\nRecovery box is compiled in the supplied QASM; it is not one elementary gate.',fontsize=11)
    fig.savefig(out/'one-update.png',dpi=170,bbox_inches='tight');fig.savefig(out/'one-update.svg',bbox_inches='tight');plt.close(fig)
    report['runtime_seconds']=time.time()-start
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',n,'resources',report['one_sweep_resources'],'seconds',time.time()-start,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spins',type=int,nargs='+',choices=[4,8],default=[4,8])
    p.add_argument('--b',type=float,default=6.)
    p.add_argument('--sweeps',type=int,default=80)
    p.add_argument('--noise',type=float,nargs='*',default=[.01,.05])
    p.add_argument('--shots',type=int,default=1024)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'quantum-glauber')
    args=p.parse_args()
    if args.sweeps<1 or args.shots<1 or any(not np.isfinite(x) or x<0 or x>1 for x in args.noise):p.error('Invalid sweep, shot count, or noise probability.')
    for n in args.spins:run(args,n)


if __name__=='__main__':main()
