"""Local Petz circuits: exact benchmarks, gate scaling, and MPS trajectories."""
import os
import tempfile
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np
from scipy.io import savemat
from qiskit import qasm2,qpy
from qiskit.quantum_info import Operator,partial_trace
from local_petz import (image_graph,build_local_petz,apply_dense_sweep,image_circuit,
                        patch_circuit,route_locally)
from quantum_glauber import (build_quantum_heat_bath,ensemble_circuit_density,apply_bit_flips,
                            recovery_circuit,compile_circuit,elementary_isometry_columns)
from run_quantum_glauber import initial_density,observe
from ladder_tensor import build_ladder,dense_row_hamiltonian
from thermal_bath import trace_distance


def write_json(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def dense(args):
    out=args.output/'dense';out.mkdir(parents=True,exist_ok=True)
    ladder=build_ladder(4);graph=image_graph(ladder.image)
    target=build_quantum_heat_bath(dense_row_hamiltonian(ladder),args.b)
    report={'b':args.b,'n_spins':8,'image':graph.image.tolist(),'sigma':graph.sigma,
            'gibbs':observe(target.sigma,target,graph.image),'models':{}}
    for mode in ['star','strip3']:
        model=build_local_petz(graph,args.b,mode);cases={};finals={}
        for kind,p in [('plus_x',0.),('mixed',0.),('ordered',0.),('plus_x',.01),('plus_x',.05)]:
            rho=initial_density(graph.image,kind);records=[]
            for s in range(args.sweeps+1):
                record=observe(rho,target,graph.image);record['sweep']=s;records.append(record)
                if s<args.sweeps:rho=apply_bit_flips(apply_dense_sweep(rho,model),p)
            name=f'{kind}-p{p:g}';finals[name]=rho
            cases[name]={'records':records,'residual':trace_distance(apply_bit_flips(apply_dense_sweep(rho,model),p),rho)}
            print(mode,name,'far',records[-1]['far_aligned'],'D',records[-1]['trace_distance_to_gibbs'],'residual',cases[name]['residual'],flush=True)
        # Ordered versus random-order schedules need not have the same fixed point.
        row=initial_density(graph.image,'plus_x')
        for _ in range(args.sweeps):row=apply_dense_sweep(row,model,order=range(8))
        report['models'][mode]={'patch_sizes':[len(u.sites) for u in model.updates],
            'layers':model.layers,'gibbs_one_sweep_defect':trace_distance(apply_dense_sweep(target.sigma,model),target.sigma),
            'cases':cases,'row_major_order_control':observe(row,target,graph.image),
            'row_major_vs_colored_trace_distance':trace_distance(row,finals['plus_x-p0'])}
        np.savez_compressed(out/f'{mode}-states.npz',rho_gibbs=target.sigma,**finals)
        sites=np.empty((1,8),object);kraus=np.empty((1,8),object)
        for j,u in enumerate(model.updates):sites[0,j]=np.array(u.sites);kraus[0,j]=np.stack(u.kraus,axis=2)
        savemat(out/f'{mode}-fixture.mat',{'image':graph.image,'H':target.hamiltonian,'sigma':graph.sigma,
            'kappa':graph.kappa,'b':args.b,'beta':args.b*8,'n_pixels':8,'sigma_ddof':1,
            'sites_zero_based':sites,'kraus':kraus,'order_zero_based':np.array(model.order),
            'rho_initial':initial_density(graph.image,'plus_x'),'rho_final':finals['plus_x-p0'],
            'rho_gibbs':target.sigma,'sweeps':args.sweeps},do_compression=True)
    write_json(out/'report.json',report)


def resources(args):
    out=args.output/'circuits';out.mkdir(parents=True,exist_ok=True);report=[]
    images=[('strip8',build_ladder(4).image,'star'),('strip16',build_ladder(8).image,'star'),
            ('strip32',build_ladder(16).image,'star'),('strip8-wide',build_ladder(4).image,'strip3')]
    rr,cc=np.indices((8,8));images.append(('square64',255*((rr//2+cc//2)%2),'star'))
    for name,image,mode in images:
        model=build_local_petz(image_graph(image),args.b,mode);result={'name':name,'mode':mode,
            'b':args.b,'image':model.graph.image.tolist(),'sigma':model.graph.sigma,
            'kappa':model.graph.kappa,'sigma_ddof':model.graph.sigma_ddof,
            'n_spins':model.graph.n,'n_ancillas':model.graph.n,'layers':model.layers,
            'max_patch_spins':max(len(u.sites) for u in model.updates),'checks':[]}
        # Validate distinct local circuits. Full unitary routing equivalence
        # proves correctness on every input, including entangled exteriors.
        signatures=set()
        for u in model.updates:
            key=(u.edges,u.delta,u.local.site)
            if key in signatures:continue
            signatures.add(key)
            columns=elementary_isometry_columns(compile_circuit(recovery_circuit(u.local)),len(u.sites)-1)
            phase=np.vdot(u.local.isometry,columns);phase/=abs(phase)
            isometry_error=float(np.linalg.norm(columns-phase*u.local.isometry))
            if isometry_error>1e-8:raise RuntimeError('Compiled recovery isometry differs from patch channel.')
            original=patch_circuit(model,u,False);routed=patch_circuit(model,u,True)
            # Remove resets to compare complete unitary circuits. Same reset
            # locations also require each between-reset segment to agree.
            def segments(c):
                blocks=[];block=c.copy_empty_like()
                for ins in c.data:
                    if ins.operation.name=='reset':blocks.append(block);block=c.copy_empty_like()
                    else:block.append(ins.operation,ins.qubits)
                blocks.append(block);return blocks
            errors=[float(np.linalg.norm(Operator(a).data-Operator(b).data))
                    for a,b in zip(segments(original),segments(routed))]
            if max(errors)>1e-9:raise RuntimeError('Routing changed the circuit channel.')
            # Independent density execution of the circuit versus patch Kraus map.
            d=2**len(u.sites);rho=np.ones((d,d))/d
            actual=partial_trace(ensemble_circuit_density(routed,np.kron(np.diag([1.,0]),rho)),[len(u.sites)]).data
            expected=sum(k@rho@k.conj().T for k in u.kraus)
            error=trace_distance(actual,expected)
            if error>1e-8:raise RuntimeError('Local circuit differs from the patch map.')
            result['checks'].append({'center':u.center,'patch_spins':len(u.sites),
                'all_initialized_columns_isometry_error':isometry_error,
                'routing_segment_operator_error':max(errors),'circuit_channel_error':error})
        for routed in [False,True]:
            circuit=image_circuit(model,routed=routed,parallel=True)
            key='routed' if routed else 'unrouted'
            result[key]={'operations':dict(circuit.count_ops()),'depth':circuit.depth(),
                         'qubits':circuit.num_qubits}
            if routed:
                with gzip.open(out/f'{name}.qasm.gz','wt') as f:f.write(qasm2.dumps(circuit))
                with (out/f'{name}.qpy').open('wb') as f:qpy.dump(circuit,f)
                with (out/f'{name}.qpy').open('rb') as f:restored=qpy.load(f)[0]
                if restored.count_ops()!=circuit.count_ops():raise RuntimeError('QPY export mismatch.')
                with gzip.open(out/f'{name}.qasm.gz','rt') as f:parsed=qasm2.loads(f.read())
                if parsed.count_ops()!=circuit.count_ops():raise RuntimeError('QASM export mismatch.')
                error=0.
                for a,c in zip(circuit.data,parsed.data):
                    if a.operation.name!=c.operation.name or [circuit.find_bit(q).index for q in a.qubits]!=[parsed.find_bit(q).index for q in c.qubits]:
                        raise RuntimeError('QASM changed an operation or qubit.')
                    if a.operation.params:error=max(error,max(abs(float(x)-float(y)) for x,y in zip(a.operation.params,c.operation.params)))
                if error>1e-10:raise RuntimeError('Excessive QASM angle rounding.')
                result['qasm_all_gate_angle_error']=error
        report.append(result);write_json(out/'report.json',report)
        print('RESOURCE',name,result['unrouted'],result['routed'],flush=True)


def trajectories(args):
    from local_petz_trajectories import (trajectory_plan,initial_trajectory,apply_trajectory_sweep,trajectory_observe)
    out=args.output/'trajectories';out.mkdir(parents=True,exist_ok=True)
    checkpoints=sorted(set([0,1,2,4,8,12,20,*range(25,args.sweeps+1,5),args.sweeps]))
    checkpoints=[t for t in checkpoints if t<=args.sweeps]
    for width in args.widths:
        model=build_local_petz(image_graph(build_ladder(width).image),args.b,'star');plan=trajectory_plan(model)
        for kind in args.initial:
            t0=time.time();gammas=[];means=[];diagnostics=[]
            name=f'L{width}-{kind}-chi{args.bond}-r{args.replicas}'
            for replica in range(args.replicas):
                trajectory=initial_trajectory(model,kind,args.seed+replica)
                rg=[];rm=[]
                for sweep in range(args.sweeps+1):
                    if sweep in checkpoints:
                        g,m=trajectory_observe(trajectory);rg.append(g);rm.append(m)
                    if sweep<args.sweeps:apply_trajectory_sweep(trajectory,plan,args.bond,args.cutoff)
                gammas.append(rg);means.append(rm)
                diagnostics.append([trajectory.discarded_weight,trajectory.max_probability_sum_error,trajectory.state.max_bond(),trajectory.svd_fallbacks])
                if (replica+1)%16==0 or replica+1==args.replicas:
                    np.savez_compressed(out/f'{name}.npz',gamma=np.array(gammas),means=np.array(means),
                                        diagnostics=np.array(diagnostics),sweeps=checkpoints)
                    print(name,replica+1,'/',args.replicas,'elapsed',round(time.time()-t0,1),flush=True)
            gamma=np.mean(gammas,axis=0);se=np.std(gammas,axis=0,ddof=1)/np.sqrt(args.replicas)
            report={'width':width,'n_spins':2*width,'b':args.b,'sigma':model.graph.sigma,
                'mode':'star','initial':kind,'seed':args.seed,'replicas':args.replicas,'max_bond':args.bond,
                'cutoff':args.cutoff,'sweeps':checkpoints,'seeds':[0,width//4],
                'gamma':gamma.tolist(),'gamma_se':se.tolist(),'means':np.mean(means,axis=0).tolist(),
                'far_aligned':(-gamma[:,0,1,-1]).tolist(),
                'far_se':se[:,0,1,-1].tolist(),'max_discarded_weight_sum':float(np.max(np.array(diagnostics)[:,0])),
                'max_probability_sum_error':float(np.max(np.array(diagnostics)[:,1])),
                'svd_fallbacks':int(np.sum(np.array(diagnostics)[:,3])),
                'runtime_seconds':time.time()-t0,
                'uncertainty':'Independent Kraus trajectories; SE is ensemble uncertainty, not MPS error. Confidence intervals are approximate and pointwise. No quantum hardware.'}
            write_json(out/f'{name}.json',report)
            print('DONE',name,'far',report['far_aligned'][-1],'+/-',1.96*report['far_se'][-1],
                  'max discarded',report['max_discarded_weight_sum'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['dense','resources','trajectories'])
    p.add_argument('--b',type=float,default=6.)
    p.add_argument('--sweeps',type=int,default=40)
    p.add_argument('--widths',type=int,nargs='+',default=[4,8,16])
    p.add_argument('--initial',nargs='+',choices=['plus_x','ordered','mixed'],default=['plus_x'])
    p.add_argument('--replicas',type=int,default=128)
    p.add_argument('--bond',type=int,default=32)
    p.add_argument('--cutoff',type=float,default=1e-12)
    p.add_argument('--seed',type=int,default=1200)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'local-petz')
    args=p.parse_args()
    if args.sweeps<1 or args.replicas<2 or args.bond<1 or not np.isfinite(args.cutoff) or args.cutoff<0:
        p.error('Invalid sweep count, replicas, bond, or cutoff.')
    {'dense':dense,'resources':resources,'trajectories':trajectories}[args.action](args)


if __name__=='__main__':main()
