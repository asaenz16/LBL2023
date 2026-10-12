"""Density-MPS simulation of bounded star channels on two-row image strips.

MPS sites are physical spins in rung order (top,bottom per column), each
with interleaved ket/bra dimension four. Compression can violate positivity;
use exact small-system comparisons, bond refinement, and marginal diagnostics.
"""
import numpy as np
import quimb.tensor as qtn
from local_petz import local_superoperator
from ladder_bath import density_profile, density_reduced, density_energy


def spin_order(model):
    rows,width=model.graph.image.shape[:2]
    if rows!=2:raise ValueError('The tensor benchmark is for two-row strips.')
    return [r*width+c for c in range(width) for r in range(2)]


def tensor_plan(model):
    order=spin_order(model);inverse=np.argsort(order);plan=[]
    for i in model.order:
        update=model.updates[i];where=[int(inverse[p]) for p in update.sites]
        sort=np.argsort(where);where=tuple(sorted(where));k=len(where)
        gate=local_superoperator(update).reshape((4,)*(2*k))
        gate=gate.transpose(tuple(sort)+tuple(sort+k)).reshape(4**k,4**k)
        mpo=qtn.MatrixProductOperator.from_dense(gate,dims=[4]*k,sites=where,L=model.graph.n,
                                               cutoff=1e-14,cutoff_mode='abs')
        error=np.linalg.norm(mpo.to_dense()-gate)
        if error>1e-10:raise RuntimeError('Local channel MPO factorization failed.')
        plan.append((where,mpo,float(error)))
    return plan


def initial_density_mps(model,kind):
    order=spin_order(model);n=model.graph.n
    if kind=='plus_x':return qtn.MPS_product_state([np.ones(4)/2 for _ in range(n)])
    if kind=='mixed':return qtn.MPS_product_state([np.array([1.,0,0,1])/2 for _ in range(n)])
    if kind!='ordered':raise ValueError('Use plus_x, mixed, or ordered.')
    truth=np.where(model.graph.image.ravel()[order]==0,1,-1)
    def product(sign):
        return qtn.MPS_product_state([np.array([1.,0,0,0]) if sign*z==1 else np.array([0.,0,0,1]) for z in truth])
    state=product(1).add_MPS(product(-1));state.multiply_(.5)
    return state


def as_rung_density(state):
    """Regroup neighboring spin density sites for existing observable helpers."""
    copy=state.copy();copy.permute_arrays('lrp')
    # Sub-MPO contraction can reorder the internal tensor dictionary.
    arrays=[copy[i].data for i in range(copy.L)]
    arrays[0]=arrays[0][None];arrays[-1]=arrays[-1][:,None]
    combined=[]
    for i in range(0,len(arrays),2):
        a,b=arrays[i:i+2]
        tensor=np.einsum('aip,ibq->abpq',a,b).reshape(a.shape[0],b.shape[1],2,2,2,2)
        tensor=tensor.transpose(0,1,2,4,3,5).reshape(a.shape[0],b.shape[1],16)
        combined.append(tensor)
    combined[0]=combined[0][0];combined[-1]=combined[-1][:,0]
    return qtn.MatrixProductState(combined,shape='lrp')


def apply_tensor_sweep(state,plan,max_bond=32,cutoff=1e-12):
    for where,mpo,_ in plan:
        state.gate_with_submpo_(mpo,where=where,method='direct',max_bond=max_bond,
                               cutoff=cutoff,cutoff_mode='abs')
    trace=density_profile(as_rung_density(state))['trace']
    if abs(trace)<1e-12:raise RuntimeError('Density tensor lost its trace.')
    state.multiply_(np.real_if_close(1/trace).item())
    return float(abs(trace-1))


def tensor_observe(state,ladder):
    rho=as_rung_density(state)
    profiles=[density_profile(rho,seed=(0,c)) for c in [0,1]]
    gamma=np.stack([p['gamma'] for p in profiles])
    diagnostics=[]
    for count in (1,2):
        for start in range(ladder.width-count+1):
            marginal=density_reduced(rho,start,count)
            diagnostics.append((np.linalg.norm(marginal-marginal.conj().T),
                                np.linalg.eigvalsh((marginal+marginal.conj().T)/2).min()))
    truth=np.where(ladder.image==0,1,-1)
    return {'gamma':gamma.tolist(),'connected':np.stack([p['connected'] for p in profiles]).tolist(),
            'means':profiles[0]['means'].tolist(),'far_aligned':float(gamma[0,-1,-1]*truth[0,0]*truth[-1,-1]),
            'energy':density_energy(rho,ladder),'max_bond':state.max_bond(),
            'trace_error':float(abs(profiles[0]['trace']-1)),
            'max_marginal_hermiticity_error':float(max(x[0] for x in diagnostics)),
            'min_one_or_two_rung_eigenvalue':float(min(x[1] for x in diagnostics))}
