"""Positive pure-state MPS trajectories for local Petz channels on strips.

The four Kraus outcomes are sampled with their Born probabilities. Their
ensemble is exactly the channel before MPS truncation. No trajectory is
postselected or discarded. Finite-ensemble error and MPS error are separate.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import svd
import quimb.tensor as qtn
from local_petz_tensor import spin_order
from ladder_tensor import seed_profile


def lift_operator(operator,sites,n):
    """Embed a small operator in a bounded contiguous MPS window."""
    sites=tuple(sites);k=len(sites);dim=2**n
    rows=np.arange(dim);shifts=np.array([n-1-i for i in sites])
    local=((rows[:,None]>>shifts)&1)@(1<<np.arange(k-1,-1,-1))
    mask=sum(1<<int(s) for s in shifts);cleared=rows&~mask
    output=np.zeros((dim,dim),complex)
    for dest in range(2**k):
        new=cleared.copy()
        for j,shift in enumerate(shifts):new|=((dest>>(k-1-j))&1)<<int(shift)
        output[new,rows]=operator[dest,local]
    return np.real_if_close(output)


def trajectory_plan(model):
    order=spin_order(model);inverse=np.argsort(order);plan=[]
    for i in model.order:
        u=model.updates[i];where=[int(inverse[p]) for p in u.sites]
        start,end=min(where),max(where)
        kraus=tuple(lift_operator(k,[p-start for p in where],end-start+1) for k in u.kraus)
        if np.linalg.norm(sum(k.conj().T@k for k in kraus)-np.eye(2**(end-start+1)))>1e-10:
            raise RuntimeError('Lifted channel is not trace preserving.')
        plan.append((start,end,kraus))
    return plan


@dataclass
class Trajectory:
    state: object
    rng: object
    info: dict
    discarded_weight: float = 0.
    max_probability_sum_error: float = 0.
    svd_fallbacks: int = 0


def initial_trajectory(model,kind,seed):
    rng=np.random.default_rng(seed);order=spin_order(model)
    if kind=='plus_x':vectors=[np.ones(2)/np.sqrt(2) for _ in order]
    elif kind=='ordered':
        sign=rng.choice([-1,1]);truth=np.where(model.graph.image.ravel()[order]==0,1,-1)*sign
        vectors=[np.eye(2)[int(z<0)] for z in truth]
    elif kind=='mixed':vectors=[np.eye(2)[int(rng.integers(2))] for _ in order]
    else:raise ValueError('Use plus_x, ordered, or mixed.')
    return Trajectory(qtn.MPS_product_state(vectors),rng,{'cur_orthog':(0,0)})


def apply_trajectory_update(trajectory,item,max_bond=32,cutoff=1e-12):
    start,end,kraus=item;state=trajectory.state
    state.canonicalize_((start,end),info=trajectory.info)
    state.permute_arrays('lrp')
    arrays=[]
    for i in range(start,end+1):
        a=state[i].data
        if i==0:a=a[None]
        if i==state.L-1:a=a[:,None]
        arrays.append(a.transpose(0,2,1))
    theta=arrays[0]
    for a in arrays[1:]:theta=np.tensordot(theta,a,axes=([-1],[0]))
    theta=theta.reshape(theta.shape[0],2**(end-start+1),theta.shape[-1])
    candidates=[np.einsum('ab,lbr->lar',k,theta,optimize=True) for k in kraus]
    probabilities=np.array([np.vdot(x,x).real for x in candidates])
    total=probabilities.sum()
    trajectory.max_probability_sum_error=max(trajectory.max_probability_sum_error,float(abs(total-1)))
    if not np.isfinite(total) or total<=0:raise RuntimeError('Invalid Kraus probabilities.')
    outcome=int(trajectory.rng.choice(len(kraus),p=probabilities/total))
    theta=candidates[outcome]/np.sqrt(probabilities[outcome])
    left=theta.shape[0];right=theta.shape[-1]
    for i in range(start,end):
        matrix=theta.reshape(left*2,-1)
        try:
            u,s,vh=svd(matrix,full_matrices=False,lapack_driver='gesdd')
        except np.linalg.LinAlgError:
            # Nearly rank-deficient post-measurement tensors can defeat the
            # divide-and-conquer driver. Retry the same finite matrix with QR SVD.
            u,s,vh=svd(matrix,full_matrices=False,lapack_driver='gesvd')
            trajectory.svd_fallbacks+=1
        keep=max(1,min(max_bond,int(np.count_nonzero(s>cutoff))))
        trajectory.discarded_weight+=float(np.sum(s[keep:]**2))
        a=u[:,:keep].reshape(left,2,keep).transpose(0,2,1)
        state[i].modify(data=a[0] if i==0 else a)
        theta=s[:keep,None]*vh[:keep];left=keep
    a=theta.reshape(left,2,right).transpose(0,2,1)
    a/=np.linalg.norm(a)  # The outside tensors and new left tensors are isometries.
    state[end].modify(data=a[0,0] if state.L==1 else a[:,0] if end==state.L-1 else a)
    trajectory.info['cur_orthog']=(end,end)
    return outcome


def apply_trajectory_sweep(trajectory,plan,max_bond=32,cutoff=1e-12):
    for item in plan:apply_trajectory_update(trajectory,item,max_bond,cutoff)


def as_rung_state(state):
    state=state.copy();state.permute_arrays('lrp')
    arrays=[state[i].data for i in range(state.L)]
    arrays[0]=arrays[0][None];arrays[-1]=arrays[-1][:,None]
    result=[]
    for i in range(0,state.L,2):
        a,b=arrays[i:i+2]
        result.append(np.einsum('aip,ibq->abpq',a,b).reshape(a.shape[0],b.shape[1],4))
    result[0]=result[0][0];result[-1]=result[-1][:,0]
    return qtn.MatrixProductState(result,shape='lrp')


def trajectory_observe(trajectory):
    state=as_rung_state(trajectory.state)
    profiles=[seed_profile(state,(0,c)) for c in [0,state.L//4]]
    return np.stack([p['gamma'] for p in profiles]),profiles[0]['means']
