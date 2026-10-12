"""Overlapping four-spin baths and coherent evolution for image ladders.

Both are modifications of the local-bond model, not full-H Gibbs samplers.
Each plaquette uses its complete internal H. Pauli proposal weights match
the total per-spin proposal weight in the original bond model.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

from ladder_tensor import I4, X, Y, Z, OPS
from ladder_bath import rung_gate, horizontal_gate, rung_interleave_superoperator, density_profile
from thermal_bath import build_bath, davies_generator


def plaquette_hamiltonian(delta):
    rung=-.25*(np.kron(X,X)+np.kron(Y,Y)+2*np.kron(Z,Z))
    horizontal=sum(-.25*coefficient*np.kron(OPS[axis][row],OPS[axis][row])
                   for row in range(2) for axis,coefficient in zip('XYZ',[1,1,delta]))
    return np.real_if_close(np.kron(rung,I4)+np.kron(I4,rung)+horizontal)


@lru_cache(maxsize=32)
def plaquette_bath(delta,b):
    return build_bath(plaquette_hamiltonian(delta),beta=4*b)


@lru_cache(maxsize=64)
def plaquette_generator(delta,b,left_weight,right_weight):
    bath=plaquette_bath(delta,b)
    # Grouping by the original proposal keeps equal-frequency degenerate blocks.
    result=np.zeros((256,256),complex)
    identity=np.eye(16)
    for contact in bath.contacts:
        weight=left_weight if int(contact.label[1:])<2 else right_weight
        a,p=contact.lowering,contact.excited_probability
        jumps=[a] if contact.omega==0 else [np.sqrt(2*(1-p))*a,np.sqrt(2*p)*a.conj().T]
        for jump in jumps:
            decay=jump.conj().T@jump
            result+=weight*(np.kron(jump.conj(),jump)-.5*np.kron(identity,decay)-.5*np.kron(decay.T,identity))
    return np.real_if_close(result)


def column_to_row_superoperator(matrix,d):
    order=np.arange(d*d).reshape(d,d).T.reshape(-1)
    return matrix[np.ix_(order,order)]


@lru_cache(maxsize=128)
def plaquette_gate(delta,b,duration,left_weight,right_weight):
    channel=expm(duration*plaquette_generator(delta,b,left_weight,right_weight))
    row=column_to_row_superoperator(channel,16)
    return np.real_if_close(rung_interleave_superoperator(row))


def proposal_weights(width):
    degree=np.full(width,3.);degree[[0,-1]]=2.
    coverage=np.full(width,2.);coverage[[0,-1]]=1.
    return degree/coverage


def bath_plan(ladder,model,b,duration,plaquette_weighting='degree'):
    L=ladder.width;plan=[]
    if model=='bond':
        plan.extend([((j,),rung_gate(b,duration/2)) for j in range(L)])
    elif model!='plaquette':
        raise ValueError('Use bond or plaquette.')
    if plaquette_weighting not in ['degree','uniform']:raise ValueError('Unknown plaquette weighting.')
    weights=proposal_weights(L) if plaquette_weighting=='degree' else np.ones(L)
    for parity,factor in [(0,.5),(1,1.),(0,.5)]:
        for j in range(parity,L-1,2):
            delta=float(ladder.horizontal_delta[j])
            gate=(horizontal_gate(delta,b,duration*factor) if model=='bond' else
                  plaquette_gate(delta,b,duration*factor,float(weights[j]),float(weights[j+1])))
            plan.append(((j,j+1),gate))
    if model=='bond':plan.extend([((j,),rung_gate(b,duration/2)) for j in range(L)])
    return plan


def unitary_superoperator(h,duration,two_rungs=False):
    u=expm(-1j*duration*h)
    channel=np.kron(u,u.conj())
    return rung_interleave_superoperator(channel) if two_rungs else channel


def coherent_plan(ladder,duration):
    """Second-order spatial split of exp(-i H duration)."""
    plan=[((j,),unitary_superoperator(ladder.h_rung,duration/2)) for j in range(ladder.width)]
    for parity,factor in [(0,.5),(1,1.),(0,.5)]:
        for j in range(parity,ladder.width-1,2):
            plan.append(((j,j+1),unitary_superoperator(ladder.h_bonds[j,j+1],duration*factor,True)))
    plan.extend([((j,),unitary_superoperator(ladder.h_rung,duration/2)) for j in range(ladder.width)])
    return plan


def make_plan(ladder,model='bond',b=6.,dt=.1,eta=1.,coherent=False,integrator='fused',plaquette_weighting='degree'):
    if dt<=0 or eta<=0:raise ValueError('dt and eta must be positive.')
    bath=bath_plan(ladder,model,b,eta*dt,plaquette_weighting)
    if not coherent:return bath
    if integrator=='fused' and model=='bond':
        # Each bond Davies dissipator commutes with its own coherent generator.
        # Compose the exact local channels, then spatially Strang-split them.
        coherent_layers=coherent_plan(ladder,dt)
        if [s for s,_ in coherent_layers]!=[s for s,_ in bath]:
            raise RuntimeError('Coherent and dissipative bond layouts differ.')
        return [(sites,u@g) for (sites,u),(_,g) in zip(coherent_layers,bath)]
    if integrator not in ['fused','separate']:raise ValueError('Unknown integrator.')
    half=coherent_plan(ladder,dt/2)
    return half+bath+half


def apply_mps_plan(rho,plan,bond,cutoff=1e-12):
    info={}
    for sites,gate in plan:
        if len(sites)==1:
            rho.gate_(gate,sites[0],contract=True)
            info.clear()  # General dissipative gates need not preserve canonical form.
        else:
            rho.gate_with_auto_swap_(gate,sites,info=info,max_bond=bond,cutoff=cutoff,cutoff_mode='rsum2')
    trace=density_profile(rho)['trace']
    rho.multiply_(np.real_if_close(1/trace).item())
    return abs(trace-1)


def dense_vector(rho,width):
    order=tuple(j for col in range(width) for j in (col,col+width))
    return np.asarray(rho).reshape((4,)*(2*width)).transpose(order).reshape(-1)


def dense_matrix(vector,width):
    order=tuple(range(0,2*width,2))+tuple(range(1,2*width,2))
    return vector.reshape((4,)*(2*width)).transpose(order).reshape(4**width,4**width)


def apply_dense_plan(vector,plan,width):
    """Apply local superoperators without constructing the global channel."""
    for sites,gate in plan:
        order=sites+tuple(j for j in range(width) if j not in sites)
        block=vector.reshape((16,)*width).transpose(order).reshape(16**len(sites),-1)
        block=gate@block
        vector=block.reshape((16,)*width).transpose(np.argsort(order)).reshape(-1)
    return vector
