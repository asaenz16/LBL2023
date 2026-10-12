"""A scalable *local-bond* Davies model, simulated as a vectorized density MPS.

Each two-spin bath generator obeys detailed balance for that bond Hamiltonian.
The overlapping noncommuting bonds mean this is NOT an exact Gibbs sampler
for the full ladder, nor a controlled approximation to its global Davies
generator. This model distinction is essential when interpreting robustness.

Density MPS physical index = (rung ket, rung bra), dimension 16. Truncation
does not guarantee global positivity: validate small systems exactly and
increase the bond dimension before drawing conclusions on larger systems.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
import quimb.tensor as qtn

from ladder_tensor import I2,I4,X,Y,Z,OPS
from thermal_bath import build_bath,davies_generator


@lru_cache(maxsize=64)
def bond_kraus(delta, b, duration):
    h = -.25*(np.kron(X,X)+np.kron(Y,Y)+delta*np.kron(Z,Z))
    local = build_bath(h,beta=2*b)
    superop = expm(duration*davies_generator(local))
    choi = superop.reshape(4,4,4,4,order="F").transpose(0,2,1,3).reshape(16,16)
    values,vectors = np.linalg.eigh((choi+choi.conj().T)/2)
    if values.min() < -1e-10:
        raise RuntimeError("Local bath channel is not completely positive.")
    kraus = tuple(np.sqrt(value)*vectors[:,k].reshape(4,4)
                  for k,value in enumerate(values) if value > 1e-13)
    if np.linalg.norm(sum(k.conj().T@k for k in kraus)-np.eye(4)) > 1e-10:
        raise RuntimeError("Local bath Kraus operators are not trace preserving.")
    return kraus


def lift_pair(kraus, row):
    """Embed a two-spin operator on one leg of two neighboring rungs."""
    output=np.zeros((16,16),complex)
    shifts=[3-row,1-row]
    masks=[1<<shift for shift in shifts]
    for src in range(16):
        source=2*((src>>shifts[0])&1)+((src>>shifts[1])&1)
        cleared=src & ~(masks[0]|masks[1])
        for target in range(4):
            dest=cleared|((target>>1)<<shifts[0])|((target&1)<<shifts[1])
            output[dest,src]=kraus[target,source]
    return output


def rung_interleave_superoperator(superop):
    """Two-rung row-vectorized channel -> two dimension-16 MPS sites."""
    return superop.reshape((4,)*8).transpose(0,2,1,3,4,6,5,7).reshape(256,256)


@lru_cache(maxsize=64)
def rung_gate(b,duration):
    return np.real_if_close(sum(np.kron(k,k.conj()) for k in bond_kraus(2.,b,duration)))


@lru_cache(maxsize=64)
def horizontal_gate(delta,b,duration):
    local = bond_kraus(delta,b,duration)
    gates=[]
    for row in range(2):
        lifted=[lift_pair(k,row) for k in local]
        gates.append(rung_interleave_superoperator(sum(np.kron(k,k.conj()) for k in lifted)))
    return np.real_if_close(gates[1]@gates[0])


def noise_gate(axis, probability):
    gate=np.eye(16,dtype=complex)
    for op in OPS[axis]:
        gate=((1-probability)*np.eye(16)+probability*np.kron(op,op.conj()))@gate
    return np.real_if_close(gate)


def mixed_density(width):
    return qtn.MPS_product_state([np.eye(4).reshape(16)/4 for _ in range(width)])


def plus_x_density(width):
    """Pure |+> on every physical spin, without dephasing in magnetization."""
    return qtn.MPS_product_state([np.ones(16)/4 for _ in range(width)])


def density_arrays(rho):
    result=rho.copy(); result.permute_arrays("lrp")
    arrays=list(result.arrays); arrays[0]=arrays[0][None]; arrays[-1]=arrays[-1][:,None]
    return arrays


def _insert(array,operator):
    return np.einsum("abp,p->ab",array,operator.T.reshape(16))


def density_profile(rho,seed=(0,0),axis='Z'):
    if axis not in OPS:raise ValueError('Use X, Y or Z for the spin observable.')
    arrays=density_arrays(rho); L=len(arrays); row,col=seed
    matrices=[_insert(a,I4) for a in arrays]
    left=[np.ones(1,complex)]
    for m in matrices: left.append(left[-1]@m)
    right=[None]*(L+1); right[L]=np.ones(1,complex)
    for j in reversed(range(L)): right[j]=matrices[j]@right[j+1]
    trace=left[-1][0]; spin=[o/2 for o in OPS[axis]]
    means=np.empty((2,L)); gamma=np.empty((2,L))
    for j in range(L):
        for r in range(2): means[r,j]=np.real(left[j]@_insert(arrays[j],spin[r])@right[j+1]/trace)
    for r in range(2):
        gamma[r,col]=np.real(left[col]@_insert(arrays[col],spin[row]@spin[r])@right[col+1]/trace)
    marked=left[col]@_insert(arrays[col],spin[row])
    for j in range(col+1,L):
        for r in range(2): gamma[r,j]=np.real(marked@_insert(arrays[j],spin[r])@right[j+1]/trace)
        marked=marked@matrices[j]
    marked=_insert(arrays[col],spin[row])@right[col+1]
    for j in reversed(range(col)):
        for r in range(2): gamma[r,j]=np.real(left[j]@_insert(arrays[j],spin[r])@marked/trace)
        marked=matrices[j]@marked
    return {"gamma":gamma,"connected":gamma-means[row,col]*means,"means":means,
            "trace":complex(trace)}


def density_reduced(rho,start,count=1):
    arrays=density_arrays(rho); L=len(arrays)
    if count not in (1,2) or not 0 <= start <= L-count:
        raise ValueError("Only one or two adjacent rungs are supported.")
    left=np.ones(1,complex);right=np.ones(1,complex)
    for j in range(start): left=left@_insert(arrays[j],I4)
    for j in reversed(range(start+count,L)): right=_insert(arrays[j],I4)@right
    block=left
    for j in range(start,start+count): block=np.einsum('...a,abp->...pb',block,arrays[j])
    tensor=np.einsum('...a,a->...',block,right).reshape((4,)*(2*count))
    reduced=tensor.transpose(tuple(range(0,2*count,2))+tuple(range(1,2*count,2))).reshape(4**count,4**count)
    return reduced/np.trace(reduced)


def dense_to_density(rho,width,cutoff=0):
    """Input uses rung ket order, followed by rung bra order."""
    if width > 4:
        raise ValueError("Dense conversion is only for the smallest strip.")
    order=tuple(j for col in range(width) for j in (col,col+width))
    vector=np.asarray(rho).reshape((4,)*(2*width)).transpose(order).reshape(-1)
    return qtn.MatrixProductState.from_dense(vector,dims=[16]*width,cutoff=cutoff)


def density_to_dense(rho):
    if rho.L > 4:
        raise ValueError("Dense conversion is only for the smallest strip.")
    tensor=np.asarray(rho.to_dense()).reshape((4,)*(2*rho.L))
    order=tuple(range(0,2*rho.L,2))+tuple(range(1,2*rho.L,2))
    return tensor.transpose(order).reshape(4**rho.L,4**rho.L)


def density_energy(rho,ladder):
    """Energy of the full image Hamiltonian from one- and two-rung marginals."""
    value=sum(np.trace(density_reduced(rho,j,1)@ladder.h_rung) for j in range(ladder.width))
    value+=sum(np.trace(density_reduced(rho,j,2)@ladder.h_bonds[j,j+1]) for j in range(ladder.width-1))
    return float(np.real(value))


def bath_step(rho,ladder,b=6.,duration=.1,axis="Z",probability=0.,max_bond=32,cutoff=1e-10):
    """Strang composition of local dissipators, followed by local noise.

    p is probability per full step, not per gate. Repeated layers produce
    a driven channel; callers refining duration must also rescale p.
    """
    info={}
    rg=rung_gate(b,duration/2)
    for j in range(ladder.width): rho.gate_(rg,j,contract=True)
    for parity,factor in [(0,.5),(1,1.),(0,.5)]:
        for j in range(parity,ladder.width-1,2):
            gate=horizontal_gate(float(ladder.horizontal_delta[j]),b,duration*factor)
            rho.gate_with_auto_swap_(gate,(j,j+1),info=info,max_bond=max_bond,
                                     cutoff=cutoff,cutoff_mode="rsum2")
    for j in range(ladder.width): rho.gate_(rg,j,contract=True)
    if probability:
        gate=noise_gate(axis,probability)
        for j in range(ladder.width): rho.gate_(gate,j,contract=True)
    trace=density_profile(rho)['trace']
    rho.multiply_(np.real_if_close(1/trace).item())
    return abs(trace-1)
