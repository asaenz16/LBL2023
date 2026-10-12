"""Finite Hamiltonian/bath stages, with exact small-system generators.

The bath is the existing sum of edge-local Davies dissipators. For eight
spins we exponentiate that sum directly, avoiding spatial Trotter error.
The optional invariant zero magnetization-difference operator sector is
verified before use. Full-space mode preserves coherences in every sector.
"""
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from ladder_tensor import X, Y, Z, dense_row_hamiltonian, rung_to_row_indices
from thermal_bath import build_bath, davies_generator
from bath_modifications import column_to_row_superoperator


def spin_vector(matrix, n):
    order=tuple(k for j in range(n) for k in (j,j+n))
    return np.asarray(matrix).reshape((2,)*(2*n)).transpose(order).reshape(-1)


def spin_matrix(vector, n):
    order=tuple(range(0,2*n,2))+tuple(range(1,2*n,2))
    return np.asarray(vector).reshape((2,)*(2*n)).transpose(order).reshape(2**n,2**n)


def embed_pair_generator(local, sites, n):
    """Embed a row-vectorized two-spin superoperator in spin Liouville order."""
    if len(sites)!=2 or len(set(sites))!=2 or min(sites)<0 or max(sites)>=n:
        raise ValueError('Require two distinct spin sites.')
    order=np.arange(16).reshape(2,2,2,2).transpose(0,2,1,3).reshape(-1)
    local=local[np.ix_(order,order)]
    rr,cc=np.where(abs(local)>1e-13)
    others=[j for j in range(n) if j not in sites]
    digits=np.array(np.unravel_index(np.arange(4**(n-2)),(4,)*(n-2))) if n>2 else np.empty((0,1),int)
    base=np.zeros(4**(n-2),dtype=np.int64)
    for j,d in zip(others,digits):base+=d*4**(n-1-j)
    weights=[4**(n-1-j) for j in sites]
    rows=base[None,:]+(rr//4*weights[0]+rr%4*weights[1])[:,None]
    cols=base[None,:]+(cc//4*weights[0]+cc%4*weights[1])[:,None]
    values=np.broadcast_to(local[rr,cc,None],rows.shape)
    return sparse.coo_matrix((values.ravel(),(rows.ravel(),cols.ravel())),shape=(4**n,4**n)).tocsr()


@lru_cache(maxsize=16)
def local_generators(delta,b):
    h=-.25*(np.kron(X,X)+np.kron(Y,Y)+delta*np.kron(Z,Z))
    bath=column_to_row_superoperator(davies_generator(build_bath(h,beta=2*b)),4)
    coherent=-1j*(np.kron(h,np.eye(4))-np.kron(np.eye(4),h.T))
    return np.real_if_close(bath),coherent


class ExactEdgeDynamics:
    """Untruncated density evolution, optionally in an exact invariant sector.

    Use sector='all' for coherent initial states such as |+><+| tensor n.
    The default zero charge sector is appropriate for the maximally mixed start.
    """
    def __init__(self,ladder,b=6.,sector='zero'):
        if ladder.n_spins>8:raise ValueError('Exact reference is limited to eight spins.')
        if sector not in ['zero','all']:raise ValueError('Use zero or all operator sectors.')
        self.operator_sector=sector
        self.width=ladder.width;self.n=ladder.n_spins;self.d=2**self.n
        perm=rung_to_row_indices(self.width)
        self.h=dense_row_hamiltonian(ladder)[np.ix_(perm,perm)]
        self.evals,self.evecs=np.linalg.eigh(self.h)
        edges=[(2*j,2*j+1,2.) for j in range(self.width)]
        edges.extend((2*j+r,2*(j+1)+r,float(ladder.horizontal_delta[j]))
                     for j in range(self.width-1) for r in range(2))
        full_d=sparse.csr_matrix((4**self.n,)*2,dtype=float)
        full_c=sparse.csr_matrix((4**self.n,)*2,dtype=complex)
        for i,j,delta in edges:
            diss,coh=local_generators(delta,b)
            full_d+=embed_pair_generator(diss,(i,j),self.n)
            full_c+=embed_pair_generator(coh,(i,j),self.n)
        indices=np.arange(4**self.n);charge=np.zeros(len(indices),int)
        for j in range(self.n):charge+=np.array([0,-1,1,0])[(indices//4**j)%4]
        self.zero_charge_indices=np.flatnonzero(charge==0)
        self.sector=self.zero_charge_indices if sector=='zero' else indices
        outside=np.flatnonzero(charge!=0) if sector=='zero' else np.empty(0,int)
        self.outside=outside
        self.sector_leakage=0.
        for generator in [full_d,full_c]:
            leak=generator[outside][:,self.sector]
            self.sector_leakage=max(self.sector_leakage,float(np.max(abs(leak.data),initial=0)))
        if self.sector_leakage>1e-11:raise RuntimeError('Magnetization-difference sector is not invariant.')
        self.diss=full_d[self.sector][:,self.sector].tocsr()
        self.coh=full_c[self.sector][:,self.sector].tocsr()
        self.identity=self.pack(np.eye(self.d)/self.d)

    def pack(self,matrix):
        v=spin_vector(matrix,self.n)
        if np.linalg.norm(v[self.outside])>1e-11:
            raise ValueError('State is outside the invariant operator sector.')
        return v[self.sector]

    def unpack(self,vector):
        out=np.zeros(4**self.n,dtype=vector.dtype);out[self.sector]=vector
        return spin_matrix(out,self.n)

    def unitary(self,duration):
        return (self.evecs*np.exp(-1j*duration*self.evals))@self.evecs.conj().T

    def h_stage(self,vector,unitary):
        matrix=self.unpack(vector)
        return self.pack(unitary@matrix@unitary.conj().T)

    def bath_stage(self,vector,duration):
        generator=duration*self.diss
        return expm_multiply(generator,vector,traceA=generator.diagonal().sum())

    def simultaneous_stage(self,vector,t_h,t_b):
        generator=t_h*self.coh+t_b*self.diss
        return expm_multiply(generator,vector,traceA=generator.diagonal().sum())

    def cycle(self,vector,t_h,t_b,protocol='sequential',unitary=None):
        if min(t_h,t_b)<0 or not np.isfinite(t_h+t_b) or t_h+t_b==0:
            raise ValueError('Require finite nonnegative stage durations, at least one positive.')
        if protocol=='simultaneous':
            return None,self.simultaneous_stage(vector,t_h,t_b)
        if protocol!='sequential':raise ValueError('Unknown protocol.')
        after_h=self.h_stage(vector,self.unitary(t_h) if unitary is None else unitary)
        return after_h,self.bath_stage(after_h,t_b)
