"""Bounded-patch Petz channels for the image XXZ model.

Each channel uses the Gibbs state of the INDUCED PATCH HAMILTONIAN, not a
marginal of the global Gibbs state. Thus construction is local and bounded
in image size, but overlapping updates need not preserve global Gibbs.
Physical pixel order remains row-major. No classical spin dephasing is used.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from algorithm1 import xxz_hamiltonian
from quantum_glauber import (build_quantum_heat_bath, full_kraus, update_circuit,
                            compile_circuit)


@dataclass
class ImageGraph:
    image: np.ndarray
    edges: np.ndarray
    delta: np.ndarray
    sigma: float
    kappa: float
    sigma_ddof: int

    @property
    def n(self): return int(np.prod(self.image.shape[:2]))


@dataclass
class PatchUpdate:
    center: int
    sites: tuple
    edges: tuple
    delta: tuple
    local: object
    sigma: np.ndarray
    hamiltonian: np.ndarray
    kraus: tuple


@dataclass
class LocalPetz:
    graph: ImageGraph
    b: float
    mode: str
    updates: list
    layers: list

    @property
    def order(self): return [i for layer in self.layers for i in layer]


def image_graph(image, kappa=2., sigma=None, sigma_ddof=1):
    """O(N) model construction without constructing a 2**N Hamiltonian."""
    image=np.asarray(image,dtype=float)
    if image.ndim not in (2,3) or (image.ndim==3 and image.shape[2]!=3):
        raise ValueError('Use grayscale or RGB image arrays.')
    if 0 in image.shape or not np.all(np.isfinite(image)):
        raise ValueError('Image must be nonempty and finite.')
    if not np.isfinite(kappa) or kappa<0 or sigma_ddof not in (0,1):
        raise ValueError('Invalid kappa or standard deviation convention.')
    rows,cols=image.shape[:2]
    edges=[]
    for r in range(rows):
        for c in range(cols):
            p=r*cols+c
            if c+1<cols:edges.append((p,p+1))
            if r+1<rows:edges.append((p,p+cols))
    edges=np.array(edges,dtype=int).reshape(-1,2)
    colors=image.reshape(rows*cols,-1)
    distance=np.linalg.norm(colors[edges[:,0]]-colors[edges[:,1]],axis=1)
    if sigma is None:
        if len(distance)<=sigma_ddof:raise ValueError('Supply positive sigma for too few edges.')
        sigma=float(np.std(distance,ddof=sigma_ddof))
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Supply a positive sigma for zero contrast variance.')
    return ImageGraph(image.copy(),edges,2-kappa*distance/sigma,float(sigma),float(kappa),sigma_ddof)


@lru_cache(maxsize=256)
def _patch_data(size, edges, delta, b, center):
    h=xxz_hamiltonian(size,np.array(edges,dtype=int).reshape(-1,2),np.array(delta))
    bath=build_quantum_heat_bath(h,b)
    update=bath.updates[center]
    return update,bath.sigma,h,tuple(full_kraus(update))


def build_local_petz(graph,b=6.,mode='star'):
    """star: one site + all neighbors (<=5). strip3: up to three rungs (<=6).

    strip3 is a larger-patch control restricted to two-row images. Both use
    all original edges internal to each patch, with no sigma recalculation.
    """
    if not np.isfinite(b) or b<0:raise ValueError('Require finite b >= 0.')
    rows,cols=graph.image.shape[:2]
    if mode not in ('star','strip3') or (mode=='strip3' and rows!=2):
        raise ValueError('Use star, or strip3 on a two-row image.')
    adjacency=[{} for _ in range(graph.n)]
    for (a,c),delta in zip(graph.edges,graph.delta):
        adjacency[a][int(c)]=float(delta);adjacency[c][int(a)]=float(delta)
    updates=[]
    for center in range(graph.n):
        r,c=divmod(center,cols)
        if mode=='star':sites=tuple(sorted([center,*adjacency[center]]))
        else:sites=tuple(rr*cols+cc for rr in range(2) for cc in range(max(0,c-1),min(cols,c+2)))
        index={p:i for i,p in enumerate(sites)}
        edges=[];deltas=[]
        for p in sites:
            for q,delta in adjacency[p].items():
                if p<q and q in index:edges.append((index[p],index[q]));deltas.append(delta)
        edges=tuple(edges);deltas=tuple(deltas)
        u,sigma,h,kraus=_patch_data(len(sites),edges,deltas,float(b),index[center])
        updates.append(PatchUpdate(center,sites,edges,deltas,u,sigma,h,kraus))
    # Stars centered at r+2c mod 5 are disjoint on a square grid. Three-rung
    # windows use six layers: one row and one column residue modulo three.
    colors=5 if mode=='star' else 6
    layers=[[] for _ in range(colors)]
    for i in range(graph.n):
        r,c=divmod(i,cols);color=(r+2*c)%5 if mode=='star' else 3*r+c%3
        layers[color].append(i)
    layers=[layer for layer in layers if layer]
    for layer in layers:
        seen=set()
        for i in layer:
            if seen.intersection(updates[i].sites):raise RuntimeError('Overlapping parallel patches.')
            seen.update(updates[i].sites)
    return LocalPetz(graph,float(b),mode,updates,layers)


def apply_dense_update(rho,update,n):
    """Exact local channel tensored with identity on the entire exterior.

    Keeps correlations with the exterior; never substitutes a product of
    the patch marginal and an exterior state. Dense validation only.
    """
    if n>10 or np.shape(rho)!=(2**n,2**n):raise ValueError('Dense validation requires N <= 10.')
    sites=update.sites;rest=tuple(i for i in range(n) if i not in sites)
    order=sites+rest+tuple(n+i for i in sites+rest)
    d=2**len(sites);e=2**len(rest)
    tensor=np.asarray(rho).reshape((2,)*(2*n)).transpose(order).reshape(d,e,d,e)
    result=sum(np.einsum('ai,irjs,bj->arbs',k,tensor,k.conj(),optimize=True) for k in update.kraus)
    return result.reshape((2,)*(2*n)).transpose(np.argsort(order)).reshape(2**n,2**n)


def apply_dense_sweep(rho,model,order=None):
    order=model.order if order is None else list(order)
    if sorted(order)!=list(range(model.graph.n)):raise ValueError('Visit each center exactly once.')
    for i in order:rho=apply_dense_update(rho,model.updates[i],model.graph.n)
    return rho


def _path(edges,source,target):
    queue=deque([(source,)]);seen={source}
    while queue:
        path=queue.popleft()
        if path[-1]==target:return path
        for a,b in edges:
            v=b if a==path[-1] else a if b==path[-1] else None
            if v is not None and v not in seen:seen.add(v);queue.append(path+(v,))
    raise ValueError('Disconnected patch coupling graph.')


def route_locally(circuit,edges):
    """Route each CNOT along patch edges and undo swaps, preserving bit layout.

    Both CNOT directions are allowed. This conservative router avoids hidden
    final permutations; it is a cost bound, not an optimized compilation.
    """
    result=circuit.copy_empty_like();result.global_phase=circuit.global_phase
    def swap(a,b):result.cx(a,b);result.cx(b,a);result.cx(a,b)
    for instruction in circuit.data:
        qs=[circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name!='cx':
            result.append(instruction.operation,qs);continue
        path=_path(edges,*qs)
        hops=list(zip(path[:-2],path[1:-1]))
        for a,b in hops:swap(a,b)
        result.cx(path[-2],path[-1])
        for a,b in reversed(hops):swap(a,b)
    return result


@lru_cache(maxsize=256)
def _compiled_patch(size,edges,delta,b,center,routed):
    update,_,_,_=_patch_data(size,edges,delta,b,center)
    compiled=compile_circuit(update_circuit(update))
    if routed:
        connections=tuple((size-1-a,size-1-c) for a,c in edges)+((size,size-1-center),)
        compiled=route_locally(compiled,connections)
    return compiled


def patch_circuit(model,update,routed=True):
    return _compiled_patch(len(update.sites),update.edges,update.delta,model.b,update.local.site,routed).copy()


def image_circuit(model,sweeps=1,routed=True,parallel=True):
    """Parallel: N spins + N locally attached ancillas, <=5 star layers/sweep.

    Serial mode uses one ancilla with unrestricted access to each patch;
    moving this shared ancilla across a physical image is not costed.
    """
    if not isinstance(sweeps,int) or sweeps<1:raise ValueError('sweeps must be a positive integer.')
    n=model.graph.n
    circuit=QuantumCircuit(QuantumRegister(n,'spin'),QuantumRegister(n if parallel else 1,'bath'))
    for _ in range(sweeps):
        for layer in model.layers:
            for i in layer:
                update=model.updates[i]
                qs=[n-1-p for p in reversed(update.sites)]
                qs.append(2*n-1-update.center if parallel else n)
                circuit.compose(patch_circuit(model,update,routed),qubits=qs,inplace=True)
            circuit.barrier()
    return circuit


def local_superoperator(update):
    """Interleaved (ket,bra) physical dimension 4 for each patch spin."""
    k=len(update.sites)
    if k>5:raise ValueError('Dense superoperator limited to five-spin patches; use Kraus contractions for larger patches.')
    matrix=sum(np.kron(a,a.conj()) for a in update.kraus)
    order=tuple(x for j in range(k) for x in (j,j+k))
    order=order+tuple(x+2*k for x in order)
    return np.real_if_close(matrix.reshape((2,)*(4*k)).transpose(order).reshape(4**k,4**k))
