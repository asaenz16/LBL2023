"""Immediate-neighbor Petz bath for diagonal image Ising Hamiltonians.

H=-sum J_ij Z_i Z_j, J=atan(alpha*(threshold-distance))/4.
Unlike the XXZ patch approximation, this is the exact global single-site
Petz channel: outside interactions cancel in the conditional Gibbs ratio.
All public spin arrays/matrices use row-major pixels, pixel zero as MSB.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.special import expit
from qiskit import QuantumCircuit, QuantumRegister
from ising_monte_carlo import signed_image


@dataclass
class IsingPetz:
    graph: object
    b: float
    k: float
    neighbors: list
    probabilities: list  # P(new bit=0 | neighbors), first neighbor is index LSB
    angles: list
    layers: list

    @property
    def n(self): return self.graph.n

    @property
    def order(self): return [i for layer in self.layers for i in layer]


def alternating_image(width):
    """Two rows repeating black/white 2x2 blocks, starting with black."""
    if not isinstance(width,Integral) or width<4 or width%4:
        raise ValueError('Use L=4,8,12,... for complete black/white block pairs.')
    row=255*((np.arange(width)//2)%2)
    return np.stack([row,row]).astype(np.uint8)


def build_ising_petz(image, *, alpha=.001, threshold=100., k=.8, b=None):
    graph=signed_image(image,alpha,threshold=threshold)
    scale=float(np.median(abs(graph.delta)))/4 if len(graph.delta) else 0.
    if b is None:
        if not np.isfinite(k) or k<=0 or scale<=1e-12:
            raise ValueError('Require positive K and nonzero median coupling, or supply b explicitly.')
        b=float(k/scale)
    if not np.isfinite(b) or b<0: raise ValueError('Require finite b>=0.')
    neighbors=[];probabilities=[];angles=[]
    for i in range(graph.n):
        valid=graph.neighbors[i]>=0
        nb=graph.neighbors[i,valid].astype(int);js=graph.couplings[i,valid]
        order=np.argsort(nb);nb=nb[order];js=js[order];d=len(nb)
        bits=(np.arange(2**d)[:,None]>>np.arange(d))&1
        field=(1-2*bits)@js
        p=expit(2*b*field)
        # atan2 avoids subtracting p from 1 in cold regimes.
        theta=2*np.arctan2(np.sqrt(expit(-2*b*field)),np.sqrt(p))
        neighbors.append(tuple(int(v) for v in nb));probabilities.append(p);angles.append(theta)
    rows,cols=graph.image.shape[:2]
    # Four disjoint-star layers suffice on a two-row ladder; five on a grid.
    layers=[[] for _ in range(4 if rows==2 else 5)]
    for i in range(graph.n):
        r,c=divmod(i,cols);color=(c+2*r)%4 if rows==2 else (r+2*c)%5
        layers[color].append(i)
    layers=[layer for layer in layers if layer]
    for layer in layers:
        sites=[p for i in layer for p in (i,*neighbors[i])]
        if len(sites)!=len(set(sites)): raise RuntimeError('Parallel stars overlap.')
    return IsingPetz(graph,float(b),float(b*scale),neighbors,probabilities,angles,layers)


def gray_rotations(angles):
    """Walsh transform in cyclic Gray order; first control is the index LSB.

Each row is (RY angle, CNOT control slot after that rotation). The last
CNOT closes the Gray cycle. Zero controls give a single RY and no CNOT.
"""
    angles=np.asarray(angles,float);count=len(angles);d=count.bit_length()-1
    if angles.ndim!=1 or count!=2**d or not np.isfinite(angles).all():
        raise ValueError('Require 2**degree finite angles.')
    if d==0:return [(float(angles[0]),None)]
    result=[]
    for j in range(count):
        gray=j^(j>>1)
        signs=np.array([(-1)**((i&gray).bit_count()) for i in range(count)])
        value=float(signs@angles/count)
        nxt=(j+1)%count;change=gray^(nxt^(nxt>>1))
        control=change.bit_length()-1
        result.append((value,control))
    return result


def append_recovery(circuit, center, neighbors, bath, angles):
    for theta,control in gray_rotations(angles):
        if abs(theta)>1e-14:circuit.ry(theta,center)
        if control is not None:circuit.cx(neighbors[control],center)
    circuit.cx(center,bath)  # discard this bath to remove only the new spin coherence


def local_circuit(model,center):
    """d+1 spins plus one bath; local spin order (center, sorted neighbors).

Reset bath, swap out old center, reset bath, conditional RY on fresh center,
copy the new center to bath, reset bath. All reset outcomes are retained.
Every two-qubit gate is center-neighbor or center-bath; no routing required.
"""
    d=len(model.neighbors[center]);n=d+1
    circuit=QuantumCircuit(QuantumRegister(n,'spin'),QuantumRegister(1,'bath'),name=f'Petz_{center}')
    q=n-1;bath=n;nb=[n-2-j for j in range(d)]
    circuit.reset(bath)
    circuit.cx(q,bath);circuit.cx(bath,q);circuit.cx(q,bath)
    circuit.reset(bath)
    append_recovery(circuit,q,nb,bath,model.angles[center])
    circuit.reset(bath)
    return circuit


def image_circuit(model,sweeps=1,*,parallel=True,initial=None,measure=False):
    if not isinstance(sweeps,Integral) or sweeps<1:raise ValueError('Require positive integer sweeps.')
    n=model.n;na=n if parallel else 1
    circuit=QuantumCircuit(QuantumRegister(n,'spin'),QuantumRegister(na,'bath'),name='Ising_Petz_Algorithm1')
    if initial=='plus_x':circuit.h(range(n))
    elif initial not in (None,'all_up'):raise ValueError('Use plus_x or all_up initial state.')
    for _ in range(sweeps):
        for layer in model.layers:
            for center in layer:
                sites=(center,*model.neighbors[center]);size=len(sites)
                qs=[n-1-p for p in reversed(sites)]+[n+center if parallel else n]
                circuit.compose(local_circuit(model,center),qubits=qs,inplace=True)
            circuit.barrier()
    if measure:
        from qiskit import ClassicalRegister
        bits=ClassicalRegister(n,'pixels');circuit.add_register(bits)
        # Qiskit displayed strings are MSB-first, matching row-major pixels.
        for p in range(n):circuit.measure(n-1-p,n-1-p)
    return circuit


def correlations_from_counts(counts,n,seeds=(0,),confidence=.95):
    """Algorithm 1 readout from MSB-first strings exported by image_circuit.

One all-Z measurement supplies every seed correlation. The Hoeffding bound
is simultaneous across all requested seed/pixel pairs for independent shots.
It quantifies readout statistics, not bias from a finite number of sweeps.
"""
    if not isinstance(n,Integral) or n<1 or not 0<confidence<1:raise ValueError('Invalid readout parameters.')
    if any(not isinstance(s,Integral) or not 0<=s<n for s in seeds):raise ValueError('Invalid seed.')
    total=0;gamma=np.zeros((len(seeds),n))
    for bits,count in counts.items():
        bits=bits.replace(' ','')
        if len(bits)!=n or set(bits)-{'0','1'} or not isinstance(count,Integral) or count<0:
            raise ValueError('Counts must contain length-N binary strings and nonnegative integers.')
        z=1-2*np.fromiter((int(v) for v in bits),int)
        gamma+=count*z[np.asarray(seeds),None]*z[None,:]/4;total+=count
    if total<2 or not len(seeds):raise ValueError('Require at least two shots and one seed.')
    gamma/=total;se=np.sqrt(np.maximum(0,1/16-gamma**2)/(total-1))
    radius=np.sqrt(np.log(2*n*len(seeds)/(1-confidence))/(8*total))
    lower=np.maximum(-.25,gamma-radius);upper=np.minimum(.25,gamma+radius)
    for j,s in enumerate(seeds):lower[j,s]=upper[j,s]=.25
    return dict(gamma=gamma,se=se,shots=total,lower=lower,upper=upper,
        resolved=(lower>0)|(upper<0),same_segment=gamma>0,simultaneous_radius=radius)


def local_kraus(model,center):
    """Four Kraus operators |new><old| sqrt[p(new|neighbors)]."""
    d=len(model.neighbors[center]);dim=2**(d+1);rest=2**d
    # Array tensor factors are MSB-first; lookup's first neighbor is LSB.
    reverse=np.array([int(f'{i:0{d}b}'[::-1],2) if d else 0 for i in range(rest)])
    p=model.probabilities[center][reverse];out=[]
    for new in range(2):
        amplitude=np.sqrt(p if new==0 else 1-p)
        for old in range(2):
            a=np.zeros((dim,dim));idx=np.arange(rest)
            a[new*rest+idx,old*rest+idx]=amplitude;out.append(a)
    return tuple(out)


def apply_dense_update(rho,model,center):
    """Exact channel on an arbitrary entangled input, limited to N<=10."""
    n=model.n
    if n>10 or np.shape(rho)!=(2**n,2**n):raise ValueError('Dense simulation is restricted to N<=10.')
    sites=(center,*model.neighbors[center]);rest=tuple(i for i in range(n) if i not in sites)
    order=sites+rest+tuple(n+i for i in sites+rest)
    d=2**len(sites);e=2**len(rest)
    x=np.asarray(rho).reshape((2,)*(2*n)).transpose(order).reshape(d,e,d,e)
    out=sum(np.einsum('ai,irjs,bj->arbs',a,x,a.conj(),optimize=True) for a in local_kraus(model,center))
    return out.reshape((2,)*(2*n)).transpose(np.argsort(order)).reshape(2**n,2**n)


def apply_dense_sweep(rho,model):
    for center in model.order:rho=apply_dense_update(rho,model,center)
    return rho


def ladder_gibbs(model,seeds=(0,)):
    """Exact 4x4 transfer contraction for ANY two-row Ising image.

Computes seed correlations with O(L) small-matrix work per seed, and O(L)
storage. No Monte Carlo and no full Hilbert-space enumeration are used.
"""
    rows,width=model.graph.image.shape[:2]
    if rows!=2:raise ValueError('Transfer reference requires a two-row image.')
    if any(not isinstance(s,Integral) or not 0<=s<model.n for s in seeds):raise ValueError('Invalid seed.')
    z=np.array([[1,1],[1,-1],[-1,1],[-1,-1]],float)
    vertical=np.zeros(width);horizontal=np.zeros((2,max(0,width-1)))
    for (p,q),delta in zip(model.graph.edges,model.graph.delta):
        r,c=divmod(int(p),width)
        if q==p+width:vertical[c]=delta/4
        else:horizontal[r,c]=delta/4
    v=np.exp(model.b*vertical[:,None]*(z[:,0]*z[:,1])[None,:]-abs(model.b*vertical[:,None]))
    ts=[]
    for c in range(width-1):
        log=model.b*sum(horizontal[r,c]*z[:,r,None]*z[None,:,r] for r in range(2))
        log+=np.log(v[c+1])[None,:];ts.append(np.exp(log-log.max()))
    f=np.empty((width,4));g=np.empty_like(f);norms=[];back=[]
    f[0]=v[0]/v[0].sum()
    for c,t in enumerate(ts):
        value=f[c]@t;norms.append(value.sum());f[c+1]=value/value.sum()
    g[-1]=1.
    for c in range(width-2,-1,-1):
        value=ts[c]@g[c+1];back.append(value.sum());g[c]=value/value.sum()
    # Store backward factors in increasing bond order.
    back=back[::-1];gamma=[]
    for seed in seeds:
        sr,sc=divmod(int(seed),width);values=np.zeros((2,width))
        tagged=f[sc]*z[:,sr]
        for c in range(sc,width):
            if c>sc:tagged=tagged@ts[c-1]/norms[c-1]
            for r in range(2):values[r,c]=(tagged*z[:,r])@g[c]/(f[c]@g[c])/4
        tagged=g[sc]*z[:,sr]
        for c in range(sc-1,-1,-1):
            tagged=ts[c]@tagged/back[c]
            for r in range(2):values[r,c]=(f[c]*z[:,r])@tagged/(f[c]@g[c])/4
        values[sr,sc]=.25;gamma.append(values.ravel())
    return np.asarray(gamma)
