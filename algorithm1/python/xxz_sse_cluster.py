"""Additional exact worldline-cluster update for the SSE joint distribution.

For a fixed operator string, flipping both entire endpoint worldlines of
every exchange vertex preserves its weight and legality. These exchanges
are hard bonds. A diagonal vertex has W=C+Delta*s_i*s_j/4 and its flipped
weight Wopp=C-Delta*s_i*s_j/4. If W>Wopp, freeze the relative worldline
orientation with probability 1-Wopp/W. Otherwise leave it unfrozen.
Flip each resulting component independently with probability 1/2.
This is a signed Swendsen-Wang conditional on the operator string. It is
combined with the existing diagonal heat bath and directed loops, which
also change that string. No physical quantum bath or real-time claim.
"""
import numpy as np
from numba import njit
from xxz_glauber import _initialize,_loop_sweep
from xxz_sse_fast import _diagonal,_observe_fast,expansion_cutoff


@njit(cache=True)
def _root(parent,p):
    while parent[p]!=p:
        parent[p]=parent[parent[p]];p=parent[p]
    return p


@njit(cache=True)
def _worldline_cluster(spins,operators,edges,delta,shift):
    parent=np.arange(len(spins));sizes=np.ones(len(spins),np.int64);state=spins.copy()
    for op in operators:
        if not op:continue
        e=(op-2)//2;i,j=edges[e];join=False
        if op&1:
            if state[i]==state[j]:raise RuntimeError('Illegal exchange before cluster move.')
            join=True;state[i]*=-1;state[j]*=-1
        else:
            aligned=delta[e]*state[i]*state[j]
            if aligned>0:join=np.random.random()<aligned/(2*shift[e]+aligned/2)
        if join:
            a,c=_root(parent,i),_root(parent,j)
            if a!=c:
                if sizes[a]<sizes[c]:a,c=c,a
                parent[c]=a;sizes[a]+=sizes[c]
    if np.any(state!=spins):raise RuntimeError('Nonperiodic path before cluster move.')
    flips=np.empty(len(spins),np.int64)
    for i in range(len(spins)):flips[i]=1 if np.random.random()<.5 else -1
    for i in range(len(spins)):spins[i]*=flips[_root(parent,i)]


@njit(cache=True)
def _run_cluster(edges,delta,shift,exits,incident,initial,b,capacity,times,seeds,nloops,seed):
    np.random.seed(seed);spins=initial.copy()
    operators=_initialize(spins,edges,delta,shift,b,capacity)
    profiles=np.empty((len(times),len(seeds),len(spins)))
    orders=np.empty(len(times),np.int64);sectors=np.empty(len(times),np.int64)
    maximum=np.count_nonzero(operators);t=0
    for k in range(len(times)):
        while t<times[k]:
            count=_diagonal(spins,operators,edges,delta,shift,incident,b)
            maximum=max(maximum,count)
            _loop_sweep(spins,operators,edges,exits,nloops)
            _worldline_cluster(spins,operators,edges,delta,shift);t+=1
        profiles[k],orders[k]=_observe_fast(spins,operators,edges,seeds)
        sectors[k]=spins.sum()
    return profiles,orders,sectors,maximum


def run_cluster(model,initial,b,times,seeds,seed,capacity=None,nloops=8):
    initial=np.asarray(initial,dtype=np.int64);times=np.asarray(times,dtype=np.int64)
    seeds=np.asarray(seeds,dtype=np.int64)
    if initial.shape!=(model.n,) or not np.isin(initial,[-1,1]).all():raise ValueError('Invalid initial path.')
    if not len(times) or np.any(times<0) or np.any(np.diff(times)<=0):raise ValueError('Invalid times.')
    if not len(seeds) or np.any(seeds<0) or np.any(seeds>=model.n):raise ValueError('Invalid seeds.')
    minimum,_=expansion_cutoff(model,b)
    if capacity is None:capacity=minimum
    if capacity<minimum:raise ValueError('Expansion cutoff is too small.')
    degree=np.bincount(model.edges.ravel(),minlength=model.n)
    incident=np.full((model.n,int(degree.max())),-1,np.int64);counts=np.zeros(model.n,int)
    for e,pair in enumerate(model.edges):
        for p in pair:incident[p,counts[p]]=e;counts[p]+=1
    return _run_cluster(model.edges,model.delta,model.shift,model.exits,incident,initial,float(b),
        int(capacity),times,seeds,nloops,int(seed))
