"""Same SSE heat-bath/directed-loop measure with sparse diagonal bookkeeping.

Fenwick trees replace full-edge scans after each vertex. The diagonal
conditional and fixed-number loop update are unchanged. The ZZ estimator
accumulates only changes instead of visiting every site at every vertex.
"""
import numpy as np
from numba import njit
from xxz_glauber import _initialize,_loop_sweep


def expansion_cutoff(model,b,log_tolerance=-50.):
    """Conservative log Chernoff tail, avoiding Poisson survival underflow.

For k=M+1>mu, log P(Poisson(mu)>=k) <= -mu+k*(1+log(mu/k)).
Include the trace lower-bound prefactor from the original SSE implementation.
"""
    if not np.isfinite(b) or b<=0:raise ValueError('Require finite positive b.')
    row_bound=float(np.maximum(model.shift+model.delta/4,model.shift-model.delta/4+.5).sum())
    mean=b*row_bound;offset=b*(row_bound-model.shift.sum())
    cutoff=max(32,int(np.ceil(mean)))
    while True:
        k=cutoff+1;bound=offset-mean+k*(1+np.log(mean/k))
        if bound<log_tolerance:return cutoff,float(bound)
        cutoff+=max(4,int(np.sqrt(mean)))


@njit(cache=True)
def _add(tree,index,value):
    index+=1
    while index<len(tree):
        tree[index]+=value;index+=index&-index


@njit(cache=True)
def _draw(tree,total):
    target=np.random.random()*total;index=0;bit=1
    while bit*2<len(tree):bit*=2
    while bit:
        candidate=index+bit
        if candidate<len(tree) and tree[candidate]<=target:
            target-=tree[candidate];index=candidate
        bit//=2
    return min(index,len(tree)-2)


@njit(cache=True)
def _diagonal(spins,operators,edges,delta,shift,incident,b):
    state=spins.copy();weights=np.empty(len(edges));tree=np.zeros(len(edges)+1)
    total=0.;count=np.count_nonzero(operators)
    for e in range(len(edges)):
        i,j=edges[e];weights[e]=shift[e]+delta[e]*state[i]*state[j]/4
        _add(tree,e,weights[e]);total+=weights[e]
    for p in range(len(operators)):
        op=operators[p]
        if op&1:
            i,j=edges[(op-2)//2];state[i]*=-1;state[j]*=-1
            for endpoint in (i,j):
                for e in incident[endpoint]:
                    if e<0:break
                    a,c=edges[e];new=shift[e]+delta[e]*state[a]*state[c]/4
                    change=new-weights[e];weights[e]=new;total+=change;_add(tree,e,change)
        else:
            rest=count-(1 if op else 0)
            probability=b*total/(len(operators)-rest+b*total)
            if np.random.random()<probability:
                operators[p]=2*_draw(tree,total)+2;count=rest+1
            else:operators[p]=0;count=rest
    if np.any(state!=spins):raise RuntimeError('Nonperiodic SSE path.')
    return count


@njit(cache=True)
def _observe_fast(spins,operators,edges,seeds):
    n=len(spins);state=spins.copy();current=np.empty((len(seeds),n))
    total=np.zeros_like(current);last=np.zeros((len(seeds),n),np.int64)
    for s in range(len(seeds)):current[s]=state[seeds[s]]*state/4.
    count=0
    for op in operators:
        if not op:continue
        count+=1
        if op&1:
            i,j=edges[(op-2)//2];state[i]*=-1;state[j]*=-1
            for s in range(len(seeds)):
                if seeds[s]==i or seeds[s]==j:
                    for p in range(n):
                        total[s,p]+=(count-last[s,p])*current[s,p]
                        current[s,p]=state[seeds[s]]*state[p]/4;last[s,p]=count
                else:
                    for p in (i,j):
                        total[s,p]+=(count-last[s,p])*current[s,p]
                        current[s,p]*=-1;last[s,p]=count
    if np.any(state!=spins):raise RuntimeError('Observation of nonperiodic path.')
    total+=(count+1-last)*current
    return total/(count+1),count


@njit(cache=True)
def _run_fast(edges,delta,shift,exits,incident,initial,b,capacity,times,seeds,nloops,seed):
    np.random.seed(seed);spins=initial.copy()
    operators=_initialize(spins,edges,delta,shift,b,capacity)
    profiles=np.empty((len(times),len(seeds),len(spins)))
    orders=np.empty(len(times),np.int64);sectors=np.empty(len(times),np.int64)
    maximum=np.count_nonzero(operators);t=0
    for k in range(len(times)):
        while t<times[k]:
            count=_diagonal(spins,operators,edges,delta,shift,incident,b)
            maximum=max(maximum,count)
            _loop_sweep(spins,operators,edges,exits,nloops);t+=1
        profiles[k],orders[k]=_observe_fast(spins,operators,edges,seeds)
        sectors[k]=spins.sum()
    return profiles,orders,sectors,maximum


def run_fast(model,initial,b,times,seeds,seed,capacity=None,nloops=8):
    initial=np.asarray(initial,dtype=np.int64);times=np.asarray(times,dtype=np.int64)
    seeds=np.asarray(seeds,dtype=np.int64)
    if initial.shape!=(model.n,) or not np.isin(initial,[-1,1]).all():raise ValueError('Invalid initial path.')
    if not len(times) or np.any(times<0) or np.any(np.diff(times)<=0):raise ValueError('Invalid times.')
    if not len(seeds) or np.any(seeds<0) or np.any(seeds>=model.n):raise ValueError('Invalid seeds.')
    minimum,_=expansion_cutoff(model,b)
    if capacity is None:capacity=minimum
    if capacity<minimum:raise ValueError('Expansion cutoff is too small.')
    if not isinstance(nloops,int) or nloops<1:raise ValueError('Require a fixed positive loop count.')
    degree=np.bincount(model.edges.ravel(),minlength=model.n)
    incident=np.full((model.n,int(degree.max())),-1,np.int64);counts=np.zeros(model.n,int)
    for e,pair in enumerate(model.edges):
        for p in pair:incident[p,counts[p]]=e;counts[p]+=1
    return _run_fast(model.edges,model.delta,model.shift,model.exits,incident,initial,float(b),
        int(capacity),times,seeds,nloops,int(seed))
