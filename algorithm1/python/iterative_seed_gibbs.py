"""Fresh-round diagonal Gibbs measurements and adaptive signed seed readouts.

This module does not clamp spins or change the Hamiltonian after promotion.
SSE updates are the previously validated ones; the new observable is a Z-basis
snapshot, permitting every ZZ readout from the same commuting measurement.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from numba import njit
from scipy.stats import t as student_t
from xxz_glauber import graph,_initialize,_loop_sweep
from xxz_sse_fast import _diagonal,expansion_cutoff
from xxz_sse_cluster import _worldline_cluster
from xxz_image_gibbs import starts,statistics
from xxz_two_seed import parameterize as old_parameterize
from ising_monte_carlo import signed_image
from ising_seed_sampling import setup_sampler,sample_spins


def parameterize(image,rule):
    if rule['threshold']!='fixed125':return old_parameterize(image,rule)
    raw=signed_image(image,.001);d=raw.distances
    spread=max(float(np.diff(np.quantile(d,[.1,.9]))[0]),1.)
    alpha=.01 if rule['sensitivity']=='fixed01' else 1/spread
    model=signed_image(image,alpha,threshold=125.)
    scale=float(np.median(abs(model.delta)));fallback=scale<1e-8
    if fallback:scale=max(float(np.mean(abs(model.delta))),.001)
    g=rule['strength']*scale;b=4*rule['K']/scale
    return model,dict(threshold=125.,alpha=alpha,edge_spread=spread,median_abs_delta=scale,
        normalization_fallback=fallback,g=g,g_over_scale=g/scale,b=b,beta_paper=b*model.n,
        ferromagnetic_fraction=float(np.mean(model.delta>g)),intermediate_fraction=float(np.mean(abs(model.delta)<=g)),
        antiferromagnetic_fraction=float(np.mean(model.delta < -g)))


@njit(cache=True)
def _snapshots(edges,delta,shift,exits,incident,initial,b,capacity,burn,count,thin,seed):
    np.random.seed(seed);spins=initial.copy();operators=_initialize(spins,edges,delta,shift,b,capacity)
    snapshots=np.empty((count,len(spins)),np.int8);orders=np.empty(count);sectors=np.empty(count)
    maximum=np.count_nonzero(operators);out=0
    for t in range(burn+count*thin):
        order=_diagonal(spins,operators,edges,delta,shift,incident,b)
        maximum=max(maximum,order);_loop_sweep(spins,operators,edges,exits,8)
        _worldline_cluster(spins,operators,edges,delta,shift)
        if t>=burn and (t-burn)%thin==0:
            snapshots[out]=spins;orders[out]=order;sectors[out]=spins.sum();out+=1
    return snapshots,orders,sectors,maximum


def iter_rounds(image,rule,*,rounds,samples,burn,thin,chains,rng_seed):
    model,param=parameterize(image,rule);g=param['g'];b=param['b']
    if g:
        quantum=graph(model.n,model.edges,model.delta/g);capacity,tail=expansion_cutoff(quantum,b*g)
        degree=np.bincount(model.edges.ravel(),minlength=model.n)
        incident=np.full((model.n,int(degree.max())),-1,np.int64);counts=np.zeros(model.n,int)
        for e,pair in enumerate(model.edges):
            for p in pair:incident[p,counts[p]]=e;counts[p]+=1
    else:sampler=setup_sampler(image.shape,model.edges,model.delta,b);capacity=0;tail=None
    # Every round is a separately seeded, independently initialized ensemble.
    children=np.random.SeedSequence(rng_seed).spawn(rounds*chains)
    for t in range(rounds):
        zz=[];scalars=[];maximum=0
        for c,initial in enumerate(starts(image,chains,int(children[t*chains].generate_state(1)[0]))):
            seed=int(children[t*chains+c].generate_state(1)[0]%(2**31-1))
            if g:
                z,orders,sectors,m=_snapshots(quantum.edges,quantum.delta,quantum.shift,quantum.exits,
                    incident,initial,b*g,capacity,burn,samples,thin,seed)
                e=g*(quantum.shift.sum()-orders/(b*g))/model.n;maximum=max(maximum,int(m))
                mag=(sectors/model.n)**2
            else:
                z=sample_spins(sampler,samples,np.random.default_rng(seed))
                e=-(z[:,model.edges[:,0]]*z[:,model.edges[:,1]])@model.delta/4/model.n
                mag=z.mean(axis=1)**2
            zz.append(z);scalars.append(np.column_stack([e,mag]))
        sm,se,ess,rh,_=statistics(np.asarray(scalars),blocks=16)
        diagnostic=dict(scalar_rhat=float(rh.max()) if np.isfinite(rh.max()) else None,
            scalar_ess=ess.tolist(),energy_per_spin=float(sm[0]),magnetization_squared=float(sm[1]),
            capacity=capacity,maximum_order=maximum,log_tail_bound=tail)
        yield np.packbits(np.asarray(zz)>0,axis=-1,bitorder='little'),param,diagnostic


def sample_rounds(image,rule,**kwargs):
    rows=list(iter_rounds(image,rule,**kwargs))
    return np.stack([r[0] for r in rows]),rows[0][1],[r[2] for r in rows]


def unpack(packed,n):
    return (2*np.unpackbits(packed,axis=-1,count=n,bitorder='little').astype(np.int8)-1)


class RoundMoments:
    """Sufficient second moments for any signed mean of ZZ correlations.

    For score observation Zi*(w.Z), its second moment is E[(w.Z)^2],
    independent of i. Thus means, batch errors and full-series split-Rhat
    can all be recovered without storing a sample-by-N-by-N tensor.
    """
    def __init__(self,spins,blocks=16,scalar_rhat=1.):
        z=np.asarray(spins,dtype=np.float32);self.chains,self.count,self.n=z.shape
        if self.count%blocks or blocks%2:raise ValueError('Even blocks dividing sample count required.')
        self.blocks=blocks;self.scalar_rhat=scalar_rhat
        q=z.reshape(self.chains,blocks,self.count//blocks,self.n)
        self.cov=np.matmul(q.swapaxes(-1,-2),q)/(self.count//blocks)

    def infer(self,foreground,background,max_rounds=10):
        idx=np.array(tuple(foreground)+tuple(background),dtype=int)
        if not len(idx) or len(set(idx))!=len(idx):raise ValueError('Nonempty disjoint seed sets required.')
        weights=np.r_[np.full(len(foreground),1/len(foreground)) if foreground else [],
            np.full(len(background),-1/len(background)) if background else []]
        batch=np.sum(self.cov[:,:,idx,:]*weights[None,None,:,None],axis=2,dtype=np.float64)
        return infer_batches(batch,idx,weights,self.cov,self.count,self.scalar_rhat,max_rounds)


def infer_batches(batch,idx,weights,cov,count,scalar_rhat,max_rounds):
    chains,blocks,n=batch.shape;point=batch.mean(axis=(0,1));cm=batch.mean(axis=1)
    se=np.maximum(batch.reshape(chains*blocks,n).std(axis=0,ddof=1)/np.sqrt(chains*blocks),
        cm.std(axis=0,ddof=1)/np.sqrt(chains))
    mult=float(student_t.ppf(1-.05/(2*n*max_rounds),chains*blocks-1));radius=mult*se
    half=batch.reshape(chains,2,blocks//2,n).mean(axis=2).reshape(chains*2,n)
    second=np.sum(half[:,idx]*weights[None,:],axis=1);m=count//2
    within=np.maximum(second[:,None]-half**2,0.)*m/(m-1)
    w=within.mean(axis=0);between=m*half.var(axis=0,ddof=1)
    rh=np.ones(n);vary=w>1e-14
    rh[vary]=np.sqrt((m-1)/m+between[vary]/(m*w[vary]))
    rh[(~vary)&(between>1e-14)]=np.inf
    total=chains*count;second_all=float(np.mean(np.sum(cm[:,idx]*weights,axis=1)))
    variance=np.maximum(second_all-point**2,0.)*total/(total-1)
    ess=np.full(n,float(total));np.divide(variance,se**2,out=ess,where=se>1e-14);ess=np.minimum(ess,total)
    max_rhat=max(float(rh.max()),float(scalar_rhat) if scalar_rhat is not None else np.inf)
    return dict(score=point,radius=radius,se=se,batches=batch,rhat=rh,ess=ess,
        converged=bool(max_rhat<1.05),max_rhat=max_rhat,
        minimum_ess=float(ess[variance>1e-12].min()) if np.any(variance>1e-12) else float(total))


def candidates(inference,foreground,background,cut_fg,cut_bg,gate):
    score=inference['score'];radius=inference['radius'];available=np.ones(len(score),bool)
    available[list(foreground)+list(background)]=False
    if not inference['converged']:return np.array([],int),np.array([],int)
    if gate=='confidence':
        positive=(score-radius>cut_fg);negative=(score+radius < -cut_bg)
    elif gate=='point_resolved':
        positive=(score>cut_fg)&(score>radius);negative=(score < -cut_bg)&(score < -radius)
    else:raise ValueError('Unknown candidate gate.')
    return np.flatnonzero(available&positive),np.flatnonzero(available&negative)


def stop_index(counts,terminals,cap,stopping,n_unknown):
    """Return zero-based last observed round and the reason, before promotion."""
    best=counts[0];stale=0;same=0
    for k,count in enumerate(counts):
        if terminals[k]:return k,terminals[k]
        if k+1>=cap:return k,'round_cap'
        if k:
            same=same+1 if count==counts[k-1] else 0
            tolerance=0 if stopping!='tolerance_1pct' else max(1,int(np.ceil(.01*n_unknown)))
            if count<best-tolerance:best=count;stale=0
            else:stale+=1
            if stopping=='same_count' and same>=2:return k,'stable_count'
            if stopping in ('no_improvement','tolerance_1pct') and stale>=2:return k,'no_improvement'
        if stopping not in ('none','same_count','no_improvement','tolerance_1pct'):raise ValueError('Unknown stopping rule.')
    return len(counts)-1,'round_cap'
