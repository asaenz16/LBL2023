"""Arctangent image Ising/XXZ Gibbs sampling, without quantum circuits.

H = -sum [g(XX+YY) + Delta ZZ]/4. For g>0, H = g H_SSE(Delta/g),
so the existing independently tested directed-loop sampler applies at b*g.
The diagonal limit uses signed clusters and local heat-bath sweeps.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from types import SimpleNamespace
import numpy as np
from numba import njit
from scipy.stats import t as student_t
from ising_monte_carlo import signed_image,cluster_step,energy
from ising_grid_transfer import exact_grid_gibbs
from ising_thresholds import otsu_edge_threshold
from xxz_glauber import graph,run_chain,exact_gibbs,expansion_cutoff
from xxz_sse_fast import run_fast,expansion_cutoff
from weizmann_petz_metrics import mask_metrics,segmentation


def parameterize(image,rule):
    initial=signed_image(image,.001);d=initial.distances
    kind=rule['threshold']
    if kind=='fixed50':tau=50.
    elif kind=='fixed100':tau=100.
    elif kind=='q75':tau=max(float(np.quantile(d[d>0],.75)) if np.any(d>0) else 100.,1.)
    elif kind=='edge_otsu':tau=max(otsu_edge_threshold(d,fallback=100.),1.)
    else:raise ValueError('Unknown threshold rule.')
    spread=max(float(np.diff(np.quantile(d,[.1,.9]))[0]),1.)
    sensitivity=rule['sensitivity']
    if sensitivity=='fixed001':alpha=.001
    elif sensitivity=='fixed01':alpha=.01
    elif sensitivity=='inverse_spread':alpha=1/spread
    else:raise ValueError('Unknown sensitivity rule.')
    model=signed_image(image,alpha,threshold=tau)
    scale=float(np.median(abs(model.delta)))
    # An explicit image-only fallback avoids an undefined beta at quantile ties.
    fallback=scale<1e-8
    if fallback:scale=max(float(np.mean(abs(model.delta))),1e-3)
    transverse=rule['transverse']
    g=float(rule['strength'])*(scale if transverse=='relative' else 1.)
    if g<0 or not np.isfinite(g):raise ValueError('Require nonnegative transverse exchange.')
    if rule['temperature']=='longitudinal':denominator=scale
    elif rule['temperature']=='combined':denominator=scale+g
    else:raise ValueError('Unknown temperature rule.')
    b=4*rule['K']/denominator
    if b<=0:raise ValueError('Require positive K.')
    halfwidth=float(np.tan(g)/alpha) if 0<g<np.pi/2 else None
    regime=dict(ferromagnetic_fraction=float(np.mean(model.delta>g)),
        intermediate_fraction=float(np.mean(abs(model.delta)<=g)),
        antiferromagnetic_fraction=float(np.mean(model.delta < -g)),
        halfwidth=halfwidth,lower_distance=tau-halfwidth if halfwidth is not None else None,
        upper_distance=tau+halfwidth if halfwidth is not None else None)
    return model,dict(threshold=tau,alpha=alpha,edge_spread=spread,median_abs_delta=scale,
        normalization_fallback=fallback,g=g,g_over_scale=g/scale,b=b,beta_paper=b*model.n,**regime)


@njit(cache=True)
def _ising_chain(edges,j,neighbors,couplings,b,initial,seeds,burn,samples,thin,rng_seed):
    np.random.seed(rng_seed);z=initial.copy();prob=-np.expm1(-2*b*np.abs(j))
    gamma=np.empty((samples,len(seeds),len(z)));observables=np.empty((samples,2))
    total=burn+samples*thin;out=0
    for step in range(total):
        for i in range(len(z)):
            h=0.
            for k in range(4):
                nb=neighbors[i,k]
                if nb>=0:h+=couplings[i,k]*z[nb]
            z[i]=1 if np.random.random()<(1+np.tanh(b*h))/2 else -1
        corr=cluster_step(z,edges,j,prob,seeds)
        if step>=burn and (step-burn)%thin==0:
            gamma[out]=corr;observables[out,0]=energy(z,edges,j)/len(z)
            observables[out,1]=(z.sum()/len(z))**2;out+=1
    return gamma,observables


def starts(image,count,seed):
    rng=np.random.default_rng(seed);n=image.size
    checker=1-2*(np.indices(image.shape).sum(axis=0).ravel()%2)
    kinds=[np.ones(n,dtype=np.int64),rng.choice([-1,1],n),checker.astype(np.int64),
           np.where(image.ravel()>np.median(image),-1,1).astype(np.int64)]
    return [kinds[i%4].copy() if i<4 else rng.choice([-1,1],n) for i in range(count)]


def rhat(values):
    """Unranked split R-hat; constants agreeing in every chain return 1."""
    n=values.shape[1]//2
    split=np.concatenate([values[:,:n],values[:,-n:]],axis=0)
    within=np.var(split,axis=1,ddof=1).mean(axis=0)
    between=n*np.var(split.mean(axis=1),axis=0,ddof=1)
    result=np.ones_like(within)
    np.divide((n-1)*within/n+between/n,within,out=result,where=within>1e-30)
    result=np.sqrt(result)
    result=np.where((within<=1e-30)&(between>1e-30),np.inf,result)
    return result


def statistics(samples,blocks=16):
    chains,count=samples.shape[:2];usable=count-count%blocks
    batch=samples[:,:usable].reshape(chains,blocks,usable//blocks,*samples.shape[2:]).mean(axis=2)
    mean=samples.mean(axis=(0,1))
    batch_se=batch.reshape(chains*blocks,*samples.shape[2:]).std(axis=0,ddof=1)/np.sqrt(chains*blocks)
    between_se=samples.mean(axis=1).std(axis=0,ddof=1)/np.sqrt(chains)
    se=np.maximum(batch_se,between_se)
    variance=samples.reshape(-1,*samples.shape[2:]).var(axis=0,ddof=1)
    ess=np.full_like(se,float(chains*count))
    np.divide(variance,se**2,out=ess,where=se>1e-15)
    return mean,se,np.minimum(ess,chains*count),rhat(samples),batch


def evaluate(gamma,radius,truth,seeds):
    labels=truth.ravel()[seeds];pred=np.stack([segmentation(g,l) for g,l in zip(gamma,labels)])
    resolved=abs(gamma)>np.maximum(radius,1e-12)
    correct=pred==truth.ravel()[None,:]
    primary=pred[0];known=resolved[0];t=truth.ravel()
    tp=np.sum(primary&known&t);fp=np.sum(primary&known&~t)
    denominator=int(t.sum()+fp+np.sum(~known&~t))
    lower=float(tp/denominator) if denominator else 1.
    upper_numerator=int(tp+np.sum(~known&t));upper_denominator=int(t.sum()+fp)
    upper=float(upper_numerator/upper_denominator) if upper_denominator else 1.
    return dict(**mask_metrics(primary,t),iou_lower=lower,iou_upper=upper,
        resolved_fraction=float(resolved.mean()),primary_resolved_fraction=float(known.mean()),
        correct_resolved_fraction=float(np.mean(resolved&correct)),
        wrong_resolved_fraction=float(np.mean(resolved&~correct)),
        secondary_iou=mask_metrics(pred[-1],t)['iou'],seed_agreement=float(np.mean(pred[0]==pred[-1])))


def sample(image,rule,seeds,*,burn=1000,samples=2048,thin=2,chains=4,rng_seed=71823):
    model,params=parameterize(image,rule);b=params['b'];g=params['g']
    initial=starts(image,chains,rng_seed);profiles=[];scalars=[];maximum=0
    if g>0:
        quantum=graph(model.n,model.edges,model.delta/g)
        capacity,logtail=expansion_cutoff(quantum,b*g)
    else:capacity=0;logtail=None
    for c in range(chains):
        if g>0:
            r=run_fast(quantum,initial[c],b*g,burn+1+thin*np.arange(samples),seeds,
                seed=(rng_seed+7919*c)%(2**31-1),capacity=capacity)
            profiles.append(r[0]);scalars.append(np.column_stack([
                g*(quantum.shift.sum()-r[1]/(b*g))/model.n,(r[2]/model.n)**2]))
            maximum=max(maximum,int(r[-1]))
        else:
            p,e=_ising_chain(model.edges,model.delta/4,model.neighbors,model.couplings,b,
                initial[c],np.asarray(seeds),burn,samples,thin,(rng_seed+7919*c)%(2**31-1))
            profiles.append(p);scalars.append(e)
    profiles=np.asarray(profiles);scalars=np.asarray(scalars)
    mean,se,ess,rh,batches=statistics(profiles)
    # Approximate simultaneous MCMC intervals, not iid Hoeffding guarantees.
    # Batch means and between-chain disagreement both contribute to the SE.
    multiplier=float(student_t.ppf(1-.05/(2*mean.size),chains*16-1))
    radius=multiplier*se
    scalar_mean,scalar_se,scalar_ess,scalar_rhat,_=statistics(scalars)
    variable=np.std(profiles,axis=(0,1))>1e-12
    diag=dict(max_correlation_rhat=float(np.max(rh)),max_scalar_rhat=float(np.max(scalar_rhat)),
        minimum_variable_correlation_ess=float(np.min(ess[variable])) if variable.any() else float(chains*samples),
        maximum_correlation_se=float(se.max()),maximum_radius=float(radius.max()),
        energy_per_spin=float(scalar_mean[0]),energy_per_spin_se=float(scalar_se[0]),
        magnetization_squared=float(scalar_mean[1]),scalar_ess=scalar_ess.tolist(),
        capacity=capacity,maximum_order=maximum,log_relative_expansion_tail_bound=logtail,
        converged=bool(np.max(rh)<1.05 and np.max(scalar_rhat)<1.05),
        burn=burn,samples_per_chain=samples,thin=thin,chains=chains,interval_multiplier=multiplier,
        fixed_directed_loops_per_sweep=8 if g>0 else 0)
    # Infinity is meaningful diagnostically, but use a JSON-compatible sentinel.
    for key in ('max_correlation_rhat','max_scalar_rhat'):
        if not np.isfinite(diag[key]):diag[key]=None
    reference=None
    if g==0:reference=exact_grid_gibbs(SimpleNamespace(graph=model,b=b,n=model.n),seeds)
    elif model.n<=10:reference=exact_gibbs(quantum,b*g,seeds)['gamma']
    if reference is not None:
        diag['gamma_linf_vs_exact']=float(abs(mean-reference).max())
        diag['exact_outside_interval_fraction']=float(np.mean(abs(mean-reference)>radius+1e-12))
    return dict(params=params,diagnostics=diag,gamma=mean,se=se,radius=radius,
        batches=batches,chain_means=profiles.mean(axis=1),scalar_series=scalars,reference=reference)
