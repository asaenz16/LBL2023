"""Gibbs estimates with both directed loops and worldline-cluster updates.

The original estimator and parameter conventions are retained. The additional
conditional cluster move addresses poor mixing in weak-exchange regimes.
"""
import numpy as np
from scipy.stats import t as student_t
from xxz_image_gibbs import parameterize,starts,statistics,sample as sample_v1
from xxz_glauber import graph,exact_gibbs
from xxz_sse_fast import expansion_cutoff
from xxz_sse_cluster import run_cluster


def sample(image,rule,seeds,*,burn=1000,samples=2048,thin=2,chains=4,rng_seed=71823):
    if rule['strength']==0:
        return sample_v1(image,rule,seeds,burn=burn,samples=samples,thin=thin,chains=chains,rng_seed=rng_seed)
    model,params=parameterize(image,rule);b=params['b'];g=params['g']
    quantum=graph(model.n,model.edges,model.delta/g)
    capacity,logtail=expansion_cutoff(quantum,b*g)
    profiles=[];scalars=[];maximum=0
    for c,initial in enumerate(starts(image,chains,rng_seed)):
        r=run_cluster(quantum,initial,b*g,burn+1+thin*np.arange(samples),seeds,
            seed=(rng_seed+7919*c)%(2**31-1),capacity=capacity)
        profiles.append(r[0]);scalars.append(np.column_stack([
            g*(quantum.shift.sum()-r[1]/(b*g))/model.n,(r[2]/model.n)**2]))
        maximum=max(maximum,int(r[-1]))
    profiles=np.asarray(profiles);scalars=np.asarray(scalars)
    mean,se,ess,rh,batches=statistics(profiles)
    multiplier=float(student_t.ppf(1-.05/(2*mean.size),chains*16-1));radius=multiplier*se
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
        fixed_directed_loops_per_sweep=8,worldline_cluster_updates_per_sweep=1)
    for key in ('max_correlation_rhat','max_scalar_rhat'):
        if not np.isfinite(diag[key]):diag[key]=None
    reference=exact_gibbs(quantum,b*g,seeds)['gamma'] if model.n<=10 else None
    if reference is not None:
        diag['gamma_linf_vs_exact']=float(abs(mean-reference).max())
        diag['exact_outside_interval_fraction']=float(np.mean(abs(mean-reference)>radius+1e-12))
    return dict(params=params,diagnostics=diag,gamma=mean,se=se,radius=radius,
        batches=batches,chain_means=profiles.mean(axis=1),scalar_series=scalars,reference=reference)
