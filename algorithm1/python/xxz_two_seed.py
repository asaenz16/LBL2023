"""Paired one/two-seed ZZ inference and additive XXZ sampling extensions.

Correlations retain the repository Gamma=<ZZ>/4 convention. New parameter
rules and metrics live here so the previous frozen samplers are unchanged.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.stats import t as student_t
from xxz_image_gibbs import parameterize as legacy_parameterize,starts,statistics
from xxz_sse_cluster import run_cluster
from xxz_glauber import graph,exact_gibbs
from xxz_sse_fast import expansion_cutoff
from ising_monte_carlo import signed_image
from ising_seed_sampling import setup_sampler,sample_spins,exact_correlations
from weizmann_petz_metrics import mask_metrics
from segmentation_bpm import bpm_metrics

DECISIONS=('one_background','two_max','one_foreground')


def parameterize(image,rule):
    if rule['threshold']!='q90':return legacy_parameterize(image,rule)
    # q90 extends only the threshold; all other physical conventions match v2.
    raw=signed_image(image,.001);d=raw.distances
    tau=max(float(np.quantile(d[d>0],.9)) if np.any(d>0) else 125.,1.)
    spread=max(float(np.diff(np.quantile(d,[.1,.9]))[0]),1.)
    alpha={'fixed001':.001,'fixed01':.01,'inverse_spread':1/spread}[rule['sensitivity']]
    model=signed_image(image,alpha,threshold=tau);scale=float(np.median(abs(model.delta)))
    fallback=scale<1e-8
    if fallback:scale=max(float(np.mean(abs(model.delta))),.001)
    g=rule['strength']*scale;b=4*rule['K']/scale
    if rule['transverse']!='relative' or rule['temperature']!='longitudinal':raise ValueError('q90 uses relative exchange and longitudinal normalization.')
    return model,dict(threshold=tau,alpha=alpha,edge_spread=spread,median_abs_delta=scale,
        normalization_fallback=fallback,g=g,g_over_scale=g/scale,b=b,beta_paper=b*model.n,
        ferromagnetic_fraction=float(np.mean(model.delta>g)),intermediate_fraction=float(np.mean(abs(model.delta)<=g)),
        antiferromagnetic_fraction=float(np.mean(model.delta < -g)))


def scores(gamma):
    a=np.asarray(gamma)
    if a.shape[-2]!=2:raise ValueError('Expected background and foreground seed axes.')
    return np.stack((-a[...,0,:],a[...,1,:]-a[...,0,:],a[...,1,:]),axis=-2)


def paired_inference(gamma,batches,chain_means):
    """Approximate simultaneous t intervals; retains paired seed covariance.

    The maximum of batch and between-chain SE is used, as in the old engine.
    These are MCMC error estimates, not finite-time coverage guarantees.
    """
    point=scores(gamma);batch=scores(batches);chains=scores(chain_means)
    count,blocks=batch.shape[:2]
    se=np.maximum(batch.reshape(count*blocks,*point.shape).std(axis=0,ddof=1)/np.sqrt(count*blocks),
        chains.std(axis=0,ddof=1)/np.sqrt(count))
    multiplier=float(student_t.ppf(1-.05/(2*point.size),count*blocks-1))
    return point,se,multiplier*se,multiplier


def decision_metrics(score,radius,truth,seeds,converged=True):
    truth=np.asarray(truth,bool);t=truth.ravel();score=np.asarray(score)
    pred=score>1e-12;resolved=abs(score)>np.maximum(radius,1e-12)
    unknown=np.ones(t.size,bool);unknown[np.asarray(seeds)]=False
    correct=pred==t
    tp=int(np.sum(pred&resolved&t));fp=int(np.sum(pred&resolved&~t))
    lower_den=int(t.sum()+fp+np.sum(~resolved&~t));upper_den=int(t.sum()+fp)
    lower=tp/lower_den if lower_den else 1.;upper=(tp+np.sum(~resolved&t))/upper_den if upper_den else 1.
    return dict(**mask_metrics(pred,t),**bpm_metrics(pred.reshape(truth.shape),truth),
        unresolved=float((~resolved)[unknown].mean()),correct_resolved=float((correct&resolved)[unknown].mean()),
        wrong_resolved=float((~correct&resolved)[unknown].mean()),unknown_iou=mask_metrics(pred[unknown],t[unknown])['iou'],
        iou_lower=float(lower),iou_upper=float(upper),mixing_lower=float(lower if converged else 0.),
        mixing_upper=float(upper if converged else 1.),mixing_unresolved=float((~resolved)[unknown].mean() if converged else 1.),
        seed_accuracy=float(correct[np.asarray(seeds)].mean()))


def evaluate_arrays(arrays,truth,seeds,converged=True):
    point,se,radius,multiplier=paired_inference(arrays['gamma'],arrays['batches'],arrays['chain_means'])
    result={d:decision_metrics(point[i],radius[i],truth,seeds,converged) for i,d in enumerate(DECISIONS)}
    if 'reference' in arrays:
        exact={d:decision_metrics(s,np.zeros_like(s),truth,seeds) for d,s in zip(DECISIONS,scores(arrays['reference']))}
    else:exact=None
    return result,exact,dict(score=point,se=se,radius=radius,multiplier=multiplier)


def sample(image,rule,seeds,*,burn,samples,thin,chains,rng_seed):
    model,params=parameterize(image,rule);g=params['g'];b=params['b'];profiles=[];scalars=[]
    if g:
        quantum=graph(model.n,model.edges,model.delta/g);capacity,logtail=expansion_cutoff(quantum,b*g);maximum=0
        for c,initial in enumerate(starts(image,chains,rng_seed)):
            r=run_cluster(quantum,initial,b*g,burn+1+thin*np.arange(samples),seeds,
                seed=(rng_seed+7919*c)%(2**31-1),capacity=capacity)
            profiles.append(r[0]);scalars.append(np.column_stack([g*(quantum.shift.sum()-r[1]/(b*g))/model.n,(r[2]/model.n)**2]))
            maximum=max(maximum,int(r[-1]))
        reference=exact_gibbs(quantum,b*g,seeds)['gamma'] if model.n<=10 else None
    else:
        sampler=setup_sampler(image.shape,model.edges,model.delta,b)
        reference=exact_correlations(sampler,seeds)/4
        for c in range(chains):
            z=sample_spins(sampler,samples,np.random.default_rng(rng_seed+7919*c))
            profiles.append(z[:,np.asarray(seeds),None]*z[:,None,:]/4)
            e=-(z[:,model.edges[:,0]]*z[:,model.edges[:,1]])@model.delta/4/model.n
            scalars.append(np.column_stack([e,(z.mean(axis=1))**2]))
        capacity=0;logtail=None;maximum=0
    profiles=np.asarray(profiles);scalars=np.asarray(scalars)
    mean,se,ess,rh,batches=statistics(profiles,blocks=32)
    sm,sse,sess,srh,_=statistics(scalars,blocks=32)
    _,dse,dess,drh,_=statistics(scores(profiles),blocks=32)
    varying=np.std(profiles,axis=(0,1))>1e-12;varying_d=np.std(scores(profiles),axis=(0,1))>1e-12
    max_rhat=max(float(rh.max()),float(srh.max()),float(drh.max()))
    diag=dict(max_rhat=max_rhat if np.isfinite(max_rhat) else None,
        max_correlation_rhat=float(rh.max()) if np.isfinite(rh.max()) else None,
        max_score_rhat=float(drh.max()) if np.isfinite(drh.max()) else None,
        max_scalar_rhat=float(srh.max()) if np.isfinite(srh.max()) else None,
        minimum_correlation_ess=float(ess[varying].min()) if varying.any() else float(chains*samples),
        minimum_score_ess=float(dess[varying_d].min()) if varying_d.any() else float(chains*samples),
        converged=bool(max_rhat<1.05),energy_per_spin=float(sm[0]),magnetization_squared=float(sm[1]),
        capacity=capacity,maximum_order=maximum,log_tail_bound=logtail,
        method='SSE with directed loops and whole-worldline clusters' if g else 'iid exact frontier Gibbs',
        burn=burn if g else 0,samples_per_chain=samples,thin=thin if g else 1,chains=chains)
    arrays=dict(gamma=mean,se=se,batches=batches,chain_means=profiles.mean(axis=1),score_rhat=drh,score_ess=dess,
        scalar_series=scalars)
    if reference is not None:
        arrays['reference']=reference;diag['gamma_linf_vs_exact']=float(abs(mean-reference).max())
    return arrays,params,diag
