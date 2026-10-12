from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ising_seed_sampling import (setup_sampler,exact_correlations,sample_spins,sufficient_counts,
    parameterize,FUNCTIONS,classify_counts,segmentation_metrics,exact_decision,_mean_spins)
from ising_algorithm1 import grid_edges
from ising_grid_transfer import exact_grid_gibbs


def enumeration(shape,edges,j,b,seeds):
    n=np.prod(shape);z=1-2*((np.arange(1<<n)[:,None]>>np.arange(n))&1)
    logp=b/4*(z[:,edges[:,0]]*z[:,edges[:,1]])@j
    p=np.exp(logp-logp.max());p/=p.sum()
    return p,z,np.einsum('d,ds,dn->sn',p,z[:,seeds],z)


@pytest.mark.parametrize('shape',[(1,5),(5,1),(2,4),(4,2),(3,3)])
def test_frontier_and_forced_seed_match_enumeration(shape):
    edges=grid_edges(*shape);j=np.random.default_rng(572).normal(size=len(edges));b=3.1;seeds=[0,np.prod(shape)-1]
    p,z,expected=enumeration(shape,edges,j,b,seeds)
    sampler=setup_sampler(shape,edges,j,b)
    assert_allclose(exact_correlations(sampler,seeds),expected,atol=2e-13)
    assert_allclose(_mean_spins(sampler['probabilities'],sampler['width']),0,atol=2e-14)
    # Multiplying the stored sequential conditionals reproduces every probability.
    actual=[]
    for config in z:
        state=0;prob=1.
        for k,site in enumerate(sampler['order']):
            q=sampler['probabilities'][k,state];minus=config[site]<0
            prob*=1-q if minus else q;bit=1<<(k%sampler['width']);state=(state&~bit)|(bit if minus else 0)
        actual.append(prob)
    assert_allclose(actual,p,atol=2e-14)


def test_independent_samples_and_max_correlation_identity():
    shape=(3,3);edges=grid_edges(*shape);j=np.random.default_rng(8).normal(size=len(edges));sampler=setup_sampler(shape,edges,j,2.)
    seeds=[0,8];samples=sample_spins(sampler,100000,np.random.default_rng(2));counts=sufficient_counts(samples,seeds,[100000])[0]
    c=np.array([(samples*samples[:,s,None]).mean(axis=0) for s in seeds]);expected=exact_correlations(sampler,seeds)
    assert_allclose(c,expected,atol=.012)
    assert_allclose(c[1]-c[0],2*(2*counts[1]-counts[3])/100000,atol=2e-16)
    assert np.array_equal(c[1]>c[0]+1e-12,2*counts[1]>counts[3])


@pytest.mark.parametrize('function',FUNCTIONS)
def test_growth_functions_preserve_sign_and_match_old_transfer(function):
    image=np.array([[0.,20.,250.],[30.,60.,130.],[10.,25.,200.]])
    rule=dict(function=function,threshold=dict(kind='fixed',value=50.),sensitivity='a01',K=.8)
    edges,j,p=parameterize(image,rule)
    d=abs(image.ravel()[edges[:,0]]-image.ravel()[edges[:,1]])
    assert np.array_equal(np.sign(j),np.sign(50-d))
    sampler=setup_sampler(image.shape,edges,j,p['b']);c=exact_correlations(sampler,[0,8])
    model=SimpleNamespace(n=image.size,b=p['b'],graph=SimpleNamespace(image=image,edges=edges,delta=j))
    assert_allclose(c,4*exact_grid_gibbs(model,[0,8]),atol=1e-12)


def test_extreme_bonds_do_not_overflow_log_sampling():
    shape=(2,3);edges=grid_edges(*shape);j=np.array([1e5,-1e5,2.,-3.,4.,1e5,-1e5])
    sampler=setup_sampler(shape,edges,j,4.);_,_,expected=enumeration(shape,edges,j,4.,[0,5])
    assert np.isfinite(sampler['probabilities']).all()
    assert_allclose(exact_correlations(sampler,[0,5]),expected,atol=1e-9)


def test_binomial_winner_resolution_and_no_disagreement():
    counts=np.array([[0,2000,1000],[0,0,0],[0,2000,1000],[0,0,0]])
    pred,resolved=classify_counts(counts,2000,'two_max');assert not pred.any() and not resolved.any()
    pred,resolved=classify_counts(counts,2000,'one_background');assert pred.tolist()==[False,True,False]
    assert resolved.tolist()==[True,True,False]
    counts[1]=[0,1000,500];counts[3]=1000
    pred,resolved=classify_counts(counts,2000,'two_max');assert pred.tolist()==[False,True,False]
    assert resolved.tolist()==[True,True,False]


def test_uncertain_label_iou_bounds_and_excluded_seeds():
    truth=np.array([0,1,0,1],bool);m=segmentation_metrics([0,1,1,0],[1,1,0,0],truth,[0,1])
    assert m['unresolved']==1 and m['iou_lower']==1/3 and m['iou_upper']==1
    corr=np.array([[1,.3,.2,-.1],[.3,1,.4,-.05]])
    assert exact_decision(corr,'two_max')[0].tolist()==[False,True,True,True]


def test_exact_binomial_interval_coverage():
    from scipy.stats import binom
    from ising_seed_sampling import binomial_interval
    for n in (1,10,100):
        k=np.arange(n+1);lo,hi=binomial_interval(k,np.full(n+1,n),.005)
        for p in (0.,.01,.25,.5,.9,1.):
            coverage=binom.pmf(k,n,p)[(lo<=p)&(p<=hi)].sum()
            assert coverage>=.99-1e-12


def test_parameter_scan_is_unique_and_linear_sensitivity_is_not_redundant():
    from run_weizmann_ising_seeds import candidates,identifier
    rules=candidates()
    assert len(rules)==len({identifier(r) for r in rules})==351
    assert {r['sensitivity'] for r in rules if r['function']=='linear'}=={'adaptive'}
