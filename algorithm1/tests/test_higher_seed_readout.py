import itertools
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import pytest
from higher_seed_readout import (triple_functions, select_triple, pair_coefficients,
    Evidence, infer_observations, observation_moments, trajectory, evaluate, CONFIGS, POLICIES, METRICS)


def enumeration():
    z=np.array(list(itertools.product((-1.,1.),repeat=5)))
    rng=np.random.default_rng(12);j=rng.normal(size=(5,5));j=(j+j.T)/2
    p=np.exp(.13*np.einsum('si,ij,sj->s',z,j,z));p/=p.sum()
    return z,p


@pytest.mark.parametrize('eta',[.25,.5,.75,.9,1.])
def test_soft_matches_exact_conditional_distribution(eta):
    z,p=enumeration();roles={0:1,1:-1,2:1};t=(0,1,2)
    a,b=triple_functions(z[None],t,roles,0,'soft',eta)
    w=(1+z[:,0])*(1-eta*z[:,1])*(1+eta*z[:,2])
    direct=(p*w*z[:,3]).sum()/(p*w).sum()
    symmetric=(p*z[:,3]*a[0]).sum()/(p*b[0]).sum()
    # Production feature weights intentionally use float32, below sampling error.
    assert symmetric==pytest.approx(direct,abs=3e-8)


def test_majority_is_four_point_and_flip_even():
    z,p=enumeration();roles={0:-1,1:1,2:-1}
    a,b=triple_functions(z[None],(0,1,2),roles,0,'majority',0.)
    u=z[:,:3]*np.array([-1,1,-1]);expanded=(u.sum(1)-u.prod(1))/2
    np.testing.assert_array_equal(a[0],expanded)
    assert (p*z[:,0]*z[:,1]*z[:,2]).sum()==pytest.approx(0,abs=1e-12)
    assert (p*z[:,3]*a[0]).sum()==pytest.approx((p*z[:,3]*expanded).sum())


def test_sampling_error_uses_paired_denominator_influence():
    rng=np.random.default_rng(93);z=rng.choice([-1.,1.],(4,512,5)).astype('f')
    a,b=triple_functions(z,(0,1,2),{0:1,1:1,2:-1},0,'soft',.9)
    d=b.mean();num=z[...,3]*a;m=num.mean()/d
    influence=num/d-m*(b/d-1)
    result=infer_observations(influence[...,None])
    assert result['score'][0]==pytest.approx(m,abs=1e-7)
    # Independent influence estimate; numerator-only SE would be different.
    block=influence.reshape(4,16,32).mean(2)
    expected=max(block.std(ddof=1)/8,block.mean(1).std(ddof=1)/2)
    assert result['se'][0]==pytest.approx(expected,rel=1e-6)


def test_triples_exclude_target_and_anchor_when_requested():
    priority=np.array([.7,.2,.9,.3,.4])
    assert select_triple(range(5),0,'anchor',priority)==(0,1,3)
    assert select_triple(range(5),0,'anchor',priority,exclude=0)==(1,3,4)
    assert select_triple(range(3),0,'anchor',priority,exclude=2)==()


def test_pair_normalization_and_leave_one_out():
    np.testing.assert_allclose(pair_coefficients([1,1,-1]),[.25,.25,-.5])
    rng=np.random.default_rng(1);z=rng.choice([-1,1],(4,512,4))
    e=Evidence(np.packbits(z>0,axis=-1,bitorder='little'),4,1.,512,np.arange(4))
    obs=e.pair_observations((0,1,2),{0:1,1:1,2:-1},0)
    np.testing.assert_allclose(obs[...,1],z[...,1]*(z[...,0]-z[...,2])/2)
    np.testing.assert_allclose(obs[...,2],z[...,2]*(z[...,0]+z[...,1])/2)


def test_matched_linear_control_distinct_from_majority():
    z,_=enumeration();r={0:1,1:1,2:1}
    a,_=triple_functions(z[None],(0,1,2),r,0,'majority',0)
    b,_=triple_functions(z[None],(0,1,2),r,0,'linear',0)
    assert np.any(a!=b)


def test_full_replay_deterministic_and_truth_not_used_for_promotions():
    rng=np.random.default_rng(4);z=rng.choice([-1,1],(3,4,256,9))
    # One partially correlated group; other pixels remain noisy/unresolved.
    z[...,1]=z[...,0];z[...,2]=-z[...,0]
    packed=np.packbits(z>0,axis=-1,bitorder='little');diag=[{'scalar_rhat':1.}]*3
    policy=dict(next(c for c in CONFIGS if c['family']=='soft_score'),cap=3,stopping='none')
    truth=np.zeros((3,3),bool);truth.flat[0]=True
    x=evaluate(packed,diag,truth,0,'foreground',[policy],1,31,256,True)
    y=evaluate(packed,diag,~truth,0,'foreground',[policy],1,31,256,True)
    assert x[2][0]['history']==y[2][0]['history']
    np.testing.assert_array_equal(x[1],y[1])
    history=x[2][0]['history']
    for h in history:
        assert len([i for i in h['added'] if i>=0])<=2
    rounds=x[0][0,0,METRICS.index('rounds')]
    assert x[0][0,0,METRICS.index('shots')]==rounds*4*256


def test_grid_has_unique_policies_and_four_requested_families():
    assert len({tuple(sorted(p.items())) for p in POLICIES})==len(POLICIES)
    assert {p['family'] for p in POLICIES}=={'pair','majority_gate','soft_gate','majority_score','soft_score'}


def test_fast_moments_match_direct_float64_statistics():
    obs=np.random.default_rng(77).normal(size=(4,512,15)).astype('f')
    mean,second=observation_moments(obs,16)
    q=obs.astype('d').reshape(4,16,32,15)
    np.testing.assert_allclose(mean,q.mean(2),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(second,(q*q).mean(2),rtol=1e-12,atol=1e-12)
    inf=infer_observations(obs)
    within=obs.astype('d').reshape(8,256,15).var(1,ddof=1).mean(0)
    between=256*obs.astype('d').reshape(8,256,15).mean(1).var(0,ddof=1)
    expected=np.sqrt(255/256+between/(256*within))
    assert inf['max_rhat']==pytest.approx(max(1.,expected.max()),abs=1e-12)


def test_pure_four_spin_signal_survives_when_all_pairs_vanish():
    # A spin-flip symmetric distribution with a genuine fourth-order signal.
    z=np.array(list(itertools.product((-1.,1.),repeat=4)))
    prob=(1+.6*z.prod(1))/16
    for a,b in itertools.combinations(range(4),2):
        assert (prob*z[:,a]*z[:,b]).sum()==pytest.approx(0,abs=1e-12)
    roles={0:1,1:1,2:1}
    a,b=triple_functions(z[None],(0,1,2),roles,0,'majority',0)
    assert (prob*z[:,3]*a[0]).sum()==pytest.approx(-.3)
    a,b=triple_functions(z[None],(0,1,2),roles,0,'soft',.5)
    assert (prob*z[:,3]*a[0]).sum()/(prob*b[0]).sum()==pytest.approx(.15)


@pytest.mark.parametrize('method',['majority','soft'])
def test_reversing_seed_roles_reverses_readout(method):
    z,_=enumeration();roles={0:1,1:-1,2:1}
    a,b=triple_functions(z[None],(0,1,2),roles,0,method,.75)
    aa,bb=triple_functions(z[None],(0,1,2),{k:-v for k,v in roles.items()},0,method,.75)
    np.testing.assert_allclose(a,-aa);np.testing.assert_allclose(b,bb)
