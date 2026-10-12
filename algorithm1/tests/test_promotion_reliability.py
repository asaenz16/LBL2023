import inspect
import numpy as np
import pytest
from iterative_seed_gibbs import RoundMoments,sample_rounds
from iterative_seed_readout import trajectory as old_trajectory,score_metrics
from promotion_reliability import *
from promotion_reliability_bank import iter_tail


def test_grid():
    assert len(CONFIGS)==250 and len(POLICIES)==2250
    assert len({tuple(p.items()) for p in POLICIES})==2250


@pytest.mark.parametrize('norm',['relative','absolute'])
def test_weighted_moments_and_leave_one_out(norm):
    rng=np.random.default_rng(34);z=rng.choice([-1,1],(4,512,7));mm=RoundMoments(z);ev=Evidence(mm)
    ids=[0,1,3,5];roles=[1,1,-1,-1];weights=[1.,.3,.5,.7];w=coefficients(roles,weights,norm)
    inf,loo=ev.infer(ids,roles,weights,norm)
    obs=z*np.sum(z[:,:,ids]*w,axis=-1)[:,:,None]
    np.testing.assert_allclose(inf['score'],obs.mean((0,1)),atol=1e-14)
    for i,p in enumerate(ids):
        keep=np.arange(len(ids))!=i;a=coefficients(np.array(roles)[keep],np.array(weights)[keep],norm)
        y=z[:,:,p]*np.sum(z[:,:,np.array(ids)[keep]]*a,axis=-1);b=y.reshape(4,16,32).mean(2)
        se=max(b.std(ddof=1)/8,b.mean(1).std(ddof=1)/2)
        np.testing.assert_allclose(loo['score'][i],y.mean(),atol=1e-14)
        np.testing.assert_allclose(loo['radius'][i],student_t.ppf(1-.05/(2*7*10),63)*se,atol=1e-14)


def test_lone_anti_seed_normalization():
    np.testing.assert_array_equal(coefficients([1,-1],[1.,.1],'relative'),[1.,-1.])
    np.testing.assert_array_equal(coefficients([1,-1],[1.,.1],'absolute'),[1.,-.1])


def test_spatial_exclusion_and_roles():
    same=spatial_mask((8,8),[0,63],[1,-1],1,'same_250');all_=spatial_mask((8,8),[0,63],[1,-1],1,'all_250')
    assert not same[1] and not same[2] and same[3] and same[63]
    assert not all_[63] and not all_[62]


def test_maintenance_anchor_and_persistence():
    loo=dict(score=np.array([-1.,-.9,.8]),radius=np.array([.1,.1,.1]),converged=np.ones(3,bool))
    ids=[0,1,2];roles=[1,1,-1];weights=[1.,1.,1.]
    w,retired,strikes,_=maintain(ids,roles,weights,0,loo,'remove1',{})
    assert retired==[1,2] and w=={0:1.}
    w,retired,strikes,_=maintain(ids,roles,weights,0,loo,'remove2',{})
    assert retired==[]
    assert maintain(ids,roles,weights,0,loo,'remove2',strikes)[1]==[1,2]
    w,retired,_,_=maintain(ids,roles,weights,0,loo,'soft_absolute',{})
    assert w[0]==1. and w[1]==.55 and w[2]==.55 and not retired


def test_consensus_and_self_correlation_exclusion():
    rng=np.random.default_rng(47);z=rng.choice([-1,1],(4,512,6));z[:,:,1]=z[:,:,0];z[:,:,2]=-z[:,:,0]
    ev=Evidence(RoundMoments(z));f,b=ev.votes([0,2],[1,-1],1.)
    assert f[0] and f[1] and b[2]
    _,loo=ev.infer([0,1],[1,-1],[1.,1.],'relative')
    assert loo['score'][1]==1. # the erroneous anti-seed cannot validate itself


def toy():
    rng=np.random.default_rng(333);z=rng.choice([-1,1],(10,4,256,6)).astype(np.int8)
    z[:,:,:,1]=z[:,:,:,0];z[:,:,:,2]=-z[:,:,:,0]
    return np.packbits(z>0,axis=-1,bitorder='little'),[dict(scalar_rhat=1.)]*10


def test_truth_independence_and_prefix_causality():
    packed,diag=toy();truth=np.array([[1,1,0],[0,1,0]],bool);base=dict(start='foreground',cut_fg=.25,cut_bg=.25,gate='confidence')
    policy=dict(space='none',agreement='consensus75_temporal2',maintenance='soft_absolute',cut='base',cap=6,stopping='none')
    a,_,ta=evaluate(packed,diag,truth,[2,0],base,[policy],2,'fixture',True)
    _,_,tb=evaluate(packed,diag,~truth,[2,0],base,[policy],2,'fixture',True)
    for x,y in zip(ta,tb):
        assert x['prediction']==y['prediction'] and x['stop']==y['stop']
        assert [r['added'] for r in x['history']]==[r['added'] for r in y['history']]
        assert all(r['weights'][0]==1. for r in x['history'])
        assert x['history'][0]['added']==[-1,-1]
    changed=packed.copy();changed[6:]=np.random.default_rng(52).integers(0,256,changed[6:].shape,dtype=np.uint8)
    b,_,_=evaluate(changed,diag,truth,[2,0],base,[policy],2,'fixture')
    np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('g',[0.,.1])
def test_tail_preserves_original_round_schedule(g):
    image=np.array([[10.,10.,160.],[20.,25.,180.]]);rule=dict(threshold='q90',sensitivity='inverse_spread',K=.6,transverse='relative',strength=g,temperature='longitudinal')
    args=dict(rounds=3,samples=128,burn=50,thin=2,chains=4,rng_seed=625)
    full,params,diag=sample_rounds(image,rule,**args)
    tail=list(iter_tail(image,rule,**args,start_round=1))
    for i,(z,p,d) in enumerate(tail):
        np.testing.assert_array_equal(z,full[i+1]);assert p==params and d==diag[i+1]


def test_stopping_minimum_audit():
    h=[dict(terminal='all_resolved',maintenance_pending=False,unresolved_count=0) for _ in range(10)]
    assert endpoint(h,dict(cap=10,stopping='none'))==(0,'all_resolved')
    assert endpoint(h,dict(cap=10,stopping='audit3'))==(2,'all_resolved')


def test_stable_semantic_error_is_not_a_correlation_contradiction():
    # The second pixel's image label is background, but the model aligns it
    # exactly with the trusted foreground pixel. Fresh same-model evidence
    # cannot discover this semantic mistake through consistency alone.
    rng=np.random.default_rng(56);z=rng.choice([-1,1],(4,512,6));z[:,:,1]=z[:,:,0]
    ev=Evidence(RoundMoments(z));_,loo=ev.infer([0,1],[1,1],[1.,1.],'relative')
    weights,retired,_,_=maintain([0,1],[1,1],[1.,1.],0,loo,'remove1',{})
    assert loo['score'][1]==1 and not retired and weights[1]==1.


def test_planted_inconsistent_seed_detected_without_self_vote():
    rng=np.random.default_rng(662);q=np.array([1,1,1,-1,-1,-1]);global_spin=rng.choice([-1,1],(4,2048,1))
    noise=np.where(rng.random((4,2048,6))<.02,-1,1);z=global_spin*q*noise
    ids=list(range(6));roles=[1,1,1,-1,-1,1] # deliberate wrong label on index5
    ev=Evidence(RoundMoments(z));_,loo=ev.infer(ids,roles,[1.]*6,'relative')
    weights,removed,_,_=maintain(ids,roles,[1.]*6,0,loo,'remove1',{})
    assert removed==[5] and len(weights)==5


def test_unused_round_elision_preserves_all_three_round_policies():
    packed,diag=toy();truth=np.array([[1,1,0],[0,1,0]],bool);base=dict(start='foreground',cut_fg=.25,cut_bg=.25,gate='confidence')
    policies=[p for p in POLICIES if p['cap']==3]
    a,r,ta=evaluate(packed,diag,truth,[2,0],base,policies,1,'prefix-elision',True)
    b,s,tb=evaluate(packed[:3],diag[:3],truth,[2,0],base,policies,1,'prefix-elision',True)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(r,s);assert ta==tb
