import numpy as np
import pytest
from iterative_seed_gibbs import RoundMoments,candidates,stop_index,sample_rounds,unpack,parameterize
from iterative_seed_readout import trajectory,score_metrics,BASES,POLICIES,pooled_inference,RequiredRounds,evaluate_grid
from xxz_image_gibbs import statistics
from xxz_glauber import graph,exact_gibbs


@pytest.mark.parametrize('fg,bg', [([0],[]),([],[1]),([0,2],[1]),([0,2,3],[1,4])])
def test_moments_equal_direct_samples(fg,bg):
    z=np.random.default_rng(911).choice([-1,1],(4,512,7)).astype(np.int8)
    mm=RoundMoments(z);out=mm.infer(fg,bg)
    q=(z[:,:,fg].mean(axis=-1) if fg else 0)-(z[:,:,bg].mean(axis=-1) if bg else 0)
    y=z*q[:,:,None];mean,se,ess,rh,batches=statistics(y,blocks=16)
    np.testing.assert_allclose(out['score'],mean,atol=1e-14)
    np.testing.assert_allclose(out['se'],se,atol=1e-14)
    np.testing.assert_allclose(out['rhat'],rh,atol=1e-12)
    np.testing.assert_allclose(out['ess'],ess,atol=1e-9)


def test_seed_only_background_mirror():
    z=np.random.default_rng(76).choice([-1,1],(4,128,6))
    mm=RoundMoments(z);a=mm.infer([0],[]);b=mm.infer([],[0])
    np.testing.assert_array_equal(a['score'],-b['score'])
    np.testing.assert_array_equal(a['radius'],b['radius'])


def test_candidate_eligibility_and_disjoint_roles():
    inf=dict(score=np.array([1.,.8,.6,-.9,-.7,.1]),radius=np.array([0.,.1,.2,.1,.3,.2]),converged=True)
    a,b=candidates(inf,[0],[],.5,.5,'confidence');assert a.tolist()==[1] and b.tolist()==[3]
    a,b=candidates(inf,[0],[3],.5,.5,'point_resolved');assert a.tolist()==[1,2] and b.tolist()==[4]
    inf['converged']=False;assert all(len(x)==0 for x in candidates(inf,[0],[],.5,.5,'confidence'))


def test_stopping_and_no_candidates():
    assert stop_index([9,9,9,8],['']*4,10,'same_count',63)==(2,'stable_count')
    assert stop_index([9,8,8,10],['']*4,10,'no_improvement',63)==(3,'no_improvement')
    assert stop_index([9,8,7,6],['']*4,3,'none',63)==(2,'round_cap')
    assert stop_index([9,8],['','no_candidates'],10,'none',63)==(1,'no_candidates')
    assert stop_index([9,0],['','all_resolved'],10,'none',63)==(1,'all_resolved')


def test_promoted_pixels_not_removed_from_unresolved():
    inf=dict(score=np.array([1.,0.,.8,-.8]),radius=np.array([0.,.1,.1,.1]),converged=True,minimum_ess=1000.,max_rhat=1.)
    m=score_metrics(inf,np.array([[1,1],[1,0]],bool),0)
    assert m['unresolved']==pytest.approx(1/3)


def test_pooled_control_matches_concatenated_ensembles():
    rng=np.random.default_rng(82);z=rng.choice([-1,1],(3,4,128,6))
    mm=[RoundMoments(x) for x in z];actual=pooled_inference(mm,[0],[])
    expected=RoundMoments(z.reshape(12,128,6)).infer([0],[])
    for k in ('score','se','radius','rhat','ess'):np.testing.assert_allclose(actual[k],expected[k],atol=1e-12)


@pytest.mark.parametrize('g',[0.,.1,.5])
def test_new_snapshot_sampler_against_exact_gibbs(g):
    image=np.array([[10.,10.,160.],[20.,25.,180.]])
    rule=dict(threshold='q90',sensitivity='inverse_spread',K=.6,transverse='relative',strength=g,temperature='longitudinal')
    packed,params,diag=sample_rounds(image,rule,rounds=2,samples=4096,burn=1200,thin=2,chains=4,rng_seed=721)
    assert packed.shape==(2,4,4096,1)
    z=unpack(packed,6);assert np.isin(z,[-1,1]).all();model,_=parameterize(image,rule)
    # graph uses unit XX exchange; g=0 reference is computed as a diagonal H.
    if g:
        q=graph(6,model.edges,model.delta/params['g']);reference=4*exact_gibbs(q,params['b']*params['g'],np.arange(6))['gamma']
    else:
        states=1-2*((np.arange(64)[:,None]>>np.arange(6))&1)
        energy=-np.sum(states[:,model.edges[:,0]]*states[:,model.edges[:,1]]*model.delta/4,axis=1)
        weights=np.exp(-params['b']*(energy-energy.min()));weights/=weights.sum()
        reference=np.einsum('s,si,sj->ij',weights,states,states)
    zz=z.reshape(-1,6).astype(float);empirical=zz.T@zz/len(zz)
    assert np.max(abs(empirical-reference))<.06
    assert all(d['maximum_order']<=d['capacity'] for d in diag)


def test_grid_cross_product():
    assert len(BASES)==36 and len(POLICIES)==432
    assert len({tuple(sorted(p.items())) for p in POLICIES})==432


def toy_bank():
    rng=np.random.default_rng(993);z=rng.choice([-1,1],(10,4,256,6)).astype(np.int8)
    z[:,:,:,1]=z[:,:,:,0];z[:,:,:,2]=-z[:,:,:,0];z[:,:,:,4]=z[:,:,:,0]
    return np.packbits(z>0,axis=-1,bitorder='little'),[dict(scalar_rhat=1.)]*10


def test_lazy_generation_matches_full_grid_prefixes():
    bank,diag=toy_bank();policies=[dict(BASES[0],cap=cap,stopping=stop) for cap in (3,6,10) for stop in ('none','same_count')]
    guard=RequiredRounds([2,0],policies,2,'toy',6)
    for k,(z,d) in enumerate(zip(bank,diag)):
        if not guard.consume(z,d):break
    truth=np.array([[1,1,0],[0,1,0]],bool)
    a,ar,_=evaluate_grid(bank,diag,truth,[2,0],2,'toy',policies)
    b,br,traces=evaluate_grid(bank[:k+1],diag[:k+1],truth,[2,0],2,'toy',policies,save_traces=True)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ar,br)
    for trace in traces:
        for row in trace['history']:
            assert len(set(row['foreground'])&set(row['background']))==0
            assert len([i for i in row['added'] if i>=0])<=2


def test_truth_cannot_change_promotions():
    bank,diag=toy_bank();policies=[dict(BASES[0],cap=6,stopping='none')]
    truth=np.array([[1,1,0],[0,1,0]],bool)
    _,_,a=evaluate_grid(bank,diag,truth,[2,0],2,'toy',policies,save_traces=True)
    changed=truth.copy();changed.ravel()[[1,3,4,5]]=~changed.ravel()[[1,3,4,5]]
    _,_,b=evaluate_grid(bank,diag,changed,[2,0],2,'toy',policies,save_traces=True)
    for x,y in zip(a,b):
        assert x['prediction']==y['prediction'] and x['stop']==y['stop']
        assert [r['added'] for r in x['history']]==[r['added'] for r in y['history']]
