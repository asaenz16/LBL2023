import numpy as np
import pytest
from segmentation_bpm import boundary_proximity
from xxz_two_seed import parameterize,scores,paired_inference,evaluate_arrays,sample
from xxz_image_gibbs import parameterize as old_parameterize


def test_bpm_translated_thin_crack():
    g=np.zeros((9,9),bool);g[2:7,3]=True;p=np.roll(g,1,axis=1)
    assert boundary_proximity(p,g,0)['iou']==0
    assert boundary_proximity(p,g,1)['iou']==1
    assert boundary_proximity(p,g,1)['f1']==1
    assert boundary_proximity(np.roll(g,2,axis=1),g,1)['iou']==0


def test_bpm_nonnegative_and_empty_cases():
    a=np.zeros((7,7),bool);b=a.copy();b[3,3]=True
    assert boundary_proximity(a,a)['iou']==1
    assert boundary_proximity(a,b)['iou']==0
    assert boundary_proximity(b,a)['iou']==0
    with pytest.raises(ValueError):boundary_proximity(a,b,-1)


def test_bpm_asymmetry_is_explicit():
    g=np.zeros((9,9),bool);p=g.copy();g[4,2]=True;g[4,7]=True;p[3:6,2]=True
    a=boundary_proximity(p,g,1);b=boundary_proximity(g,p,1)
    assert a['iou']==pytest.approx(3/4)
    assert b['iou']==pytest.approx(1/2)
    assert a['symmetric_f1']==pytest.approx(b['symmetric_f1'])


def test_paired_errors_cancel_shared_noise():
    rng=np.random.default_rng(3);common=rng.normal(0,.01,(4,32,6))
    batches=np.stack([common+.05,common+.07],axis=2)
    point,se,radius,m=paired_inference(batches.mean(axis=(0,1)),batches,batches.mean(axis=1))
    assert np.max(se[1])<1e-15 and np.min(se[0])>1e-5
    assert np.allclose(point[1],.02)


def test_same_profiles_give_ties_and_unresolved():
    rng=np.random.default_rng(4);one=rng.normal(0,.02,(4,32,6));b=np.stack([one,one],axis=2)
    arrays=dict(gamma=b.mean(axis=(0,1)),batches=b,chain_means=b.mean(axis=1))
    truth=np.array([[0,0,0],[1,1,1]],bool);m,_,_=evaluate_arrays(arrays,truth,[0,5])
    assert m['two_max']['iou']==0 and m['two_max']['unresolved']==1


@pytest.mark.parametrize('g',[0.,.25,.75])
def test_original_parameter_conventions(g):
    im=np.array([[0,20,160,255],[5,25,150,240]],float)
    rule=dict(threshold='edge_otsu',sensitivity='fixed01',K=.8,transverse='relative',strength=g,temperature='longitudinal')
    new,p=parameterize(im,rule);old,q=old_parameterize(im,rule)
    assert np.array_equal(new.delta,old.delta) and p==q


def test_q90_positive_difference_fallback_and_temperature():
    im=np.zeros((2,4));im[:,2:]=255
    rule=dict(threshold='q90',sensitivity='inverse_spread',K=.8,transverse='relative',strength=.5,temperature='longitudinal')
    m,p=parameterize(im,rule)
    assert p['threshold']==255 and np.all(m.delta>=0)
    assert p['b']*np.median(abs(m.delta))/4==pytest.approx(.8)
    _,flat=parameterize(np.zeros((2,4)),rule)
    assert flat['threshold']==125 and np.isfinite(flat['b'])


@pytest.mark.parametrize('g',[0.,.25,.75])
def test_small_gibbs_exact_reference_and_paired_score(g):
    im=np.array([[0,15,200,255],[10,20,190,230]],float)
    rule=dict(threshold='edge_otsu',sensitivity='fixed01',K=.8,transverse='relative',strength=g,temperature='longitudinal')
    arrays,p,d=sample(im,rule,[0,3],burn=600,samples=4096,thin=2,chains=4,rng_seed=1293)
    assert d['gamma_linf_vs_exact']<.012
    assert np.max(abs(scores(arrays['gamma'])-scores(arrays['reference'])))<.017
    assert np.allclose(arrays['gamma'][[0,1],[0,3]],.25)
