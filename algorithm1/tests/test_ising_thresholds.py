import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from ising_algorithm1 import ising_model, gibbs_ising, spin_basis
from ising_monte_carlo import signed_image, sample_gibbs, energy
from ising_thresholds import otsu_edge_threshold, threshold_for_image, normalized_model, K_REFERENCE


def test_custom_threshold_energy_and_old_default():
    image=np.array([[0,70,180],[250,20,95]])
    for tau in [0,50,75.5,125,200]:
        m=signed_image(image,.001,threshold=tau)
        exact=ising_model(image,-.001,threshold=tau)
        assert_allclose(m.delta,exact.delta)
        assert_allclose([energy(z,m.edges,m.delta/4) for z in spin_basis(6)],exact.energies)
    assert_array_equal(signed_image(image,.001).delta,signed_image(image,.001,threshold=125).delta)
    for tau in [-1,np.inf,np.nan]:
        with pytest.raises(ValueError):signed_image(image,.001,threshold=tau)


def test_otsu_edge_split_against_direct_within_class_variance():
    d=np.array([0]*40+[1,3,6,11,37,80,100,200,255])
    for positive in [False,True]:
        x=d[d>0] if positive else d
        values=np.unique(x);candidates=(values[:-1]+values[1:])/2
        costs=[]
        for tau in candidates:
            low=x[x<tau];high=x[x>tau]
            costs.append(np.sum((low-low.mean())**2)+np.sum((high-high.mean())**2))
        assert otsu_edge_threshold(d,positive_only=positive)==candidates[np.argmin(costs)]
    assert otsu_edge_threshold([0,0])==125
    assert otsu_edge_threshold([0,255],positive_only=True)==125
    assert otsu_edge_threshold([0,255])==127.5
    with pytest.raises(ValueError):otsu_edge_threshold([-1,2])


def test_quantile_excludes_constant_edges_and_thermal_normalization():
    image=np.array([[0,0,0,20,60,140,255]])
    assert threshold_for_image(image,dict(kind='quantile',value=.5))==60
    assert threshold_for_image(np.zeros((4,4)),dict(kind='quantile',value=.75))==125
    for kind in [dict(kind='fixed',value=75),dict(kind='quantile',value=.75),
                 dict(kind='otsu',positive_only=False),dict(kind='otsu',positive_only=True)]:
        m,bs=normalized_model(image,kind,[.5,K_REFERENCE,.8])
        assert_allclose(bs*np.median(abs(m.delta))/4,[.5,K_REFERENCE,.8])
    with pytest.raises(ValueError):normalized_model([[0,75]],dict(kind='fixed',value=75),[.5])


def test_custom_threshold_sampler_vs_exact_gibbs():
    image=np.array([[0,170,80],[200,20,255],[40,190,90]])
    for tau in [50,87.5,100]:
        m=signed_image(image,.001,threshold=tau)
        exact=ising_model(image,-.001,threshold=tau)
        bs=[8.,20.,40.]
        r=sample_gibbs(m,bs,seeds=[0,4,8],burn=512,samples=4096,rng_seed=8271+int(tau))
        truth=np.array([gibbs_ising(exact,b*image.size).gamma[[0,4,8]] for b in bs])
        error=abs(r['gamma']-truth)
        assert error.max()<.015
        assert np.all(error<6*r['se']+.002)
        assert r['rhat_observables'].max()<1.03
