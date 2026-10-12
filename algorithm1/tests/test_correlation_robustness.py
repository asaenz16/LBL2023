import numpy as np
from numpy.testing import assert_allclose

from algorithm1 import seed_correlations
from correlation_robustness import (correlation_metrics, ensemble_linf_diameter,
                                    initial_ensemble, pauli_noise, permanent_crossing,
                                    spin_correlations)
from thermal_bath import trace_distance


def test_product_order_and_classical_mixture_have_different_connected_correlations():
    product = np.diag([1.,0,0,0])
    mixture = np.diag([.5,0,0,.5])
    raw,connected,means = spin_correlations(product)
    assert_allclose(raw,[.25,.25]); assert_allclose(connected,[0,0])
    assert_allclose(means,[.5,.5])
    assert_allclose(spin_correlations(mixture)[0],[.25,.25])
    assert_allclose(spin_correlations(mixture)[1],[.25,.25])


def test_coherence_changes_state_without_changing_zz_correlations():
    bell = np.array([1,0,0,1])/np.sqrt(2)
    rho = np.outer(bell,bell)
    dephased = pauli_noise(rho,2,"Z",.5)
    assert_allclose(seed_correlations(dephased,0),seed_correlations(rho,0))
    assert_allclose(trace_distance(dephased,rho),.5)
    assert_allclose(dephased,np.diag(np.diag(rho)),atol=1e-15)


def test_independent_bit_flips_attenuate_every_offsite_zz_pair():
    rho = np.diag([.5,0,0,.5]); p=.17
    noisy = pauli_noise(rho,2,"X",p)
    assert_allclose(seed_correlations(noisy,0),[.25,.25*(1-2*p)**2])


def test_distance_metrics_match_manual_examples_and_trace_bound():
    ref = np.diag([.5,0,0,.5])
    opposite = np.diag([0,.5,.5,0]); states=np.array([ref,opposite])
    previous = np.array([[.25,0],[.25,0]])
    metrics=correlation_metrics(states,ref,previous=previous)
    assert_allclose(metrics["linf_reference"],[0,.5])
    assert_allclose(metrics["linf_step"],[.25,.25])
    assert_allclose(metrics["signed_margin"],[.25,-.25])
    assert np.all(metrics["linf_reference"] <= .5*metrics["trace_distance"]+1e-14)
    assert_allclose(ensemble_linf_diameter(metrics["gamma"]),.5)


def test_random_ensemble_is_reproducible_positive_and_normalized():
    states,labels,families=initial_ensemble(4,3,12)
    assert len(states) == 10 and len(labels) == len(families) == 10
    assert_allclose(states,initial_ensemble(4,3,12)[0])
    assert_allclose(np.trace(states,axis1=1,axis2=2),1)
    assert np.linalg.eigvalsh(states).min() > -1e-12
    assert_allclose(spin_correlations(states[3])[1],0)


def test_persistent_threshold_rejects_early_transient_crossings():
    assert permanent_crossing([False,True,False,True,True]) == 3
    assert permanent_crossing([True,True]) == 0
    assert permanent_crossing([True,False]) is None
