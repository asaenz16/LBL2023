import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from ising_algorithm1 import ising_model, gibbs_ising, spin_basis
from ising_monte_carlo import signed_image, energy, sample_gibbs, split_rhat


def test_sign_scale_and_exact_energy_compatibility():
    image = np.array([[0, 180, 90], [210, 20, 255]])
    model = signed_image(image, .03); old = ising_model(image, -.03)
    assert_allclose(model.delta, old.delta)
    assert_allclose([energy(z, model.edges, model.delta/4) for z in spin_basis(6)], old.energies)
    assert model.delta[0] < 0
    for alpha in (0, -1, np.inf):
        with pytest.raises(ValueError): signed_image(image, alpha)


def test_signed_cluster_conditional_estimator_exact_bond_sum():
    # Frustrated plaquette: prove the estimator independently by summing every
    # spin and bond configuration, including zero and antiferromagnetic edges.
    image = [[0, 90], [250, 170]]
    m = signed_image(image, .03); exact = ising_model(image, -.03)
    assert np.prod(np.sign(m.delta)) == -1
    b = 3.; g = gibbs_ising(exact, beta=4*b); zall = spin_basis(4)
    result = np.zeros((4, 4))
    for z, weight in zip(zall, g.weights):
        prob = np.where(m.delta*z[m.edges[:,0]]*z[m.edges[:,1]] > 0,
                        -np.expm1(-b*abs(m.delta)/2), 0.)
        for bits in itertools.product((0, 1), repeat=len(prob)):
            bonds = np.array(bits, bool); mass = np.prod(np.where(bonds, prob, 1-prob))
            connected = np.eye(4, dtype=bool)
            for (i,j), on in zip(m.edges, bonds):
                if on: connected[i,j] = connected[j,i] = True
            for k in range(4): connected |= connected[:,k,None] & connected[None,k,:]
            result += weight*mass*connected*z[:,None]*z[None,:]/4
    assert_allclose(result, g.gamma, atol=2e-15)


def test_sampler_matches_exact_frustrated_gibbs():
    image = np.array([[0,170,80],[200,20,255],[40,190,90]])
    m = signed_image(image, .03)
    r = sample_gibbs(m, [0,1,2,4,8], seeds=[0,4], burn=1024, samples=4096)
    exact = ising_model(image, -.03)
    truth = np.array([gibbs_ising(exact, b*9).gamma[[0,4]] for b in r['bs']])
    error = abs(r['gamma']-truth)
    assert np.max(error) < .012
    assert np.all(error < 6*r['se']+.002)
    assert np.max(r['rhat_observables']) < 1.03
    assert_array_equal(r['gamma'][0], np.eye(9)[[0,4]]/4)


def test_disconnected_zero_correlations_and_diagnostic():
    r = sample_gibbs(signed_image([[0,125]], .1), [2], samples=256, burn=32)
    assert_array_equal(r['gamma'], [[[.25,0]]])
    rng = np.random.default_rng(8)
    x = rng.normal(size=(4, 1024, 2)); x[0,:,1] += 3
    rh = split_rhat(x)
    assert rh[0] < 1.01 and rh[1] > 1.5


def test_parameter_rule_and_metrics_do_not_confuse_background_accuracy():
    from analyze_emnist_ising import continuous_rule, choose_grid_rule
    from run_emnist_ising import choose_seeds, proxy_metrics
    image=np.zeros((4,4));image[1:3,1:3]=255
    seeds=choose_seeds(image);assert_array_equal(seeds,[0,5])
    a,b,d=continuous_rule(image,.125,.7)
    assert_allclose(a,.125/130)
    assert_allclose(b*np.median(abs(signed_image(image,a).delta))/4,.7)
    aa,bb,_=choose_grid_rule(image,.7,.125,[(.001,16),(.001,24),(.01,3)])
    assert aa==.001 and bb==24
    # All-background segmentation can have high accuracy and zero foreground IoU.
    r=proxy_metrics(np.full(16,.2),np.zeros(16),image,0)
    assert r['accuracy']==.75 and r['iou']==0 and r['balanced_accuracy']==.5
    with pytest.raises(ValueError):continuous_rule(np.array([[0,125]]),.1,.7)
    with pytest.raises(ValueError):sample_gibbs(signed_image([[0,255]],.1),[1,1])
