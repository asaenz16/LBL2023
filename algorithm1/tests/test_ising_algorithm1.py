import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import expm
from ising_algorithm1 import ising_model, gibbs_ising, algorithm1_labels


@pytest.mark.parametrize('alpha', [-.1, 0., .1])
def test_two_spin_analytic_sign_and_scale(alpha):
    for image in [[[0, 0]], [[0, 255]], [[0, 125]]]:
        m = ising_model(image, alpha); r = gibbs_ising(m, beta=8.)
        d = m.delta[0]
        assert_allclose(m.energies, [-d/4, d/4, d/4, -d/4])
        assert_allclose(r.gamma[0, 1], np.tanh(r.beta_effective*d/4)/4, atol=1e-14)
        assert_allclose(r.magnetization, 0, atol=0)


def test_independent_kronecker_hamiltonian_and_exponential():
    m = ising_model([[0, 31], [188, 239]], -.03); h = np.zeros((16, 16))
    for (p, q), d in zip(m.edges, m.delta):
        operator = np.array([[1.]])
        for i in range(4): operator = np.kron(operator, np.diag([1., -1.]) if i in (p, q) else np.eye(2))
        h -= d*operator/4
    assert_allclose(np.diag(m.energies), h)
    rho = expm(-3*h); rho /= np.trace(rho)
    r = gibbs_ising(m, beta=12.)
    assert_allclose(r.weights, rho.diagonal(), atol=1e-14)
    from algorithm1 import seed_correlations
    for seed in range(4): assert_allclose(r.gamma[seed], seed_correlations(rho, seed), atol=1e-14)


def test_bipartite_sign_change_and_expected_clean_segmentation():
    image = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0]])
    positive = gibbs_ising(ising_model(image, .03), beta=36.)
    negative = gibbs_ising(ising_model(image, -.03), beta=36.)
    parity = (-1.)**np.indices(image.shape).sum(axis=0).ravel()
    assert_allclose(positive.gamma, negative.gamma*parity[:, None]*parity[None, :], atol=2e-14)
    truth = image.ravel()[:, None] == image.ravel()[None, :]
    assert_array_equal(negative.gamma > 0, truth)
    assert_array_equal(algorithm1_labels(negative.gamma, 4), truth[4])


def test_threshold_components_and_zero_controls():
    image = np.tile([0, 0, 125, 125, 0, 0], (2, 1))
    r = gibbs_ising(ising_model(image, -.03), beta=48.)
    assert_array_equal(r.gamma[0, [2, 3, 4, 5, 8, 9, 10, 11]], 0.)
    assert r.gamma[0, 1] > 0
    for m, beta in [(ising_model(image, 0), 48.), (ising_model(image, -.03), 0.)]:
        assert_array_equal(gibbs_ising(m, beta).gamma, np.eye(12)/4)


def test_rgb_raw_distance_and_uniform_images():
    m = ising_model(np.array([[[0, 0, 0], [3, 4, 12]]]), .1)
    assert_allclose(m.distances, [13.])
    assert_allclose(m.delta, [np.arctan(.1*(13-125))])
    assert np.all(ising_model(np.zeros((2, 2)), -.01).delta > 0)
    assert gibbs_ising(ising_model([[99]], .03), 4).gamma[0, 0] == .25


def test_compatible_with_existing_small_gibbs_circuit():
    from algorithm1 import gibbs_circuit, compile_circuit, system_density
    m = ising_model([[0, 0], [255, 255]], -.03)
    circuit = compile_circuit(gibbs_circuit(np.diag(m.energies), beta=16.))
    rho = system_density(circuit, 4)
    assert_allclose(rho, np.diag(gibbs_ising(m, 16.).weights), atol=2e-12)


def test_invalid_inputs():
    for image, alpha in [([[np.nan]], .1), ([[256]], .1), (np.zeros((2,2,2)), .1), ([[0]], np.inf), (np.zeros((5,5)), .1)]:
        with pytest.raises(ValueError): ising_model(image, alpha)
    with pytest.raises(ValueError): gibbs_ising(ising_model([[0]], .1), -1)
    with pytest.raises(ValueError): algorithm1_labels(np.eye(2), -1)
