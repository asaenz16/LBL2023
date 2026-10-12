import numpy as np
import pytest
from numpy.testing import assert_allclose
from xxz_glauber import (graph, image_graph, heatbath_exits, expansion_cutoff,
                        run_chain, exact_gibbs)


def test_vertex_weights_reconstruct_full_xxz_bond():
    x = np.array([[0., 1.], [1., 0.]])
    y = np.array([[0., -1j], [1j, 0.]])
    z = np.diag([1., -1.])
    for delta in [-3.2, 0., 2.]:
        m = graph(2, [[0, 1]], [delta])
        h = -(np.kron(x, x)+np.kron(y, y)+delta*np.kron(z, z))/4
        assert_allclose(m.weights[0].reshape(4, 4), m.shift[0]*np.eye(4)-h)


def test_all_local_scatterings_obey_detailed_balance():
    for delta in [-3.2, -.1, 0., 2., 5.]:
        m = graph(2, [[0, 1]], [delta])
        w = m.weights[0]
        p = heatbath_exits(w)
        assert_allclose(p[w > 0].sum(axis=-1), 1)
        for v in np.flatnonzero(w):
            for entrance in range(4):
                for leave in range(4):
                    other = v ^ (1 << (3-entrance)) ^ (1 << (3-leave))
                    assert_allclose(w[v]*p[v, entrance, leave],
                                    w[other]*p[other, leave, entrance], atol=1e-14)


def test_image_conventions_and_dense_benchmark_match_existing_code():
    pytest.importorskip("quimb")
    from ladder_tensor import build_ladder, dense_row_hamiltonian
    for width in [4, 8, 12, 16]:
        m, old = image_graph(width), build_ladder(width)
        assert_allclose(m.image, old.image)
        assert_allclose(m.sigma, old.sigma)
        assert_allclose(m.delta[width:width+width-1], old.horizontal_delta)
    m = image_graph(4)
    assert_allclose(exact_gibbs(m, 1.)["hamiltonian"], dense_row_hamiltonian(build_ladder(4)))


def test_cutoff_bound_and_input_validation():
    m = image_graph(16)
    capacity, log_bound = expansion_cutoff(m, 6.)
    assert log_bound < -50
    with pytest.raises(ValueError):
        run_chain(m, np.ones(32), 6., [0, 1], capacity=capacity-1)
    for b in [-1., 0., np.nan]:
        with pytest.raises(ValueError):
            expansion_cutoff(m, b)
    with pytest.raises(ValueError):
        run_chain(m, np.ones(32), 6., [0, .5])


def test_path_histogram_reproduces_correlations():
    m = image_graph(4)
    r = run_chain(m, np.ones(8), 1., [0, 1, 10, 100], seeds=[0, 1], record_distribution=True)
    z = 1-2*((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1)
    assert_allclose(r[5].sum(axis=1), 1)
    for t in range(4):
        assert_allclose(r[0][t], np.array([(r[5][t]*z[:, s]) @ z/4 for s in [0, 1]]), atol=1e-14)


@pytest.mark.parametrize("delta", [-2.7, 0., 2.])
def test_two_spin_sampling_including_magnetization_sectors(delta):
    m = graph(2, [[0, 1]], [delta])
    exact = exact_gibbs(m, 2.)
    result = run_chain(m, [1, -1], 2., np.arange(2000, 66000), seed=715)
    samples = np.column_stack([result[0][:, 0, 1], m.shift.sum()-result[2]/2,
                               (result[3]/2)**2])
    batches = samples.reshape(64, 1000, 3).mean(axis=1)
    error = batches.std(axis=0, ddof=1)/8
    target = [exact["gamma"][0, 1], exact["energy"], exact["magnetization_second_moment"]]
    assert np.all(abs(batches.mean(axis=0)-target) < 5*error + 1e-3)
    assert set(result[3]) == {-2, 0, 2}


def test_eight_spin_high_temperature_against_exact_gibbs():
    m = image_graph(4)
    exact = exact_gibbs(m, 1., [0, 1])
    result = run_chain(m, np.ones(8), 1., np.arange(2000, 42000), seeds=[0, 1], seed=819)
    batch = result[0].reshape(40, 1000, 2, 8).mean(axis=1)
    error = batch.std(axis=0, ddof=1)/np.sqrt(40)
    assert np.all(abs(batch.mean(axis=0)-exact["gamma"]) < 5*error + 1e-3)
    energies = (m.shift.sum()-result[2]).reshape(40, 1000).mean(axis=1)
    assert abs(energies.mean()-exact["energy"]) < 5*energies.std(ddof=1)/np.sqrt(40)
    assert len(np.unique(result[3])) >= 5
