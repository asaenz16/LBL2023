"""Independent physics, circuit, ordering, and exchange checks."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit, qasm2
from scipy.io import loadmat
from scipy.linalg import expm

from algorithm1 import (compile_circuit, estimate_correlations, gibbs_circuit,
                        image_model, sample_counts, seed_correlations,
                        system_density, xxz_hamiltonian)


def reference_gibbs(h, beta, n):
    shifted = h - np.linalg.eigvalsh(h)[0] * np.eye(len(h))
    rho = expm(-beta / n * shifted)
    return rho / np.trace(rho)


@pytest.mark.parametrize("delta", [-2.0, 0.0, 1.0, 2.0])
def test_two_site_matrix_and_spectrum_from_existing_notebook(delta):
    h = xxz_hamiltonian(2, [[0, 1]], [delta])
    expected = np.diag([-delta/4, delta/4, delta/4, -delta/4])
    expected[1, 2] = expected[2, 1] = -0.5
    assert_allclose(h, expected)
    assert_allclose(np.linalg.eigvalsh(h), sorted([-delta/4, -delta/4,
                                                (delta-2)/4, (delta+2)/4]), atol=1e-14)


def test_grid_hamiltonian_against_independent_kronecker_construction():
    model = image_model([[0, 12], [47, 90]], sigma_ddof=0)
    assert model.edges.tolist() == [[0, 1], [0, 2], [1, 3], [2, 3]]
    assert_allclose(model.distances, [12, 47, 78, 43])
    assert_allclose(model.sigma, np.std([12, 47, 78, 43]))
    paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
              np.diag([1, -1])]
    h = np.zeros((16, 16), complex)
    for (p, q), delta in zip(model.edges, model.delta):
        for pauli, coefficient in zip(paulis, [1, 1, delta]):
            term = np.array([[1]])
            for site in range(4):
                term = np.kron(term, pauli if site in (p, q) else np.eye(2))
            h -= coefficient * term / 4
    assert_allclose(model.hamiltonian, h)


def test_rgb_and_standard_deviation_conventions():
    rgb = np.array([[[0, 0, 0], [3, 4, 0], [3, 4, 12]]])
    sample = image_model(rgb)
    population = image_model(rgb, sigma_ddof=0)
    assert_allclose(sample.distances, [5, 12])
    assert_allclose(sample.sigma, population.sigma * np.sqrt(2))
    with pytest.raises(ValueError, match="sigma"):
        image_model(np.zeros((2, 2)))
    assert_allclose(image_model(np.zeros((2, 2)), sigma=1).delta, 2)
    with pytest.raises(ValueError, match="sigma"):
        image_model([[0, 1]])


@pytest.mark.parametrize("beta", [0.0, 0.7, 24.0, 200.0])
def test_compiled_circuit_density_matches_matrix_exponential(beta):
    model = image_model([[0, 12], [47, 90]])
    prepared = gibbs_circuit(model.hamiltonian, beta)
    compiled = compile_circuit(prepared)
    assert set(compiled.count_ops()) <= {"u", "cx"}
    rho = system_density(compiled, 4)
    assert_allclose(rho, reference_gibbs(model.hamiltonian, beta, 4), atol=2e-12)
    assert_allclose(np.trace(rho), 1, atol=2e-12)


def test_two_site_analytic_thermal_correlation():
    delta, beta = 1.8, 3.0
    h = xxz_hamiltonian(2, [[0, 1]], [delta])
    rho = system_density(compile_circuit(gibbs_circuit(h, beta)), 2)
    t = beta/2
    a, b = np.exp(t*delta/4), np.exp(-t*delta/4)*np.cosh(t/2)
    analytic = (a-b)/(4*(a+b))
    assert_allclose(seed_correlations(rho, 0), [0.25, analytic], atol=1e-12)


def test_zero_temperature_limit_keeps_degenerate_ground_space_mixture():
    h = xxz_hamiltonian(2, [[0, 1]], [2.0])
    rho = system_density(compile_circuit(gibbs_circuit(h, 200.0)), 2)
    assert_allclose(rho, np.diag([0.5, 0, 0, 0.5]), atol=1e-12)
    assert_allclose(np.trace(rho @ rho), 0.5, atol=1e-12)


def test_infinite_temperature_correlations():
    model = image_model([[0, 3, 10]])
    rho = system_density(compile_circuit(gibbs_circuit(model.hamiltonian, 0)), 3)
    assert_allclose(rho, np.eye(8)/8, atol=1e-12)
    assert_allclose(seed_correlations(rho, 1), [0, 0.25, 0], atol=1e-12)


def test_pixel_qubit_measurement_order_and_nonzero_seed():
    circuit = QuantumCircuit(6)
    circuit.x(2)  # Pixel zero is the most significant SYSTEM qubit.
    counts = sample_counts(circuit, 3, shots=32)
    assert counts == {"100": 32}
    result = estimate_correlations(counts, 3, seed=2)
    assert_allclose(result.gamma, [-0.25, 0.25, 0.25])
    assert_allclose(seed_correlations(system_density(circuit, 3), 2), result.gamma)


def test_ties_and_uncertainty_are_reported():
    estimate = estimate_correlations({"00": 50, "01": 50}, 2, 0)
    assert_allclose(estimate.gamma, [0.25, 0])
    assert estimate.same_segment.tolist() == [True, False]
    assert estimate.sign_resolved.tolist() == [True, False]
    assert estimate.lower[1] < 0 < estimate.upper[1]
    assert estimate.lower[0] == estimate.upper[0] == 0.25


@pytest.mark.parametrize("image", [np.zeros((3, 3)), [[np.nan, 1]], np.zeros((1, 1, 2))])
def test_invalid_images_are_rejected(image):
    with pytest.raises(ValueError):
        image_model(image)


def test_bad_hamiltonians_edges_and_counts_are_rejected():
    with pytest.raises(ValueError, match="once"):
        xxz_hamiltonian(2, [[0, 1], [1, 0]], [1, 1])
    with pytest.raises(ValueError, match="Hermitian"):
        gibbs_circuit([[0, 1], [0, 0]], 1)
    with pytest.raises(ValueError):
        gibbs_circuit(np.eye(3), 1)
    with pytest.raises(ValueError):
        estimate_correlations({"000": 10}, 2, 0)
    with pytest.raises(ValueError):
        estimate_correlations({"00": 0}, 2, 0)


def test_demo_qasm_and_matlab_gate_exchange(tmp_path):
    script = Path(__file__).resolve().parents[1] / "python" / "run_demo.py"
    subprocess.run([sys.executable, str(script), "--output", str(tmp_path)],
                   check=True, capture_output=True, text=True)
    f = loadmat(tmp_path / "fixture.mat")
    report = json.loads((tmp_path / "report.json").read_text())
    n = int(f["n_pixels"].item())
    imported = qasm2.loads((tmp_path / "circuit.qasm").read_text())
    rho = system_density(imported.remove_final_measurements(inplace=False), n)
    assert_allclose(rho, f["rho_reference"], atol=1e-12)
    assert_allclose(f["gamma_reference"].ravel(), report["gamma_exact"])
    assert report["same_segment_shots"] == [[True, True], [False, False]]
    assert all(report["sign_resolved"])
    assert np.all(np.array(report["ci_lower"]) <= np.array(report["gamma_exact"]) + 1e-12)
    assert np.all(np.array(report["ci_upper"]) >= np.array(report["gamma_exact"]) - 1e-12)

    # Independently simulate the MAT gate encoding in QCLAB's MSB convention.
    # This verifies the bridge algebra; it does NOT execute MATLAB/QCLAB itself.
    state = np.zeros([2] * (2*n), complex)
    state[(0,) * (2*n)] = 1

    def mapped(q):
        return n - 1 - q if q < n else 3*n - 1 - q

    def apply_one(matrix, axis):
        nonlocal state
        state = np.moveaxis(np.tensordot(matrix, state, axes=(1, axis)), 0, axis)

    for (code, first, second), (theta, phi, lam) in zip(f["gates"], f["u_angles"]):
        q = mapped(int(first))
        if code == 1:
            apply_one(np.diag([1, np.exp(1j*lam)]), q)
            apply_one(np.array([[np.cos(theta/2), -np.sin(theta/2)],
                                [np.sin(theta/2), np.cos(theta/2)]]), q)
            apply_one(np.diag([1, np.exp(1j*phi)]), q)
        else:
            target = mapped(int(second))
            state = np.moveaxis(state, [q, target], [0, 1])
            state[1] = state[1, ::-1].copy()
            state = np.moveaxis(state, [0, 1], [q, target])
    amplitude_matrix = state.reshape(2**n, 2**n)
    assert_allclose(amplitude_matrix @ amplitude_matrix.conj().T,
                    f["rho_reference"], atol=2e-12)
