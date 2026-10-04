"""Small-instance circuit reference for Algorithm 1 in MainV2.tex.

Pixel indices are row-major and zero-based. In dense matrices pixel 0 is
the leftmost (most significant) tensor factor, matching MATLAB kron order.
Exact diagonalization and generic circuit synthesis are exponential in N.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector, partial_trace

MAX_PIXELS = 6


def check_n(n: int) -> None:
    if not isinstance(n, Integral) or not 1 <= n <= MAX_PIXELS:
        raise ValueError(f"This dense reference supports 1 to {MAX_PIXELS} pixels.")


@dataclass
class ImageModel:
    image: np.ndarray
    edges: np.ndarray
    distances: np.ndarray
    delta: np.ndarray
    sigma: float
    sigma_ddof: int
    kappa: float
    hamiltonian: np.ndarray

    @property
    def n(self) -> int:
        return int(np.prod(self.image.shape[:2]))


def xxz_hamiltonian(n: int, edges: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """H = -1/4 sum_{undirected edges} (XX + YY + delta ZZ).

The manuscript's ordered-neighbor sum has a factor 1/2. Listing each edge
once removes that factor. Bit flips implement XX+YY without dense krons.
"""
    check_n(n)
    edges = np.asarray(edges)
    delta = np.asarray(delta, dtype=float)
    if edges.shape != (len(delta), 2) or not np.all(np.isfinite(delta)):
        raise ValueError("Provide one finite anisotropy per edge, with shape (E, 2).")
    if not np.all(edges == np.floor(edges)):
        raise ValueError("Edge indices must be integers.")
    edges = edges.astype(int)
    if np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("Edges must join two distinct existing pixels.")
    if len({tuple(sorted(edge)) for edge in edges}) != len(edges):
        raise ValueError("List every undirected edge exactly once.")
    states = np.arange(2**n)
    h = np.zeros((2**n, 2**n), dtype=float)
    for (p, q), anisotropy in zip(edges, delta):
        mask_p, mask_q = 1 << (n - 1 - p), 1 << (n - 1 - q)
        opposite = ((states & mask_p) != 0) != ((states & mask_q) != 0)
        h[states, states] -= anisotropy * (1 - 2 * opposite.astype(int)) / 4
        selected = states[opposite]
        h[selected ^ mask_p ^ mask_q, selected] -= 0.5
    return h


def image_model(image, kappa: float = 2.0, *, sigma: float | None = None,
                sigma_ddof: int = 1) -> ImageModel:
    """Grayscale (rows, cols) or RGB (rows, cols, 3), Euclidean color distance.

Open boundaries, four nearest neighbors. Sigma is the standard deviation
of distances over UNIQUE undirected edges, not of pixel intensities.
The paper does not specify ddof or how to handle zero variance; require an
explicit positive sigma when it cannot be inferred.
"""
    image = np.asarray(image, dtype=float)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Use a 2D grayscale or (rows, columns, 3) RGB array.")
    if not np.all(np.isfinite(image)) or 0 in image.shape:
        raise ValueError("Image must be nonempty and finite.")
    n = int(np.prod(image.shape[:2]))
    check_n(n)
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")
    if sigma_ddof not in (0, 1):
        raise ValueError("sigma_ddof must be 0 (population) or 1 (sample).")
    rows, cols = image.shape[:2]
    edges = []
    for row in range(rows):
        for col in range(cols):
            p = row * cols + col
            if col + 1 < cols:
                edges.append((p, p + 1))
            if row + 1 < rows:
                edges.append((p, p + cols))
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    colors = image.reshape(n, -1)
    distances = np.linalg.norm(colors[edges[:, 0]] - colors[edges[:, 1]], axis=1)
    if sigma is None:
        if len(distances) <= sigma_ddof:
            raise ValueError("Too few edge distances: supply a positive sigma explicitly.")
        sigma = float(np.std(distances, ddof=sigma_ddof))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Zero/invalid distance variance: supply a positive sigma explicitly.")
    delta = 2 - kappa * distances / sigma
    return ImageModel(image.copy(), edges, distances, delta, float(sigma),
                      sigma_ddof, float(kappa), xxz_hamiltonian(n, edges, delta))


@dataclass
class GibbsCircuit:
    circuit: QuantumCircuit
    energies: np.ndarray
    weights: np.ndarray
    eigenvectors: np.ndarray
    beta_effective: float
    n: int


def gibbs_circuit(hamiltonian: np.ndarray, beta: float) -> GibbsCircuit:
    """Prepare sum_k sqrt(p_k) |E_k>_system |k>_ancilla using 2N qubits.

Eigenvectors are COLUMNS: H = U diag(E) U^dagger. With the ancilla ignored,
the system is exp(-beta H/N)/Z. Reprepare for each measurement shot.
"""
    h = np.asarray(hamiltonian, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2:
        raise ValueError("Hamiltonian must be square with dimension 2**N, N >= 1.")
    n = h.shape[0].bit_length() - 1
    check_n(n)
    if h.shape[0] != 2**n or not np.all(np.isfinite(h)):
        raise ValueError("Hamiltonian must be finite with dimension 2**N.")
    if not np.allclose(h, h.conj().T, atol=1e-12, rtol=0):
        raise ValueError("Hamiltonian must be Hermitian.")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    energies, u = np.linalg.eigh(h)
    beta_effective = float(beta) / n
    # Subtracting Emin preserves rho and avoids overflow at low temperature.
    weights = np.exp(-beta_effective * (energies - energies[0]))
    weights /= weights.sum()
    # Avoid 's': OpenQASM's qelib1.inc already defines a gate with that name.
    system, ancilla = QuantumRegister(n, "sysreg"), QuantumRegister(n, "envreg")
    circuit = QuantumCircuit(system, ancilla, name="algorithm1_gibbs")
    loading = QuantumCircuit(n, name="Gibbs weights")
    loading.append(StatePreparation(np.sqrt(weights)), range(n))
    # A named block keeps the overview diagram readable; synthesis expands it.
    circuit.append(loading.to_gate(label="sqrt(p)"), ancilla)
    for k in range(n):
        circuit.cx(ancilla[k], system[k])
    circuit.append(UnitaryGate(u, label="energy basis U"), system)
    circuit.metadata = {"n_pixels": n, "beta_paper": float(beta),
                        "pixel_p_to_system_qubit": "N-1-p",
                        "method": "exact spectral purification (exponential)"}
    return GibbsCircuit(circuit, energies, weights, u, beta_effective, n)


def compile_circuit(preparation: GibbsCircuit) -> QuantumCircuit:
    """Explicit one-qubit U and CNOT gates; no device routing/noise model."""
    return transpile(preparation.circuit, basis_gates=["u", "cx"],
                     optimization_level=1, seed_transpiler=1200)


def system_density(circuit: QuantumCircuit, n: int) -> np.ndarray:
    check_n(n)
    if circuit.num_qubits != 2 * n:
        raise ValueError("Expected N system qubits followed by N ancilla qubits.")
    return partial_trace(Statevector.from_instruction(circuit), list(range(n, 2*n))).data


def z_table(n: int) -> np.ndarray:
    """Rows are dense basis states; columns are row-major pixel indices."""
    check_n(n)
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return 1 - 2 * bits


def seed_correlations(rho: np.ndarray, seed: int) -> np.ndarray:
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError("rho must be square with dimension 2**N.")
    n = rho.shape[0].bit_length() - 1
    check_n(n)
    if rho.shape[0] != 2**n or not isinstance(seed, Integral) or not 0 <= seed < n:
        raise ValueError("Invalid density dimension or seed index.")
    z = z_table(n)
    return np.diag(rho).real @ (z[:, seed, None] * z) / 4


def measurement_circuit(circuit: QuantumCircuit, n: int) -> QuantumCircuit:
    check_n(n)
    if circuit.num_qubits != 2*n or circuit.num_clbits:
        raise ValueError("Expected an unmeasured purification circuit on 2N qubits.")
    measured = circuit.copy()
    bits = ClassicalRegister(n, "pixels")
    measured.add_register(bits)
    measured.measure(range(n), bits)
    # Qiskit prints c[N-1]...c[0], hence pixel 0...pixel N-1, left to right.
    return measured


def sample_counts(circuit: QuantumCircuit, n: int, *, shots: int = 8192,
                  rng_seed: int = 1200) -> dict[str, int]:
    if not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer.")
    measured = measurement_circuit(circuit, n)
    return dict(BasicSimulator().run(measured, shots=shots, seed_simulator=rng_seed)
                .result().get_counts())


@dataclass
class CorrelationEstimate:
    gamma: np.ndarray
    same_segment: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sign_resolved: np.ndarray
    shots: int


def estimate_correlations(counts: Mapping[str, int], n: int, seed: int,
                          *, alpha: float = 0.05) -> CorrelationEstimate:
    """Shot estimates and simultaneous Hoeffding bounds for all seed pairs.

Each outcome is +/-1/4. Union bound: radius=sqrt(log(2*N/alpha)/(8*shots)).
The strict >0 rule matches Algorithm 1. sign_resolved separately flags
labels whose sign is supported by the confidence bounds.
"""
    check_n(n)
    if not isinstance(seed, Integral) or not 0 <= seed < n or not 0 < alpha < 1:
        raise ValueError("Invalid seed or alpha.")
    total, gamma = 0, np.zeros(n)
    for bits, count in counts.items():
        if len(bits) != n or set(bits) - {"0", "1"}:
            raise ValueError("Counts must use N-bit strings in row-major pixel order.")
        if not isinstance(count, Integral) or count < 0:
            raise ValueError("Counts must be nonnegative integers.")
        z = 1 - 2 * np.array([int(bit) for bit in bits])
        gamma += count * z[seed] * z / 4
        total += count
    if total == 0:
        raise ValueError("At least one shot is required.")
    gamma /= total
    radius = np.sqrt(np.log(2 * n / alpha) / (8 * total))
    lower, upper = np.maximum(-0.25, gamma - radius), np.minimum(0.25, gamma + radius)
    lower[seed] = upper[seed] = 0.25
    return CorrelationEstimate(gamma, gamma > 0, lower, upper,
                               (lower > 0) | (upper < 0), total)
