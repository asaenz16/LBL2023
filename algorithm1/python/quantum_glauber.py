"""Quantum heat-bath channels and their resettable-ancilla circuits.

For sigma=exp(-b H)/Z, erase spin A and recover it using the Petz map:
Phi_A(rho)=sqrt(sigma) [I_A tensor sigma_B^(-1/2) Tr_A(rho)
                       sigma_B^(-1/2)] sqrt(sigma).

This is the Schrodinger adjoint of the minimal conditional expectation
in arXiv:1409.3435, Eq. (13). It preserves a noncommuting Gibbs state but
is generally NONLOCAL. Neither that paper's commuting-local efficiency
theorem nor arXiv:1105.5970's Ising path-dynamics theorem is assumed here.

The explicit circuit swaps the erased spin into an ancilla, resets it,
applies a recovery isometry, then resets the same ancilla. All outcomes
are retained. Dense preprocessing and generic synthesis are exponential.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.linalg import svd
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, partial_trace


@dataclass
class SiteUpdate:
    site: int
    n: int
    permutation: np.ndarray
    recovery: np.ndarray  # two Kraus matrices, each (2**n, 2**(n-1)), in A,B order
    isometry: np.ndarray  # output order (environment,A,B), input B
    singular_values: np.ndarray
    isometry_error: float
    gibbs_error: float


@dataclass
class QuantumHeatBath:
    hamiltonian: np.ndarray
    b: float
    n: int
    sigma: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray
    updates: list[SiteUpdate]


def site_permutation(n, site):
    if not isinstance(n, Integral) or not 1 <= n <= 8 or not isinstance(site, Integral) or not 0 <= site < n:
        raise ValueError("Require 1 <= n <= 8 and a valid row-major pixel index.")
    axes = [site]+[j for j in range(n) if j != site]
    return np.arange(2**n).reshape([2]*n).transpose(axes).ravel()


def build_quantum_heat_bath(hamiltonian, b):
    """Construct stable Petz recovery isometries for at most eight spins.

    Form T by stacking the two blocks of sqrt(sigma). Then T^dagger T is
    sigma_B. The polar factor V=polar(T)=T sigma_B^(-1/2) is the required
    isometry. Thin SVD computes it without explicitly inverting a marginal
    or discarding small eigenvalues. Check both TP and Gibbs preservation.
    """
    h = np.asarray(hamiltonian, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or len(h) < 2:
        raise ValueError("H must be a square matrix of dimension 2**n.")
    n = len(h).bit_length()-1
    if n < 1 or n > 8 or len(h) != 2**n:
        raise ValueError("Dense circuit reference supports 1 to 8 spins.")
    if not np.isfinite(b) or b < 0 or not np.all(np.isfinite(h)):
        raise ValueError("Require finite H and finite b >= 0.")
    if not np.allclose(h, h.conj().T, atol=1e-12, rtol=0):
        raise ValueError("H must be Hermitian.")
    energies, u = np.linalg.eigh(h)
    amplitudes = np.exp(-.5*b*(energies-energies[0]))
    if np.any(amplitudes == 0):
        raise ValueError("Thermal amplitudes underflow; lower b for this dense reference.")
    amplitudes /= np.linalg.norm(amplitudes)
    root = (u*amplitudes)@u.conj().T
    sigma = (u*amplitudes**2)@u.conj().T
    d = 2**(n-1)
    updates = []
    for site in range(n):
        p = site_permutation(n, site)
        s = root[np.ix_(p, p)]
        t = np.vstack([s[:, :d], s[:, d:]])
        left, values, right = svd(t, full_matrices=False, lapack_driver="gesvd")
        # Below this threshold, polar factors in near-null directions are not
        # resolved reliably. Reject rather than silently regularize the map.
        if values[-1] <= 100*np.finfo(float).eps*values[0]:
            raise ValueError("Gibbs marginal is too ill-conditioned for this reference; lower b.")
        v = left@right
        recovery = v.reshape(2, 2*d, d)
        iso_error = float(np.linalg.norm(v.conj().T@v-np.eye(d)))
        ordered = sigma[np.ix_(p, p)]
        marginal = ordered[:d, :d]+ordered[d:, d:]
        reconstructed = sum(k@marginal@k.conj().T for k in recovery)
        gibbs_error = float(np.linalg.norm(reconstructed-ordered))
        if iso_error > 1e-10 or gibbs_error > 1e-10:
            raise RuntimeError("Failed isometry or Gibbs fixed-point check.")
        updates.append(SiteUpdate(site, n, p, recovery, v, values, iso_error, gibbs_error))
    return QuantumHeatBath(h, float(b), n, sigma, energies, u, updates)


def apply_update(update, rho):
    d = 2**(update.n-1)
    rho = np.asarray(rho, complex)
    if rho.shape != (2*d, 2*d):
        raise ValueError("Density matrix dimension does not match the update.")
    p = update.permutation
    ordered = rho[np.ix_(p, p)]
    marginal = ordered[:d, :d]+ordered[d:, d:]
    result = sum(k@marginal@k.conj().T for k in update.recovery)
    output = np.empty_like(result)
    output[np.ix_(p, p)] = result
    return output


def apply_sweep(bath, rho, order=None):
    order = list(range(bath.n)) if order is None else list(order)
    if sorted(order) != list(range(bath.n)):
        raise ValueError("A sweep must visit every site exactly once.")
    result = np.asarray(rho, complex).copy()
    for site in order:
        result = apply_update(bath.updates[site], result)
    return result


def full_kraus(update):
    """Four full-system Kraus matrices, in the existing row-major bit basis."""
    d = 2**(update.n-1)
    output = []
    p = update.permutation
    for r in update.recovery:
        for old_spin in range(2):
            ordered = np.zeros((2*d, 2*d), complex)
            ordered[:, old_spin*d:(old_spin+1)*d] = r
            k = np.empty_like(ordered)
            k[np.ix_(p, p)] = ordered
            output.append(k)
    return output


def recovery_circuit(update):
    """Isometry on B (low bits), fresh erased spin A, and fresh environment.

    The two highest input qubits MUST be zero. This circuit can be embedded
    in a unitary on n+1 qubits; only these initialized columns are specified.
    """
    iso = Isometry(update.isometry, 0, 0, epsilon=1e-12)
    iso.label = f"Recover spin {update.site}"
    circuit = QuantumCircuit(update.n+1, name=f"recovery_{update.site}")
    circuit.append(iso, range(update.n+1))
    return circuit


def update_circuit(update):
    """One real quantum-state update using N system qubits and one reusable bath."""
    n = update.n
    system, ancilla = QuantumRegister(n, "spin"), QuantumRegister(1, "bath")
    circuit = QuantumCircuit(system, ancilla, name=f"heat_bath_site_{update.site}")
    q = n-1-update.site  # paper pixel 0 is the most significant system bit
    rest = [j for j in range(n) if j != q]
    circuit.reset(ancilla[0])
    circuit.swap(system[q], ancilla[0])
    circuit.reset(ancilla[0])
    circuit.compose(recovery_circuit(update), qubits=rest+[q, n], inplace=True)
    circuit.reset(ancilla[0])
    return circuit


def sweep_circuit(bath, sweeps=1, order=None):
    if not isinstance(sweeps, Integral) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer.")
    order = list(range(bath.n)) if order is None else list(order)
    if sorted(order) != list(range(bath.n)):
        raise ValueError("A sweep must visit every site exactly once.")
    circuit = QuantumCircuit(QuantumRegister(bath.n, "spin"), QuantumRegister(1, "bath"),
                             name="quantum_heat_bath_sweep")
    for _ in range(sweeps):
        for site in order:
            circuit.compose(update_circuit(bath.updates[site]), inplace=True)
    return circuit


def preparation_circuit(n, kind, image_spins=None, corruption=.1):
    """Physical preparations on system+one bath; bath ends reset to |0>."""
    if not isinstance(n, Integral) or not 1 <= n <= 8:
        raise ValueError("Require 1 <= n <= 8.")
    circuit = QuantumCircuit(QuantumRegister(n, "spin"), QuantumRegister(1, "bath"))
    if kind == "plus_x":
        circuit.h(range(n))
    elif kind == "mixed":
        for q in range(n):
            circuit.h(q)
            circuit.cx(q, n)
            circuit.reset(n)
    elif kind in ("ordered", "corrupted"):
        z = np.asarray(image_spins)
        if z.shape != (n,) or not np.all(np.isin(z, [-1, 1])):
            raise ValueError("Supply one +/-1 image spin per pixel.")
        for site in np.flatnonzero(z == -1):
            circuit.x(n-1-int(site))
        # Equal mixture of the image and its global inverse, with zero Z means.
        circuit.h(n)
        for q in range(n):
            circuit.cx(n, q)
        circuit.reset(n)
        if kind == "corrupted":
            if not np.isfinite(corruption) or not 0 <= corruption <= 1:
                raise ValueError("corruption must be in [0,1].")
            for q in range(n):
                circuit.ry(2*np.arcsin(np.sqrt(corruption)), n)
                circuit.cx(n, q)
                circuit.reset(n)
    else:
        raise ValueError("Use plus_x, mixed, ordered, or corrupted.")
    return circuit


def ensemble_circuit_density(circuit, initial=None):
    """Execute reset and unread measurements as channels, retaining every outcome.

    Qiskit simulates the gates themselves. For compiled circuits, this is
    independent of the direct Petz formula and recovery-isometry contractions.
    """
    state = DensityMatrix.from_int(0, 2**circuit.num_qubits) if initial is None else DensityMatrix(initial)
    if state.num_qubits != circuit.num_qubits:
        raise ValueError("Initial state and circuit sizes differ.")
    for instruction in circuit.data:
        qargs = [circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "barrier":
            continue
        if instruction.operation.name == "measure":
            state = state.evolve(Kraus([np.diag([1., 0]), np.diag([0., 1])]), qargs)
        else:
            state = state.evolve(instruction.operation, qargs)
    return state


def simulate_compiled_sweep(bath, rho, compiled=None):
    compiled = compile_circuit(sweep_circuit(bath)) if compiled is None else compiled
    combined = np.kron(np.diag([1., 0]), rho)
    evolved = ensemble_circuit_density(compiled, combined)
    return partial_trace(evolved, [bath.n]).data


def compile_circuit(circuit):
    return transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1, seed_transpiler=1200)


def bit_flip_layer(n, probability):
    """Independent physical X errors, realized with one repeatedly reset ancilla."""
    if not isinstance(n, Integral) or not 1 <= n <= 8:
        raise ValueError("Require 1 <= n <= 8.")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Require probability in [0,1].")
    circuit = QuantumCircuit(QuantumRegister(n, "spin"), QuantumRegister(1, "bath"))
    for q in range(n):
        circuit.reset(n)
        circuit.ry(2*np.arcsin(np.sqrt(probability)), n)
        circuit.cx(n, q)
        circuit.reset(n)
    return circuit


def apply_bit_flips(rho, probability):
    rho = np.asarray(rho, complex)
    n = len(rho).bit_length()-1
    if rho.shape != (2**n, 2**n) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid density dimension or bit-flip probability.")
    result = rho.copy()
    indices = np.arange(len(rho))
    for q in range(n):
        flipped = indices ^ (1 << q)
        result = (1-probability)*result+probability*result[np.ix_(flipped, flipped)]
    return result


def compiled_isometry_error(update, compiled=None):
    """Exhaustive column check; intended for small cases (at most four spins)."""
    if update.n > 4:
        raise ValueError("Exhaustive compiled-operator check is limited to four spins.")
    compiled = compile_circuit(recovery_circuit(update)) if compiled is None else compiled
    matrix = Operator(compiled).data[:, :2**(update.n-1)]
    overlap = np.vdot(update.isometry, matrix)
    phase = overlap/abs(overlap)
    return float(np.linalg.norm(matrix-phase*update.isometry))


def elementary_isometry_columns(circuit, input_qubits):
    """Simulate ALL initialized input columns of a u3/cx isometry circuit.

    This avoids constructing the unused columns of a full 2**(n+1) unitary.
    It executes the exported elementary gates, independently of the recovery
    tensor. Tests compare it to Qiskit's Operator on small circuits.
    """
    if not isinstance(input_qubits, Integral) or not 0 <= input_qubits <= circuit.num_qubits:
        raise ValueError("Invalid input qubit count.")
    rows = np.arange(2**circuit.num_qubits)
    columns = 2**input_qubits
    matrix = np.zeros((len(rows), columns), complex)
    matrix[np.arange(columns), np.arange(columns)] = 1
    low = {q: rows[(rows & (1 << q)) == 0] for q in range(circuit.num_qubits)}
    for instruction in circuit.data:
        name = instruction.operation.name
        qs = [circuit.find_bit(q).index for q in instruction.qubits]
        if name == "u3":
            q, = qs
            lo, hi = low[q], low[q] | (1 << q)
            a, c = matrix[lo].copy(), matrix[hi].copy()
            u = instruction.operation.to_matrix()
            matrix[lo] = u[0,0]*a+u[0,1]*c
            matrix[hi] = u[1,0]*a+u[1,1]*c
        elif name == "cx":
            control, target = qs
            permutation = rows ^ (((rows >> control) & 1) << target)
            matrix = matrix[permutation]
        else:
            raise ValueError(f"Expected only u3/cx gates, got {name}.")
    return np.exp(1j*float(circuit.global_phase))*matrix
