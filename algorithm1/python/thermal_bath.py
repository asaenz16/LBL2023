"""Frequency-resolved thermal collisions, with an exact Davies reference.

Based on the Davies/Glauber definitions in arXiv:2410.01206v2, Sec. 3.
The finite-contact dilation below is our adaptation, not the toric-code
algorithm or a claim of efficient energy resolution for general XXZ models.
Dense spectral preprocessing is intentionally limited to four system spins.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import expm
from scipy.special import expit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate

PAULIS = {"X": np.array([[0, 1], [1, 0]], complex),
          "Y": np.array([[0, -1j], [1j, 0]], complex),
          "Z": np.diag([1, -1]).astype(complex)}


def local_pauli(n, site, axis):
    result = np.array([[1]], complex)
    for p in range(n):
        result = np.kron(result, PAULIS[axis] if p == site else np.eye(2))
    return result


def _validate_h(h):
    h = np.asarray(h, complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H must be square.")
    n = h.shape[0].bit_length() - 1
    if not 1 <= n <= 4 or h.shape[0] != 2**n:
        raise ValueError("The dense bath reference supports 1 to 4 system qubits.")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.conj().T, atol=1e-12, rtol=0):
        raise ValueError("H must be finite and Hermitian.")
    return h, n


@dataclass
class Contact:
    label: str
    omega: float
    lowering: np.ndarray
    excited_probability: float
    unitary: np.ndarray  # bath tensor system; Qiskit bath is qubit N
    kraus: tuple[np.ndarray, ...]


@dataclass
class ThermalBath:
    hamiltonian: np.ndarray
    beta: float
    n: int
    dt: float
    contacts: list[Contact]
    energy_tolerance: float
    operator_tolerance: float

    @property
    def beta_effective(self):
        return self.beta / self.n


def thermal_contact(label, omega, lowering, beta_effective, dt):
    """An energy-conserving contact with a fresh thermal qubit.

For omega>0, [H,A]=-omega*A and V=|1><0|_b tensor A + h.c.
The bath gap is omega, p1=1/(1+exp(beta_effective*omega)).
Use exp(-i sqrt(2dt) V); its small-dt generator is
2*(1-p1)*D[A] + 2*p1*D[A^dagger]. At omega=0, A is Hermitian,
and exp(-i sqrt(dt) X_b tensor A) gives D[A], counted only once.
"""
    if not np.isfinite(omega) or omega < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require omega >= 0 and finite dt > 0.")
    if not np.isfinite(beta_effective) or beta_effective < 0:
        raise ValueError("Require finite beta_effective >= 0.")
    a = np.asarray(lowering, complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError("The transition operator must be a finite square matrix.")
    if omega == 0 and not np.allclose(a, a.conj().T, atol=1e-12, rtol=0):
        raise ValueError("A zero-frequency contact requires a Hermitian operator.")
    d = len(a)
    v = np.block([[np.zeros_like(a), a.conj().T], [a, np.zeros_like(a)]])
    angle = np.sqrt(dt if omega == 0 else 2*dt)
    unitary = expm(-1j*angle*v)
    p1 = float(expit(-beta_effective * omega))
    kraus = tuple(np.sqrt(probability) * unitary[out*d:(out+1)*d, inc*d:(inc+1)*d]
                  for inc, probability in enumerate([1-p1, p1]) for out in range(2))
    return Contact(label, float(omega), a, p1, unitary, kraus)


def build_bath(hamiltonian, beta, *, dt=0.1, axes="XYZ", global_flip=False,
               energy_tolerance=1e-10, operator_tolerance=1e-12):
    """Resolve local coupling operators into lowering/zero-frequency parts.

Equal Bohr frequencies are grouped; degeneracies retain full matrix blocks,
not individual rank-one jumps. Tolerances remove numerical eigensolver noise.
Default XYZ proposals span the local Pauli algebra. Z-only is exposed for
the conservation-law counterexample and does not guarantee thermalization.
"""
    h, n = _validate_h(hamiltonian)
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require finite beta >= 0 and dt > 0.")
    if not axes or any(axis not in PAULIS for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError("axes must be a nonempty subset of XYZ without repeats.")
    if (not np.isfinite(energy_tolerance) or energy_tolerance <= 0
            or not np.isfinite(operator_tolerance) or operator_tolerance <= 0):
        raise ValueError("Tolerances must be finite and positive.")
    energies, u = np.linalg.eigh(h)
    # Positive gap means column/source energy is ABOVE row/target energy.
    gaps = energies[None, :] - energies[:, None]
    zero = np.abs(gaps) <= energy_tolerance
    group = np.full(gaps.shape, -1, int)
    frequencies = []
    remaining = sorted((float(gaps[i,j]), i, j)
                       for i, j in zip(*np.where(gaps > energy_tolerance)))
    anchor = -np.inf
    for gap, i, j in remaining:
        if gap - anchor > energy_tolerance:
            anchor = gap
            frequencies.append(gap)
        group[i, j] = len(frequencies)-1
    proposals = [(f"{axis}{site}", local_pauli(n, site, axis))
                 for site in range(n) for axis in axes]
    if global_flip:
        flip = np.array([[1]], complex)
        for _ in range(n):
            flip = np.kron(flip, PAULIS["X"])
        if np.linalg.norm(h@flip-flip@h) > 1e-10:
            raise ValueError("The proposed global X flip must commute with H.")
        proposals.append(("global_X", flip))
    contacts = []
    for label, proposal in proposals:
        matrix = u.conj().T @ proposal @ u
        a0 = u @ np.where(zero, matrix, 0) @ u.conj().T
        a0 = (a0+a0.conj().T)/2
        reconstructed = a0.copy()
        if np.linalg.norm(a0) > operator_tolerance:
            contacts.append(thermal_contact(label, 0, a0, beta/n, dt))
        for index, frequency in enumerate(frequencies):
            a = u @ np.where(group == index, matrix, 0) @ u.conj().T
            reconstructed += a+a.conj().T
            if np.linalg.norm(a) > operator_tolerance:
                contacts.append(thermal_contact(label, frequency, a, beta/n, dt))
        if not np.allclose(reconstructed, proposal, atol=1e-9, rtol=0):
            raise RuntimeError("Frequency decomposition failed to reconstruct the coupling.")
    return ThermalBath(h, float(beta), n, float(dt), contacts,
                       float(energy_tolerance), float(operator_tolerance))


def apply_contact(contact, rho):
    return sum(k @ rho @ k.conj().T for k in contact.kraus)


def apply_sweep(bath, rho):
    result = np.asarray(rho, complex).copy()
    for contact in bath.contacts:
        result = apply_contact(contact, result)
    return result


def channel_matrix(bath):
    """Dense sweep superoperator, column-major vectorization (MATLAB rho(:))."""
    d = 2**bath.n
    channel = np.eye(d*d, dtype=complex)
    for contact in bath.contacts:
        step = sum(np.kron(k.conj(), k) for k in contact.kraus)
        channel = step @ channel
    return channel


def davies_generator(bath):
    """Schrodinger-picture dissipator; coherent H evolution is omitted.

This is the adjoint of the paper's Heisenberg-picture generator. The paper
uses positive omega for energy-RAISING operators, the opposite sign to our
positive lowering-gap convention. Therefore its gamma(-omega)=2*p0 here.
"""
    d = 2**bath.n
    identity = np.eye(d)
    result = np.zeros((d*d, d*d), complex)
    for contact in bath.contacts:
        a, p1 = contact.lowering, contact.excited_probability
        jumps = [a] if contact.omega == 0 else [np.sqrt(2*(1-p1))*a, np.sqrt(2*p1)*a.conj().T]
        for jump in jumps:
            decay = jump.conj().T @ jump
            result += (np.kron(jump.conj(), jump)
                       - 0.5*np.kron(identity, decay) - 0.5*np.kron(decay.T, identity))
    return result


def reference_gibbs(bath):
    h = bath.hamiltonian
    shifted = h - np.linalg.eigvalsh(h)[0]*np.eye(len(h))
    rho = expm(-bath.beta_effective * shifted)
    return rho / np.trace(rho)


def contact_circuit(contact, n):
    """One contact on N system qubits + one reusable bath qubit.

The bath measurement is NONSELECTIVE: its result is unused. RY alone would
prepare a coherent superposition, not a thermal mixed state. The measurement
dephases it to diag(p0,p1); no runs are postselected. Reset removes system-bath
correlations at the end. Equivalent hardware option: choose |0> or |1> with
classical probabilities p0,p1 and initialize the bath accordingly.
"""
    if not isinstance(n, Integral) or not 1 <= n <= 4 or contact.unitary.shape != (2**(n+1),)*2:
        raise ValueError("Contact dimension must match 1 to 4 system qubits.")
    system = QuantumRegister(n, "sysreg")
    bath = QuantumRegister(1, "bath")
    scratch = ClassicalRegister(1, "bath_draw")
    circuit = QuantumCircuit(system, bath, scratch, name=f"bath_{contact.label}")
    circuit.reset(bath[0])
    circuit.ry(2*np.arcsin(np.sqrt(contact.excited_probability)), bath[0])
    circuit.measure(bath[0], scratch[0])
    circuit.append(UnitaryGate(contact.unitary, label=f"exchange {contact.label}"),
                   list(system)+list(bath))
    circuit.reset(bath[0])
    return circuit


def sweep_circuit(bath, sweeps=1):
    if not isinstance(sweeps, Integral) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer.")
    circuit = QuantumCircuit(QuantumRegister(bath.n, "sysreg"), QuantumRegister(1, "bath"),
                             ClassicalRegister(1, "bath_draw"), name="thermal_bath_sweeps")
    for _ in range(sweeps):
        for contact in bath.contacts:
            circuit.compose(contact_circuit(contact, bath.n), inplace=True)
    return circuit


def trace_distance(rho, sigma):
    return float(np.linalg.svd(rho-sigma, compute_uv=False).sum()/2)
