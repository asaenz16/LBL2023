import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import DensityMatrix, Kraus, partial_trace

from algorithm1 import image_model, seed_correlations, xxz_hamiltonian
from thermal_bath import (apply_contact, apply_sweep, build_bath, channel_matrix,
                          contact_circuit, davies_generator, reference_gibbs,
                          thermal_contact, trace_distance)


def test_one_spin_thermalization_matches_analytic_population():
    # E0=0, E1=1; a fully excited spin exchanges energy with the thermal bath.
    bath = build_bath(np.diag([0., 1.]), beta=2, axes="X", dt=0.2)
    contact, = bath.contacts
    rho = np.diag([0., 1.])
    evolved = apply_contact(contact, rho)
    p1 = 1/(1+np.exp(2))
    expected_p1 = 1 - (1-p1)*np.sin(np.sqrt(0.4))**2
    assert_allclose(evolved, np.diag([1-expected_p1, expected_p1]), atol=1e-14)
    for _ in range(150):
        rho = apply_sweep(bath, rho)
    assert_allclose(rho, np.diag([1-p1,p1]), atol=1e-12)


def test_glauber_rates_have_correct_sign_and_beta_over_n():
    bath = build_bath(xxz_hamiltonian(2, [[0,1]], [-1.5]), beta=3)
    for contact in bath.contacts:
        assert_allclose(contact.excited_probability/(1-contact.excited_probability),
                        np.exp(-1.5*contact.omega), atol=1e-13)
        a = contact.lowering
        assert_allclose(bath.hamiltonian@a-a@bath.hamiltonian,
                        -contact.omega*a, atol=1e-12)


@pytest.mark.parametrize("beta", [0, 2, 24, 200])
def test_each_contact_is_trace_preserving_and_gibbs_preserving(beta):
    bath = build_bath(xxz_hamiltonian(2, [[0,1]], [2.0]), beta)
    rho = reference_gibbs(bath)
    for contact in bath.contacts:
        assert_allclose(sum(k.conj().T@k for k in contact.kraus), np.eye(4), atol=1e-12)
        assert_allclose(apply_contact(contact, rho), rho, atol=1e-12)
        hb = np.diag([0., contact.omega])
        free = np.kron(np.eye(2),bath.hamiltonian)+np.kron(hb,np.eye(4))
        assert_allclose(contact.unitary@free, free@contact.unitary, atol=1e-12)
    assert_allclose(davies_generator(bath)@rho.ravel(order="F"), 0, atol=1e-12)


def test_zero_frequency_retains_full_degenerate_block():
    bath = build_bath(np.zeros((4,4)), beta=10, dt=0.1)
    assert len(bath.contacts) == 6
    assert all(contact.omega == 0 for contact in bath.contacts)
    state = np.diag([1.,0,0,0])
    for _ in range(100):
        state = apply_sweep(bath,state)
    assert_allclose(state,np.eye(4)/4,atol=1e-12)


def test_small_contact_limit_agrees_with_davies_generator():
    h = xxz_hamiltonian(2, [[0,1]], [-0.7])
    errors = []
    for dt in [0.004,0.002,0.001]:
        bath = build_bath(h,3,dt=dt)
        approximation = (channel_matrix(bath)-np.eye(16))/dt
        errors.append(np.linalg.norm(approximation-davies_generator(bath)))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2


def test_circuit_measurement_and_reset_match_contact_channel():
    bath = build_bath(xxz_hamiltonian(2,[[0,1]],[1.7]),beta=3)
    psi = np.array([1,1j,2,-1j],complex)/np.sqrt(7)
    initial = np.outer(psi,psi.conj())
    for contact in [bath.contacts[0], next(c for c in bath.contacts if c.omega > 0)]:
        circuit = contact_circuit(contact,2)
        state = DensityMatrix(np.kron(np.diag([1.,0]),initial))
        for instruction in circuit.data:
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            if instruction.operation.name == "measure":
                # Exact ensemble of both measurement outcomes; no postselection.
                state = state.evolve(Kraus([np.diag([1,0]),np.diag([0,1])]),qubits)
            else:
                state = state.evolve(instruction.operation,qubits)
        assert_allclose(partial_trace(state,[2]).data,apply_contact(contact,initial),atol=1e-12)
        assert_allclose(partial_trace(state,[0,1]).data,np.diag([1,0]),atol=1e-12)


def test_actual_dynamic_circuit_shots_match_thermal_contact():
    contact = build_bath(np.diag([0.,1.]),2,axes="X",dt=0.4).contacts[0]
    circuit = QuantumCircuit(2,2)
    circuit.x(0)
    circuit.compose(contact_circuit(contact,1),qubits=[0,1],clbits=[0],inplace=True)
    circuit.measure(0,1)
    circuit = transpile(circuit,basis_gates=["u","cx"],optimization_level=1)
    counts = BasicSimulator().run(circuit,shots=8192,seed_simulator=1200).result().get_counts()
    excited = sum(count for bits,count in counts.items() if bits[0] == "1")/8192
    expected = apply_contact(contact,np.diag([0.,1.]))[1,1].real
    assert abs(excited-expected) < 0.025


def test_z_only_coupling_fails_to_escape_polarized_sector():
    model = image_model([[0,0],[255,255]])
    bad = build_bath(model.hamiltonian,24,axes="Z")
    polarized = np.diag([1.]+[0.]*15)
    assert_allclose(apply_sweep(bad,polarized),polarized,atol=1e-12)
    assert trace_distance(polarized,reference_gibbs(bad)) > 0.5


def test_noncommuting_four_pixel_model_converges_from_distinct_states():
    model = image_model([[0,0],[255,255]])
    bath = build_bath(model.hamiltonian,24)
    channel = channel_matrix(bath)
    target = reference_gibbs(bath)
    psi = np.ones(16)/4
    initials = [np.diag([1.]+[0.]*15), np.eye(16)/16, np.outer(psi,psi)]
    states = np.column_stack([rho.ravel(order="F") for rho in initials])
    for _ in range(60):
        states = channel@states
    for vector in states.T:
        rho = vector.reshape(16,16,order="F")
        assert trace_distance(rho,target) < 1e-8
        assert np.linalg.eigvalsh(rho).min() > -1e-12
        assert_allclose(np.trace(rho),1,atol=1e-11)
        assert_allclose(seed_correlations(rho,0),seed_correlations(target,0),atol=1e-8)


def test_global_flip_changes_magnetization_but_not_zz_correlations():
    h = xxz_hamiltonian(2, [[0,1]], [2.0])
    ordinary = build_bath(h,20)
    assisted = build_bath(h,20,global_flip=True)
    rho = np.diag([1.,0,0,0])
    left,right = apply_sweep(ordinary,rho),apply_sweep(assisted,rho)
    assert trace_distance(left,right) > 0.01
    assert_allclose(seed_correlations(left,0),seed_correlations(right,0),atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"beta":-1}, {"dt":0}, {"axes":"ZZ"},
                                    {"energy_tolerance":0}])
def test_invalid_bath_parameters(kwargs):
    params = {"beta":1,**kwargs}
    with pytest.raises(ValueError):
        build_bath(np.diag([0.,1.]),**params)


def test_invalid_zero_frequency_contact():
    with pytest.raises(ValueError,match="Hermitian"):
        thermal_contact("bad",0,np.array([[0,1],[0,0]]),1,0.1)
