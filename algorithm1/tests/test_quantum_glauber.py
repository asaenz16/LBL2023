import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit, qasm2
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace
from algorithm1 import image_model, xxz_hamiltonian
from quantum_glauber import (build_quantum_heat_bath, apply_update, apply_sweep,
    full_kraus, update_circuit, sweep_circuit, compile_circuit, recovery_circuit,
    compiled_isometry_error, ensemble_circuit_density, preparation_circuit,
    elementary_isometry_columns, bit_flip_layer, apply_bit_flips)


def random_density(d, seed=12):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d))+1j*rng.normal(size=(d, d))
    rho = a@a.conj().T
    return rho/np.trace(rho)


@pytest.mark.parametrize("b", [0., 1., 6.])
def test_cptp_and_full_gibbs_preservation(b):
    bath = build_quantum_heat_bath(image_model([[0,0],[255,255]]).hamiltonian, b)
    rho = random_density(16)
    for update in bath.updates:
        ks = full_kraus(update)
        assert_allclose(sum(k.conj().T@k for k in ks), np.eye(16), atol=2e-13)
        assert_allclose(apply_update(update, bath.sigma), bath.sigma, atol=2e-13)
        result = apply_update(update, rho)
        assert_allclose(result, sum(k@rho@k.conj().T for k in ks), atol=2e-13)
        assert_allclose(np.trace(result), 1, atol=2e-13)
        assert np.linalg.eigvalsh(result).min() >= -2e-13


def test_classical_limit_is_single_spin_conditional_heat_bath():
    # Frustrated three-site diagonal model with longitudinal fields: unequal
    # probabilities ensure both thermal sign and row-major placement matter.
    z = 1-2*((np.arange(8)[:,None] >> np.arange(2,-1,-1)) & 1)
    energy = -.7*z[:,0]*z[:,1]+.3*z[:,1]*z[:,2]-.2*z[:,0]+.4*z[:,2]
    b = 1.7
    bath = build_quantum_heat_bath(np.diag(energy), b)
    rng = np.random.default_rng(23)
    p = rng.random(8); p /= p.sum()
    for site, update in enumerate(bath.updates):
        expected = np.zeros(8)
        mask = 1 << (2-site)
        for j in range(8):
            if j & mask:
                continue
            pair = [j, j|mask]
            weights = np.exp(-b*energy[pair]); weights /= weights.sum()
            expected[pair] = p[pair].sum()*weights
        assert_allclose(apply_update(update, np.diag(p)), np.diag(expected), atol=1e-13)


def test_polar_construction_matches_direct_petz_formula_when_well_conditioned():
    from scipy.linalg import sqrtm
    bath = build_quantum_heat_bath(image_model([[0,0],[255,255]]).hamiltonian, .5)
    rho = random_density(16)
    for update in bath.updates:
        p = update.permutation
        target = bath.sigma[np.ix_(p,p)]
        marginal = target[:8,:8]+target[8:,8:]
        inv = np.linalg.inv(sqrtm(marginal))
        inputp = rho[np.ix_(p,p)]
        middle = inv@(inputp[:8,:8]+inputp[8:,8:])@inv
        root = sqrtm(target)
        expected = root@np.kron(np.eye(2),middle)@root
        actual = apply_update(update,rho)[np.ix_(p,p)]
        assert_allclose(actual,expected,atol=1e-12)


def test_entire_compiled_recovery_isometry_and_dynamic_sweep():
    bath = build_quantum_heat_bath(image_model([[0,0],[255,255]]).hamiltonian, 6.)
    for update in bath.updates:
        compiled_recovery = compile_circuit(recovery_circuit(update))
        assert compiled_isometry_error(update,compiled_recovery) < 2e-11
        from qiskit.quantum_info import Operator
        assert_allclose(elementary_isometry_columns(compiled_recovery,3),
                        Operator(compiled_recovery).data[:,:8],atol=1e-12)
    rho = random_density(16,83)
    compiled = compile_circuit(sweep_circuit(bath))
    result = ensemble_circuit_density(compiled, np.kron(np.diag([1.,0]),rho))
    assert_allclose(partial_trace(result,[4]).data, apply_sweep(bath,rho), atol=2e-11)
    assert_allclose(partial_trace(result,list(range(4))).data,np.diag([1.,0]),atol=2e-11)
    assert set(compiled.count_ops()) <= {"u3","cx","reset"}
    # Portable OpenQASM round trip retains all resets and elementary gates.
    restored = qasm2.loads(qasm2.dumps(compiled))
    repeat = ensemble_circuit_density(restored, np.kron(np.diag([1.,0]),rho))
    assert_allclose(repeat.data,result.data,atol=2e-11)


@pytest.mark.parametrize("kind", ["mixed","plus_x","ordered","corrupted"])
def test_physical_preparation_circuits(kind):
    n=4;z=np.array([1,1,-1,-1]); p=.1
    circuit=preparation_circuit(n,kind,z,corruption=p)
    state=ensemble_circuit_density(circuit)
    if kind=="mixed": expected=np.eye(16)/16
    elif kind=="plus_x": expected=np.ones((16,16))/16
    else:
        bits=1-2*((np.arange(16)[:,None]>>np.arange(3,-1,-1))&1)
        probability=np.zeros(16)
        for image in [z,-z]:
            if kind=="ordered": probability+=np.all(bits==image,axis=1)/2
            else: probability+=np.prod(np.where(bits==image,1-p,p),axis=1)/2
        expected=np.diag(probability)
    assert_allclose(partial_trace(state,[n]).data,expected,atol=1e-13)
    assert_allclose(partial_trace(state,list(range(n))).data,np.diag([1.,0]),atol=1e-13)


def test_actual_reset_circuit_shots_match_channel():
    bath=build_quantum_heat_bath(xxz_hamiltonian(2,[[0,1]],[-1.7]),2.)
    circuit=QuantumCircuit(3,2)
    circuit.h([0,1])
    circuit.compose(sweep_circuit(bath),inplace=True)
    circuit.measure([0,1],[0,1])
    compiled=compile_circuit(circuit)
    counts=BasicSimulator().run(compiled,shots=2048,seed_simulator=1200).result().get_counts()
    measured=np.array([counts.get(f"{j:02b}",0) for j in range(4)])/2048
    target=np.diag(apply_sweep(bath,np.ones((4,4))/4)).real
    assert np.max(abs(measured-target)) < .04


def test_boxed_qpy_export_round_trip_preserves_dynamic_channel():
    import io
    from qiskit import qpy
    from run_quantum_glauber import build_elementary_update, boxed_sweep
    bath=build_quantum_heat_bath(xxz_hamiltonian(2,[[0,1]],[-1.7]),2.)
    blocks=[build_elementary_update(compile_circuit(recovery_circuit(u)),2,u.site)
            for u in bath.updates]
    circuit=boxed_sweep(blocks)
    buffer=io.BytesIO();qpy.dump(circuit,buffer);buffer.seek(0)
    restored=qpy.load(buffer)[0]
    rho=random_density(4,56)
    actual=ensemble_circuit_density(restored,np.kron(np.diag([1.,0]),rho))
    assert_allclose(partial_trace(actual,[2]).data,apply_sweep(bath,rho),atol=1e-11)
    assert_allclose(partial_trace(actual,[0,1]).data,np.diag([1.,0]),atol=1e-11)


def test_bit_flip_circuit_matches_channel_and_zz_attenuation():
    rho=random_density(16, 64)
    p=.05
    result=ensemble_circuit_density(compile_circuit(bit_flip_layer(4,p)),
        np.kron(np.diag([1.,0]),rho))
    direct=apply_bit_flips(rho,p)
    assert_allclose(partial_trace(result,[4]).data,direct,atol=1e-12)
    z=1-2*((np.arange(16)[:,None]>>np.arange(3,-1,-1))&1)
    assert_allclose(np.diag(direct)@(z[:,0]*z[:,3]),
                    (1-2*p)**2*(np.diag(rho)@(z[:,0]*z[:,3])),atol=1e-12)


def test_eight_spin_full_model_preserves_gibbs_and_converges():
    pytest.importorskip("quimb")
    from ladder_tensor import build_ladder,dense_row_hamiltonian
    bath=build_quantum_heat_bath(dense_row_hamiltonian(build_ladder(4)),6.)
    rho=np.ones((256,256))/256
    for _ in range(60): rho=apply_sweep(bath,rho)
    assert np.linalg.norm(rho-bath.sigma) < 1e-7
    assert_allclose(np.trace(rho),1,atol=1e-11)
    assert max(u.isometry_error for u in bath.updates)<1e-11
    assert max(u.gibbs_error for u in bath.updates)<1e-11


@pytest.mark.parametrize("h,b", [(np.eye(3),1),(np.eye(2),-1),
    (np.array([[0,1],[0,0]]),1),(np.diag([0.,100.]),100.)])
def test_invalid_inputs_fail_explicitly(h,b):
    with pytest.raises(ValueError): build_quantum_heat_bath(h,b)
