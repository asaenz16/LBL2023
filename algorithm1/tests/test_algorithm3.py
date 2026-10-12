import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from local_petz import image_graph, build_local_petz, apply_dense_update
from algorithm1 import xxz_hamiltonian
from algorithm3 import (active_terms, build_active_petz, field_patch, capped_gibbs_magnetization,
                        dense_local_magnetization, mps_plan, segment, preparation_circuit, active_sweep_circuit)


def test_frozen_field_matches_independent_partial_expectation():
    graph = image_graph([[0, 0, 255], [0, 255, 255]])
    frozen = np.array([-.5, np.nan, np.nan, np.nan, np.nan, .25])
    active, edges, delta, fields = active_terms(graph, frozen)
    h = field_patch(4, edges, delta, tuple(fields), 1., 0)[2]
    full = xxz_hamiltonian(6, graph.edges, graph.delta)
    expected = np.zeros_like(h)
    for a in range(2):
        for b in range(2):
            indices = []
            for state in range(16):
                bits = [a] + [(state >> shift) & 1 for shift in range(3, -1, -1)] + [b]
                indices.append(sum(bit << (5-i) for i, bit in enumerate(bits)))
            weight = (.5+frozen[0] if a == 0 else .5-frozen[0]) * (.5+frozen[5] if b == 0 else .5-frozen[5])
            expected += weight * full[np.ix_(indices, indices)]
    expected -= np.trace(expected-h)/16 * np.eye(16)
    assert_allclose(h, expected, atol=1e-13)


def test_unfrozen_model_matches_existing_local_petz():
    graph = image_graph([[0, 0, 255], [0, 255, 255]])
    old = build_local_petz(graph, 2.)
    new = build_active_petz(graph, np.full(6, np.nan), 2.)
    assert old.layers == new.layers
    for a, b in zip(old.updates, new.updates):
        assert a.sites == b.sites
        assert_allclose(a.hamiltonian, b.hamiltonian, atol=1e-14)
        assert_allclose(a.kraus, b.kraus, atol=1e-13)


def test_cap_counts_up_spins_and_full_cap_matches_gibbs():
    graph = image_graph([[0, 0], [255, 255]])
    frozen = np.array([-.5, np.nan, np.nan, np.nan])
    means, details = capped_gibbs_magnetization(graph, frozen, b=0., k=1)
    assert details['dimension'] == 4
    assert_allclose(means, -.25, atol=1e-14)
    active, edges, delta, fields = active_terms(graph, frozen)
    h = field_patch(3, edges, delta, tuple(fields), 1.2, 0)[2]
    rho = expm(-1.2*h); rho /= np.trace(rho)
    z = 1 - 2*((np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1)
    exact = rho.diagonal() @ z / 2
    actual, _ = capped_gibbs_magnetization(graph, frozen, b=1.2, k=3)
    assert_allclose(actual, exact, atol=1e-13)


def test_freezing_is_synchronous_monotone_and_keeps_measured_fields(monkeypatch):
    import algorithm3
    graph = image_graph([[0, 0], [255, 255]])
    seen = []
    def oracle(model, sweeps):
        seen.append(model)
        return (np.array([-.2, .3, -.1]) if model.n == 3 else np.array([.3])), {}
    monkeypatch.setattr(algorithm3, 'dense_local_magnetization', oracle)
    result = segment(graph, [0], b=1.)
    assert result['rounds'][0]['newly_frozen'] == [1, 3]
    assert result['rounds'][1]['newly_frozen'] == []
    assert result['same_as_seed'] == [[True, True], [False, True]]
    assert result['frozen_magnetization'] == [-.5, -.2, None, -.1]
    _, _, _, expected = active_terms(graph, [-.5, -.2, np.nan, -.1])
    assert_allclose(seen[1].fields, expected)


def test_zero_magnetization_is_not_frozen():
    graph = image_graph([[0, 0], [255, 255]])
    result = segment(graph, [0], b=0., sweeps=2)
    assert result['same_as_seed'] == [[True, False], [False, False]]
    assert result['rounds'][0]['newly_frozen'] == []


def test_patches_with_fields_are_cptp_and_local_on_large_image():
    r, c = np.indices((16, 16)); graph = image_graph(255*((r//2+c//2) % 2))
    frozen = np.full(256, np.nan); frozen[::7] = -.3
    model = build_active_petz(graph, frozen, b=2.)
    assert model.n < 256 and len(model.layers) <= 5
    for u in model.updates:
        assert len(u.sites) <= 5
        assert_allclose(sum(k.conj().T @ k for k in u.kraus), np.eye(2**len(u.sites)), atol=1e-12)
        assert_allclose(sum(k @ u.sigma @ k.conj().T for k in u.kraus), u.sigma, atol=1e-12)
        for p in u.sites:
            if p == u.center: continue
            a, b = model.active[p], model.active[u.center]
            assert abs(a//16-b//16)+abs(a%16-b%16) == 1


def test_single_remaining_spin_trajectory():
    import quimb.tensor as qtn
    from local_petz_trajectories import Trajectory, apply_trajectory_update
    graph = image_graph([[0, 0], [255, 255]])
    model = build_active_petz(graph, [-.5, np.nan, -.2, -.3], b=2.)
    _, plan = mps_plan(model); result = np.zeros((2, 2), complex)
    for outcome in range(4):
        class Forced:
            def choice(self, n, p): self.probability = p[outcome]; return outcome
        rng = Forced()
        state = qtn.MPS_product_state([np.ones(2)/np.sqrt(2)])
        tr = Trajectory(state, rng, {'cur_orthog': (0, 0)})
        apply_trajectory_update(tr, plan[0], max_bond=32)
        vector = state.to_dense().ravel()
        result += rng.probability * np.outer(vector, vector.conj())
    assert_allclose(result, model.updates[0].sigma, atol=1e-12)


def test_compact_mps_with_holes_matches_dense_instrument():
    import quimb.tensor as qtn
    from local_petz_trajectories import Trajectory, apply_trajectory_update
    graph = image_graph([[0, 0, 255, 255], [0, 0, 255, 255]])
    frozen = np.full(8, np.nan); frozen[[0, 4]] = [-.5, -.2]
    model = build_active_petz(graph, frozen, b=2.)
    order, plan = mps_plan(model); rng = np.random.default_rng(1200)
    vector = rng.normal(size=2**model.n)+1j*rng.normal(size=2**model.n); vector /= np.linalg.norm(vector)
    reordered = vector.reshape([2]*model.n).transpose(order).ravel()
    selected = 2; expected = apply_dense_update(np.outer(vector, vector.conj()), model.updates[model.order[selected]], model.n)
    result = np.zeros_like(expected)
    for outcome in range(4):
        class Forced:
            def choice(self, n, p): self.probability = p[outcome]; return outcome
        choice = Forced(); state = qtn.MatrixProductState.from_dense(reordered, dims=[2]*model.n, cutoff=0)
        tr = Trajectory(state, choice, {'cur_orthog': 'calc'})
        apply_trajectory_update(tr, plan[selected], max_bond=64, cutoff=0)
        observed = state.to_dense().reshape([2]*model.n).transpose(np.argsort(order)).ravel()
        result += choice.probability*np.outer(observed, observed.conj())
    assert_allclose(result, expected, atol=1e-12)


def test_routed_round_circuit_and_qpy_match_density(tmp_path):
    from qiskit import qpy
    from qiskit.quantum_info import partial_trace
    from quantum_glauber import ensemble_circuit_density
    graph = image_graph([[0, 0], [255, 255]])
    model = build_active_petz(graph, [-.5, np.nan, np.nan, np.nan], b=2.)
    circuit = preparation_circuit(model, sweeps=2, measure=False)
    path = tmp_path/'round.qpy'
    with path.open('wb') as f: qpy.dump(circuit, f)
    with path.open('rb') as f: loaded = qpy.load(f)[0]
    actual = partial_trace(ensemble_circuit_density(loaded), range(model.n, 2*model.n)).data
    _, reference = dense_local_magnetization(model, 2)
    assert_allclose(actual, reference['density'], atol=1e-10)
    physical = [(model.n-1-p, model.n-1-q) for p, q in model.edges]
    physical += [(model.n-1-i, 2*model.n-1-i) for i in range(model.n)]
    allowed = {frozenset(edge) for edge in physical}
    for instruction in active_sweep_circuit(model).data:
        if instruction.operation.name == 'cx':
            assert frozenset(circuit.find_bit(q).index for q in instruction.qubits) in allowed


def test_invalid_parameters():
    graph = image_graph([[0, 0], [255, 255]])
    with pytest.raises(ValueError): active_terms(graph, [-.5, np.nan])
    with pytest.raises(ValueError): active_terms(graph, [-.6, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError): segment(graph, [])
    with pytest.raises(ValueError): segment(graph, [0, 0])
    with pytest.raises(ValueError): capped_gibbs_magnetization(graph, [np.nan]*4, k=3, max_dimension=2)


def test_counts_bit_order_and_feedback():
    from algorithm3 import magnetization_from_counts
    mean, se, shots = magnetization_from_counts({'001': 3, '110': 1}, 3)
    assert_allclose(mean, [.25, .25, -.25]); assert shots == 4
    assert_allclose(se, .25)
    graph = image_graph([[0, 0], [255, 255]])
    def counts(model, sweeps, iteration):
        return {'100': 10000} if iteration == 0 else {'00': 10000}
    result = segment(graph, [0], backend='counts', count_provider=counts, decision='resolved')
    assert result['same_as_seed'] == [[True, True], [False, False]]
    assert result['unresolved_active'] == []
    with pytest.raises(ValueError): magnetization_from_counts({'01': 1}, 2)
    with pytest.raises(ValueError): magnetization_from_counts({'0x': 2}, 2)


def test_resolved_rule_defers_uncertain_signs():
    graph = image_graph([[0, 0], [255, 255]])
    def oracle(*args): return np.array([-.01, .01, .4]), {'se': np.array([.05, .05, .01])}
    result = segment(graph, [0], backend='mps', decision='resolved', mps_estimator=oracle, replicas=64)
    assert result['rounds'][0]['newly_frozen'] == []
    assert result['unresolved_active'] == [1, 2]


@pytest.mark.parametrize('name', ['center9', 'L9'])
def test_five_spin_field_recovery_synthesis_on_arbitrary_input(name):
    from algorithm3 import verified_field_recovery, compiled_field_patch
    from run_algorithm3 import benchmark
    from qiskit.quantum_info import partial_trace
    from quantum_glauber import ensemble_circuit_density
    image, _, seeds = benchmark(name); graph = image_graph(image)
    frozen = np.full(9, np.nan); frozen[seeds] = -.5
    model = build_active_petz(graph, frozen, b=6.)
    u = model.updates[model.active.index(4)]
    args = (len(u.sites), u.edges, u.delta, tuple(model.fields[list(u.sites)]), model.b, u.local.site)
    _, check = verified_field_recovery(*args)
    assert check['verified_error'] < 1e-8
    rng = np.random.default_rng(814); vector = rng.normal(size=32)+1j*rng.normal(size=32); vector /= np.linalg.norm(vector)
    rho = np.outer(vector, vector.conj())
    initial = np.kron(np.diag([1., 0.]), rho)
    output = ensemble_circuit_density(compiled_field_patch(*args, routed=True), initial)
    assert_allclose(partial_trace(output, [5]).data, apply_dense_update(rho, u, 5), atol=1e-9)
