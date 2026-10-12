"""Algorithm 3 adaptation: local Petz preparation, magnetization, freezing.

Uses the sign-consistent interpretation of MainV2.tex: seeds have m=-1/2,
negative active magnetizations join the frozen seed segment, and frozen
values generate h_i=sum_j Delta_ij m_j. Local preparation does NOT impose
the original global up-spin cap K. A separate capped-Gibbs oracle provides
the small-system reference. Original image normalization is kept throughout.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from algorithm1 import xxz_hamiltonian
from local_petz import PatchUpdate, apply_dense_update, route_locally
from quantum_glauber import (build_quantum_heat_bath, full_kraus, compile_circuit,
                            recovery_circuit, elementary_isometry_columns)


@dataclass
class ActiveModel:
    graph: object
    active: tuple  # compact tensor index -> original row-major pixel
    edges: tuple
    delta: tuple
    fields: np.ndarray
    b: float
    updates: list
    layers: list

    @property
    def n(self): return len(self.active)

    @property
    def order(self): return [i for layer in self.layers for i in layer]


def active_terms(graph, frozen):
    frozen = np.asarray(frozen, float)
    if frozen.shape != (graph.n,) or np.any(np.isinf(frozen)) or np.any(abs(frozen[np.isfinite(frozen)]) > .5):
        raise ValueError('One frozen magnetization in [-1/2,1/2], or NaN for active, per pixel.')
    active = tuple(np.flatnonzero(np.isnan(frozen)).tolist())
    index = {p: i for i, p in enumerate(active)}
    fields = np.zeros(len(active)); edges = []; delta = []
    for (p, q), d in zip(graph.edges, graph.delta):
        if p in index and q in index:
            edges.append((index[p], index[q])); delta.append(float(d))
        elif p in index:
            fields[index[p]] += d * frozen[q]
        elif q in index:
            fields[index[q]] += d * frozen[p]
    return active, tuple(edges), tuple(delta), fields


@lru_cache(maxsize=256)
def field_patch(size, edges, delta, fields, b, center):
    h = xxz_hamiltonian(size, np.array(edges, int).reshape(-1, 2), np.array(delta))
    basis = np.arange(2**size)
    z = 1 - 2 * ((basis[:, None] >> np.arange(size-1, -1, -1)) & 1)
    h[np.diag_indices_from(h)] -= z @ np.asarray(fields) / 2
    bath = build_quantum_heat_bath(h, b)
    update = bath.updates[center]
    return update, bath.sigma, h, tuple(full_kraus(update))


def build_active_petz(graph, frozen, b=6.):
    if not np.isfinite(b) or b < 0: raise ValueError('Require finite b >= 0.')
    active, edges, delta, fields = active_terms(graph, frozen)
    adjacency = [dict() for _ in active]
    for (p, q), d in zip(edges, delta): adjacency[p][q] = d; adjacency[q][p] = d
    updates = []
    for center in range(len(active)):
        sites = tuple(sorted([center, *adjacency[center]])); index = {p: i for i, p in enumerate(sites)}
        pe = []; pd = []
        for p in sites:
            for q, d in adjacency[p].items():
                if p < q and q in index: pe.append((index[p], index[q])); pd.append(d)
        u, sigma, h, kraus = field_patch(len(sites), tuple(pe), tuple(pd),
                                        tuple(fields[list(sites)]), float(b), index[center])
        updates.append(PatchUpdate(center, sites, tuple(pe), tuple(pd), u, sigma, h, kraus))
    layers = [[] for _ in range(5)]
    width = graph.image.shape[1]
    for i, pixel in enumerate(active):
        row, col = divmod(pixel, width); layers[(row + 2*col) % 5].append(i)
    layers = [layer for layer in layers if layer]
    for layer in layers:
        sites = [p for i in layer for p in updates[i].sites]
        if len(sites) != len(set(sites)): raise RuntimeError('Parallel patches overlap.')
    return ActiveModel(graph, active, edges, delta, fields, float(b), updates, layers)


def dense_local_magnetization(model, sweeps=50):
    if model.n > 9: raise ValueError('Exact density benchmark limited to nine active spins.')
    if not isinstance(sweeps, int) or sweeps < 1: raise ValueError('Positive integer sweep count required.')
    rho = np.ones((2**model.n, 2**model.n)) / 2**model.n  # fresh all +x
    for _ in range(sweeps):
        for i in model.order: rho = apply_dense_update(rho, model.updates[i], model.n)
    z = 1 - 2 * ((np.arange(2**model.n)[:, None] >> np.arange(model.n-1, -1, -1)) & 1)
    means = rho.diagonal().real @ z / 2
    return means, {'density_trace_error': float(abs(np.trace(rho)-1)), 'density': rho}


def capped_gibbs_magnetization(graph, frozen, b=6., k=3, max_dimension=2048):
    """Exact H^[K] reference without ever forming the full 2**N Hamiltonian.

Up means Z=+1 (bit 0); the retained basis has at most K up-spins. Every
sector, including the zero-up sector, is included with its thermal weight.
"""
    active, edges, delta, fields = active_terms(graph, frozen); n = len(active)
    if not isinstance(k, int) or k < 0 or not np.isfinite(b) or b < 0:
        raise ValueError('Nonnegative integer K and finite b >= 0 required.')
    dimension = sum(comb(n, j) for j in range(min(k, n)+1))
    if dimension > max_dimension: raise ValueError(f'Capped reference dimension {dimension} exceeds {max_dimension}.')
    states = [sum(1 << i for i in up) for j in range(min(k, n)+1) for up in combinations(range(n), j)]
    index = {state: i for i, state in enumerate(states)}
    spins = np.array([[.5 if state & (1 << i) else -.5 for i in range(n)] for state in states])
    h = np.diag(-spins @ fields)
    for (p, q), d in zip(edges, delta):
        h[np.diag_indices_from(h)] -= d * spins[:, p] * spins[:, q]
        for row, state in enumerate(states):
            if bool(state & (1 << p)) != bool(state & (1 << q)):
                h[index[state ^ (1 << p) ^ (1 << q)], row] -= .5
    energies, vectors = np.linalg.eigh(h)
    weights = np.exp(-b*(energies-energies[0])); weights /= weights.sum()
    diagonal = (abs(vectors)**2) @ weights
    return diagonal @ spins, {'dimension': dimension, 'k': k, 'basis_up_masks': states,
                              'hamiltonian': h, 'probabilities': diagonal}


def mps_plan(model):
    from local_petz_trajectories import lift_operator
    width = model.graph.image.shape[1]
    # Column order keeps the two-row strips in the existing rung order.
    order = sorted(range(model.n), key=lambda i: (model.active[i] % width, model.active[i] // width))
    inverse = np.argsort(order); plan = []
    for i in model.order:
        update = model.updates[i]; where = [int(inverse[p]) for p in update.sites]
        start, end = min(where), max(where)
        if end-start+1 > 12: raise ValueError('MPS window exceeds 12 spins; use a gate-MPO method for wider 2D grids.')
        plan.append((start, end, tuple(lift_operator(k, [p-start for p in where], end-start+1) for k in update.kraus)))
    return order, plan


def mps_local_magnetization(model, sweeps=50, replicas=128, bond=32, cutoff=1e-12, seed=1200):
    import quimb.tensor as qtn
    from local_petz_trajectories import Trajectory, apply_trajectory_sweep
    if replicas < 2 or sweeps < 1 or bond < 1 or cutoff < 0: raise ValueError('Invalid trajectory parameters.')
    order, plan = mps_plan(model); samples = []; diagnostics = []
    sz = np.diag([.5, -.5])
    for replica in range(replicas):
        state = qtn.MPS_product_state([np.ones(2)/np.sqrt(2) for _ in order])
        tr = Trajectory(state, np.random.default_rng(seed+replica), {'cur_orthog': (0, 0)})
        for _ in range(sweeps): apply_trajectory_sweep(tr, plan, bond, cutoff)
        if model.n == 1:
            vector = state.to_dense().ravel(); means = [np.vdot(vector, sz @ vector).real]
        else:
            values = state.compute_local_expectation_canonical({(i,): sz for i in range(model.n)}, return_all=True)
            means = [float(np.real(values[(i,)])) for i in range(model.n)]
        samples.append(np.array(means)[np.argsort(order)])
        diagnostics.append([tr.discarded_weight, tr.max_probability_sum_error, tr.svd_fallbacks])
    samples = np.array(samples); diagnostics = np.array(diagnostics)
    return samples.mean(axis=0), {'samples': samples, 'se': samples.std(axis=0, ddof=1)/np.sqrt(replicas),
        'max_discarded_weight_sum': float(diagnostics[:, 0].max()),
        'max_probability_sum_error': float(diagnostics[:, 1].max()),
        'svd_fallbacks': int(diagnostics[:, 2].sum())}


def magnetization_from_counts(counts, n):
    """Decode Qiskit Z strings: leftmost bit is compact active pixel 0."""
    total = 0; sums = np.zeros(n)
    for key, count in counts.items():
        bits = str(key).replace(' ', '')
        if len(bits) != n or any(b not in '01' for b in bits) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError('Require nonnegative integer counts of N-bit binary strings.')
        total += int(count); sums += count * (.5-np.array(list(bits), int))
    if total < 2: raise ValueError('At least two independent shots required.')
    means = sums/total
    return means, np.sqrt(np.maximum(0., .25-means**2)/(total-1)), total


def segment(graph, seeds, *, b=6., backend='dense', sweeps=50, k=3, replicas=128,
            bond=32, cutoff=1e-12, random_seed=1200, decision='strict', alpha=.05,
            sign_tolerance=1e-10, callback=None, mps_estimator=None, count_provider=None):
    """Fresh preparation each round, then synchronous irreversible freezing.

    Seeds all belong to one supplied segment; no truth labels are consulted.
    Strict: freeze m < -sign_tolerance (roundoff only). Resolved: additionally
    require a negative simultaneous Hoeffding upper bound across at most N^2
    adaptive decisions. These bounds control trajectory sampling, not MPS or
    channel bias. Frozen fields retain the measured value, not its sign alone.
    """
    if backend not in ('dense', 'mps', 'capped', 'counts'): raise ValueError('Use dense, mps, capped, or counts backend.')
    if backend == 'counts' and count_provider is None: raise ValueError('Counts backend needs a circuit count provider.')
    if decision not in ('strict', 'resolved') or not 0 < alpha < 1 or sign_tolerance < 0:
        raise ValueError('Invalid sign decision options.')
    seeds = tuple(int(p) for p in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or max(seeds) >= graph.n:
        raise ValueError('Supply distinct seed pixels belonging to one segment.')
    frozen = np.full(graph.n, np.nan); frozen[list(seeds)] = -.5
    frozen_round = np.full(graph.n, -1, int); frozen_round[list(seeds)] = 0
    rounds = []; max_rounds = graph.n-len(seeds)+1
    for iteration in range(max_rounds):
        active = np.flatnonzero(np.isnan(frozen))
        if not len(active): break
        before = frozen.copy()
        if backend == 'capped':
            means, details = capped_gibbs_magnetization(graph, frozen, b, k)
            se = np.zeros(len(active)); radius = 0.
        else:
            model = build_active_petz(graph, frozen, b)
            if backend == 'dense':
                means, details = dense_local_magnetization(model, sweeps)
                se = np.zeros(len(active)); radius = 0.
            elif backend == 'counts':
                counts = count_provider(model, sweeps, iteration)
                means, se, shots = magnetization_from_counts(counts, model.n)
                details = {'shots': shots, 'counts': counts}
                radius = float(np.sqrt(np.log(2*graph.n*max_rounds/alpha)/(2*shots)))
            else:
                estimator = mps_local_magnetization if mps_estimator is None else mps_estimator
                means, details = estimator(model, sweeps, replicas, bond, cutoff,
                                           random_seed+100000*iteration)
                se = details['se']
                radius = float(np.sqrt(np.log(2*graph.n*max_rounds/alpha)/(2*replicas)))
        upper = means + (radius if decision == 'resolved' else 0.)
        chosen = upper < -sign_tolerance; added = active[chosen]
        frozen[added] = means[chosen]; frozen_round[added] = iteration+1
        record = {'round': iteration+1, 'active': active.tolist(), 'magnetization': means.tolist(),
                  'se': se.tolist(), 'simultaneous_sampling_radius': radius,
                  'newly_frozen': added.tolist(), 'frozen_before': [None if np.isnan(x) else float(x) for x in before],
                  'uncertain_freezes': active[chosen & (means+radius >= -sign_tolerance)].tolist(),
                  'reused_samples': details.get('reused_samples', False),
                  'shots': details.get('shots'),
                  'max_discarded_weight_sum': details.get('max_discarded_weight_sum', 0.),
                  'max_probability_sum_error': details.get('max_probability_sum_error', 0.),
                  'reference_dimension': details.get('dimension')}
        rounds.append(record)
        if callback is not None: callback(record, details)
        if not len(added): break
    return {'backend': backend, 'decision': decision, 'b': b, 'beta_paper': b*graph.n,
            'sweeps_per_round': sweeps if backend != 'capped' else None,
            'k': k if backend == 'capped' else None, 'seeds': list(seeds),
            'replicas': replicas if backend == 'mps' else None, 'bond': bond if backend == 'mps' else None,
            'cutoff': cutoff if backend == 'mps' else None, 'random_seed': random_seed,
            'alpha': alpha, 'sign_tolerance': sign_tolerance, 'rounds': rounds,
            'same_as_seed': np.isfinite(frozen).reshape(graph.image.shape[:2]).tolist(),
            'frozen_round': frozen_round.reshape(graph.image.shape[:2]).tolist(),
            'frozen_magnetization': [None if np.isnan(x) else float(x) for x in frozen],
            'termination': 'all_frozen' if np.all(np.isfinite(frozen)) else 'no_new_freezes',
            'unresolved_active': [p for p, m in zip(rounds[-1]['active'], rounds[-1]['magnetization'])
                                  if abs(m) <= rounds[-1]['simultaneous_sampling_radius']+sign_tolerance]
                                  if rounds and not rounds[-1]['newly_frozen'] else []}


@lru_cache(maxsize=256)
def verified_field_recovery(size, edges, delta, fields, b, center):
    """Verify synthesized initialized columns; use a unitary completion if needed.

    The installed Qiskit isometry synthesis fails this check for some five-spin
    field-dependent patches. A QR-completed unitary provides a checked fallback.
    It increases the fixed patch cost; no unverified circuit is returned.
    """
    from qiskit.circuit.library import UnitaryGate
    update = field_patch(size, edges, delta, fields, b, center)[0]
    def error(circuit):
        columns = elementary_isometry_columns(circuit, size-1)
        overlap = np.vdot(update.isometry, columns)
        phase = overlap/abs(overlap) if abs(overlap) else 1.
        return float(np.linalg.norm(columns-phase*update.isometry))
    circuit = compile_circuit(recovery_circuit(update)); initial_error = error(circuit)
    method = 'isometry'
    if initial_error > 1e-8:
        v = update.isometry; q, _ = np.linalg.qr(v, mode='complete')
        unitary = np.column_stack([v, q[:, v.shape[1]:]])
        if np.linalg.norm(unitary.conj().T @ unitary-np.eye(len(unitary))) > 1e-10:
            raise RuntimeError('Invalid recovery unitary completion.')
        raw = QuantumCircuit(size+1); raw.append(UnitaryGate(unitary), range(size+1))
        circuit = compile_circuit(raw); method = 'verified_unitary_completion'
    final_error = error(circuit)
    if final_error > 1e-8: raise RuntimeError(f'Compiled recovery fails validation: {final_error}.')
    return circuit, {'method': method, 'initial_isometry_error': initial_error, 'verified_error': final_error}


@lru_cache(maxsize=256)
def compiled_field_patch(size, edges, delta, fields, b, center, routed):
    recovery, _ = verified_field_recovery(size, edges, delta, fields, b, center)
    circuit = QuantumCircuit(size+1)
    erased = size-1-center
    circuit.reset(size); circuit.swap(erased, size); circuit.reset(size)
    circuit.compose(recovery, [q for q in range(size) if q != erased]+[erased, size], inplace=True)
    circuit.reset(size); circuit = compile_circuit(circuit)
    if routed:
        connections = tuple((size-1-p, size-1-q) for p, q in edges) + ((size, size-1-center),)
        circuit = route_locally(circuit, connections)
    return circuit


def active_sweep_circuit(model, routed=True):
    """Elementary sweep on compact active registers, with local reset ancillas.

    Compact spin q=Nactive-1-i lives at original image pixel model.active[i].
    Thus every routed two-qubit gate still acts on a physical image edge or
    the center's attached ancilla. No connectivity from deleted spins is used.
    """
    if not model.n: raise ValueError('No circuit needed when all pixels are frozen.')
    circuit = QuantumCircuit(QuantumRegister(model.n, 'spin'), QuantumRegister(model.n, 'bath'))
    for layer in model.layers:
        for i in layer:
            u = model.updates[i]
            block = compiled_field_patch(len(u.sites), u.edges, u.delta, tuple(model.fields[list(u.sites)]),
                                         model.b, u.local.site, routed)
            circuit.compose(block, [model.n-1-p for p in reversed(u.sites)]+[2*model.n-1-i], inplace=True)
        circuit.barrier()
    return circuit


def preparation_circuit(model, sweeps=50, routed=True, measure=True):
    """Executable QPY circuit for one classically selected freezing round.

    A composite instruction reuses the fully elementary sweep definition;
    counting this instruction as one elementary gate would be incorrect.
    Reset and reprepare the state for every independent Z-measurement shot.
    """
    if not isinstance(sweeps, int) or sweeps < 1: raise ValueError('Require positive sweeps.')
    body = active_sweep_circuit(model, routed)
    circuit = body.copy_empty_like()
    circuit.h(range(model.n))
    instruction = body.to_instruction(); instruction.name = 'local_petz_sweep'
    for _ in range(sweeps): circuit.append(instruction, range(2*model.n))
    if measure:
        from qiskit import ClassicalRegister
        register = ClassicalRegister(model.n, 'z'); circuit.add_register(register)
        circuit.measure(range(model.n), register)
    return circuit
