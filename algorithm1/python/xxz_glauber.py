"""XXZ path-space heat-bath sampler using stochastic series expansion (SSE).

This adapts the *sampling* idea of Martinelli--Wouts (arXiv:1105.5970), not
their transverse-Ising single-site generator. Exchange vertices require
linked paths: we use the directed-loop heat-bath solution of Syljuasen and
Sandvik, Phys. Rev. E 66, 046701 (2002). Sweeps are not physical bath time.

H = -sum_e (XX + YY + delta_e ZZ)/4, and W_e = C_e I - h_e.
All allowed vertex weights are positive, even for negative delta_e.
No imaginary-time mesh, tensor truncation, or magnetization restriction.
The fixed expansion cutoff has a conservative, reported tail bound.
"""
from dataclasses import dataclass
from numbers import Integral
import os
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "quantum-segmentation-numba"))
import numpy as np
from numba import njit
from scipy.stats import poisson


@dataclass
class XXZGraph:
    n: int
    edges: np.ndarray
    delta: np.ndarray
    shift: np.ndarray
    weights: np.ndarray
    exits: np.ndarray
    image: np.ndarray | None = None
    sigma: float | None = None


def vertex_weights(delta, shift):
    """Four-bit code [in_i,in_j,out_i,out_j], bit 0 means spin +1."""
    weights = np.zeros(16)
    weights[0] = weights[15] = shift + delta / 4
    weights[5] = weights[10] = shift - delta / 4
    weights[6] = weights[9] = .5
    return weights


def heatbath_exits(weights):
    """P(exit | vertex, entrance) proportional to the resulting vertex weight.

    Include entrance==exit (bounce). Flipping both legs maps reverse and
    forward choices to the same normalization, giving W(v)P=W(v')P'.
    """
    p = np.zeros((16, 4, 4))
    for v in range(16):
        if weights[v] <= 0:
            continue
        for entrance in range(4):
            for leave in range(4):
                changed = v ^ (1 << (3 - entrance)) ^ (1 << (3 - leave))
                p[v, entrance, leave] = weights[changed]
            p[v, entrance] /= p[v, entrance].sum()
    return p


def graph(n, edges, delta, epsilon=.25):
    edges = np.asarray(edges)
    delta = np.asarray(delta, dtype=float)
    if not isinstance(n, Integral) or n < 1 or edges.shape != (len(delta), 2) or not len(delta):
        raise ValueError("Require a positive spin count and an (E,2) edge array.")
    if not np.all(edges == np.floor(edges)) or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("Invalid spin indices.")
    edges = edges.astype(np.int64)
    if any(i == j for i, j in edges) or len({tuple(sorted(e)) for e in edges}) != len(edges):
        raise ValueError("List each non-self undirected edge exactly once.")
    if not np.all(np.isfinite(delta)) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Require finite anisotropies and epsilon > 0.")
    shift = np.abs(delta) / 4 + epsilon
    weights = np.array([vertex_weights(d, c) for d, c in zip(delta, shift)])
    exits = np.array([heatbath_exits(w).cumsum(axis=-1) for w in weights])
    return XXZGraph(n, edges, delta, shift, weights, exits)


def image_graph(width, kappa=2., normalization="sample", epsilon=.25):
    """Same image, unique-edge sigma, row-major indices and couplings as the MPS code."""
    if not isinstance(width, Integral) or width < 4 or width % 4:
        raise ValueError("Require width=4N for positive integer N.")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Require finite nonnegative kappa.")
    image = np.tile(255 * ((np.arange(width) // 2) % 2), (2, 1))
    edges = np.array([(c, width+c) for c in range(width)] +
                     [(r*width+c, r*width+c+1) for r in range(2) for c in range(width-1)])
    contrasts = np.abs(image.ravel()[edges[:, 0]] - image.ravel()[edges[:, 1]])
    if normalization not in ("sample", "bulk"):
        raise ValueError("normalization must be sample or bulk.")
    sigma = np.std(contrasts, ddof=1) if normalization == "sample" else 255*np.sqrt(2)/3
    result = graph(2*width, edges, 2-kappa*contrasts/sigma, epsilon)
    result.image, result.sigma = image, float(sigma)
    return result


def expansion_cutoff(model, b, log_tolerance=-50.):
    """Bound omitted partition weight / full weight using a row-sum bound R.

    Tr exp(b W) >= 2**n exp(b sum C) by Jensen, while
    Tr W**k <= 2**n R**k. Hence relative tail <=
    exp[b(R-sum C)] Pr{Poisson(bR)>M}. This also bounds total variation
    between full and truncated path measures (not a mixing bound).
    """
    if not np.isfinite(b) or b <= 0:
        raise ValueError("Require finite positive effective inverse temperature b.")
    r = np.sum(np.maximum(model.shift + model.delta/4,
                          model.shift - model.delta/4 + .5))
    mean = b*r
    cutoff = max(32, int(np.ceil(mean)))
    while b*(r-model.shift.sum()) + poisson.logsf(cutoff, mean) > log_tolerance:
        cutoff += max(4, int(np.sqrt(mean)))
    return cutoff, float(b*(r-model.shift.sum()) + poisson.logsf(cutoff, mean))


@njit(cache=True)
def _choose(weights, total):
    target = np.random.random() * total
    acc = 0.
    for i in range(len(weights)):
        acc += weights[i]
        if target < acc:
            return i
    return len(weights)-1


@njit(cache=True)
def _initialize(spins, edges, delta, shift, b, capacity):
    weights = np.empty(len(edges))
    for e in range(len(edges)):
        i, j = edges[e]
        weights[e] = shift[e] + delta[e]*spins[i]*spins[j]/4
    total = weights.sum()
    order = np.random.poisson(b*total)
    while order > capacity:
        order = np.random.poisson(b*total)
    operators = np.zeros(capacity, np.int64)
    positions = np.random.permutation(capacity)
    for p in positions[:order]:
        operators[p] = 2*_choose(weights, total) + 2
    return operators


@njit(cache=True)
def _diagonal_sweep(spins, operators, edges, delta, shift, b):
    """Exact heat-bath conditional at each string slot, keeping off-diagonals.

    Given the rest of the string with n_rest operators, the possible weights
    are M-n_rest (identity), and b*W_e(s,s) for each diagonal bond e.
    This includes replacement of one diagonal bond label by another.
    """
    state = spins.copy()
    count = np.count_nonzero(operators)
    weights = np.empty(len(edges))
    for e in range(len(edges)):
        i, j = edges[e]
        weights[e] = shift[e] + delta[e]*state[i]*state[j]/4
    total = weights.sum()
    for p in range(len(operators)):
        op = operators[p]
        if op & 1:
            i, j = edges[(op-2)//2]
            state[i] *= -1
            state[j] *= -1
            # Small bounded-degree graphs; this simple recomputation is safe.
            for e in range(len(edges)):
                a, c = edges[e]
                weights[e] = shift[e] + delta[e]*state[a]*state[c]/4
            total = weights.sum()
        else:
            rest = count - (1 if op else 0)
            probability = b*total/(len(operators)-rest+b*total)
            if np.random.random() < probability:
                operators[p] = 2*_choose(weights, total) + 2
                count = rest + 1
            else:
                operators[p] = 0
                count = rest
    if np.any(state != spins):
        raise RuntimeError("Nonperiodic SSE string.")
    return count


@njit(cache=True)
def _loop_sweep(spins, operators, edges, exit_cdf, nloops):
    """A FIXED number of complete loops, not a state-dependent stopping budget."""
    positions = np.flatnonzero(operators)
    nv = len(positions)
    vertices = np.empty(nv, np.int64)
    bonds = np.empty(nv, np.int64)
    links = np.empty(4*nv, np.int64)
    first = np.full(len(spins), -1, np.int64)
    last = first.copy()
    state = spins.copy()
    for v in range(nv):
        op = operators[positions[v]]
        bond = (op-2)//2
        bonds[v] = bond
        i, j = edges[bond]
        code = ((1-state[i])//2)*8 + ((1-state[j])//2)*4
        if op & 1:
            if state[i] == state[j]:
                raise RuntimeError("Illegal parallel-spin exchange.")
            state[i] *= -1
            state[j] *= -1
        code += ((1-state[i])//2)*2 + ((1-state[j])//2)
        vertices[v] = code
        for k in range(2):
            site = edges[bond, k]
            leg = 4*v+k
            if last[site] < 0:
                first[site] = leg
            else:
                links[last[site]] = leg
                links[leg] = last[site]
            last[site] = leg+2
    for site in range(len(spins)):
        if first[site] >= 0:
            links[last[site]] = first[site]
            links[first[site]] = last[site]
    steps = 0
    bounces = 0
    for _ in range(nloops if nv else 0):
        start = np.random.randint(4*nv)
        entrance_leg = start
        loop_steps = 0
        while True:
            v, entrance = entrance_leg//4, entrance_leg % 4
            code = vertices[v]
            r = np.random.random()
            leave = 0
            while leave < 3 and r >= exit_cdf[bonds[v], code, entrance, leave]:
                leave += 1
            vertices[v] = code ^ (1 << (3-entrance)) ^ (1 << (3-leave))
            steps += 1
            loop_steps += 1
            bounces += int(leave == entrance)
            exit_leg = 4*v+leave
            # A return to the starting leg closes the defect before traversal;
            # arriving there via its temporal link also closes the loop.
            if exit_leg == start:
                break
            entrance_leg = links[exit_leg]
            if entrance_leg == start:
                break
            if loop_steps > 10000000:
                raise RuntimeError("Loop limit reached: abort, never accept an unfinished loop.")
    for v in range(nv):
        code = vertices[v]
        off_diagonal = int((code >> 2) != (code & 3))
        operators[positions[v]] = 2*bonds[v] + 2 + off_diagonal
    for site in range(len(spins)):
        if first[site] < 0:
            spins[site] = 1 if np.random.random() < .5 else -1
        else:
            leg = first[site]
            spins[site] = 1-2*((vertices[leg//4] >> (3-leg%4)) & 1)
    return steps, bounces


@njit(cache=True)
def _observe(spins, operators, edges, seeds, histogram_size=0):
    """Imaginary-time diagonal estimator, integrating out ordered event times.

    For n events, the n+1 gaps have conditional mean b/(n+1). The initial
    and final states coincide and both are included. Non-equilibrium values
    describe this specified path ensemble, not a physical quantum channel.
    """
    state = spins.copy()
    gamma = np.empty((len(seeds), len(spins)))
    for k in range(len(seeds)):
        gamma[k] = state[seeds[k]]*state/4.
    means = state.astype(np.float64)/2
    histogram = np.zeros(histogram_size)
    if histogram_size:
        index = 0
        for spin in state:
            index = 2*index + (1-spin)//2
        histogram[index] += 1
    count = 0
    for op in operators:
        if not op:
            continue
        count += 1
        if op & 1:
            i, j = edges[(op-2)//2]
            state[i] *= -1
            state[j] *= -1
        for k in range(len(seeds)):
            gamma[k] += state[seeds[k]]*state/4.
        means += state/2.
        if histogram_size:
            index = 0
            for spin in state:
                index = 2*index + (1-spin)//2
            histogram[index] += 1
    if np.any(state != spins):
        raise RuntimeError("Loop update broke periodicity.")
    return gamma/(count+1), means/(count+1), count, histogram/(count+1)


@njit(cache=True)
def _run(edges, delta, shift, exit_cdf, initial, b, capacity, times, seeds, nloops, seed, histogram_size):
    np.random.seed(seed)
    spins = initial.copy()
    operators = _initialize(spins, edges, delta, shift, b, capacity)
    profiles = np.empty((len(times), len(seeds), len(spins)))
    means = np.empty((len(times), len(spins)))
    orders = np.empty(len(times), np.int64)
    sectors = np.empty(len(times), np.int64)
    work = np.zeros((len(times), 2), np.int64)
    histograms = np.empty((len(times), histogram_size))
    maximum_order = np.count_nonzero(operators)
    t = 0
    total_steps, total_bounces = 0, 0
    for k in range(len(times)):
        while t < times[k]:
            count = _diagonal_sweep(spins, operators, edges, delta, shift, b)
            maximum_order = max(maximum_order, count)
            steps, bounces = _loop_sweep(spins, operators, edges, exit_cdf, nloops)
            total_steps += steps
            total_bounces += bounces
            t += 1
        profiles[k], means[k], orders[k], histograms[k] = _observe(spins, operators, edges, seeds, histogram_size)
        sectors[k] = spins.sum()
        work[k, 0], work[k, 1] = total_steps, total_bounces
    return profiles, means, orders, sectors, work, histograms, maximum_order


def run_chain(model, initial, b, times, seeds=(0,), seed=1200, nloops=None, capacity=None, record_distribution=False):
    initial = np.asarray(initial, dtype=np.int64)
    times, seeds = np.asarray(times), np.asarray(seeds)
    if initial.shape != (model.n,) or not np.all(np.isin(initial, [-1, 1])):
        raise ValueError("Initial Z paths must be +/-1, one per spin.")
    if times.ndim != 1 or not len(times) or np.any(times < 0) or np.any(np.diff(times) <= 0) or not np.all(times == np.floor(times)):
        raise ValueError("Use strictly increasing nonnegative integer sweep checkpoints.")
    if seeds.ndim != 1 or not len(seeds) or not np.all(seeds == np.floor(seeds)) or np.any(seeds < 0) or np.any(seeds >= model.n):
        raise ValueError("Invalid row-major seeds.")
    safe_capacity, _ = expansion_cutoff(model, b)
    capacity = safe_capacity if capacity is None else capacity
    if not isinstance(capacity, Integral) or capacity < safe_capacity:
        raise ValueError("Capacity must be at least the conservative expansion cutoff.")
    nloops = model.n if nloops is None else nloops
    if not isinstance(nloops, Integral) or nloops < 1:
        raise ValueError("Use a fixed positive integer loop count per sweep.")
    if record_distribution and model.n > 8:
        raise ValueError("Full Z-distribution output is limited to eight spins.")
    return _run(model.edges, model.delta, model.shift, model.exits, initial, float(b),
                capacity, times.astype(np.int64), seeds.astype(np.int64), nloops, seed,
                2**model.n if record_distribution else 0)


def exact_gibbs(model, b, seeds=(0,)):
    """Independent dense XXZ benchmark, limited to ten spins."""
    if model.n > 10:
        raise ValueError("Dense validation supports at most ten spins.")
    n = model.n
    basis = np.arange(2**n)
    z = 1-2*((basis[:, None] >> np.arange(n-1, -1, -1)) & 1)
    h = np.zeros((len(basis), len(basis)))
    for (i, j), delta in zip(model.edges, model.delta):
        h[basis, basis] -= delta*z[:, i]*z[:, j]/4
        selected = basis[z[:, i] != z[:, j]]
        h[selected ^ (1 << (n-1-i)) ^ (1 << (n-1-j)), selected] -= .5
    energy, vec = np.linalg.eigh(h)
    weights = np.exp(-b*(energy-energy[0]))
    weights /= weights.sum()
    probabilities = abs(vec)**2 @ weights
    gamma = np.array([(probabilities*z[:, s]) @ z/4 for s in seeds])
    return {"hamiltonian": h, "gamma": gamma, "energy": float(energy@weights),
            "z_probabilities": probabilities, "ground_energy": float(energy[0]),
            "magnetization_second_moment": float(probabilities @ (z.sum(axis=1)/2)**2)}
