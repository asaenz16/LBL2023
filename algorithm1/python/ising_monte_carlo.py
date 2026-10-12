"""Signed Ising Gibbs sampling: heat bath + signed clusters + parallel tempering.

Sign-compatible convention: Delta=atan(alpha*(threshold-d)), alpha>0,
H=-sum Delta*Z_i*Z_j/4. This module takes b=beta_paper/N explicitly.
No spin is pinned. A seed is only a reference for a two-point observable.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/private/tmp/quantum-segmentation-numba')
from dataclasses import dataclass
import numpy as np
from numba import njit
from ising_algorithm1 import grid_edges


@dataclass
class SignedImage:
    image: np.ndarray
    edges: np.ndarray
    distances: np.ndarray
    delta: np.ndarray
    alpha: float
    neighbors: np.ndarray
    couplings: np.ndarray
    threshold: float = 125.

    @property
    def n(self): return int(np.prod(self.image.shape[:2]))


def signed_image(image, alpha, *, threshold=125.):
    image = np.asarray(image, dtype=float)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('Use grayscale or RGB images.')
    if image.size == 0 or not np.isfinite(image).all() or np.any((image < 0) | (image > 255)):
        raise ValueError('Raw image channels must be finite in [0,255].')
    if not np.isfinite(alpha) or alpha <= 0: raise ValueError('Sensitivity alpha must be positive.')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('The edge threshold must be finite and nonnegative.')
    n = int(np.prod(image.shape[:2])); colors = image.reshape(n, -1)
    edges = grid_edges(*image.shape[:2])
    distances = np.linalg.norm(colors[edges[:, 0]]-colors[edges[:, 1]], axis=1)
    delta = np.arctan(alpha*(threshold-distances))
    neighbors = np.full((n, 4), -1, dtype=np.int64); couplings = np.zeros((n, 4))
    counts = np.zeros(n, dtype=int)
    for (p, q), d in zip(edges, delta):
        for i, j in [(p, q), (q, p)]:
            neighbors[i, counts[i]] = j; couplings[i, counts[i]] = d/4; counts[i] += 1
    return SignedImage(image.copy(), edges, distances, delta, float(alpha), neighbors, couplings, float(threshold))


def image_parameter_rule(image, coefficient, reduced_coupling):
    """Image-statistic rule with coefficients calibrated externally on training data.

q is the median distance from the fixed 125 edge threshold, over nonconstant
edges. Alpha=coefficient/q; b sets the median dimensionless coupling. This is
a starting point, not a guarantee that the couplings encode useful segments.
"""
    if not np.isfinite(reduced_coupling) or reduced_coupling<=0:
        raise ValueError('The reduced coupling must be finite and positive.')
    if not np.isfinite(coefficient) or coefficient<=0:
        raise ValueError('The sensitivity coefficient must be finite and positive.')
    initial=signed_image(image,.01)
    margins=abs(125-initial.distances[initial.distances>0])
    q=max(float(np.median(margins)) if len(margins) else 125.,1.)
    alpha=coefficient/q;model=signed_image(image,alpha)
    median_delta=float(np.median(abs(model.delta))) if len(model.delta) else 0.
    if median_delta<=4e-12:
        raise ValueError('Median coupling is zero: revise the model rather than choosing an enormous beta.')
    b=4*reduced_coupling/median_delta
    return dict(alpha=alpha,b=b,beta_paper=b*model.n,margin_median=q,
                median_abs_delta=median_delta,negative_edges=int(np.sum(model.delta<0)),
                applicability='all-ferromagnetic; cannot form multiple sign-based segments'
                    if np.all(model.delta>0) else 'mixed signs; validate correlations and boundary suitability')


def temperature_ladder(model, targets):
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 1 or len(targets) == 0 or not np.isfinite(targets).all() or np.any(targets < 0):
        raise ValueError('Targets must be a nonempty vector of finite b>=0.')
    # More replicas near the ordering transition; include b=0 and every target.
    scale = max(float(np.max(abs(model.delta), initial=0.))/4, 1e-8)
    end = targets.max()*scale
    reduced = np.arange(0, min(end, 1.)+.00000001, .04)
    if end > 1: reduced = np.r_[reduced, np.geomspace(1, end, int(np.ceil(np.log(end)/.12))+1)]
    return np.unique(np.r_[reduced/scale, targets])


@njit(cache=True)
def energy(z, edges, j):
    value = 0.
    for e in range(len(j)): value -= j[e]*z[edges[e, 0]]*z[edges[e, 1]]
    return value


@njit(cache=True)
def _root(parent, p):
    while parent[p] != p:
        parent[p] = parent[parent[p]]; p = parent[p]
    return p


@njit(cache=True)
def cluster_step(z, edges, j, probabilities, seeds):
    """Signed SW move and conditional (Rao-Blackwell) seed correlations.

Open only satisfied edges, with P(open)=1-exp(-2*b*abs(J)). Components
flip independently; relative signs within a component are inherited from z.
Conditional <Z_seed Z_p> is their relative sign if connected and zero otherwise.
This identity holds for mixed signs too: occupied cycles cannot be frustrated.
"""
    n = len(z); parent = np.arange(n); size = np.ones(n, dtype=np.int64)
    for e in range(len(j)):
        p, q = edges[e]
        if j[e]*z[p]*z[q] > 0 and np.random.random() < probabilities[e]:
            p = _root(parent, p); q = _root(parent, q)
            if p != q:
                if size[p] < size[q]: p, q = q, p
                parent[q] = p; size[p] += size[q]
    roots = np.empty(n, np.int64)
    for p in range(n): roots[p] = _root(parent, p)
    corr = np.zeros((len(seeds), n))
    for s in range(len(seeds)):
        seed = seeds[s]
        for p in range(n):
            if roots[p] == roots[seed]: corr[s, p] = z[seed]*z[p]/4.
    flips = np.empty(n, np.int8)
    for p in range(n): flips[p] = 1 if np.random.random() < .5 else -1
    for p in range(n): z[p] *= flips[roots[p]]
    return corr


@njit(cache=True)
def _run_chain(edges, j, neighbors, couplings, bs, targets, seeds,
               burn, samples, block_size, rng_seed, initial):
    np.random.seed(rng_seed)
    n = len(neighbors); nt = len(bs); nobs = len(targets); ns = len(seeds)
    states = np.ones((nt, n), np.int8)
    if initial == 0:
        for t in range(nt):
            for p in range(n): states[t, p] = 1 if np.random.random() < .5 else -1
    probabilities = np.empty((nt, len(j)))
    for t in range(nt): probabilities[t] = -np.expm1(-2*bs[t]*np.abs(j))
    target_slot = np.full(nt, -1, dtype=np.int64)
    for i in range(nobs): target_slot[targets[i]] = i
    blocks = np.zeros((samples//block_size, nobs, ns, n))
    series = np.zeros((samples, nobs, 3))  # E/N, mean Gamma, mean |conditional Gamma|
    energy_blocks = np.zeros((samples//block_size, nobs))
    swap_attempts = np.zeros(nt-1, np.int64); swap_accepts = np.zeros(nt-1, np.int64)
    # Count completed hot->cold->hot journeys of labeled replicas, after warmup.
    tags = np.arange(nt); phase = np.zeros(nt, np.int8); trips = np.zeros(nt, np.int64)
    energies = np.zeros(nt)
    for step in range(burn+samples):
        for t in range(nt):
            z = states[t]
            # Systematic-scan heat bath: composition preserves Gibbs, even though
            # a whole sweep need not itself be reversible. No field or seed clamp.
            for p in range(n):
                h = 0.
                for k in range(4):
                    q = neighbors[p, k]
                    if q >= 0: h += couplings[p, k]*z[q]
                prob = (1+np.tanh(bs[t]*h))/2
                z[p] = 1 if np.random.random() < prob else -1
            corr = cluster_step(z, edges, j, probabilities[t], seeds)
            energies[t] = energy(z, edges, j)
            oi = target_slot[t]
            if step >= burn and oi >= 0:
                it = step-burn; ib = it//block_size
                blocks[ib, oi] += corr/block_size
                energy_blocks[ib, oi] += energies[t]/n/block_size
                series[it, oi, 0] = energies[t]/n
                series[it, oi, 1] = np.mean(corr[0])
                series[it, oi, 2] = np.mean(np.abs(corr[0]))
        # Adjacent exchange ratio for exp(-b E): (b_i-b_j)*(E_i-E_j).
        for t in range(step % 2, nt-1, 2):
            if step >= burn: swap_attempts[t] += 1
            log_ratio = (bs[t]-bs[t+1])*(energies[t]-energies[t+1])
            if log_ratio >= 0 or np.log(np.random.random()) < log_ratio:
                tmp = states[t].copy(); states[t] = states[t+1]; states[t+1] = tmp
                tag = tags[t]; tags[t] = tags[t+1]; tags[t+1] = tag
                if step >= burn: swap_accepts[t] += 1
        if step >= burn:
            tag = tags[0]
            if phase[tag] == 2: trips[tag] += 1
            phase[tag] = 1
            if phase[tags[-1]] == 1: phase[tags[-1]] = 2
    return blocks, energy_blocks, series, swap_accepts, swap_attempts, trips


def split_rhat(samples):
    """Unranked split R-hat across axis-0 chains and axis-1 observations.

Constant, agreeing series are marked 1; disagreements with zero within-chain
variance are infinity. This diagnostic is necessary, not proof of convergence.
"""
    samples = np.asarray(samples, dtype=float); half = samples.shape[1]//2
    if half < 2: raise ValueError('Need >=4 observations for split R-hat.')
    split = np.concatenate([samples[:, :half], samples[:, -half:]], axis=0)
    w = split.var(axis=1, ddof=1).mean(axis=0)
    between = half*split.mean(axis=1).var(axis=0, ddof=1)
    v = (half-1)/half*w+between/half
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sqrt(v/w)
    return np.where(w == 0, np.where(between == 0, 1., np.inf), result)


def effective_samples(x):
    """Single-chain FFT autocorrelation, Geyer initial-positive paired sums.

Zero-variance observables return zero ESS (no empirical mixing information).
"""
    x = np.asarray(x, float); x = x-x.mean(); n = len(x)
    if not np.any(x): return 0.
    f = np.fft.rfft(x, n=2*n); ac = np.fft.irfft(f*f.conj(), n=2*n)[:n]
    ac /= ac[0]; total = 0.; last = np.inf
    # Pairs rho(0)+rho(1), rho(2)+rho(3), ...; tau=-1+2*sum pairs.
    for t in range(0, n-1, 2):
        pair = min(float(ac[t]+ac[t+1]), last)
        if pair <= 0: break
        total += pair; last = pair
    return float(min(n, n/max(1., -1+2*total)))


def sample_gibbs(model, bs, seeds=(0,), *, burn=1024, samples=2048,
                 block_size=64, chains=2, rng_seed=20261007):
    if burn < 0 or samples < 4*block_size or samples % block_size or chains < 2:
        raise ValueError('Use >=2 chains, >=4 full blocks and nonnegative burn-in.')
    bs = np.asarray(bs, float); seeds = np.asarray(seeds, dtype=np.int64)
    if len(np.unique(bs)) != len(bs):raise ValueError('Requested temperatures must be distinct.')
    if seeds.ndim != 1 or len(seeds) == 0 or np.any((seeds < 0) | (seeds >= model.n)):
        raise ValueError('Invalid reference spins.')
    ladder = temperature_ladder(model, bs)
    targets = np.searchsorted(ladder, bs)
    outputs = [_run_chain(model.edges, model.delta/4, model.neighbors, model.couplings,
        ladder, targets, seeds, burn, samples, block_size, rng_seed+1000003*c, c % 2)
        for c in range(chains)]
    blocks, energies, series, accepts, attempts, trips = [np.stack(x) for x in zip(*outputs)]
    gamma = blocks.mean(axis=(0, 1))
    # Batch SE plus between-chain discrepancy safeguard. Neither certifies bias.
    batch_se = blocks.reshape(-1, *blocks.shape[2:]).std(axis=0, ddof=1)/np.sqrt(chains*len(blocks[0]))
    chain_se = blocks.mean(axis=1).std(axis=0, ddof=1)/np.sqrt(chains)
    se = np.maximum(batch_se, chain_se)
    with np.errstate(invalid='ignore', divide='ignore'):
        swap = accepts.sum(axis=0)/attempts.sum(axis=0)
    ess = np.array([[effective_samples(series[c, :, t, k]) for k in range(3)]
                    for c in range(chains) for t in range(len(bs))]).reshape(chains, len(bs), 3).sum(axis=0)
    return dict(gamma=gamma, se=se, blocks=blocks, energy_blocks=energies,
        series=series, bs=bs, beta_paper=bs*model.n, ladder=ladder, seeds=seeds,
        swap_acceptance=swap, round_trips=trips,
        rhat_observables=split_rhat(series), rhat_gamma=split_rhat(blocks), ess_observables=ess,
        chain_linf=np.ptp(blocks.mean(axis=1), axis=0).max(axis=-1),
        burn=burn, samples=samples, block_size=block_size, chains=chains, rng_seed=rng_seed)
