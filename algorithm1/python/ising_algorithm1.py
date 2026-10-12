"""Exact diagonal-Ising Gibbs reference for the modified Algorithm 1.

H = -sum_edges Delta S^z_p S^z_q = -sum_edges Delta Z_p Z_q / 4,
Delta = arctan(alpha * (Euclidean_color_distance - 125)).
The formula is used literally, including its sign. There are no XX, YY,
fields, frozen seeds, color rescaling, or sigma normalization. Enumeration
is exponential but avoids both diagonalization and a 2**N by 2**N matrix.
"""
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
import numpy as np

MAX_PIXELS = 20


@lru_cache(maxsize=2)
def spin_basis(n):
    """Row-major pixels, pixel 0 the most significant bit, Z=+/-1."""
    if not isinstance(n, Integral) or not 1 <= n <= MAX_PIXELS:
        raise ValueError(f'Exact enumeration supports 1 to {MAX_PIXELS} pixels.')
    states = np.arange(2**n, dtype=np.uint64)
    bits = (states[:, None] >> np.arange(n-1, -1, -1, dtype=np.uint64)) & 1
    z = 1.-2.*bits
    z.setflags(write=False)
    return z


def grid_edges(rows, cols):
    return np.array([(r*cols+c, q) for r in range(rows) for c in range(cols)
                     for q in ([r*cols+c+1] if c+1 < cols else [])
                     + ([(r+1)*cols+c] if r+1 < rows else [])], dtype=int).reshape(-1, 2)


def component_labels(n, edges, delta):
    """Components of nonzero couplings; separate components have zero <ZZ>."""
    parent = list(range(n))
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]; p = parent[p]
        return p
    for (p, q), d in zip(edges, delta):
        if d != 0: parent[find(int(p))] = find(int(q))
    return np.array([find(p) for p in range(n)])


@dataclass
class IsingImage:
    image: np.ndarray
    edges: np.ndarray
    distances: np.ndarray
    delta: np.ndarray
    alpha: float
    threshold: float
    energies: np.ndarray
    components: np.ndarray

    @property
    def n(self): return int(np.prod(self.image.shape[:2]))


def ising_model(image, alpha, *, threshold=125.):
    image = np.asarray(image, dtype=float)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('Use a grayscale matrix or (rows, columns, 3) RGB array.')
    if 0 in image.shape or not np.isfinite(image).all() or np.any((image < 0) | (image > 255)):
        raise ValueError('Supply finite raw color channels in [0,255]; no rescaling is applied.')
    if not np.isfinite(alpha) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Require finite alpha and a finite nonnegative threshold.')
    rows, cols = image.shape[:2]; n = rows*cols; z = spin_basis(n)
    edges = grid_edges(rows, cols); colors = image.reshape(n, -1)
    distances = np.linalg.norm(colors[edges[:, 0]]-colors[edges[:, 1]], axis=1)
    delta = np.arctan(float(alpha)*(distances-threshold))
    energies = np.zeros(2**n)
    for (p, q), d in zip(edges, delta): energies -= d*z[:, p]*z[:, q]/4
    return IsingImage(image.copy(), edges, distances, delta, float(alpha),
                      float(threshold), energies, component_labels(n, edges, delta))


@dataclass
class IsingGibbs:
    weights: np.ndarray
    gamma: np.ndarray
    magnetization: np.ndarray
    beta_effective: float
    log_partition: float
    mean_energy: float
    numerical_magnetization_residual: float
    numerical_disconnected_correlation_residual: float


def gibbs_ising(model, beta):
    """Full, unconditioned Gibbs state exp[-(beta/N)H]/Z; beta is paper beta.

Every computational-basis state is included. Known exact symmetries enforce
zero one-spin means and zero correlations between disconnected components.
This prevents roundoff-sized structural zeros becoming apparent sign data.
"""
    if not np.isfinite(beta) or beta < 0: raise ValueError('Require finite beta >= 0.')
    b = float(beta)/model.n; e = model.energies; emin = float(e.min())
    weights = np.exp(-b*(e-emin)); shifted_partition = float(weights.sum())
    weights /= shifted_partition; z = spin_basis(model.n)
    gamma = z.T @ (weights[:, None]*z)/4
    gamma = (gamma+gamma.T)/2
    residual = float(np.max(abs(weights @ z / 2)))
    disconnected = model.components[:, None] != model.components[None, :]
    disconnected_error = float(np.max(abs(gamma[disconnected]))) if disconnected.any() else 0.
    gamma[disconnected] = 0.
    if b == 0: gamma[:] = 0.
    np.fill_diagonal(gamma, .25)
    return IsingGibbs(weights, gamma, np.zeros(model.n), b,
        float(np.log(shifted_partition)-b*emin), float(weights @ e), residual, disconnected_error)


def algorithm1_labels(gamma, seed=0):
    """Original rule: positive correlation means same class; zero is opposite."""
    gamma = np.asarray(gamma)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1] or not np.isfinite(gamma).all():
        raise ValueError('Supply a finite square correlation matrix.')
    if not isinstance(seed, Integral) or not 0 <= seed < len(gamma):
        raise ValueError('Invalid zero-based seed pixel.')
    return gamma[seed] > 0
