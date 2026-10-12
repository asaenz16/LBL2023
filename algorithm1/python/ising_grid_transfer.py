"""Exact finite-temperature Ising correlations by row transfer, width <= 16.

The between-row matrix is a tensor product of 2x2 matrices. Applying it by
butterflies costs O(width * 2**width), without storing a dense transfer matrix.
The contraction is exponential only in the shorter side, not the pixel count.
All public seed/pixel indices preserve the repository's row-major convention.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/private/tmp/quantum-segmentation-numba')
import numpy as np
from numba import njit


@njit(cache=True)
def _cross(vector, same, different):
    out = vector.copy()
    for c in range(len(same)):
        bit = 1 << c
        for base in range(0, len(out), 2*bit):
            for j in range(bit):
                a, b = out[base+j], out[base+j+bit]
                out[base+j] = same[c]*a + different[c]*b
                out[base+j+bit] = different[c]*a + same[c]*b
    return out


@njit(cache=True)
def _contract(horizontal, vertical, b, seeds):
    rows, width = horizontal.shape[0], vertical.shape[1]
    dim = 1 << width
    spins = np.empty((dim, width))
    for state in range(dim):
        for c in range(width):
            spins[state,c] = 1-2*((state >> c) & 1)
    weights = np.zeros((rows, dim))
    for r in range(rows):
        for state in range(dim):
            for c in range(width-1):
                weights[r,state] += b*horizontal[r,c]*spins[state,c]*spins[state,c+1]
        largest = weights[r].max()
        weights[r] = np.exp(weights[r]-largest)
    same = np.exp(b*vertical-np.abs(b*vertical))
    different = np.exp(-b*vertical-np.abs(b*vertical))
    f = np.empty((rows,dim)); g = np.empty((rows,dim))
    fnorm = np.ones(rows); gnorm = np.ones(rows)
    f[0] = weights[0]/weights[0].sum(); g[-1] = np.ones(dim)/dim
    for r in range(1,rows):
        v = weights[r]*_cross(f[r-1],same[r-1],different[r-1])
        fnorm[r] = v.sum(); f[r] = v/fnorm[r]
    for r in range(rows-2,-1,-1):
        v = _cross(weights[r+1]*g[r+1],same[r],different[r])
        gnorm[r] = v.sum(); g[r] = v/gnorm[r]
    gamma = np.zeros((len(seeds),rows*width))
    for si in range(len(seeds)):
        sr, sc = seeds[si]//width, seeds[si]%width
        tag = f[sr]*spins[:,sc]
        for r in range(sr,rows):
            if r > sr:
                tag = weights[r]*_cross(tag,same[r-1],different[r-1])/fnorm[r]
            denominator = (f[r]*g[r]).sum()
            for c in range(width):
                gamma[si,r*width+c] = (tag*spins[:,c]*g[r]).sum()/denominator/4
        tag = g[sr]*spins[:,sc]
        for r in range(sr-1,-1,-1):
            tag = _cross(weights[r+1]*tag,same[r],different[r])/gnorm[r]
            denominator = (f[r]*g[r]).sum()
            for c in range(width):
                gamma[si,r*width+c] = (f[r]*spins[:,c]*tag).sum()/denominator/4
        gamma[si,seeds[si]] = .25
    return gamma


def exact_grid_gibbs(model, seeds=(0,)):
    """Exact Gibbs seed correlations; no equilibration or Monte Carlo error."""
    shape = model.graph.image.shape[:2]
    grid = np.arange(model.n).reshape(shape)
    if shape[0] < shape[1]:
        grid = grid.T
    rows,width = grid.shape
    if width > 16:
        raise ValueError('Exact transfer is restricted to shorter side <=16.')
    inverse = np.argsort(grid.ravel())
    seeds = np.asarray(seeds)
    if seeds.ndim != 1 or not len(seeds) or not np.issubdtype(seeds.dtype,np.integer):
        raise ValueError('Seeds must be a nonempty integer vector.')
    if np.any((seeds<0)|(seeds>=model.n)):
        raise ValueError('Invalid seed.')
    horizontal = np.zeros((rows,width-1)); vertical = np.zeros((rows-1,width))
    for (p,q),delta in zip(model.graph.edges,model.graph.delta):
        a,c = sorted((inverse[p],inverse[q])); r,j = divmod(a,width)
        if c-a == width:
            vertical[r,j] = delta/4
        else:
            horizontal[r,j] = delta/4
    result = _contract(horizontal,vertical,model.b,inverse[seeds])[:,inverse]
    if not np.isfinite(result).all() or np.max(abs(result))>.2500000001:
        raise FloatingPointError('Unstable transfer contraction at this temperature.')
    return result
