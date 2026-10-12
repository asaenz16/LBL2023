"""Observable-focused diagnostics, independent of Gibbs-state preparation.

Spin normalization and site ordering match algorithm1.py. The reference
state can be any density matrix; it is only a comparison, not a target.
"""
from numbers import Integral

import numpy as np

from algorithm1 import check_n, z_table
from thermal_bath import local_pauli, trace_distance


def spin_correlations(rho, seed=0):
    """Return raw seed <Sz Sz>, connected correlations, and local <Sz>."""
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError("rho must be square with dimension 2**N.")
    n = rho.shape[0].bit_length()-1
    check_n(n)
    if rho.shape[0] != 2**n or not isinstance(seed, Integral) or not 0 <= seed < n:
        raise ValueError("Invalid density dimension or seed.")
    spins = z_table(n)/2
    probabilities = np.diag(rho).real
    means = probabilities @ spins
    raw = probabilities @ (spins[:, seed, None]*spins)
    return raw, raw-means[seed]*means, means


def correlation_metrics(states, reference, seed=0, previous=None, initial=None):
    """States have shape (runs,d,d); exclude the constant self pair from norms.

    signed_margin > 0 means every seed label agrees with the reference.
    Near-zero reference pairs are excluded from this label comparison.
    One must inspect margin size, not only numerical sign agreement.
    """
    states = np.asarray(states)
    if states.ndim != 3:
        raise ValueError("Provide states with shape (runs,d,d).")
    raw, connected, means = (np.asarray(v) for v in zip(
        *(spin_correlations(rho, seed) for rho in states)))
    target, connected_target, _ = spin_correlations(reference, seed)
    offseed = np.arange(len(target)) != seed
    if not np.any(offseed):
        raise ValueError("Correlation comparisons require at least two spins.")
    comparison = offseed & (np.abs(target) > 1e-12)
    if not np.any(comparison):
        margin = np.full(len(states), np.nan)
    else:
        margin = np.min(raw[:, comparison]*np.sign(target[comparison]), axis=1)
    result = {
        "gamma": raw, "connected": connected, "magnetization": means,
        "linf_reference": np.max(np.abs(raw[:, offseed]-target[offseed]), axis=1),
        "connected_linf_reference": np.max(
            np.abs(connected[:, offseed]-connected_target[offseed]), axis=1),
        "trace_distance": np.array([trace_distance(rho, reference) for rho in states]),
        "signed_margin": margin,
    }
    for name, baseline in [("linf_step", previous), ("linf_initial", initial)]:
        result[name] = (np.full(len(states), np.nan) if baseline is None else
                        np.max(np.abs(raw[:, offseed]-np.asarray(baseline)[:, offseed]), axis=1))
    return result


def ensemble_linf_diameter(correlations, seed=0):
    """Exact maximum pairwise l-infinity distance in a finite sample."""
    values = np.asarray(correlations)
    offseed = np.arange(values.shape[-1]) != seed
    return float(np.max(np.ptp(values[:, offseed], axis=0)))


def pauli_noise(rho, n, axis, probability):
    """Independent local Pauli channels (1-p)rho+p P_j rho P_j.

    p=1/2 for Z is complete dephasing, not p=1 (a deterministic Z pulse).
    """
    check_n(n)
    if axis not in "XYZ" or len(axis) != 1 or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Require one Pauli axis and probability in [0,1].")
    state = np.asarray(rho, complex).copy()
    if state.shape != (2**n, 2**n):
        raise ValueError("The state dimension must match n.")
    for site in range(n):
        pauli = local_pauli(n, site, axis)
        state = (1-probability)*state+probability*pauli@state@pauli
    return state


def initial_ensemble(n, samples_per_family=32, rng_seed=1200):
    """Three named states, 32 Haar product states, 32 global Haar pure states.

    The named domain state below is only defined for the 2x2 image in the
    demo; it is an interpretable retention control, not an arbitrary input.
    """
    check_n(n)
    if not isinstance(samples_per_family, Integral) or samples_per_family < 1:
        raise ValueError("samples_per_family must be positive.")
    rng = np.random.default_rng(rng_seed)
    d = 2**n
    states = [np.diag([1.]+[0.]*(d-1)), np.eye(d)/d, np.ones((d,d))/d]
    labels = ["Polarized |0000>", "Maximally mixed", "Coherent |++++>"] if n == 4 else [
        "Polarized", "Maximally mixed", "Coherent plus"]
    families = ["named"]*3
    if n == 4:
        state = np.zeros(d); state[3] = 1  # |0011>, the two image regions
        states.append(np.outer(state,state)); labels.append("Domain product |0011>")
        families.append("named")
    for family in ["random_product", "haar_pure"]:
        for k in range(samples_per_family):
            vector = np.array([1.],complex)
            if family == "random_product":
                for _ in range(n):
                    spin = rng.normal(size=2)+1j*rng.normal(size=2)
                    vector = np.kron(vector,spin/np.linalg.norm(spin))
            else:
                vector = rng.normal(size=d)+1j*rng.normal(size=d)
                vector /= np.linalg.norm(vector)
            states.append(np.outer(vector,vector.conj()))
            labels.append(f"{family}_{k:03d}"); families.append(family)
    return np.asarray(states,complex), labels, families


def permanent_crossing(mask):
    """First true step after which all remaining sampled steps are true.

    This is persistence only through the finite observation window.
    Return None if the condition fails at the last observed step.
    """
    mask = np.asarray(mask,dtype=bool)
    failures = np.flatnonzero(~mask)
    step = 0 if len(failures) == 0 else int(failures[-1]+1)
    return step if step < len(mask) else None
