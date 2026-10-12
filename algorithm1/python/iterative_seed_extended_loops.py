"""Versioned numerical amendment: extend the abort guard, never truncate loops.

Derived verbatim from the frozen sampler functions except names and the loop
step safety limit (10 million to 100 million). No transition probabilities,
random draws, retained samples, burn-in, seeds or Hamiltonians are changed.
"""
import numpy as np
from numba import njit
from xxz_glauber import _initialize
from xxz_sse_fast import _diagonal
from xxz_sse_cluster import _worldline_cluster

@njit(cache=True)
def _loop_sweep_extended(spins, operators, edges, exit_cdf, nloops):
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
            if loop_steps > 100000000:
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
def _snapshots_extended(edges,delta,shift,exits,incident,initial,b,capacity,burn,count,thin,seed):
    np.random.seed(seed);spins=initial.copy();operators=_initialize(spins,edges,delta,shift,b,capacity)
    snapshots=np.empty((count,len(spins)),np.int8);orders=np.empty(count);sectors=np.empty(count)
    maximum=np.count_nonzero(operators);out=0
    for t in range(burn+count*thin):
        order=_diagonal(spins,operators,edges,delta,shift,incident,b)
        maximum=max(maximum,order);_loop_sweep_extended(spins,operators,edges,exits,8)
        _worldline_cluster(spins,operators,edges,delta,shift)
        if t>=burn and (t-burn)%thin==0:
            snapshots[out]=spins;orders[out]=order;sectors[out]=spins.sum();out+=1
    return snapshots,orders,sectors,maximum
