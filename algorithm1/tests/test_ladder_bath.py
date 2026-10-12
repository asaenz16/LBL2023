import numpy as np
from numpy.testing import assert_allclose
import pytest
pytest.importorskip("quimb")
from scipy.linalg import expm
from qiskit.quantum_info import DensityMatrix,Kraus

from ladder_tensor import build_ladder,I2,X,Y,Z,dense_row_hamiltonian,rung_to_row_indices
from ladder_bath import (bath_step,bond_kraus,dense_to_density,density_profile,
                         density_reduced,density_to_dense,mixed_density,plus_x_density,density_energy)


def apply_dense_channel(rho,kraus,sites,n=8):
    # sites are left-to-right physical spins in rung order.
    qargs=[n-1-j for j in reversed(sites)]
    return DensityMatrix(rho).evolve(Kraus(list(kraus)),qargs).data


def dense_step(rho,ladder,b,duration,axis,probability):
    L=ladder.width
    for j in range(L): rho=apply_dense_channel(rho,bond_kraus(2.,b,duration/2),(2*j,2*j+1))
    for parity,factor in [(0,.5),(1,1.),(0,.5)]:
        for j in range(parity,L-1,2):
            for row in range(2):
                rho=apply_dense_channel(rho,bond_kraus(float(ladder.horizontal_delta[j]),b,duration*factor),
                                        (2*j+row,2*j+row+2))
    for j in range(L): rho=apply_dense_channel(rho,bond_kraus(2.,b,duration/2),(2*j,2*j+1))
    op={"X":X,"Z":Z}[axis]
    for j in range(2*L): rho=apply_dense_channel(rho,[np.sqrt(1-probability)*I2,np.sqrt(probability)*op],(j,))
    return rho


def test_bond_channel_is_cptp_and_preserves_its_own_gibbs_state():
    for delta in [2.,-2.74]:
        h=-.25*(np.kron(X,X)+np.kron(Y,Y)+delta*np.kron(Z,Z))
        target=expm(-6*(h-np.linalg.eigvalsh(h)[0]*np.eye(4))); target/=np.trace(target)
        kraus=bond_kraus(delta,6.,.1)
        assert_allclose(sum(k.conj().T@k for k in kraus),np.eye(4),atol=1e-12)
        assert_allclose(sum(k@target@k.conj().T for k in kraus),target,atol=1e-12)


def test_vectorized_density_and_measurements_match_independent_dense_calculation():
    rng=np.random.default_rng(55)
    vector=rng.normal(size=256)+1j*rng.normal(size=256); vector/=np.linalg.norm(vector)
    dense=.6*np.outer(vector,vector.conj())+.4*np.eye(256)/256
    rho=dense_to_density(dense,4)
    assert_allclose(density_to_dense(rho),dense,atol=1e-13)
    spins=(1-2*((np.arange(256)[:,None]>>np.arange(7,-1,-1))&1))/2
    for seed in [(0,0),(1,2)]:
        stats=density_profile(rho,seed)
        index=2*seed[1]+seed[0]
        raw=(np.diag(dense).real@(spins*spins[:,index,None])).reshape(4,2).T
        assert_allclose(stats['gamma'],raw,atol=1e-12)
    reduced=DensityMatrix(dense)
    from qiskit.quantum_info import partial_trace
    assert_allclose(density_reduced(rho,1,2),partial_trace(reduced,[0,1,6,7]).data,atol=1e-12)


def test_local_bath_step_matches_exact_density_matrix():
    ladder=build_ladder(4)
    initial=np.eye(256)/256
    expected=dense_step(initial,ladder,6.,.1,"X",.02)
    rho=mixed_density(4)
    drift=bath_step(rho,ladder,b=6.,duration=.1,axis="X",probability=.02,max_bond=256,cutoff=0)
    assert drift < 1e-10
    assert_allclose(density_to_dense(rho),expected,atol=1e-11)
    assert np.linalg.eigvalsh(density_to_dense(rho)).min() > -1e-12
    indices=rung_to_row_indices(4)
    h=dense_row_hamiltonian(ladder)[np.ix_(indices,indices)]
    assert_allclose(density_energy(rho,ladder),np.trace(expected@h).real,atol=1e-11)


def test_plus_x_product_state_and_bath_step_keep_full_initial_coherence():
    rho=plus_x_density(4);initial=np.ones((256,256))/256;ladder=build_ladder(4)
    assert_allclose(density_to_dense(rho),initial,atol=1e-14)
    assert_allclose(density_energy(rho,ladder),-2.5,atol=1e-12)
    x=density_profile(rho,axis='X');z=density_profile(rho)
    assert_allclose(x['means'],.5,atol=1e-14)
    assert_allclose(x['gamma'],.25,atol=1e-14)
    assert_allclose(z['means'],0,atol=1e-14)
    expected=np.zeros((2,4));expected[0,0]=.25
    assert_allclose(z['gamma'],expected,atol=1e-14)
    expected=dense_step(initial,ladder,6.,.1,'Z',0.)
    bath_step(rho,ladder,duration=.1,probability=0.,max_bond=256,cutoff=0.)
    assert_allclose(density_to_dense(rho),expected,atol=1e-11)
