import numpy as np
from numpy.testing import assert_allclose
import pytest

pytest.importorskip("quimb")
import quimb.tensor as qtn
from ladder_tensor import (build_ladder, dense_row_hamiltonian, domain_product,
                           ground_state, rotate, rung_to_row_indices, seed_profile)


def test_rung_mpo_matches_independent_pixel_hamiltonian():
    ladder=build_ladder(4)
    row=dense_row_hamiltonian(ladder); perm=rung_to_row_indices(4)
    assert_allclose(ladder.mpo.to_dense(),row[np.ix_(perm,perm)],atol=1e-13)
    assert_allclose(ladder.sigma,np.std([0.]*8+[255.]*2,ddof=1))


@pytest.mark.parametrize("seed",[(0,0),(0,1),(1,2),(1,3)])
def test_transfer_profiles_match_dense_complex_state(seed):
    rng=np.random.default_rng(37)
    vector=rng.normal(size=256)+1j*rng.normal(size=256); vector/=np.linalg.norm(vector)
    psi=qtn.MatrixProductState.from_dense(vector,dims=[4]*4,cutoff=0)
    stats=seed_profile(psi,seed)
    indices=rung_to_row_indices(4)
    spins=((1-2*((indices[:,None]>>np.arange(7,-1,-1))&1))/2).reshape(256,2,4)
    probabilities=abs(vector)**2
    means=np.einsum('i,irc->rc',probabilities,spins)
    raw=np.einsum('i,i,irc->rc',probabilities,spins[:,seed[0],seed[1]],spins)
    assert_allclose(stats['gamma'],raw,atol=1e-12)
    assert_allclose(stats['means'],means,atol=1e-12)
    assert_allclose(stats['connected'],raw-means[seed]*means,atol=1e-12)


def test_dmrg_energy_and_correlations_match_exact_small_strip():
    ladder=build_ladder(4); perm=rung_to_row_indices(4)
    e,v=np.linalg.eigh(dense_row_hamiltonian(ladder)[np.ix_(perm,perm)])
    psi,info=ground_state(ladder,bond_dims=(8,16,32),max_sweeps=12)
    assert abs(info['energy']-e[0]) < 1e-8
    assert info['variance'] < 1e-7
    exact=qtn.MatrixProductState.from_dense(v[:,0],dims=[4]*4)
    assert_allclose(seed_profile(psi)['gamma'],seed_profile(exact)['gamma'],atol=2e-5)


def test_tebd_time_step_reduction_matches_exact_evolution():
    from scipy.linalg import expm
    ladder=build_ladder(4); perm=rung_to_row_indices(4)
    h=dense_row_hamiltonian(ladder)[np.ix_(perm,perm)]
    psi=rotate(domain_product(ladder),.3)
    exact=expm(-.2j*h)@np.asarray(psi.to_dense()).ravel()
    errors=[]
    for dt in [.05,.025]:
        evol=qtn.TEBD(psi,ladder.local_ham,dt=dt,split_opts={'max_bond':16,'cutoff':0},progbar=False)
        evol.update_to(.2,order=2,progbar=False)
        v=np.asarray(evol.pt.to_dense()).ravel()
        errors.append(np.linalg.norm(v-exact))
    assert errors[1] < 1e-4
    assert 3.5 < errors[0]/errors[1] < 4.5
