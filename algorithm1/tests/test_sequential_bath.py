import numpy as np
from numpy.testing import assert_allclose
import pytest
pytest.importorskip('quimb')
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from ladder_tensor import build_ladder
from bath_modifications import bath_plan,apply_dense_plan,dense_matrix,dense_vector
from sequential_bath import ExactEdgeDynamics,spin_vector,spin_matrix,embed_pair_generator,local_generators


@pytest.fixture(scope='module')
def dynamics():return ExactEdgeDynamics(build_ladder(4))


def test_local_embedding_layout_and_roundtrip():
    rng=np.random.default_rng(1006);r=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));r=r@r.conj().T;r/=np.trace(r)
    d,_=local_generators(-2.3,6.)
    expected=(d@r.reshape(-1)).reshape(4,4)
    assert_allclose(spin_matrix(embed_pair_generator(d,(0,1),2)@spin_vector(r,2),2),expected,atol=1e-12)
    assert_allclose(spin_matrix(spin_vector(r,2),2),r)


def test_exact_coherent_generator_and_sector(dynamics):
    d=dynamics;r=d.unpack(d.bath_stage(d.identity,.08))
    expected=-1j*(d.h@r-r@d.h)
    assert_allclose(d.unpack(d.coh@d.pack(r)),expected,atol=1e-12)
    assert d.sector_leakage<1e-12
    coherent=d.h_stage(d.pack(r),d.unitary(.7));after=d.unpack(coherent)
    assert_allclose(np.trace(after@d.h),np.trace(r@d.h),atol=1e-12)
    assert_allclose(np.linalg.eigvalsh(after),np.linalg.eigvalsh(r),atol=1e-12)
    assert_allclose(d.h_stage(d.identity,d.unitary(2.)),d.identity,atol=1e-12)
    bad=r.copy();bad[0,1]=.001
    with pytest.raises(ValueError):d.pack(bad)


def test_full_bath_agrees_with_refined_existing_edge_baths(dynamics):
    d=dynamics;ladder=build_ladder(4);exact=d.unpack(d.bath_stage(d.identity,.08));errors=[]
    for substeps in [4,8]:
        v=dense_vector(d.unpack(d.identity),4);plan=bath_plan(ladder,'bond',6.,.08/substeps)
        for _ in range(substeps):v=apply_dense_plan(v,plan,4)
        errors.append(np.linalg.norm(dense_matrix(v,4)-exact))
    assert 3.8<errors[0]/errors[1]<4.2
    assert errors[-1]<2e-4
    assert_allclose(d.bath_stage(d.bath_stage(d.identity,.04),.04),d.pack(exact),atol=1e-12)
    assert_allclose(np.trace(exact),1.,atol=1e-12)
    assert np.linalg.eigvalsh(exact).min()>-1e-12


def test_finite_sequential_cycle_differs_but_small_cycle_limit_agrees(dynamics):
    d=dynamics;initial=d.bath_stage(d.identity,.1);errors=[]
    for dt in [.01,.005]:
        _,seq=d.cycle(initial,dt,dt)
        _,joint=d.cycle(initial,dt,dt,'simultaneous')
        errors.append(np.linalg.norm(seq-joint))
    assert errors[0]>1e-7
    assert 3.5<errors[0]/errors[1]<4.5
    # An independent sparse exponential of the summed generator.
    expected=expm_multiply(.03*(d.coh+2*d.diss),initial)
    _,observed=d.cycle(initial,.03,.06,'simultaneous')
    assert_allclose(observed,expected,atol=1e-12)


def test_unrestricted_evolution_keeps_plus_x_coherences_and_matches_local_channels(dynamics):
    ladder=build_ladder(4);full=ExactEdgeDynamics(ladder,sector='all')
    initial=np.ones((256,256))/256
    with pytest.raises(ValueError):dynamics.pack(initial)
    assert_allclose(full.unpack(full.pack(initial)),initial,atol=1e-14)
    assert_allclose(np.trace(initial@initial),1.,atol=1e-14)
    # Full operator space and the invariant sector agree on a mixed start.
    assert_allclose(full.unpack(full.bath_stage(full.identity,.04)),
                    dynamics.unpack(dynamics.bath_stage(dynamics.identity,.04)),atol=1e-12)
    expected=full.unpack(full.bath_stage(full.pack(initial),.04));errors=[]
    for substeps in [4,8]:
        v=dense_vector(initial,4);plan=bath_plan(ladder,'bond',6.,.04/substeps)
        for _ in range(substeps):v=apply_dense_plan(v,plan,4)
        errors.append(np.linalg.norm(dense_matrix(v,4)-expected))
    assert 3.8<errors[0]/errors[1]<4.2
    assert_allclose(np.trace(expected),1.,atol=1e-12)
    assert np.linalg.eigvalsh(expected).min()>-1e-12
