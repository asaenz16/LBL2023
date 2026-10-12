import numpy as np
from numpy.testing import assert_allclose
import pytest
pytest.importorskip('quimb')
from scipy.linalg import expm
from ladder_tensor import build_ladder,dense_row_hamiltonian,rung_to_row_indices
from ladder_bath import dense_to_density,density_to_dense
from thermal_bath import local_pauli,build_bath,davies_generator
from bath_modifications import (plaquette_hamiltonian,plaquette_generator,plaquette_gate,
    dense_vector,dense_matrix,apply_dense_plan,make_plan,apply_mps_plan,coherent_plan,proposal_weights,
    column_to_row_superoperator,unitary_superoperator)


def test_plaquette_hamiltonian_and_weighted_generator():
    delta=-2.3
    h=sum(-.25*c*local_pauli(4,i,a)@local_pauli(4,j,a)
          for i,j,d in [(0,1,2),(2,3,2),(0,2,delta),(1,3,delta)] for a,c in zip('XYZ',[1,1,d]))
    assert_allclose(plaquette_hamiltonian(delta),h,atol=1e-13)
    assert_allclose(plaquette_generator(delta,6.,1.,1.),davies_generator(build_bath(h,beta=24)),atol=1e-12)
    L=plaquette_generator(delta,6.,2.,1.5)
    target=expm(-6*(h-np.linalg.eigvalsh(h)[0]*np.eye(16)));target/=np.trace(target)
    assert_allclose(L@target.reshape(-1,order='F'),0,atol=1e-12)
    assert_allclose(np.eye(16).reshape(-1)@L,0,atol=1e-12)
    channel=expm(.1*L)
    choi=channel.reshape(16,16,16,16,order='F').transpose(0,2,1,3).reshape(256,256)
    assert np.linalg.eigvalsh((choi+choi.conj().T)/2).min()>-1e-11


def test_proposal_weight_accounting():
    for L in [4,8,16]:
        w=proposal_weights(L);total=np.zeros(L)
        for j in range(L-1):total[j:j+2]+=w[j:j+2]
        assert_allclose(total,[2]+[3]*(L-2)+[2])


def test_channel_layout_matches_column_vectorization():
    rng=np.random.default_rng(903);v=rng.normal(size=16)+1j*rng.normal(size=16);v/=np.linalg.norm(v)
    rho=np.outer(v,v.conj())
    expected=(expm(.07*plaquette_generator(-2.3,6.,2.,1.5))@rho.reshape(-1,order='F')).reshape(16,16,order='F')
    observed=dense_matrix(plaquette_gate(-2.3,6.,.07,2.,1.5)@dense_vector(rho,2),2)
    assert_allclose(observed,expected,atol=1e-12)


@pytest.mark.parametrize('model',['bond','plaquette'])
def test_mps_modified_step_matches_dense_complex_input(model):
    rng=np.random.default_rng(904);v=rng.normal(size=256)+1j*rng.normal(size=256);v/=np.linalg.norm(v)
    initial=.3*np.outer(v,v.conj())+.7*np.eye(256)/256
    ladder=build_ladder(4);plan=make_plan(ladder,model,dt=.03,eta=.2,coherent=True)
    expected=dense_matrix(apply_dense_plan(dense_vector(initial,4),plan,4),4)
    state=dense_to_density(initial,4)
    drift=apply_mps_plan(state,plan,256,cutoff=0)
    assert drift<1e-10
    assert_allclose(density_to_dense(state),expected,atol=1e-11)
    assert np.linalg.eigvalsh(expected).min()>-1e-11


def test_coherent_split_converges_to_full_hamiltonian():
    rng=np.random.default_rng(905);v=rng.normal(size=256)+1j*rng.normal(size=256);v/=np.linalg.norm(v)
    initial=np.outer(v,v.conj());ladder=build_ladder(4);perm=rung_to_row_indices(4)
    h=dense_row_hamiltonian(ladder)[np.ix_(perm,perm)];errors=[]
    for dt in [.04,.02]:
        u=expm(-1j*dt*h);exact=u@initial@u.conj().T
        approx=dense_matrix(apply_dense_plan(dense_vector(initial,4),coherent_plan(ladder,dt),4),4)
        errors.append(np.linalg.norm(approx-exact))
    assert 7.5<errors[0]/errors[1]<8.5


def test_fused_bond_channel_matches_joint_lindblad_exponential():
    from ladder_bath import rung_gate
    h=build_ladder(4).h_rung;identity=np.eye(4)
    dissipator=davies_generator(build_bath(h,beta=12.))
    coherent=-1j*(np.kron(identity,h)-np.kron(h.T,identity))
    expected=column_to_row_superoperator(expm(.07*(coherent+.2*dissipator)),4)
    observed=unitary_superoperator(h,.07)@rung_gate(6.,.2*.07)
    assert_allclose(observed,expected,atol=1e-12)
