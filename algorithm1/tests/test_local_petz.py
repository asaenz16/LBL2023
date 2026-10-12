import io
import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import qpy, qasm2
from qiskit.quantum_info import partial_trace
from algorithm1 import image_model
from quantum_glauber import build_quantum_heat_bath, ensemble_circuit_density
from local_petz import (image_graph,build_local_petz,apply_dense_update,apply_dense_sweep,
                        image_circuit,patch_circuit)


def random_density(n):
    rng=np.random.default_rng(149)
    a=rng.normal(size=(2**n,2**n))+1j*rng.normal(size=(2**n,2**n))
    rho=a@a.conj().T
    return rho/np.trace(rho)


@pytest.mark.parametrize('image', [[[0,0],[255,255]],np.arange(18).reshape(2,3,3)])
def test_scalable_graph_matches_existing_image_conventions(image):
    a=image_model(image);b=image_graph(image)
    assert_allclose(a.edges,b.edges);assert_allclose(a.delta,b.delta)
    assert a.sigma==b.sigma


def test_patches_cptp_and_local_gibbs_preservation_but_global_bias():
    m=build_local_petz(image_graph([[0,0],[255,255]]),b=2.)
    for u in m.updates:
        assert_allclose(sum(k.conj().T@k for k in u.kraus),np.eye(2**len(u.sites)),atol=1e-12)
        assert_allclose(sum(k@u.sigma@k.conj().T for k in u.kraus),u.sigma,atol=1e-12)
    target=build_quantum_heat_bath(image_model(m.graph.image).hamiltonian,2.).sigma
    assert np.linalg.norm(apply_dense_sweep(target,m)-target)>1e-4


def test_local_map_preserves_entangled_exterior_marginal():
    m=build_local_petz(image_graph([[0,0],[255,255]]),b=2.)
    rho=random_density(4);u=m.updates[0];result=apply_dense_update(rho,u,4)
    traced=[3-i for i in u.sites]
    assert_allclose(partial_trace(result,traced).data,partial_trace(rho,traced).data,atol=1e-12)
    assert_allclose(np.trace(result),1,atol=1e-12)
    assert np.linalg.eigvalsh(result).min()>-1e-12


def test_parallel_schedule_has_bounded_patches_on_large_square_image():
    rr,cc=np.indices((16,16));image=255*((rr//2+cc//2)%2)
    model=build_local_petz(image_graph(image),b=1.)
    assert len(model.updates)==256 and len(model.layers)==5
    assert sorted(model.order)==list(range(256))
    assert max(len(u.sites) for u in model.updates)==5
    for layer in model.layers:
        sites=[p for i in layer for p in model.updates[i].sites]
        assert len(sites)==len(set(sites))
    assert max(u.hamiltonian.shape[0] for u in model.updates)==32


@pytest.mark.parametrize('parallel',[False,True])
def test_routed_full_circuit_and_exports_match_dense_channel(parallel):
    model=build_local_petz(image_graph([[0,0],[255,255]]),b=2.)
    circuit=image_circuit(model,parallel=parallel,routed=True)
    # Every routed CNOT is either on an image edge or the selected site's bath.
    n=model.graph.n;allowed={frozenset((n-1-a,n-1-b)) for a,b in model.graph.edges}
    if parallel:allowed.update(frozenset((n-1-i,2*n-1-i)) for i in range(n))
    else:allowed.update(frozenset((n-1-i,n)) for i in range(n))
    for ins in circuit.data:
        if ins.operation.name=='cx':
            assert frozenset(circuit.find_bit(q).index for q in ins.qubits) in allowed
    rho=random_density(n);anc=np.zeros((2**(circuit.num_qubits-n),)*2);anc[0,0]=1
    actual=ensemble_circuit_density(circuit,np.kron(anc,rho))
    assert_allclose(partial_trace(actual,list(range(n,circuit.num_qubits))).data,
                    apply_dense_sweep(rho,model),atol=1e-10)
    assert_allclose(partial_trace(actual,list(range(n))).data,anc,atol=1e-10)
    buffer=io.BytesIO();qpy.dump(circuit,buffer);buffer.seek(0)
    assert qpy.load(buffer)[0].count_ops()==circuit.count_ops()
    assert qasm2.loads(qasm2.dumps(circuit)).count_ops()==circuit.count_ops()


def test_tensor_channel_matches_dense_on_eight_entangled_spins():
    pytest.importorskip('quimb')
    from local_petz_tensor import tensor_plan,as_rung_density,apply_tensor_sweep
    from ladder_tensor import rung_to_row_indices,build_ladder
    from ladder_bath import density_to_dense
    import quimb.tensor as qtn
    model=build_local_petz(image_graph(build_ladder(4).image),b=2.)
    rho=random_density(8);rung=rung_to_row_indices(4)
    ordered=rho[np.ix_(rung,rung)]
    permutation=tuple(x for j in range(8) for x in (j,j+8))
    vector=ordered.reshape((2,)*16).transpose(permutation).reshape(-1)
    state=qtn.MatrixProductState.from_dense(vector,dims=[4]*8,cutoff=0)
    drift=apply_tensor_sweep(state,tensor_plan(model),max_bond=256,cutoff=1e-14)
    expected=apply_dense_sweep(rho,model)[np.ix_(rung,rung)]
    assert_allclose(density_to_dense(as_rung_density(state)),expected,atol=1e-9)
    assert drift<1e-9


@pytest.mark.parametrize('kind',['plus_x','mixed','ordered'])
def test_tensor_initial_states_match_row_major_reference(kind):
    pytest.importorskip('quimb')
    from local_petz_tensor import as_rung_density,initial_density_mps
    from ladder_tensor import rung_to_row_indices,build_ladder
    from ladder_bath import density_to_dense
    from run_quantum_glauber import initial_density
    model=build_local_petz(image_graph(build_ladder(4).image));rung=rung_to_row_indices(4)
    expected=initial_density(model.graph.image,kind)[np.ix_(rung,rung)]
    assert_allclose(density_to_dense(as_rung_density(initial_density_mps(model,kind))),expected,atol=1e-13)


def test_invalid_local_parameters_are_rejected():
    graph=image_graph([[0,0],[255,255]])
    with pytest.raises(ValueError):build_local_petz(graph,-1)
    with pytest.raises(ValueError):image_graph(np.zeros((3,3)))
    with pytest.raises(ValueError):build_local_petz(graph,mode='global')
    with pytest.raises(ValueError):apply_dense_sweep(np.eye(16)/16,build_local_petz(graph),order=[0,1])


def test_trajectory_instrument_average_matches_exact_entangled_update():
    pytest.importorskip('quimb')
    import quimb.tensor as qtn
    from local_petz_trajectories import Trajectory,trajectory_plan,apply_trajectory_update
    from ladder_tensor import build_ladder,rung_to_row_indices
    model=build_local_petz(image_graph(build_ladder(4).image),b=2.)
    rng=np.random.default_rng(413);psi=rng.normal(size=256)+1j*rng.normal(size=256);psi/=np.linalg.norm(psi)
    indices=rung_to_row_indices(4);rung=psi[indices]
    plan=trajectory_plan(model);which=model.order.index(1)
    expected=apply_dense_update(np.outer(psi,psi.conj()),model.updates[1],8)[np.ix_(indices,indices)]
    average=np.zeros((256,256),complex)
    for outcome in range(4):
        class ForcedOutcome:
            probability=None
            def choice(self,n,p):self.probability=p[outcome];return outcome
        selection=ForcedOutcome()
        state=qtn.MatrixProductState.from_dense(rung,dims=[2]*8,cutoff=0)
        trajectory=Trajectory(state,selection,{'cur_orthog':'calc'})
        apply_trajectory_update(trajectory,plan[which],max_bond=16,cutoff=0)
        vector=np.asarray(trajectory.state.to_dense()).ravel()
        assert_allclose(np.linalg.norm(vector),1,atol=1e-12)
        assert trajectory.max_probability_sum_error<1e-12
        average+=selection.probability*np.outer(vector,vector.conj())
    assert_allclose(average,expected,atol=1e-12)
