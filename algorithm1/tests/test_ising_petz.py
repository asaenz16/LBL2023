import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from qiskit import qasm2
from qiskit.quantum_info import partial_trace
from ising_algorithm1 import ising_model,gibbs_ising
from ising_petz import (alternating_image,build_ising_petz,local_kraus,local_circuit,
    image_circuit,apply_dense_update,apply_dense_sweep,ladder_gibbs)
from quantum_glauber import build_quantum_heat_bath,apply_update,ensemble_circuit_density


def random_density(n,seed=173):
    rng=np.random.default_rng(seed);a=rng.normal(size=(2**n,2**n))+1j*rng.normal(size=(2**n,2**n))
    rho=a@a.conj().T;return rho/np.trace(rho)


def test_local_ising_petz_equals_global_petz_on_entangled_inputs():
    image=np.array([[0,170,40],[90,255,80]])
    m=build_ising_petz(image,b=13.);exact=ising_model(image,-.001,threshold=100.)
    global_bath=build_quantum_heat_bath(np.diag(exact.energies),m.b)
    rho=random_density(6)
    for i in range(6):
        assert_allclose(apply_dense_update(rho,m,i),apply_update(global_bath.updates[i],rho),atol=2e-14)
        assert_allclose(apply_dense_update(global_bath.sigma,m,i),global_bath.sigma,atol=2e-14)
        ks=local_kraus(m,i);d=len(ks[0])
        assert_allclose(sum(a.T@a for a in ks),np.eye(d),atol=2e-14)


@pytest.mark.parametrize('center',[0,1])
def test_compiled_local_circuit_matches_petz_on_arbitrary_density(center):
    m=build_ising_petz(alternating_image(4));n=len(m.neighbors[center])+1
    rho=random_density(n);c=local_circuit(m,center)
    actual=partial_trace(ensemble_circuit_density(c,np.kron(np.diag([1.,0.]),rho)),[n]).data
    expected=sum(a@rho@a.T for a in local_kraus(m,center))
    assert_allclose(actual,expected,atol=3e-14)


@pytest.mark.parametrize('parallel',[False,True])
def test_full_circuit_channel_order_and_physical_connectivity(parallel):
    m=build_ising_petz([[0,0],[255,255]]);n=m.n
    c=image_circuit(m,parallel=parallel);rho=random_density(n)
    na=c.num_qubits-n;anc=np.zeros((2**na,2**na));anc[0,0]=1.
    actual=ensemble_circuit_density(c,np.kron(anc,rho))
    assert_allclose(partial_trace(actual,list(range(n,n+na))).data,apply_dense_sweep(rho,m),atol=3e-14)
    assert_allclose(partial_trace(actual,list(range(n))).data,anc,atol=3e-14)
    allowed={frozenset((n-1-int(a),n-1-int(b))) for a,b in m.graph.edges}
    allowed.update(frozenset((n-1-i,n+i if parallel else n)) for i in range(n))
    for op in c.data:
        if op.operation.name=='cx':assert frozenset(c.find_bit(q).index for q in op.qubits) in allowed
    parsed=qasm2.loads(qasm2.dumps(c));assert parsed.count_ops()==c.count_ops()
    for a,b in zip(c.data,parsed.data):
        assert a.operation.name==b.operation.name
        assert [c.find_bit(q).index for q in a.qubits]==[parsed.find_bit(q).index for q in b.qubits]
        assert_allclose(a.operation.params,b.operation.params,atol=1e-14)


@pytest.mark.parametrize('b',[0.,13.,32.106383702576444])
def test_transfer_reference_matches_full_enumeration(b):
    image=np.array([[0,40,255,80],[90,210,20,255]])
    m=build_ising_petz(image,b=b);seeds=[0,2,5,7]
    expected=gibbs_ising(ising_model(image,-.001,threshold=100.),b*8).gamma[seeds]
    assert_allclose(ladder_gibbs(m,seeds),expected,atol=1e-14)


def test_plus_x_dephases_after_a_complete_sweep_and_gibbs_is_fixed():
    m=build_ising_petz([[0,0],[255,255]])
    plus=apply_dense_sweep(np.ones((16,16))/16,m)
    mixed=apply_dense_sweep(np.eye(16)/16,m)
    assert_allclose(plus,np.diag(np.diag(plus)),atol=1e-14)
    assert_allclose(plus,mixed,atol=1e-14)
    target=gibbs_ising(ising_model(m.graph.image,-.001,threshold=100.),m.b*m.n)
    assert_allclose(apply_dense_sweep(np.diag(target.weights),m),np.diag(target.weights),atol=1e-14)


def test_large_ladder_resource_bound_without_dense_construction():
    for width in [4,8,16,64]:
        m=build_ising_petz(alternating_image(width));n=m.n;c=image_circuit(m)
        assert len(m.layers)==4 and max(map(len,m.neighbors))==3
        assert_array_equal(sorted(m.order),range(n))
        assert c.count_ops()['cx']==12*n-16
        assert c.count_ops()['ry']==5*n-8
        assert c.count_ops()['reset']==3*n
        assert c.num_qubits==2*n and c.depth()<=80
        for layer in m.layers:
            sites=[v for i in layer for v in (i,*m.neighbors[i])]
            assert len(sites)==len(set(sites))


def test_invalid_inputs():
    for width in [0,3,4.5]:
        with pytest.raises(ValueError):alternating_image(width)
    for b in [-1,np.nan]:
        with pytest.raises(ValueError):build_ising_petz([[0,255]],b=b)
    with pytest.raises(ValueError):build_ising_petz([[0,100]],k=.8)


def test_population_reduction_and_sampling_match_finite_circuit_time():
    from ising_petz_populations import exact_population_sweep,population_trajectories
    from ising_algorithm1 import spin_basis
    m=build_ising_petz([[0,0],[255,255]]);p=np.ones(16)/16;rho=np.ones((16,16))/16
    z=spin_basis(4);exact=[]
    for t in range(6):
        exact.append((p*z[:,0])@z/4)
        if t<5:
            p=exact_population_sweep(p,m);rho=apply_dense_sweep(rho,m)
            assert_allclose(p,np.diag(rho),atol=1e-14)
    r=population_trajectories(m,list(range(6)),shots=32768,seeds=[0],rng_seed=82954)
    error=abs(r['gamma'][:,0]-exact)
    assert error.max()<.008
    assert np.all(error<6*r['se'][:,0]+.001)


def test_algorithm1_readout_bit_order_and_finite_shot_bounds():
    from ising_petz import correlations_from_counts
    result=correlations_from_counts({'0011':100,'1100':100},4,seeds=[0,2])
    assert_allclose(result['gamma'],[[.25,.25,-.25,-.25],[-.25,-.25,.25,.25]])
    assert result['resolved'].all()
    assert_array_equal(result['same_segment'],[[True,True,False,False],[False,False,True,True]])
    for counts in [{'100':2},{'0000':1},{'abcd':10}]:
        with pytest.raises(ValueError):correlations_from_counts(counts,4)
