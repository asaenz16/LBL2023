import numpy as np
import pytest
from numpy.testing import assert_allclose
from ising_petz import build_ising_petz,ladder_gibbs,alternating_image
from ising_algorithm1 import ising_model,gibbs_ising
from ising_grid_transfer import exact_grid_gibbs


@pytest.mark.parametrize('shape',[(2,4),(3,3),(4,3),(1,4),(4,1)])
@pytest.mark.parametrize('b',[0.,17.,42.])
def test_transfer_against_enumeration(shape,b):
    image=np.random.default_rng(8320).uniform(0,255,shape)
    m=build_ising_petz(image,b=b)
    seeds=[0,m.n//2,m.n-1]
    ref=gibbs_ising(ising_model(image,-.001,threshold=100),b*m.n).gamma[seeds]
    assert_allclose(exact_grid_gibbs(m,seeds),ref,atol=2e-13)


def test_long_strip_transfer_and_transpose():
    image=alternating_image(64);m=build_ising_petz(image)
    seeds=[0,2,63,91]
    assert_allclose(exact_grid_gibbs(m,seeds),ladder_gibbs(m,seeds),atol=2e-13)
    transposed=build_ising_petz(image.T)
    mapping=np.arange(m.n).reshape(image.shape).T.ravel();inv=np.argsort(mapping)
    assert_allclose(exact_grid_gibbs(transposed,inv[seeds])[:,inv],exact_grid_gibbs(m,seeds),atol=2e-13)


def test_degree_four_elementary_circuit_on_coherent_input():
    from qiskit.quantum_info import partial_trace
    from ising_petz import local_circuit,local_kraus,image_circuit
    from quantum_glauber import ensemble_circuit_density
    m=build_ising_petz(np.array([[0,110,20],[210,50,170],[100,240,70]]))
    rng=np.random.default_rng(824);a=rng.normal(size=(32,32))+1j*rng.normal(size=(32,32))
    rho=a@a.conj().T;rho/=np.trace(rho)
    c=local_circuit(m,4)
    actual=partial_trace(ensemble_circuit_density(c,np.kron(np.diag([1.,0.]),rho)),[5]).data
    expected=sum(k@rho@k.T for k in local_kraus(m,4))
    assert_allclose(actual,expected,atol=3e-14)
    for side in (8,16):
        model=build_ising_petz(rng.uniform(0,255,(side,side)))
        assert len(model.layers)==5
        circuit=image_circuit(model)
        assert circuit.num_qubits==2*side**2
        assert circuit.depth()<=200
