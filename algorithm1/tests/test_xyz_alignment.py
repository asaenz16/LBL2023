import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from xyz_alignment import (hamiltonian,thermal_distributions,correlations,
    parameterize,draw_correlations,uncertainty_radius,metrics,DECISIONS,basis)


def kron_all(items):
    a=np.array([[1.]])
    for b in items:a=np.kron(a,b)
    return a


def test_xyz_matches_pauli_and_all_axis_probabilities():
    x=np.array([[0,1],[1,0]]);y=np.array([[0,-1j],[1j,0]]);z=np.diag([1,-1]);paulis=[x,y,z]
    edges=np.array([[0,1],[1,2],[0,2]]);j=np.array([[.8,-.3,.7],[-.9,.4,-.2],[.1,.6,-.7]])
    h=np.zeros((8,8),complex)
    for edge,couplings in zip(edges,j):
        for a,c in zip(paulis,couplings):h-=c*kron_all([a if q in edge else np.eye(2) for q in range(3)])/4
    assert_allclose(hamiltonian(3,edges,j)[0],h,atol=1e-14)
    p,diag=thermal_distributions(3,edges,j,[0.,.4,2.8])
    had=np.array([[1,1],[1,-1]])/np.sqrt(2);sy=had@np.diag([1,-1j])
    for k,b in enumerate([0.,.4,2.8]):
        rho=expm(-b*h);rho/=np.trace(rho)
        for axis,rotation in enumerate([had,sy,np.eye(2)]):
            u=kron_all([rotation]*3)
            assert_allclose(p[k,axis],np.diag(u@rho@u.conj().T).real,atol=1e-13)
        c=correlations(p[k],[0,2])
        for axis,a in enumerate(paulis):
            for q in range(3):
                o=np.eye(8) if q==0 else kron_all([a if site in [0,q] else np.eye(2) for site in range(3)])
                assert_allclose(c[axis,0,q],np.trace(rho@o).real,atol=1e-13)
    assert diag['eigen_residual']<1e-12


def test_global_axis_permutation_preserves_alignment():
    j=np.random.default_rng(1).normal(size=(4,3));edges=np.array([[0,1],[0,2],[1,3],[2,3]])
    p,_=thermal_distributions(4,edges,j,[1.3]);q,_=thermal_distributions(4,edges,j[:,[2,0,1]],[1.3])
    assert_allclose(correlations(p,[0,2]).sum(axis=1),correlations(q,[0,2]).sum(axis=1),atol=2e-13)


def test_isotropic_ising_infinite_temperature_and_sampling():
    edges=np.array([[0,1],[1,2],[2,3]]);j=np.array([[.3,.3,.3],[-.6,-.6,-.6],[.8,.8,.8]])
    p,_=thermal_distributions(4,edges,j,[0.,1.7])
    c=correlations(p,[0,3]);assert_allclose(c[1,0],c[1,1],atol=2e-14);assert_allclose(c[1,0],c[1,2],atol=2e-14)
    assert_allclose(p[0],1/16,atol=2e-15)
    ising=j.copy();ising[:,:2]=0
    q,_=thermal_distributions(4,edges,ising,[1.7]);assert_allclose(q[0,:2],1/16)
    for observable in ['alignment','zz']:
        samples=np.array([draw_correlations(p[1],[0,3],6144,np.random.default_rng(seed),observable) for seed in range(40)])
        target=c[1].mean(axis=0) if observable=='alignment' else c[1,2]
        assert abs(samples.mean(axis=0)-target).max()<.006
        assert np.mean(np.max(abs(samples-target),axis=(1,2))>uncertainty_radius(4,6144,DECISIONS[1]))<.05


def test_parameter_signs_scales_and_affine_contrast_invariance():
    r=dict(model='XYZ',threshold='edge_otsu',width=.5,sensitivity='inverse_spread',xy_scale=.25,K=.8,normalization='sum')
    image=np.array([[10.,40.,190.],[50.,70.,140.]])
    _,j,p=parameterize(image,r);_,k,q=parameterize(15+.7*image,r)
    assert_allclose(j,k,atol=1e-14);assert_allclose(p['b'],q['b'],atol=1e-13)
    assert p['thresholds'][0]<p['thresholds'][1]<p['thresholds'][2]
    assert p['beta_paper']==image.size*p['b']


def test_metrics_bounds_contain_every_unresolved_label_assignment():
    from itertools import product
    from weizmann_petz_metrics import mask_metrics
    truth=np.array([0,1,0,1],bool);seeds=[0,1];c=np.array([[1,-.2,0,0],[-.2,1,0,0]])
    m=metrics(c,truth,seeds,DECISIONS[1],.1)
    for a,b in product([0,1],repeat=2):
        assert m['iou_lower']<=mask_metrics([0,1,a,b],truth)['iou']<=m['iou_upper']


def test_alignment_is_not_three_times_the_information_at_equal_copies():
    assert uncertainty_radius(9,24576,DECISIONS[-1])==2*uncertainty_radius(9,24576,DECISIONS[1])
    with pytest.raises(ValueError):draw_correlations(np.full((3,4),.25),[0,1],10,np.random.default_rng(1),'alignment')


def test_alignment_singlet_triplet_range_and_high_temperature_slope():
    # A=XX+YY+ZZ has a triplet eigenvalue +1 and singlet eigenvalue -3.
    x=np.array([[0,1],[1,0]]);y=np.array([[0,-1j],[1j,0]]);z=np.diag([1,-1])
    a=sum(np.kron(p,p) for p in (x,y,z))
    assert_allclose(np.linalg.eigvalsh(a),[-3,1,1,1],atol=1e-14)
    edge=np.array([[0,1]]);j=np.array([[.7,-.2,.4]]);b=1e-6
    p,_=thermal_distributions(2,edge,j,[b])
    measured=correlations(p[0],[0])[:,0,1]
    assert_allclose(measured/b,j[0]/4,atol=1e-7)
    # With positive amplitudes, the leading-order effective coupling is
    # monotonically decreasing in color difference despite staggered thresholds.
    d=np.linspace(0,255,300);thresholds=np.array([25.,75.,125.])
    effective=(np.arctan(.01*(thresholds-d[:,None]))*np.array([.25,.25,1])).sum(axis=1)
    assert np.all(np.diff(effective)<0)


def test_frozen_scan_has_unique_rules_and_reuses_temperature_spectra():
    from run_weizmann_xyz import candidates,identifier,coupling_identifier,tied
    rules=candidates()
    assert len(rules)==len({identifier(r) for r in rules})==459
    assert len({coupling_identifier(r) for r in rules})==81
    assert sum(r['model']=='XYZ' and tied(r) for r in rules)==108


def test_analytic_bond_alignment_matches_full_gibbs():
    from xyz_bond_diagnostic import bond_alignment
    edges=np.array([[0,1]])
    for j in [np.array([[.7,-.2,.4]]),np.array([[-.9,.3,-.6]])]:
        for b in (0.,.1,2.,100.):
            p,_=thermal_distributions(2,edges,j,[b])
            exact=correlations(p[0],[0])[:,0,1].sum()
            assert_allclose(bond_alignment(j[0],b),exact,atol=3e-14)
