import numpy as np
import pytest
from numpy.testing import assert_allclose
from xxz_glauber import graph,exact_gibbs,_initialize,_diagonal_sweep,_loop_sweep,_observe
from xxz_sse_fast import _observe_fast,run_fast,expansion_cutoff
from xxz_image_gibbs import parameterize,sample,evaluate,statistics


def rule(strength=1.,**kwargs):
    return dict(threshold='fixed50',sensitivity='fixed01',K=.5,
        transverse='relative',strength=strength,temperature='combined',**kwargs)


def test_rescaling_matches_pauli_hamiltonian_and_ising_limit():
    image=np.array([[0.,100.],[40.,210.]])
    m,p=parameterize(image,rule())
    xxz=exact_gibbs(graph(4,m.edges,m.delta/p['g']),p['b']*p['g'],[0])
    z=np.array([[1,0],[0,-1]]);x=np.array([[0,1],[1,0]]);y=np.array([[0,-1j],[1j,0]])
    def embed(pauli,site):
        out=np.array([[1.]])
        for i in range(4):out=np.kron(out,pauli if i==site else np.eye(2))
        return out
    h=np.zeros((16,16),complex)
    for (i,j),d in zip(m.edges,m.delta):
        h-=(p['g']*(embed(x,i)@embed(x,j)+embed(y,i)@embed(y,j))+d*embed(z,i)@embed(z,j))/4
    assert_allclose(h,p['g']*xxz['hamiltonian'],atol=1e-14)
    _,zero=parameterize(image,rule(0));assert zero['g']==0
    assert_allclose(np.arctan(p['alpha']*(p['threshold']-p['lower_distance'])),p['g'])


def test_sparse_observer_matches_complete_imaginary_time_average():
    m=graph(4,[[0,1],[0,2],[1,3],[2,3]],[-.5,.2,1.,-.9])
    z=np.array([1,-1,1,-1]);ops=_initialize(z,m.edges,m.delta,m.shift,2.,100)
    for _ in range(30):
        _diagonal_sweep(z,ops,m.edges,m.delta,m.shift,2.)
        _loop_sweep(z,ops,m.edges,m.exits,4)
        a=_observe(z,ops,m.edges,np.array([0,2]));b=_observe_fast(z,ops,m.edges,np.array([0,2]))
        assert_allclose(a[0],b[0],atol=2e-14);assert a[2]==b[1]


@pytest.mark.parametrize('delta',[-2.7,0.,2.])
def test_fast_sampling_exact_two_spin_and_sector_changes(delta):
    m=graph(2,[[0,1]],[delta]);r=run_fast(m,[1,-1],2.,np.arange(2000,66000),[0],715)
    exact=exact_gibbs(m,2.)
    a=np.column_stack([r[0][:,0,1],m.shift.sum()-r[1]/2,(r[2]/2)**2])
    blocks=a.reshape(64,1000,3).mean(axis=1);se=blocks.std(axis=0,ddof=1)/8
    target=[exact['gamma'][0,1],exact['energy'],exact['magnetization_second_moment']]
    assert np.all(abs(blocks.mean(axis=0)-target)<5*se+1e-3)
    assert set(r[2])=={-2,0,2}


@pytest.mark.parametrize('strength',[0.,.5,2.])
def test_arctangent_image_sampler_against_exact_gibbs(strength):
    image=np.array([[0.,100.,200.,20.],[60.,150.,90.,250.]])
    result=sample(image,rule(strength),[0,4],burn=1500,samples=8192,chains=4,rng_seed=917)
    assert np.max(abs(result['gamma']-result['reference'])/(result['se']+1e-3))<5
    assert result['diagnostics']['max_correlation_rhat']<1.05


def test_iou_interval_includes_all_unresolved_completions():
    from itertools import product
    truth=np.array([[0,1],[1,0]],bool);seeds=np.array([0,1])
    gamma=np.array([[.25,-.2,0.,0.],[-.2,.25,0.,0.]])
    q=evaluate(gamma,.1,truth,seeds)
    for a,b in product([False,True],repeat=2):
        pred=np.array([False,True,a,b]);score=np.sum(pred&truth.ravel())/np.sum(pred|truth.ravel())
        assert q['iou_lower']<=score<=q['iou_upper']


def test_expansion_bound_remains_finite_for_large_prefactors():
    from scipy.stats import poisson
    m=graph(2,[[0,1]],[.4])
    for b in [1.,100.,5000.]:
        cutoff,bound=expansion_cutoff(m,b)
        assert np.isfinite(bound) and bound< -50
        mean=b*max(m.shift[0]+m.delta[0]/4,m.shift[0]-m.delta[0]/4+.5)
        exact=b*(mean/b-m.shift.sum())+poisson.logsf(cutoff,mean)
        assert exact<=bound+1e-10


@pytest.mark.parametrize('threshold',['q75','edge_otsu'])
def test_adaptive_rule_contrast_invariance_away_from_floors(threshold):
    image=np.random.default_rng(261).uniform(20,220,(8,8))
    settings=dict(threshold=threshold,sensitivity='inverse_spread',K=.8,
        transverse='relative',strength=.5,temperature='combined')
    first,p=parameterize(image,settings);second,q=parameterize(35+.6*image,settings)
    assert_allclose(first.delta,second.delta,atol=3e-15)
    assert_allclose([p['g'],p['b']],[q['g'],q['b']],atol=3e-14)
