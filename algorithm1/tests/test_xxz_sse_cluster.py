import numpy as np
import pytest
from numpy.testing import assert_allclose
from xxz_glauber import graph,exact_gibbs,_observe
from xxz_sse_cluster import _worldline_cluster,run_cluster
from xxz_image_gibbs import statistics


@pytest.mark.parametrize('d',[-4.,-.2,.1,3.])
def test_fixed_diagonal_string_conditional_distribution(d):
    # One diagonal bond repeated five times: conditional weights are W^5.
    m=graph(2,[[0,1]],[d]);spins=np.array([1,-1]);ops=np.full(5,2,np.int64)
    result=[]
    for i in range(22000):
        _worldline_cluster(spins,ops,m.edges,m.delta,m.shift)
        if i>=2000:result.append(spins[0]*spins[1])
    a=(m.shift[0]+d/4)**5;b=(m.shift[0]-d/4)**5
    assert abs(np.mean(result)-(a-b)/(a+b))<.03


def test_hard_exchange_constraints_are_preserved():
    m=graph(4,[[0,1],[1,2],[2,3]],[.3,-.7,1.])
    # Two exchanges on bond 0, with diagonal vertices between.
    ops=np.array([3,4,6,3,2,4,6]);spins=np.array([1,-1,1,-1])
    for _ in range(100):
        _worldline_cluster(spins,ops,m.edges,m.delta,m.shift)
        r=_observe(spins,ops,m.edges,np.array([0,2]))
        assert np.isfinite(r[0]).all()


def test_mixed_string_conditional_distribution_with_exchange_parities():
    m=graph(4,[[0,1],[1,2],[2,3]],[.3,-.7,1.])
    ops=np.array([3,4,6,3,2,4,6]);basis=1-2*((np.arange(16)[:,None]>>np.arange(3,-1,-1))&1)
    weights=np.zeros(16)
    for k,z in enumerate(basis):
        state=z.copy();w=1.
        for op in ops:
            e=(op-2)//2;i,j=m.edges[e]
            if op&1:
                if state[i]==state[j]:w=0.;break
                w*=.5;state[i]*=-1;state[j]*=-1
            else:w*=m.shift[e]+m.delta[e]*state[i]*state[j]/4
        weights[k]=w
    weights/=weights.sum();expected=(weights*basis[:,0])@basis
    spins=np.array([1,-1,1,-1]);values=[]
    for step in range(54000):
        _worldline_cluster(spins,ops,m.edges,m.delta,m.shift)
        if step>=4000:values.append(spins[0]*spins)
    assert_allclose(np.mean(values,axis=0),expected,atol=.018)


@pytest.mark.parametrize('scale',[.2,1.,3.])
def test_full_cluster_sse_against_eight_spin_exact(scale):
    edges=np.array([[0,1],[1,2],[2,3],[4,5],[5,6],[6,7],[0,4],[1,5],[2,6],[3,7]])
    d=np.array([1.2,-.7,.2,-1.2,.4,1.7,.1,1.4,-.8,.9])/scale
    m=graph(8,edges,d);b=4*scale;seeds=np.array([0,4]);out=[]
    for c in range(4):
        initial=np.random.default_rng(920+c).choice([-1,1],8)
        out.append(run_cluster(m,initial,b,np.arange(2000,18384,2),seeds,1903+c)[0])
    mean,se,_,rh,_=statistics(np.asarray(out));ref=exact_gibbs(m,b,seeds)['gamma']
    assert np.all(abs(mean-ref)<6*se+.002)
    assert np.max(rh)<1.05
