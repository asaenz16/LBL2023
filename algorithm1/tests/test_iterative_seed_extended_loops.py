"""The numerical amendment changes only the safety guard."""
import inspect
import numpy as np
import pytest
from xxz_glauber import graph,_loop_sweep
from xxz_sse_fast import expansion_cutoff
from iterative_seed_gibbs import _snapshots
from iterative_seed_extended_loops import _loop_sweep_extended,_snapshots_extended


def test_only_loop_guard_changed():
    expected=inspect.getsource(_loop_sweep.py_func).replace('def _loop_sweep(', 'def _loop_sweep_extended(').replace('loop_steps > 10000000','loop_steps > 100000000')
    assert expected.strip()==inspect.getsource(_loop_sweep_extended.py_func).strip()
    expected=inspect.getsource(_snapshots.py_func).replace('def _snapshots(', 'def _snapshots_extended(').replace('_loop_sweep(', '_loop_sweep_extended(')
    assert expected.strip()==inspect.getsource(_snapshots_extended.py_func).strip()


@pytest.mark.parametrize('g',[.1,.25,.5])
def test_snapshot_samples_bitwise_equal(g):
    edges=np.array([[0,1],[1,2],[3,4],[4,5],[0,3],[1,4],[2,5]],np.int64)
    delta=np.array([.3,-.7,.6,-.2,.4,.9,-.5]);model=graph(6,edges,delta/g)
    b=3.;capacity,_=expansion_cutoff(model,b*g)
    incident=np.full((6,3),-1,np.int64);counts=np.zeros(6,int)
    for e,pair in enumerate(edges):
        for p in pair:incident[p,counts[p]]=e;counts[p]+=1
    args=(model.edges,model.delta,model.shift,model.exits,incident,np.array([1,-1,1,-1,1,-1],np.int8),b*g,capacity,100,256,2,125)
    original=_snapshots(*args);extended=_snapshots_extended(*args)
    for a,b in zip(original,extended):np.testing.assert_array_equal(a,b)
