import numpy as np
from numpy.testing import assert_allclose
from xxz_glauber import graph,exact_gibbs
from weizmann_xxz_controls import isolated_bond_shift


def test_isolated_bond_classical_mapping_is_exact_for_zz_only():
    for g in [.1,.5,1.]:
        for b in [.5,3.,20.]:
            for delta in [-1.2,-.2,0.,.2,1.2]:
                ref=exact_gibbs(graph(2,[[0,1]],[delta/g]),b*g,[0])['gamma'][0,1]
                classical=np.tanh(b*(delta-isolated_bond_shift(b,g))/4)/4
                assert_allclose(ref,classical,atol=3e-14)
