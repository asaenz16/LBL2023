import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
from higher_seed_readout import CONFIGS
from run_higher_seed_bounded import ORIGINAL_EVALUATE,bounded_evaluate


def test_bounded_execution_preserves_caps_randomness_and_every_saved_trace():
    rng=np.random.default_rng(17);z=rng.choice([-1,1],(10,4,256,16))
    z[...,:12]=z[...,0,None];z[...,1]=-z[...,0]
    packed=np.packbits(z>0,axis=-1,bitorder='little')
    policies=[dict(next(c for c in CONFIGS if c['family']==family),cap=cap,stopping='none')
              for family,cap in [('pair',6),('majority_gate',3),('soft_score',3)]]
    args=(packed,[{'scalar_rhat':1.}]*10,np.eye(4,dtype=bool),0,'foreground',policies,2,981)
    a=ORIGINAL_EVALUATE(*args,samples=256,traces=True)
    b=bounded_evaluate(*args,samples=256,traces=True)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert a[2]==b[2]
