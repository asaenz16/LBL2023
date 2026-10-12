import numpy as np
from numpy.testing import assert_allclose
from weizmann_petz_data import seed_pixels,resize_mask,decode_annotation
from weizmann_petz_metrics import (classical_baselines,coupling_diagnostics,mask_metrics,segmentation)
from ising_petz import build_ising_petz


def test_seed_baselines_and_reference_labels():
    truth=np.zeros((8,8),bool);truth[2:6,2:6]=True
    seeds=seed_pixels(truth)
    assert len(seeds)==2 and not truth.ravel()[seeds[0]] and truth.ravel()[seeds[1]]
    for inverted in (False,True):
        image=30+200*(~truth if inverted else truth)
        baselines=classical_baselines(image,truth,seeds)
        for p in baselines.values():assert mask_metrics(p,truth)['iou']==1
    assert not segmentation(np.zeros(3),False).any()
    assert not segmentation(np.zeros(3),True).any()


def test_competing_bonds_and_mask_interpolation():
    m=build_ising_petz([[0,110],[40,80]])
    d=coupling_diagnostics(m,np.array([[0,1],[0,1]],bool))
    assert d['frustrated_plaquette_fraction']==1
    mask=np.array([[False,True],[True,False]])
    assert resize_mask(mask,8).dtype==bool
    assert_allclose(resize_mask(mask,8).mean(),.5)
    overlay=np.array([[[255,0,0],[250,10,10],[255,255,255],[20,20,20],[128,100,100]]],np.uint8)
    assert_allclose(decode_annotation(overlay),[[1,1,0,0,0]])


def test_single_class_control_is_separate_and_well_defined():
    for label in (False,True):
        truth=np.full((2,16),label);seeds=seed_pixels(truth)
        assert len(seeds)==1
        for p in classical_baselines(np.full(truth.shape,128.),truth,seeds).values():
            assert mask_metrics(p,truth)['iou']==1
