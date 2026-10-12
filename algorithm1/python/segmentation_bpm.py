"""Skeleton proximity metrics following arXiv:2410.10713v1, Eqs. (3)-(4).

The paper leaves the disk radius and skeletonization algorithm unspecified,
and prints negative FP/FN indicator differences followed by an l1 norm.
We explicitly count nonnegative unmatched skeleton pixels. The resulting
TP-based IoU/F1 are asymmetric for unequal skeleton lengths at r>0; a
symmetric precision/recall F1 is also returned, under a separate name.
This is a documented operational interpretation, not verified author code.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def boundary_proximity(prediction,truth,radius=1):
    pred=np.asarray(prediction,bool);truth=np.asarray(truth,bool)
    if pred.ndim!=2 or pred.shape!=truth.shape:raise ValueError('Require equal 2D mask shapes.')
    if not np.isfinite(radius) or radius<0:raise ValueError('Radius must be finite and nonnegative.')
    p=skeletonize(pred,method='zhang');g=skeletonize(truth,method='zhang')
    npix=int(p.sum());ng=int(g.sum())
    near_g=distance_transform_edt(~g)<=radius if ng else np.zeros(g.shape,bool)
    near_p=distance_transform_edt(~p)<=radius if npix else np.zeros(p.shape,bool)
    tp=int((p&near_g).sum());fp=npix-tp;matched_g=int((g&near_p).sum());fn=ng-matched_g
    both_empty=not npix and not ng
    precision=tp/npix if npix else float(both_empty)
    recall=matched_g/ng if ng else float(both_empty)
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        symmetric_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        precision=precision,recall=recall,tp=tp,fp=fp,fn=fn,
        predicted_skeleton=npix,truth_skeleton=ng,matched_truth=matched_g,radius=float(radius))


def bpm_metrics(prediction,truth,radii=(0,1,2)):
    out={}
    for r in radii:
        v=boundary_proximity(prediction,truth,r)
        for key in ('iou','f1','symmetric_f1'):out[f'bpm_{key}_r{r}']=v[key]
    return out
