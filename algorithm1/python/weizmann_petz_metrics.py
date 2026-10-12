"""Evaluation and classical baselines for the seeded Ising benchmark."""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from ising_algorithm1 import grid_edges
from ising_thresholds import otsu_edge_threshold


def segmentation(gamma,seed_foreground):
    return gamma>1e-12 if seed_foreground else gamma < -1e-12


def mask_metrics(prediction,truth):
    p=np.asarray(prediction,bool).ravel();t=np.asarray(truth,bool).ravel()
    tp=np.sum(p&t);union=np.sum(p|t);total=p.sum()+t.sum()
    return dict(iou=float(tp/union) if union else 1.,dice=float(2*tp/total) if total else 1.,
                accuracy=float(np.mean(p==t)))


def classical_baselines(image,truth,seeds):
    labels=truth.ravel()[seeds]
    if len(seeds)==1:
        pred=np.full(truth.shape,labels[0]);return dict(otsu=pred,random_walker=pred)
    tau=otsu_edge_threshold(image.ravel(),fallback=float(np.median(image)))
    thresholded=image>tau;fg=seeds[int(np.flatnonzero(labels)[0])]
    otsu=thresholded if thresholded.ravel()[fg] else ~thresholded
    edges=grid_edges(*image.shape);n=image.size;pixels=image.ravel()
    distances=abs(pixels[edges[:,0]]-pixels[edges[:,1]])
    # Fixed image-only scale: 10 percent of the robust intensity range.
    # A median of nonzero differences is unsuitable for two-level images:
    # there it equals the object boundary contrast and suppresses it too weakly.
    scale=max(float(np.diff(np.quantile(pixels,[.05,.95]))[0])*.1,1.)
    conductance=np.exp(-(distances/scale)**2)+1e-6
    rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
    adjacency=coo_matrix((np.tile(conductance,2),(rows,cols)),shape=(n,n)).tocsr()
    laplacian=diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
    unknown=np.setdiff1d(np.arange(n),seeds);probability=np.zeros(n)
    probability[seeds]=labels
    if len(unknown):
        probability[unknown]=spsolve(laplacian[unknown][:,unknown],
                                    -laplacian[unknown][:,seeds]@labels.astype(float))
    return dict(otsu=otsu,random_walker=(probability>.5).reshape(image.shape))


def coupling_diagnostics(model,truth):
    h,w=truth.shape;signs={tuple(edge):np.sign(d) for edge,d in zip(model.graph.edges,model.graph.delta)}
    frustrated=[]
    for r in range(h-1):
        for c in range(w-1):
            i=r*w+c
            product=signs[(i,i+1)]*signs[(i,i+w)]*signs[(i+1,i+w+1)]*signs[(i+w,i+w+1)]
            frustrated.append(product<0)
    crossing=truth.ravel()[model.graph.edges[:,0]]!=truth.ravel()[model.graph.edges[:,1]]
    negative=model.graph.delta<0
    return dict(negative_bond_fraction=float(negative.mean()),zero_bond_fraction=float(np.mean(model.graph.delta==0)),
        frustrated_plaquette_fraction=float(np.mean(frustrated)) if frustrated else 0.,
        boundary_antiferromagnetic_fraction=float(negative[crossing].mean()) if crossing.any() else None,
        interior_antiferromagnetic_fraction=float(negative[~crossing].mean()) if (~crossing).any() else None,
        median_neighbor_difference=float(np.median(model.graph.distances)),
        all_ferromagnetic=bool(np.all(model.graph.delta>0)))


def correlation_metrics(gamma,reference,truth,seeds,radius=0.):
    labels=truth.ravel()[seeds];pred=[segmentation(g,lab) for g,lab in zip(gamma,labels)]
    result=mask_metrics(pred[0],truth)
    threshold=max(radius,1e-12);resolved=abs(gamma)>threshold
    correct=np.stack([p==truth.ravel() for p in pred])
    result.update(gamma_linf=float(np.max(abs(gamma-reference))),
        resolved_fraction=float(resolved.mean()),correct_resolved_fraction=float(np.mean(resolved&correct)),
        resolved_error_fraction=float(np.mean(resolved&~correct)),
        primary_resolved_fraction=float(resolved[0].mean()),
        secondary_iou=mask_metrics(pred[-1],truth)['iou'],
        seed_agreement=float(np.mean(pred[0]==pred[-1])),
        min_abs_gamma=float(np.min(abs(gamma[0]))))
    return result
