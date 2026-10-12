"""Image-only global thresholds for sign-compatible Ising segmentation.

Otsu operates on edge differences, never on pixel intensities or evaluation masks.
All rules are computed once, before Gibbs sampling. They preserve edge symmetry.
"""
import numpy as np
from ising_monte_carlo import signed_image

ALPHA = .001
K_REFERENCE = 20*np.arctan(ALPHA*125)/4
K_GRID = [.5, float(K_REFERENCE), .8]
RULES = [dict(kind='fixed', value=t) for t in (50., 75., 100., 125.)]
RULES += [dict(kind='quantile', value=q) for q in (.5, .75, .9)]
RULES += [dict(kind='otsu', positive_only=p) for p in (False, True)]


def rule_name(rule):
    if rule['kind']=='fixed': return f"fixed-{rule['value']:g}"
    if rule['kind']=='quantile': return f"quantile-{rule['value']:g}"
    if rule['kind']=='otsu': return 'otsu-positive' if rule['positive_only'] else 'otsu-all'
    raise ValueError('Unknown threshold rule.')


def otsu_edge_threshold(distances, *, positive_only=False, fallback=125.):
    """Exact Otsu split of the empirical edge distribution, with no bin rounding.

Search splits between distinct observed differences. Equal maxima choose the
lowest split; tau is the midpoint of the adjacent observed values. Thus no edge
has zero coupling. A one-valued distribution cannot be split: use fallback.
For uint8 grayscale this is equivalent to a histogram Otsu split, with an
explicit midpoint convention for empty-bin plateaus.
"""
    d=np.asarray(distances, float)
    if d.ndim!=1 or not np.isfinite(d).all() or np.any(d<0):
        raise ValueError('Distances must be a finite nonnegative vector.')
    if positive_only: d=d[d>0]
    values, counts=np.unique(d, return_counts=True)
    if len(values)<2: return float(fallback)
    p=counts/counts.sum(); w=np.cumsum(p)[:-1]
    mu=np.cumsum(p*values)[:-1]; total=np.sum(p*values)
    scores=(total*w-mu)**2/(w*(1-w))
    i=int(np.argmax(scores))
    return float((values[i]+values[i+1])/2)


def threshold_for_image(image, rule):
    d=signed_image(image, ALPHA).distances
    kind=rule['kind']
    if kind=='fixed': tau=float(rule['value'])
    elif kind=='quantile':
        q=float(rule['value'])
        if not np.isfinite(q) or not 0<q<1: raise ValueError('Quantile must lie in (0,1).')
        positive=d[d>0]
        tau=float(np.quantile(positive,q)) if len(positive) else 125.
        # Retain the actual quantile, including genuine zero couplings at ties.
        tau=max(tau,1.)
    elif kind=='otsu':
        tau=max(otsu_edge_threshold(d,positive_only=rule['positive_only']),1.)
    else: raise ValueError('Unknown threshold rule.')
    if not np.isfinite(tau) or tau<0: raise ValueError('Invalid threshold.')
    return tau


def normalized_model(image, rule, ks, alpha=ALPHA):
    ks=np.asarray(ks,float)
    if ks.ndim!=1 or not len(ks) or not np.isfinite(ks).all() or np.any(ks<=0):
        raise ValueError('Thermal scales must be a nonempty positive vector.')
    tau=threshold_for_image(image,rule)
    model=signed_image(image,alpha,threshold=tau)
    scale=float(np.median(abs(model.delta)))/4 if len(model.delta) else 0.
    if scale<=1e-12:
        raise ValueError('Median coupling is zero; thermal normalization is undefined.')
    return model,ks/scale
