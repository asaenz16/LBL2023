"""Exact iid Ising grid sampling and one/two-seed segmentation.

H=-sum_e J_e Z_i Z_j/4, rho=exp(-b H)/Z. A frontier of at most
16 spins gives exact sequential conditionals in O(N 2**width), followed by
O(N B) independent samples. Unlike local MCMC, there is no burn-in or mixing.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/quantum-segmentation-numba')
import numpy as np
from numba import njit
from scipy.stats import beta as beta_distribution
from ising_algorithm1 import grid_edges
from ising_thresholds import threshold_for_image
from weizmann_petz_metrics import mask_metrics

FUNCTIONS=['arctan','arctan_quadratic','arctan_cubic','linear','linear_cubic']
DECISIONS=['one_background','two_max','one_foreground']


def parameterize(image,rule):
    image=np.asarray(image,float);edges=grid_edges(*image.shape)
    distances=abs(image.ravel()[edges[:,0]]-image.ravel()[edges[:,1]])
    tau=threshold_for_image(image,rule['threshold'])
    spread=max(float(np.diff(np.quantile(distances,[.1,.9]))[0]),1.)
    alpha={'a001':.001,'a01':.01,'adaptive':1/spread}[rule['sensitivity']]
    x=alpha*(tau-distances);delta=np.arctan(x);name=rule['function']
    if name=='arctan':j=delta
    elif name=='arctan_quadratic':j=delta+delta*abs(delta)
    elif name=='arctan_cubic':j=delta+4*delta**3
    elif name=='linear':j=x
    elif name=='linear_cubic':j=x+x**3
    else:raise ValueError('Unknown coupling function.')
    scale=float(np.median(abs(j)));fallback=scale<1e-12
    if fallback:scale=max(float(abs(j).mean()),.001)
    b=4*rule['K']/scale
    params=dict(threshold=tau,alpha=alpha,spread=spread,scale=scale,scale_fallback=fallback,
        b=b,beta_paper=b*image.size,median_abs_theta=float(np.median(abs(b*j/4))),
        max_abs_theta=float(max(abs(b*j/4))),zero_bond_fraction=float(np.mean(j==0)))
    return edges,j,params


def frontier_geometry(shape,edges,theta):
    grid=np.arange(np.prod(shape)).reshape(shape)
    if shape[0]<shape[1]:grid=grid.T
    rows,width=grid.shape
    if width>16:raise ValueError('Shorter image side must not exceed 16.')
    order=grid.ravel();inverse=np.argsort(order);left=np.zeros(len(order));up=np.zeros(len(order))
    for (i,j),weight in zip(edges,theta):
        p,q=sorted([inverse[i],inverse[j]])
        if q-p==width:up[q]=weight
        elif q-p==1 and q//width==p//width:left[q]=weight
        else:raise ValueError('Non-neighbor coupling.')
    return order,inverse,left,up,width


@njit(cache=True)
def _conditionals(left,up,width,forced=-1):
    n=len(left);dim=1<<width;prob=np.empty((n,dim));future=np.zeros(dim)
    for k in range(n-1,-1,-1):
        c=k%width;bit=1<<c;current=np.empty(dim)
        for state in range(dim):
            field=up[k]*(1-2*((state>>c)&1))
            if c>0:field+=left[k]*(1-2*((state>>(c-1))&1))
            plus=future[state&~bit]+field
            minus=future[state|bit]-field
            if k==forced:
                prob[k,state]=1.;current[state]=plus
            elif plus>=minus:
                w=np.exp(minus-plus);prob[k,state]=1/(1+w);current[state]=plus+np.log1p(w)
            else:
                w=np.exp(plus-minus);prob[k,state]=w/(1+w);current[state]=minus+np.log1p(w)
        current-=current.max();future=current
    return prob


@njit(cache=True)
def _mean_spins(prob,width):
    n,dim=prob.shape;mass=np.zeros(dim);mass[0]=1.;means=np.empty(n)
    for k in range(n):
        bit=1<<(k%width);next_mass=np.zeros(dim);mean=0.
        for state in range(dim):
            plus=mass[state]*prob[k,state];minus=mass[state]-plus
            next_mass[state&~bit]+=plus;next_mass[state|bit]+=minus;mean+=plus-minus
        means[k]=mean;mass=next_mass
    return means


def setup_sampler(shape,edges,j,b):
    if len(shape)!=2 or min(shape)<1 or not np.isfinite(b) or b<0 or not np.isfinite(j).all():
        raise ValueError('Finite couplings, nonnegative inverse temperature, and a 2D grid required.')
    order,inverse,left,up,width=frontier_geometry(shape,edges,b*np.asarray(j)/4)
    return dict(order=order,inverse=inverse,left=left,up=up,width=width,
        probabilities=_conditionals(left,up,width))


def exact_correlations(sampler,seeds):
    # Zero-field global spin-flip symmetry gives C_si=E[Zi | Zs=+1].
    result=[]
    for seed in seeds:
        p=_conditionals(sampler['left'],sampler['up'],sampler['width'],sampler['inverse'][seed])
        result.append(_mean_spins(p,sampler['width'])[sampler['inverse']])
    return np.asarray(result)


def sample_spins(sampler,budget,rng):
    if budget<1:raise ValueError('Positive sample budget required.')
    p=sampler['probabilities'];width=sampler['width'];state=np.zeros(budget,dtype=np.int64)
    spins=np.empty((budget,len(p)),dtype=np.int8)
    for k in range(len(p)):
        minus=rng.random(budget)>=p[k,state];bit=1<<(k%width)
        state=(state&~bit)|(minus.astype(np.int64)*bit)
        spins[:,sampler['order'][k]]=1-2*minus.astype(np.int8)
    return spins


def sufficient_counts(spins,seeds,budgets):
    results=[]
    for budget in budgets:
        s=spins[:budget];bg=s[:,seeds[0]];fg=s[:,seeds[1]];disagree=bg!=fg
        one=(s!=bg[:,None]).sum(axis=0)
        fore=(s==fg[:,None]).sum(axis=0)
        two=(s[disagree]==fg[disagree,None]).sum(axis=0)
        results.append(np.stack([one,two,fore,np.full(s.shape[1],int(disagree.sum()))]))
    return np.asarray(results)


def binomial_interval(successes,trials,tail):
    k=np.asarray(successes);n=np.asarray(trials);safe=np.maximum(n,1)
    lower=np.where(k==0,0.,beta_distribution.ppf(tail,k,safe-k+1))
    upper=np.where(k==n,1.,beta_distribution.ppf(1-tail,k+1,np.maximum(n-k,1)))
    return np.where(n==0,0.,lower),np.where(n==0,1.,upper)


def classify_counts(counts,budget,decision,delta=.05,comparisons=9):
    k=counts[DECISIONS.index(decision)];n=counts[3] if decision=='two_max' else np.full(len(k),budget)
    lo,hi=binomial_interval(k,n,delta/(2*comparisons*len(k)))
    pred=2*k>n;resolved=(lo>.5)|(hi<.5)
    return pred,resolved


def exact_decision(corr,decision):
    if decision=='one_background':score=-corr[0]
    elif decision=='one_foreground':score=corr[1]
    elif decision=='two_max':score=corr[1]-corr[0]
    else:raise ValueError('Unknown seed decision.')
    return score>1e-10,abs(score)>1e-10


def segmentation_metrics(pred,resolved,truth,seeds):
    truth=np.asarray(truth,bool).ravel();pred=np.asarray(pred,bool);resolved=np.asarray(resolved,bool)
    unknown=np.ones(len(truth),bool);unknown[np.asarray(seeds)]=False;correct=pred==truth
    tp=int(np.sum(pred&resolved&truth));fp=int(np.sum(pred&resolved&~truth))
    den=int(truth.sum()+fp+np.sum(~resolved&~truth));upperden=int(truth.sum()+fp)
    return dict(**mask_metrics(pred,truth),unresolved=float(np.mean(~resolved[unknown])),
        correct_resolved=float(np.mean((resolved&correct)[unknown])),wrong_resolved=float(np.mean((resolved&~correct)[unknown])),
        iou_lower=tp/den if den else 1.,iou_upper=(tp+int(np.sum(~resolved&truth)))/upperden if upperden else 1.,
        unknown_iou=mask_metrics(pred[unknown],truth[unknown])['iou'],seed_accuracy=float(correct[seeds].mean()))


def evaluate_counts(counts,budget,truth,seeds):
    return {decision:segmentation_metrics(*classify_counts(counts,budget,decision),truth,seeds) for decision in DECISIONS}
