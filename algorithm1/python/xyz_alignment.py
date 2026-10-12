"""Signed XYZ Gibbs references and iid Pauli-basis sampling for Algorithm 1.

H=-sum_e sum_a J[e,a] sigma[a,i] sigma[a,j]/4. Unlike the old XXZ SSE,
this engine accepts arbitrary signed, unequal XX and YY couplings. Exact
parity-block diagonalization is exponential and deliberately limited to 12
spins. All parity/magnetization sectors are included. Samples are independent
projective outcomes from the exact Gibbs state, not an approximate QMC chain.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from ising_algorithm1 import grid_edges
from ising_thresholds import otsu_edge_threshold


@lru_cache(maxsize=6)
def basis(n):
    if not 1<=n<=12:raise ValueError('Exact XYZ engine supports 1 to 12 spins.')
    states=np.arange(1<<n,dtype=np.int64)
    bits=(states[:,None]>>np.arange(n-1,-1,-1))&1
    return states,1-2*bits,bits.sum(axis=1)


def parameterize(image,rule):
    image=np.asarray(image,float);edges=grid_edges(*image.shape)
    d=abs(image.ravel()[edges[:,0]]-image.ravel()[edges[:,1]])
    spread=max(float(np.diff(np.quantile(d,[.1,.9]))[0]),1.)
    name=rule['threshold']
    if name=='fixed50':center=50.
    elif name=='edge_otsu':center=max(float(otsu_edge_threshold(d,fallback=100.)),1.)
    elif name=='q75':center=max(float(np.quantile(d[d>0],.75)) if np.any(d>0) else 100.,1.)
    elif name in ('quantiles2575','quantiles5090','fixed_triple'):center=None
    else:raise ValueError('Unknown threshold family.')
    if center is not None:thresholds=center+float(rule['width'])*spread*np.array([-1.,0.,1.])
    elif name=='fixed_triple':thresholds=np.array([25.,75.,125.])
    else:thresholds=np.quantile(d,[.25,.5,.75] if name=='quantiles2575' else [.5,.75,.9])
    sensitivity=rule['sensitivity']
    alpha={'fixed001':.001,'fixed01':.01,'inverse_spread':1/spread}[sensitivity]
    coupling=np.arctan(alpha*(thresholds[None,:]-d[:,None]))
    coupling[:,:2]*=float(rule['xy_scale'])
    if rule['model']=='Ising':coupling[:,:2]=0.
    elif rule['model']!='XYZ':raise ValueError('Unknown model.')
    # Report both normalizations; b is set by the experiment rule.
    scales={}
    fallback={}
    for norm,values in [('sum',abs(coupling).sum(axis=1)),('max',abs(coupling).max(axis=1))]:
        value=float(np.median(values));fallback[norm]=value<1e-8
        scales[norm]=max(float(values.mean()),.001) if value<1e-8 else value
    b=4*float(rule['K'])/scales[rule['normalization']]
    params=dict(thresholds=thresholds.tolist(),alpha=alpha,edge_spread=spread,
        scales=scales,normalization_fallback=fallback,b=b,beta_paper=image.size*b,
        distinct_thresholds=len(np.unique(thresholds)),
        negative_fraction=(coupling<0).mean(axis=0).tolist(),
        mixed_axis_sign_fraction=float(np.mean((coupling.min(axis=1)<0)&(coupling.max(axis=1)>0))),
        positive_offdiagonal_fraction=float(np.mean(np.r_[coupling[:,0]+coupling[:,1],coupling[:,0]-coupling[:,1]]<0)))
    return edges,coupling,params


def hamiltonian(n,edges,coupling,sector=None):
    states,z,pop=basis(n);edges=np.asarray(edges,int);coupling=np.asarray(coupling,float)
    if edges.shape!=(len(coupling),2) or coupling.shape[1:]!=(3,):raise ValueError('Invalid XYZ bonds.')
    if not np.isfinite(coupling).all():raise ValueError('Nonfinite coupling.')
    selected=states if sector is None else states[pop%2==sector]
    inverse=np.full(len(states),-1,int);inverse[selected]=np.arange(len(selected))
    h=np.zeros((len(selected),len(selected)));diagonal=np.zeros(len(selected))
    for (i,j),(jx,jy,jz) in zip(edges,coupling):
        if i==j or min(i,j)<0 or max(i,j)>=n:raise ValueError('Invalid edge.')
        zz=z[selected,i]*z[selected,j];diagonal-=jz*zz/4
        target=selected^(1<<(n-1-i))^(1<<(n-1-j))
        h[inverse[target],np.arange(len(selected))]-=(jx-jy*zz)/4
    h[np.arange(len(selected)),np.arange(len(selected))]+=diagonal
    return h,selected


def hadamard_rows(v):
    """Apply H tensor n to the basis index (row axis), without building it."""
    out=np.array(v,copy=True);size=len(out);columns=out.shape[1];step=1
    while step<size:
        chunks=out.reshape(-1,2,step,columns);a=chunks[:,0].copy();b=chunks[:,1].copy()
        chunks[:,0]=a+b;chunks[:,1]=a-b;step*=2
    return out/np.sqrt(size)


def thermal_distributions(n,edges,coupling,betas):
    """Return exact all-spin outcome probabilities in uniform X, Y, Z bases.

    Shape: (number of temperatures, 3 axes, 2**n). Rotated eigenvectors
    are contracted with all Boltzmann weights before proceeding to next axis.
    """
    states,z,pop=basis(n);betas=np.asarray(betas,float)
    if betas.ndim!=1 or np.any(betas<0) or not np.isfinite(betas).all():raise ValueError('Invalid inverse temperatures.')
    if np.all(np.asarray(coupling)[:,:2]==0):
        energies=np.zeros(len(states))
        for (i,j),(_,_,jz) in zip(edges,coupling):energies-=jz*z[:,i]*z[:,j]/4
        weights=np.exp(-betas[:,None]*(energies-energies.min()))
        weights/=weights.sum(axis=1,keepdims=True)
        p=np.full((len(betas),3,len(states)),1/len(states));p[:,2]=weights
        return p,dict(energy=(weights@energies).tolist(),eigen_residual=0.,method='exact diagonal Ising')
    sectors=[];residual=0.
    for sector in (0,1):
        h,chosen=hamiltonian(n,edges,coupling,sector)
        energies,vectors=eigh(h,check_finite=False,driver='evr')
        # Every matrix is small enough to check a deterministic eigenvector subset.
        picks=np.unique(np.linspace(0,len(energies)-1,5,dtype=int))
        residual=max(residual,float(abs(h@vectors[:,picks]-vectors[:,picks]*energies[picks]).max()))
        sectors.append((chosen,energies,vectors))
    e0=min(s[1][0] for s in sectors);partition=np.zeros(len(betas));energy=np.zeros(len(betas))
    p=np.zeros((len(betas),3,len(states)))
    phase=1-2*((pop//2)%2)
    for chosen,energies,vectors in sectors:
        weights=np.exp(-(energies[:,None]-e0)*betas[None,:]);partition+=weights.sum(axis=0)
        energy+=energies@weights
        p[:,2,chosen]=(vectors**2@weights).T
        for axis in (0,1):
            full=np.zeros((len(states),len(chosen)))
            full[chosen]=vectors if axis==0 else vectors*phase[chosen,None]
            # S-dagger tensor n differs from phase by a constant per parity sector.
            transformed=hadamard_rows(full)
            p[:,axis]+=(transformed**2@weights).T
    p/=partition[:,None,None]
    if np.min(p)<-1e-13 or not np.allclose(p.sum(axis=2),1.,atol=2e-12):raise RuntimeError('Invalid Gibbs probability.')
    p=np.maximum(p,0);p/=p.sum(axis=2,keepdims=True)
    return p,dict(energy=(energy/partition).tolist(),eigen_residual=residual,method='exact all-parity XYZ')


def correlations(probabilities,seeds):
    p=np.asarray(probabilities);n=int(np.log2(p.shape[-1]));_,z,_=basis(n)
    products=z[:,np.asarray(seeds)][:,:,None]*z[:,None,:]
    return np.einsum('...d,dsn->...sn',p,products,optimize=True)


def draw_correlations(probabilities,seeds,budget,rng,observable):
    """Equal total independent Gibbs copies for alignment and ZZ.

    Alignment uses budget/3 global measurements in each Pauli basis. One
    outcome estimates correlations to both seeds and every pixel at once.
    ZZ uses all budget copies in the Z basis. No circuit is simulated.
    """
    p=np.asarray(probabilities)
    if budget<1 or (observable=='alignment' and budget%3):raise ValueError('Alignment budget must be divisible by three.')
    axes=(0,1,2) if observable=='alignment' else (2,)
    if observable not in ('alignment','zz'):raise ValueError('Unknown observable.')
    shots=budget//len(axes);total=0.
    for axis in axes:
        counts=rng.multinomial(shots,p[axis]);total=total+correlations(counts/shots,seeds)
    return total/len(axes)


DECISIONS=[dict(kind='background',threshold=t) for t in (-.1,0.,.1)]+[dict(kind='contrast',threshold=0.)]


def decision_score(corr,decision):
    return -corr[0] if decision['kind']=='background' else corr[1]-corr[0]


def uncertainty_radius(n,budget,decision,delta=.05):
    # Simultaneous over both observables, both score types, three budgets,
    # and every pixel. Shifted cuts share their background-score event.
    comparisons=12*n
    radius=np.sqrt(2*np.log(2*comparisons/delta)/budget)
    return radius*(2 if decision['kind']=='contrast' else 1)


def metrics(corr,truth,seeds,decision,radius=0.):
    from weizmann_petz_metrics import mask_metrics
    truth=np.asarray(truth,bool).ravel();score=decision_score(corr,decision)-decision['threshold']
    pred=score>1e-12;resolved=abs(score)>max(radius,1e-12)
    # Known seed labels are not imposed on the output; no spins are pinned.
    unknown=np.ones(len(truth),bool);unknown[np.asarray(seeds)]=False
    correct=pred==truth;known=resolved
    tp=np.sum(pred&known&truth);fp=np.sum(pred&known&~truth)
    denominator=int(truth.sum()+fp+np.sum(~known&~truth))
    lower=float(tp/denominator) if denominator else 1.
    upper_den=int(truth.sum()+fp);upper=float((tp+np.sum(~known&truth))/upper_den) if upper_den else 1.
    # Self/seed entries are known correlations but excluded from resolution rates.
    return dict(**mask_metrics(pred,truth),iou_lower=lower,iou_upper=upper,
        resolved=float(resolved[unknown].mean()),correct_resolved=float((resolved&correct)[unknown].mean()),
        wrong_resolved=float((resolved&~correct)[unknown].mean()),predicted_foreground=float(pred[unknown].mean()),
        unknown_iou=mask_metrics(pred[unknown],truth[unknown])['iou'],
        minimum_margin=float(abs(score[unknown]).min()),seed_accuracy=float(correct[np.asarray(seeds)].mean()))
