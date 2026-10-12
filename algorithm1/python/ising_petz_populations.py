"""Population simulation of the gate-validated Ising Petz circuit.

Diagonal output probabilities depend only on input diagonal probabilities.
After a full sweep the state is diagonal. Thus +x and I/d have identical Z
statistics at every sweep, and the following is an exact reduction of the
circuit, not a different sampler or an assumption of Gibbs equilibration.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/quantum-segmentation-numba')
import numpy as np
from numba import njit


def exact_population_sweep(population,model):
    if model.n>20:raise ValueError('Full populations are limited to N<=20.')
    p=np.asarray(population,float).copy();n=model.n;states=np.arange(2**n)
    if p.shape!=(2**n,):raise ValueError('Wrong population dimension.')
    for i in model.order:
        bit=1<<(n-1-i);lo=states[(states&bit)==0];hi=lo|bit
        index=np.zeros(len(lo),int)
        for j,nb in enumerate(model.neighbors[i]):index|=((lo>>(n-1-nb))&1)<<j
        p0=model.probabilities[i][index];total=p[lo]+p[hi]
        p[lo]=p0*total;p[hi]=(1-p0)*total
    return p


@njit(cache=True)
def _simulate(neighbors,degree,p0,order,seeds,checkpoints,shots,rng_seed,initial):
    np.random.seed(rng_seed);n=len(neighbors)
    states=np.ones((n,shots),np.int8)
    if initial==0:
        for i in range(n):
            for r in range(shots):states[i,r]=1 if np.random.random()<.5 else -1
    gamma=np.zeros((len(checkpoints),len(seeds),n));slot=0
    for sweep in range(checkpoints[-1]+1):
        if sweep==checkpoints[slot]:
            for s in range(len(seeds)):
                for i in range(n):
                    total=0.
                    for r in range(shots):total+=states[seeds[s],r]*states[i,r]/4.
                    gamma[slot,s,i]=total/shots
            slot+=1
            if slot==len(checkpoints):break
        for center in order:
            for r in range(shots):
                config=0
                for j in range(degree[center]):
                    if states[neighbors[center,j],r]<0:config|=1<<j
                states[center,r]=1 if np.random.random()<p0[center,config] else -1
    return gamma


def population_trajectories(model,checkpoints,*,shots=32768,seeds=(0,),rng_seed=1200,initial='plus_x'):
    checkpoints=np.asarray(checkpoints,dtype=int)
    if checkpoints.ndim!=1 or len(checkpoints)==0 or checkpoints[0]!=0 or np.any(np.diff(checkpoints)<=0):
        raise ValueError('Checkpoints must increase strictly from zero.')
    if not isinstance(shots,(int,np.integer)) or shots<2:raise ValueError('Require at least two independent trajectories.')
    if initial not in ('plus_x','mixed','all_up'):raise ValueError('Unsupported initial state.')
    seeds=np.asarray(seeds,dtype=int)
    if seeds.ndim!=1 or not len(seeds) or np.any((seeds<0)|(seeds>=model.n)):raise ValueError('Invalid seeds.')
    degree=np.array(list(map(len,model.neighbors)),dtype=int)
    neighbors=np.full((model.n,4),-1,dtype=int);p0=np.zeros((model.n,16))
    for i in range(model.n):
        neighbors[i,:degree[i]]=model.neighbors[i];p0[i,:2**degree[i]]=model.probabilities[i]
    gamma=_simulate(neighbors,degree,p0,np.array(model.order),seeds,checkpoints,shots,rng_seed,int(initial=='all_up'))
    se=np.sqrt(np.maximum(0,1/16-gamma**2)/(shots-1))
    # Known initial expectation, no reason to report sampling noise at t=0.
    gamma[0]=.25 if initial=='all_up' else 0.
    for j,s in enumerate(seeds):gamma[0,j,s]=.25
    se[0]=0.
    radius=np.sqrt(np.log(2*model.n/.05)/(8*shots))
    return dict(gamma=gamma,se=se,sweeps=checkpoints,shots=shots,seeds=seeds,
                simultaneous_radius=radius,initial=initial,rng_seed=rng_seed)
