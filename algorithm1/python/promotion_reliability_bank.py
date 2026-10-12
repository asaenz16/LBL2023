"""Independent round tails with exactly the original seed spawning schedule.

The function body is derived from the frozen iter_rounds function; only the
start-round range is added. Existing sample prefixes are never regenerated.
"""
from iterative_seed_gibbs import parameterize,graph,expansion_cutoff,setup_sampler,starts,statistics,sample_spins,np
from iterative_seed_extended_loops import _snapshots_extended as _snapshots

def iter_tail(image,rule,*,rounds,samples,burn,thin,chains,rng_seed,start_round=0):
    model,param=parameterize(image,rule);g=param['g'];b=param['b']
    if g:
        quantum=graph(model.n,model.edges,model.delta/g);capacity,tail=expansion_cutoff(quantum,b*g)
        degree=np.bincount(model.edges.ravel(),minlength=model.n)
        incident=np.full((model.n,int(degree.max())),-1,np.int64);counts=np.zeros(model.n,int)
        for e,pair in enumerate(model.edges):
            for p in pair:incident[p,counts[p]]=e;counts[p]+=1
    else:sampler=setup_sampler(image.shape,model.edges,model.delta,b);capacity=0;tail=None
    # Every round is a separately seeded, independently initialized ensemble.
    children=np.random.SeedSequence(rng_seed).spawn(rounds*chains)
    for t in range(start_round,rounds):
        zz=[];scalars=[];maximum=0
        for c,initial in enumerate(starts(image,chains,int(children[t*chains].generate_state(1)[0]))):
            seed=int(children[t*chains+c].generate_state(1)[0]%(2**31-1))
            if g:
                z,orders,sectors,m=_snapshots(quantum.edges,quantum.delta,quantum.shift,quantum.exits,
                    incident,initial,b*g,capacity,burn,samples,thin,seed)
                e=g*(quantum.shift.sum()-orders/(b*g))/model.n;maximum=max(maximum,int(m))
                mag=(sectors/model.n)**2
            else:
                z=sample_spins(sampler,samples,np.random.default_rng(seed))
                e=-(z[:,model.edges[:,0]]*z[:,model.edges[:,1]])@model.delta/4/model.n
                mag=z.mean(axis=1)**2
            zz.append(z);scalars.append(np.column_stack([e,mag]))
        sm,se,ess,rh,_=statistics(np.asarray(scalars),blocks=16)
        diagnostic=dict(scalar_rhat=float(rh.max()) if np.isfinite(rh.max()) else None,
            scalar_ess=ess.tolist(),energy_per_spin=float(sm[0]),magnetization_squared=float(sm[1]),
            capacity=capacity,maximum_order=maximum,log_tail_bound=tail)
        yield np.packbits(np.asarray(zz)>0,axis=-1,bitorder='little'),param,diagnostic
