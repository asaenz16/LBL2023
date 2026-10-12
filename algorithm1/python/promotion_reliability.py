"""Causal promotion, spatial diversity and leave-one-out seed maintenance.

Decisions never receive truth. A round's weights are fixed before its samples
are inspected; maintenance decisions only affect the next fresh round.
"""
import itertools
import numpy as np
from scipy.stats import t as student_t
from iterative_seed_gibbs import RoundMoments,infer_batches,candidates,unpack
from iterative_seed_readout import score_metrics,controls_for,METRICS as OLD_METRICS
from weizmann_petz_data import seed_for

SPACES=('none','same_125','same_250','all_125','all_250')
AGREEMENTS=('none','consensus75','consensus100','temporal2','consensus75_temporal2')
MAINTENANCE=('keep','remove1','remove2','soft_relative','soft_absolute')
MODES=('none','same_count','audit3')
CONFIGS=[dict(space=s,agreement=a,maintenance=m,cut=c) for s,a,m,c in itertools.product(SPACES,AGREEMENTS,MAINTENANCE,('base','strict75'))]
POLICIES=[dict(**c,cap=cap,stopping=stop) for c,cap,stop in itertools.product(CONFIGS,(3,6,10),MODES)]
METRICS=OLD_METRICS+('retired_count','retired_wrong','ever_promoted','ever_wrong','wrong_weight','promoted_weight','weight_reductions','agreement_rejections','spatial_rejections')
REASONS=('all_resolved','no_candidates','unreliable_sampling','round_cap','stable_count')


def config_of(p):return {k:p[k] for k in ('space','agreement','maintenance','cut')}


def coefficients(roles,weights,normalization):
    roles=np.asarray(roles);weights=np.asarray(weights,float);out=np.zeros(len(weights))
    for sign in (1,-1):
        ix=roles==sign
        if ix.any():out[ix]=sign*weights[ix]/(ix.sum() if normalization=='absolute' else weights[ix].sum())
    return out


class Evidence:
    def __init__(self,mm):self.mm=mm;self.cache={};self.pairs=None

    def infer(self,ids,roles,weights,normalization):
        key=(tuple(ids),tuple(roles),tuple(weights),normalization)
        if key not in self.cache:
            w=coefficients(roles,weights,normalization);idx=np.asarray(ids,int)
            batch=np.sum(self.mm.cov[:,:,idx,:]*w[None,None,:,None],axis=2,dtype=np.float64)
            inf=infer_batches(batch,idx,w,None,self.mm.count,self.mm.scalar_rhat,10);inf.pop('batches')
            self.cache[key]=(inf,self.leave_out(idx,np.asarray(roles),np.asarray(weights),normalization))
        return self.cache[key]

    def leave_out(self,ids,roles,weights,normalization):
        k=len(ids);a=np.zeros((k,k))
        for i in range(k):
            keep=np.arange(k)!=i
            if keep.any():a[i,keep]=coefficients(roles[keep],weights[keep],normalization)
        cov=self.mm.cov[:,:,ids,:][:,:,:,ids]
        batch=np.einsum('cbji,ij->cbi',cov,a,optimize=True)
        chains,blocks,k=batch.shape;cm=batch.mean(1);mean=cm.mean(0)
        se=np.maximum(batch.reshape(chains*blocks,k).std(0,ddof=1)/np.sqrt(chains*blocks),cm.std(0,ddof=1)/np.sqrt(chains))
        radius=student_t.ppf(1-.05/(2*self.mm.n*10),chains*blocks-1)*se
        hc=cov.reshape(chains,2,blocks//2,k,k).mean(2).reshape(chains*2,k,k)
        half=batch.reshape(chains,2,blocks//2,k).mean(2).reshape(chains*2,k)
        second=np.einsum('ij,hjl,il->hi',a,hc,a,optimize=True);n=self.mm.count//2
        within=np.maximum(second-half**2,0)*n/(n-1);w=within.mean(0);between=n*half.var(0,ddof=1)
        rh=np.ones(k);v=w>1e-14;rh[v]=np.sqrt((n-1)/n+between[v]/(n*w[v]));rh[(~v)&(between>1e-14)]=np.inf
        return dict(score=mean,radius=radius,converged=(rh<1.05)&(self.mm.scalar_rhat is not None and self.mm.scalar_rhat<1.05))

    def votes(self,ids,roles,q):
        if self.pairs is None:
            cov=self.mm.cov;c,b,n,_=cov.shape;cm=cov.mean(1,dtype=np.float64);mean=cm.mean(0)
            se=np.maximum(cov.reshape(c*b,n,n).std(0,ddof=1,dtype=np.float64)/np.sqrt(c*b),cm.std(0,ddof=1)/np.sqrt(c))
            radius=student_t.ppf(1-.05/(2*n*n*10),c*b-1)*se
            half=cov.reshape(c,2,b//2,n,n).mean(2,dtype=np.float64).reshape(2*c,n,n);m=self.mm.count//2
            w=((1-half**2)*m/(m-1)).mean(0);between=m*half.var(0,ddof=1);rh=np.ones((n,n));v=w>1e-14
            rh[v]=np.sqrt((m-1)/m+between[v]/(m*w[v]));rh[(~v)&(between>1e-14)]=np.inf
            self.pairs=(mean,radius,rh<1.05)
        mean,radius,valid=self.pairs;v=mean[ids]*np.asarray(roles)[:,None];r=radius[ids]
        required=int(np.ceil(q*len(ids)-1e-12))
        return ((v-r>0)&valid[ids]).sum(0)>=required,((v+r<0)&valid[ids]).sum(0)>=required


def spatial_mask(shape,ids,roles,sign,space):
    n=int(np.prod(shape));ok=np.ones(n,bool)
    if space=='none':return ok
    scope,scale=space.split('_');radius=min(shape)*(int(scale)/1000)
    refs=[p for p,y in zip(ids,roles) if scope=='all' or y==sign]
    if not refs:return ok
    xy=np.column_stack(np.unravel_index(np.arange(n),shape));ref=xy[refs]
    return np.min(np.sum((xy[:,None]-ref[None,:])**2,axis=2),axis=1)>radius**2


def maintain(ids,roles,weights,initial,loo,kind,strikes):
    new={};retire=[];next_strikes={};decreases=0
    for j,(p,y,w) in enumerate(zip(ids,roles,weights)):
        if p==initial:new[p]=1.;next_strikes[p]=0;continue
        contradictory=bool(loo['converged'][j] and y*loo['score'][j]+loo['radius'][j]<0)
        count=strikes.get(p,0)+1 if contradictory else 0;next_strikes[p]=count
        if kind.startswith('remove') and count>=int(kind[-1]):retire.append(p);continue
        value=w
        if kind.startswith('soft') and loo['converged'][j]:
            target=float(np.clip((y*loo['score'][j]-loo['radius'][j])/.5,.1,1.))
            proposed=round(.5*w+.5*target,4)
            if abs(proposed-w)>=.05:value=proposed
        decreases+=int(value<w);new[p]=value
    return new,retire,next_strikes,decreases


def trajectory(evidence,shape,initial,start,base,config,uniforms):
    """Return a universal prefix supporting ordinary and minimum-three audits."""
    active={int(initial):(1 if start=='foreground' else -1,1.)};retired={};ever={};strikes={};history=[]
    previous=(np.zeros(int(np.prod(shape)),bool),np.zeros(int(np.prod(shape)),bool))
    cumulative=dict(weight_reductions=0,agreement_rejections=0,spatial_rejections=0)
    for t,ev in enumerate(evidence):
        ids=sorted(active);roles=[active[p][0] for p in ids];weights=[active[p][1] for p in ids]
        norm='absolute' if config['maintenance']=='soft_absolute' else 'relative'
        inf,loo=ev.infer(ids,roles,weights,norm);resolved=abs(inf['score'])>np.maximum(inf['radius'],1e-12)
        unknown=np.arange(len(resolved))!=initial;count=int((~resolved[unknown]).sum()) if inf['converged'] else int(unknown.sum())
        fg=[p for p in ids if active[p][0]==1];bg=[p for p in ids if active[p][0]==-1]
        cuts=(.75,.75) if config['cut']=='strict75' else (base['cut_fg'],base['cut_bg'])
        pf,pb=candidates(inf,fg,bg,*cuts,base['gate']);masks=[]
        for sign,pool in ((1,pf),(-1,pb)):
            mask=np.zeros(len(resolved),bool);mask[pool]=True;mask[list(retired)]=False
            space=spatial_mask(shape,ids,roles,sign,config['space']);cumulative['spatial_rejections']+=int((mask&~space).sum());masks.append(mask&space)
        agree=config['agreement'];q=1. if '100' in agree else .75
        if 'consensus' in agree:
            voted=ev.votes(ids,roles,q)
            for j in range(2):cumulative['agreement_rejections']+=int((masks[j]&~voted[j]).sum());masks[j]&=voted[j]
        current=tuple(m.copy() for m in masks)
        if 'temporal2' in agree:
            for j in range(2):cumulative['agreement_rejections']+=int((masks[j]&~previous[j]).sum());masks[j]&=previous[j]
        previous=current
        new_weights,removals,new_strikes,decreases=maintain(ids,roles,weights,initial,loo,config['maintenance'],strikes)
        if not inf['converged']:new_weights=dict(zip(ids,weights));removals=[];decreases=0
        changed=bool(removals or any(new_weights[p]!=active[p][1] for p in new_weights))
        pending=bool((t==0 and 'temporal2' in agree and any(m.any() for m in current)) or
            (config['maintenance']=='remove2' and any(v==1 for p,v in new_strikes.items() if p!=initial)))
        terminal='unreliable_sampling' if not inf['converged'] else ('all_resolved' if count==0 else ('no_candidates' if not any(m.any() for m in masks) and not changed and not pending else ''))
        row=dict(round=t+1,foreground=fg,background=bg,weights=dict(zip(ids,weights)),retired=retired.copy(),ever=ever.copy(),inference=inf,
            unresolved_count=count,terminal=terminal,maintenance_pending=changed or pending,added=[-1,-1],removed=[],next_weights={},counts=cumulative.copy())
        history.append(row)
        if not inf['converged'] or t==len(evidence)-1 or (terminal and t>=2 and not changed and not pending):break
        for p in removals:retired[p]=active[p][0];active.pop(p)
        for p,w in new_weights.items():active[p]=(active[p][0],w)
        strikes=new_strikes;cumulative['weight_reductions']+=decreases;row['removed']=removals;row['next_weights']=new_weights
        # No promotion on a fully resolved round. Audit mode can still measure
        # a fresh round or apply previously decided seed maintenance.
        if count:
            order=(0,1) if uniforms[t,2]<.5 else (1,0)
            chosen=[]
            for j in order:
                pool=np.flatnonzero(masks[j]);pool=np.array([p for p in pool if p not in retired],int)
                if config['space'].startswith('all') and chosen:
                    keep=spatial_mask(shape,chosen,[1]*len(chosen),1,config['space']);pool=pool[keep[pool]]
                if len(pool):
                    p=int(pool[min(int(uniforms[t,j]*len(pool)),len(pool)-1)]);sign=1 if j==0 else -1
                    active[p]=(sign,1.);ever[p]=sign;row['added'][j]=p;chosen.append(p)
    return history


def endpoint(history,policy):
    same=0
    for k,h in enumerate(history):
        reason=h['terminal']
        if policy['stopping']=='audit3' and reason!='unreliable_sampling':
            if k<2 or h['maintenance_pending']:reason=''
        if reason:return k,reason
        if k+1>=policy['cap']:return k,'round_cap'
        if k:
            same=same+1 if h['unresolved_count']==history[k-1]['unresolved_count'] else 0
            if policy['stopping']=='same_count' and same>=2:return k,'stable_count'
    return len(history)-1,'round_cap'


def history_metrics(history,truth,initial,cache):
    flat=np.asarray(truth,bool).ravel();out=[]
    for h in history:
        m=score_metrics(h['inference'],truth,initial,cache)
        promoted={p:1 for p in h['foreground'] if p!=initial}|{p:-1 for p in h['background'] if p!=initial}
        wrong=lambda d:sum(bool(flat[p])!=(y==1) for p,y in d.items())
        errors=wrong(promoted);weights=h['weights']
        m.update(rounds=h['round'],foreground_seeds=len(h['foreground']),background_seeds=len(h['background']),
            promoted_count=len(promoted),promoted_wrong=errors,promoted_error=errors/len(promoted) if promoted else 0.,
            initial_iou=out[0]['iou'] if out else m['iou'],iou_change=m['iou']-(out[0]['iou'] if out else m['iou']),
            retired_count=len(h['retired']),retired_wrong=wrong(h['retired']),ever_promoted=len(h['ever']),ever_wrong=wrong(h['ever']),
            wrong_weight=sum(weights[p] for p,y in promoted.items() if bool(flat[p])!=(y==1)),promoted_weight=sum(weights[p] for p in promoted),**h['counts'])
        out.append(m)
    return out


def evaluate(packed,diagnostics,truth,seeds,base,policies,repeats,random_key,traces=False):
    moments=[RoundMoments(unpack(z,np.size(truth)),scalar_rhat=d['scalar_rhat']) for z,d in zip(packed,diagnostics)]
    evidence=[Evidence(m) for m in moments];initial=int(seeds[1 if base['start']=='foreground' else 0]);cache={};histories={};control_cache={}
    uniforms=[np.random.default_rng(seed_for(random_key+':'+str(r))).random((10,3)) for r in range(repeats)]
    values=np.empty((len(policies),repeats,len(METRICS)));reasons=np.empty((len(policies),repeats),np.int8);saved=[]
    for pi,policy in enumerate(policies):
        config=config_of(policy);key0=tuple(config.items())
        for rep in range(repeats):
            key=(key0,rep)
            if key not in histories:
                h=trajectory(evidence,np.shape(truth),initial,base['start'],base,config,uniforms[rep]);histories[key]=(h,history_metrics(h,truth,initial,cache))
            h,metrics=histories[key];k,reason=endpoint(h,policy)
            values[pi,rep]=[metrics[k][m] for m in METRICS];reasons[pi,rep]=REASONS.index(reason)
            if traces:
                if k not in control_cache:control_cache[k]=controls_for(moments[:k+1],truth,seeds,initial,base['start'],cache)
                rows=[dict(**{a:r[a] for a in ('round','foreground','background','weights','retired','ever')},
                    added=r['added'] if i<k else [-1,-1],removed=r['removed'] if i<k else [],next_weights=r['next_weights'] if i<k else {},metrics=metrics[i]) for i,r in enumerate(h[:k+1])]
                saved.append(dict(policy_index=pi,repeat=rep,stop=reason,history=rows,
                    controls=control_cache[k],
                    prediction=np.packbits(h[k]['inference']['score']>1e-12).tolist(),
                    resolved=np.packbits(abs(h[k]['inference']['score'])>np.maximum(h[k]['inference']['radius'],1e-12)).tolist()))
    return values,reasons,saved
