"""Policy grid and reproducible seed promotion, independent of image truth."""
import itertools
import numpy as np
from iterative_seed_gibbs import RoundMoments,unpack,candidates,stop_index,infer_batches
from weizmann_petz_data import seed_for
from weizmann_petz_metrics import mask_metrics
from segmentation_bpm import bpm_metrics

MAX_ROUNDS=10
BASES=[dict(start=s,cut_fg=f,cut_bg=b,gate=g) for s,f,b,g in itertools.product(
    ('foreground','background'),(.25,.5,.75),(.25,.5,.75),('confidence','point_resolved'))]
POLICIES=[dict(**base,stopping=stop,cap=cap) for base,stop,cap in itertools.product(
    BASES,('none','same_count','no_improvement','tolerance_1pct'),(3,6,10))]
METRICS=('iou','unknown_iou','unresolved','mixing_unresolved','iou_lower','iou_upper','mixing_lower',
    'correct_resolved','wrong_resolved','bpm_iou_r0','bpm_iou_r1','bpm_iou_r2',
    'rounds','foreground_seeds','background_seeds','promoted_error','promoted_count','promoted_wrong',
    'converged','minimum_ess','max_rhat','initial_iou','iou_change')
REASONS=('all_resolved','no_candidates','unreliable_sampling','round_cap','stable_count','no_improvement')


def base_of(policy):return {k:policy[k] for k in ('start','cut_fg','cut_bg','gate')}


def initial_sets(base,initial_seed):
    return ([initial_seed],[]) if base['start']=='foreground' else ([],[initial_seed])


class RequiredRounds:
    """Stop generating unused draws once every requested trajectory stops.

    It deliberately has no truth argument. Candidate draws exactly match the
    subsequent replay; stopping variants share a common random prefix.
    """
    def __init__(self,seeds,policies,repeats,random_key,n):
        self.n=n;self.states=[]
        bases={BASES.index(base_of(p)):base_of(p) for p in policies}
        for bi,base in bases.items():
            initial=int(seeds[1 if base['start']=='foreground' else 0])
            relevant=[p for p in policies if base_of(p)==base]
            for rep in range(repeats):
                fg,bg=initial_sets(base,initial)
                self.states.append(dict(base=base,initial=initial,fg=fg,bg=bg,counts=[],terminals=[],policies=relevant,
                    rng=np.random.default_rng(seed_for(random_key+':'+str(bi)+':'+str(rep))),active=True))

    def consume(self,packed,diagnostic):
        mm=RoundMoments(unpack(packed,self.n),scalar_rhat=diagnostic['scalar_rhat']);cache={}
        for state in self.states:
            if not state['active']:continue
            f,b=state['fg'],state['bg'];base=state['base'];key=(tuple(sorted(f)),tuple(sorted(b)))
            if key not in cache:cache[key]=mm.infer(*key,MAX_ROUNDS)
            inf=cache[key];unknown=np.ones(self.n,bool);unknown[state['initial']]=False
            known=abs(inf['score'])>np.maximum(inf['radius'],1e-12)
            count=int(np.sum(~known[unknown])) if inf['converged'] else self.n-1
            pf,pb=candidates(inf,f,b,base['cut_fg'],base['cut_bg'],base['gate'])
            terminal='all_resolved' if not count else ('unreliable_sampling' if not inf['converged'] else ('no_candidates' if len(pf)+len(pb)==0 else ''))
            state['counts'].append(count);state['terminals'].append(terminal);length=len(state['counts'])
            active=False
            for policy in state['policies']:
                k,reason=stop_index(state['counts'],state['terminals'],policy['cap'],policy['stopping'],self.n-1)
                if k==length-1 and reason=='round_cap' and length<policy['cap']:active=True
            state['active']=active
            if active:
                if len(pf):f.append(int(state['rng'].choice(pf)))
                if len(pb):b.append(int(state['rng'].choice(pb)))
        return any(s['active'] for s in self.states)


def trajectory(moments,diagnostics,base,initial_seed,rng_seed,inference_cache=None):
    """No truth argument: only initial label, measurements and random choices."""
    cache={} if inference_cache is None else inference_cache
    fg,bg=initial_sets(base,int(initial_seed));rng=np.random.default_rng(rng_seed);history=[]
    for t,mm in enumerate(moments):
        key=(t,tuple(sorted(fg)),tuple(sorted(bg)))
        if key not in cache:
            inf=mm.infer(key[1],key[2],MAX_ROUNDS);inf.pop('batches');cache[key]=inf
        inf=cache[key];known=abs(inf['score'])>np.maximum(inf['radius'],1e-12)
        unknown=np.ones(mm.n,bool);unknown[initial_seed]=False
        count=int(np.sum(~known[unknown])) if inf['converged'] else int(unknown.sum())
        pf,pb=candidates(inf,fg,bg,base['cut_fg'],base['cut_bg'],base['gate'])
        reason='all_resolved' if count==0 else ('unreliable_sampling' if not inf['converged'] else
            ('no_candidates' if len(pf)+len(pb)==0 else ''))
        row=dict(round=t+1,foreground=fg.copy(),background=bg.copy(),inference=inf,
            unresolved_count=count,terminal=reason,eligible_foreground=len(pf),eligible_background=len(pb),added=[-1,-1])
        history.append(row)
        if reason or t+1>=MAX_ROUNDS:break
        if len(pf):p=int(rng.choice(pf));fg.append(p);row['added'][0]=p
        if len(pb):p=int(rng.choice(pb));bg.append(p);row['added'][1]=p
    return history


def score_metrics(inference,truth,initial_seed,bpm_cache=None):
    t=np.asarray(truth,bool);flat=t.ravel();p=inference['score']>1e-12
    resolved=abs(inference['score'])>np.maximum(inference['radius'],1e-12)
    unknown=np.ones(len(flat),bool);unknown[initial_seed]=False
    correct=p==flat;tp=int(np.sum(p&resolved&flat));fp=int(np.sum(p&resolved&~flat))
    lowerden=int(flat.sum()+fp+np.sum(~resolved&~flat));upperden=int(flat.sum()+fp)
    lower=tp/lowerden if lowerden else 1.;upper=(tp+np.sum(~resolved&flat))/upperden if upperden else 1.
    key=np.packbits(p).tobytes()
    if bpm_cache is None:bpm=bpm_metrics(p.reshape(t.shape),t)
    else:
        if key not in bpm_cache:bpm_cache[key]=bpm_metrics(p.reshape(t.shape),t)
        bpm=bpm_cache[key]
    return dict(**mask_metrics(p,flat),**bpm,unknown_iou=mask_metrics(p[unknown],flat[unknown])['iou'],
        unresolved=float(np.mean(~resolved[unknown])),mixing_unresolved=float(np.mean(~resolved[unknown]) if inference['converged'] else 1.),
        iou_lower=float(lower),iou_upper=float(upper),mixing_lower=float(lower if inference['converged'] else 0.),
        correct_resolved=float(np.mean((correct&resolved)[unknown])),wrong_resolved=float(np.mean((~correct&resolved)[unknown])),
        converged=float(inference['converged']),minimum_ess=inference['minimum_ess'],max_rhat=min(inference['max_rhat'],1e6))


def evaluated_history(history,truth,initial_seed,bpm_cache):
    result=[];flat=np.asarray(truth,bool).ravel()
    for row in history:
        m=score_metrics(row['inference'],truth,initial_seed,bpm_cache)
        foreground=[p for p in row['foreground'] if p!=initial_seed]
        background=[p for p in row['background'] if p!=initial_seed]
        errors=int(np.sum(~flat[foreground]))+int(np.sum(flat[background]));promoted=len(foreground)+len(background)
        m.update(rounds=row['round'],foreground_seeds=len(row['foreground']),background_seeds=len(row['background']),
            promoted_error=errors/promoted if promoted else 0.,promoted_count=promoted,promoted_wrong=errors,
            initial_iou=result[0]['iou'] if result else m['iou'],iou_change=m['iou']-(result[0]['iou'] if result else m['iou']))
        result.append(m)
    return result


def make_moments(packed,n,diagnostics):
    return [RoundMoments(unpack(z,n),scalar_rhat=d['scalar_rhat']) for z,d in zip(packed,diagnostics)]


def evaluate_grid(packed,diagnostics,truth,seeds,repeats,random_key,policies=None,save_traces=False):
    policies=POLICIES if policies is None else policies;n=np.asarray(truth).size
    moments=make_moments(packed,n,diagnostics);cache={};bpm_cache={};histories={}
    table=np.empty((len(policies),repeats,len(METRICS)));reasons=np.empty((len(policies),repeats),np.int8);traces=[]
    for pi,policy in enumerate(policies):
        base=base_of(policy);bi=BASES.index(base);initial=int(seeds[1 if base['start']=='foreground' else 0])
        for rep in range(repeats):
            key=(bi,rep)
            if key not in histories:
                h=trajectory(moments,diagnostics,base,initial,seed_for(random_key+':'+str(bi)+':'+str(rep)),cache)
                histories[key]=(h,evaluated_history(h,truth,initial,bpm_cache))
            history,metrics=histories[key]
            k,reason=stop_index([h['unresolved_count'] for h in history],[h['terminal'] for h in history],policy['cap'],policy['stopping'],n-1)
            table[pi,rep]=[metrics[k][m] for m in METRICS];reasons[pi,rep]=REASONS.index(reason)
            if save_traces:
                trace=[dict(**{key:r[key] for key in ('round','foreground','background','eligible_foreground','eligible_background')},
                    added=r['added'] if q<k else [-1,-1],metrics=metrics[q]) for q,r in enumerate(history[:k+1])]
                controls=controls_for(moments[:k+1],truth,seeds,initial,policy['start'],bpm_cache)
                traces.append(dict(policy_index=pi,repeat=rep,stop=reason,history=trace,controls=controls,
                    prediction=np.packbits(history[k]['inference']['score']>1e-12).tolist(),
                    resolved=np.packbits(abs(history[k]['inference']['score'])>np.maximum(history[k]['inference']['radius'],1e-12)).tolist()))
    return table,reasons,traces


def pooled_inference(moments,foreground,background):
    idx=np.array(tuple(foreground)+tuple(background),int)
    weights=np.r_[np.full(len(foreground),1/len(foreground)) if foreground else [],np.full(len(background),-1/len(background)) if background else []]
    batch=np.concatenate([np.sum(mm.cov[:,:,idx,:]*weights[None,None,:,None],axis=2,dtype=np.float64) for mm in moments])
    scalar=max(d.scalar_rhat if d.scalar_rhat is not None else np.inf for d in moments)
    return infer_batches(batch,idx,weights,None,moments[0].count,scalar,MAX_ROUNDS)


def controls_for(moments,truth,seeds,initial,start,bpm_cache):
    f,b=initial_sets(dict(start=start),initial)
    return {name:score_metrics(inf,truth,initial,bpm_cache) for name,inf in (
        ('initial',moments[0].infer(f,b,MAX_ROUNDS)),
        ('resample_only',moments[-1].infer(f,b,MAX_ROUNDS)),
        ('pooled_one_seed',pooled_inference(moments,f,b)),
        ('oracle_two_seed',pooled_inference(moments,[int(seeds[1])],[int(seeds[0])]))) }
