"""Causal three-seed Gibbs readouts; no truth enters inference or promotion.

Scores share [-1,1] normalization. Promoted pixels are evaluated leave-one-out,
including in the pairwise control. Soft conditioning explicitly averages each
configuration with its global spin flip (one draw, not two independent draws).
"""
import itertools
from functools import lru_cache
import numpy as np
from numba import njit
from scipy.stats import t as student_t
from iterative_seed_gibbs import unpack, stop_index
from segmentation_bpm import bpm_metrics

FAMILIES = ('pair', 'majority_gate', 'soft_gate', 'majority_score', 'soft_score')
REASONS = ('all_resolved', 'no_candidates', 'unreliable_sampling', 'round_cap', 'stable_count')
METRICS = ('iou', 'unresolved', 'mixing_unresolved', 'mixing_lower', 'bpm_iou_r1',
           'bpm_iou_r0', 'bpm_iou_r2', 'promoted_count', 'promoted_wrong', 'rounds',
           'shots', 'minimum_ess', 'max_rhat', 'converged', 'higher_rounds',
           'activated', 'weighted_ess', 'gate_rejections', 'score_evaluations')


def configurations():
    out = []
    for cut, margin in itertools.product((.25, .5, .75), (0., .05)):
        base = dict(cut=cut, margin=margin, family='pair', group='anchor', eta=0.,
                    blend=0., gate_margin=0.)
        out.append(base)
        for group in ('anchor', 'uniform'):
            for method, eta in [('majority', 0.)] + [('soft', e) for e in (.25, .5, .75, .9)]:
                for gate in (0., .15):
                    out.append(dict(base, family=method+'_gate', group=group, eta=eta,
                                    gate_margin=gate))
                for blend in (.25, .5, 1.):
                    out.append(dict(base, family=method+'_score', group=group, eta=eta,
                                    blend=blend))
    return out


CONFIGS = configurations()
POLICIES = [dict(**c, cap=cap, stopping=stop) for c, cap, stop in
            itertools.product(CONFIGS, (3, 6, 10), ('none', 'same_count'))]


@lru_cache(None)
def multiplier(n, chains, blocks):
    return float(student_t.ppf(1-.05/(2*n*10), chains*blocks-1))


@njit(cache=True)
def observation_moments(obs, blocks):
    """One contiguous pass; avoids repeated large variance temporaries."""
    c,s,n=obs.shape;size=s//blocks
    first=np.zeros((c,blocks,n),np.float64)
    second=np.zeros((c,blocks,n),np.float64)
    for a in range(c):
        for b in range(blocks):
            for j in range(b*size,(b+1)*size):
                for i in range(n):
                    x=np.float64(obs[a,j,i]);first[a,b,i]+=x;second[a,b,i]+=x*x
    return first/size,second/size


def infer_observations(obs, scalar_rhat=1., blocks=16):
    """Batch/chain SE and split Rhat of paired influence observations.

    For a ratio, obs has its estimated mean plus the ratio's delta-method
    influence. This retains numerator/denominator covariance. Coverage is
    approximate and is not simultaneous across parameter configurations.
    """
    c, s, n = obs.shape
    if s % blocks or blocks % 2 or c < 2:
        raise ValueError('Even blocks dividing samples and >=2 chains required')
    batch,second = observation_moments(obs, blocks)
    cm = batch.mean(1); point = cm.mean(0)
    se = np.maximum(batch.reshape(c*blocks, n).std(0, ddof=1)/np.sqrt(c*blocks),
                    cm.std(0, ddof=1)/np.sqrt(c))
    hm = batch.reshape(2*c,blocks//2,n).mean(1)
    hs = second.reshape(2*c,blocks//2,n).mean(1)
    within = (np.maximum(hs-hm**2,0)*(s//2)/(s//2-1)).mean(0)
    between = (s//2)*hm.var(0, ddof=1)
    rh = np.ones(n); varying = within > 1e-12
    rh[varying] = np.sqrt((s//2-1)/(s//2)+between[varying]/((s//2)*within[varying]))
    rh[(~varying) & (between > 1e-12)] = np.inf
    variance = np.maximum(second.mean((0,1))-point**2,0)*(c*s)/(c*s-1)
    ess = np.full(n, float(c*s))
    np.divide(variance, se**2, out=ess, where=se > 1e-14)
    ess = np.minimum(ess, c*s)
    maximum = max(float(rh.max()), float(scalar_rhat) if scalar_rhat is not None else np.inf)
    return dict(score=point, se=se, radius=multiplier(n, c, blocks)*se,
                converged=bool(maximum < 1.05), max_rhat=maximum,
                minimum_ess=float(ess[variance > 1e-12].min()) if np.any(variance > 1e-12) else float(c*s))


def pair_coefficients(roles):
    roles = np.asarray(roles)
    w = np.zeros(len(roles)); classes = len(set(roles.tolist()))
    for sign in (-1, 1):
        ix = roles == sign
        if ix.any(): w[ix] = sign / ix.sum() / classes
    return w


def select_triple(ids, initial, group, priority, exclude=None):
    ids = [p for p in ids if p != exclude]
    if len(ids) < 3: return ()
    ordered = sorted(ids, key=lambda p: (priority[p], p))
    if group == 'anchor' and initial in ids:
        return tuple(sorted([initial]+[p for p in ordered if p != initial][:2]))
    return tuple(sorted(ordered[:3]))


def triple_functions(z, triple, roles, initial, method, eta):
    """Return odd numerator A and even denominator B for flip symmetry."""
    u = np.stack([roles[p]*z[..., p] for p in triple], axis=-1)
    if method == 'majority':
        return np.sign(u.sum(-1)), np.ones(z.shape[:2], np.float32)
    if method == 'linear':
        return u.mean(-1), np.ones(z.shape[:2], np.float32)
    if method != 'soft': raise ValueError(method)
    strength = np.array([1. if p == initial else eta for p in triple], np.float32)
    v = u*strength
    a = v.sum(-1)+v.prod(-1)
    b = 1+v[..., 0]*v[..., 1]+v[..., 0]*v[..., 2]+v[..., 1]*v[..., 2]
    return a, b


class Evidence:
    def __init__(self, packed, n, scalar_rhat, samples, priority):
        self.z = unpack(packed[:, :samples], n).astype(np.float32)
        self.scalar = scalar_rhat; self.priority = priority
        self.cache = {}; self.calls = 0

    def pair_observations(self, ids, roles, initial):
        z = self.z; w = pair_coefficients([roles[p] for p in ids])
        vote = np.sum(z[..., ids]*w, -1, dtype=np.float32)
        obs = z*vote[..., None]
        # No promoted pixel resolves itself through its own seed label.
        for p in ids:
            if p == initial: continue
            other = [j for j in ids if j != p]
            ww = pair_coefficients([roles[j] for j in other])
            obs[..., p] = z[..., p]*np.sum(z[..., other]*ww, -1)
        obs[..., initial] = roles[initial]
        return obs

    def infer(self, active, initial, config):
        ids = tuple(sorted(active)); family = config['family']
        rolekey = tuple(active[p] for p in ids)
        pairkey = (ids, rolekey, 'pair')
        if pairkey not in self.cache:
            self.cache[pairkey] = infer_observations(self.pair_observations(ids, active, initial), self.scalar)
        base = self.cache[pairkey]
        if family == 'pair' or len(ids) < 3: return base, base, False, float(np.prod(self.z.shape[:2]))
        method = family.split('_')[0]
        key = (ids, rolekey, method, config['group'], config['eta'])
        if key not in self.cache:
            pair = self.pair_observations(ids, active, initial)
            triple = select_triple(ids, initial, config['group'], self.priority)
            a, b = triple_functions(self.z, triple, active, initial, method, config['eta'])
            denominator = float(b.mean(dtype=np.float64))
            numerator = self.z*a[..., None]
            mean = numerator.mean((0, 1), dtype=np.float64)/denominator
            high = numerator/denominator-mean*(b[..., None]/denominator-1)
            weights = [float(b.sum(dtype=np.float64)**2/np.square(b).sum(dtype=np.float64))]
            for p in ids:
                if p == initial: high[..., p] = active[p]; continue
                t = select_triple(ids, initial, config['group'], self.priority, exclude=p)
                if not t: high[..., p] = pair[..., p]; continue
                aa, bb = triple_functions(self.z, t, active, initial, method, config['eta'])
                dd = float(bb.mean(dtype=np.float64)); nn = self.z[..., p]*aa
                mm = float(nn.mean(dtype=np.float64))/dd
                high[..., p] = nn/dd-mm*(bb/dd-1)
                weights.append(float(bb.sum(dtype=np.float64)**2/np.square(bb).sum(dtype=np.float64)))
            high = high.astype(np.float32)
            self.cache[key] = ({v: infer_observations((1-v)*pair+v*high, self.scalar)
                                for v in (.25, .5, 1.)}, min(weights))
            self.calls += 1
        values, weighted_ess = self.cache[key]
        # Kish ESS is a weight concentration diagnostic, NOT an MCMC ESS.
        if weighted_ess < 100:
            values = {k: dict(v, converged=False) for k, v in values.items()}
        score = base if family.endswith('_gate') else values[config['blend']]
        return score, values[1.], True, weighted_ess


def trajectory(evidence, initial, start, config, uniforms):
    active = {int(initial): 1 if start == 'foreground' else -1}
    history = []; high_rounds = 0; rejected = 0; evals = 0
    for t, ev in enumerate(evidence):
        inf, high, used, wess = ev.infer(active, initial, config)
        high_rounds += used; evals += len(inf['score'])*(1+int(used))
        resolved = abs(inf['score']) > inf['radius']+config['margin']+1e-12
        unknown = np.arange(len(resolved)) != initial
        count = int((~resolved[unknown]).sum()) if inf['converged'] else int(unknown.sum())
        available = np.ones(len(resolved), bool); available[list(active)] = False
        pools = []
        for y in (1, -1):
            ok = available & (y*inf['score']-inf['radius'] > config['cut']) & resolved
            if used and config['family'].endswith('_gate'):
                accept = y*high['score']-high['radius'] > config['gate_margin']
                if not high['converged']: accept[:] = False
                rejected += int((ok & ~accept).sum()); ok &= accept
            pools.append(np.flatnonzero(ok))
        reason = ('unreliable_sampling' if not inf['converged'] else
                  'all_resolved' if count == 0 else 'no_candidates' if not any(len(p) for p in pools) else '')
        row = dict(inference=inf, active=active.copy(), round=t+1, unresolved_count=count,
                   terminal=reason, added=[-1, -1], higher_rounds=high_rounds,
                   weighted_ess=wess, gate_rejections=rejected, score_evaluations=evals)
        history.append(row)
        if reason or t == len(evidence)-1: break
        for j, pool in enumerate(pools):
            if len(pool):
                p = int(pool[min(int(uniforms[t, j]*len(pool)), len(pool)-1)])
                active[p] = 1 if j == 0 else -1; row['added'][j] = p
    return history


def metrics(row, truth, initial, margin, samples, cache):
    flat = truth.ravel().astype(bool); inf = row['inference']
    pred = inf['score'] > 1e-12
    resolved = abs(inf['score']) > inf['radius']+margin+1e-12
    unknown = np.arange(len(flat)) != initial
    tp = int((pred & flat).sum()); union = int((pred | flat).sum())
    cert_tp = int((pred & resolved & flat).sum())
    lower_den = int(flat.sum()+(pred & resolved & ~flat).sum()+(~resolved & ~flat).sum())
    key = np.packbits(pred).tobytes()
    if key not in cache: cache[key] = bpm_metrics(pred.reshape(truth.shape), truth)
    promotions = {p:y for p,y in row['active'].items() if p != initial}
    return dict(iou=tp/union if union else 1., unresolved=float((~resolved[unknown]).mean()),
                mixing_unresolved=float((~resolved[unknown]).mean()) if inf['converged'] else 1.,
                mixing_lower=(cert_tp/lower_den if lower_den else 1.) if inf['converged'] else 0.,
                **cache[key], promoted_count=len(promotions),
                promoted_wrong=sum(bool(flat[p]) != (y == 1) for p,y in promotions.items()),
                rounds=row['round'], shots=row['round']*4*samples,
                minimum_ess=inf['minimum_ess'], max_rhat=min(inf['max_rhat'], 1e6),
                converged=float(inf['converged']), higher_rounds=row['higher_rounds'],
                activated=float(row['higher_rounds'] > 0), weighted_ess=row['weighted_ess'],
                gate_rejections=row['gate_rejections'], score_evaluations=row['score_evaluations'])


def evaluate(packed, diagnostics, truth, initial, start, policies, repeats, seed,
             samples=2048, traces=False):
    """Common random promotion draws and truth-independent triplet priorities."""
    rng = np.random.default_rng(seed); n = truth.size
    uniforms = rng.random((repeats, 10, 2)); priorities = rng.random((repeats, 10, n))
    table = np.empty((len(policies), repeats, len(METRICS)))
    reasons = np.empty((len(policies), repeats), np.int8); saved = []; metric_cache = {}
    for rep in range(repeats):
        evidence = [Evidence(p, n, d['scalar_rhat'], samples, priorities[rep,t])
                    for t,(p,d) in enumerate(zip(packed, diagnostics))]
        histories = {}; measured = {}
        for pi, policy in enumerate(policies):
            config = {k:v for k,v in policy.items() if k not in ('cap', 'stopping')}
            key = tuple(sorted(config.items()))
            if key not in histories:
                histories[key] = trajectory(evidence, initial, start, config, uniforms[rep])
            history = histories[key]
            k, reason = stop_index([r['unresolved_count'] for r in history],
                                   [r['terminal'] for r in history], policy['cap'],
                                   policy['stopping'], n-1)
            if (key,k) not in measured:
                measured[key,k] = metrics(history[k], truth, initial, policy['margin'], samples, metric_cache)
            table[pi,rep] = [measured[key,k][m] for m in METRICS]
            reasons[pi,rep] = REASONS.index(reason)
            if traces:
                r = history[k]
                saved.append(dict(policy_index=pi, repeat=rep, stop=reason,
                    history=[dict(round=h['round'], active=h['active'], added=h['added'] if j<k else [-1,-1],
                                  unresolved_count=h['unresolved_count'], higher_rounds=h['higher_rounds'])
                             for j,h in enumerate(history[:k+1])],
                    prediction=np.packbits(r['inference']['score']>1e-12).tolist()))
    return table, reasons, saved
