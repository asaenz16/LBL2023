import numpy as np
import torch
import pytest
from correlation_nn_data import moments,init
from correlation_nn import Decoder,permutations,transform,train,probabilities
from xxz_image_gibbs import statistics


def test_all_pair_moments_match_direct_observations():
    z=np.random.default_rng(516).choice([-1,1],(4,512,7)).astype(np.int8)
    packed=np.packbits(z>0,axis=-1,bitorder='little')
    c,se,rh,ess=moments(packed,7,(512,))
    direct=z[:,:,:,None]*z[:,:,None,:]
    expected=statistics(direct,blocks=16)
    for a,b in zip((c[0],se,ess,rh),expected[:4]):np.testing.assert_allclose(a,b,atol=2e-3,rtol=2e-6)


def test_dihedral_action_transforms_both_axes():
    z=torch.tensor(np.random.default_rng(45).choice([-1,1],(2,16)),dtype=torch.float32)
    c=z[:,:,None]*z[:,None,:];seed=torch.zeros_like(z);seed[:,6]=1
    for p in permutations(4):
        cp,im,sp,yp=transform(c,z,seed,z,p)
        torch.testing.assert_close(cp,im[:,:,None]*im[:,None,:])
        torch.testing.assert_close(cp@sp[:,:,None],(c@seed[:,:,None])[:,p])


@pytest.mark.parametrize('architecture',['volume','graph'])
def test_full_decoder_gradient_reaches_distant_correlations(architecture):
    torch.manual_seed(127);model=Decoder(4,12,architecture,'full',True)
    c=torch.randn(2,16,16,requires_grad=True);im=torch.zeros(2,16);seed=torch.zeros(2,16);seed[:,3]=1
    output=model(c,im,seed);assert output.shape==(2,16)
    output.sum().backward();assert torch.isfinite(c.grad).all() and c.grad[0,0,15].abs()>0


@pytest.mark.parametrize('architecture',['volume','graph'])
def test_prior_and_image_controls_do_not_read_correlations(architecture):
    for kind in ('prior','image'):
        model=Decoder(4,8,architecture,kind,True)
        c=torch.randn(2,16,16);im=torch.rand(2,16);seed=torch.zeros(2,16);seed[:,4]=1
        torch.testing.assert_close(model(c,im,seed),model(torch.zeros_like(c),im,seed))


@pytest.mark.parametrize('architecture',['volume','graph'])
def test_prior_has_same_positional_capacity_as_full_matrix_decoder(architecture):
    torch.manual_seed(314)
    full=Decoder(4,8,architecture,'full',True);prior=Decoder(4,8,architecture,'prior',True)
    prior.load_state_dict(full.state_dict())
    seed=torch.zeros(2,16);seed[:,5]=1;im=torch.rand(2,16)
    eye=torch.eye(16)[None].expand(2,-1,-1)
    torch.testing.assert_close(full(eye,im,seed),prior(torch.randn(2,16,16),im,seed))


def test_full_decoder_does_not_read_intensities():
    model=Decoder(4,8,'volume','full',False);c=torch.eye(16)[None];seed=torch.zeros(1,16)
    torch.testing.assert_close(model(c,torch.zeros(1,16),seed),model(c,torch.rand(1,16),seed))


@pytest.mark.parametrize('architecture',['volume','graph'])
def test_column_control_reads_only_the_supplied_seed_column(architecture):
    torch.manual_seed(891)
    model=Decoder(4,8,architecture,'column',True)
    c=torch.randn(2,16,16,requires_grad=True);im=torch.rand(2,16)
    seed=torch.zeros(2,16);indices=torch.tensor([3,7]);seed[torch.arange(2),indices]=1
    changed=torch.randn_like(c)
    changed[torch.arange(2),:,indices]=c.detach()[torch.arange(2),:,indices]
    output=model(c,im,seed)
    torch.testing.assert_close(output,model(changed,im,seed))
    output.sum().backward()
    allowed=torch.zeros_like(c,dtype=torch.bool);allowed[torch.arange(2),:,indices]=True
    assert torch.count_nonzero(c.grad[~allowed])==0
    assert c.grad[allowed].abs().sum()>0


def test_seedless_decoder_ignores_supplied_seed_locations():
    model=Decoder(4,8,'volume','full',False);c=torch.eye(16)[None];im=torch.zeros(1,16)
    a=torch.zeros(1,16);b=a.clone();a[:,1]=1;b[:,9]=1
    torch.testing.assert_close(model(c,im,a),model(c,im,b))


def test_sampling_quality_eligibility_does_not_use_outer_test_cases(tmp_path,monkeypatch):
    import json
    import run_correlation_nn as runner
    monkeypatch.setattr(runner,'OUT',tmp_path)
    ids=[f'i{k}' for k in range(100)]
    p=dict(cases=[dict(source_id=s) for s in ids],rules={'r':{}},folds=[dict(train=ids[:64],validation=ids[64:80],test=ids[80:])])
    d=[dict(converged=i<80) for i in range(100)]
    (tmp_path/'diagnostics-r.json').write_text(json.dumps(d))
    assert runner.reliable_rules(p,0)==['r']
    for i in range(5):d[i]['converged']=False
    for i in range(80,100):d[i]['converged']=True
    (tmp_path/'diagnostics-r.json').write_text(json.dumps(d))
    assert runner.reliable_rules(p,0)==[]


@pytest.mark.parametrize('g',[0.,.1,.5])
def test_complete_loop_all_pair_samples_agree_with_dense_gibbs(g):
    from promotion_reliability_bank import iter_tail
    from iterative_seed_gibbs import parameterize
    from xxz_glauber import graph,exact_gibbs
    image=np.array([[10.,10.,160.],[20.,25.,180.]])
    rule=dict(threshold='q90',sensitivity='inverse_spread',K=.6,transverse='relative',strength=g,temperature='longitudinal')
    packed,params,_=next(iter_tail(image,rule,rounds=1,samples=4096,burn=1200,thin=2,chains=4,rng_seed=721))
    model,_=parameterize(image,rule)
    if g:
        q=graph(6,model.edges,model.delta/params['g']);reference=4*exact_gibbs(q,params['b']*params['g'],np.arange(6))['gamma']
    else:
        z=1-2*((np.arange(64)[:,None]>>np.arange(6))&1)
        e=-np.sum(z[:,model.edges[:,0]]*z[:,model.edges[:,1]]*model.delta/4,axis=1)
        w=np.exp(-params['b']*(e-e.min()));w/=w.sum();reference=np.einsum('s,si,sj->ij',w,z,z)
    c,*_=moments(packed,6,(4096,))
    assert np.max(abs(c[0]-reference))<.06


def test_folds_are_source_disjoint_and_cover_once():
    p=init();covered=[]
    for f in p['folds']:
        a,b,c=map(set,(f['train'],f['validation'],f['test']))
        assert (len(a),len(b),len(c))==(64,16,20)
        assert not (a&b or a&c or b&c)
        covered+=f['test']
    assert len(covered)==len(set(covered))==100


def test_training_can_fit_a_small_correlation_segmentation_problem():
    y=np.array([[0,0,1,1]*4,[0]*8+[1]*8],np.float32)
    z=2*y-1;c=z[:,:,None]*z[:,None,:];s=np.zeros_like(y);s[:,14]=1
    model,result,_=train(c,y,s,y,[0,1],[],dict(architecture='volume',width=16,lr=.01,weight_decay=0.,epochs=100),rng_seed=149)
    with torch.no_grad():p=probabilities(model,torch.tensor(c),torch.tensor(y),torch.tensor(s))
    assert np.mean((p.numpy()>.5)==y)> .95
