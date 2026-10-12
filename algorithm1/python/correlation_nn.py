"""Small neural decoders of the entire pair-correlation matrix.

Input includes pixel coordinates (known lattice geometry), but no intensities
except in the explicitly named image control. Label arrays never enter forward.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def permutations(side):
    a=np.arange(side*side).reshape(side,side)
    return torch.tensor(np.stack([np.rot90(np.fliplr(a) if flip else a,k).ravel().copy()
                                 for flip in (False,True) for k in range(4)]),dtype=torch.long)


def transform(c,image,seed,truth,p):
    """p maps new pixel positions to old; transform BOTH correlation axes."""
    return c[:,p][:,:,p],image[:,p],seed[:,p],truth[:,p]


def iou_values(pred,truth):
    pred=pred.bool();truth=truth.bool();inter=(pred&truth).sum(-1);union=(pred|truth).sum(-1)
    return torch.where(union>0,inter/union,torch.ones_like(union,dtype=torch.float32))


class Decoder(nn.Module):
    def __init__(self,side=16,width=24,architecture='volume',input_kind='full',seeded=True):
        super().__init__();self.side=side;self.n=side*side;self.architecture=architecture
        self.input_kind=input_kind;self.seeded=seeded
        y,x=torch.meshgrid(torch.linspace(-1,1,side),torch.linspace(-1,1,side),indexing='ij')
        self.register_buffer('xy',torch.stack([x.ravel(),y.ravel(),x.square().ravel(),y.square().ravel()]))
        grid=np.arange(self.n).reshape(side,side);neighbors=[];valid=[]
        for dy,dx in ((0,-1),(0,1),(-1,0),(1,0)):
            yy,xx=np.indices((side,side));ok=(yy+dy>=0)&(yy+dy<side)&(xx+dx>=0)&(xx+dx<side)
            neighbors.append(grid[np.clip(yy+dy,0,side-1),np.clip(xx+dx,0,side-1)].ravel());valid.append(ok.ravel())
        self.register_buffer('neighbors',torch.tensor(np.stack(neighbors),dtype=torch.long))
        self.register_buffer('valid',torch.tensor(np.stack(valid),dtype=torch.float32))
        # 4 coordinates, 4 seed features, 4 local correlations, 3 row moments.
        extra=15
        # Preserve the same positional capacity in controls: the constant
        # diagonal of a full C matrix already acts as a pixel-position code.
        main=(self.n if architecture=='volume' else 0)+(1 if input_kind=='image' else 0)
        self.project=nn.Conv2d(extra+main,width,1)
        if architecture=='volume':
            self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=d,dilation=d),nn.GELU(),
                nn.Conv2d(width,width,1)) for d in (1,2,4,1)])
        elif architecture=='graph':
            self.messages=nn.ModuleList([nn.Conv1d(3*width,width,1) for _ in range(3)])
            self.locals=nn.ModuleList([nn.Conv2d(width,width,3,padding=1) for _ in range(3)])
        else:raise ValueError(architecture)
        self.head=nn.Conv2d(width,1,1)

    def forward(self,c,image,seed):
        b,n,_=c.shape;s=self.side
        identity=torch.eye(n,dtype=c.dtype,device=c.device)[None].expand(b,-1,-1)
        source=c if self.input_kind in ('full','column') else identity
        feature_c=c if self.input_kind=='full' else identity
        xy=self.xy[None].expand(b,-1,-1)
        seed=seed if self.seeded else torch.zeros_like(seed)
        seedcorr=(source@seed[:,:,None]).squeeze(-1)
        sx=(seed*self.xy[0]).sum(-1,keepdim=True);sy=(seed*self.xy[1]).sum(-1,keepdim=True)
        exists=seed.sum(-1,keepdim=True)
        seedfeat=torch.stack([seed,seedcorr,(xy[:,0]-sx)*exists,(xy[:,1]-sy)*exists],1)
        index=torch.arange(n,device=c.device)[None,:]
        local=feature_c[:,index,self.neighbors]*self.valid[None]
        rows=torch.stack([feature_c.mean(-1),feature_c.abs().mean(-1),feature_c.square().mean(-1)],1)
        features=[xy,seedfeat,local,rows]
        if self.architecture=='volume':features.append(feature_c.transpose(1,2))
        if self.input_kind=='image':features.append(image[:,None]*2-1)
        h=F.gelu(self.project(torch.cat(features,1).reshape(b,-1,s,s)))
        if self.architecture=='volume':
            for block in self.blocks:h=F.gelu(h+block(h))
        else:
            mask=1-torch.eye(n,device=c.device)
            pos=F.relu(feature_c)*mask;neg=F.relu(-feature_c)*mask
            pos=pos/pos.sum(-1,keepdim=True).clamp_min(1e-6);neg=neg/neg.sum(-1,keepdim=True).clamp_min(1e-6)
            for msg,local in zip(self.messages,self.locals):
                q=h.flatten(2).transpose(1,2)
                m=msg(torch.cat([q,pos@q,neg@q],-1).transpose(1,2)).reshape(b,-1,s,s)
                h=F.gelu(h+m+local(h))
        return self.head(h).flatten(1)


def probabilities(model,c,image,seed):
    p=torch.sigmoid(model(c,image,seed))
    if model.seeded:p=torch.where(seed.bool(),torch.ones_like(p),p)
    return p


def loss(logits,truth):
    p=torch.sigmoid(logits)
    intersection=(p*truth).sum(-1)
    union=(p+truth-p*truth).sum(-1)
    return .5*F.binary_cross_entropy_with_logits(logits,truth)+.5*(1-(intersection+1)/(union+1)).mean()


def threshold_select(probs,truth):
    best=(-1,None)
    for t in np.arange(.25,.751,.05):
        score=float(iou_values(probs>t,truth).mean())
        if score>best[0]+1e-10:best=(score,float(round(t,2)))
    return best


def train(c,image,seed,truth,train_ids,validation_ids,config,*,input_kind='full',seeded=True,rng_seed=811,epochs=None):
    start=time.perf_counter();seed_everything(rng_seed)
    c=torch.as_tensor(c,dtype=torch.float32);image=torch.as_tensor(image,dtype=torch.float32).flatten(1)
    seed=torch.as_tensor(seed,dtype=torch.float32);truth=torch.as_tensor(truth,dtype=torch.float32).flatten(1)
    model=Decoder(int(np.sqrt(c.shape[1])),config['width'],config['architecture'],input_kind,seeded)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    nepoch=int(epochs or config['epochs']);generator=np.random.default_rng(rng_seed)
    permutations_=permutations(model.side);tr=np.asarray(train_ids,dtype=int);va=np.asarray(validation_ids,dtype=int)
    history=[];best=-1;best_state=None;best_epoch=nepoch;threshold=.5
    for epoch in range(1,nepoch+1):
        model.train();order=generator.permutation(tr);losses=[]
        for batch in range(0,len(order),16):
            idx=order[batch:batch+16];p=permutations_[generator.integers(8)]
            a,b,s,y=transform(c[idx],image[idx],seed[idx],truth[idx],p)
            optimizer.zero_grad(set_to_none=True);value=loss(model(a,b,s),y);value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(value.detach()))
        # Fixed schedule, independent of validation observations.
        for group in optimizer.param_groups:group['lr']=config['lr']*(.2+.8*(1+np.cos(np.pi*epoch/config['epochs']))/2)
        if len(va) and (epoch%5==0 or epoch==nepoch):
            model.eval()
            with torch.no_grad():
                vp=probabilities(model,c[va],image[va],seed[va]);vi,vt=threshold_select(vp,truth[va])
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_iou=vi,threshold=vt))
            if vi>best+1e-9:
                best=vi;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;threshold=vt
    if best_state is not None:model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp=probabilities(model,c[tr],image[tr],seed[tr]);train_iou=float(iou_values(tp>threshold,truth[tr]).mean())
        vp=probabilities(model,c[va],image[va],seed[va]).numpy() if len(va) else np.empty((0,c.shape[1]))
    result=dict(validation_iou=best if len(va) else None,train_iou=train_iou,epoch=best_epoch,threshold=threshold,
        seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),history=history,
        rng_seed=rng_seed,config=config,input_kind=input_kind,seeded=seeded)
    return model,result,vp
