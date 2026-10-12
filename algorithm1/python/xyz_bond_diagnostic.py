"""Analytic isolated-bond diagnostic; not a substitute for image Gibbs tests."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/xyz-matplotlib')
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq


def bond_alignment(couplings,b):
    """Exact raw <XX+YY+ZZ> for H=-(Jx XX+Jy YY+Jz ZZ)/4."""
    j=np.asarray(couplings,float)
    log_ratios=b*.5*np.stack([j[...,0]+j[...,1],j[...,0]+j[...,2],j[...,1]+j[...,2]],axis=-1)
    logs=np.concatenate([np.zeros((*log_ratios.shape[:-1],1)),log_ratios],axis=-1)
    singlet_probability=np.exp(-logsumexp(logs,axis=-1))
    return 1-4*singlet_probability


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from weizmann_petz_data import ROOT,write_json
    d=np.linspace(0,180,1000);thresholds=np.array([25.,75.,125.]);alpha=.01
    j=np.arctan(alpha*(thresholds-d[:,None]));fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k,color in enumerate(['#cb4b16','#268bd2','#6c71c4']):axes[0].plot(d,j[:,k],color=color,label=['Jx','Jy','Jz'][k])
    axes[0].axhline(0,color='gray',lw=.8);axes[0].set(xlabel='Neighbor intensity difference d',ylabel='Coupling J',title='Three distinct interaction thresholds')
    axes[0].legend();roots={}
    for b in [.5,2.,8.]:
        y=bond_alignment(j,b);axes[1].plot(d,y,label=f'b = {b:g}')
        roots[str(b)]=float(brentq(lambda x:bond_alignment(np.arctan(alpha*(thresholds-x)),b),0,180))
        axes[1].plot(roots[str(b)],0,'o',color=axes[1].lines[-1].get_color())
    axes[1].axhline(0,color='gray',lw=.8);axes[1].axvline(75,color='gray',ls=':',label='High-temperature limit: 75')
    axes[1].axvline(100,color='gray',ls='--',label='Zero-temperature crossing limit: 100')
    axes[1].set(xlabel='Neighbor intensity difference d',ylabel='Raw alignment <XX+YY+ZZ>',title='One alignment sign crossing at each temperature',ylim=(-3.1,1.1))
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Isolated-bond diagnostic: staggered thresholds move the effective boundary\nIllustrative fixed thresholds (25,75,125), alpha = 0.01, equal axis amplitudes')
    fig.tight_layout();out=ROOT/'results/weizmann-xyz/figures';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'bond-thresholds.png',dpi=180,bbox_inches='tight');fig.savefig(out/'bond-thresholds.pdf',bbox_inches='tight')
    write_json(ROOT/'weizmann-xyz-bond-diagnostic.json',dict(thresholds=thresholds.tolist(),alpha=alpha,alignment_zero_by_b=roots,
        statement='For an isolated bond at fixed positive inverse temperature, thresholds, positive arctangent sensitivities and amplitudes, raw alignment is strictly decreasing in d and has one sign crossing over the real line. This does not establish monotonicity when adaptive parameters are recalculated or for correlations in a full image.'))
    print(roots)


if __name__=='__main__':main()
