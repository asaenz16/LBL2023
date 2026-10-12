"""Figures for local XXZ Petz circuits, exact bias, and size scaling."""
import os,tempfile
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'local-petz-mpl'))
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from qiskit import QuantumCircuit,QuantumRegister
from local_petz import image_graph,build_local_petz
from quantum_glauber import update_circuit
from ladder_tensor import build_ladder


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/f'{name}.svg',bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'local-petz')
    args=p.parse_args();out=args.input;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    dense=json.loads((out/'dense'/'report.json').read_text())
    global_path=out.parent/'quantum-glauber'/'n8'/'report.json'
    global_data=json.loads(global_path.read_text()) if global_path.exists() else None
    files=[out/'trajectories'/f'L{w}-plus_x-chi32-r128.json' for w in [4,8,16]]
    available=[json.loads(f.read_text()) for f in files if f.exists()]
    endpoints={r['sweeps'][-1] for r in available}
    if len(endpoints)>1:raise ValueError('Complete all strip runs before comparing their final profiles.')
    endpoint=next(iter(endpoints),25)
    colors=['#177ea1','#c96535','#333333']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    keys=['far_aligned','profile_linf_to_gibbs','trace_distance_to_gibbs']
    labels=[r'Sign-aligned far correlation $\Gamma$',r'$\|\Gamma-\Gamma_{\rm Gibbs}\|_\infty$',r'$D(\rho,\rho_{\rm Gibbs})$']
    cases=[('Nearest-neighbor patch',dense['models']['star']['cases']['plus_x-p0']['records']),
           ('Three-rung patch',dense['models']['strip3']['cases']['plus_x-p0']['records'])]
    if global_data is not None and global_data['b']==dense['b']:
        cases.append(('Global recovery',global_data['trajectories']['plus_x-p0']['records']))
    for (label,records),color in zip(cases,colors):
        for ax,key in zip(axes,keys):
            ax.plot([r['sweep'] for r in records],[r[key] for r in records],lw=2,color=color,label=label)
    axes[0].axhline(dense['gibbs']['far_aligned'],color='gray',ls=':',label='Gibbs reference')
    for ax,label in zip(axes,labels):ax.set_xlabel('Sweeps');ax.set_ylabel(label);ax.set_xlim(0,endpoint);ax.grid(alpha=.2)
    axes[0].legend(fontsize=9,loc='lower right')
    fig.suptitle(f"8 XXZ spins · all-+x start · b = {dense['b']:g}\nLocal patches retain correlations while converging to different states",fontsize=13)
    save(fig,out,'local-vs-global')

    resources=json.loads((out/'circuits'/'report.json').read_text())
    strips=[r for r in resources if r['name'] in ['strip8','strip16','strip32']]
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for mode,color,label in [('unrouted',colors[0],'Unrestricted gates within each patch'),('routed',colors[1],'Routed on local edges')]:
        n=[r['n_spins'] for r in strips]
        axes[0].plot(n,[r[mode]['operations']['cx'] for r in strips],'o-',color=color,label=label)
        axes[1].plot(n,[r[mode]['depth'] for r in strips],'o-',color=color,label=label)
    for ax in axes:ax.set_xlabel('Image spins N');ax.grid(alpha=.2);ax.set_xticks(n)
    axes[0].set_ylabel('CNOTs per sweep');axes[1].set_ylabel('Circuit depth per sweep')
    axes[0].legend(fontsize=9)
    fig.suptitle('Nearest-neighbor Petz patches on 2 × L strips\nFive disjoint layers; one locally attached reset ancilla per spin',fontsize=13)
    save(fig,out,'circuit-scaling')

    model=build_local_petz(image_graph(build_ladder(4).image));u=model.updates[1]
    circuit=QuantumCircuit(*[QuantumRegister(1,f'p{p}') for p in reversed(u.sites)],QuantumRegister(1,'bath'))
    block=update_circuit(u.local)
    for instruction in block.data:
        if instruction.operation.name=='isometry':instruction.operation.label='Local recovery'
    circuit.compose(block,inplace=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={'width_ratios':[1,2]})
    ax=axes[0]
    for a,b in model.graph.edges:
        ra,ca=divmod(a,4);rb,cb=divmod(b,4)
        ax.plot([ca,cb],[-ra,-rb],color='#aaaaaa',zorder=0)
    for i in range(8):
        r,c=divmod(i,4);inside=i in u.sites
        ax.add_patch(Circle((c,-r),.22,facecolor='black' if model.graph.image[r,c]==0 else 'white',
                            edgecolor='#cc4d28' if i==1 else colors[0] if inside else '#aaaaaa',linewidth=4 if inside else 1))
        ax.text(c,-r,str(i),ha='center',va='center',color='white' if model.graph.image[r,c]==0 else 'black',fontsize=11)
    ax.set_xlim(-.4,3.4);ax.set_ylim(-1.8,.8);ax.set_aspect('equal');ax.axis('off')
    ax.set_title('Erase pixel 1; recover on {0, 1, 2, 5}',fontsize=11)
    ax.text(1.5,-1.65,'Other pixels are untouched by this update.',ha='center',fontsize=10)
    circuit.draw(output='mpl',ax=axes[1],fold=-1,style={'name':'bw','fontsize':10,'subfontsize':8})
    fig.suptitle('A bounded quantum heat-bath update\nThe recovery box uses only the selected patch; exports contain its elementary gates.',fontsize=13)
    fig.subplots_adjust(top=.81,wspace=.15)
    save(fig,out,'local-update')

    if available:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
        for r,color in zip(available,colors):
            if r['b']!=dense['b']:raise ValueError('Dense and trajectory experiments use different temperatures.')
            times=np.array(r['sweeps']);far=np.array(r['far_aligned']);err=1.96*np.array(r['far_se'])
            axes[0].plot(times,far,'o-',color=color,label=f"{r['n_spins']} spins")
            axes[0].fill_between(times,np.maximum(-.25,far-err),np.minimum(.25,far+err),color=color,alpha=.15)
            width=r['width'];g=np.array(r['gamma'])[-1,0,1];se=np.array(r['gamma_se'])[-1,0,1]
            sign=1-2*((np.arange(width)//2)%2)
            axes[2].errorbar(np.arange(width)+1,g*sign,yerr=1.96*se,fmt='o-',capsize=2,color=color,label=f"{r['n_spins']} spins")
            refpath=out.parent/'xxz-glauber'/'long-run'/f"L{width}-b{r['b']:g}.json"
            if width==4:
                reference=np.array(dense['gibbs']['gamma'])
            elif refpath.exists():
                ref=json.loads(refpath.read_text())
                if ref['b']!=r['b'] or ref['seeds']!=r['seeds'] or not np.isclose(ref['sigma'],r['sigma']):
                    raise ValueError('Reference and trajectory Hamiltonians or seeds differ.')
                reference=np.array(ref['reference']['gamma']).reshape(2,2,width)
            else:reference=None
            if reference is not None:
                distance=np.max(abs(np.array(r['gamma'])-reference),axis=(1,2,3))
                axes[1].plot(times,distance,'o-',color=color)
        axes[0].set_xlabel('Sweeps');axes[0].set_ylabel('Sign-aligned far-corner Γ');axes[0].legend()
        axes[1].set_xlabel('Sweeps');axes[1].set_ylabel(r'Estimated $\|\Gamma-\Gamma_{\rm Gibbs}\|_\infty$')
        axes[2].set_xlabel('Manhattan distance from top-left seed');axes[2].set_ylabel('Sign-aligned Γ to bottom-row pixels')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f'Local star channel · 128 quantum trajectories · all-+x start · final profiles at sweep {endpoint}\n'
                     '95% pointwise ensemble intervals on Γ; profile errors are point estimates, using exact / prior SSE Gibbs references.',fontsize=12)
        save(fig,out,'larger-strips')

    ordered_path=out/'trajectories'/'L16-ordered-chi32-r64.json'
    plus=next((r for r in available if r['width']==16),None)
    if ordered_path.exists() and plus is not None:
        ordered=json.loads(ordered_path.read_text())
        for key in ['b','sigma','mode','seeds','sweeps','max_bond','cutoff']:
            if ordered[key]!=plus[key]:raise ValueError(f'Initial-state controls differ in {key}.')
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
        for r,color,label in [(plus,colors[0],'All +x'),(ordered,colors[1],'Ordered image / inverse mixture')]:
            times=np.array(r['sweeps']);far=np.array(r['far_aligned']);err=1.96*np.array(r['far_se'])
            axes[0].plot(times,far,'o-',color=color,label=f"{label} ({r['replicas']} trajectories)")
            axes[0].fill_between(times,np.maximum(-.25,far-err),np.minimum(.25,far+err),color=color,alpha=.15)
            width=r['width'];sign=1-2*((np.arange(width)//2)%2)
            axes[1].errorbar(np.arange(width)+1,np.array(r['gamma'])[-1,0,1]*sign,
                             yerr=1.96*np.array(r['gamma_se'])[-1,0,1],fmt='o-',capsize=2,color=color)
        axes[0].set_xlabel('Sweeps');axes[0].set_ylabel('Sign-aligned far-corner Γ');axes[0].legend(fontsize=9)
        axes[1].set_xlabel('Manhattan distance from top-left seed');axes[1].set_ylabel(f'Sign-aligned Γ at sweep {endpoint}')
        for ax in axes:ax.axhline(0,color='gray',lw=.8,ls=':');ax.grid(alpha=.2)
        fig.suptitle('32 spins · local star channel · initial-state comparison\n95% pointwise ensemble intervals',fontsize=12)
        save(fig,out,'initial-state-comparison')


if __name__=='__main__':main()
