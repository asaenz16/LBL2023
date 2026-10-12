"""Plot exact comparisons, tensor sensitivity and compiled circuit estimates."""
import os
import json
import csv
from pathlib import Path
import numpy as np


def read(path):
    value=json.loads(path.read_text())
    if 'model' in value and value.get('plaquette_weighting') not in ['degree','uniform']:
        value['plaquette_weighting']='uniform' if path.stem.endswith('-uniform') else 'degree'
    return value


def main():
    root=Path(__file__).resolve().parents[1]/'results'/'bath-modifications'
    old=root.parent/'ladders'/'sample';out=root/'figures';out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    runs=[read(p) for p in sorted(root.glob('*.json')) if '-H' in p.name]
    rows=[]
    for a in runs:
        r=a['records'][-1];L=a['width'];profiles=r['seed_profiles']
        sign=1-2*((np.arange(L)//2)%2)
        far=profiles[0]['gamma'][1][-1]*sign[-1]
        interior=profiles[1]['gamma'][1][3*L//4]*sign[L//4]*sign[3*L//4]
        rows.append({k:a[k] for k in ['model','coherent','eta','width','dt','backend','max_bond_setting']}|
                    {'integrator':a.get('integrator','separate'),'plaquette_weighting':a.get('plaquette_weighting','degree'),'far_aligned':far,'interior_aligned':interior,
                     'energy_above_ground_per_spin':r['energy_above_ground_per_spin'],'distance_gibbs':r.get('distance_gibbs'),
                     'time':r['time'],'exposure':r['exposure'],'stabilized':a['stabilized'],
                     'max_trace_renormalization':a['max_trace_renormalization'],
                     'minimum_reduced_eigenvalue':a.get('minimum_two_rung_eigenvalue'),
                     'final_cycle_residual':a.get('final_cycle_residual')})
    (root/'comparison-summary.json').write_text(json.dumps({'runs':rows},indent=2)+'\n')
    with (root/'comparison-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    from scipy.io import savemat
    def cells(values):
        result=np.empty((1,len(values)),dtype=object)
        for j,value in enumerate(values):result[0,j]=np.asarray(value)
        return result
    savemat(root/'bath-modifications.mat',{
        'widths':[a['width'] for a in runs],'eta':[a['eta'] for a in runs],
        'coherent':[a['coherent'] for a in runs],'dt':[a['dt'] for a in runs],
        'model':cells([a['model'] for a in runs]),'backend':cells([a['backend'] for a in runs]),
        'max_bond':[a['max_bond_setting'] or 0 for a in runs],
        'plaquette_weighting':cells([a.get('plaquette_weighting','degree') for a in runs]),
        'integrator':cells([a.get('integrator','separate') for a in runs]),
        'edge_gamma':cells([a['records'][-1]['seed_profiles'][0]['gamma'] for a in runs]),
        'interior_gamma':cells([a['records'][-1]['seed_profiles'][1]['gamma'] for a in runs])})
    def pick(model,coherent,eta,L=4,dt=.025,backend='dense',bond=None):
        candidates=[a for a in runs if (a['model'],a['coherent'],a['eta'],a['width'],a['dt'],a['backend'])==(model,coherent,eta,L,dt,backend)
                    and a.get('plaquette_weighting','degree')=='degree'
                    and (not coherent or a.get('integrator','separate')=='fused') and (bond is None or a['max_bond_setting']==bond)]
        return candidates[0] if candidates else None
    specs=[('bond',False,1.,'Original bond','#526878'),('plaquette',False,1.,'Plaquette','#cd7c24'),
           ('bond',True,1.,'H + bath, eta=1','#237e91'),('bond',True,.2,'H + bath, eta=0.2','#965f9a'),
           ('bond',True,.05,'H + bath, eta=0.05','#b94352')]
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for model,coherent,eta,label,color in specs:
        a=pick(model,coherent,eta);r=a['records'];u=[x['exposure'] for x in r]
        axes[0,0].plot(u,[-x['seed_profiles'][0]['gamma'][1][-1] for x in r],color=color,label=label)
        axes[0,1].plot(u,[x['seed_profiles'][0]['linf_ground'] for x in r],color=color,label=label)
        axes[1,0].plot(u,[x['energy_above_ground_per_spin'] for x in r],color=color,label=label)
        axes[1,1].plot(u,[x['distance_gibbs'] for x in r],color=color,label=label)
    for ax in axes.flat:ax.set_xlabel(r'Dissipative exposure $u=\eta t$');ax.grid(alpha=.2)
    axes[0,0].set(title='Distant correlation: larger is better',ylabel='Sign-aligned far-corner correlation')
    axes[0,1].set(title='Correlation error from ground reference',ylabel=r'$\|\Gamma-\Gamma_{GS}\|_\infty$')
    axes[1,0].set(title='Energy of the full image Hamiltonian',ylabel='Excess energy per spin')
    axes[1,1].set(title='Global Gibbs state as a diagnostic',ylabel='Trace distance to Gibbs')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Exact eight-spin density matrices | b=6 | step 0.025 | initially maximally mixed\n'
                 'A weaker bath needs more elapsed time to reach the same exposure',fontsize=12)
    save(fig,'exact-comparison')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    widths=[4,8,12,16]
    for ax,interior in zip(axes,[False,True]):
        def value(a):
            L=a['width'];p=a['records'][-1]['seed_profiles'][int(interior)];j=3*L//4 if interior else L-1
            seed=L//4 if interior else 0;sign=1-2*((np.arange(L)//2)%2)
            return np.asarray(p['gamma'])[1,j]*sign[seed]*sign[j]
        baselines=[read(old/f'bath-L{L}-Z-p0-dt0.1-chi64.json') for L in widths]
        distances=[L//2+1 if interior else L for L in widths]
        ax.semilogy(distances,[value(a) for a in baselines],'o-',color='#526878',label='Original bond, chi=64')
        for model,coherent,color,label in [('plaquette',False,'#cd7c24','Plaquette'),('bond',True,'#237e91','H + bond, eta=1')]:
            for chi,style in [(32,':'),(64,'--'),(128,'-')]:
                values=[];xs=[]
                for L in widths:
                    a=pick(model,coherent,1.,L,.1,'mps',chi)
                    if a is not None:
                        v=value(a);values.append(v if v>0 else np.nan);xs.append(L//2+1 if interior else L)
                if values:ax.semilogy(xs,values,'o'+style,color=color,label=f'{label}, chi={chi}',ms=4)
        ax.set(xlabel='Pair separation',ylabel='Sign-aligned correlation',title='Interior pairs' if interior else 'Far corners')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Larger strips: show bond-dimension sensitivity explicitly\n'
                 'Tiny plaquette signals are not certified as converged; step 0.1',fontsize=12)
    save(fig,'size-and-truncation')

    fig,ax=plt.subplots(figsize=(8.2,4.8),layout='constrained')
    resource=read(root/'circuit-resources.json')['ladder_estimates']
    n=[r['system_qubits'] for r in resource]
    bond=np.array([r['bond_cx_per_collision_sweep_estimate'] for r in resource])
    plaque=[r['plaquette_cx_per_collision_sweep_estimate'] for r in resource]
    coherent=np.array([r['coherent_added_cx_per_strang_step'] for r in resource])
    ax.semilogy(n,bond,'o-',label='Bond collisions')
    ax.semilogy(n,bond+coherent,'s--',label='Bond collisions + coherent gates')
    ax.semilogy(n,plaque,'D-',label='Plaquette collisions')
    ax.set(xlabel='System spins',ylabel='Estimated CNOTs per unsubdivided collision sweep',xticks=n)
    ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Generic local synthesis: linear size scaling, very different constants\n'
                 'Excludes contact subdivision, repeated sweeps, shots, routing and error correction',fontsize=11)
    save(fig,'circuit-costs')
    print('Plots and comparison summary:',root)


if __name__=='__main__':main()
