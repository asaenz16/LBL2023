"""Assemble size/separation plots and numerical checks from saved ladder runs."""
import os
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np


def positive(values):
    """Leave nonpositive values missing on a log axis, never invent a floor."""
    values=np.asarray(values)
    return np.where(values>0,values,np.nan)


def read(path):
    return json.loads(path.read_text())


def aligned(profile,width,seed=(0,0),kind='gamma'):
    sign=1-2*((np.arange(width)//2)%2)
    return np.asarray(profile[kind])*sign[None,:]*sign[seed[1]]


def choose_quench(root,width,angle):
    candidates=[root/f'quench-L{width}-uniform-theta{angle:g}-dt0.02-chi128.json',
                root/f'quench-L{width}-uniform-theta{angle:g}-dt0.04-chi64.json']
    return read(next(p for p in candidates if p.exists()))


def main():
    base=Path(__file__).resolve().parents[1]/'results'/'ladders';root=base/'sample'
    out=base/'figures';out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    widths=[4,8,12,16];colors=['#267a9e','#d58b20','#8759a9','#35876a']
    grounds={L:read(root/f'ground-L{L}.json') for L in widths}
    quenches={(L,a):choose_quench(root,L,a) for L in widths for a in [.3,.6]}
    baths={}
    for L in widths:
        for axis,p in [('Z',0.),('Z',.02),('X',.02)]:
            path=root/f'bath-L{L}-{axis}-p{p:g}-dt0.1-chi64.json'
            if not path.exists():path=root/f'bath-L{L}-{axis}-p{p:g}-dt0.1-chi32.json'
            baths[L,axis,p]=read(path)
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180)
        fig.savefig(out/f'{name}.svg');plt.close(fig)

    fig,axes=plt.subplots(4,1,figsize=(10,4.7),layout='constrained',sharex=True)
    for ax,L in zip(axes,widths):
        ax.set_facecolor('#e7e7e7')
        ax.imshow(np.asarray(grounds[L]['image']),cmap='gray',vmin=0,vmax=255,extent=(-.5,L-.5,1.5,-.5))
        ax.add_patch(Rectangle((-.5,-.5),L,2,fill=False,edgecolor='#333',lw=1.5))
        ax.scatter([0,L//4],[0,0],facecolors='none',edgecolors=['#1aa7ec','#ed8b23'],s=85,lw=2)
        ax.set(yticks=[0,1],ylabel=f'2 x {L}',xlim=(-.5,15.5))
        ax.set_xticks(np.arange(-.5,L,1),minor=True);ax.set_yticks([-.5,.5,1.5],minor=True)
        ax.grid(which='minor',color='#777',lw=.5);ax.tick_params(which='minor',length=0)
    axes[-1].set_xlabel('Column (zero-based)')
    fig.suptitle('Alternating black/white 2x2 blocks | blue: edge seed; orange: interior seed')
    save(fig,'lattices')

    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for L,color in zip(widths,colors):
        for ax,a in zip(axes[0],[.3,.6]):
            record=quenches[L,a]['records'][-1]
            ax.plot(np.arange(1,L+1),aligned(record['seed_profiles'][0],L)[1],'.-',color=color,label=f'2 x {L}')
            ax.set(xlabel='Manhattan distance from upper-left seed',ylabel='Sign-aligned correlation',ylim=(0,.26))
            ax.set_title(f'After isolated evolution to t=6; rotation angle {a:g}')
        q=quenches[L,.3];times=[r['time'] for r in q['records']]
        axes[1,0].plot(times,[aligned(r['seed_profiles'][0],L)[1,-1] for r in q['records']],color=color,label=f'2 x {L}')
        axes[1,1].plot(times,[r['seed_profiles'][0]['linf_reference'] for r in q['records']],color=color,label=f'2 x {L}')
    axes[1,0].set(title='Farthest-pair correlation after the 0.3-radian pulse',xlabel='Hamiltonian evolution time',ylabel='Sign-aligned correlation')
    axes[1,1].set(title='Correlation l-infinity distance from ground reference',xlabel='Hamiltonian evolution time',ylabel=r'$\|\Gamma(t)-\Gamma_{GS}\|_\infty$')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Existing image order survives these finite-time perturbations\n'
                 'Sign-aligned means multiplied by the expected black/white sign; positive = correct label',fontsize=12)
    save(fig,'isolated-robustness')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for seed_index,ax in enumerate(axes):
        distances=[L if seed_index==0 else L//2+1 for L in widths]
        def value(profiles,L):
            target=L-1 if seed_index==0 else 3*L//4
            return aligned(profiles[seed_index],L,tuple(profiles[seed_index]['seed']))[1,target]
        ax.semilogy(distances,[value(grounds[L]['seed_profiles'],L) for L in widths],'o-',label='Ground reference',color='#666')
        ax.semilogy(distances,[value(quenches[L,.3]['records'][-1]['seed_profiles'],L) for L in widths],'^-',label='Isolated: rotation 0.3, t=6',color='#267a9e')
        for axis,p,label,color,style in [('Z',0.,'Local bath, no added noise','#34885b','o-'),
                                       ('Z',.02,'Local bath + Z noise, p=0.02','#965bad','s--'),
                                       ('X',.02,'Local bath + X noise, p=0.02','#d58b20','D-')]:
            ax.semilogy(distances,positive([value(baths[L,axis,p]['records'][-1]['seed_profiles'],L) for L in widths]),style,label=label,color=color)
        ax.set(xlabel='Pair separation (lattice edges)',ylabel='Sign-aligned correlation',title='Edge seed, opposite far corner' if seed_index==0 else 'Interior seed and distant interior column')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Size and separation: robustness depends on the prepared state\n'
                 'Local-bond bath is a separate preparation model; it does not preserve the full Gibbs state',fontsize=12)
    save(fig,'size-and-separation')

    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for L,color in zip(widths,colors):
        bath=baths[L,'Z',0.]
        axes[0].semilogy(np.arange(1,L+1),positive(aligned(bath['records'][-1]['seed_profiles'][0],L)[1]),'.-',color=color,label=f'2 x {L}')
    for axis,p,color,label in [('Z',0.,'#34885b','No added noise'),('Z',.02,'#965bad','Z noise p=0.02'),('X',.02,'#d58b20','X noise p=0.02')]:
        bath=baths[16,axis,p]
        axes[1].semilogy(np.arange(1,17),positive(aligned(bath['records'][-1]['seed_profiles'][0],16)[1]),'.-',color=color,label=label)
    axes[0].set(title='Local bath without added noise',xlabel='Distance from upper-left seed',ylabel='Sign-aligned correlation')
    axes[1].set(title='2 x 16 local bath: sustained perturbations',xlabel='Distance from upper-left seed',ylabel='Sign-aligned correlation')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Late-sweep bath profiles: strong short-distance signals, suppressed distant signals')
    save(fig,'bath-spatial-correlations')

    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    for row,(axis,p,label) in enumerate([('Z',0.,'No added noise'),
                                         ('Z',.02,'Z noise p=0.02'),
                                         ('X',.02,'X noise p=0.02')]):
        for L,color in zip(widths,colors):
            coarse=read(root/f'bath-L{L}-{axis}-p{p:g}-dt0.1-chi32.json')
            reference=np.asarray(grounds[L]['seed_profiles'][0]['gamma'])
            for run,style in [(coarse,'-'),(baths[L,axis,p],'--')]:
                records=run['records'];sweeps=[r['sweep'] for r in records]
                far=[aligned(r['seed_profiles'][0],L)[1,-1] for r in records]
                linf=[np.max(abs(np.asarray(r['seed_profiles'][0]['gamma'])-reference)) for r in records]
                axes[row,0].semilogy(sweeps,positive(far),style,color=color,
                                    label=f'2 x {L}' if style=='-' else None)
                axes[row,1].plot(sweeps,linf,style,color=color)
        axes[row,0].set(title=label,ylabel='Farthest-pair aligned correlation')
        axes[row,1].set(title=label,ylabel=r'$\|\Gamma(s)-\Gamma_{GS}\|_\infty$')
        axes[row,0].legend(fontsize=8,ncol=2)
    for ax in axes.flat:ax.set_xlabel('Bath sweep');ax.grid(alpha=.2)
    fig.suptitle('Bath evolution from the maximally mixed state\n'
                 'Solid: bond dimension 32; dashed: continuation at 64. Reference: ground-state profile.',fontsize=12)
    save(fig,'bath-sweeps')

    summary={'widths':widths,'n_spins':[2*L for L in widths],
             'versions':{k:version(k) for k in ['numpy','scipy','quimb','qiskit','matplotlib']},
             'isolated':[],'bath':[],'quench_refinements':[],'bath_bond_refinements':[]}
    for L in widths:
        for angle in [.3,.6]:
            q=quenches[L,angle];last=q['records'][-1]
            summary['isolated'].append({'width':L,'angle':angle,'dt':q['dt'],'bond_setting':q['max_bond_setting'],
              'far_gamma_aligned':float(aligned(last['seed_profiles'][0],L)[1,-1]),
              'bulk_far_gamma_aligned':float(aligned(last['seed_profiles'][1],L,(0,L//4))[1,3*L//4]),
              'minimum_margin_over_times_and_both_seeds':min(p['signed_margin'] for r in q['records'] for p in r['seed_profiles']),
              'distance_from_ground':last['distance_from_ground'],
              'energy_above_ground_per_spin':(q['records'][0]['energy']-grounds[L]['energy'])/(2*L),
              'max_energy_drift':q['max_energy_drift'],'max_local_magnetization':q['max_local_magnetization']})
            coarse=read(root/f'quench-L{L}-uniform-theta{angle:g}-dt0.04-chi64.json')
            if q['max_bond_setting']>64:
                differences=[np.max(abs(np.asarray(a['gamma'])-np.asarray(b['gamma'])))
                             for r,s in zip(coarse['records'],q['records'])
                             for a,b in zip(r['seed_profiles'],s['seed_profiles'])]
                summary['quench_refinements'].append({'width':L,'angle':angle,'maximum_gamma_difference':float(max(differences))})
        for axis,p in [('Z',0.),('Z',.02),('X',.02)]:
            b=baths[L,axis,p];last=b['records'][-1]
            summary['bath'].append({'width':L,'axis':axis,'p':p,'bond_setting':b['max_bond_setting'],
              'last_sweep':b['last_sweep'],'observable_stabilized':b['observable_stabilized'],
              'last_record_change':last.get('change_since_previous_record'),
              'far_gamma_aligned':float(aligned(last['seed_profiles'][0],L)[1,-1]),
              'bulk_far_gamma_aligned':float(aligned(last['seed_profiles'][1],L,(0,L//4))[1,3*L//4]),
              'minimum_reduced_eigenvalue':b['minimum_two_rung_eigenvalue'],
              'max_trace_renormalization':b['max_trace_renormalization']})
            if b['max_bond_setting']>32:
                coarse=read(root/f'bath-L{L}-{axis}-p{p:g}-dt0.1-chi32.json')
                differences=[np.max(abs(np.asarray(a['gamma'])-np.asarray(c['gamma']))) for a,c in zip(
                    last['seed_profiles'],coarse['records'][-1]['seed_profiles'])]
                summary['bath_bond_refinements'].append({'width':L,'axis':axis,'p':p,
                  'maximum_gamma_difference':float(max(differences)),
                  'far_gamma_difference':abs(last['seed_profiles'][0]['gamma'][1][-1]-coarse['records'][-1]['seed_profiles'][0]['gamma'][1][-1])})
    from ladder_tensor import load_mps,build_ladder
    from ladder_bath import density_energy
    for row in summary['bath']:
        L=row['width'];axis=row['axis'];p=row['p'];bond=row['bond_setting']
        rho=load_mps(root/f'bath-L{L}-{axis}-p{p:g}-dt0.1-chi{bond}.npz')
        e=density_energy(rho,build_ladder(L))
        row['energy_per_spin']=e/(2*L)
        row['energy_above_ground_per_spin']=(e-grounds[L]['energy'])/(2*L)
    (base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import csv
    with (base/'size-comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['width','spins','edge_pair_distance','isolated_theta0.3_t6','bath_p0','bath_Z_p0.02','bath_X_p0.02'])
        for L in widths:
            writer.writerow([L,2*L,L,aligned(quenches[L,.3]['records'][-1]['seed_profiles'][0],L)[1,-1],
              *[aligned(baths[L,axis,p]['records'][-1]['seed_profiles'][0],L)[1,-1] for axis,p in [('Z',0.),('Z',.02),('X',.02)]]])
    from scipy.io import savemat
    def cells(values):
        result=np.empty((1,len(values)),dtype=object)
        for i,value in enumerate(values):result[0,i]=np.asarray(value)
        return result
    savemat(base/'ladder-correlations.mat',{'widths':widths,'spins':np.array(widths)*2,
      'ground_gamma':cells([grounds[L]['seed_profiles'][0]['gamma'] for L in widths]),
      'isolated_theta03_gamma_t6':cells([quenches[L,.3]['records'][-1]['seed_profiles'][0]['gamma'] for L in widths]),
      'bath_p0_gamma':cells([baths[L,'Z',0.]['records'][-1]['seed_profiles'][0]['gamma'] for L in widths]),
      'bath_Z_p002_gamma':cells([baths[L,'Z',.02]['records'][-1]['seed_profiles'][0]['gamma'] for L in widths]),
      'bath_X_p002_gamma':cells([baths[L,'X',.02]['records'][-1]['seed_profiles'][0]['gamma'] for L in widths])})
    print('Figures:',out)
    print('Summary:',base/'summary.json')


if __name__=='__main__':main()
