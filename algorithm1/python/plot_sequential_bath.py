"""Stage-resolved finite-cycle results, matching joint evolution by exposure."""
import os
import json
import csv
from pathlib import Path
import numpy as np
from scipy.io import savemat


def main():
    root=Path(__file__).resolve().parents[1]/'results'/'sequential-bath'
    out=root/'figures';out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    seq=[json.loads(p.read_text()) for p in sorted(root.glob('sequential-*.json'))]
    joint=[json.loads(p.read_text()) for p in sorted(root.glob('simultaneous-*.json'))]
    def match(a):
        choices=[j for j in joint if np.isclose(j['H_to_bath_ratio'],a['H_to_bath_ratio'],atol=1e-12,rtol=0)]
        return choices[0] if choices else None
    rows=[]
    for a in seq:
        final=a['records'][-1];h=a['records'][-2];control=match(a)
        row={k:a[k] for k in ['t_h','t_b','H_to_bath_ratio','stabilized']}
        row.update(far_after_bath=final['far_aligned'],far_after_H=h['far_aligned'],
            interior_after_bath=final['interior_aligned'],
            linf_ground_after_bath=max(p['linf_ground'] for p in final['seed_profiles']),
            linf_ground_after_H=max(p['linf_ground'] for p in h['seed_profiles']),
            energy_above_ground_per_spin=final['energy_above_ground_per_spin'],
            simultaneous_far=control['records'][-1]['far_aligned'] if control else None,
            difference_from_simultaneous=final['far_aligned']-control['records'][-1]['far_aligned'] if control else None,
            final_cycle=final['cycle'],final_bath_exposure=final['bath_exposure'],
            **a['diagnostics'])
        rows.append(row)
    (root/'summary.json').write_text(json.dumps({'runs':rows},indent=2)+'\n')
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def cells(values):
        result=np.empty((1,len(values)),object)
        for i,v in enumerate(values):result[0,i]=np.asarray(v)
        return result
    savemat(root/'sequential-bath.mat',{
        't_h':[a['t_h'] for a in seq],'t_b':[a['t_b'] for a in seq],
        'edge_gamma_after_bath':cells([a['records'][-1]['seed_profiles'][0]['gamma'] for a in seq]),
        'edge_gamma_after_H':cells([a['records'][-2]['seed_profiles'][0]['gamma'] for a in seq]),
        'interior_gamma_after_bath':cells([a['records'][-1]['seed_profiles'][1]['gamma'] for a in seq]),
        'cycles':cells([[r['cycle'] for r in a['records']] for a in seq]),
        'phase':cells([[r['phase'] for r in a['records']] for a in seq]),
        'far_trajectories':cells([[r['far_aligned'] for r in a['records']] for a in seq])})
    # Default grid only; extra duration controls remain in exports.
    hs=[0.,.05,.2,.5,1.,2.,4.];bs=[.05,.2,.5,1.,2.]
    def field(key):
        return np.array([[next((r[key] for r in rows if r['t_h']==h and r['t_b']==b),np.nan) for b in bs] for h in hs],float)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    for ax,key,title,cmap in zip(axes.flat,
            ['far_after_bath','far_after_H','difference_from_simultaneous','energy_above_ground_per_spin'],
            ['Far correlation after bath','Far correlation after Hamiltonian',
             'After-bath correlation minus simultaneous control','Excess energy per spin'],
            ['viridis','viridis','RdBu_r','magma_r']):
        data=field(key);kw={}
        if key.startswith('far_'):kw={'vmin':0,'vmax':.085}
        if key=='difference_from_simultaneous':
            limit=max(np.nanmax(abs(data)),1e-5);kw={'vmin':-limit,'vmax':limit}
        im=ax.imshow(data,aspect='auto',cmap=cmap,**kw)
        ax.set(xticks=range(len(bs)),xticklabels=bs,yticks=range(len(hs)),yticklabels=hs,
               xlabel=r'Bath duration $\tau_B$',ylabel=r'Hamiltonian duration $\tau_H$',title=title)
        for i in range(len(hs)):
            for j in range(len(bs)):
                value=data[i,j]
                if np.isfinite(value):
                    red,green,blue,_=im.cmap(im.norm(value))
                    color='black' if .2126*red+.7152*green+.0722*blue>.5 else 'white'
                    label=f'{value:.4f}' if key!='difference_from_simultaneous' else f'{value if abs(value)>=.00005 else 0:+.4f}'
                    ax.text(j,i,label,ha='center',va='center',fontsize=8,color=color)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Finite mix-then-bath cycles | 2×4 image | b=6 | initial state I/256\n'
                 'Full generator exponentials; controls match integrated H and bath exposure',fontsize=12)
    save(fig,'duration-grid')

    selected=[(0.,.2),(.2,.2),(1.,.2),(2.,.5),(4.,1.)]
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    for h,b in selected:
        candidates=[a for a in seq if a['t_h']==h and a['t_b']==b]
        if not candidates:continue
        a=candidates[0];r=[x for x in a['records'] if x['phase']!='after_H'];u=[x['bath_exposure'] for x in r]
        label=fr'$\tau_H={h:g},\ \tau_B={b:g}$'
        axes[0,0].plot([x['cycle'] for x in r],[x['far_aligned'] for x in r],'o-',ms=3,label=label)
        axes[0,1].plot(u,[x['far_aligned'] for x in r],'o-',ms=3,label=label)
        axes[1,0].plot(u,[max(p['linf_ground'] for p in x['seed_profiles']) for x in r],'o-',ms=3,label=label)
        axes[1,1].plot(u,[x['energy_above_ground_per_spin'] for x in r],'o-',ms=3,label=label)
    axes[0,0].set(xlabel='Cycle (H stage followed by bath stage)',ylabel='Sign-aligned far correlation',xlim=(0,25))
    axes[0,1].set(xlabel=r'Accumulated bath exposure $u=m\tau_B$',ylabel='Sign-aligned far correlation',xlim=(0,4))
    axes[1,0].set(xlabel=r'Accumulated bath exposure $u=m\tau_B$',ylabel=r'$\max_s\|\Gamma_s-\Gamma_{s,GS}\|_\infty$',xlim=(0,4))
    axes[1,1].set(xlabel=r'Accumulated bath exposure $u=m\tau_B$',ylabel='Excess energy per spin',xlim=(0,4))
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('After-bath measurements: cycle count and equal bath exposure answer different questions\n'
                 'Markers are stage endpoints; connecting lines do not resolve the intervening evolution',fontsize=11)
    save(fig,'cycle-trajectories')

    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for h,b in [(1.,.2),(2.,.5),(4.,1.)]:
        candidates=[a for a in seq if a['t_h']==h and a['t_b']==b]
        if not candidates:continue
        a=candidates[0];r=a['records'][-2:];label=fr'$\tau_H={h:g},\ \tau_B={b:g}$'
        # Show a complete late-time cycle, including its initial after-bath point.
        phases=['Cycle start','After H','After bath']
        axes[0].plot(phases,[r[-1]['far_aligned'],r[0]['far_aligned'],r[-1]['far_aligned']],'o-',label=label)
        axes[1].plot(phases,[r[-1]['energy_above_ground_per_spin'],r[0]['energy_above_ground_per_spin'],r[-1]['energy_above_ground_per_spin']],'o-',label=label)
    axes[0].set(ylabel='Sign-aligned far correlation',title='Correlation can oscillate within a cycle')
    axes[1].set(ylabel='Excess energy per spin',title='H conserves energy; net bath change vanishes at the cycle limit')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Late-time stage endpoints | a stationary cycle need not be a stationary state',fontsize=12)
    save(fig,'stage-endpoints')
    print('Plots and exports:',root)


if __name__=='__main__':main()
