"""Compare all-+x and maximally mixed starts, retaining tensor error diagnostics."""
import os
import tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
for key in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(key,'1')
import json
import csv
from pathlib import Path
import numpy as np
from scipy.io import savemat
from ladder_tensor import build_ladder,load_mps
from ladder_bath import density_energy


def main():
    root=Path(__file__).resolve().parents[1]/'results'/'bath-plus-x'
    old=root.parent/'ladders'/'sample';out=root/'figures';out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    exact=json.loads((root/'exact-comparison.json').read_text());records=exact['records']
    early_path=root/'early-time'/'exact-comparison.json'
    if early_path.exists():
        early=json.loads(early_path.read_text())['records']
        records=early+[r for r in records if r['time']>early[-1]['time']+1e-12]
    t=np.array([r['time'] for r in records]);fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for name,label,color,style in [('mixed','Initially I/256','#526878','-'),('plus_x','Initially all +x','#cf7d25','--')]:
        data=[r['states'][name] for r in records]
        for ax,values in zip(axes.flat[:5],[[r['far_aligned'] for r in data],
                 [max(p['linf_ground'] for p in r['seed_profiles']) for r in data],
                 [r['energy_above_ground_per_spin'] for r in data],
                 [r['x_magnetization_per_spin'] for r in data],[r['purity'] for r in data]]):
            ax.plot(t,values,style,color=color,lw=2,label=label)
    axes[0,0].set(ylabel='Sign-aligned far-corner correlation',title='Image correlation',xlim=(0,2))
    axes[0,1].set(ylabel=r'$\max_s\|\Gamma_s-\Gamma_{s,GS}\|_\infty$',title='Ground-reference profile error',xlim=(0,2))
    axes[0,2].set(ylabel='Excess energy per spin',title='Full image energy',xlim=(0,1))
    axes[1,0].set(ylabel=r'$\frac{1}{n}\sum_i\langle S_i^x\rangle$',title='Initial x polarization decays',xlim=(0,1))
    axes[1,1].set(ylabel=r'$\mathrm{Tr}(\rho^2)$',title='Purity',xlim=(0,1))
    axes[1,2].semilogy(t,[r['trace_distance_between_initializations'] for r in records],label='Full-state trace distance',color='#237e91')
    profile_differences=np.array([r['profile_linf_between_initializations'] for r in records])
    axes[1,2].semilogy(t,np.where(profile_differences>0,profile_differences,np.nan),
                       label=r'Correlation-profile $\ell_\infty$ difference',color='#965f9a')
    axes[1,2].set(ylabel='Distance between the two trajectories',title='The initial-state difference disappears',ylim=(1e-14,2))
    for ax in axes.flat:ax.set_xlabel('Bath evolution time');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Bath only, 2×4 image | b=6 | full density operators in all magnetization-difference sectors\n'
                 'No coherent H stage, Trotter splitting, tensor truncation, or added noise',fontsize=12)
    fig.savefig(out/'exact-initial-state-comparison.png',dpi=180);fig.savefig(out/'exact-initial-state-comparison.svg');plt.close(fig)

    rows=[];tensor_runs=[]
    for path in sorted((root/'sample').glob('*-initplus_x.json')):
        a=json.loads(path.read_text());L=a['width'];chi=a['max_bond_setting']
        baseline_path=old/f'bath-L{L}-Z-p0-dt{a["dt"]:g}-chi{chi}.json'
        if not baseline_path.exists():continue
        baseline=json.loads(baseline_path.read_text());r=a['records'][-1];b=baseline['records'][-1]
        ground=json.loads((old/f'ground-L{L}.json').read_text())
        distance=max(float(np.max(abs(np.asarray(x['gamma'])-np.asarray(y['gamma']))))
                     for x,y in zip(r['seed_profiles'],b['seed_profiles']))
        ground_distance=max(float(np.max(abs(np.asarray(x['gamma'])-np.asarray(y['gamma']))))
                     for x,y in zip(r['seed_profiles'],ground['seed_profiles']))
        mixed_energy=density_energy(load_mps(baseline_path.with_suffix('.npz')),build_ladder(L))
        row={'width':L,'n_spins':2*L,'max_bond':chi,'dt':a['dt'],
             'plus_x_far':-r['seed_profiles'][0]['gamma'][1][-1],
             'mixed_far':-b['seed_profiles'][0]['gamma'][1][-1],
             'profile_linf_between_initializations':distance,'plus_x_profile_linf_ground':ground_distance,
             'plus_x_excess_energy_per_spin':(r['energy']-ground['energy'])/(2*L),
             'mixed_excess_energy_per_spin':(mixed_energy-ground['energy'])/(2*L),
             'final_x_magnetization_per_spin':r['x_magnetization_per_spin'],
             'final_sweep':a['last_sweep'],'stabilized':a['observable_stabilized'],
             'max_trace_renormalization':a['max_trace_renormalization'],
             'minimum_reduced_hermitian_eigenvalue':a['minimum_two_rung_eigenvalue'],
             'max_reduced_hermiticity_error':a['maximum_two_rung_hermiticity_error'],
             'resumed_from_bond':a['resumed_from_bond']}
        rows.append(row);tensor_runs.append(a)
    report={'exact':{'final_time':records[-1]['time'],'stabilized':exact['stabilized'],
                    'initial':records[0],'final':records[-1],
                    'maximum_profile_difference':max(r['profile_linf_between_initializations'] for r in records),
                    'maximum_plus_x_energy':max(r['states']['plus_x']['energy'] for r in records),
                    'time_of_maximum_plus_x_energy':max(records,key=lambda r:r['states']['plus_x']['energy'])['time'],
                    'diagnostics':exact['diagnostics']},'tensor_runs':rows}
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if rows:
        with (root/'size-comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
        for chi,style in [(32,':'),(64,'-')]:
            subset=sorted([r for r in rows if r['max_bond']==chi],key=lambda r:r['width'])
            if not subset:continue
            for name,label,color,marker in [('mixed','Mixed','#526878','o'),('plus_x','All +x','#cf7d25','x')]:
                axes[0].semilogy([r['width'] for r in subset],[r[f'{name}_far'] for r in subset],
                                 marker+style,color=color,label=f'{label}, chi={chi}',ms=7)
            axes[1].semilogy([r['width'] for r in subset],
                             np.maximum([r['profile_linf_between_initializations'] for r in subset],1e-16),
                             'o'+style,label=f'chi={chi}')
        axes[0].set(ylabel='Late-time sign-aligned far correlation',title='Compare initializations at matching chi')
        axes[1].set(ylabel=r'$\max_s\|\Gamma_s^{+x}-\Gamma_s^{mixed}\|_\infty$',title='Difference between final correlation profiles')
        for ax in axes:ax.set_xlabel('Far-corner separation / strip width L');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('2×L strips | same edge bath | tensor step 0.1\n'
                     'Agreement between starts does not remove bond-dimension or time-step error',fontsize=12)
        fig.savefig(out/'size-comparison.png',dpi=180);fig.savefig(out/'size-comparison.svg');plt.close(fig)
    def cells(values):
        result=np.empty((1,len(values)),object)
        for i,v in enumerate(values):result[0,i]=np.asarray(v)
        return result
    savemat(root/'bath-plus-x.mat',{
        'exact_time':t,
        'exact_mixed_edge_gamma':cells([r['states']['mixed']['seed_profiles'][0]['gamma'] for r in records]),
        'exact_plus_x_edge_gamma':cells([r['states']['plus_x']['seed_profiles'][0]['gamma'] for r in records]),
        'exact_trace_distance':[r['trace_distance_between_initializations'] for r in records],
        'exact_profile_linf_difference':[r['profile_linf_between_initializations'] for r in records],
        'tensor_widths':[r['width'] for r in rows],'tensor_max_bond':[r['max_bond'] for r in rows],
        'tensor_plus_x_edge_gamma':cells([a['records'][-1]['seed_profiles'][0]['gamma'] for a in tensor_runs])})
    print('Saved plots, summary, CSV and MATLAB export:',root)


if __name__=='__main__':main()
