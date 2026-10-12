"""Resolve correlations throughout a late-time cycle, not just at its endpoints."""
import os
import tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
for key in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(key,'1')
import json
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
from sequential_bath import ExactEdgeDynamics
from ladder_tensor import build_ladder
from run_bath_modifications import dense_profiles,decorate


def main():
    root=Path(__file__).resolve().parents[1]/'results'/'sequential-bath'
    ladder=build_ladder(4);d=ExactEdgeDynamics(ladder)
    ground=json.loads((root.parent/'ladders'/'sample'/'ground-L4.json').read_text())
    reports=[]
    for th,tb in [(1.,.2),(2.,.05),(4.,.05),(4.,.2),(4.,1.)]:
        path=root/f'sequential-L4-H{th:g}-B{tb:g}.npz'
        start=d.pack(np.load(path)['rho']);stages=[]
        for samples in [101,201]:
            h_times=np.linspace(0,th,samples);b_times=np.linspace(0,tb,samples)
            h_states=[d.h_stage(start,d.unitary(t)) for t in h_times]
            b_states=expm_multiply(d.diss,h_states[-1],start=0,stop=tb,num=samples,endpoint=True,
                                  traceA=d.diss.diagonal().sum())
            records=[]
            for phase,times,vectors in [('H',h_times,h_states),('bath',b_times,b_states)]:
                for t,v in zip(times,vectors):
                    rho=d.unpack(v);profiles=decorate(dense_profiles(rho,4),ladder,ground['seed_profiles'])
                    records.append({'phase':phase,'stage_time':float(t),'cycle_time':float(t if phase=='H' else th+t),
                        'far_aligned':-profiles[0]['gamma'][1][-1],
                        'energy_above_ground_per_spin':float((np.einsum('ij,ji->',rho,d.h).real-d.evals[0])/8),
                        'seed_profiles':profiles})
            values=[r['far_aligned'] for r in records]
            mean=float((np.trapezoid(values[:samples],h_times)+np.trapezoid(values[samples:],b_times))/(th+tb))
            stages.append({'samples_per_stage':samples,'mean_far_over_physical_cycle':mean,
                           'minimum_sampled_far':min(values),'maximum_sampled_far':max(values),'records':records})
        report={'t_h':th,'t_b':tb,**stages[-1],
                'mean_sampling_refinement_change':abs(stages[0]['mean_far_over_physical_cycle']-stages[1]['mean_far_over_physical_cycle']),
                'minimum_sampling_refinement_change':abs(stages[0]['minimum_sampled_far']-stages[1]['minimum_sampled_far'])}
        reports.append(report)
        print(th,tb,'mean',report['mean_far_over_physical_cycle'],'min',report['minimum_sampled_far'],flush=True)
    (root/'within-cycle.json').write_text(json.dumps({'runs':reports},indent=2)+'\n')
    out=root/'figures';out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for th,tb,ax in [(4.,.05,axes[0]),(4.,1.,axes[1])]:
        a=next(x for x in reports if x['t_h']==th and x['t_b']==tb)
        for phase,color,label in [('H','#237e91','Hamiltonian stage'),('bath','#ce842d','Bath stage')]:
            r=[x for x in a['records'] if x['phase']==phase]
            ax.plot([x['cycle_time'] for x in r],[x['far_aligned'] for x in r],color=color,lw=2.5,label=label)
        joint=[json.loads(p.read_text()) for p in root.glob('simultaneous-*.json')]
        control=next(x for x in joint if np.isclose(x['H_to_bath_ratio'],th/tb))
        ax.axhline(control['records'][-1]['far_aligned'],ls='--',color='#965f9a',label='Matched simultaneous')
        ax.axhline(.080641341049451,ls=':',color='#526878',label='Bath only')
        ax.axvspan(th,th+tb,color='#ce842d',alpha=.12)
        ax.axhline(a['mean_far_over_physical_cycle'],ls='-.',color='#528446',label='Physical-time cycle average')
        ax.set(title=fr'$\tau_H={th:g},\ \tau_B={tb:g}$',xlabel='Time within late-time cycle',ylabel='Sign-aligned far correlation')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('The timing effect depends on when correlations are measured | 201 samples per stage')
    fig.savefig(out/'within-cycle.png',dpi=180);fig.savefig(out/'within-cycle.svg');plt.close(fig)


if __name__=='__main__':main()
