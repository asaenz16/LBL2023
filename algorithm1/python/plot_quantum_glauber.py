"""Plot verified quantum heat-bath trajectories, including sustained X noise."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(),'quantum-heat-bath-mpl'))
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig, output, name):
    fig.savefig(output/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(output/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'quantum-glauber')
    args=parser.parse_args()
    reports=[json.loads((args.input/f'n{n}'/'report.json').read_text()) for n in [4,8]]
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    colors=['#1b6a9b','#da7633','#2c9064','#9459a4']
    cases=['mixed-p0','plus_x-p0','ordered-p0','corrupted-p0']
    labels=['Maximally mixed','All +x (coherent)','Image / inverse mixture','10% corrupted image mixture']
    keys=['far_aligned','profile_linf_to_gibbs','trace_distance_to_gibbs']
    ylabels=[r'Sign-aligned far correlation $\Gamma$',r'$\|\Gamma-\Gamma_{\rm Gibbs}\|_\infty$',r'Trace distance $D(\rho,\rho_{\rm Gibbs})$']
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    summary=[]
    for row,report in enumerate(reports):
        for case,label,color in zip(cases,labels,colors):
            records=report['trajectories'][case]['records']
            for col,key in enumerate(keys):
                values=np.array([r[key] for r in records])
                axes[row,col].plot([r['sweep'] for r in records],np.maximum(values,1e-14) if col else values,
                                   color=color,label=label,lw=1.7)
            first=next((r['sweep'] for r in records if r['trace_distance_to_gibbs']<1e-6),None)
            summary.append({'spins':report['n_system'],'case':case,'first_sweep_trace_distance_below_1e_6':first,
                            'final_far_aligned':records[-1]['far_aligned'],'final_trace_distance':records[-1]['trace_distance_to_gibbs']})
        axes[row,0].axhline(report['gibbs']['far_aligned'],color='black',ls='--',lw=1,label='Exact Gibbs')
        for col in range(3):
            ax=axes[row,col];ax.set_xlabel('Sweeps');ax.set_ylabel(ylabels[col]);ax.grid(alpha=.18)
            ax.set_title(f"{report['n_system']} spins · {['far correlation','profile error','full quantum state'][col]}")
            ax.set_xlim(0, min(25,records[-1]['sweep']))
            if col:ax.set_yscale('log');ax.set_ylim(1e-14,1.1)
    axes[0,0].legend(fontsize=9,loc='lower right')
    fig.suptitle('Physical XXZ quantum heat bath · effective inverse temperature b = 6\n'
                 'One sweep visits every spin; curves use the verified channel. Errors below 10⁻¹⁴ are clipped.',fontsize=13)
    save(fig,args.input,'quantum-convergence')

    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    report=reports[1]
    for p,color in zip([0,.01,.05],['#1b6a9b','#da7633','#9459a4']):
        name=f'plus_x-p{p:g}';records=report['trajectories'][name]['records']
        label='No added noise' if p==0 else f'{p:.0%} X error / spin / sweep'
        for col,key in enumerate(keys):
            values=np.array([r[key] for r in records])
            axes[col].plot([r['sweep'] for r in records],np.maximum(values,1e-14) if col else values,
                           color=color,label=label,lw=2)
        if p:
            summary.append({'spins':8,'case':name,'final_far_aligned':records[-1]['far_aligned'],
                            'retained_far_fraction':records[-1]['far_aligned']/report['gibbs']['far_aligned'],
                            'final_profile_linf':records[-1]['profile_linf_to_gibbs'],
                            'final_trace_distance':records[-1]['trace_distance_to_gibbs'],
                            'final_sweep_residual':report['trajectories'][name]['final_sweep_residual']})
    axes[0].axhline(report['gibbs']['far_aligned'],color='black',ls='--',lw=1,label='Exact Gibbs')
    for col,ax in enumerate(axes):
        ax.set_xlabel('Sweeps');ax.set_ylabel(ylabels[col]);ax.grid(alpha=.18);ax.set_xlim(0,min(25,records[-1]['sweep']))
        if col:ax.set_yscale('log');ax.set_ylim(1e-14,1.1)
    axes[0].legend(fontsize=9,loc='lower right')
    fig.suptitle('8-spin 2 × 4 image · coherent all-+x start · sustained physical X errors\n'
                 'Measure after the noise layer. Nonzero plateaus describe states away from Gibbs.',fontsize=13)
    save(fig,args.input,'quantum-robustness')
    (args.input/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
