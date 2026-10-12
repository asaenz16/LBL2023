"""Compare plaquette thermalization and coherent evolution with bond baths."""
import os
import tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',os.path.join(tempfile.gettempdir(),'quantum-segmentation-numba'))
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from ladder_tensor import build_ladder,dense_row_hamiltonian,rung_to_row_indices,save_mps,load_mps
from ladder_bath import mixed_density,density_profile,density_energy,density_reduced
from thermal_bath import trace_distance
from bath_modifications import make_plan,apply_mps_plan,apply_dense_plan,dense_vector,dense_matrix


def dense_profiles(rho,L):
    spins=((1-2*((np.arange(4**L)[:,None]>>np.arange(2*L-1,-1,-1))&1))/2).reshape(4**L,L,2).transpose(0,2,1)
    prob=np.diag(rho).real;means=np.einsum('i,irc->rc',prob,spins)
    return [{'seed':[0,col],'gamma':np.einsum('i,i,irc->rc',prob,spins[:,0,col],spins),
             'means':means,'connected':np.einsum('i,i,irc->rc',prob,spins[:,0,col],spins)-means[0,col]*means}
            for col in [0,L//4]]


def decorate(profiles,ladder,ground):
    truth=np.where(ladder.image==0,1.,-1.);out=[]
    for row,ref in zip(profiles,ground):
        seed=tuple(row['seed']);g=np.asarray(row['gamma']);mask=np.ones(g.shape,bool);mask[seed]=False
        margin=g*truth*truth[seed]
        out.append({**{k:np.asarray(v).tolist() for k,v in row.items()},
                    'linf_ground':float(np.max(abs(g-np.asarray(ref['gamma'])))),
                    'signed_margin':float(margin[mask].min())})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--widths',type=int,nargs='+',default=[4])
    p.add_argument('--model',choices=['bond','plaquette'],default='plaquette')
    p.add_argument('--plaquette-weighting',choices=['degree','uniform'],default='degree')
    p.add_argument('--coherent',action='store_true')
    p.add_argument('--integrator',choices=['fused','separate'],default='fused')
    p.add_argument('--etas',type=float,nargs='+',default=[1.])
    p.add_argument('--backend',choices=['dense','mps'],default='dense')
    p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--bond',type=int,default=32)
    p.add_argument('--b',type=float,default=6.)
    p.add_argument('--exposure',type=float,default=8.,help='Maximum dissipative exposure eta*t.')
    p.add_argument('--minimum-exposure',type=float,default=3.)
    p.add_argument('--tolerance',type=float,default=1e-8)
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'bath-modifications')
    a=p.parse_args()
    if a.backend=='dense' and any(L!=4 for L in a.widths):p.error('Dense benchmark is restricted to width 4.')
    if min(a.etas)<=0 or a.dt<=0 or a.exposure<=0:p.error('Require positive rates, step and exposure.')
    a.output.mkdir(parents=True,exist_ok=True)
    for L in a.widths:
        ladder=build_ladder(L)
        ground_path=Path(__file__).resolve().parents[1]/'results'/'ladders'/'sample'/f'ground-L{L}.json'
        gs=json.loads(ground_path.read_text())
        if a.backend=='dense':
            perm=rung_to_row_indices(L);h=dense_row_hamiltonian(ladder)[np.ix_(perm,perm)]
            gibbs=expm(-a.b*(h-np.linalg.eigvalsh(h)[0]*np.eye(4**L)));gibbs/=np.trace(gibbs)
        for eta in a.etas:
            stem=f'{a.model}-H{int(a.coherent)}-eta{eta:g}-L{L}-dt{a.dt:g}-{a.backend}-chi{a.bond if a.backend=="mps" else 0}'
            if a.coherent and a.integrator=='fused':stem+='-fused'
            if a.model=='plaquette' and a.plaquette_weighting=='uniform':stem+='-uniform'
            if a.b!=6.:stem+=f'-b{a.b:g}'
            path=a.output/(stem+'.json')
            if path.exists():
                print('Existing',stem,flush=True);continue
            start=time.time();plan=make_plan(ladder,a.model,a.b,a.dt,eta,a.coherent,a.integrator,a.plaquette_weighting)
            setup=time.time()-start;start_u=0.
            if a.resume:
                prior=json.loads(a.resume.read_text())
                expected=(L,a.model,a.coherent,eta,a.dt,a.b)
                actual=tuple(prior[k] for k in ['width','model','coherent','eta','dt','b'])
                if expected!=actual or a.backend!='mps':p.error('Resume must match model, width, rate, dt, b and MPS backend.')
                if prior.get('integrator','separate')!=a.integrator and a.coherent:p.error('Resume integrator must match.')
                prior_weight=prior.get('plaquette_weighting','degree')
                if prior_weight not in ['degree','uniform']:
                    prior_weight='uniform' if a.resume.stem.endswith('-uniform') else 'degree'
                if prior_weight!=a.plaquette_weighting:p.error('Resume weighting must match.')
                rho=load_mps(a.resume.with_suffix('.npz'));start_u=prior['records'][-1]['exposure']
            else:rho=(dense_vector(np.eye(4**L)/4**L,L) if a.backend=='dense' else mixed_density(L))
            records=[];drifts=[];stable=0;previous_matrix=None
            sample_steps=max(1,round(.5/(eta*a.dt)))
            max_steps=int(np.ceil(a.exposure/(eta*a.dt)))
            def measure(step):
                nonlocal previous_matrix
                row={'sweep':step,'time':start_u/eta+step*a.dt,'exposure':start_u+step*eta*a.dt}
                if a.backend=='dense':
                    matrix=dense_matrix(rho,L)
                    profiles=dense_profiles(matrix,L);e=float(np.trace(matrix@h).real)
                    row['distance_gibbs']=trace_distance(matrix,gibbs)
                    if previous_matrix is not None:row['state_change']=trace_distance(matrix,previous_matrix)
                    previous_matrix=matrix.copy()
                else:
                    profiles=[{'seed':seed,**{k:v for k,v in density_profile(rho,seed).items() if k!='trace'}} for seed in [(0,0),(0,L//4)]]
                    e=density_energy(rho,ladder);row['max_bond']=rho.max_bond()
                row.update(energy=e,energy_above_ground_per_spin=(e-gs['energy'])/(2*L),
                           seed_profiles=decorate(profiles,ladder,gs['seed_profiles']))
                if records:
                    row['profile_change']=max(np.max(abs(np.asarray(x['gamma'])-np.asarray(y['gamma'])))
                                              for x,y in zip(row['seed_profiles'],records[-1]['seed_profiles']))
                return row
            records.append(measure(0))
            for step in range(1,max_steps+1):
                if a.backend=='dense':rho=apply_dense_plan(rho,plan,L)
                else:drifts.append(apply_mps_plan(rho,plan,a.bond))
                if step%sample_steps==0 or step==max_steps:
                    row=measure(step);records.append(row)
                    change=row.get('state_change',row['profile_change'])
                    stable=stable+1 if change<a.tolerance else 0
                    print(stem,'u',round(row['exposure'],3),'far',f"{row['seed_profiles'][0]['gamma'][1][-1]:.6g}",'change',f'{change:.2e}','sec',round(time.time()-start,1),flush=True)
                    if row['exposure']>=a.minimum_exposure and stable>=3:break
            report={'width':L,'n_spins':2*L,'model':a.model,'coherent':a.coherent,'eta':eta,'dt':a.dt,'b':a.b,
                    'normalization':'sample','backend':a.backend,'integrator':a.integrator,'max_bond_setting':a.bond if a.backend=='mps' else None,
                    'records':records,'stabilized':stable>=3,'tolerance':a.tolerance,
                    'convergence_observable':'full-state trace distance' if a.backend=='dense' else 'two seed profiles',
                    'max_trace_renormalization':max(drifts,default=0.),'setup_seconds':setup,'runtime_seconds':time.time()-start,
                    'initial_state':'resumed' if a.resume else 'maximally mixed','resumed_from':str(a.resume) if a.resume else None,
                    'plaquette_weighting':a.plaquette_weighting,
                    'plaquette_weighting_description':'degree matches original per-spin bond proposal weight; uniform uses weight one; full internal plaquette H',
                    'limitations':['Neither modification guarantees the global Gibbs fixed point.',
                                   'MPS truncation is not positivity preserving.','Finite time and finite size results.']}
            if a.backend=='dense':
                matrix=dense_matrix(rho,L);future=dense_matrix(apply_dense_plan(rho,plan,L),L)
                report.update(minimum_eigenvalue=float(np.linalg.eigvalsh((matrix+matrix.conj().T)/2).min()),
                              trace=float(np.trace(matrix).real),hermiticity_error=float(np.linalg.norm(matrix-matrix.conj().T)),
                              final_cycle_residual=trace_distance(future,matrix),
                              gibbs_stationarity_residual=trace_distance(dense_matrix(apply_dense_plan(dense_vector(gibbs,L),plan,L),L),gibbs))
                np.savez_compressed(a.output/(stem+'.npz'),rho=matrix)
            else:
                reduced=[density_reduced(rho,j,2) for j in range(L-1)]
                report.update(minimum_two_rung_eigenvalue=min(float(np.linalg.eigvalsh((r+r.conj().T)/2).min()) for r in reduced),
                              maximum_two_rung_hermiticity_error=max(float(np.linalg.norm(r-r.conj().T)) for r in reduced))
                save_mps(a.output/(stem+'.npz'),rho)
            path.write_text(json.dumps(report,indent=2)+'\n')
            print('Finished',path,'seconds',report['runtime_seconds'],flush=True)


if __name__=='__main__':main()
