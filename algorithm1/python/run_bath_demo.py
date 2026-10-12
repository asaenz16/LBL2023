"""Validate thermal-bath preparation and export a MATLAB/QCLAB replay fixture."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
from scipy.io import savemat
from qiskit import qasm2, qpy, transpile

from algorithm1 import image_model, seed_correlations
from thermal_bath import (apply_contact, build_bath, channel_matrix, contact_circuit,
                          davies_generator, reference_gibbs, sweep_circuit, trace_distance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta",type=float,default=24)
    parser.add_argument("--dt",type=float,default=0.1)
    parser.add_argument("--sweeps",type=int,default=60)
    parser.add_argument("--global-flip",action="store_true")
    parser.add_argument("--compile-sweep",action="store_true",help="Report gates and export QASM for ONE sweep")
    parser.add_argument("--plot",action="store_true",help="Requires matplotlib and pylatexenc")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"results"/"bath")
    args = parser.parse_args()
    if args.sweeps < 1:
        parser.error("--sweeps must be positive")
    model = image_model([[0,0],[255,255]],kappa=2)
    bath = build_bath(model.hamiltonian,args.beta,dt=args.dt,global_flip=args.global_flip)
    target = reference_gibbs(bath)
    d = len(target)
    channel = channel_matrix(bath)
    generator = davies_generator(bath)
    stationary_residual = trace_distance((channel@target.ravel(order="F")).reshape(d,d,order="F"),target)
    contact_residual = max(trace_distance(apply_contact(c,target),target) for c in bath.contacts)
    generator_residual = float(np.linalg.norm(generator@target.ravel(order="F")))
    if max(stationary_residual,contact_residual,generator_residual) > 1e-9:
        raise RuntimeError("Thermal fixed-point checks failed.")
    eigenvalues = np.linalg.eigvals(channel)
    moduli = np.sort(np.abs(eigenvalues))
    gap = float(1-moduli[-2])
    labels = ["polarized |0000>","maximally mixed","coherent |++++>"]
    initials = [np.diag([1.]+[0.]*(d-1)),np.eye(d)/d,np.ones((d,d))/d]
    vectors = np.column_stack([rho.ravel(order="F") for rho in initials]).astype(complex)
    errors,gamma_errors = [],[]
    gamma_target = seed_correlations(target,0)
    for sweep in range(args.sweeps+1):
        states = [vector.reshape(d,d,order="F") for vector in vectors.T]
        errors.append([trace_distance(rho,target) for rho in states])
        gamma_errors.append([float(np.max(np.abs(seed_correlations(rho,0)-gamma_target))) for rho in states])
        if sweep < args.sweeps:
            vectors = channel@vectors
    args.output.mkdir(parents=True,exist_ok=True)
    # One actual nonzero-energy contact, including preparation, unread measurement, reset.
    example_index = next(i for i,c in enumerate(bath.contacts) if c.omega > 0)
    example = contact_circuit(bath.contacts[example_index],bath.n)
    example_compiled = transpile(example,basis_gates=["u3","cx"],optimization_level=1,seed_transpiler=1200)
    (args.output/"one-contact.qasm").write_text(qasm2.dumps(example_compiled))
    one_sweep = sweep_circuit(bath)
    with (args.output/"one-sweep.qpy").open("wb") as handle:
        qpy.dump(one_sweep,handle)
    resources = None
    if args.compile_sweep:
        compiled = transpile(one_sweep,basis_gates=["u3","cx"],optimization_level=1,seed_transpiler=1200)
        resources = {"operations":dict(compiled.count_ops()),"depth":compiled.depth(),
                     "scope":"ONE sweep, all-to-all, no noise or device routing"}
        (args.output/"one-sweep.qasm").write_text(qasm2.dumps(compiled))
    unitaries = np.stack([c.unitary for c in bath.contacts],axis=2)
    lowering = np.stack([c.lowering for c in bath.contacts],axis=2)
    savemat(args.output/"bath-fixture.mat",{
        "image":model.image,"kappa":model.kappa,"sigma":model.sigma,
        "sigma_ddof":model.sigma_ddof,"H":model.hamiltonian,
        "n_pixels":bath.n,"beta":bath.beta,"beta_effective":bath.beta_effective,
        "dt":bath.dt,"sweeps":args.sweeps,"omega":[c.omega for c in bath.contacts],
        "p_excited":[c.excited_probability for c in bath.contacts],
        "lowering":lowering,"unitaries_bath_first":unitaries,
        "rho_initial":initials[0],"rho_final":states[0],"rho_gibbs":target,
        "gamma_gibbs":gamma_target,"gamma_final":seed_correlations(states[0],0),
        "channel_column_major":channel,"davies_generator_column_major":generator,
    },do_compression=True)
    report = {
        "method":"frequency-resolved, energy-conserving thermal collisions",
        "reference":"arXiv:2410.01206v2 Section 3; finite-contact dilation is our adaptation",
        "n_pixels":bath.n,"system_plus_bath_qubits":bath.n+1,"beta_paper":bath.beta,
        "beta_effective":bath.beta_effective,"dt":bath.dt,"sweeps":args.sweeps,
        "contacts_per_sweep":len(bath.contacts),"global_flip":args.global_flip,
        "energy_tolerance":bath.energy_tolerance,"operator_tolerance":bath.operator_tolerance,
        "max_single_contact_stationarity_residual":contact_residual,
        "sweep_stationarity_residual":stationary_residual,"davies_stationarity_residual":generator_residual,
        "sweep_spectral_gap_1_minus_subleading_modulus":gap,
        "initial_states":labels,"trace_distance_by_sweep":errors,
        "max_correlation_error_by_sweep":gamma_errors,
        "gamma_gibbs":gamma_target.tolist(),"gamma_final":seed_correlations(states[0],0).tolist(),
        "same_segment_final":(seed_correlations(states[0],0)>0).reshape(2,2).tolist(),
        "one_contact_resources":dict(example_compiled.count_ops()),
        "one_sweep_resources":resources,
        "limitations":["Dense spectral preprocessing is exponential.",
                       "The toric-code polynomial mixing theorem is not a theorem for image XXZ.",
                       "These are ensemble density-matrix results, not hardware results or segmentation shot estimates.",
                       "dt is the small-step Davies time parameter, not hardware wall-clock time."],
        "versions":{name:version(name) for name in ["numpy","scipy","qiskit"]},
    }
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if args.plot:
        import os
        os.environ.setdefault("MPLCONFIGDIR",str(args.output/".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(8,4.8),layout="constrained")
        for i,label in enumerate(labels):
            ax.semilogy(range(args.sweeps+1),np.asarray(errors)[:,i],label=label,lw=2)
        ax.set(xlabel=f"Bath sweeps ({len(bath.contacts)} contacts per sweep)",
               ylabel="Trace distance to exact Gibbs state",
               title="Four-pixel XXZ model: convergence under a thermal bath")
        ax.legend(); ax.grid(alpha=.2)
        fig.savefig(args.output/"convergence.png",dpi=180)
        fig.savefig(args.output/"convergence.svg")
        plt.close(fig)
        fig = example.draw(output="mpl",fold=-1,style={"name":"bw"})
        fig.savefig(args.output/"one-contact.png",dpi=160,bbox_inches="tight")
        plt.close(fig)
    print(f"{bath.n} system qubits + 1 reusable thermal bath qubit")
    print(f"{len(bath.contacts)} contacts/sweep; {args.sweeps} sweeps")
    print(f"Gibbs stationarity residual: {stationary_residual:.3e}")
    for label,error in zip(labels,errors[-1]):
        print(f"Final trace distance from {label}: {error:.3e}")
    print("Final seed correlations:",seed_correlations(states[0],0))
    print("One-sweep compiled resources:",resources)
    print("Outputs:",args.output.resolve())


if __name__ == "__main__":
    main()
