"""Plot observable robustness during bath sweeps and under sustained noise.

The Gibbs state is a benchmark, not a preparation success criterion. This
four-spin experiment distinguishes correlation values, label margins,
connected correlations, and full-state distances. No long-range or
universality conclusion follows from a single 2x2 image.
"""
import argparse
import csv
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.io import savemat

from algorithm1 import image_model
from correlation_robustness import (correlation_metrics, ensemble_linf_diameter,
                                    initial_ensemble, pauli_noise, permanent_crossing,
                                    spin_correlations)
from thermal_bath import build_bath, channel_matrix, reference_gibbs, trace_distance


def evolve(channel, states):
    return np.stack([(channel@rho.ravel(order="F")).reshape(rho.shape,order="F")
                     for rho in states])


def run_trajectories(channel, initials, reference, sweeps):
    states = initials.copy()
    history = []
    initial_gamma = np.array([spin_correlations(rho)[0] for rho in states])
    previous = None
    for step in range(sweeps+1):
        metrics = correlation_metrics(states,reference,previous=previous,initial=initial_gamma)
        history.append(metrics); previous = metrics["gamma"]
        if step < sweeps:
            states = evolve(channel,states)
    return {key:np.asarray([item[key] for item in history]) for key in history[0]}, states


def driven_test(channel, reference, h, n, sweeps):
    """Observe after noise, every sweep; unlike ordinary thermalization this
    channel generally does not fix the original Gibbs state. Start from I/d.
    """
    rows = []; trajectories = []
    for axis in ["Z","X"]:
        for probability in [0.,.01,.02,.05,.1,.25,.5]:
            state = np.eye(len(reference))/len(reference)
            path = []
            for step in range(sweeps+1):
                stats = correlation_metrics([state],reference)
                path.append(np.r_[stats["gamma"][0],stats["linf_reference"][0],
                                  stats["trace_distance"][0],stats["signed_margin"][0]])
                if step < sweeps:
                    state = pauli_noise(evolve(channel,[state])[0],n,axis,probability)
            next_state = pauli_noise(evolve(channel,[state])[0],n,axis,probability)
            residual = trace_distance(next_state,state)
            if residual > 1e-9:
                raise RuntimeError("Driven channel has not stabilized; increase --noise-sweeps.")
            raw, connected, means = spin_correlations(state)
            rows.append({"axis":axis,"p":probability,"gamma":raw.tolist(),
                         "connected":connected.tolist(),"magnetization":means.tolist(),
                         "linf_reference":float(stats["linf_reference"][0]),
                         "trace_distance":float(stats["trace_distance"][0]),
                         "signed_margin":float(stats["signed_margin"][0]),
                         "fixed_point_residual":residual,
                         "normalized_commutator_fro":float(np.linalg.norm(h@state-state@h)/np.linalg.norm(h)),
                         "energy_per_spin":float(np.trace(h@state).real/n)})
            trajectories.append(path)
    return rows,np.asarray(trajectories)


def save_figures(out, history, labels, families, gamma_ref, driven, thresholds):
    os.environ.setdefault("MPLCONFIGDIR",str(out/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    steps = np.arange(len(history["gamma"]))
    colors = ["#1b7494","#da8422","#8056a3","#429663"]
    named = [i for i,family in enumerate(families) if family == "named"]
    def finish(fig,name):
        fig.savefig(out/f"{name}.png",dpi=170)
        fig.savefig(out/f"{name}.svg")
        plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(11,7),layout="constrained",sharex=True,sharey=True)
    for ax,index in zip(axes.flat,named):
        for p,color in zip(range(1,4),colors):
            label = f"pixel {p} (distance {1 if p < 3 else 2})"
            ax.plot(steps,history["gamma"][:,index,p],color=color,label=label,lw=2)
            ax.axhline(gamma_ref[p],color=color,linestyle="--",alpha=.65,lw=1)
        ax.axhline(0,color="black",lw=.6); ax.set_title(labels[index])
        ax.set(xlabel="Bath sweep",ylabel=r"$\Gamma_{0j}=\langle S_0^z S_j^z\rangle$",ylim=(-.265,.265))
        ax.grid(alpha=.15)
    axes[0,0].legend(loc="upper right",fontsize=9)
    fig.suptitle("Seed correlations during sweeps | dashed = Gibbs comparison\n"
                 "2x2 image [[0,0],[255,255]]; kappa=2, beta/N=6; self pair omitted",fontsize=13)
    finish(fig,"correlations-vs-sweeps")

    fig,axes = plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
    panels = [("linf_reference",r"$\|\Gamma(s)-\Gamma_{\mathrm{Gibbs}}\|_\infty$"),
              ("linf_step",r"$\|\Gamma(s)-\Gamma(s-1)\|_\infty$"),
              ("trace_distance",r"$D(\rho_s,\rho_{\mathrm{Gibbs}})$")]
    for ax,(metric,title) in zip(axes,panels):
        for index,color in zip(named,colors):
            ax.semilogy(steps,history[metric][:,index],color=color,label=labels[index],lw=2)
        ax.set(title=title,xlabel="Bath sweep",ylim=(1e-12,1.3)); ax.grid(alpha=.2)
    axes[0].axhline(thresholds["correlation_tolerance"],color="gray",ls=":",label="absolute tolerance 0.01")
    axes[2].axhline(thresholds["state_far_threshold"],color="gray",ls=":",label="state distance 0.1")
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.suptitle("Correlation accuracy, change per sweep, and full-state distance are different tests",fontsize=13)
    finish(fig,"correlation-distances")

    fig,axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for family,color in zip(["random_product","haar_pure"],colors):
        idx = np.flatnonzero(np.asarray(families) == family)
        label = f"{family.replace('_',' ')} (n={len(idx)})"
        for ax,metric in [(axes[0,0],"linf_reference"),(axes[1,0],"trace_distance")]:
            q = np.quantile(history[metric][:,idx],[.1,.5,.9],axis=1)
            ax.semilogy(steps,q[1],color=color,label=label,lw=2)
            ax.fill_between(steps,q[0],q[2],color=color,alpha=.18)
        diameters = [ensemble_linf_diameter(value[idx]) for value in history["gamma"]]
        axes[0,1].semilogy(steps,diameters,color=color,label=label,lw=2)
        success = history["signed_margin"][:,idx] >= thresholds["label_margin"]
        axes[1,1].plot(steps,success.mean(axis=1),color=color,label=label,lw=2)
    axes[0,0].set_title("Correlation distance to Gibbs comparison")
    axes[0,1].set_title("Maximum distance between runs (no Gibbs reference)")
    axes[1,0].set_title("Full-state trace distance to Gibbs comparison")
    axes[1,1].set(title="Fraction matching all reference labels with margin >= 0.02",ylim=(-.03,1.03))
    for ax in axes.flat:
        ax.set_xlabel("Bath sweep"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Sampled initial-state robustness | bands: 10th-90th percentiles, not confidence intervals",fontsize=12)
    finish(fig,"initial-state-ensemble")

    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for index,color in zip(named,colors):
        axes[0].plot(steps,history["gamma"][:,index,3],color=color,label=labels[index],lw=2)
        axes[1].plot(steps,history["connected"][:,index,3],color=color,label=labels[index],lw=2)
    axes[0].set_title(r"Raw $\langle S_0^z S_3^z\rangle$")
    axes[1].set_title(r"Connected $\langle S_0^z S_3^z\rangle-\langle S_0^z\rangle\langle S_3^z\rangle$")
    for ax in axes:
        ax.set(xlabel="Bath sweep",ylim=(-.265,.265)); ax.axhline(0,color="gray",lw=.6)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Farthest pair (0,3): only two lattice edges apart; this is not a long-range test",fontsize=12)
    finish(fig,"connected-correlations")

    fig,axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    fig.get_layout_engine().set(w_pad=.12,wspace=.05)
    for axis,color in zip(["Z","X"],colors):
        data = [row for row in driven if row["axis"] == axis]
        p = [row["p"] for row in data]
        label = "Z dephasing" if axis == "Z" else "X bit flips"
        for ax,key,title in [(axes[0,0],"trace_distance","Full-state distance to original Gibbs state"),
                             (axes[0,1],"linf_reference","Correlation l-infinity distance"),
                             (axes[1,0],"signed_margin","Minimum signed seed correlation")]:
            ax.plot(p,[row[key] for row in data],"o-",color=color,label=label,lw=2)
            ax.set_title(title)
        axes[1,1].plot(p,[row["connected"][3] for row in data],"o-",color=color,label=label,lw=2)
    axes[1,1].set_title("Farthest connected correlation (0,3)")
    axes[1,0].axhline(thresholds["label_margin"],color="gray",ls=":",label="margin 0.02")
    for ax in axes.flat:
        ax.set_xlabel("Local error probability per spin after every sweep")
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("Sustained perturbations: observe after the noise, once the driven channel stabilizes\n"
                 "These are different channels; their stationary states need not be Gibbs",fontsize=12)
    finish(fig,"driven-robustness")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps",type=int,default=60)
    parser.add_argument("--samples-per-family",type=int,default=32)
    parser.add_argument("--noise-sweeps",type=int,default=200)
    parser.add_argument("--rng-seed",type=int,default=1200)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"results"/"correlations")
    args = parser.parse_args()
    if min(args.sweeps,args.samples_per_family,args.noise_sweeps) < 1:
        parser.error("Sweep and sample counts must be positive.")
    args.output.mkdir(parents=True,exist_ok=True)
    model = image_model([[0,0],[255,255]],kappa=2)
    bath = build_bath(model.hamiltonian,24,dt=.1)
    channel = channel_matrix(bath); reference = reference_gibbs(bath)
    initials,labels,families = initial_ensemble(model.n,args.samples_per_family,args.rng_seed)
    history,finals = run_trajectories(channel,initials,reference,args.sweeps)
    thresholds = {"correlation_tolerance":.01,"label_margin":.02,"state_far_threshold":.1}
    # General observable bound: max |Delta <Sz Sz>| <= D(rho,sigma)/2.
    if np.any(history["linf_reference"] > .5*history["trace_distance"]+1e-11):
        raise RuntimeError("Correlation/state distance consistency check failed.")
    if np.max(np.abs(np.trace(finals,axis1=1,axis2=2)-1)) > 1e-10 or np.linalg.eigvalsh(finals).min() < -1e-10:
        raise RuntimeError("The evolved states failed density-matrix checks.")
    summaries = []
    for index,label in enumerate(labels):
        entry = {"label":label,"family":families[index]}
        for name,mask in [("correlations_within_0.01",history["linf_reference"][:,index] <= .01),
                          ("labels_with_margin_0.02",history["signed_margin"][:,index] >= .02),
                          ("state_within_0.01",history["trace_distance"][:,index] <= .01)]:
            step = permanent_crossing(mask)
            entry[name] = {"first_persistent_sweep":step,
                           "trace_distance_at_crossing":None if step is None else float(history["trace_distance"][step,index]),
                           "correlation_linf_at_crossing":None if step is None else float(history["linf_reference"][step,index])}
        summaries.append(entry)
    driven,driven_paths = driven_test(channel,reference,model.hamiltonian,model.n,args.noise_sweeps)
    dephased = pauli_noise(reference,model.n,"Z",.5)
    report = {"image":model.image.tolist(),"n":model.n,"seed_pixel":0,
              "pixel_manhattan_distances":[0,1,1,2],"kappa":model.kappa,
              "beta":bath.beta,"beta_effective":bath.beta_effective,"dt":bath.dt,
              "contacts_per_sweep":len(bath.contacts),"sweeps":args.sweeps,
              "samples_per_family":args.samples_per_family,"rng_seed":args.rng_seed,
              "thresholds":thresholds,"gamma_reference":spin_correlations(reference)[0].tolist(),
              "initial_state_summaries":summaries,"noise_sweeps":args.noise_sweeps,
              "driven_stationary_results":driven,
              "exact_dephasing_control":{
                  "trace_distance":trace_distance(dephased,reference),
                  "correlation_linf":float(np.max(abs(spin_correlations(dephased)[0]-spin_correlations(reference)[0]))),
                  "commutator_fro":float(np.linalg.norm(model.hamiltonian@dephased-dephased@model.hamiltonian)),
                  "interpretation":"All diagonal Z observables are preserved by construction. This control is not evidence of universality."},
              "limitations":["One four-spin image cannot establish long-range order or universality.",
                             "Bath trajectories approach the same known fixed point; this alone is not universality.",
                             "Label signs, correlation magnitudes, connected correlations and quantum entanglement are distinct.",
                             "Driven states are sampled after each noise step; no claim of persistence during the full cycle.",
                             "Reference-state distance alone does not rule out Gibbs at another temperature; nonzero [H,rho] does.",
                             "Initial-state samples are specified ensembles, not all states or a statistical proof.",
                             "All results are ideal ensemble calculations; no finite-shot or hardware noise uncertainty."],
              "versions":{name:version(name) for name in ["numpy","scipy","qiskit","matplotlib"]}}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(args.output/"trajectories.npz",**history,labels=np.array(labels),
                        families=np.array(families),initial_states=initials,final_states=finals,
                        reference_state=reference,driven_paths=driven_paths)
    savemat(args.output/"correlations.mat",{**history,"labels":np.array(labels,dtype=object),
            "families":np.array(families,dtype=object),"H":model.hamiltonian,"rho_reference":reference,
            "rho_initial_runs_first":initials,"rho_final_runs_first":finals,
            "driven_paths":driven_paths,"gamma_reference":spin_correlations(reference)[0],
            "sweep":np.arange(args.sweeps+1),"beta":bath.beta,"beta_effective":bath.beta_effective,
            "dt":bath.dt,"image":model.image,"kappa":model.kappa,"sigma":model.sigma})
    with (args.output/"correlations-vs-sweeps.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        keys=["linf_reference","linf_step","linf_initial","trace_distance","signed_margin"]
        writer.writerow(["sweep","run","family","gamma_01","gamma_02","gamma_03",
                         "connected_01","connected_02","connected_03",*keys])
        for step in range(args.sweeps+1):
            for index,label in enumerate(labels):
                writer.writerow([step,label,families[index],*history["gamma"][step,index,1:],
                                 *history["connected"][step,index,1:],
                                 *[history[key][step,index] for key in keys]])
    save_figures(args.output,history,labels,families,spin_correlations(reference)[0],driven,thresholds)
    print(f"{len(initials)} initial states, {args.sweeps} sweeps; {len(driven)} sustained-noise settings")
    for row in summaries[:4]:
        print(json.dumps(row))
    print("Complete dephasing control:",report["exact_dephasing_control"])
    for row in driven:
        if row["p"] in [.02,.5]:
            print("Driven:",json.dumps(row))
    print("Outputs:",args.output.resolve())


if __name__ == "__main__":
    main()
