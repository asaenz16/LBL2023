"""XXZ heat-bath loop study: independent transients and separate equilibrium checks."""
import os
for key in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
import argparse
import json
import time
from pathlib import Path
import numpy as np
from xxz_glauber import image_graph, run_chain, exact_gibbs, expansion_cutoff


def initial_path(model, kind, rng):
    truth = np.where(model.image.ravel() == 0, 1, -1)
    if kind == "random":
        return rng.choice([-1, 1], model.n)
    state = truth.copy()
    if kind == "corrupted":
        state *= np.where(rng.random(model.n) < .1, -1, 1)
    elif kind != "ordered":
        raise ValueError(kind)
    return state * rng.choice([-1, 1])


def mean_se(array, axis=0):
    return array.mean(axis=axis), array.std(axis=axis, ddof=1)/np.sqrt(array.shape[axis])


def split_rhat(values):
    # shape: independent chains, successive scalar samples
    half = values.shape[1]//2
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    w = np.mean(np.var(split, axis=1, ddof=1))
    between = half*np.var(split.mean(axis=1), ddof=1)
    if w == 0:
        return 1. if between == 0 else float("inf")
    return float(np.sqrt(((half-1)*w/half+between/half)/w))


def reference_stats(samples, nblocks=16):
    chains, count = samples.shape[:2]
    usable = count - count % nblocks
    batch = samples[:, :usable].reshape(chains, nblocks, usable//nblocks, *samples.shape[2:]).mean(axis=2)
    mean = samples.mean(axis=(0, 1))
    batch_se = batch.reshape(chains*nblocks, *samples.shape[2:]).std(axis=0, ddof=1)/np.sqrt(chains*nblocks)
    chain_se = samples.mean(axis=1).std(axis=0, ddof=1)/np.sqrt(chains)
    return mean, np.maximum(batch_se, chain_se)


def study(width, b, args):
    start = time.time()
    model = image_graph(width, normalization=args.normalization, epsilon=args.epsilon)
    seeds = np.array([0, width//4])
    rng = np.random.default_rng(args.seed+width*1000+int(b*100))
    base_seed = int(rng.integers(1, 2**30))
    capacity, log_bound = expansion_cutoff(model, b)
    if args.capacity_multiplier != 1:
        capacity *= args.capacity_multiplier
    reference_times = args.burn + np.arange(args.reference_samples)*args.thin
    ref_gamma, ref_orders, ref_sectors = [], [], []
    maximum_order = 0
    ref_kinds = ["ordered" if c % 2 == 0 else "random" for c in range(args.reference_chains)]
    for c, kind in enumerate(ref_kinds):
        r = run_chain(model, initial_path(model, kind, rng), b, reference_times,
                      seeds, seed=base_seed+c, capacity=capacity)
        ref_gamma.append(r[0]); ref_orders.append(r[2]); ref_sectors.append(r[3])
        maximum_order = max(maximum_order, r[-1])
    ref_gamma, ref_orders, ref_sectors = map(np.asarray, [ref_gamma, ref_orders, ref_sectors])
    ref_mean, ref_se = reference_stats(ref_gamma)
    ref_energy = model.shift.sum()-ref_orders/b
    e_mean, e_se = reference_stats(ref_energy)
    far = -ref_gamma[:, :, 0, -1]
    rhat = {"far": split_rhat(far), "energy": split_rhat(ref_energy),
            "magnetization_squared": split_rhat((ref_sectors/2)**2)}
    exact = exact_gibbs(model, b, seeds) if model.n <= 10 else None
    reference = {"gamma": ref_mean.tolist(), "gamma_se": ref_se.tolist(),
                 "far_aligned": float(-ref_mean[0, -1]), "far_se": float(ref_se[0, -1]),
                 "energy": float(e_mean), "energy_se": float(e_se), "split_rhat": rhat,
                 "chain_far_means": far.mean(axis=1).tolist(), "initializations": ref_kinds,
                 "burn": args.burn, "thin": args.thin, "samples_per_chain": args.reference_samples}
    if exact:
        reference["exact"] = {k: np.asarray(exact[k]).tolist() for k in
                              ["gamma", "energy", "ground_energy", "magnetization_second_moment"]}
        reference["max_profile_error_vs_exact"] = float(np.max(abs(ref_mean-exact["gamma"])))
        reference["energy_error_vs_exact"] = float(e_mean-exact["energy"])
    print(f"Reference L={width}, b={b:g}: far={-ref_mean[0,-1]:.6f} +/- {1.96*ref_se[0,-1]:.6f}, Rhat={rhat}, seconds={time.time()-start:.1f}", flush=True)
    arrays = {"reference_gamma": ref_gamma, "reference_orders": ref_orders,
              "reference_magnetization_twice": ref_sectors, "times": np.array(args.times)}
    trajectories = {}
    for index, kind in enumerate(args.initials):
        samples, orders, work, sectors, distributions = [], [], [], [], []
        for c in range(args.replicas):
            r = run_chain(model, initial_path(model, kind, rng), b, args.times, seeds,
                          seed=base_seed+10000*(index+1)+c, capacity=capacity,
                          record_distribution=model.n <= 8)
            samples.append(r[0]); orders.append(r[2]); sectors.append(r[3]); work.append(r[4])
            if model.n <= 8:
                distributions.append(r[5])
            maximum_order = max(maximum_order, r[-1])
        samples, orders, work, sectors = map(np.asarray, [samples, orders, work, sectors])
        mean, se = mean_se(samples)
        trajectories[kind] = {"gamma": mean.tolist(), "gamma_se": se.tolist(),
                              "order_mean": orders.mean(axis=0).tolist(),
                              "loop_vertex_steps_mean": work[:, :, 0].mean(axis=0).tolist(),
                              "bounce_fraction": float(work[:, -1, 1].sum()/max(1,work[:, -1, 0].sum()))}
        arrays[kind+"_gamma"] = samples
        arrays[kind+"_orders"] = orders
        arrays[kind+"_magnetization_twice"] = sectors
        arrays[kind+"_work"] = work
        if distributions:
            # Symmetrize the global spin flip exactly; ZZ values do not change.
            probabilities = np.asarray(distributions)
            probabilities = (probabilities + probabilities[:, :, ::-1])/2
            arrays[kind+"_z_probabilities"] = probabilities
            from scipy.linalg import expm
            rho = expm(-b*(exact["hamiltonian"]-exact["ground_energy"]*np.eye(2**model.n)))
            rho /= np.trace(rho)
            trajectories[kind]["diagonal_output_trace_distance_to_gibbs"] = [
                float(np.abs(np.linalg.eigvalsh(np.diag(p)-rho)).sum()/2)
                for p in probabilities.mean(axis=0)]
            trajectories[kind]["z_distribution_tv_to_gibbs"] = [
                float(np.abs(p-exact["z_probabilities"]).sum()/2)
                for p in probabilities.mean(axis=0)]
            reference["exact_gibbs_to_dephased_trace_distance"] = float(
                np.abs(np.linalg.eigvalsh(np.diag(np.diag(rho))-rho)).sum()/2)
        print(f"L={width}, b={b:g}, {kind}: far(final)={-mean[-1,0,-1]:.6f}, seconds={time.time()-start:.1f}", flush=True)
    report = {"width": width, "n_spins": model.n, "b": b, "beta_paper": b*model.n,
              "normalization": args.normalization, "kappa": 2., "sigma": model.sigma,
              "epsilon": args.epsilon, "capacity": capacity, "conservative_log_tail_bound": log_bound,
              "max_order": maximum_order, "replicas": args.replicas, "random_seed": args.seed,
              "loop_attempts_per_sweep": model.n, "times": args.times, "seeds": seeds.tolist(),
              "reference": reference, "trajectories": trajectories,
              "runtime_seconds": time.time()-start,
              "interpretation": "Path-space sampling dynamics, not a real-time quantum channel; full XXZ retained.",
              "connected": "Global spin-flip symmetrization of each path ensemble gives zero one-point means and connected=raw Gamma.",
              "initialization": "Constant imaginary-time Z paths; diagonal operators drawn conditionally. These are not simulations of a pure +x density matrix."}
    stem = args.output/f"L{width}-b{b:g}"
    Path(str(stem)+".json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(str(stem)+".npz", **arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--b", type=float, nargs="+", default=[6.])
    parser.add_argument("--replicas", type=int, default=512)
    parser.add_argument("--reference-chains", type=int, default=8)
    parser.add_argument("--reference-samples", type=int, default=2048)
    parser.add_argument("--burn", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--times", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400, 800])
    parser.add_argument("--initials", nargs="+", choices=["ordered", "random", "corrupted"], default=["ordered", "random", "corrupted"])
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--normalization", choices=["sample", "bulk"], default="sample")
    parser.add_argument("--epsilon", type=float, default=.25)
    parser.add_argument("--capacity-multiplier", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"results"/"xxz-glauber")
    args = parser.parse_args()
    if min(args.replicas, args.reference_chains) < 2 or args.reference_samples < 32 or args.thin < 1 or args.burn < 1 or args.capacity_multiplier < 1:
        parser.error("Require >=2 replicas/chains, >=32 reference samples, and positive burn, thin and capacity multiplier.")
    args.output.mkdir(parents=True, exist_ok=True)
    for b in args.b:
        for width in args.widths:
            study(width, b, args)


if __name__ == "__main__":
    main()
