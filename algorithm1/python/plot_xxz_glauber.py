"""Plot XXZ heat-bath transients with independent-replica uncertainty."""
import os
import json
import csv
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]/"results"/"xxz-glauber"
COLORS = {"ordered": "#218379", "corrupted": "#c98723", "random": "#8266ab"}
LABELS = {"ordered": "Image-ordered", "corrupted": "10% spin corruption", "random": "Random Z paths"}


def read(path):
    return json.loads(path.read_text())


def bounded_mean_interval(mean, count, alpha=.05):
    """Invert the bounded-variable Chernoff/KL inequality for Gamma in [-1/4,1/4].

    Independent replicas are the samples. Unlike +/-1.96 empirical SE, this
    has nonzero width even when every sampled correlation equals +1/4.
    Apply Bonferroni over pixels for a simultaneous trajectory-profile band.
    """
    q = np.clip(2*np.asarray(mean)+.5, 0, 1)
    threshold = np.log(2/alpha)/np.asarray(count)
    def divergence(p):
        p = np.clip(p, 1e-300, 1-1e-15)
        return (q*np.log(np.maximum(q, 1e-300)/p) +
                (1-q)*np.log(np.maximum(1-q, 1e-300)/(1-p)))
    left, right = np.zeros_like(q), q.copy()
    for _ in range(55):
        mid = (left+right)/2
        outside = divergence(mid) > threshold
        left = np.where(outside, mid, left)
        right = np.where(outside, right, mid)
    lower = (left+right)/2
    left, right = q.copy(), np.ones_like(q)
    for _ in range(55):
        mid = (left+right)/2
        outside = divergence(mid) > threshold
        right = np.where(outside, mid, right)
        left = np.where(outside, left, mid)
    upper = (left+right)/2
    return (lower-.5)/2, (upper-.5)/2


def joined(base, long, kind):
    t = np.asarray(base["times"])
    mean = np.asarray(base["trajectories"][kind]["gamma"])
    se = np.asarray(base["trajectories"][kind]["gamma_se"])
    if long:
        keep = np.asarray(long["times"]) > t[-1]
        t = np.r_[t, np.asarray(long["times"])[keep]]
        mean = np.concatenate([mean, np.asarray(long["trajectories"][kind]["gamma"])[keep]])
        se = np.concatenate([se, np.asarray(long["trajectories"][kind]["gamma_se"])[keep]])
    return t, mean, se


def main():
    out = ROOT/"figures"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    widths = [4, 8, 12, 16]
    bases = {w: read(ROOT/f"L{w}-b6.json") for w in widths}
    longs = {w: read(ROOT/"long-run"/f"L{w}-b6.json") if (ROOT/"long-run"/f"L{w}-b6.json").exists() else None for w in widths}
    refs = {w: (longs[w] or bases[w])["reference"] for w in widths}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), layout="constrained", sharex="col", sharey="row")
    table = []
    exports = []
    for col, w in enumerate(widths):
        ref = refs[w]
        target = np.asarray(ref["exact"]["gamma"] if "exact" in ref else ref["gamma"])
        target_se = np.zeros_like(target) if "exact" in ref else np.asarray(ref["gamma_se"])
        for kind in COLORS:
            t, mean, se = joined(bases[w], longs[w], kind)
            far, far_se = -mean[:, 0, -1], se[:, 0, -1]
            counts = np.where(t <= bases[w]["times"][-1], bases[w]["replicas"],
                              (longs[w] or bases[w])["replicas"])
            far_low, far_high = bounded_mean_interval(far, counts)
            axes[0, col].plot(t, far, "o-", ms=3, lw=1.5, color=COLORS[kind], label=LABELS[kind])
            axes[0, col].fill_between(t, far_low, far_high, color=COLORS[kind], alpha=.14)
            distance = np.max(abs(mean-target), axis=(1, 2))
            low, high = bounded_mean_interval(mean, counts[:, None, None], alpha=.025/target.size)
            radius = np.max(np.maximum(mean-low, high-mean) +
                            norm.ppf(1-.025/(2*target.size))*target_se, axis=(1, 2))
            axes[1, col].plot(t, distance, "o-", ms=3, lw=1.5, color=COLORS[kind])
            axes[1, col].fill_between(t, np.maximum(0, distance-radius), distance+radius, color=COLORS[kind], alpha=.10)
            table.append({"width": w, "spins": 2*w, "initial": kind, "last_sweep": int(t[-1]),
                          "far": float(far[-1]), "far_se": float(far_se[-1]),
                          "far_ci95_low": float(far_low[-1]), "far_ci95_high": float(far_high[-1]),
                          "linf_reference": float(distance[-1]), "linf_radius_approx95": float(radius[-1]),
                          "reference_far": ref["far_aligned"], "reference_far_se": ref["far_se"],
                          "reference_max_rhat": max(ref["split_rhat"].values())})
            exports.append({"width": w, "b": 6., "initial": kind, "sweeps": t,
                            "seeds_row_column": np.array([[0, 0], [0, w//4]]),
                            "gamma": mean.reshape(len(t), 2, 2, w),
                            "gamma_se": se.reshape(len(t), 2, 2, w),
                            "reference_gamma": target.reshape(2, 2, w)})
        axes[0, col].axhline(-target[0, -1], color="black", ls="--", lw=1, label="Exact Gibbs" if w == 4 else "Gibbs estimate")
        axes[0, col].set_title(f"2×{w} image • {2*w} spins")
        axes[0, col].set_ylim(-.035, .278)
        axes[1, col].set_ylim(0, .31)
        for row in range(2):
            axes[row, col].set_xscale("symlog", linthresh=1)
            axes[row, col].set_xlim(0, t[-1]*1.15)
            axes[row, col].grid(alpha=.2)
        axes[1, col].set_xlabel("Monte Carlo sweeps")
    axes[0, 0].set_ylabel(r"Aligned far-corner $\Gamma=\langle Z_s Z_j\rangle/4$")
    axes[1, 0].set_ylabel(r"$\max_s\|\Gamma_s-\Gamma_{s,\mathrm{ref}}\|_\infty$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Full XXZ Hamiltonian | b=6 | heat-bath directed loops\n"
                 "512 independent paths through 800 sweeps; 128 in longer runs. Bands account for bounded-observable sampling error.")
    fig.savefig(out/"sweep-correlations.png", dpi=180)
    fig.savefig(out/"sweep-correlations.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    w = 16
    times, means, errors = joined(bases[w], longs[w], "random")
    truth = 1-2*((np.arange(w)//2) % 2)
    for requested in [0, 20, 200, 800, 12800]:
        if requested not in times:
            continue
        k = int(np.flatnonzero(times == requested)[0])
        profile = means[k, 0, w:]*truth
        axes[0].plot(np.arange(1, w+1), profile, "o-", ms=3, label=f"Sweep {requested}")
    axes[0].plot(np.arange(1, w+1), np.asarray(refs[w]["gamma"])[0, w:]*truth, "k--", lw=1.5, label="Gibbs estimate")
    axes[0].set(title="Correlations build across the 2×16 strip", xlabel="Seed-to-pixel Manhattan distance", ylabel="Sign-aligned correlation")
    axes[0].legend(fontsize=8)
    for kind in COLORS:
        vals, err = [], []
        for width in widths:
            k = bases[width]["times"].index(800)
            vals.append(-np.asarray(bases[width]["trajectories"][kind]["gamma"])[k, 0, -1])
            low, high = bounded_mean_interval(vals[-1], bases[width]["replicas"])
            err.append([vals[-1]-float(low), float(high)-vals[-1]])
        axes[1].errorbar(widths, vals, yerr=np.array(err).T, marker="o", label=LABELS[kind], color=COLORS[kind], capsize=3)
    axes[1].plot(widths, [refs[w]["far_aligned"] for w in widths], "k--", label="Gibbs estimate")
    axes[1].set(title="At the same 800-sweep budget", xlabel="Far-corner separation L", ylabel="Sign-aligned correlation")
    axes[1].legend(fontsize=8)
    for width, color in [(4, "#227f91"), (16, "#a35362")]:
        controls = [read(ROOT/"temperature-control"/f"L{width}-b{b}.json")["reference"] for b in [1, 2]]+[refs[width]]
        axes[2].errorbar([1, 2, 6], [c["far_aligned"] for c in controls],
                         yerr=[1.96*c["far_se"] for c in controls], marker="o", capsize=4,
                         label=f"2×{width} image", color=color)
        if width == 4:
            axes[2].plot([1, 2, 6], [-np.asarray(c["exact"]["gamma"])[0, -1] for c in controls], "kx", ms=8, label="Exact 8-spin Gibbs")
    axes[2].set(title="Temperature limits the retained correlation", xlabel="Effective inverse temperature b", ylabel="Stationary far-corner correlation")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_ylim(-.035, .275)
    fig.suptitle("Full XXZ model: retention, preparation, and temperature are distinct tests", fontsize=13)
    fig.savefig(out/"size-separation-temperature.png", dpi=180)
    fig.savefig(out/"size-separation-temperature.svg")
    plt.close(fig)

    small = bases[4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    target = np.asarray(small["reference"]["exact"]["gamma"])
    for kind in COLORS:
        d = small["trajectories"][kind]
        axes[0].plot(small["times"], np.max(abs(np.asarray(d["gamma"])-target), axis=(1, 2)), "o-", ms=3, color=COLORS[kind], label=LABELS[kind])
        axes[1].plot(small["times"], d["diagonal_output_trace_distance_to_gibbs"], "o-", ms=3, color=COLORS[kind])
    axes[1].axhline(small["reference"]["exact_gibbs_to_dephased_trace_distance"], color="black", ls="--", label="Exact dephased Gibbs state")
    axes[0].set(title="Z-correlation profiles approach Gibbs values", ylabel=r"$\max_s\|\Gamma_s-\Gamma_{s,\beta}\|_\infty$")
    axes[1].set(title="The diagonal output remains a different state", ylabel=r"Trace distance $D(\rho_{\rm diag},\rho_\beta)$")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, small["times"][-1]*1.15)
        ax.set_xlabel("Monte Carlo sweeps")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("2×4 image | b=6 | diagonal states formed from sampled Z configurations\n"
                 "These are separable output mixtures; the sampler is not physical quantum dynamics.", fontsize=12)
    fig.savefig(out/"correlations-vs-state-distance.png", dpi=180)
    fig.savefig(out/"correlations-vs-state-distance.svg")
    plt.close(fig)
    (ROOT/"summary.json").write_text(json.dumps(table, indent=2)+"\n")
    with (ROOT/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    savemat(ROOT/"xxz-glauber.mat", {"runs": np.asarray(exports, dtype=object),
            "note": "Gamma axes: checkpoint, seed, image row, image column; raw signed SzSz. Seeds are row-major zero-based.",
            "sweep_definition": "One diagonal conditional scan and N fixed complete loop attempts; not physical time."}, long_field_names=True)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
