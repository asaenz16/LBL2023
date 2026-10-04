"""Run a four-pixel example and export gates/results for MATLAB/QCLAB."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from qiskit import qasm2, transpile
from scipy.io import savemat
from scipy.linalg import expm

from algorithm1 import (compile_circuit, estimate_correlations, gibbs_circuit,
                        image_model, measurement_circuit, sample_counts,
                        seed_correlations, system_density)


def export_mat(path, model, prepared, compiled, rho_reference, rho_circuit,
               correlations, counts, estimate, beta, seed, rng_seed):
    """Dense arrays use MATLAB kron order; exported gate qubits use Qiskit order."""
    gates, angles = [], []
    for instruction in compiled.data:
        indices = [compiled.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "u":
            gates.append([1, indices[0], -1])
            angles.append([float(x) for x in instruction.operation.params])
        elif instruction.operation.name == "cx":
            gates.append([2, *indices])
            angles.append([0, 0, 0])
        else:
            raise ValueError(f"Unsupported export gate: {instruction.operation.name}")
    savemat(path, {
        "image": model.image, "kappa": model.kappa, "beta": beta,
        "beta_effective": prepared.beta_effective, "n_pixels": model.n,
        "sigma": model.sigma, "sigma_ddof": model.sigma_ddof,
        "seed_zero_based": seed, "rng_seed": rng_seed, "shots": estimate.shots,
        "edges_zero_based": model.edges, "distances": model.distances,
        "delta": model.delta, "H": model.hamiltonian,
        "rho_reference": rho_reference, "rho_circuit": rho_circuit,
        "gamma_reference": correlations, "gamma_shots": estimate.gamma,
        "ci_lower": estimate.lower, "ci_upper": estimate.upper,
        "same_segment_shots": estimate.same_segment,
        "count_basis_indices": [int(bits, 2) for bits in counts],
        "count_values": list(counts.values()),
        "gates": np.asarray(gates, dtype=int).reshape(-1, 3),
        "u_angles": np.asarray(angles).reshape(-1, 3),
        "global_phase": float(compiled.global_phase),
        "basis_order": "pixel 0 is MSB; row-major pixels; Z|0>=+|0>",
        "gate_encoding": "1=U(theta,phi,lambda),2=CX; Qiskit 0-based qubits",
    }, do_compression=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-json", type=Path, help="JSON grayscale or RGB array; <=6 pixels")
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=24.0, help="Paper beta; divided by N internally")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--sigma-ddof", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=0, help="Zero-based row-major seed pixel")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--rng-seed", type=int, default=1200)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "demo")
    args = parser.parse_args()
    image = (json.loads(args.image_json.read_text()) if args.image_json
             else [[0, 0], [255, 255]])
    model = image_model(image, args.kappa, sigma=args.sigma, sigma_ddof=args.sigma_ddof)
    prepared = gibbs_circuit(model.hamiltonian, args.beta)
    # Independent reference: matrix exponential, not the circuit's spectral weights.
    shifted_h = model.hamiltonian - np.linalg.eigvalsh(model.hamiltonian)[0] * np.eye(2**model.n)
    rho_ref = expm(-args.beta / model.n * shifted_h)
    rho_ref /= np.trace(rho_ref)
    exact = seed_correlations(rho_ref, args.seed)
    compiled = compile_circuit(prepared)
    rho = system_density(compiled, model.n)
    trace_distance = float(np.linalg.svd(rho - rho_ref, compute_uv=False).sum() / 2)
    if trace_distance > 1e-9:
        raise RuntimeError(f"Circuit verification failed: trace distance {trace_distance}")
    counts = sample_counts(compiled, model.n, shots=args.shots, rng_seed=args.rng_seed)
    estimate = estimate_correlations(counts, model.n, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    export_mat(args.output / "fixture.mat", model, prepared, compiled, rho_ref, rho,
               exact, counts, estimate, args.beta, args.seed, args.rng_seed)
    # OpenQASM 2 qelib1 defines u3; Qiskit's lowercase 'u' needs an extension.
    qasm_circuit = transpile(measurement_circuit(compiled, model.n),
                            basis_gates=["u3", "cx"], optimization_level=0)
    (args.output / "circuit.qasm").write_text(qasm2.dumps(qasm_circuit))
    (args.output / "circuit.txt").write_text(str(prepared.circuit.draw(output="text", fold=-1)))
    report = {
        "method": "exact spectral purification; exponential reference, ideal simulator",
        "image": model.image.tolist(), "n_pixels": model.n, "qubits": 2 * model.n,
        "kappa": model.kappa, "beta_paper": args.beta, "beta_effective": prepared.beta_effective,
        "sigma": model.sigma, "sigma_ddof": model.sigma_ddof,
        "sigma_source": "explicit" if args.sigma is not None else "unique edge distances",
        "edges_zero_based": model.edges.tolist(), "delta": model.delta.tolist(),
        "seed_pixel_zero_based": args.seed, "seed_simulator": args.rng_seed,
        "shots": estimate.shots, "gate_counts": dict(compiled.count_ops()),
        "depth": compiled.depth(), "routing": "all-to-all; no hardware routing or noise",
        "trace_distance_to_expm": trace_distance,
        "gamma_exact": exact.tolist(), "gamma_shots": estimate.gamma.tolist(),
        "ci_lower": estimate.lower.tolist(), "ci_upper": estimate.upper.tolist(),
        "ci_description": "simultaneous >=95% Hoeffding bounds; ideal independent shots",
        "sign_resolved": estimate.sign_resolved.tolist(),
        "same_segment_exact": (exact > 0).reshape(model.image.shape[:2]).tolist(),
        "same_segment_shots": estimate.same_segment.reshape(model.image.shape[:2]).tolist(),
        "counts": counts,
        "versions": {name: version(name) for name in ["numpy", "scipy", "qiskit"]},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{model.n} pixels, {2*model.n} qubits, {args.shots} shots")
    print(f"Compiled gates: {dict(compiled.count_ops())}; depth {compiled.depth()}")
    print(f"Trace distance from expm reference: {trace_distance:.3e}")
    print("Pixel  Exact Gamma    Shot Gamma     95% simultaneous interval   Sign resolved")
    for p in range(model.n):
        print(f"{p:5d}  {exact[p]: .8f}    {estimate.gamma[p]: .8f}   "
              f"[{estimate.lower[p]: .8f}, {estimate.upper[p]: .8f}]  {estimate.sign_resolved[p]}")
    print("Same segment as seed (Algorithm 1, shot estimates):")
    print(estimate.same_segment.reshape(model.image.shape[:2]).astype(int))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
