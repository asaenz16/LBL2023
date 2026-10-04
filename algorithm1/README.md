# Algorithm 1: quantum circuit reference

This folder implements the **Quantum Segmentation Algorithm (Algorithm 1)**
in the supplied `Quantum_paper/MainV2.tex`, in Python/Qiskit and MATLAB/QCLAB.
It is an additive prototype; the existing MATLAB scripts are unchanged.

The first objective is a correct, measurable circuit for small instances. It
uses **exact spectral purification**, including classical diagonalization of
the image Hamiltonian. This has exponential cost and does **not** demonstrate
efficient Gibbs preparation, a quantum speedup, or the paper's proposed bath
implementation. It supplies a reference against which those methods can be
tested. The prototype deliberately rejects images larger than six pixels.

## Circuit and paper conventions

For each unique undirected horizontal/vertical nearest-neighbor edge, use

\[
\Delta_{pq}=2-\kappa\,\|c_p-c_q\|_2/\sigma,\qquad
H=-\frac14\sum_{\{p,q\}}(X_pX_q+Y_pY_q+\Delta_{pq}Z_pZ_q).
\]

There are open boundaries. The paper's ordered neighbor sum counts each edge
twice and has an extra factor of one half. Its spin operators are Pauli
matrices divided by two. Grayscale distances are absolute differences;
RGB distances are Euclidean distances. No color rescaling is applied.

`sigma` is the standard deviation of **neighbor distances**, not of pixel
intensities. The default is the sample standard deviation (`ddof=1`) over
unique undirected edges. The manuscript does not fix sample versus population
standard deviation, or whether to duplicate the edges when computing it.
This choice must be checked when the missing generating scripts arrive.
Set `sigma_ddof=0` / `'SigmaDDOF',0` for population standard deviation, or
provide a positive `sigma` / `'Sigma',value` to specify the normalization
directly. Zero or undefined variance requires an explicit positive value;
the implementation does not silently choose a model for it.

Let `N` be the number of pixels and let `beta` be the paper's input parameter.
With eigenvectors in the **columns** of `U`,

\[
H=U\operatorname{diag}(E)U^\dagger,\qquad
p_k=\frac{\exp[-(\beta/N)(E_k-E_{\min})]}
{\sum_j\exp[-(\beta/N)(E_j-E_{\min})]}.
\]

The circuit uses `N` system qubits and `N` ancillary qubits, initially zero:

1. Prepare the ancilla in `sum_k sqrt(p_k)|k>`.
2. Apply `N` CNOT gates, copying the ancilla basis index to the system.
3. Apply the eigenvector unitary `U` to the system.
4. Measure the system in the computational (`Z`) basis; ignore the ancilla.

Before measurement the joint state is

\[
|\Psi\rangle=\sum_k\sqrt{p_k}|E_k\rangle_S|k\rangle_A,
\quad \operatorname{Tr}_A|\Psi\rangle\langle\Psi|
=\frac{e^{-(\beta/N)H}}{\operatorname{Tr}e^{-(\beta/N)H}}.
\]

Qiskit synthesizes the preparation and unitary into single-qubit `U` and
two-qubit CNOT gates. The native MATLAB circuit represents preparation with
a Householder matrix block and `U` with a QCLAB `MatrixGate`. The exchange
checker also recreates the **decomposed Qiskit circuit** using QCLAB `Phase`,
`RotationY`, and `CNOT` gates. A matrix block must not be counted as one
elementary gate in a resource claim.

For each measured bitstring `b`, set `z[p]=(-1)^b[p]`. Estimate
`Gamma(seed,p) = mean(z[seed]*z[p])/4`. All pairs commute and share the same
shots; the circuit is re-prepared for every shot. The self-correlation is
exactly `1/4`. A pixel receives the seed's label iff its estimate is **strictly
positive**, including the paper's zero-to-opposite tie rule.

The reports also flag unresolved signs using simultaneous 95% Hoeffding
bounds. For `M` ideal independent shots the conservative radius is
`sqrt(log(2*N/0.05)/(8*M))`, clipped to `[-1/4,1/4]`. These bounds account for
sampling uncertainty, not state-preparation error or hardware noise. Near
zero correlation, the paper's hard label can be unstable. The sample size
sufficient for additive error `epsilon` in all correlations with failure
probability at most `alpha` is `ceil(log(2*N/alpha)/(8*epsilon^2))`.

## Ordering shared with MATLAB

| Quantity | Convention |
| --- | --- |
| Pixel order | Row-major: left to right, then top to bottom |
| Python seed and exported edges | Zero-based |
| MATLAB function seed and native edges | One-based |
| Dense `H`, `rho`, eigenvectors | Pixel 0 is the leftmost / most significant tensor factor |
| Spin convention | `Z|0> = +|0>`, following the explicit matrices in the paper |
| Qiskit system qubit for pixel `p` | `N-1-p`, where `p` is zero-based |
| QCLAB system qubit for pixel `p` | `p`, where `p` is zero-based |
| Python measurement count strings | Left to right = pixel 0 through pixel N-1 |

The exported MAT arrays already use MATLAB's `kron` basis order. Do not
bit-reverse these matrices. Gate indices in the MAT file use Qiskit's order;
`verify_python_fixture.m` performs the required gate-index conversion.
Different eigensolvers can choose different bases in a degenerate eigenspace;
compare density matrices and observables, not eigenvector columns or gate lists.
At low temperature a degenerate ground space retains its thermal mixture.

## Run Python

From the repository root, with Python 3.12 (the tested version):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r algorithm1/requirements.txt
.venv/bin/python -m pytest -c algorithm1/pytest.ini -q
.venv/bin/python algorithm1/python/run_demo.py
```

The environment has already been created in this checkout. Dependencies are
isolated from the existing code. `requirements-lock.txt` records the complete
tested environment if exact transitive dependency versions are desired.

The demo image is `[[0,0],[255,255]]`, with `kappa=2`, `beta=24`, seed pixel 0,
8,192 shots, and random seed 1200. Its output is in `algorithm1/results/demo/`
(ignored by Git):

- `report.json`: parameters, distances, couplings, counts, correlations,
  confidence bounds, labels, versions, gate counts, and trace distance.
- `fixture.mat`: the image, Hamiltonian, reference/circuit density matrices,
  correlations, counts, and a portable elementary gate list for MATLAB.
- `circuit.qasm`: OpenQASM 2 circuit, including measurements, using `u3` and
  `cx`. Export/reimport is checked against the reference density matrix.
- `circuit.txt`: readable diagram of the preparation circuit's logical blocks.

For another tiny image, save its numeric array in a JSON file and run:

```sh
.venv/bin/python algorithm1/python/run_demo.py --image-json image.json \
  --beta 24 --kappa 2 --seed 0 --shots 20000 --output algorithm1/results/custom
```

Use `--help` for explicit sigma, variance convention, and random-seed options.

## Run MATLAB / QCLAB

Use MATLAB R2020b or newer and the
[QCLAB toolbox](https://github.com/QuantumComputingLab/qclab). The API was
inspected at commit `b9de8ca2d2e78186dc159a6ac6340646b7d603aa` (v1.2.0).
Install QCLAB outside this repository and add its root to your MATLAB path,
as for the repository's existing `new/simple_circuit.m` and `first_QC.m`.

From the repository root in MATLAB:

```matlab
addpath('/path/to/qclab');
addpath('algorithm1/matlab');
r = algorithm1_qclab([0,0;255,255], 2, 24, 1);
r.circuit.draw();
disp(r.gamma_exact);
disp(r.gamma_shots);
disp(r.same_segment);

% After running the Python demo:
test_algorithm1('algorithm1/results/demo/fixture.mat');
comparison = verify_python_fixture('algorithm1/results/demo/fixture.mat');
```

MATLAB and Python random number generators differ; individual shot counts
need not agree. The tests compare `H`, exact reduced density matrices, and
exact correlations. QCLAB's `MatrixGate` does not export to QASM; use the
decomposed Python export for the portable circuit.

**Validation status:** the Python simulator, physics tests, OpenQASM round
trip, and independent simulation of the MATLAB gate encoding were executed.
MATLAB/QCLAB execution remains unverified because MATLAB is not installed
in this workspace. Run `test_algorithm1` in MATLAB before claiming parity.

## First observed result and next experiments

The four-pixel demo uses eight qubits and, with the pinned Qiskit version,
176 single-qubit `U` gates and 109 CNOTs, at depth 211. These are ideal
all-to-all counts before device routing; measurements are excluded.
The reduced density matrix agrees with the independent SciPy `expm`
reference to about `1.95e-14` in trace distance. The exact seed correlations
are approximately `[0.25, 0.16506253, -0.24444022, -0.17022277]`.
The sampled labels are `[[1,1],[0,0]]`, and every sign is resolved by the
simultaneous bounds. This is a synthetic sanity check, not reproduction of
the manuscript's missing 4x4-image experiments.

The tests also check the two-site matrix/spectrum from `XXZ2_Spectrum.nb`,
an analytic two-site thermal correlation, an independent Kronecker-product
Hamiltonian, zero inverse temperature, degenerate ground states, RGB color
distances, pixel ordering, variance edge cases, and shot uncertainty.

The next research steps are to replace spectral purification with a local
system-bath circuit, measure its convergence to this exact reference, and
track preparation bias and gate cost as temperature and image size change.
Then add shot-budget sweeps, a device noise model and routed gate counts,
before hardware execution. A bath method's mixing-time assumptions need
verification for these image-dependent XXZ Hamiltonians. The cited
[Ding et al. paper](https://arxiv.org/abs/2508.05703) establishes guarantees
for the physically relevant models it analyzes; those guarantees cannot
be assumed for every image Hamiltonian.

Implementation references: IBM's
[state-preparation API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.StatePreparation)
and [bit-ordering conventions](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering).
