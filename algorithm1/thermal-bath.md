# Preparing Algorithm 1's Gibbs state with a thermal bath

The target remains `rho = exp(-b H)/Z`, where **b = beta/N**. The image
Hamiltonian, pixel ordering, spin normalization, and segmentation rule are
unchanged. This extension replaces preparation of an entire Gibbs
purification with repeated interactions with a fresh thermal bath qubit.

## What the requested paper provides

[Ding, Landau, Li, Lin and Zhang, arXiv:2410.01206v2](https://arxiv.org/abs/2410.01206v2)
proves a polynomial mixing bound for a particular toric-code Gibbs sampler.
Its local Davies terms are supplemented by global logical jumps that connect
otherwise poorly connected sectors. The commuting stabilizer structure is
essential to that analysis. Section 3 supplies the general Davies/Glauber
construction that we use as a reference. The finite-contact circuit below is
our adaptation, rather than a circuit reproduced from that paper.

Our image model contains noncommuting terms on overlapping XXZ bonds.
Its dimension, temperature normalization, couplings, and symmetry sectors
differ from the toric code. We therefore have no inherited polynomial-time
preparation theorem. A stationary Gibbs state and a useful mixing rate are
separate requirements.

The references most relevant to progressing beyond this prototype are:

| Reference in the requested paper | Relevance to this project |
| --- | --- |
| [17] [Chen, Kastoryano, Brandao and Gilyen, 2303.18224](https://arxiv.org/abs/2303.18224) | Finite-resolution thermal Lindbladians and control of approximation errors. |
| [18] [Chen, Kastoryano and Gilyen, 2311.09207](https://arxiv.org/abs/2311.09207) | Exactly detailed-balanced generators for noncommuting Hamiltonians using filtered operators and a coherent correction. Efficient implementation still has a mixing-time dependence. |
| [23] [Ding, Li and Lin, 2404.05998](https://arxiv.org/abs/2404.05998) | KMS-detailed-balanced generators with finitely many filtered jumps; a promising general construction for our noncommuting model. The introduction explains the energy-resolution problem with exact Davies jumps. |

A later, related paper already cited in the manuscript is
[Ding, Zhan, Preskill and Lin, 2508.05703v3](https://arxiv.org/abs/2508.05703v3).
It is **not** a reference of the May 2025 toric-code paper. Its circuit uses
one resettable thermal qubit, randomly chosen bath frequencies and couplings,
and smoothly modulated interactions. Section VII gives a second-order
Trotter implementation. This is the natural next circuit to test when we
replace the exact frequency resolution used below. Its stated mixing bounds
cover particular model classes and do not establish a bound for general
image-dependent XXZ models.

## Implemented bath contacts

We first choose local coupling proposals `X_j`, `Y_j`, and `Z_j` at every
pixel. Let `A(omega)` lower the system energy by `omega > 0`:

\[
A(\omega)=\sum_{E_h-E_l=\omega}\Pi_l A\Pi_h,
\qquad [H,A(\omega)]=-\omega A(\omega).
\]

The code computes these components by dense diagonalization. Equal energies
and equal Bohr frequencies are grouped, retaining complete degenerate
blocks. Although `A` is local, `A(omega)` generally acts on the whole system.
This spectral preprocessing and the subsequent generic unitary synthesis
are exponential; the new reference is limited to **four system qubits**.

For each component, prepare one bath qubit with gap `omega` at inverse
temperature `b`:

\[
H_B=\omega|1\rangle\langle1|,\qquad
\tau_\omega=\frac{|0\rangle\langle0|+e^{-b\omega}|1\rangle\langle1|}
{1+e^{-b\omega}}.
\]

Use an energy-exchange interaction, written here in system-then-bath order:

\[
V_\omega=A(\omega)\otimes|1\rangle\langle0|
+A(\omega)^\dagger\otimes|0\rangle\langle1|,\quad
W_\omega=e^{-i\sqrt{2\,dt}\,V_\omega}.
\]

One contact implements

\[
\Phi_\omega(\rho)=\operatorname{Tr}_B\!\left[
W_\omega(\rho\otimes\tau_\omega)W_\omega^\dagger\right].
\]

The interaction commutes with `H + H_B`. Consequently it commutes with the
joint Gibbs state, and **every contact preserves the target Gibbs state**,
even at finite contact angle. This is a direct property of our resolved
interaction, not an appeal to the toric-code mixing theorem. A full sweep
visits every nonzero component of every coupling proposal.

For the zero-frequency component `A(0)`, use the maximally mixed bath and
`W_0 = exp(-i sqrt(dt) A(0) tensor X)`. The different factor avoids counting
the zero-frequency dissipator twice. It also permits mixing inside degenerate
energy subspaces. Numerical grouping/pruning tolerances are recorded; the
program checks stationarity rather than assuming finite-precision resolution
is exact.

The quantum circuit for one contact is:

```text
system rho --------------------[ W_omega ]---------------- rho_next
                                     |
bath |0> -- RY(theta) -- M -----[         ]-- reset |0>
                        |
                   result unused
```

Here `theta = 2 asin(sqrt(p1))`, `p1 = 1/(1+exp(b*omega))`. The unread
measurement converts the rotated bath qubit into a **mixed** thermal state.
Keeping both measurement outcomes is essential: there is no postselection.
`RY` alone would create a pure coherent bath state and would not implement
this channel. An equivalent option is to initialize `|0>` or `|1>` using a
classical random draw with the specified probabilities.

The bath reset removes its correlations with the system, allowing the same
qubit to be reused. Thus four pixels require **five quantum wires** plus one
reused classical bit. Gate synthesis may introduce additional ancillas on
other backends; the current generic synthesis uses these five wires.

## Relation to the Davies generator

Writing `D[A](rho)=A rho A^dagger - {A^dagger A,rho}/2`, the small-`dt` limit
of a nonzero-frequency contact is

\[
\Phi_\omega=I+dt\left[2p_0D[A(\omega)]+2p_1D[A(\omega)^\dagger]\right]
+O(dt^2),\qquad p_1/p_0=e^{-b\omega}.
\]

At zero frequency the limit is `I + dt D[A(0)] + O(dt^2)`. Their sum gives
the reference dissipator. The sign convention differs from the source:
its positive frequency labels energy-raising components, while ours is a
positive energy *loss*. This accounts for the apparent reversal of the
Glauber rates. The code acts on density matrices (Schrodinger picture).
The commuting free Hamiltonian evolution is omitted in this dissipative
reference. It can be added without changing Gibbs stationarity.

The finite-contact sweep is not exactly `exp(dt L)`; we test its approach
to the Davies generator as `dt` decreases. `dt` is the effective Davies
time per sweep in that limit, not a hardware duration. Finite-step mixing
must be checked independently; stationary-state preservation alone does not
prove uniqueness or exclude special nonmixing contact angles.

## Tests and observed results

The default experiment uses the original image `[[0,0],[255,255]]`, `kappa=2`,
`beta=24`, and `dt=0.1`. There are 160 contacts per sweep. Starting separately
from `|0000>`, the maximally mixed state, and `|++++>`, the ensemble density
matrix approaches the same Gibbs state. After 60 sweeps, the observed trace
distances were respectively about `4.0e-11`, `2.5e-11`, and `5.3e-11`.
The one-sweep Gibbs stationarity residual was about `3.1e-15`.

The seed correlations converge to the original reference:
`[0.25, 0.16506253, -0.24444022, -0.17022277]`.
These are ideal ensemble results, not hardware measurements or shot-based
segmentation accuracy. The tests include an actual Qiskit shot simulation
of a single thermal contact to check its intermediate measurement and reset.

Other new checks cover trace preservation, joint energy conservation,
upward/downward rates, complete degenerate blocks, high and low temperature,
the Davies small-step limit, and qubit ordering.

Two useful findings for the paper revision:

- **Z-only bath couplings fail.** The image Hamiltonian and these couplings
  conserve total Z magnetization. A fully polarized state cannot escape its
  sector and therefore does not reach the full Gibbs state. The default
  bath includes X and Y couplings to allow sector transitions.
- **Global flips need an observable-specific assessment.** Our XXZ model
  commutes with `F = X tensor ... tensor X`. An optional zero-frequency
  contact using F preserves Gibbs and can change magnetization. However,
  seed ZZ correlations are even under F. The local contact channels are
  F-covariant, so adding this commuting flip channel leaves their ZZ
  correlation evolution unchanged for the same number of sweeps. It does
  not automatically accelerate the segmentation observable. The tests
  compare the two constructions explicitly.

Generic compilation of **one** default sweep produced 111,545 single-qubit
`u3` gates, 67,680 CNOTs, 320 resets, and 160 bath measurements; circuit depth
was 133,689. These all-to-all counts exclude system readout and hardware
routing. Simply repeating this sweep 60 times would use 4,060,800 CNOTs
before any cross-sweep optimization. Thus this is a correctness reference,
not a practical hardware proposal or evidence of quantum advantage.

## Reproduce and inspect

From the repository root, using the existing virtual environment:

```sh
.venv/bin/python -m pytest -c algorithm1/pytest.ini -q
.venv/bin/python algorithm1/python/run_bath_demo.py
# Optional gate counts, portable one-sweep circuit, and figures:
.venv/bin/python -m pip install -r algorithm1/requirements-plot.txt
.venv/bin/python algorithm1/python/run_bath_demo.py --compile-sweep --plot
```

The output directory `algorithm1/results/bath/` is ignored by Git:

- `report.json`: parameters, convergence trajectories, residuals and resources.
- `bath-fixture.mat`: Hamiltonian, bath probabilities, lowering operators,
  contact unitaries, channel, and initial/final/reference states.
- `one-contact.qasm`: a compiled contact with thermalization measurement/reset.
- `one-sweep.qpy`: the complete sweep with exact unitary blocks, loadable in Qiskit.
- `one-sweep.qasm`: compiled single sweep, when `--compile-sweep` is specified.
- `convergence.png`, `convergence.svg`, `one-contact.png`: optional figures.

The exported single sweep is **not** a full 60-sweep preparation circuit.
Use `sweep_circuit(bath, sweeps=60)` to construct repeated preparation.
To estimate Algorithm 1's correlations, append the existing system Z readout
after preparation, reprepare the entire system for each independent shot,
and pass only the system pixel bits to `estimate_correlations` (exclude the
bath scratch bit). Correlations from successive sweeps of one trajectory
are not independent preparation shots.

For MATLAB:

```matlab
addpath('algorithm1/matlab');
r = verify_thermal_bath_fixture('algorithm1/results/bath/bath-fixture.mat');
% With QCLAB on the path, additionally exercise its MatrixGate representation:
r = verify_thermal_bath_fixture('algorithm1/results/bath/bath-fixture.mat', true);
```

The MATLAB code rebuilds the image Hamiltonian, bath probabilities and
exchange unitaries, then replays the partial-trace channels. It reuses the
exported resolved operators. Its QCLAB option converts the bath-first dense
basis to system-first basis. MATLAB execution is still unverified because
there is no MATLAB runtime in this workspace.

## Next implementation decision

For a circuit with simple local interactions, implement the smoothly driven,
resettable-bath protocol of 2508.05703. Its finite-time energy filtering avoids
constructing each exact spectral projector. Benchmark bath-window size,
coupling strength, frequency sampling and Hamiltonian Trotter step against
this reference, separating stationary-state bias from incomplete mixing.

For an exact Gibbs fixed point at the generator level with noncommuting
interactions, implement the KMS-filtered construction in reference [23],
including its coherent correction. Plain smoothed Davies jumps do not in
general preserve the exact Gibbs state. This offers another controlled
comparison, with mixing-time analysis still required for our model.
