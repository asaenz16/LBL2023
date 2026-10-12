# Robustness on alternating-block strips

These experiments extend the four-spin study to 8, 16, 24, and 32 spins.
For the tested finite times, image correlations survive rotations of an
ordered ground state followed by isolated evolution. A separate local-bond
bath, starting from the maximally mixed state, produces correlations that
decrease strongly with separation. This is evidence for robustness of a
particular family of ordered states, not a demonstration of universality.

## Geometry and compatibility

Use open strips of height 2 and width L = 4N, with N = 1, 2, 3, 4. Both
rows are `[0, 0, 255, 255, 0, 0, 255, 255, ...]`. Each pixel is a spin.
The Hamiltonian retains the existing Algorithm 1 conventions:

\[
 H=-\frac14\sum_{\{i,j\}}(X_iX_j+Y_iY_j+\Delta_{ij}Z_iZ_j),
 \qquad \Delta_{ij}=2-2|c_i-c_j|/\sigma.
\]

Each undirected horizontal or vertical edge is included once. The main
series uses the sample standard deviation of the edge contrasts, as in
the existing prototype. Vertical and same-color bonds have Delta = 2.
The cross-color Delta changes from -2.7434 at L=4 to -2.2991 at L=16
because the finite-size contrast distribution changes. A control fixes
sigma = 255 sqrt(2)/3, its infinite-strip value, so that cross-color Delta
is always 2 - 3 sqrt(2). The qualitative results persist in that control.
The manuscript's unresolved sigma convention still needs comparison with
the missing original generating scripts.

For efficient tensor contractions, a single MPS site holds one column's
two spins, in the basis `00, 01, 10, 11` (top spin first). Pure-state sites
have dimension 4; vectorized density sites have dimension 16. Public image
arrays and correlations have shape `(2,L)` and preserve `(row,column)`
indexing. This representation changes neither the physical Hamiltonian
nor the observable convention. The MATLAB export contains these same
arrays; MATLAB execution has not been tested in this workspace.

## Observables

We measure the paper's spin correlation and its connected counterpart:

\[
 \Gamma_{sj}=\langle S_s^z S_j^z\rangle
 =\tfrac14\langle Z_sZ_j\rangle,\qquad
 C_{sj}=\Gamma_{sj}-\langle S_s^z\rangle\langle S_j^z\rangle.
\]

The two seeds are `(0,0)` and `(0,L/4)`, using zero-based indices. The
edge-seed distant pair ends at `(1,L-1)`, at Manhattan distance L. The
interior-seed pair ends at `(1,3L/4)`, at distance L/2+1. Thus the test is
not restricted to corners.

Let s_j be +1 on black pixels and -1 on white pixels. Plots marked
"sign-aligned" show s_s s_j Gamma_sj. A positive value is consistent with
the intended segmentation; it is not the raw correlation. In particular,
the far-corner raw correlation is negative. The minimum sign-aligned
correlation excludes the seed's trivial self-correlation 1/4. We also
record the maximum absolute change in the correlation profile,
`max_j |Gamma_sj - Gamma_GS,sj|`, separately for each seed.

Ground states are projected into the even global spin-flip sector. The
Rx pulses and the Hamiltonian preserve that symmetry, which forces local
z magnetizations to vanish in exact arithmetic. Consequently C and Gamma
agree, up to numerical symmetry error. This does not establish that
entanglement is necessary: a symmetric classical mixture of ordered
configurations can also have zero means and nonzero connected correlations.

## Experiment A: retain existing order under isolated evolution

DMRG supplies an ordered ground reference. Apply a single layer
`product_j exp(-i theta X_j/2)`, then evolve under the full image
Hamiltonian to t=6 with second-order TEBD. Time is in the units fixed by
the displayed Hamiltonian, with hbar=1; it is not a bath sweep count.
The broad run uses theta = 0, 0.15, 0.3, 0.6 radians, step 0.04 and maximum
bond dimension 64. The main nonzero cases, 0.3 and 0.6, are refined with
step 0.02 and maximum bond dimension 128. Profiles are sampled every 0.2.

For theta=0.3 at t=6:

| Strip | Spins | Far-corner aligned Gamma | Interior-pair aligned Gamma |
|---|---:|---:|---:|
| 2 x 4 | 8 | 0.20558 | 0.21015 |
| 2 x 8 | 16 | 0.20244 | 0.19617 |
| 2 x 12 | 24 | 0.20204 | 0.17807 |
| 2 x 16 | 32 | 0.20202 | 0.17426 |

The smallest sign-aligned correlation over both seeds, all sampled times,
and all non-seed pixels is about 0.174 in the 32-spin theta=0.3 run. Its
pure-state trace distance from the ground reference at t=6 is about 0.794,
using sqrt(1-|<GS|psi>|^2). That distance is from the ground state, not
from a finite-temperature Gibbs state. A large global state distance can
coexist with stable local observables and is not by itself universality.
The stronger theta=0.6 pulse reduces the correlations considerably. At
32 spins and t=6, the refined far-corner and interior-pair values are
0.11119 and 0.04227, respectively. Every tested sign remains correct;
the smallest margin over both seeds and sampled times is 0.04227. The
pure-state distance from the ground reference is about 0.992. Detailed
values and diagnostics are in `results/ladders/summary.json`.

Controls use fixed bulk sigma at all four sizes and one reproducible
random-sign Rx pattern at 32 spins (theta=0.3, seed 1216). Both retain the
signs. The random-pattern control has only one realization and the
coarser time step/bond dimension, so it is exploratory rather than a
statistical ensemble test.

## Experiment B: prepare with a local-bond bath and sustained noise

The earlier globally resolved bath requires full diagonalization. The
scalable alternative here constructs an exact Davies dissipator for each
two-spin bond separately, using local X, Y, Z couplings and effective
inverse temperature b=6. Thus beta_paper = b times the number of spins.
For each bond, the exponential channel preserves that bond's Gibbs state.
The overlapping noncommuting bond generators **do not preserve the Gibbs
state of the full image Hamiltonian**. This is a distinct preparation
model, not a controlled approximation to the previous global generator.
There is no additional coherent Hamiltonian evolution in this protocol.

Start from I/2^(2L). One bath sweep uses Strang composition of vertical,
even-horizontal and odd-horizontal dissipative layers with duration 0.1.
Then apply independent local Pauli channels `(1-p)rho + p P rho P` on
every spin. Cases are no added noise, Z noise p=0.02, and X noise p=0.02.
The continuous-noise rate is held fixed when testing duration 0.05.

The density MPS first runs with maximum bond dimension 32, then continues
from that endpoint with dimension 64. Every five sweeps we check both
seed profiles. Three successive changes below 1e-7 are a stopping rule;
some runs exhaust their allotted refinement steps first. The final
profiles change little, but this is **observable stabilization**, not a
full-state mixing guarantee. Solid/dashed lines in `bath-sweeps.png`
identify the two bond dimensions.

Late-sweep far-corner sign-aligned correlations at dimension 64 are:

| Strip | No added noise | Z noise, p=0.02 | X noise, p=0.02 |
|---|---:|---:|---:|
| 2 x 4 | 0.07753 | 0.07753 | 0.05428 |
| 2 x 8 | 0.004000 | 0.004000 | 0.002070 |
| 2 x 12 | 0.0001520 | 0.0001519 | 0.00006328 |
| 2 x 16 | 0.000006108 | 0.000006104 | 0.000001975 |

Z noise barely changes these values; X noise suppresses them further.
Retaining a mathematical sign when the magnitude is tiny does not imply
practical segmentation with a fixed measurement budget. Small signals
require much more precise estimation. The fixed-sigma no-noise control
also falls from about 0.0758 at 8 spins to 5.43e-6 at 32 spins.

Exact eight-spin calibration demonstrates the model difference. One
local-bath cycle moves the full Gibbs state by trace distance 0.284. The
no-added-noise stationary state is distance 0.625 from that Gibbs state;
its far-corner correlation is 0.077527, compared with 0.249501 in the
global Gibbs state. Therefore these bath results cannot be used to
reject robustness for an accurately implemented global thermal bath.

## Numerical checks and limits

The Python suite has 54 passing tests, including independent dense
Hamiltonian construction, row/rung ordering, complex-state correlation
contractions, small-system ground-state and TEBD comparisons, local
channel complete positivity/trace preservation and detailed balance,
and comparison of a complete density-MPS bath sweep with Qiskit Kraus
evolution. An additional eight-spin dense run checks 60 sweeps for all
three bath cases; its final one-cycle full-state residual is below
2.5e-11. The far-corner dimension-64 bath values agree with it within
3.1e-7.

At 32 spins, restarting DMRG from all spins along X reaches the same
energy as the ordered initial guess within 6.1e-10. Ground-state energy
variances in the main series are below 2e-9. The restart is recorded in
`sample/ground-L16-restart.json`; it rules out dependence on that one
initial guess, not all possible variational bias.

The combined TEBD time-step/bond refinement changes theta=0.3
correlations by at most about 0.00182 over both seeds and all sampled
times. For theta=0.6 the maximum change is 0.00259. These are sensitivity
checks, not rigorous error bounds. In the refined 32-spin runs, total
energy drifts are 0.00176 and 0.0168 for the two angles, respectively.
Per-case diagnostics are in `summary.json`. The bath refinement changes full seed profiles
by less than 0.00065; relative errors in the tiny distant signals can
still be appreciable. Halving the eight-spin bath step changes its
far-corner correlation by up to 0.003, so the bath table describes the
specified discrete protocol, not a converged continuous-time limit.

Density-MPS truncation does not guarantee positivity of the global
density operator. All checked adjacent two-rung reduced states are
positive, and the largest dimension-64 per-step trace correction is
about 3.3e-4. These diagnostics plus bond refinement support the observed
trend but are not a certificate of global positivity for 32 spins.

The two experiments are not energy-matched: at 32 spins, the theta=0.3
isolated state has excess energy about 0.060 per spin, whereas the
no-noise bath state has about 0.404 per spin. The observed distinction
could reflect the energy window as well as the preparation and state
structure. Experiment A starts with long-range order already present;
it does not demonstrate formation of that order from an uncorrelated
initial state. Finite time t<=6, width<=16, fixed height 2, and this
specific periodic image cannot establish infinite-size, infinite-time,
or two-dimensional universality.

## Reproduce and inspect

From the repository root:

```sh
.venv/bin/python -m pip install -r algorithm1/requirements-tensor.txt
.venv/bin/python -m pytest -c algorithm1/pytest.ini -q
.venv/bin/python algorithm1/python/run_ladder_quench.py
.venv/bin/python algorithm1/python/run_ladder_quench.py --angles 0.3 0.6 --dt 0.02 --bond 128
.venv/bin/python algorithm1/python/run_ladder_bath.py
.venv/bin/python algorithm1/python/run_ladder_bath.py --resume-bond 32 --bond 64 --sweeps 25
.venv/bin/python algorithm1/python/check_ladder_bath_dense.py
.venv/bin/python algorithm1/python/run_ladder_quench.py --normalization bulk --angles 0.3
.venv/bin/python algorithm1/python/run_ladder_bath.py --normalization bulk --widths 4 16 --cases Z:0
.venv/bin/python algorithm1/python/run_ladder_quench.py --widths 16 --angles 0.3 --pattern random
.venv/bin/python algorithm1/python/run_ladder_bath.py --widths 4 --duration 0.05 --bond 64 --sweeps 120
.venv/bin/python algorithm1/python/plot_ladder_study.py
```

Runners reuse cached files with matching names; use a new `--output`
directory for independent reruns or altered code. The main plots require
all four default widths. Larger widths are accepted in multiples of four,
but cost and required bond dimension can grow substantially.

`results/ladders/` contains per-run JSON, saved ground and density MPS
arrays, `summary.json`, `size-comparison.csv`, `ladder-correlations.mat`,
and PNG/SVG figures. Results are ignored by Git; the code and this note
are the reproducible record. The MAT file contains cells of raw `(2,L)`
edge-seed correlations, not sign-aligned values.

Core code: `python/ladder_tensor.py`, `python/ladder_bath.py`. The public
[Quimb TEBD guide](https://quimb.readthedocs.io/en/latest/examples/ex_TEBD_evo.html)
describes the underlying time-evolution API. This work is a classical
tensor-network validation, not a compiled large-system hardware circuit.

## Next tests of the conjecture

1. **Compare states at matched energy density.** Build ensembles of local
   rotations, symmetry-preserving circuits, and mixed states in the same
   energy window. Measure Gamma, connected correlations, and the minimum
   sign margin against energy and separation. This isolates whether energy
   explains the robustness better than proximity to a Gibbs state.
2. **Increase time together with separation.** At fixed time, pre-existing
   order can persist before perturbations propagate across a larger
   sample. Extend t in proportion to L, with independent time-step and
   bond convergence; use an operational margin threshold for lifetimes.
3. **Test recovery as well as retention.** Vary initial states of the same
   bath, including ordered, rotated and maximally mixed states. Replace
   the bond-only bath by larger overlapping-cluster constructions and
   calibrate their global thermal bias on exactly solvable sizes.
4. **Sample disorder and geometry.** Repeat many random perturbation
   realizations, vary block lengths and kappa, and include taller strips.
   Test bulk pairs at fixed separations across sizes to separate boundary
   effects from distance effects. Report uncertainty across realizations.

The useful conjecture to test next is robustness across a specified
energy-and-symmetry class of states, with a nonvanishing measurable
correlation margin as separation and observation time grow.

## XXZ worldline heat-bath dynamics

The October 6 study preserves the **full XXZ image Hamiltonian**, including
both exchange terms. [Martinelli and Wouts, arXiv:1105.5970v1](https://arxiv.org/pdf/1105.5970v1)
define single-site heat-bath resampling of imaginary-time paths for the
transverse-field Ising model. Their generator and their regular-tree mixing
results do not apply unchanged to this XXZ ladder. Here an exchange changes
two neighboring paths together. Independently resampling a single path
would violate the exchange constraints or leave the sampler nonergodic.

The implemented adaptation uses stochastic series expansion (SSE), with
heat-bath directed loops following the framework of
[Syljuasen and Sandvik, Phys. Rev. E 66, 046701 (2002), Sections II–III](https://arxiv.org/pdf/cond-mat/0202316).
It implements the path-space heat-bath idea with an XXZ-compatible update;
it is not a reproduction of the Ising paper's generator. **Monte Carlo
sweeps are algorithmic updates, not real-time thermal-bath evolution.**

### What the code samples

Keep `H=-sum_e(XX+YY+Delta_e ZZ)/4`, the same unique-edge sample sigma,
row-major image indexing, `kappa=2`, and effective inverse temperature
`b=beta_paper/N`. Write `W_e=C_e I-h_e` with
`C_e=abs(Delta_e)/4+epsilon`, default `epsilon=1/4`. Its six nonzero
computational-basis matrix elements are

\[
W(\uparrow\uparrow\to\uparrow\uparrow)
=W(\downarrow\downarrow\to\downarrow\downarrow)=C_e+\Delta_e/4,
\]
\[
W(\uparrow\downarrow\to\uparrow\downarrow)
=W(\downarrow\uparrow\to\downarrow\uparrow)=C_e-\Delta_e/4,
\qquad
W(\uparrow\downarrow\leftrightarrow\downarrow\uparrow)=1/2.
\]

All are positive for the image couplings. Expanding
`exp[b sum_e W_e]` gives positive weights for periodic operator strings;
the scalar shift cancels from normalized physical observables. The code
uses the following updates, derived for these particular weights:

1. Scan the padded string. Given the other slots with `m_rest` operators,
   resample a diagonal/identity slot with relative weights `M-m_rest` for
   identity and `b W_e(s,s)` for each diagonal bond label. Exchange slots
   are propagated, updating the instantaneous Z configuration.
2. Link each vertex leg to the next occurrence of that spin, closing the
   imaginary-time boundary. Start a loop at a uniformly selected leg.
   For each possible exit, flip the entrance and exit spins and choose
   with probability proportional to the new allowed vertex weight. Include
   a bounce. The code checks the local balance relation
   `W(v)P(v,a→v',b)=W(v')P(v',b→v,a)` for every allowed move.
3. Close each loop before measuring. One sweep uses **N fixed loop
   attempts**, including immediate bounces. Do not stop a sweep after a
   state-dependent number of visited legs: that changes the sampling
   clock. Winding loops can change the total-magnetization sector.

There is no Trotter time step or MPS truncation. The finite expansion
capacity is chosen using a conservative omitted-partition-weight bound
below `exp(-50)`. For `W=sum_e W_e` with row-sum bound `R`, the bound is
`exp[b(R-sum C_e)] Pr(Poisson(bR)>M)`. It follows from
`Tr W^m <= 2^N R^m` and `Tr exp(bW) >= 2^N exp(b sum C_e)`.
This controls expansion truncation; it says nothing about mixing time.

At each checkpoint the observable integrates over the imaginary-time
gaps, yielding `Gamma_sj=<Z_s Z_j>/4`. Both the corner seed and the
interior seed `(0,L/4)` are measured. For interpretation of connected
correlations, symmetrize each sampled path with its global spin inverse:
one-point Z means are then exactly zero, and connected equals raw Gamma.
Sign alignment uses the intended black/white labels only when plotting.

### Experiments and interpretation

The main `b=6` series uses 512 independent replicas for each size and each
initialization through 800 sweeps. Initial paths are constant in imaginary
time: image-ordered, image-ordered with independent 10% spin corruption,
or independent random Z spins. The initial diagonal operators are drawn
conditionally on those paths. These initializations are not an evolution
of the pure all-+x state studied in the previous bath experiment.

Longer runs use 128 independent replicas on 16, 24, and 32 spins through
12,800 sweeps. Separate reference chains start alternately ordered and
random. The extended references discard 12,000 sweeps, then retain 4,096
samples per chain separated by ten sweeps, over eight chains. Error
estimates retain autocorrelation by taking the larger of batch-mean and
between-chain standard errors. Split R-hat is monitored for the far
correlation, expansion-order energy estimator, and magnetization squared.
These are convergence diagnostics, not a proof of complete mixing.

The initial 32-spin reference had correlation R-hat about 1.20 despite
apparently stationary energy. It was superseded by the extended reference;
the shorter run is retained as evidence of the mixing problem. The
correlation-versus-sweep plot uses the extended references when available.

At 800 sweeps, the far-corner correlations from random paths are
approximately 0.2497, 0.2318, 0.1574, and 0.0986 for 8, 16, 24, and 32
spins. This decline at a fixed computational budget is chiefly a
relaxation effect: the extended 32-spin random-start run reaches about
0.2456. Its ordered-start value is about 0.2496. The long reference is
about 0.2488, with an approximate 95% sampling interval of +/-0.0014;
finite-replica transient intervals are wider. Values near the maximum
0.25 must not be assigned zero uncertainty just because a finite sample
contains no distant defects.

The plotting code therefore uses bounded-variable Chernoff/KL intervals
for the independent transient replicas. Far-corner bands are pointwise
95% intervals. The profile-distance bands use Bonferroni over the two seed
profiles, with reference uncertainty added using batch-based standard
errors. Reference intervals remain conditional on adequate mixing; the
plots are not simultaneous confidence statements across all sweep times.
The ell-infinity curve is the distance between estimated profiles, not
the expectation of a single-path maximum.

Temperature provides a counter-test to unrestricted robustness. For the
32-spin reference, the far correlation is about `0.057 +/- 0.014` at
`b=2`, and `0.0014 +/- 0.0051` at `b=1` (approximate 95% intervals).
The latter is consistent with zero at this precision. Strong correlation
is retained and recovered in the tested cold regime, not across arbitrary
temperatures or states. The strip still has fixed height two; none of
these finite-size data establish thermodynamic long-range order.

### A state-level check away from the quantum Gibbs state

For eight spins the code also records the distribution of Z configurations
from each path ensemble. Preparing these configurations as a classical
mixture defines a valid **separable diagonal output state**. It reproduces
the measured ZZ profiles but is not the evolving density matrix of a
physical bath channel. Its trace distance to the exact quantum Gibbs
state is plotted separately from the correlation-profile distance.

At stationarity this output is `D_Z(rho_b)`, complete dephasing of the
Gibbs state in the computational basis. An independent exact calculation
at `b=6` gives

\[
\tfrac12\|\mathcal D_Z(\rho_b)-\rho_b\|_1=0.2000109318,
\qquad
\Gamma_{sj}(\mathcal D_Z(\rho_b))=\Gamma_{sj}(\rho_b).
\]

This is a concrete example of correlations surviving a substantial state
change. Their preservation under Z dephasing follows by construction,
because ZZ observables are diagonal; it does not establish robustness to
generic noise or a quantum preparation advantage. It also shows why
these correlations alone cannot certify entanglement. Combining the
sampling tests with matched-energy physical perturbations remains the
appropriate next test of the broader conjecture.

### Reproduction, validation, and files

From the repository root, using the existing Python environment:

```sh
.venv/bin/python -m pip install -r algorithm1/requirements-glauber.txt
.venv/bin/python -m pytest -q -c algorithm1/pytest.ini
.venv/bin/python algorithm1/python/run_xxz_glauber.py --replicas 512
.venv/bin/python algorithm1/python/run_xxz_glauber.py --widths 8 12 16 \
  --replicas 128 --burn 12000 --reference-samples 4096 --thin 10 \
  --times 0 100 400 800 1600 3200 6400 12800 \
  --output algorithm1/results/xxz-glauber/long-run
.venv/bin/python algorithm1/python/run_xxz_glauber.py --widths 4 16 \
  --b 1 2 --replicas 256 \
  --output algorithm1/results/xxz-glauber/temperature-control
.venv/bin/python algorithm1/python/plot_xxz_glauber.py
```

Use a different `--output` directory to preserve an earlier run: this
runner overwrites matching filenames rather than reusing a cache.
`python/xxz_glauber.py` contains the sampler; `run_xxz_glauber.py` contains
the experimental protocol; `plot_xxz_glauber.py` produces three PNG/SVG
figures, summary JSON/CSV, and `xxz-glauber.mat`. The MAT export preserves
raw signed Gamma with axes `(checkpoint, seed, image row, image column)`.
MATLAB execution was not performed. These results remain Git-ignored.

The suite checks reconstruction of the full XXZ bond, every local
scattering balance equation, compatibility with the existing Hamiltonian,
periodic path closure, sampling of magnetization sectors, two-spin
benchmarks with positive/negative/zero anisotropy, and eight-spin exact
Gibbs benchmarks. It also verifies that path histograms reproduce Gamma.
All 76 tests passed after the sampler was added.

Storage grows with the expansion order, typically proportional to
`b |E|`, instead of a `2^N` density matrix. The current implementation
uses simple linear scans over bond labels, so it is not optimized for very
large graphs. Loop lengths and mixing can grow sharply with size, as the
random-start trajectories demonstrate. This is classical numerical
validation of the unchanged quantum Hamiltonian; no quantum circuit or
scalable physical thermalization claim follows from these Monte Carlo
timings.
