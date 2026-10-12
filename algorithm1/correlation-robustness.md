# Correlation robustness away from a Gibbs state

The research objective is to determine whether image-dependent XXZ
Hamiltonians support useful spatial correlations across a broad class of
states. Accurate Gibbs-state preparation is no longer the success criterion.
The original Gibbs state is retained as one reproducible comparison.

The current experiment uses the existing 2x2 image `[[0,0],[255,255]]`,
`kappa=2`, `beta=24` (`b=beta/N=6`), and bath parameter `dt=0.1`.
One sweep applies the same 160 contacts as the bath reference. All numerical
results are ideal ensemble density-matrix calculations. They are not
hardware results, shot estimates, or tests on larger images.

## Observables and operational questions

For seed pixel 0, measure

\[
\Gamma_j(s)=\operatorname{Tr}[\rho_s S_0^z S_j^z],\qquad
C_j(s)=\Gamma_j(s)-\langle S_0^z\rangle_s\langle S_j^z\rangle_s.
\]

Raw Gamma is the quantity used by Algorithm 1. Connected C removes the
product of local magnetizations. Nonzero connected correlations need not
be quantum entanglement: a classical mixture can have them. A polarized
product state has nonzero raw correlations and zero connected correlations.

The plots and exports include:

1. Every off-seed Gamma value versus sweep. The self pair is always 1/4 and
   is excluded from the comparison norms.
2. `e_inf(s) = max_{j != 0} |Gamma_j(s) - Gamma_j(reference)|`.
   This uses the Gibbs state only as a baseline.
3. `delta_inf(s) = max_{j != 0} |Gamma_j(s) - Gamma_j(s-1)|`.
   This measures change per sweep without a Gibbs reference. It is undefined
   at sweep zero. Small change alone can also describe a frozen wrong state.
4. Departure from the initial correlations, exported as `linf_initial`, to
   distinguish retention of existing order from formation of correlations.
5. The maximum pairwise l-infinity distance between sampled runs at each
   sweep, an initial-state sensitivity measure with no Gibbs reference.
6. `D(s) = ||rho_s - rho_reference||_1/2`, to see whether useful observable
   behavior occurs while the full state is still far from the reference.
7. The minimum signed correlation
   `m(s) = min_{j != 0} sign(Gamma_j(reference))*Gamma_j(s)`.
   A positive margin means all seed labels agree with the reference; its
   magnitude measures distance from an ambiguous zero. In this synthetic
   image the reference labels also match the known two-region partition.

For raw Gamma the bound `e_inf <= D/2` holds because `||Sz Sz|| = 1/4`.
There is no reverse implication: matching these few observables need not
make two density matrices close. The code checks this bound along every run.

Diagnostic thresholds are an absolute correlation tolerance 0.01, label
margin 0.02, and state distance 0.1 for flagging a state far from the chosen
reference. These are explicit experimental choices, not theoretical phase
boundaries. All individual curves and data remain available. A reported
"persistent" threshold crossing means persistence through the observed
60 sweeps only.

## What the first sweep plots show

Four named states are compared with 32 random product pure states and 32
global Haar-random pure states. Product spins are sampled independently
from the single-spin Haar measure. Random seed is 1200. These ensembles
sample different starting states; they are not a guarantee over all states.

| Initial state | First sweep with all label margins >= 0.02 | State distance at that sweep | First persistent e_inf <= 0.01 | First persistent D <= 0.01 |
| --- | ---: | ---: | ---: | ---: |
| Polarized 0000 | 3 | 0.5531 | 11 | 14 |
| Maximally mixed | 1 | 0.7062 | 10 | 13 |
| Coherent ++++ | 1 | 0.5774 | 8 | 12 |
| Domain product 0011 | 0 | 0.7312 | 4 | 12 |

All 32 sampled random product states achieve the label margin by sweep 3;
all 32 global Haar states do so by sweep 2. Correlation distance drops below
0.01 at sweeps 8-10 for these random ensembles, while state distance drops
below 0.01 at sweeps 12-14.

The evidence is strongest for **label robustness before state equilibration**.
The numerical correlation values generally relax on a time scale comparable
to the state, with a modest earlier threshold crossing. The domain initial
state is an exception with substantially closer correlations than its full
state distance would suggest. These results do not establish a parametrically
large separation of relaxation times or a universality mechanism.

All these trajectories approach the same deliberately constructed bath
fixed point. Their eventual agreement is expected and is not by itself
evidence of universality.

## Sustained perturbations: states that remain nonthermal

After every bath sweep apply independent local channels
`E_j(rho) = (1-p)rho + p P_j rho P_j`, with either P=Z or P=X.
Each parameter setting starts from I/16 and runs for 200 sweeps. We observe
immediately after the noise operation. Every final one-cycle residual is
checked against 1e-9. This is a stationary state of a driven discrete channel,
not necessarily a stationary state of isolated H evolution.

| Repeated perturbation | D from original Gibbs | e_inf | Minimum signed correlation |
| --- | ---: | ---: | ---: |
| Z noise, p=0.02 | 0.14646 | 0.00934 | 0.16506 |
| Z noise, p=0.50 | 0.68239 | 0.08558 | 0.12156 |
| X noise, p=0.02 | 0.20089 | 0.05289 | 0.13220 |
| X noise, p=0.50 | 0.83582 | 0.24444 | 0 |

At p=0.5, independent Z noise completely dephases every spin. The
post-noise state is diagonal in the product Z basis and is therefore a
separable classical mixture. Its farthest-pair connected correlation is
still -0.12156; segmentation can survive without entanglement in that
observed state. No assertion is made about entanglement during the cycle.

For these nonzero perturbations, the nonzero commutator `[H,rho]` certifies
that the final state is not `exp(-b'H)/Z` at any temperature b'. For example,
at Z noise p=0.02, `||[H,rho]||_F / ||H||_F = 0.005284`.
Distance from just one Gibbs temperature would not establish this fact.

At p=0.5, X noise sets every offsite ZZ correlation to zero at the
observation time. This is an explicit failure control. More generally one
application of independent X noise multiplies any offsite raw ZZ
correlation by `(1-2p)^2`. The driven test therefore probes different
sensitivities rather than assuming robustness to arbitrary noise.

There is also an exact, deliberately simple control: fully dephase the
reference state once in the Z basis. All ZZ correlations are unchanged,
but its state distance is 0.60814. This follows from diagonality of the
observables and is **not evidence of universality**. The repeated-noise
experiment asks a stronger dynamical question because the intervening bath
contacts can change the populations and correlations on the next sweep.

## Tests that would discriminate a mechanism

The present largest separation is two lattice edges. It cannot establish
long-range order. The next experiments should be prioritized as follows.

1. **Increase system size and separation.** Use chains and narrow ladders
   with repeated two-region image patterns, then genuinely larger 2D images.
   Plot both raw and connected correlations at r proportional to L and their
   full spatial profile, splitting within-region and across-boundary pairs.
   Track regional magnetizations too: a symmetry-broken ordered state can
   have long-range raw order while its connected correlations decay. Compare a
   nonzero plateau, algebraic decay, and exponential decay; do not cancel
   positive and negative correlations by indiscriminate averaging. Check
   dependence on boundary conditions and seed positions. Long-range order,
   finite correlation length, and useful finite-image segmentation are
   distinct outcomes. MPS/MPO or trajectory calculations need bond-dimension,
   time-step, and sampling convergence checks. The exact frequency-resolved
   bath in this prototype must be replaced for that scaling study.

2. **Preserve physical temperature conventions when varying size.** Here
   the exponent is -(beta/N)H with extensive H. Hold b=beta/N fixed for a
   temperature-controlled size comparison. Holding the paper parameter beta
   fixed changes b with N and would confound size and temperature. If that
   scaling is itself the hypothesis, test and label both conventions.

3. **Turn off the bath after forming correlations.** Evolve with H alone,
   with a quenched H, and with alternative local bath protocols. Record
   Gamma, connected C, label margins, and how long they survive. A plateau
   shared across preparation methods is stronger evidence than convergence
   under one channel designed to reach a fixed state. Compare correlation
   decay times versus state relaxation times as functions of L.

4. **Specify and sample the state class.** Test low-energy mixtures,
   energy-matched states, eigenstate mixtures, tilted domain states, and
   weakly entangled states, with fixed and varied magnetization sectors.
   Define energy windows and perturbation strengths before selecting
   examples. The maximally mixed state has zero offsite correlations, so
   robustness cannot hold for all states. Retention and generation from an
   initially uncorrelated state should be reported separately.

5. **Break candidate protections deliberately.** Compare homogeneous XXZ
   chains with image-dependent couplings, random bonds, small longitudinal
   and transverse fields, and added longer-range bonds. Track which
   perturbations preserve U(1) magnetization, global spin-flip symmetry, or
   integrability. A failure boundary can identify the relevant protection.
   Complete generalized Gibbs ensembles describe short-range post-quench
   correlators in integrable XXZ examples, making conserved quantities one
   candidate mechanism; this does not imply that the inhomogeneous 2D image
   Hamiltonian is integrable or has the same behavior. See
   [Piroli et al., 1611.08859](https://arxiv.org/abs/1611.08859).

6. **Separate classical robustness from quantum effects.** Retain the
   dephasing control, compare with diagonal Ising models and classical spin
   dynamics, and measure entanglement separately if claiming a quantum
   mechanism. At matched energy/correlation strength, test whether XX+YY
   terms improve the observable's robustness or only its preparation.

7. **Test readout and labels on varied images.** Repeat on different
   geometries, contrast, anisotropies, and seeds. Estimate the fraction of
   pixels with confidence intervals crossing zero, and segmentation quality
   against supplied image labels. Strong signs can survive large amplitude
   errors, while weak long-distance signals may require many shots. A claim
   of universality would additionally need a common scaling law across
   system sizes and model variations, not only several successful examples.

## Reproduce and inspect

From the repository root:

```sh
.venv/bin/python -m pip install -r algorithm1/requirements-plot.txt
.venv/bin/python algorithm1/python/run_correlation_study.py
.venv/bin/python -m pytest -c algorithm1/pytest.ini -q
```

Outputs go into the Git-ignored `algorithm1/results/correlations/`:

- `correlations-vs-sweeps.png/.svg`: individual seed correlations.
- `correlation-distances.png/.svg`: reference error, change per sweep, state distance.
- `initial-state-ensemble.png/.svg`: random-state bands, between-run diameter, label success.
- `connected-correlations.png/.svg`: raw versus connected farthest-pair correlations.
- `driven-robustness.png/.svg`: sustained-noise comparison.
- `correlations-vs-sweeps.csv`: every run and sweep, including correlation values.
- `trajectories.npz`: arrays plus initial, final, and reference states.
- `correlations.mat`: the same observables for MATLAB; trajectory arrays are
  `(sweep, run, pixel)`; density arrays ending in `_runs_first` are `(run,d,d)`.
  `driven_paths` has shape `(setting,sweep,7)`: Gamma for pixels 0-3 followed
  by correlation l-infinity error, trace distance, and minimum signed margin.
  Parameter-setting order is the `driven_stationary_results` order in JSON.
- `report.json`: parameters, threshold crossings, noise results, and limitations.

MATLAB can load the exported arrays without using Qiskit. For example,
`f=load('algorithm1/results/correlations/correlations.mat');`
`plot(f.sweep,squeeze(f.gamma(:,1,2:4)));`
plots the first run's three off-seed correlations. MATLAB execution has not
been checked in this workspace.
