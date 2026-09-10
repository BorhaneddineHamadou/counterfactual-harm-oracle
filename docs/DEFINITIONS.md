# The measure and the instrument, fully specified

Symbols follow the paper. Machine-readable values: `configs/measure.json`
(the measure and the campaigns) and `configs/field_tier.json` (the field
oracle).

## 1. Counterfactual harm

For a recorded run `e` (one executed scenario, recorded as a telemetry
trace) the reference tier computes

```
H(e) = (1/M) * sum_{i=1..M} iota(s_i)                       (Eq. 1)
```

- each of the `M` replays re-executes `e` under one disturbance drawn from
  the kernel κ, with the system under test reacting freshly inside the
  replay (closed loop, never a kinematic extrapolation);
- `s_i` is the impact speed of replay `i`, the relative speed at first
  contact in m/s, and 0 if the replay ends without contact;
- `iota` is the severity curve with `iota(0) = 0`.

`H_kappa(e) = E_{eps ~ kappa}[ iota(s(e ⊕ eps)) ]` (Definition 2.1) is what
Eq. 1 estimates; the Monte-Carlo standard error `std(iota(s_i)) / sqrt(M)`
is stored with every label (`se` in `data/<subject>/reference_labels.npz`).
The campaigns use `M = 100`.

**Remark 1 (the binary verdict).** With the zero kernel (`Kernel.zero()`)
and the step curve (`injury.step`, any contact → 1), `H` is exactly the
collision bit `B(e)` whenever the replay has no stochastic continuation
left to re-randomize; `tests/test_core.py::test_binary_special_case`
checks this on the pilot world.

## 2. The kernel κ

`proxima/kernel.py`. Three independent components, one per factor on the
conflict-to-contact path (paper Sec. 2.3 and 3.4):

| component | draw per replay | campaign values |
|---|---|---|
| reaction latency | `dlat = clip(N(0, sigma), -bound, bound)`; a negative draw acts as zero delay, so the ego never reacts sooner than the stack did (rectified at zero, bounded) | sigma 0.10 s, bound 0.25 s |
| surface friction | uniform multiplicative scale on achievable deceleration, realised as a gain on the brake command in both adapters | U(0.80, 1.05) |
| actor noise | Ornstein–Uhlenbeck acceleration noise on the threat actor, longitudinal and (drift / cut-in) lateral, mean-reverting | sigma_long 0.50 m/s², sigma_lat 0.12 m/s, theta 1 |

`Kernel.scaled(alpha)` multiplies every spread by `alpha` with the nominal
centres fixed; RQ3 (Sec. 5.4) uses `alpha ∈ {0.5, 2}`. The per-replay draws
of the two ego components are recorded for all 30,000 reference replays
(`rep_dlat`, `rep_gain` in `reference_labels.npz`); the actor-noise stream
is realised inside the simulator from the replay seed
(`execution_seed * 100 + m`).

**Sampling.** `harvest.branch_jobs` seeds `numpy.random.default_rng([seed, 999])`
with the reference execution's seed and calls `Kernel.sample(M)`, so the
draws derive deterministically from the campaign's global seed and the
rescaled arms of RQ3 reuse the same underlying quantiles (common random
numbers). On TransFuser the RQ3 arms draw `M_sample = 100` and use the
first 20, which keeps the friction stream aligned with the reference draws.

## 3. Branch point

`t* = max(0, t_crit − 4 s)`, with `t_crit` the time of first contact or,
without contact, of minimum TTC among active closing steps (else of
minimum gap): `proxima/trace.py::Trace.crit_index`. The prefix before `t*`
is reproduced under the same seeds and disturbances act from `t*` on.
MetaDrive exposes no mid-episode snapshot, so a replay there re-executes
from the start under the original seeds and switches the disturbances on
at the branch step (`start_step = round(t*/0.05)`); the CARLA adapter runs
synchronously and gates the perturbation on the same step index. Latency
jitter applies only if the evasive reaction has not begun at `t*`.

## 4. The severity curve ι

`proxima/injury.py::occupant`:

```
logit iota = -6.068 + 0.100 * s[km/h]
```

the fitted coefficients of the MAIS2+ occupant injury-risk logistic of
Kusano and Gabler (NASS/CDS rear-end crashes), adopted unchanged with the
belt-use covariate dropped. Two declared departures from their use: `s` is
relative speed at contact, not crash-phase occupant Δv, and one curve
serves every crash mode. `proxima/injury_library.py` holds the fourteen
published alternatives of Sec. 5.4 (belt states restored, nine NHTSA
logistics from DOT HS 813 219, the equal-mass Δv = s/2 convention,
Joksch's fourth-power rule).

Reference points: iota(20 km/h) = 0.017, iota(40) = 0.11, iota(47) = 0.20,
iota(61) = 0.50.

## 5. Scenarios and campaigns (Sec. 4.2)

Five NHTSA-style pre-crash templates, implemented for each stack in
`adapters/<stack>/templates.py` with the same parameter names: lead-vehicle
deceleration (`ProxLeadDecel`), lead-vehicle stopped (`ProxLeadStopped`),
cut-in (`ProxCutIn`), oncoming drift (`ProxOncomingDrift`), crossing
vehicle (`ProxCrossingTraffic`). Parameter ranges were calibrated in pilot
rounds (`campaign/*_calibrate.py`) and frozen in
`configs/<subject>_scenario_ranges.json`; the suite is a Latin-hypercube
sample of 30 scenarios per template (`scipy.stats.qmc.LatinHypercube`,
seed `global_seed + 7919 * template_index`), stored as
`data/<subject>/campaign_scenarios.json`.

Per subject: N = 150 scenarios × k = 5 executions = 750 nominal runs;
the run-0 trace of every scenario is the reference, replayed M = 100
times = 15,000 reference replays; 15,750 traces in all. Execution seed
`global_seed + sid * 1000 + run_idx` (global seeds 20260722 openpilot,
20260806 TransFuser). A reference is used only if at least 90% of its
replays completed (all 300 did). Neither stack is modified: the adapters
wrap the stock bridge (openpilot's `tools/sim` MetaDrive bridge; the
TransFuser leaderboard agent hosted in a synchronous CARLA client) and only
seed, record and perturb.

## 6. The field oracle (Tier 2)

`proxima/field_tier.py`; drivers in `analysis/`.

- **Inputs.** 71 single-trace descriptors (`features_expanded.py` 41 +
  `features_extra.py` 30) and 71 checkpoint-trajectory descriptors
  (`features_traj.py`), all from the judged run's own trace
  (`docs/FEATURES.md`).
- **Corpus.** Every trace of the training scenarios: the k nominal
  executions (weight 0.2) and the M replays (weight 0.02), each labelled
  with its scenario's Tier-1 harm. Test rows are always run-0 traces
  scored against their own M = 100 label under scenario-level folds.
- **Readouts.** Gradient-boosted trees (sklearn HistGradientBoosting,
  400 iterations, learning rate 0.05), seed-averaged: an `H > 0`
  classifier `p`, a `log10(H + 1e-4)` regressor and a raw-`H` regressor.
- **Ordering.** `score = 0 if p < tau else 1 + rank(p) + rank(log) + rank(raw)`,
  tau = 0.05 declared. On TransFuser the driver omits `rank(p)` from the
  survivor band and selects tau per fold on its training scenarios; the
  openpilot composite applied to TransFuser gives 0.780 instead of 0.790
  (the paper states this).
- **Magnitude.** `10**log − 1e-4`, wrapped in a split-conformal 90%
  interval whose quantile is calibrated on a held-out quarter of the
  training scenarios.
- **Escalation.** The 10–15% of a test fold with the largest `p(1 − p)`
  are sent to Tier 1 for 3 replays; a clean escalation scores 0.5 and any
  harm scores 3 + mean injury.

## 7. Evaluation protocol (Sec. 4.3)

Ten repeats of five-fold cross-validation at the scenario level
(`analysis/common.py::folds_for`: `np.array_split(default_rng(rep).permutation(150), 5)`),
all predictions out of fold, reported as mean ± s.d. across repeats.
Bootstrap intervals resample scenarios (4,000 draws) and, for rows that
depend on a fit or a replay draw, also draw one repeat per sample. RQ2
scores run 0 against runs 1–4; RQ4 crosses the two stores.
