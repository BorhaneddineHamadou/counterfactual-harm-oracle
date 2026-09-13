# Counterfactual harm: a continuous test oracle for autonomous driving systems

Replication package for **"How Close Was That? Grounding Continuous Test
Oracles for Autonomous Vehicles in Counterfactual Harm"**.

The paper defines the harm of a test run as the fraction of its
kernel-perturbed replays that crash, each weighted by a published
injury-risk curve (`H_kappa`), measures it by Monte-Carlo replay on two
driving systems in two simulators (openpilot in MetaDrive, TransFuser in
CARLA), and distills a field oracle that predicts it from a single recorded
run. This repository holds the implementation, both simulator adapters,
the versioned configurations and seeded suites, the derived campaign
record of both reference campaigns (150 labelled references and 15,000
replays per subject), and one script per result in the paper. Every number
in the paper recomputes from `data/` on a CPU; the raw simulation traces
(Zenodo, DOI [10.5281/zenodo.22736471](https://doi.org/10.5281/zenodo.22736471))
and the GPU campaigns are documented separately (`docs/TRACES.md`,
`docs/CAMPAIGN.md`).

## Quick start

```bash
pip install -r requirements.txt          # numpy, scipy, scikit-learn, joblib (Python >= 3.9)
python -m pytest tests -q                # core invariants incl. Remark 1 (binary verdict as special case)

python analysis/campaign_summary.py      # Sec. 5.1, Fig. 1 example, the abstract's inversion pair   (seconds)
python analysis/rq1_report.py            # Table 1 and the RQ1 prose, from stored out-of-fold scores  (~2 min)
python analysis/rq1_calibration.py       # conformal coverage, MAE, label reliability                  (seconds)
python analysis/rq1_ablations.py all     # corpus / label-sharing / target / tau ablations             (seconds)
python analysis/rq2_predictive_validity.py   # Table 2, the outside check on held-out executions       (~3 min)
python analysis/rq3_kernel_rescaling.py  # Table 3, kernel sensitivity                                 (seconds)
python analysis/rq3_injury_curves.py     # fourteen published injury curves                             (seconds)
python analysis/rq4_portability.py       # Sec. 5.5, Fig. 3 data                                       (~10 min)
python analysis/crime_baselines.py       # the 35 CommonRoad-CriMe measures against harm               (seconds)
python analysis/regulatory_mapping.py    # the suites against UN R157 / Euro NCAP grids                 (seconds)
python analysis/axioms.py                # the metamorphic battery, 219 cases                          (minutes)
python analysis/make_figures.py          # Fig. 1 (right) and Fig. 3
```

Each script prints its numbers next to the paper's values in brackets and
writes a JSON under `results/`. Scripts whose inputs are trained models
read the stored out-of-fold scores by default; every one of them has a
`--rerun` / `--retrain` path that recomputes from `data/`, and the field
tier itself is retrained with

```bash
python analysis/rq1_train_field_tier.py openpilot     # ~1 h on 32 cores; reproduces results/rq1/openpilot_field_tier_oof.npz
python analysis/rq1_train_field_tier.py transfuser    # ~30 min
```

The full claim-to-command map with expected values is `docs/REPRODUCTION.md`.

## What is where

```
proxima/            the measure and the instrument (numpy / scipy / scikit-learn only)
  kernel.py           the declared perturbation kernel kappa, rescalable by alpha
  injury.py           the severity curve iota (Kusano-Gabler MAIS2+); injury_library.py: fourteen published alternatives
  trace.py            canonical execution trace, criticality peak t_crit
  tier1.py, sim.py    the reference oracle over the SimulatorAdapter interface (run_suite, branch)
  features*.py        the field tier's inputs: 41 + 30 single-trace and 71 checkpoint-trajectory descriptors
  field_tier.py       the field oracle: replay corpus, gradient-boosted readouts, tie rule, conformal interval, escalation
  metrics.py          harm-weighted APFD, AUC, tie statistics, split-half reliability
  crime_bridge.py     trace -> CommonRoad scenario, for the third-party criticality measures
adapters/
  metadrive_openpilot/  openpilot + MetaDrive: templates, seeded/recording/perturbing bridge, worker, harvest, SLURM
  carla_transfuser/     TransFuser + CARLA 0.9.10.1: templates, synchronous agent host, control perturbation, worker, harvest, SLURM
  pilot.py              the synthetic pilot world (CPU) on which the protocol was fixed and the axiom battery runs
campaign/           range calibration, suite generation, replay-job generation, trace -> data/ builders, CriMe runner
configs/            measure.json (kernel, curve, branch, campaign layout, seeds, subjects), field_tier.json, scenario ranges
data/               the derived campaign record of both subjects (data/README.md)
analysis/           one script per paper result (analysis/README.md), common.py = the single data loader
results/            precomputed outputs and reruns (results/README.md)
docs/               DEFINITIONS.md (the measure in full), FEATURES.md, REPRODUCTION.md, CAMPAIGN.md, TRACES.md
tests/              unit tests
tools/              pack / fetch the raw trace tarballs
```

## Subjects

| subject | system | simulator | reference campaign |
|---|---|---|---|
| openpilot | commit `492ed731` (master, 2026-05-11), stock `tools/sim` bridge | MetaDrive | 750 nominal runs + 15,000 replays (~800 GPU-h), global seed 20260722 |
| TransFuser | PAMI 2022 model, repository branch `2022`, commit `9d413b2a` | CARLA 0.9.10.1, synchronous, Town01 | 750 + 15,000 (~1,400 GPU-h), global seed 20260806 |

Five NHTSA-style pre-crash templates per subject (lead deceleration,
stopped lead, cut-in, oncoming drift, crossing vehicle), 30 Latin-hypercube
scenarios each, five executions per scenario, M = 100 kernel replays of
every scenario's first execution. Every scenario draw, execution seed and
replay disturbance derives from the global seed (`configs/measure.json`).

## Environment

Tested with Python 3.11, numpy 1.26, scipy 1.17, scikit-learn 1.8,
joblib 1.5. The gradient-boosted readouts are deterministic given their
seeds, so retraining reproduces the stored out-of-fold scores to floating
point precision on the same scikit-learn version; other versions can move
numbers in the third decimal. The simulation layer needs the pinned
openpilot / MetaDrive and CARLA / TransFuser environments described in
`adapters/*/README.md` and SLURM GPU nodes.

## License

MIT (`LICENSE`).
