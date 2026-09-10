# Re-running the reference campaigns

Nothing in `analysis/` needs a simulator: every number in the paper
recomputes from `data/` on a CPU. This document is the expensive path,
regenerating `data/` from scratch. Budget: about 800 GPU-hours for
openpilot and 1,400 for TransFuser per reference campaign (750 nominal
runs + 15,000 replays), plus 6,000 (openpilot, M = 40) or 3,000
(TransFuser, M = 20) replays for the kernel-rescaling arms of RQ3.

Both subjects follow the same four stages, driven by
`campaign/<subject>_campaign.py` and executed by SLURM array workers in
`adapters/<stack>/`. The store is `data/<subject>/` (`CH_STORE` overrides):

```
data/<subject>/
  campaign_scenarios.json      the concrete suite (150 scenarios)
  jobs/nominal_jobs.jsonl      750 nominal jobs (scenario, params, seed)
  jobs/tier1_jobs.jsonl        15,000 replay jobs (+ per-replay kernel draws)
  jobs/rq2_a05_jobs.jsonl      RQ3 arms, alpha = 0.5 and 2
  jobs/rq2_a20_jobs.jsonl
  tier1_refs.json              per reference: t_crit, t*, contact
  tier1_manifest.json          kernel, M, horizon the replays embody
  rq2_manifest.json            the rescaled kernels
  traces/{nominal,tier1,rq2_a05,rq2_a20}/*.npz   the raw traces
```

The shipped `jobs/*.jsonl` are the exact job lists the paper's campaigns
ran (trace paths relative to the store), so a re-run reproduces the same
scenarios, seeds and kernel draws.

## Stages

1. **Calibrate the ranges** (once per stack; the adopted ranges are shipped
   in `configs/<subject>_scenario_ranges.json`):
   `python campaign/<subject>_calibrate.py generate <round>` → submit the
   printed job list to `adapters/<stack>/calibrate.slurm` (openpilot) or
   `calib.slurm` (TransFuser) → `analyze <round>`. Rounds widen the boxes
   until every template's nominal outcomes span pass, near-miss and
   collision.
2. **Nominal suite**: `python campaign/<subject>_campaign.py gen-nominal`
   samples the Latin hypercube and writes `jobs/nominal_jobs.jsonl`; submit
   as printed (`JOBS_FILE=... sbatch --array=... adapters/<stack>/campaign.slurm`).
3. **Reference replays**: after all nominal traces exist,
   `gen-tier1` computes `t*` per reference and writes the 15,000 replay
   jobs with the kernel draws (`tier1_refs.json`, `tier1_manifest.json`);
   submit. `status` reports progress.
4. **Kernel rescaling** (RQ3): `gen-rq2` writes the alpha 0.5 / 2 arms.

Then build the derived record the analysis reads:

```
python campaign/build_features.py <subject>            # ~10 min, 16 cores
```

which writes `reference_labels.npz`, `features_nominal.npz`,
`features_replays.npz`, `nominal_runs.csv`, `replay_outcomes.npz` and the
`rq2/` arms. The third-party criticality measures need the CommonRoad-CriMe
toolbox and the nominal traces:
`python campaign/run_crime_baselines.py <subject>` → `data/crime/`.

## Environments

- **openpilot + MetaDrive** (`adapters/metadrive_openpilot/README.md`):
  openpilot commit `492ed731` with its simulation extras, run inside a
  Singularity image of that checkout; `OPENPILOT_ROOT`, `OPENPILOT_SIF`,
  `REPO_ROOT` in `campaign.slurm`. The worker starts one openpilot manager
  and one MetaDrive world per run under Xvfb.
- **TransFuser + CARLA 0.9.10.1** (`adapters/carla_transfuser/README.md`):
  `setup_env.sh` builds the Python 3.7 venv with torch 1.12.1, the CARLA
  egg and the TransFuser repository (commit `9d413b2a`); `campaign.slurm`
  starts a headless CARLA server per task and the worker drives the
  leaderboard agent in synchronous mode.

## Operational notes (from the campaigns behind the paper)

- Judge contact from the trace's crash flag, never from the simulator's
  outcome field, which can log `out_of_lane` after a contact.
- openpilot's bridge watchdog can end a run after the ego stops without
  passing through the normal done path; the bridge flushes the trace once
  post-loop (`aborted: true`) so crashed runs are never lost.
- Keep openpilot worker slices under ~300 jobs per task: a slow leak in
  the long-running task otherwise stalls it after ~350 jobs.
- Latching offroad alerts in the shared openpilot params directory block
  engagement for every later run; the worker clears them before each attempt.
- Threats must be rendered (`PROXIMA_RENDER_THREATS=1`, set in the SLURM
  scripts) or openpilot's camera-based stack cannot perceive them.
- CARLA under `singularity --nv` needs `libnvidia-gpucomp` bound from the
  host driver; the SLURM scripts do this.
