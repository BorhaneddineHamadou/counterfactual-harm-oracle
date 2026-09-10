# openpilot + MetaDrive adapter

Subject: openpilot, commit `492ed731` (master, 2026-05-11), executed in
MetaDrive through its official simulation bridge (`tools/sim`). The
adapter wraps the stock bridge and only seeds, records and perturbs; the
driving stack is untouched.

| file | role |
|---|---|
| `templates.py` | the five pre-crash templates as scenario classes on the harness's `SceRTScenario` interface (`map_config / env_config / on_reset / on_step`); concrete parameters arrive via `PROXIMA_SCENARIO_PARAMS`; the actor-noise component of the kernel (OU) is applied here, gated by the branch step |
| `bridge.py` | `ProximaMetaDriveBridge`: the campaign bridge with real seeding (`env.reset(seed)`), per-step telemetry written as `.npz` (`PROXIMA_TRACE_OUT`), and the ego-side kernel components gated by `start_step` (`PROXIMA_PERTURB`: `dlat_s` = actuation-latency ring buffer at 100 Hz, `brake_gain` = scale on the braking command) |
| `brake_shaping.py` | the brake-gain shaping used by the bridge |
| `worker.py` | SLURM job consumer: one fresh openpilot manager + MetaDrive world per job, idempotent (skips existing traces), engagement retries, trace-level contact/impact speed in the result record |
| `harvest.py` | `.npz` → `proxima.trace.Trace`; `nominal_jobs` / `branch_jobs` job builders (kernel draws sampled here from the declared `Kernel`) |
| `adapter.py` | `MetaDriveBatchAdapter`: the `SimulatorAdapter` interface (`run_suite`, `branch`) over a finished store, validating kernel / M / t* against the manifest |
| `campaign.slurm`, `calibrate.slurm` | array workers (3-day and 4-hour walltime) |
| `openpilot_sim/` | the simulation harness the bridge builds on: `scenario_bridge.py` (the campaign bridge over openpilot's `MetaDriveBridge`), `run_openpilot_smoke.py` (Xvfb, openpilot manager and one-run driver), `scenarios/base.py` (scenario interface and road layouts) |

## Running

```
export OPENPILOT_ROOT=~/openpilot        # the pinned checkout with tools/sim extras
export OPENPILOT_SIF=~/openpilot_dev.sif # Singularity image of that environment
export REPO_ROOT=~/counterfactual-harm-oracle
python campaign/openpilot_campaign.py gen-nominal
JOBS_FILE=$REPO_ROOT/data/openpilot/jobs/nominal_jobs.jsonl \
  sbatch --array=0-11 adapters/metadrive_openpilot/campaign.slurm
```

Environment contract of one job (set by the worker): `PROXIMA_SCENARIO_PARAMS`
(JSON), `PROXIMA_SEED`, `PROXIMA_TRACE_OUT`, `PROXIMA_PERTURB` (JSON or
unset), `PROXIMA_RENDER_THREATS=1` (threats must be rendered or the
camera-based stack cannot see them). Runs are wall-clock bounded
(`duration` = 130 s) because the bridge ends episodes on wall time.

## Branch semantics on this stack

MetaDrive exposes no mid-episode snapshot, so a replay re-executes the
scenario from the start under the original seeds, reproducing the nominal
prefix, and switches the kernel disturbances on at
`start_step = round(t*/0.05)`. `harvest.branch_jobs` derives every replay's
draws from `default_rng([seed, 999])`, so the campaign is replayable from
the global seed alone.
