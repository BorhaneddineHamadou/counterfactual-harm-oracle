# TransFuser + CARLA adapter

Subject: TransFuser (PAMI 2022 model, repository branch `2022`, commit
`9d413b2a`), executed in CARLA 0.9.10.1 through its leaderboard agent
interface, hosted in-process by a synchronous CARLA client (fixed step
0.05 s, Town01). Seeded, step-deterministic execution with a single
control hook for the ego-side kernel components.

| file | role |
|---|---|
| `templates.py` | the five pre-crash templates choreographed in the (s, lat) frame of the ego lane: physics-on threat vehicles driven by target velocity with lateral correction; same template and parameter names as the MetaDrive adapter; OU actor noise gated from t* |
| `run_one.py` | one job: server (if not up) + world + ego + agent + template + 16-column `.npz` trace |
| `agent_host.py` | minimal replacement for the leaderboard evaluator: sensor rig from `agent.sensors()`, synchronous tick queues, GPS/IMU/speedometer packets, dense global plan, `agent.run_step` → `VehicleControl` |
| `perturb.py` | `ControlPerturb`: latency = ring-buffer delay on the brake command (throttle held meanwhile), `brake_gain` = scale on the brake command; both gated from `start_step` |
| `worker.py` | jobs-file worker: one `run_one.py` subprocess per job with a wall-clock cap; recycles the CARLA server on a stall and retries once |
| `harvest.py` | `.npz` → `proxima.trace.Trace`, resampling the asynchronous rows onto the 20 Hz grid; `nominal_jobs` / `branch_jobs` |
| `campaign.slurm`, `calib.slurm` | one headless CARLA server + one worker per array task (ports spaced by 50) |
| `setup_env.sh` | one-time environment: CARLA binaries, model checkpoints, Python 3.7 venv with torch 1.12.1 + cu113 and the agent-runtime dependencies, CARLA egg |

## Running

```
bash adapters/carla_transfuser/setup_env.sh      # once (needs the CARLA and model archives)
export REPO_ROOT=~/counterfactual-harm-oracle TF_REPO=~/transfuser_repo CARLA_ROOT=~/carla_09101
export TF_CKPT=~/transfuser_assets/model_ckpt/transfuser
python campaign/transfuser_campaign.py gen-nominal
JOBS_FILE=$REPO_ROOT/data/transfuser/jobs/nominal_jobs.jsonl \
  sbatch --array=0-7 adapters/carla_transfuser/campaign.slurm
```

Runs last 40 simulated seconds (`duration`). Job records carry the same
fields as the MetaDrive adapter (`job_id, scenario_cls, params, sid,
run_idx, seed, duration, perturb, trace_out`); `CH_PERTURB` /
`PROXIMA_PERTURB` holds the perturbation blob for a replay.

## Notes recorded during the campaign

- The model's own policy cruises near 4 m/s (15 km/h) on these routes; the
  suite was calibrated to that regime (`configs/transfuser_scenario_ranges.json`).
- Run-to-run execution is not bit-deterministic through the GPU inference
  path; the residual randomness of the continuation is part of the
  neighbourhood and folds into the reported Monte-Carlo error (paper Sec. 3.4).
- `singularity --nv` does not bind `libnvidia-gpucomp`; the SLURM scripts
  bind it from the host driver so the UE4 server can initialise EGL.
