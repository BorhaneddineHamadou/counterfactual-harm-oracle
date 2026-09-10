"""Reference-campaign driver for TransFuser+CARLA (paper Sec. 4.2), the
analogue of campaign/openpilot_campaign.py with the same stages.

Stages (each idempotent; workers skip existing traces):
  gen-nominal : LHS suite (N_PER_TEMPLATE per template, calibrated ranges
                configs/transfuser_calibrated_ranges.json) x K_RUNS seeds.
  gen-tier1   : after nominal traces exist: reference = run 0 per scenario,
                t* = max(0, t_crit - T_h), M kernel replays per reference.
  status      : progress of both stages.

Cost basis (calibration 2026-08-06): ~5.5 min wall per 40 s run.
nominal = 150 x 5 = 750 jobs (~69 GPU-h); tier-1 sizing decided at
gen-tier1 time (M=100 parity costs ~1375 GPU-h -- see M_REPLAYS note).

Trace files land in <store>/traces/{nominal,tier1,rq2_a05,rq2_a20}/ and the
job lists in <store>/jobs/; the store defaults to data/transfuser (CH_STORE
overrides). Derived datasets are then built with campaign/build_features.py.

Usage: python3 campaign/transfuser_campaign.py gen-nominal|gen-tier1|gen-rq2|status
"""
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from adapters.carla_transfuser import harvest as hv  # noqa: E402
from proxima.kernel import Kernel  # noqa: E402

OUT = os.environ.get("CH_STORE", os.path.join(BASE, "data", "transfuser"))
JOBS = os.path.join(OUT, "jobs")
TR_NOM = os.path.join(OUT, "traces", "nominal")
TR_T1 = os.path.join(OUT, "traces", "tier1")
SCN_FILE = os.path.join(OUT, "campaign_scenarios.json")
RANGES_FILE = os.path.join(BASE, "configs",
                           "transfuser_calibrated_ranges.json")

N_PER_TEMPLATE = 30
K_RUNS = 5
M_REPLAYS = int(os.environ.get("TF_M_REPLAYS", "100"))
BRANCH_HORIZON_S = 4.0
DURATION = 40
GLOBAL_SEED = 20260806
SLURM = os.path.join(BASE, "adapters", "carla_transfuser", "campaign.slurm")

# RQ2: same references replayed under a rescaled kernel. M reduced vs
# tier-1 (M-sufficiency: rank rho 0.95 at M=25 subsample); branch_jobs
# reseeds rng from [seed, 999], so the alpha runs reuse tier-1's
# underlying draws exactly (CRN across scales).
RQ2_SCALES = (0.5, 2.0)
M_RQ2 = int(os.environ.get("TF_M_RQ2", "20"))


def _ranges():
    raw = json.load(open(RANGES_FILE))
    return {k: {p: tuple(v) for p, v in d.items()} for k, d in raw.items()}


def gen_nominal():
    from scipy.stats import qmc
    os.makedirs(TR_NOM, exist_ok=True)
    os.makedirs(JOBS, exist_ok=True)
    ranges = _ranges()
    scenarios = []
    for ti, (scls, rng_d) in enumerate(ranges.items()):
        names = list(rng_d)
        u = qmc.LatinHypercube(d=len(names),
                               seed=GLOBAL_SEED + 7919 * ti).random(
                                   N_PER_TEMPLATE)
        lo = np.array([rng_d[k][0] for k in names])
        hi = np.array([rng_d[k][1] for k in names])
        for row in lo + u * (hi - lo):
            scenarios.append((scls, dict(zip(names, map(float, row)))))
    json.dump([{"scls": s, "params": p} for s, p in scenarios],
              open(SCN_FILE, "w"), indent=1)
    jobs = hv.nominal_jobs(scenarios, K_RUNS, GLOBAL_SEED, TR_NOM, DURATION)
    path = os.path.join(JOBS, "nominal_jobs.jsonl")
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(scenarios)} scenarios x {K_RUNS} runs = {len(jobs)} jobs "
          f"-> {path}")
    print(f"submit:\n  JOBS_FILE={path} sbatch --array=0-7 {SLURM}")


def _nominal_job_list():
    return [json.loads(l) for l in
            open(os.path.join(JOBS, "nominal_jobs.jsonl"))]


def gen_tier1():
    from dataclasses import asdict
    os.makedirs(TR_T1, exist_ok=True)
    kernel = Kernel()
    json.dump({"kernel": asdict(kernel), "M": M_REPLAYS,
               "branch_horizon_s": BRANCH_HORIZON_S},
              open(os.path.join(OUT, "tier1_manifest.json"), "w"), indent=1)
    jobs_out = []
    refs = []
    for job in _nominal_job_list():
        if job["run_idx"] != 0:
            continue
        p = job["trace_out"]
        if not os.path.exists(p):
            continue
        tr = hv.to_trace(p, sid=job["sid"], run_idx=0,
                         global_seed=job["seed"])
        t_star = max(0.0, tr.t_crit - BRANCH_HORIZON_S)
        jobs_out.extend(hv.branch_jobs(job, t_star, kernel, M_REPLAYS,
                                       TR_T1, DURATION))
        refs.append({"sid": job["sid"], "job_id": job["job_id"],
                     "t_star": float(t_star), "t_crit": float(tr.t_crit),
                     "contact": bool(tr.contact)})
    json.dump(refs, open(os.path.join(OUT, "tier1_refs.json"), "w"),
              indent=1)
    path = os.path.join(JOBS, "tier1_jobs.jsonl")
    with open(path, "w") as f:
        for j in jobs_out:
            f.write(json.dumps(j) + "\n")
    print(f"{len(refs)} reference traces x {M_REPLAYS} replays = "
          f"{len(jobs_out)} jobs -> {path}")
    print(f"submit:\n  JOBS_FILE={path} sbatch --array=0-7 {SLURM}")


def _alpha_tag(a):
    return f"a{str(a).replace('.', '')}"     # 0.5 -> a05, 2.0 -> a20


def gen_rq2():
    """Kernel-sensitivity replays: every tier-1 reference re-replayed under
    Kernel().scaled(alpha) for alpha in RQ2_SCALES, M_RQ2 replays each.
    Idempotent; separate trace dir + jobs file per scale."""
    from dataclasses import asdict
    refs = json.load(open(os.path.join(OUT, "tier1_refs.json")))
    by_id = {j["job_id"]: j for j in _nominal_job_list()}
    m_tier1 = json.load(open(os.path.join(OUT, "tier1_manifest.json")))["M"]
    manifest = {"M": M_RQ2, "M_sample": m_tier1,
                "branch_horizon_s": BRANCH_HORIZON_S, "scales": {}}
    for a in RQ2_SCALES:
        tag = _alpha_tag(a)
        kernel = Kernel().scaled(a)
        manifest["scales"][str(a)] = asdict(kernel)
        tr_dir = os.path.join(OUT, "traces", f"rq2_{tag}")
        os.makedirs(tr_dir, exist_ok=True)
        jobs_out = []
        for ref in refs:
            job = by_id[ref["job_id"]]
            jobs_out.extend(hv.branch_jobs(job, ref["t_star"], kernel,
                                           M_RQ2, tr_dir, DURATION,
                                           M_sample=m_tier1))
        path = os.path.join(JOBS, f"rq2_{tag}_jobs.jsonl")
        with open(path, "w") as f:
            for j in jobs_out:
                f.write(json.dumps(j) + "\n")
        n_tasks = (len(jobs_out) + 249) // 250   # <=250 jobs/task (24h limit)
        print(f"alpha={a}: {len(refs)} refs x {M_RQ2} = {len(jobs_out)} jobs "
              f"-> {path}")
        print(f"submit:\n  JOBS_FILE={path} sbatch --array=0-{n_tasks-1} "
              f"{SLURM}")
    json.dump(manifest, open(os.path.join(OUT, "rq2_manifest.json"), "w"),
              indent=1)


def status():
    for name, pat, jf in (
            ("nominal", f"{TR_NOM}/*.npz", "nominal_jobs.jsonl"),
            ("tier1", f"{TR_T1}/*.npz", "tier1_jobs.jsonl"),
            ("rq2_a05", f"{OUT}/traces/rq2_a05/*.npz", "rq2_a05_jobs.jsonl"),
            ("rq2_a20", f"{OUT}/traces/rq2_a20/*.npz", "rq2_a20_jobs.jsonl")):
        jp = os.path.join(JOBS, jf)
        total = sum(1 for _ in open(jp)) if os.path.exists(jp) else 0
        done = len(glob.glob(pat))
        print(f"{name}: {done}/{total}")


if __name__ == "__main__":
    {"gen-nominal": gen_nominal, "gen-tier1": gen_tier1,
     "gen-rq2": gen_rq2, "status": status}[sys.argv[1]]()
