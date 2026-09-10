"""Reference-campaign driver for openpilot+MetaDrive (paper Sec. 4.2).

Stages (each idempotent; workers skip existing traces):
  gen-nominal : LHS suite (N_PER_TEMPLATE per template, from the calibrated
                ranges file) x K seeds -> nominal job list.
  gen-tier1   : after nominal traces exist: pick reference traces, compute
                t* = max(0, t_crit - T_h), emit M kernel replays per
                reference trace (perturb sampled from the declared Kernel).
  status      : progress of both stages.
  gen-rq2     : the kernel-rescaling replays of RQ3 (alpha 0.5 and 2, M=40).

Trace files land in <store>/traces/{nominal,tier1,rq2_a05,rq2_a20}/ and the
job lists in <store>/jobs/; the store defaults to data/openpilot (CH_STORE
overrides). Derived datasets are then built with campaign/build_features.py.

Usage: python3 campaign/openpilot_campaign.py gen-nominal|gen-tier1|gen-rq2|status
"""
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from adapters.metadrive_openpilot import harvest as hv  # noqa: E402
from proxima.kernel import Kernel  # noqa: E402

OUT = os.environ.get("CH_STORE", os.path.join(BASE, "data", "openpilot"))
JOBS = os.path.join(OUT, "jobs")
TR_NOM = os.path.join(OUT, "traces", "nominal")
TR_T1 = os.path.join(OUT, "traces", "tier1")
SCN_FILE = os.path.join(OUT, "campaign_scenarios.json")
RANGES_FILE = os.path.join(BASE, "configs", "openpilot_scenario_ranges.json")

N_PER_TEMPLATE = 30
K_RUNS = 5
M_REPLAYS = 100
BRANCH_HORIZON_S = 4.0
DURATION = 130
GLOBAL_SEED = 20260722
SLURM = os.path.join(BASE, "adapters", "metadrive_openpilot", "campaign.slurm")

# RQ2: same references replayed under a rescaled kernel. M is reduced --
# the estimand is the scenario RANKING under each scale, not per-ref
# precision -- and branch_jobs derives disturbances from the job seed, so
# the alpha-runs reuse tier1's underlying quantiles (CRN across scales:
# a pure scale effect, no fresh MC sampling noise in the comparison).
RQ2_SCALES = (0.5, 2.0)
M_RQ2 = 40

# fallback ranges (pre-calibration); overwritten by calibrated_ranges.json
DEFAULT_RANGES = {
    "ProxLeadDecel": {"gap0": (30, 70), "v_lead": (4, 10),
                      "g_trig": (20, 40), "a_lead": (2, 6)},
    "ProxLeadStopped": {"d0": (80, 200)},
    "ProxCutIn": {"gap0": (15, 45), "v_cut": (4, 10), "d_trig": (15, 35),
                  "t_lat": (1.5, 3.0), "a_cut": (0, 3)},
    "ProxOncomingDrift": {"r0": (120, 220), "v_onc": (6, 14),
                          "d_trig": (60, 110), "vy_drift": (0.4, 1.0),
                          "y_frac": (0.4, 0.9), "t_stay": (0.5, 2.5)},
    "ProxCrossingTraffic": {"d_place": (100, 180), "d_trig": (30, 70),
                            "v_cross": (2, 5), "y0": (6, 9)},
}


def _ranges():
    if os.path.exists(RANGES_FILE):
        raw = json.load(open(RANGES_FILE))
        return {k: {p: tuple(v) for p, v in d.items()}
                for k, d in raw.items()}
    return DEFAULT_RANGES


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
    print(f"submit:\n  JOBS_FILE={path} sbatch --array=0-11 {SLURM}")


def _nominal_job_list():
    return [json.loads(l) for l in
            open(os.path.join(JOBS, "nominal_jobs.jsonl"))]


def gen_tier1():
    from dataclasses import asdict
    os.makedirs(TR_T1, exist_ok=True)
    kernel = Kernel()
    # the batch adapter validates against this at harvest time: replays on
    # disk embody THIS kernel/M; a caller asking for anything else must fail
    json.dump({"kernel": asdict(kernel), "M": M_REPLAYS,
               "branch_horizon_s": BRANCH_HORIZON_S},
              open(os.path.join(OUT, "tier1_manifest.json"), "w"), indent=1)
    jobs_out = []
    refs = []
    for job in _nominal_job_list():
        if job["run_idx"] != 0:        # reference trace = run 0 per scenario
            continue
        p = job["trace_out"]
        if not os.path.exists(p):
            continue
        tr = hv.to_trace(p, sid=job["sid"], run_idx=0, global_seed=job["seed"])
        t_star = max(0.0, tr.t_crit - BRANCH_HORIZON_S)
        jobs_out.extend(hv.branch_jobs(job, t_star, kernel, M_REPLAYS,
                                       TR_T1, DURATION))
        refs.append({"sid": job["sid"], "job_id": job["job_id"],
                     "t_star": t_star, "t_crit": tr.t_crit,
                     "contact": tr.contact})
    json.dump(refs, open(os.path.join(OUT, "tier1_refs.json"), "w"), indent=1)
    path = os.path.join(JOBS, "tier1_jobs.jsonl")
    with open(path, "w") as f:
        for j in jobs_out:
            f.write(json.dumps(j) + "\n")
    print(f"{len(refs)} reference traces x {M_REPLAYS} replays = "
          f"{len(jobs_out)} jobs -> {path}")
    print(f"submit:\n  JOBS_FILE={path} sbatch --array=0-23 {SLURM}")


def _alpha_tag(a):
    return f"a{str(a).replace('.', '')}"     # 0.5 -> a05, 2.0 -> a20


def gen_rq2():
    """Kernel-sensitivity replays: every tier-1 reference re-replayed under
    Kernel().scaled(alpha) for alpha in RQ2_SCALES, M_RQ2 replays each.
    Idempotent; separate trace dir + jobs file per scale."""
    from dataclasses import asdict
    refs = json.load(open(os.path.join(OUT, "tier1_refs.json")))
    by_id = {j["job_id"]: j for j in _nominal_job_list()}
    manifest = {"M": M_RQ2, "branch_horizon_s": BRANCH_HORIZON_S,
                "scales": {}}
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
                                           M_RQ2, tr_dir, DURATION))
        path = os.path.join(JOBS, f"rq2_{tag}_jobs.jsonl")
        with open(path, "w") as f:
            for j in jobs_out:
                f.write(json.dumps(j) + "\n")
        n_tasks = (len(jobs_out) + 299) // 300   # <=300 jobs/task (leak horizon)
        print(f"alpha={a}: {len(refs)} refs x {M_RQ2} = {len(jobs_out)} jobs "
              f"-> {path}")
        print(f"submit:\n  JOBS_FILE={path} sbatch --array=0-{n_tasks-1}%7 "
              f"{SLURM}")
    json.dump(manifest, open(os.path.join(OUT, "rq2_manifest.json"), "w"),
              indent=1)


def status():
    stages = [
        ("nominal", f"{TR_NOM}/*.npz",
         len(_nominal_job_list()) if os.path.exists(
             os.path.join(JOBS, "nominal_jobs.jsonl")) else 0),
        ("tier1", f"{TR_T1}/*.npz",
         sum(1 for _ in open(os.path.join(
             OUT, "campaign_tier1_jobs.jsonl")))
         if os.path.exists(os.path.join(
             OUT, "campaign_tier1_jobs.jsonl")) else 0)]
    for a in RQ2_SCALES:
        tag = _alpha_tag(a)
        jf = os.path.join(JOBS, f"rq2_{tag}_jobs.jsonl")
        if os.path.exists(jf):
            stages.append((f"rq2_{tag}",
                           os.path.join(OUT, "traces", f"rq2_{tag}", "*.npz"),
                           sum(1 for _ in open(jf))))
    for jf in sorted(glob.glob(os.path.join(OUT,
                                            "campaign_rq3_*_jobs.jsonl"))):
        arm = os.path.basename(jf)[len("campaign_rq3_"):-len("_jobs.jsonl")]
        stages.append((f"rq3_{arm}",
                       os.path.join(OUT, f"traces_rq3_{arm}", "*.npz"),
                       sum(1 for _ in open(jf))))
    for name, pattern, total in stages:
        n = len(glob.glob(pattern))
        pct = 100 * n / total if total else 0
        print(f"{name:8s}: {n}/{total} traces ({pct:.1f}%)")


if __name__ == "__main__":
    {"gen-nominal": gen_nominal, "gen-tier1": gen_tier1, "gen-rq2": gen_rq2,
     "status": status}[sys.argv[1]]()
