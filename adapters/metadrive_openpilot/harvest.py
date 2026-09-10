"""Harvest layer: bridge .npz traces -> proxima Trace objects + branch jobs.

Runs on the login node (no MetaDrive/openpilot imports). The adapter here is
file/batch-based rather than synchronous: run_suite and branch GENERATE job
records for the SLURM worker; harvest() converts finished .npz traces into
the canonical proxima Trace schema so features/tier1/tier2/validation run
unchanged.

Geometry conventions (matching the pilot world):
  gap    : longitudinal bumper-to-bumper distance to the threat along the
           ego heading (CAR_LEN subtracted)
  y_rel  : signed lateral offset of the threat from the ego axis
  closing: relative speed projected on the ego heading
Delta-v at contact = |v_ego - v_threat| at the first crash-flagged step.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from proxima.trace import Trace  # noqa: E402

DT = 0.05          # env.step cadence (20 Hz)
CAR_LEN = 4.5

TEMPLATE_OF = {
    "ProxLeadDecel": "lead_decel",
    "ProxLeadStopped": "lead_stopped",
    "ProxCutIn": "cut_in",
    "ProxOncomingDrift": "oncoming_drift",
    "ProxCrossingTraffic": "crossing_traffic",
}
ACTOR_TYPE = {t: ("pedestrian" if t == "ped_crossing" else "occupant")
              for t in TEMPLATE_OF.values()}
OVERLAP_W = {"pedestrian": 1.0, "occupant": 1.9}


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    d = z["data"]
    meta = json.loads(str(z["meta"]))
    return {c: d[:, i] for i, c in enumerate(cols)}, meta


def to_trace(path, sid=0, run_idx=0, global_seed=0) -> Trace:
    c, meta = load_npz(path)
    scen_cls = meta["scenario"]
    template = TEMPLATE_OF.get(scen_cls, scen_cls)
    actor = ACTOR_TYPE.get(template, "occupant")
    w_ov = OVERLAP_W[actor]

    hx = np.cos(c["ego_heading"])
    hy = np.sin(c["ego_heading"])
    rx = c["th_x"] - c["ego_x"]
    ry = c["th_y"] - c["ego_y"]
    gap = rx * hx + ry * hy - (CAR_LEN if actor == "occupant"
                               else CAR_LEN / 2)
    y_rel = -rx * hy + ry * hx
    v_e = np.hypot(c["ego_vx"], c["ego_vy"])
    closing = ((c["ego_vx"] - c["th_vx"]) * hx
               + (c["ego_vy"] - c["th_vy"]) * hy)
    a_e = np.gradient(v_e, DT)
    valid_threat = np.isfinite(gap)
    gap = np.where(valid_threat, gap, 1e3)
    y_rel = np.where(valid_threat, y_rel, 1e3)
    closing = np.where(valid_threat, closing, 0.0)

    overlap = np.abs(y_rel) < w_ov
    active = (gap > 0) & (np.abs(y_rel) < w_ov + 0.7) & valid_threat

    crash_idx = np.flatnonzero(c["crash"] > 0.5)
    contact = len(crash_idx) > 0
    if contact:
        i = int(crash_idx[0])
        dv = float(np.hypot(c["ego_vx"][i] - c["th_vx"][i],
                            c["ego_vy"][i] - c["th_vy"][i]))
        if not np.isfinite(dv):
            dv = float(v_e[i])
        t_contact = i * DT
        end = i + 1
    else:
        dv, t_contact, end = 0.0, float("inf"), len(gap)

    params = json.loads(meta["params"]) if meta.get("params") else {}
    return Trace(
        template=template, sid=sid, run_idx=run_idx,
        global_seed=global_seed, params=params, actor_type=actor, dt=DT,
        v_e=v_e[:end], a_e=a_e[:end], gap=gap[:end],
        closing=closing[:end], y_rel=y_rel[:end],
        overlap=overlap[:end], active=active[:end],
        contact=contact, t_contact=t_contact, dv=dv)


# --------------------------------------------------------------- job builder
def nominal_jobs(scenarios, k, global_seed, trace_dir, duration=40):
    """scenarios: list of (scenario_cls, params) with implicit sid order."""
    jobs = []
    for sid, (scls, params) in enumerate(scenarios):
        for r in range(k):
            seed = global_seed + sid * 1000 + r
            jobs.append({
                "job_id": f"s{sid}_r{r}", "scenario_cls": scls,
                "params": params, "sid": sid, "run_idx": r, "seed": seed,
                "duration": duration, "perturb": None,
                "trace_out": os.path.join(trace_dir, f"s{sid}_r{r}.npz")})
    return jobs


def branch_jobs(job, t_star_s, kernel, M, trace_dir, duration=40):
    """Replay jobs for one nominal job: same seed, perturbations from
    t_star. kernel is a proxima.kernel.Kernel; epsilon is sampled HERE so
    the declared kernel object stays the single source of truth."""
    rng = np.random.default_rng([job["seed"], 999])
    eps = kernel.sample(M, rng)
    start_step = int(round(t_star_s / DT))
    out = []
    for m in range(M):
        out.append({
            "job_id": f"{job['job_id']}_b{m}",
            "scenario_cls": job["scenario_cls"], "params": job["params"],
            "sid": job.get("sid", 0), "run_idx": job.get("run_idx", 0),
            "seed": job["seed"], "duration": duration,
            "perturb": {
                "start_step": start_step,
                "dlat_s": float(eps["dlat"][m]),
                "brake_gain": float(eps["fric"][m]),
                "ou_sigma_long": float(eps["ou_sigma_long"]),
                "ou_sigma_lat": float(eps["ou_sigma_lat"]),
                "ou_theta": float(eps["ou_theta"]),
                "seed": int(job["seed"]) * 100 + m,
            },
            "trace_out": os.path.join(trace_dir,
                                      f"{job['job_id']}_b{m}.npz")})
    return out
