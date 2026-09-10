"""Harvest layer for the CARLA stack (TransFuser): runner .npz traces ->
proxima Trace objects + branch jobs. Login-node safe (numpy only).

The CARLA server is async, so recorded rows are NOT uniformly spaced; the
runner logs each row's actual sim time in a trailing `t` column. Because
proxima.trace.Trace assumes a uniform grid (t_crit = index * dt), harvest
RESAMPLES every channel onto a uniform DT grid with np.interp before
building the Trace — downstream (features/tier1/tier2/validation) then
runs unchanged. The crash flag is resampled conservatively: contact time =
first raw crash row's t.

Geometry conventions match the MetaDrive harvest (gap along ego heading
minus CAR_LEN, signed lateral offset, closing speed projected on heading);
the math is frame-agnostic so CARLA's left-handed frame needs no special
casing.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from proxima.trace import Trace  # noqa: E402

DT = 0.05          # uniform resampling grid (20 Hz)
CAR_LEN = 4.7      # CARLA sedan blueprints are slightly longer than 4.5

TEMPLATE_OF = {
    "ProxLeadDecel": "lead_decel",
    "ProxLeadStopped": "lead_stopped",
    "ProxCutIn": "cut_in",
    "ProxOncomingDrift": "oncoming_drift",
    "ProxCrossingTraffic": "crossing_traffic",
}
ACTOR_TYPE = {t: "occupant" for t in TEMPLATE_OF.values()}
OVERLAP_W = {"pedestrian": 1.0, "occupant": 1.9}


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    d = z["data"]
    meta = json.loads(str(z["meta"]))
    return {c: d[:, i] for i, c in enumerate(cols)}, meta


def _resample(c):
    """All channels -> uniform DT grid on the recorded sim-time axis."""
    t = c["t"]
    if len(t) < 2:
        return c, 0.0
    grid = np.arange(t[0], t[-1] + DT / 2, DT)
    out = {}
    for k, v in c.items():
        if k == "crash":
            # nearest-hold: crash is a latched flag, never smooth it
            idx = np.searchsorted(t, grid, side="right") - 1
            out[k] = v[np.clip(idx, 0, len(v) - 1)]
        else:
            out[k] = np.interp(grid, t, v)
    out["t"] = grid
    crash_raw = np.flatnonzero(c["crash"] > 0.5)
    t_crash = float(t[crash_raw[0]] - t[0]) if len(crash_raw) else np.inf
    return out, t_crash


def to_trace(path, sid=0, run_idx=0, global_seed=0) -> Trace:
    raw, meta = load_npz(path)
    scen_cls = meta["scenario"]
    template = TEMPLATE_OF.get(scen_cls, scen_cls)
    actor = ACTOR_TYPE.get(template, "occupant")
    w_ov = OVERLAP_W[actor]
    c, t_crash = _resample(raw)

    hx = np.cos(c["ego_heading"])
    hy = np.sin(c["ego_heading"])
    rx = c["th_x"] - c["ego_x"]
    ry = c["th_y"] - c["ego_y"]
    gap = rx * hx + ry * hy - CAR_LEN
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

    contact = np.isfinite(t_crash)
    if contact:
        i = min(int(round(t_crash / DT)), len(gap) - 1)
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
def nominal_jobs(scenarios, k, global_seed, trace_dir, duration=90):
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


def branch_jobs(job, t_star_s, kernel, M, trace_dir, duration=90,
                M_sample=None):
    """Replay jobs for one nominal job: same seed, perturbations from
    t_star_s (gated in SIM TIME by the runner + gate file — CARLA has no
    fixed step count). kernel epsilon is sampled HERE with the same
    [seed, 999] convention as the MetaDrive adapter.

    M_sample: draw this many eps and use only the first M. Needed for CRN
    against an existing campaign at larger M — sample() draws all normals
    then all uniforms, so the friction stream only aligns with the
    reference draws when the sample SIZE matches the reference campaign's."""
    rng = np.random.default_rng([job["seed"], 999])
    eps = kernel.sample(M_sample or M, rng)
    out = []
    for m in range(M):
        out.append({
            "job_id": f"{job['job_id']}_b{m}",
            "scenario_cls": job["scenario_cls"], "params": job["params"],
            "sid": job.get("sid", 0), "run_idx": job.get("run_idx", 0),
            "seed": job["seed"], "duration": duration,
            "perturb": {
                "t_star_s": float(t_star_s),
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
