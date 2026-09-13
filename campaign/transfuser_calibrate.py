"""Scenario-range calibration for TransFuser+CARLA (paper Sec. 4.2): LHS sweep over the 5 CARLA
templates to tune ranges for a healthy crash/near-miss spectrum — the
carla_transfuser analogue of campaign/openpilot_calibrate.py.

Ranges are scaled to TransFuser's measured ~4 m/s cruise:
threats must be slower/closer than the MetaDrive-suite mid-spectrum values or
the interaction never triggers inside the 40 s window.

generate : 8 LHS scenarios per template per round.
analyze  : per-template conflict texture (contact, min gap, TTC_min, speed
           at criticality, Delta-v) + degenerate-case flags.

Usage: python3 campaign/transfuser_calibrate.py generate|analyze [round]
The adopted ranges are configs/transfuser_scenario_ranges.json.
Submit: JOBS_FILE=<printed> sbatch --array=0-3 \
        adapters/carla_transfuser/calib.slurm
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
OUT = os.environ.get("CH_CALIB", os.path.join(BASE, "data", "transfuser", "calibration"))
DURATION = 40
SEED = 20260805

RANGES_R1 = {
    "ProxLeadDecel": {"gap0": (15, 40), "v_lead": (2, 5),
                      "g_trig": (10, 30), "a_lead": (3, 8)},
    "ProxLeadStopped": {"d0": (12, 70)},
    "ProxCutIn": {"gap0": (8, 25), "v_cut": (1, 4), "d_trig": (8, 20),
                  "t_lat": (0.8, 2.5), "a_cut": (0, 3)},
    "ProxOncomingDrift": {"r0": (60, 120), "v_onc": (4, 10),
                          "d_trig": (20, 60), "vy_drift": (0.8, 2.0),
                          "y_frac": (0.85, 1.25), "t_stay": (2.0, 4.0)},
    "ProxCrossingTraffic": {"d_place": (30, 80), "d_trig": (10, 35),
                            "v_cross": (1.0, 3.0), "y0": (6, 8)},
}

# R2 from the r1 texture: LeadDecel lacked a crash tail (0/8, best 1.1 m)
# -> harder stops; OncomingDrift was 7/8 crash -> widen toward gentle
# (earlier drift, lower peak, shorter stay); Crossing 1/8 crash w/ many
# behind-passes -> center the trigger timing. CutIn healthy (keep R1);
# LeadStopped degenerate-safe (all stops 0.9-1.6 m; bounded template).
RANGES_R2 = {
    "ProxLeadDecel": {"gap0": (12, 25), "v_lead": (2, 4),
                      "g_trig": (8, 18), "a_lead": (6, 9)},
    "ProxOncomingDrift": {"r0": (70, 120), "v_onc": (4, 8),
                          "d_trig": (30, 70), "vy_drift": (1.0, 2.2),
                          "y_frac": (0.5, 1.1), "t_stay": (1.0, 2.5)},
    "ProxCrossingTraffic": {"d_place": (40, 70), "d_trig": (18, 32),
                            "v_cross": (1.5, 3.0), "y0": (6, 8)},
}

# R3: OncomingDrift interpolates r1 (7/8 crash) and r2 (0/8, razor
# near-misses) — y_frac is the sensitive driver; Crossing centers the
# timing window (r2: 6/8 behind-passes); LeadDecel gets one extreme push
# (trigger inside the ~2-3 m braking margin) before declaring the crash
# tail unreachable at TransFuser's 4 m/s cruise.
RANGES_R3 = {
    "ProxLeadDecel": {"gap0": (8, 16), "v_lead": (2, 3.5),
                      "g_trig": (3, 9), "a_lead": (7, 9)},
    "ProxOncomingDrift": {"r0": (70, 120), "v_onc": (4, 8),
                          "d_trig": (25, 50), "vy_drift": (1.0, 2.0),
                          "y_frac": (0.8, 1.2), "t_stay": (1.5, 3.0)},
    "ProxCrossingTraffic": {"d_place": (40, 60), "d_trig": (20, 30),
                            "v_cross": (1.6, 2.6), "y0": (6, 7.5)},
}

ROUNDS = {1: RANGES_R1, 2: RANGES_R2, 3: RANGES_R3}
N_PER = 8


def _round_paths(rnd):
    tag = "" if rnd == 1 else str(rnd)
    return (os.path.join(OUT, f"traces_calib{tag}"),
            os.path.join(OUT, f"calib{tag}_jobs.jsonl"),
            "tfcal" if rnd == 1 else f"tfcal{rnd}")


def lhs_params(ranges, n, seed):
    from scipy.stats import qmc
    names = list(ranges)
    u = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    lo = np.array([ranges[k][0] for k in names])
    hi = np.array([ranges[k][1] for k in names])
    vals = lo + u * (hi - lo)
    return [dict(zip(names, map(float, row))) for row in vals]


def generate(rnd):
    ranges_by_cls = ROUNDS[rnd]
    tr, jobs_file, prefix = _round_paths(rnd)
    os.makedirs(tr, exist_ok=True)
    jobs = []
    for ti, (scls, ranges) in enumerate(ranges_by_cls.items()):
        for j, params in enumerate(
                lhs_params(ranges, N_PER, SEED + 100 * rnd + ti)):
            jobs.append({"job_id": f"{prefix}_{scls}_{j}",
                         "scenario_cls": scls, "params": params,
                         "sid": ti * N_PER + j, "run_idx": 0,
                         "seed": j, "duration": DURATION, "perturb": None,
                         "trace_out": f"{tr}/{scls}_{j}.npz"})
    with open(jobs_file, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs -> {jobs_file}")
    print("submit:\n  JOBS_FILE=" + jobs_file + " sbatch --array=0-3 " +
          os.path.join(BASE, "adapters", "carla_transfuser", "calib.slurm"))


def analyze(rnd):
    from adapters.carla_transfuser.harvest import to_trace
    from proxima.features import extract
    trdir, _, _ = _round_paths(rnd)
    rows = []
    for scls in ROUNDS[rnd]:
        for j in range(N_PER):
            p = f"{trdir}/{scls}_{j}.npz"
            if not os.path.exists(p):
                continue
            tr = to_trace(p, sid=j)
            f = extract(tr)
            i = tr.crit_index()
            rows.append({"scls": scls, "j": j, "contact": bool(tr.contact),
                         "dv": round(float(tr.dv), 1),
                         "min_gap": round(float(tr.gap.min()), 1),
                         "ttc_min": round(float(f[0]), 2),
                         "v_crit": round(float(tr.v_e[i]), 1),
                         "threat_seen": bool((tr.gap < 900).any()),
                         "steps": int(tr.T)})
    for scls in ROUNDS[rnd]:
        sub = [r for r in rows if r["scls"] == scls]
        if not sub:
            print(f"{scls}: no traces yet")
            continue
        nc = sum(r["contact"] for r in sub)
        seen = sum(r["threat_seen"] for r in sub)
        print(f"\n=== {scls}: n={len(sub)} crash={nc} threat_seen={seen} ===")
        for r in sub:
            print(f"  j={r['j']} contact={r['contact']} dv={r['dv']:5.1f} "
                  f"min_gap={r['min_gap']:6.1f} ttc={r['ttc_min']:5.2f} "
                  f"v@crit={r['v_crit']:4.1f} steps={r['steps']}")
    rep = os.path.join(OUT, f"calib_report_r{rnd}.json")
    json.dump(rows, open(rep, "w"), indent=2)
    print(f"\nreport -> {rep}")


if __name__ == "__main__":
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"generate": generate, "analyze": analyze}[sys.argv[1]](rnd)
