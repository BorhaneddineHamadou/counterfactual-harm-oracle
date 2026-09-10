"""Scenario-range calibration for openpilot+MetaDrive (paper Sec. 4.2): LHS sweep over the 5 MetaDrive templates
to tune parameter ranges for a healthy crash/near-miss spectrum (the
real-stack analogue of the pilot's texture tuning).

generate : 8 LHS scenarios per template x 1 seed per round.
analyze  : per-template conflict texture (outcome, min gap, TTC_min, speed
           at criticality, Delta-v) + flags for degenerate cases.

Rounds 2-3 were generated with ad-hoc range edits (tightened ranges live in
configs/phase1_calibrated_ranges.json); round 4 pushes the three templates
still lacking a crash tail, using the drivers identified from rounds 1-3:
LeadDecel tail needs a_lead >= 4.7; OncomingDrift needs the drift to complete
before the pass (vy_drift/y_frac up, t_stay long, d_trig matched to closing
time); CrossingTraffic near-misses cluster at d_place <= 100, v_cross >= 2.

Round 6 (2026-08-15) re-tunes LeadDecel and CutIn for the SEEING SUT. Rounds
1-5 were calibrated against the blind SUT, whose ego crawls at ~5 m/s; the
seeing ego cruises at 12.4 m/s and, because rendering the threat slows the
simulator, its run window is only ~7.7 s of sim after engagement (~60 m of
ego travel, 20.3 Hz trace). Under those ranges the ego-threat gap never
reaches the trigger distance -- the A/B of 2026-08-15 armed LeadDecel in
1/11 runs and CutIn in 2/9 -- so the staged conflict simply never happens.
The round-6 boxes were designed against the 119/117 measured launch profiles
(see the A/B session): they arm in ~94% of draws with the ego at ~11 m/s,
~2.5 s of run left, and a required-avoidance-deceleration spectrum spanning
3-5.5 m/s^2 (a third of draws above 5). Round-6 runs MUST set
PROXIMA_RENDER_THREATS=1 (proxima_phase1.slurm now does) or they recalibrate
the blind world again.

Usage: python3 campaign/openpilot_calibrate.py generate|analyze [round]
The adopted ranges are configs/openpilot_scenario_ranges.json.
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
OUT = os.environ.get("CH_CALIB", os.path.join(BASE, "data", "openpilot", "calibration"))
DURATION = 130
SEED = 20260722

RANGES_R1 = {
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

RANGES_R4 = {
    "ProxLeadDecel": {"gap0": (25, 50), "v_lead": (5, 8),
                      "g_trig": (18, 34), "a_lead": (4.5, 8)},
    "ProxOncomingDrift": {"r0": (100, 160), "v_onc": (8, 14),
                          "d_trig": (35, 70), "vy_drift": (0.9, 1.8),
                          "y_frac": (0.7, 1.15), "t_stay": (1.5, 3.5)},
    "ProxCrossingTraffic": {"d_place": (80, 110), "d_trig": (16, 30),
                            "v_cross": (1.8, 3.0), "y0": (6, 8)},
}

RANGES_R5 = {
    "ProxOncomingDrift": {"r0": (100, 160), "v_onc": (10, 16),
                          "d_trig": (22, 45), "vy_drift": (1.2, 2.2),
                          "y_frac": (0.85, 1.25), "t_stay": (2.0, 4.0)},
}

RANGES_R6 = {
    "ProxLeadDecel": {"gap0": (14, 27), "v_lead": (3, 6),
                      "g_trig": (9, 18), "a_lead": (5, 8)},
    "ProxCutIn": {"gap0": (12, 26), "v_cut": (3, 6), "d_trig": (10, 20),
                  "t_lat": (0.8, 1.6), "a_cut": (0, 3)},
}

RANGES_R7 = {
    # Round 6 measured what round 6's model could not: openpilot SEES the
    # lead, so (i) a trigger set above the closing gap never fires at all
    # (g_trig 16.6 with the gap bottoming at 19.3), (ii) firing at 10-11 m
    # with a lead that stops in ~0.6 s (a_lead 5-8) is unavoidable by
    # construction -- all 3 fires were 12 m/s rear-ends, dv 11.7-12.8, no
    # graded band in between; and (iii) a threat spawned close inhibits the
    # launch from standstill (both dead egos had gap0 <= 18; across the
    # seeing campaign the effect is strong for CutIn, gap0 18.7 dead vs
    # 31.9 alive, p<0.001, weak for LeadDecel p=0.11).
    # So: spawn far enough to launch, close slowly enough to reach the
    # trigger, and SOFTEN the manoeuvre so the ego has a chance.
    "ProxLeadDecel": {"gap0": (20, 28), "v_lead": (3.0, 4.5),
                      "g_trig": (11, 16), "a_lead": (2.0, 4.5)},
    "ProxCutIn": {"gap0": (20, 30), "v_cut": (3.5, 6.0), "d_trig": (14, 24),
                  "t_lat": (1.6, 2.6), "a_cut": (0, 1.5)},
}

# NOTE (2026-08-16): the R8 CutIn box below was adopted into
# configs/phase1_calibrated_ranges.json and then REVERTED. Its apparent
# health was an artefact: at the time, ~half of every CutIn draw spawned the
# threat in the ego's own lane (the lane_shift clamp, fixed since), so the
# "texture" was largely a slow-lead scenario. With the lane fix in place the
# same 20 scenarios start the slide in 8/20 runs and COMPLETE it in 0 -- the
# trigger is crossed at t~7-8 s with under a second of run left. Do not
# re-adopt without a round that shows the lane change finishing.
RANGES_R8 = {
    # Round 7 fired reliably but at 11-15 m, which is inside the unavoidable
    # band: 5/7 fires were 12 m/s rear-ends and the 2 "near misses" only
    # survived because the run ended first (both fired at 6.6-6.8 s with
    # <0.9 s left). The gap at firing IS g_trig, so the fix is simply to
    # trigger further out: at ~20 m a 12.5 m/s ego needs ~3.4 m/s^2 to stop
    # behind a lead that halts, which openpilot can just about deliver.
    # Round 6's non-fires at g_trig 16.6/17.9 were not evidence against
    # this -- three of those four runs had a dead or aborted ego and the
    # fourth had v_lead 5.9, too fast for the gap to close at all.
    "ProxLeadDecel": {"gap0": (24, 30), "v_lead": (3.0, 4.5),
                      "g_trig": (18, 22), "a_lead": (2.0, 4.0)},
    # CutIn crashed even when triggered at 23 m, where matching the threat's
    # speed needs only ~1.2 m/s^2 -- openpilot reacts to the lane change far
    # too late (a real SUT weakness, and useful for the oracle, but it makes
    # the template crash-saturated at any trigger distance). The lever left
    # is the speed differential itself: raise v_cut, keeping it low enough
    # that the gap still closes inside the ~7.7 s window.
    "ProxCutIn": {"gap0": (24, 30), "v_cut": (5.5, 7.5), "d_trig": (18, 26),
                  "t_lat": (1.6, 2.6), "a_cut": (0, 1.5)},
}

RANGES_R9 = {
    # LeadDecel only: round 8's CutIn box is adopted as-is (1 crash / 4
    # near-misses / 4 benign out of 9 readable runs).
    # Round 8 put 7/12 LeadDecel runs in the launch-inhibition basin. The
    # cause is NOT a slow lead (v_lead does not separate dead from alive in
    # the campaign, p=0.83, nor monotonically across rounds 6-8) but the
    # spawn distance: campaign dead rate is 17-20% for gap0 < 35 and 4-9%
    # for gap0 >= 35. The closing model gap0 + 7.7*v_lead - 60 predicts the
    # observed minimum ego-threat distance on campaign runs at r=0.83, so
    # the box below satisfies both constraints at once: gap0 >= 35 to launch,
    # gap0 + 7.7*v_lead <= 80 so the gap still crosses the trigger with time
    # to spare. g_trig moves out to 22-27 m -- round 8 fired at 18-21 m and
    # rear-ended every time (dv 11.9-12.8), so the graded band, if it exists
    # at all for this SUT, is further out.
    "ProxLeadDecel": {"gap0": (35, 42), "v_lead": (3.8, 5.0),
                      "g_trig": (22, 27), "a_lead": (2.0, 4.0)},
}

ROUNDS = {1: RANGES_R1, 4: RANGES_R4, 5: RANGES_R5, 6: RANGES_R6,
          7: RANGES_R7, 8: RANGES_R8, 9: RANGES_R9}
# rounds 7-8 run 12 draws/template: at 8, half the sample was lost to dead or
# watchdog-aborted egos and only 4 LeadDecel runs were readable
N_PER_ROUND = {7: 12, 8: 12, 9: 12}
N_PER = 8


def _round_paths(rnd):
    tag = "" if rnd == 1 else str(rnd)
    return (os.path.join(OUT, f"traces_calib{tag}"),
            os.path.join(OUT, f"calib{tag}_jobs.jsonl"),
            "cal" if rnd == 1 else f"cal{rnd}")


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
    n_per = N_PER_ROUND.get(rnd, N_PER)
    jobs = []
    for ti, (scls, ranges) in enumerate(ranges_by_cls.items()):
        for j, params in enumerate(
                lhs_params(ranges, n_per, SEED + 100 * rnd + ti)):
            jobs.append({"job_id": f"{prefix}_{scls}_{j}",
                         "scenario_cls": scls,
                         "params": params, "seed": j, "duration": DURATION,
                         "perturb": None,
                         "trace_out": f"{tr}/{scls}_{j}.npz"})
    with open(jobs_file, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs -> {jobs_file}")
    print("submit:\n  JOBS_FILE=" + jobs_file + " sbatch --array=0-3 "
          os.path.join(BASE, "adapters", "metadrive_openpilot", "calibrate.slurm"))


def analyze(rnd):
    from adapters.metadrive_openpilot.harvest import to_trace
    from proxima.features import extract
    trdir, _, _ = _round_paths(rnd)
    rows = []
    for scls in ROUNDS[rnd]:
        for j in range(N_PER_ROUND.get(rnd, N_PER)):
            p = f"{trdir}/{scls}_{j}.npz"
            if not os.path.exists(p):
                continue
            tr = to_trace(p, sid=j)
            f = extract(tr)
            i = tr.crit_index()
            rows.append({"scls": scls, "j": j, "contact": tr.contact,
                         "dv": round(tr.dv, 1),
                         "min_gap": round(float(tr.gap.min()), 1),
                         "ttc_min": round(float(f[0]), 2),
                         "v_crit": round(float(tr.v_e[i]), 1),
                         "threat_seen": bool((tr.gap < 900).any()),
                         "steps": tr.T})
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
