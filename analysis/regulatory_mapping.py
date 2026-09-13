"""Where the campaign's scenario parameters sit inside the regulators' grids.

The templates are described in the paper as "NHTSA-style pre-crash templates"
whose parameter ranges were widened in pilot rounds until outcomes spanned the
spectrum. Read uncharitably that is a tuned scenario suite. This script answers
the charge with arithmetic instead of prose: it takes the frozen campaign
configuration and asks, per template, what fraction of the sampled scenarios
falls inside the parameter box that a published regulation or consumer-test
protocol defines for the same manoeuvre -- and, symmetrically, which of the
regulator's own test points the sampled box contains.

Reference grids (quoted, with sources):

UN R157 (ALKS), Annex 3, as implemented by ika RWTH Aachen's `alks-scenarios`
generator (github.com/ika-rwth-aachen/alks-scenarios), which recreates the
regulation's Annex 3 pp. 45-56 figures:
  cut-in        ego speed ve0 in {20,30,40,50,60} km/h, paired with relative
                speed dv0 in {0,10,20,30,40} km/h (14 published pairings);
                lateral velocity vy in (0, 3.0] m/s, step 0.1;
                longitudinal distance dx0 in [0, 60] m, step 1; lane width 3.5 m
  deceleration  ego speed ve0 in [10, 60] km/h, step 1; lead deceleration
                gx in [0, 1] g, step 0.05 (0 to 9.81 m/s^2); initial time
                headway THW0 = 2 s

Euro NCAP AEB Car-to-Car test protocol v4.3 (December 2023), section 8.2:
  CCRs   stationary lead; VUT 10-50 km/h for AEB (55-80 FCW), 5 km/h steps;
         overlap -50% to 50% in 25% steps
  CCRm   moving lead; VUT 30-80 km/h, 5 km/h steps; overlap -50% to 50%
  CCRb   braking lead; VUT = GVT = 50 km/h; decelerations {2, 6} m/s^2;
         headways {12, 40} m
  CCCscp crossing straight path; VUT {stop,20,30,40,50,60} km/h x
         GVT {20,30,40,50,60} km/h

Nothing here is simulated. Ego speed per scenario is the peak speed of the
recorded reference execution (run 0), shipped as the `ego_vmax_ms` column of
data/<subject>/nominal_runs.csv; the rest comes from the frozen scenario
configuration data/<subject>/campaign_scenarios.json.

Paper (Sec. 4.2): the openpilot campaign's stopped-lead scenarios sit 93%
inside the Euro NCAP CCRs speed range and its cut-ins entirely inside
UN R157's lateral-velocity and distance boxes, while TransFuser's suite,
calibrated to that stack's 15 km/h cruise, sits below grids that start at
20 to 50 km/h.

Usage:  python analysis/regulatory_mapping.py [--out results/regulatory/regulatory_mapping.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402

LANE_W = 3.5           # both simulators' template lane width
KMH = 3.6

SUBJECTS = C.SUBJECTS
LABEL = C.LABEL

# --- regulator grids -------------------------------------------------------
R157_CUTIN_PAIRS = [(60, 0), (60, 20), (60, 30), (60, 40),
                    (50, 10), (50, 20), (50, 30), (50, 40),
                    (40, 10), (40, 20), (40, 30),
                    (30, 10), (30, 20), (20, 10)]      # (ve0, dv0) km/h
R157_CUTIN_VY = (0.1, 3.0)                             # m/s
R157_CUTIN_DX0 = (0.0, 60.0)                           # m
R157_DECEL_VE0 = (10.0, 60.0)                          # km/h
R157_DECEL_A = (0.0, 9.81)                             # m/s^2 (0 to 1 g)
R157_DECEL_THW0 = 2.0                                  # s

NCAP_CCRS_AEB = np.arange(10, 51, 5, dtype=float)      # km/h
NCAP_CCRM = np.arange(30, 81, 5, dtype=float)
NCAP_CCRB_V = 50.0
NCAP_CCRB_A = [2.0, 6.0]
NCAP_CCRB_HEADWAY = [12.0, 40.0]
NCAP_CCCSCP_VUT = [20.0, 30.0, 40.0, 50.0, 60.0]       # plus start-from-stop
NCAP_CCCSCP_GVT = [20.0, 30.0, 40.0, 50.0, 60.0]

TEMPLATE = {"ProxLeadDecel": "lead_decel", "ProxLeadStopped": "lead_stopped",
            "ProxCutIn": "cut_in", "ProxOncomingDrift": "oncoming_drift",
            "ProxCrossingTraffic": "crossing_vehicle"}


def ego_speeds(subject):
    """Peak ego speed of each reference execution (run 0), keyed by scenario id."""
    return {int(r["sid"]): float(r["ego_vmax_ms"])
            for r in C.read_csv(C.data_path(subject, "nominal_runs.csv")) if r["run"] == 0}


def frac_in(vals, lo, hi):
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    return float(((v[ok] >= lo) & (v[ok] <= hi)).mean())


def span(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return [float(v.min()), float(v.max())] if len(v) else [np.nan, np.nan]


def covered_points(points, lo, hi):
    """Which of the regulator's own test points fall in the sampled box."""
    return [float(p) for p in points if lo <= p <= hi]


def analyse(subject):
    scen = json.load(open(C.data_path(subject, "campaign_scenarios.json")))
    v_ego = ego_speeds(subject)
    by = collections.defaultdict(list)
    for sid, x in enumerate(scen):
        by[TEMPLATE.get(x["scls"], x["scls"])].append((sid, x["params"]))

    res = {}

    # ---- lead_decel vs UN R157 deceleration test and Euro NCAP CCRb -------
    rows = by["lead_decel"]
    ve = np.array([v_ego[s] * KMH for s, _ in rows])
    a_lead = np.array([p["a_lead"] for _, p in rows])
    gap0 = np.array([p["gap0"] for _, p in rows])
    g_trig = np.array([p["g_trig"] for _, p in rows])
    v_lead = np.array([p["v_lead"] * KMH for _, p in rows])
    thw0 = gap0 / np.maximum(np.array([v_ego[s] for s, _ in rows]), 1e-6)
    thw_trig = g_trig / np.maximum(np.array([v_ego[s] for s, _ in rows]), 1e-6)
    res["lead_decel"] = {
        "n": len(rows),
        "UN R157 Annex 3 deceleration": {
            "ego speed [km/h]": {
                "grid": list(R157_DECEL_VE0), "sampled": span(ve),
                "frac_inside": frac_in(ve, *R157_DECEL_VE0)},
            "lead deceleration [m/s^2]": {
                "grid": list(R157_DECEL_A), "sampled": span(a_lead),
                "frac_inside": frac_in(a_lead, *R157_DECEL_A)},
            "initial headway THW0 [s]": {
                "grid": [R157_DECEL_THW0], "sampled": span(thw0),
                "frac_inside": frac_in(thw0, 0, R157_DECEL_THW0),
                "note": "R157 fixes THW0 = 2 s; the campaign samples a range, "
                        "and the fraction reported is the share at or below it",
                "sampled_trigger_headway": span(thw_trig)},
        },
        "Euro NCAP CCRb": {
            "VUT speed [km/h]": {"grid": [NCAP_CCRB_V], "sampled": span(ve)},
            "GVT speed [km/h]": {"grid": [NCAP_CCRB_V], "sampled": span(v_lead)},
            "deceleration [m/s^2]": {
                "grid": NCAP_CCRB_A, "sampled": span(a_lead),
                "points_covered": covered_points(NCAP_CCRB_A, *span(a_lead))},
            "headway [m]": {
                "grid": NCAP_CCRB_HEADWAY, "sampled": span(g_trig),
                "points_covered": covered_points(NCAP_CCRB_HEADWAY,
                                                 *span(g_trig))},
        },
    }

    # ---- lead_stopped vs Euro NCAP CCRs ----------------------------------
    rows = by["lead_stopped"]
    ve = np.array([v_ego[s] * KMH for s, _ in rows])
    d0 = np.array([p["d0"] for _, p in rows])
    res["lead_stopped"] = {
        "n": len(rows),
        "Euro NCAP CCRs (AEB)": {
            "VUT speed [km/h]": {
                "grid": [float(NCAP_CCRS_AEB.min()),
                         float(NCAP_CCRS_AEB.max())],
                "sampled": span(ve),
                "frac_inside": frac_in(ve, NCAP_CCRS_AEB.min(),
                                       NCAP_CCRS_AEB.max()),
                "grid_points_covered": covered_points(NCAP_CCRS_AEB,
                                                      *span(ve))},
            "overlap [%]": {"grid": [-50.0, 50.0], "sampled": [100.0, 100.0],
                            "note": "the templates use full overlap only; "
                                    "partial overlap is not varied"},
            "initial distance [m]": {"grid": None, "sampled": span(d0),
                                     "note": "not a protocol parameter"},
        },
    }

    # ---- cut_in vs UN R157 cut-in ----------------------------------------
    rows = by["cut_in"]
    ve = np.array([v_ego[s] * KMH for s, _ in rows])
    v_cut = np.array([p["v_cut"] * KMH for _, p in rows])
    dv0 = ve - v_cut
    vy = LANE_W / np.array([p["t_lat"] for _, p in rows])
    dx0 = np.array([p["gap0"] for _, p in rows])
    pair_ve = np.array([p[0] for p in R157_CUTIN_PAIRS], dtype=float)
    pair_dv = np.array([p[1] for p in R157_CUTIN_PAIRS], dtype=float)
    lo_ve, hi_ve = span(ve)
    lo_dv, hi_dv = span(dv0)
    pairs_covered = [[float(a), float(b)]
                     for a, b in zip(pair_ve, pair_dv)
                     if lo_ve <= a <= hi_ve and lo_dv <= b <= hi_dv]
    res["cut_in"] = {
        "n": len(rows),
        "UN R157 Annex 3 cut-in": {
            "ego speed ve0 [km/h]": {
                "grid": [float(pair_ve.min()), float(pair_ve.max())],
                "sampled": span(ve),
                "frac_inside": frac_in(ve, pair_ve.min(), pair_ve.max())},
            "relative speed dv0 [km/h]": {
                "grid": [float(pair_dv.min()), float(pair_dv.max())],
                "sampled": span(dv0),
                "frac_inside": frac_in(dv0, pair_dv.min(), pair_dv.max())},
            "lateral velocity vy [m/s]": {
                "grid": list(R157_CUTIN_VY), "sampled": span(vy),
                "frac_inside": frac_in(vy, *R157_CUTIN_VY),
                "note": "vy = lane width / lateral transition time"},
            "longitudinal distance dx0 [m]": {
                "grid": list(R157_CUTIN_DX0), "sampled": span(dx0),
                "frac_inside": frac_in(dx0, *R157_CUTIN_DX0)},
            "regulation (ve0, dv0) pairs inside the sampled box":
                {"n_covered": len(pairs_covered), "n_total":
                 len(R157_CUTIN_PAIRS), "pairs": pairs_covered},
        },
    }

    # ---- crossing_vehicle vs Euro NCAP CCCscp ----------------------------
    rows = by["crossing_vehicle"]
    ve = np.array([v_ego[s] * KMH for s, _ in rows])
    v_cross = np.array([p["v_cross"] * KMH for _, p in rows])
    res["crossing_vehicle"] = {
        "n": len(rows),
        "Euro NCAP CCCscp": {
            "VUT speed [km/h]": {
                "grid": NCAP_CCCSCP_VUT, "sampled": span(ve),
                "grid_points_covered": covered_points(NCAP_CCCSCP_VUT,
                                                      *span(ve))},
            "GVT speed [km/h]": {
                "grid": NCAP_CCCSCP_GVT, "sampled": span(v_cross),
                "grid_points_covered": covered_points(NCAP_CCCSCP_GVT,
                                                      *span(v_cross))},
        },
    }

    # ---- oncoming_drift: no scored regulatory grid -----------------------
    rows = by["oncoming_drift"]
    ve = np.array([v_ego[s] * KMH for s, _ in rows])
    v_onc = np.array([p["v_onc"] * KMH for _, p in rows])
    vy = np.array([p["vy_drift"] for _, p in rows])
    res["oncoming_drift"] = {
        "n": len(rows),
        "Euro NCAP CCFhos": {
            "VUT speed [km/h]": {"grid": None, "sampled": span(ve)},
            "GVT speed [km/h]": {"grid": None, "sampled": span(v_onc)},
            "drift lateral velocity [m/s]": {"grid": None, "sampled": span(vy)},
            "note": "v4.3 defines the head-on straight scenario but scores no "
                    "speed grid for it; manufacturers report behaviour only, "
                    "so there is no box to fall inside",
        },
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.results_path("regulatory", "regulatory_mapping.json"))
    a = ap.parse_args()
    report = {"sources": {
        "UN R157": "UN Regulation No. 157 (ALKS), Annex 3 pp. 45-56, as "
                   "implemented by ika RWTH Aachen alks-scenarios",
        "Euro NCAP": "Euro NCAP AEB Car-to-Car Test Protocol v4.3, Dec 2023, "
                     "section 8.2"},
        "subjects": {}}
    for s in SUBJECTS:
        report["subjects"][LABEL[s]] = analyse(s)

    dest = a.out
    json.dump(report, open(dest, "w"), indent=1, default=float)
    print(f"wrote {dest}\n")

    for lab, d in report["subjects"].items():
        print(f"================ {lab}")
        for template, blocks in d.items():
            print(f"\n  {template}  (n={blocks['n']})")
            for reg, params in blocks.items():
                if reg == "n":
                    continue
                print(f"    {reg}")
                for pname, p in params.items():
                    if not isinstance(p, dict):
                        continue
                    g = p.get("grid")
                    sm = p.get("sampled")
                    gs = ("-" if g is None else
                          f"[{min(g):g}, {max(g):g}]" if len(g) > 1
                          else f"{{{g[0]:g}}}")
                    ss = ("-" if sm is None else
                          f"[{sm[0]:.1f}, {sm[1]:.1f}]")
                    fi = p.get("frac_inside")
                    cov = p.get("grid_points_covered", p.get("points_covered"))
                    extra = ""
                    if fi is not None and np.isfinite(fi):
                        extra += f"  inside={100*fi:.0f}%"
                    if cov is not None:
                        extra += f"  covers={cov}"
                    print(f"      {pname:32s} reg={gs:16s} "
                          f"sampled={ss:18s}{extra}")
                    if "n_covered" in p:
                        print(f"      {'':32s} regulation pairs covered: "
                              f"{p['n_covered']}/{p['n_total']}")


    print("\nheadline (paper Sec. 4.2):")
    for lab, d in report["subjects"].items():
        ls = d["lead_stopped"]["Euro NCAP CCRs (AEB)"]["VUT speed [km/h]"]
        ci = d["cut_in"]["UN R157 Annex 3 cut-in"]
        print(f"  {lab}: stopped-lead ego speed inside CCRs range {100*ls['frac_inside']:.0f}% "
              f"(sampled {ls['sampled'][0]:.1f}-{ls['sampled'][1]:.1f} km/h); cut-in lateral velocity "
              f"inside R157 {100*ci['lateral velocity vy [m/s]']['frac_inside']:.0f}%, distance inside "
              f"{100*ci['longitudinal distance dx0 [m]']['frac_inside']:.0f}%")
    print("  paper: openpilot 93% inside CCRs; cut-ins entirely inside the lateral-velocity and distance boxes")


if __name__ == "__main__":
    main()
