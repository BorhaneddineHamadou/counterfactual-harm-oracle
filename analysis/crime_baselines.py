"""Third-party criticality baselines: 35 CommonRoad-CriMe measures vs harm.

Every implemented measure of CommonRoad-CriMe 0.4.5 (Lin and Althoff, IV
2023) was computed on the same reference executions the Tier-1 oracle
labelled (campaign/run_crime_baselines.py, which needs the toolbox and the
raw traces). Their values are shipped in data/crime/<subject>_measures.json,
two foldings per measure:

  <M>_worst   reduced over the execution in the direction the toolbox
              declares (POS -> max, NEG -> min), the trace-level score;
  <M>_crit    evaluated at the criticality instant t_crit.

This script needs neither toolbox nor traces. It turns each measure into a
criticality score by its declared direction (data/crime/measure_monotone.json,
NEG measures negated), and reports per measure and folding the Spearman
correlation with Tier-1 harm over all 150 references (r_s), over the
non-collision references only (r_s^nc), the share of reference pairs it ties,
and the harm surfaced by its top 10%. Our own scalars and the stored
out-of-fold field tier are scored the same way.

Paper (Table 1 row "Best of 35 CriMe measures", Sec. 5.2): best on openpilot
is CPI at 0.65 (inverting to -0.23 on TransFuser), best on TransFuser a
worst-case TTC variant (WTTC) at 0.67 (0.44 on openpilot); on the
non-collision slice their best falls to 0.48 and 0.53; each measure is
reported at whichever folding orders harm better on that subject.

Usage:  python analysis/crime_baselines.py [--top 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima.metrics import surfaced                  # noqa: E402

OURS = {"ttc_min": -1, "min_clearance": -1, "speed_at_crit": +1,
        "dv_realized": +1, "peak_jerk": +1, "peak_lat_rate": +1}


def finite_rank_input(v):
    """+-inf are real answers of a NEG measure ('never in conflict'); map
    them to finite extremes that keep their rank."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return v
    lo, hi = np.min(v[ok]), np.max(v[ok])
    pad = max(1.0, abs(hi - lo))
    out = v.copy()
    out[np.isposinf(v)] = hi + pad
    out[np.isneginf(v)] = lo - pad
    return out


def tie_fraction(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    i, j = np.triu_indices(len(v), 1)
    return float(np.mean(v[i] == v[j]))


def evaluate(raw, sign, H, NC):
    v = finite_rank_input(raw) * sign
    ok = np.isfinite(v)
    res = {"valid": int(ok.sum()), "tie": tie_fraction(raw)}
    if ok.sum() < 5 or np.std(v[ok]) == 0:
        res.update(rho=np.nan, rho_nc=np.nan, s10=np.nan)
        return res
    res["rho"] = float(spearmanr(v[ok], H[ok]).statistic)
    m = ok & NC
    res["rho_nc"] = (float(spearmanr(v[m], H[m]).statistic)
                     if m.sum() > 5 and np.std(v[m]) > 0 else np.nan)
    res["s10"] = float(surfaced(v[ok], H[ok], 0.10))
    return res


def run(subject):
    S = C.load_subject(subject)
    H, NC, sids = S["y"], S["NC"], S["sids"]
    rows = {r["sid"]: r for r in json.load(open(os.path.join(C.DATA, "crime", f"{subject}_measures.json")))
            if "sid" in r}
    mono = json.load(open(os.path.join(C.DATA, "crime", "measure_monotone.json")))["monotone"]
    names = sorted({k.rsplit("_", 1)[0] for r in rows.values() for k in r if k.endswith("_worst")})
    out = {"n_references": int(len(sids)), "n_noncollision": int(NC.sum()),
           "n_third_party_measures": len(names), "measures": {}}
    for name in names:
        sign = +1 if mono.get(name, "POS") == "POS" else -1
        for variant in ("worst", "crit"):
            raw = np.array([rows[int(s)].get(f"{name}_{variant}", np.nan) if int(s) in rows
                            else np.nan for s in sids], float)
            r = evaluate(raw, sign, H, NC)
            r.update(source="CommonRoad-CriMe 0.4.5", monotone="POS" if sign > 0 else "NEG")
            out["measures"][f"{name} ({variant})"] = r
    for fname, sign in OURS.items():
        r = evaluate(S["X41"][:, S["names41"].index(fname)], sign, H, NC)
        r.update(source="this paper", monotone="POS" if sign > 0 else "NEG")
        out["measures"][f"{fname} (ours)"] = r
    r = evaluate((~NC).astype(float), +1, H, NC)
    r.update(source="this paper", monotone="POS")
    out["measures"]["binary verdict (ours)"] = r
    oof = C.load_oof(subject)["oofs"]
    per = [evaluate(o, +1, H, NC) for o in oof]
    agg = {k: float(np.nanmean([p[k] for p in per])) for k in ("rho", "rho_nc", "tie", "s10")}
    agg.update(valid=per[0]["valid"], source="this paper", monotone="POS",
               note="mean over the 10 cross-validation repeats")
    out["measures"]["field tier (ours, OOF)"] = agg
    # best third-party measure at its better folding
    best = {}
    for name in names:
        cands = [(out["measures"][f"{name} ({v})"]["rho"], v) for v in ("worst", "crit")]
        cands = [c for c in cands if np.isfinite(c[0])]
        if cands:
            best[name] = max(cands)
    top = max(best.items(), key=lambda kv: kv[1][0])
    top_nc = max(((n, max(np.nan_to_num(out["measures"][f"{n} ({v})"]["rho_nc"], nan=-9)
                           for v in ("worst", "crit"))) for n in names), key=lambda kv: kv[1])
    out["best_third_party"] = {"measure": top[0], "folding": top[1][1], "rho": top[1][0],
                               "best_rho_nc": {"measure": top_nc[0], "rho_nc": float(top_nc[1])}}
    out["best_per_measure_at_better_folding"] = {n: {"rho": v[0], "folding": v[1]} for n, v in best.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    report = {}
    for subject in C.SUBJECTS:
        report[C.LABEL[subject]] = run(subject)
    dest = C.results_path("crime", "crime_table.json")
    json.dump(report, open(dest, "w"), indent=1, default=float)
    for lab, d in report.items():
        ms = d["measures"]
        rank = sorted([k for k in ms if np.isfinite(ms[k].get("rho", np.nan))],
                      key=lambda k: -ms[k]["rho"])
        print(f"\n=== {lab}: {d['n_references']} references ({d['n_noncollision']} non-collision), "
              f"{d['n_third_party_measures']} third-party measures ===")
        print(f"{'measure':34s} {'src':6s} {'r_s':>7s} {'r_s^nc':>7s} {'tie':>6s} {'s10':>6s}")
        for k in rank[:a.top]:
            r = ms[k]
            src = "CriMe" if r["source"].startswith("CommonRoad") else "ours"
            print(f"{k[:34]:34s} {src:6s} {r['rho']:7.3f} "
                  f"{r['rho_nc'] if np.isfinite(r.get('rho_nc', np.nan)) else float('nan'):7.3f} "
                  f"{r['tie']:6.3f} {r['s10'] if np.isfinite(r.get('s10', np.nan)) else float('nan'):6.3f}")
        b = d["best_third_party"]
        print(f"  best third-party: {b['measure']} ({b['folding']}) r_s {b['rho']:.3f}; "
              f"best on the non-collision slice: {b['best_rho_nc']['measure']} "
              f"r_s^nc {b['best_rho_nc']['rho_nc']:.3f}")
        for probe in ("CPI", "WTTC"):
            v = d["best_per_measure_at_better_folding"].get(probe)
            if v:
                print(f"  {probe}: r_s {v['rho']:.3f} ({v['folding']})")
    print(f"\nsaved {dest}")
    print("paper: best CriMe openpilot CPI 0.65 (-0.23 on TransFuser); TransFuser WTTC 0.67 "
          "(0.44 on openpilot); non-collision best 0.48 / 0.53")


if __name__ == "__main__":
    main()
