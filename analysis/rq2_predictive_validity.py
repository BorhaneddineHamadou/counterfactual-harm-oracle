"""RQ2, predictive validity (paper Sec. 5.3, Table 2).

Every oracle judges run 0 of each scenario; the criterion is external to the
definition of harm: the held-out repeat executions (runs 1-4) of the same
scenario, read twice
  (a) does the scenario crash again at all?      -> AUC ("crash again")
  (b) among those that do, how much harm do the repeats carry?
                                                 -> Spearman ("severity cut")
and restricted to the scenarios whose run 0 PASSED, where the binary verdict
and realized impact speed are constants (AUC_pass).

Oracles at their simulation cost (units of one nominal run):
  binary verdict, -TTC_min, -min clearance, realized impact speed   1x
  field tier (out-of-fold, results/rq1/<subject>_field_tier_oof.npz;
      scores mid-ranked per CV repeat and averaged over the 10 repeats)  1x
  H hat (M=1) 2x, H hat (M=3) 4x: Tier-1 on a budget of M replays, the
      mean of 20 random draws of M replays from the reference's 100
  H (M=100) 101x: the reference label itself

Held-out harm of a repeat execution = iota(impact speed) if it made contact,
else 0. Confidence intervals: 2000 bootstrap resamples of scenarios (seed
20260815), the pass-subset ones resampled within the subset. Paired
differences (field tier minus a baseline) use 4000 resamples, each also
drawing one CV repeat of the field tier (the per-repeat convention of the
paper's paired tests; its point estimates use the repeat-averaged score).

Held-out runs: all 600 per subject (runs 1-4 of the 150 scenarios).

Usage:  python analysis/rq2_predictive_validity.py [--nboot 2000]
Output: results/rq2/table2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima.injury import occupant                   # noqa: E402
from proxima.metrics import auc, boot_ci, surfaced    # noqa: E402

SEED = 20260815
NBOOT_PAIRED = 4000
ROWS = ["binary verdict", "realized impact speed", "min TTC", "min clearance",
        "field tier", "H hat (M=1)", "H hat (M=3)", "H (M=100)"]
COST = {"binary verdict": 1, "realized impact speed": 1, "min TTC": 1,
        "min clearance": 1, "field tier": 1, "H hat (M=1)": 2,
        "H hat (M=3)": 4, "H (M=100)": 101}
REF = "field tier"
PAPER = {
    "openpilot": {"auc": [.731, .731, .742, .893, .953, .820, .819, .859],
                  "auc_pass": [None, .500, .522, .815, .923, .764, .764, .840]},
    "transfuser": {"auc": [.818, .820, .893, .927, .961, .885, .900, .902],
                   "auc_pass": [None, .500, .711, .815, .901, .764, .733, .742]},
}


def predictors(S, rng):
    b = C.baselines(S)
    oof = C.load_oof(S["name"])["oofs"]
    pred = {"binary verdict": b["binary verdict"],
            "realized impact speed": b["realized impact speed"],
            "min TTC": b["min TTC"],
            "min clearance": b["min clearance"],
            "field tier": np.apply_along_axis(rankdata, 1, oof).mean(axis=0),
            "H (M=100)": S["y"]}
    H = S["H_rep"]
    n = len(S["y"])
    for M in (1, 3):
        draws = np.array([[H[i, rng.permutation(H.shape[1])[:M]].mean()
                           for _ in range(20)] for i in range(n)])
        pred[f"H hat (M={M})"] = draws.mean(axis=1)
    return pred


def criterion(S, hold):
    crash_frac = np.array([np.mean([c for c, _ in hold[int(s)]]) for s in S["sids"]])
    harm = np.array([np.mean([occupant(dv) if c else 0.0 for c, dv in hold[int(s)]])
                     for s in S["sids"]])
    return crash_frac, harm


def bci(fn, n, rng, nboot, *arrays):
    vals = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    return list(boot_ci(vals))


def paired(fn, ref_reps, others, rng, nboot, idx_pool):
    """Field tier minus each baseline on the same scenario resamples. The
    field tier enters per CV repeat (mid-ranked scores of one repeat drawn
    per resample), the convention of the paper's paired tests, so its
    per-repeat AUC (not the repeat-averaged score's) is what is compared."""
    out = {}
    idxs = [idx_pool[rng.integers(0, len(idx_pool), len(idx_pool))] for _ in range(nboot)]
    reps = rng.integers(0, ref_reps.shape[0], nboot)
    for k, v in others.items():
        d = np.array([fn(ref_reps[r], i) - fn(v, i) for r, i in zip(reps, idxs)])
        d = d[np.isfinite(d)]
        out[k] = {"delta": float(d.mean()), "ci": list(boot_ci(d)),
                  "p_better": float(np.mean(d > 0))}
    return out


def run_subject(subject, nboot):
    S = C.load_subject(subject)
    hold = S["hold"]
    rng = np.random.default_rng(SEED)
    pred = predictors(S, rng)
    crash_frac, harm = criterion(S, hold)
    any_crash = crash_frac > 0
    passed = pred["binary verdict"] == 0
    n = len(S["y"])
    n_hold = sum(len(v) for v in hold.values())
    pn = int(passed.sum())
    print(f"\n{'='*70}\n=== {S['label']}: predict runs 1-4 from run 0 "
          f"({n} scenarios, {n_hold} held-out executions) ===")
    print(f"  {int(any_crash.sum())} scenarios crash at least once more; the run-0 "
          f"verdict called {int((~passed).sum())} collisions")
    print(f"  run 0 PASSED on {pn} scenarios: {int(any_crash[passed].sum())} of them "
          f"crash again, holding {harm[passed].sum()/harm.sum():.0%} of all held-out harm")
    res = {"n_scenarios": n, "held_out_runs": n_hold,
           "scenarios_crashing_again": int(any_crash.sum()),
           "run0_collisions": int((~passed).sum()), "passed": pn,
           "passed_crash_again": int(any_crash[passed].sum()),
           "passed_harm_share": float(harm[passed].sum() / harm.sum()),
           "rows": {}}

    print(f"\n  {'oracle':<22}{'cost':>6}{'AUC crash-again':>26}{'surf@10%':>10}{'surf@20%':>10}")
    for k in ROWS:
        p = pred[k]
        a = auc(p, any_crash)
        rh = spearmanr(p, harm).statistic
        lo, hi = bci(lambda x, y_: spearmanr(x, y_).statistic, n, rng, nboot, p, harm)
        alo, ahi = bci(lambda x, y_: auc(x, y_), n, rng, nboot, p, any_crash)
        res["rows"][k] = {"cost": COST[k], "auc": float(a), "auc_ci": [alo, ahi],
                          "rho_heldout_harm": float(rh), "rho_heldout_harm_ci": [lo, hi],
                          "surf10": float(surfaced(p, harm, .10)),
                          "surf20": float(surfaced(p, harm, .20))}
        print(f"  {k:<22}{COST[k]:>5}x   {a:.3f} [{alo:.2f},{ahi:.2f}]"
              f"{res['rows'][k]['surf10']:>10.1%}{res['rows'][k]['surf20']:>10.1%}")

    # severity cut: among the scenarios that crash again, order held-out harm
    m = any_crash
    print(f"\n  severity cut: Spearman vs held-out harm among the {int(m.sum())} "
          f"scenarios that crash again")
    for k in ROWS:
        rh = spearmanr(pred[k][m], harm[m]).statistic
        lo, hi = bci(lambda x, y_: spearmanr(x, y_).statistic, int(m.sum()), rng, nboot,
                     pred[k][m], harm[m])
        v = pred[k][m]
        ii, jj = np.triu_indices(len(v), 1)
        ties = float(np.mean(v[ii] == v[jj]))
        # the paper's tie figure: share of the ordered runs that do not get
        # a distinct value (1 - unique values / n)
        ties_paper = 1.0 - len(np.unique(v)) / len(v)
        res["rows"][k].update(rho_severity=float(rh), rho_severity_ci=[lo, hi],
                              tie_pairs_severity=ties, tie_share_severity=ties_paper)
        print(f"    {k:<22}{rh:>7.3f} [{lo:>6.3f},{hi:>6.3f}]   tied pairs {ties:>4.0%}"
              f"   non-distinct values {ties_paper:>4.0%}")

    # the practical case: run 0 passed
    print(f"\n  AUC on the {pn} scenarios whose run 0 passed")
    for k in ROWS:
        if k == "binary verdict":
            res["rows"][k]["auc_pass"] = None
            print(f"    {k:<22}   ---  constant on this subset")
            continue
        a = auc(pred[k][passed], any_crash[passed])
        alo, ahi = bci(lambda x, y_: auc(x, y_), pn, rng, nboot,
                       pred[k][passed], any_crash[passed])
        res["rows"][k].update(auc_pass=float(a), auc_pass_ci=[alo, ahi],
                              surf10_pass=float(surfaced(pred[k][passed], harm[passed], .10)))
        print(f"    {k:<22}AUC {a:.3f} [{alo:.2f},{ahi:.2f}]   "
              f"surf@10% {res['rows'][k]['surf10_pass']:.1%}")

    # paired bootstrap: field tier minus each baseline, same resamples
    others = {k: pred[k] for k in ROWS if k != REF}
    ft_reps = np.apply_along_axis(rankdata, 1, C.load_oof(S["name"])["oofs"])
    prng = np.random.default_rng(SEED + 1)
    res["paired_auc_all"] = paired(lambda v, i: auc(v[i], any_crash[i]), ft_reps,
                                   others, prng, NBOOT_PAIRED, np.arange(n))
    pidx = np.flatnonzero(passed)
    res["paired_auc_pass"] = paired(lambda v, i: auc(v[i], any_crash[i]), ft_reps,
                                    {k: v for k, v in others.items() if k != "binary verdict"},
                                    prng, NBOOT_PAIRED, pidx)
    print("\n  paired bootstrap (per-repeat field tier), field tier minus baseline "
          "(AUC all 150 | AUC run-0-passed):")
    for k in others:
        a = res["paired_auc_all"][k]
        b = res["paired_auc_pass"].get(k)
        print(f"    vs {k:<22} {a['delta']:+.3f} [{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}] | "
              + ("   ---" if b is None else
                 f"{b['delta']:+.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]"))

    print("\n  Table 2 check (reproduced | paper):")
    for k, a, ap in zip(ROWS, PAPER[subject]["auc"], PAPER[subject]["auc_pass"]):
        r1 = res["rows"][k]["auc"]
        r2 = res["rows"][k].get("auc_pass")
        print(f"    {k:<22} AUC {r1:.3f} | {a:.3f}    AUC_pass "
              f"{'  --  ' if r2 is None else f'{r2:.3f}'} | "
              f"{'  --  ' if ap is None else f'{ap:.3f}'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    out = {"held_out": "all 600 runs per subject", "nboot": a.nboot, "seed": SEED}
    for s in C.SUBJECTS:
        out[s] = run_subject(s, a.nboot)
    fp = C.results_path("rq2", "table2.json")
    json.dump(out, open(fp, "w"), indent=1, default=float)
    print(f"\nsaved {fp}")


if __name__ == "__main__":
    main()
