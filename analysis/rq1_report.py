"""RQ1 (fidelity): Table 1 and the Sec. 5.2 prose, from the stored
out-of-fold field-tier scores. No training.

Reproduces, per subject:
  * Table 1: harm-weighted APFD (Eq. 3) and Spearman r_s against Tier-1
    truth for every ordering at its simulation cost: binary verdict, minimum
    TTC, minimum clearance, realized impact speed, best of the 35 CriMe
    measures, the field tier, Tier-1 mini-measurement at 1 and 3 replays,
    and true harm (M=100), each with a 95% bootstrap interval over scenarios
    (a bootstrap sample also draws one CV repeat / replay draw);
  * paired margins of the field tier over every baseline (r_s margin over
    the best CriMe measure +.082 [-.098,.264] / +.124 [.011,.239]; APFD gap
    over the verdict +.153 [.039,.281] / +.073 [.032,.130]; TransFuser APFD
    gaps over TTC +.032 [.007,.053] and clearance +.034 [.008,.061]);
  * r_s on the non-collision runs (0.68 / 0.70), per template, and on
    TransFuser with the oncoming-drift template excluded (0.71 +- 0.03);
  * tie statistics on reference pairs whose true harms differ by >10x
    (binary verdict 63% on openpilot; TTC<1.5 s flag 36%; field tier 8% / 2%);
  * escalation: 10% and 15% of each test fold at 3 replays (openpilot
    r_s 0.737 at 1.3x, 0.748 at 1.45x).

Two bootstrap protocols are used, exactly as the paper's drivers did:
  A. (field-tier row, paired margins) rng seed 7, resample scenarios and one
     CV repeat per sample -- the original report driver;
  B. (baseline rows, mini-measurement rows, M=100 row) rng seed 20260815,
     the same scenario resamples for every row (the paper's bootstrap convention).

Usage:  python analysis/rq1_report.py [--nboot 4000]
Output: results/rq1/table1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                  # noqa: E402
from proxima import field_tier as FT                              # noqa: E402
from proxima.metrics import apfd_h, boot_ci, ties_10x, pairs_10x  # noqa: E402

PAPER = {
    "openpilot": {
        "field tier": (.864, .729), "binary verdict": (.714, .481), "min TTC": (.801, .417),
        "min clearance": (.820, .498), "realized impact speed": (.721, .481),
        "best CriMe": (.828, .646), "Tier-1, 1 replay": (.744, .610),
        "Tier-1, 3 replays": (.865, .782), "true harm (M=100)": (.954, 1.0),
        "rho_nc": .68, "ties_binary": .63, "ties_ttc": .36, "ties_field": .08,
        "esc10": .737, "esc15": .748},
    "transfuser": {
        "field tier": (.913, .790), "binary verdict": (.841, .555), "min TTC": (.881, .609),
        "min clearance": (.878, .704), "realized impact speed": (.867, .560),
        "best CriMe": (.884, .666), "Tier-1, 1 replay": (.877, .594),
        "Tier-1, 3 replays": (.913, .710), "true harm (M=100)": (.940, 1.0),
        "rho_nc": .70, "ties_field": .02, "rho_no_oncoming": .71},
}


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def fmtci(m, lo, hi, nd=3):
    return f"{m:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"


# ------------------------------------------------------------ protocol A
def boot_a(fn, S_rows, y, rng, nboot):
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, len(y), len(y))
        r = rng.integers(0, len(S_rows))
        v = fn(S_rows[r], idx)
        if np.isfinite(v):
            out.append(v)
    o = np.array(out)
    return float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def paired_a(fn, S_rows, y, b, rng, nboot):
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, len(y), len(y))
        r = rng.integers(0, len(S_rows))
        va, vb = fn(S_rows[r], idx), fn(b, idx)
        if np.isfinite(va) and np.isfinite(vb):
            out.append(va - vb)
    o = np.array(out)
    return float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


# ------------------------------------------------------------ protocol B
def protocol_b(S, oofs, base, nboot, seed=20260815):
    """Rows and CIs of Table 1 (APFD_H and r_s)."""
    y, n = S["y"], len(S["y"])
    rng = np.random.default_rng(seed)
    H_rep = S["H_rep"]
    preds = {k: v[None, :] for k, v in base.items()}
    preds["field tier"] = oofs
    for M in (1, 3):
        preds[f"Tier-1, {M} replay" + ("s" if M > 1 else "")] = np.array(
            [[H_rep[i, rng.permutation(H_rep.shape[1])[:M]].mean() for i in range(n)]
             for _ in range(20)])
    preds["true harm (M=100)"] = y[None, :]
    idxs = [rng.integers(0, n, n) for _ in range(nboot)]
    out = {}
    for name, stat in (("apfd", lambda s, i: apfd_h(s[i], y[i])),
                       ("rho", lambda s, i: rho(s[i], y[i]))):
        rng2 = np.random.default_rng(seed + 1)
        res = {}
        for k, M in preds.items():
            R = len(M)
            point = float(np.mean([stat(M[r], np.arange(n)) for r in range(R)]))
            pick = rng2.integers(0, R, len(idxs))
            draws = np.array([stat(M[pick[j]], i) for j, i in enumerate(idxs)])
            res[k] = (point, *boot_ci(draws))
        out[name] = res
    return out


def run(subject, nboot):
    S = C.load_subject(subject)
    y, NC, tmpl = S["y"], S["NC"], S["template"]
    Z = C.load_oof(subject)
    oofs = Z["oofs"]
    base = C.baselines(S)
    cv, ckey = C.crime_best(S)
    base["best CriMe"] = cv
    P = PAPER[subject]
    print(f"\n================ {S['label']}  (nonzero labels {int((y > 0).sum())}/{len(y)}, "
          f"best CriMe measure = {ckey})")

    # ---- field tier, protocol A
    rng = np.random.default_rng(7)
    r_all = [rho(s, y) for s in oofs]
    r_nc = [rho(s[NC], y[NC]) for s in oofs]
    ap = [apfd_h(s, y) for s in oofs]
    m, lo, hi = boot_a(lambda s, i: apfd_h(s[i], y[i]), oofs, y, rng, nboot)
    print(f"field tier   APFD_H {np.mean(ap):.3f} +- {np.std(ap):.3f}  CI [{lo:.3f},{hi:.3f}]"
          f"   [paper {P['field tier'][0]:.3f}]")
    print(f"             r_s    {np.mean(r_all):.3f} +- {np.std(r_all):.3f}"
          f"   [paper {P['field tier'][1]:.3f} +- {'.034' if subject == 'openpilot' else '.015'}]")
    print(f"             r_s non-collision {np.mean(r_nc):.3f} +- {np.std(r_nc):.3f}   [paper {P['rho_nc']:.2f}]")
    per_tmpl = {}
    for t in np.unique(tmpl):
        mm = tmpl == t
        v = [rho(s[mm], y[mm]) for s in oofs]
        per_tmpl[str(t)] = float(np.nanmean(v))
        print(f"             r_s {t:18s} {np.nanmean(v):.2f}")
    res = {"field_tier": {"apfd": float(np.mean(ap)), "apfd_sd": float(np.std(ap)),
                          "apfd_ci": [lo, hi], "rho": float(np.mean(r_all)),
                          "rho_sd": float(np.std(r_all)), "rho_nc": float(np.mean(r_nc)),
                          "rho_nc_sd": float(np.std(r_nc)), "rho_by_template": per_tmpl}}
    if subject == "transfuser":
        mm = tmpl != "oncoming_drift"
        v = [rho(s[mm], y[mm]) for s in oofs]
        res["field_tier"]["rho_excluding_oncoming"] = [float(np.mean(v)), float(np.std(v))]
        print(f"             r_s excluding oncoming_drift (n={int(mm.sum())}) "
              f"{np.mean(v):.2f} +- {np.std(v):.2f}   [paper {P['rho_no_oncoming']:.2f} +- 0.03]")
        # the paper's footnote value was taken from the earlier rank-fusion
        # composite (the one whose conformal calibration Sec. 5.2 reports);
        # its out-of-fold scores are stored with the calibration run
        cal = np.load(C.results_path("rq1", "transfuser_calibration.npz"))
        if "oof_scores" in cal.files:
            v35 = [rho(s[mm], y[mm]) for s in cal["oof_scores"]]
            res["field_tier"]["rho_excluding_oncoming_rank4_composite"] = [float(np.mean(v35)), float(np.std(v35))]
            print(f"             (same cut on the stored rank-fusion composite: {np.mean(v35):.2f} +- {np.std(v35):.2f},"
                  f" the value the footnote quotes)")
    zt = [np.mean(s == 0) for s in oofs]
    print(f"             exact-zero predictions: {100 * np.mean(zt):.0f}% of runs")

    # ---- paired margins, protocol A (rng continues, exp25 order)
    print("  paired margins of the field tier (protocol A):")
    res["paired"] = {}
    for bn, bv in base.items():
        dm, dl, dh = paired_a(lambda s, i: apfd_h(s[i], y[i]), oofs, y, bv, rng, nboot)
        out = []
        for _ in range(nboot):
            idx = rng.integers(0, len(y), len(y))
            r = rng.integers(0, len(oofs))
            a_, b_ = rho(oofs[r][idx], y[idx]), rho(bv[idx], y[idx])
            if np.isfinite(a_) and np.isfinite(b_):
                out.append(a_ - b_)
        o = np.array(out)
        rm, rl, rh = float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))
        res["paired"][bn] = {"dAPFD": [dm, dl, dh], "drho": [rm, rl, rh]}
        print(f"    vs {bn:22s} dAPFD {dm:+.3f} [{dl:+.3f},{dh:+.3f}]   dr_s {rm:+.3f} [{rl:+.3f},{rh:+.3f}]")
    pm = res["paired"]
    if subject == "openpilot":
        print(f"    [paper: r_s over best CriMe +.082 [-.098,.264]; APFD over verdict +.153 [.039,.281]]")
    else:
        print(f"    [paper: r_s over best CriMe +.124 [.011,.239]; APFD over verdict +.073 [.032,.130];"
              f" APFD over TTC +.032 [.007,.053]; over clearance +.034 [.008,.061]]")

    # ---- baseline / mini-measurement rows, protocol B
    B = protocol_b(S, oofs, base, nboot)
    print("  Table 1 rows (protocol B; brackets = 95% bootstrap over scenarios):")
    print(f"    {'ordering':24s} {'APFD_H':>22s} {'r_s':>22s}   [paper APFD / r_s]")
    res["table1"] = {}
    for k in ("binary verdict", "min TTC", "min clearance", "realized impact speed",
              "best CriMe", "field tier", "Tier-1, 1 replay", "Tier-1, 3 replays",
              "true harm (M=100)"):
        a, r = B["apfd"][k], B["rho"][k]
        if k == "field tier":   # paper reports the repeat mean + protocol-A CI
            a = (float(np.mean(ap)), lo, hi)
        res["table1"][k] = {"apfd": list(a), "rho": list(r)}
        pa, pr = P[k]
        print(f"    {k:24s} {fmtci(*a):>22s} {fmtci(*r):>22s}   [{pa:.3f} / {pr:.3f}]")

    # ---- ties on >10x-differing pairs
    i, j = pairs_10x(y)
    tb = ties_10x(base["binary verdict"], y)
    ttc_flag = (S["X41"][:, S["names41"].index("ttc_min")] < 1.5).astype(float)
    tt = ties_10x(ttc_flag, y)
    tf_ = float(np.mean([ties_10x(s, y) for s in oofs]))
    res["ties_10x"] = {"n_pairs": int(len(i)), "binary verdict": tb, "ttc_flag_1.5s": tt,
                       "field tier": tf_}
    print(f"  ties on the {len(i)} pairs whose true harms differ >10x: binary verdict "
          f"{100 * tb:.0f}%  TTC<1.5s flag {100 * tt:.0f}%  field tier {100 * tf_:.0f}%"
          + (f"   [paper 63% / 36% / 8%]" if subject == "openpilot" else
             "   [paper: field tier 2%; the paper's sentence attributes 2% to the verdict, "
             "the verdict's own rate is the number printed here]"))

    # ---- escalation (openpilot: p and tail were stored)
    if "p" in Z.files:
        res["escalation"] = {}
        for frac, Bn in ((0.10, 3), (0.15, 3)):
            rr, aa = [], []
            for r in range(len(oofs)):
                p, tl = Z["p"][r], Z["tail"][r]
                s = np.where(p < FT.TAU, 0.0, 1.0 + FT.ranks(p) + tl)
                s = FT.escalate(s, p, S["inj_by_ref"], np.arange(len(y)), frac=frac, B=Bn)
                rr.append(rho(s, y)); aa.append(apfd_h(s, y))
            res["escalation"][f"{int(100 * frac)}%x{Bn}"] = {
                "cost": 1 + frac * Bn, "rho": float(np.mean(rr)), "rho_sd": float(np.std(rr)),
                "apfd": float(np.mean(aa))}
            print(f"  escalation {int(100 * frac)}% x {Bn} replays (cost {1 + frac * Bn:.2f}x): "
                  f"r_s {np.mean(rr):.3f} +- {np.std(rr):.3f}  APFD_H {np.mean(aa):.3f}"
                  f"   [paper {P['esc' + str(int(100 * frac))]:.3f}]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=4000)
    a = ap.parse_args()
    out = {s: run(s, a.nboot) for s in C.SUBJECTS}
    fp = C.results_path("rq1", "table1.json")
    json.dump(out, open(fp, "w"), indent=1, default=float)
    print(f"\nsaved {fp}")


if __name__ == "__main__":
    main()
