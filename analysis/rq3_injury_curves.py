"""RQ3, sensitivity to the declared injury curve (paper Sec. 5.4, "Nor is the
injury curve a lever").

iota only reweights impact speeds the campaign already recorded, so every
reference can be re-scored under curves nobody here fitted, with no new
simulation. The script re-labels the 150 references of each subject
under the 14 published alternatives to the shipped Kusano-Gabler MAIS2+
logistic (belt states restored; nine NHTSA 2010-2015 NASS-CDS logistics;
the equal-mass dv = s/2 convention on two of them; Joksch's fourth-power
rule) and reports how far the reference ordering moves: mean-harm span,
Spearman against the shipped curve, share of comparable pairs that reorder,
and the stored field tier's r_s lead over the best telemetry scalar (min
TTC, min clearance, realized impact speed) when all are scored against each
curve's labels (the paper's "+0.23 / +0.08 to +0.09").

Usage:  python analysis/rq3_injury_curves.py
Output: results/rq3/injury_curves.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima.injury_library import library            # noqa: E402

SHIPPED = "Kusano-Gabler MAIS2+ (shipped)"
NBOOT = 4000
SEED = 20260824


def replay_outcomes(S):
    z = np.load(C.data_path(S["name"], "replay_outcomes.npz"))
    return z["sid"].astype(int), z["contact"].astype(bool), z["dv"].astype(float)


def harm_by_reference(S, sid, contact, dv, curve):
    w = np.where(contact, curve(dv), 0.0)
    return np.array([w[sid == s].mean() for s in S["sids"]])


def pair_flips(a, b, eps=1e-12):
    n = len(a)
    i, j = np.triu_indices(n, 1)
    da, db = a[i] - a[j], b[i] - b[j]
    comparable = np.abs(da) > eps
    flipped = comparable & (np.sign(da) != np.sign(db)) & (np.abs(db) > eps)
    return int(flipped.sum()), int(comparable.sum())


def boot_rho(a, b, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a)
    out = []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        if np.std(a[idx]) == 0 or np.std(b[idx]) == 0:
            continue
        out.append(spearmanr(a[idx], b[idx]).statistic)
    out = np.array([o for o in out if np.isfinite(o)])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if len(out) else [np.nan] * 2


def published_curves():
    lib = library()
    return {n: v for n, v in lib.items() if v[1]}          # published only


def stage1():
    lib = published_curves()
    alts = [n for n in lib if n != SHIPPED]
    report = {"curves": {n: lib[n][2] for n in lib}, "subjects": {}}
    for s in C.SUBJECTS:
        S = C.load_subject(s)
        sid, contact, dv = replay_outcomes(S)
        H0 = harm_by_reference(S, sid, contact, dv, lib[SHIPPED][0])
        assert np.abs(H0 - S["y"]).max() < 1e-9, "shipped curve must reproduce the labels"
        rows = {}
        # the field tier's stored out-of-fold scores (Table 1) and the three
        # telemetry scalars, re-scored against each curve's labels
        oofs = np.load(C.results_path("rq1", f"{s}_field_tier_oof.npz"))["oofs"]
        base = C.baselines(S)
        scal = {k: v for k, v in base.items() if k != "binary verdict"}
        print(f"\n=== {S['label']}: {len(S['sids'])} references, {len(sid)} replays, "
              f"{int(contact.sum())} crashes; {len(alts)} published alternatives ===")
        print(f"{'curve':40s} {'meanH':>8s} {'rho':>6s} {'tau':>6s} {'flip%':>6s} "
              f"{'r_s ft':>7s} {'best scalar':>22s} {'lead':>7s}")
        for n in [SHIPPED] + alts:
            H = harm_by_reference(S, sid, contact, dv, lib[n][0])
            rho = spearmanr(H0, H).statistic if np.std(H) > 0 else np.nan
            tau = kendalltau(H0, H).statistic if np.std(H) > 0 else np.nan
            fl, cmp_ = pair_flips(H0, H)
            r_ft = float(np.mean([spearmanr(o, H).statistic for o in oofs]))
            r_sc = {k: float(spearmanr(v, H).statistic) for k, v in scal.items()}
            best = max(r_sc, key=r_sc.get)
            rows[n] = {"source": lib[n][2], "mean_H": float(H.mean()),
                       "nonzero": int((H > 0).sum()), "rho_vs_shipped": float(rho),
                       "tau_vs_shipped": float(tau), "rho_ci": boot_rho(H0, H),
                       "pair_flip_frac": fl / cmp_ if cmp_ else np.nan,
                       "rho_field_tier": r_ft, "rho_scalars": r_sc, "best_scalar": best,
                       "rho_lead_over_best_scalar": r_ft - r_sc[best]}
            print(f"{n[:40]:40s} {H.mean():8.5f} {rho:6.3f} {tau:6.3f} "
                  f"{100 * fl / max(cmp_, 1):6.2f} {r_ft:7.3f} {best:>22s} {r_ft - r_sc[best]:+7.3f}")
        leads = [rows[n]["rho_lead_over_best_scalar"] for n in rows]
        means = [rows[n]["mean_H"] for n in alts]
        alt_rho = {n: rows[n]["rho_vs_shipped"] for n in alts}
        alt_flip = {n: rows[n]["pair_flip_frac"] for n in alts}
        worst = min(alt_rho, key=alt_rho.get)
        summ = {"n_alternatives": len(alts),
                "mean_H_span_factor": float(max(means) / min(means)),
                "min_rho_vs_shipped": float(alt_rho[worst]), "min_rho_curve": worst,
                "max_pair_flip_frac": float(max(alt_flip.values())),
                "max_pair_flip_curve": max(alt_flip, key=alt_flip.get),
                "max_pair_flip_frac_excluding_joksch": float(max(
                    v for n, v in alt_flip.items() if "Joksch" not in n)),
                "rho_lead_range": [float(min(leads)), float(max(leads))]}
        print(f"  mean-harm span x{summ['mean_H_span_factor']:.0f}; min rho vs shipped "
              f"{summ['min_rho_vs_shipped']:.3f} ({worst}); max pair flips "
              f"{100*summ['max_pair_flip_frac']:.1f}% ({summ['max_pair_flip_curve']}), "
              f"{100*summ['max_pair_flip_frac_excluding_joksch']:.1f}% without Joksch; "
              f"r_s lead over the best scalar [{min(leads):+.3f}, {max(leads):+.3f}]")
        report["subjects"][s] = {"curves": rows, "summary": summ}
    fp = C.results_path("rq3", "injury_curves.json")
    json.dump(report, open(fp, "w"), indent=1, default=float)
    print(f"\nsaved {fp}\npaper: span x25 (openpilot) / x220 (TransFuser); "
          "r_s >= 0.998 / >= 0.922; pairs reordering 1.3% / 9.4% (Joksch alone); "
          "r_s lead over the best telemetry scalar +0.23 (openpilot), +0.08 to +0.09 (TransFuser)")


def main():
    stage1()


if __name__ == "__main__":
    main()
