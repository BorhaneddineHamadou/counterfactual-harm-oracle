"""RQ3, sensitivity to the declared injury curve (paper Sec. 5.4, "Nor is the
injury curve a lever").

iota only reweights impact speeds the campaign already recorded, so every
reference can be re-scored under curves nobody here fitted, with no new
simulation. Stage 1 (default) re-labels the 150 references of each subject
under the 14 published alternatives to the shipped Kusano-Gabler MAIS2+
logistic (belt states restored; nine NHTSA 2010-2015 NASS-CDS logistics;
the equal-mass dv = s/2 convention on two of them; Joksch's fourth-power
rule) and reports how far the reference ordering moves: mean-harm span,
Spearman against the shipped curve, share of comparable pairs that reorder.

Stage 2 (--sweep) re-distills the field tier under each curve's labels and
reports its harm-weighted APFD lead over the best telemetry scalar (min TTC,
min clearance, realized impact speed, each scored against the same
relabelled truth). Retraining 14 oracles is expensive; by default the sweep
uses 3 CV repeats and 3 seeds per readout (the shipped protocol is 10
repeats and 6 seeds on openpilot), which is documented in the output.

Usage:  python analysis/rq3_injury_curves.py                 # stage 1, both subjects
        python analysis/rq3_injury_curves.py --sweep --subject openpilot [--reps 3] [--seeds 3]
Output: results/rq3/injury_curves.json ; results/rq3/injury_sweep_<subject>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima import field_tier as FT                  # noqa: E402
from proxima.injury_library import library            # noqa: E402
from proxima.metrics import apfd_h, ranks             # noqa: E402

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
        print(f"\n=== {S['label']}: {len(S['sids'])} references, {len(sid)} replays, "
              f"{int(contact.sum())} crashes; {len(alts)} published alternatives ===")
        print(f"{'curve':40s} {'meanH':>8s} {'rho':>6s} {'tau':>6s} {'flip%':>6s}")
        for n in [SHIPPED] + alts:
            H = harm_by_reference(S, sid, contact, dv, lib[n][0])
            rho = spearmanr(H0, H).statistic if np.std(H) > 0 else np.nan
            tau = kendalltau(H0, H).statistic if np.std(H) > 0 else np.nan
            fl, cmp_ = pair_flips(H0, H)
            rows[n] = {"source": lib[n][2], "mean_H": float(H.mean()),
                       "nonzero": int((H > 0).sum()), "rho_vs_shipped": float(rho),
                       "tau_vs_shipped": float(tau), "rho_ci": boot_rho(H0, H),
                       "pair_flip_frac": fl / cmp_ if cmp_ else np.nan}
            print(f"{n[:40]:40s} {H.mean():8.5f} {rho:6.3f} {tau:6.3f} "
                  f"{100 * fl / max(cmp_, 1):6.2f}")
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
                    v for n, v in alt_flip.items() if "Joksch" not in n))}
        print(f"  mean-harm span x{summ['mean_H_span_factor']:.0f}; min rho vs shipped "
              f"{summ['min_rho_vs_shipped']:.3f} ({worst}); max pair flips "
              f"{100*summ['max_pair_flip_frac']:.1f}% ({summ['max_pair_flip_curve']}), "
              f"{100*summ['max_pair_flip_frac_excluding_joksch']:.1f}% without Joksch")
        report["subjects"][s] = {"curves": rows, "summary": summ}
    fp = C.results_path("rq3", "injury_curves.json")
    json.dump(report, open(fp, "w"), indent=1, default=float)
    print(f"\nsaved {fp}\npaper: span x25 (openpilot) / x220 (TransFuser); "
          "r_s >= 0.998 / >= 0.922; pairs reordering 1.3% / 9.4% (Joksch alone)")


# ------------------------------------------------------------------ sweep
def distill_oof(S, y, reps, seeds):
    """Field-tier OOF scores under labels y (declared tau, shipped composite
    of the subject, no inner CV: tau = 0.05 on both subjects here)."""
    n = len(y)
    S2 = dict(S)
    S2["y"] = y
    oofs = []
    for rep in range(reps):
        oof = np.full(n, np.nan)
        for te in C.folds_for(rep, n):
            tri = np.setdiff1d(np.arange(n), te)
            Xtr, ytr, w, _ = FT.corpus_ordered(S2, tri)
            parts = FT.fit_readouts(Xtr, ytr, w, rep, seeds)
            p, tz, tr = FT.predict_readouts(parts, S["X2"][te])
            oof[te] = FT.structured_score(p, tz, tr, FT.TAU, p_in_tail=S["p_in_tail"])
        oofs.append(oof)
    return np.array(oofs)


def sweep(subject, reps, seeds):
    S = C.load_subject(subject)
    lib = published_curves()
    sid, contact, dv = replay_outcomes(S)
    base = C.baselines(S)
    scal = {k: v for k, v in base.items() if k != "binary verdict"}
    out = {"protocol": f"{reps} repeats x 5 folds, {seeds} seeds per readout, tau 0.05",
           "curves": {}}
    fp = C.results_path("rq3", f"injury_sweep_{subject}.json")
    t0 = time.time()
    for n in lib:
        y = harm_by_reference(S, sid, contact, dv, lib[n][0])
        oofs = distill_oof(S, y, reps, seeds)
        ap_ft = float(np.mean([apfd_h(o, y) for o in oofs]))
        ap_sc = {k: float(apfd_h(v, y)) for k, v in scal.items()}
        ap_bin = float(apfd_h(base["binary verdict"], y))
        best = max(ap_sc, key=ap_sc.get)
        rho = float(np.mean([spearmanr(o, y).statistic for o in oofs]))
        out["curves"][n] = {"mean_H": float(y.mean()), "apfd_field_tier": ap_ft,
                            "rho_field_tier": rho, "apfd_scalars": ap_sc,
                            "apfd_binary": ap_bin, "best_scalar": best,
                            "lead_over_best_scalar": ap_ft - ap_sc[best],
                            "lead_over_verdict": ap_ft - ap_bin}
        print(f"[{time.time()-t0:.0f}s] {n[:40]:40s} meanH {y.mean():.5f} "
              f"APFD_H ft {ap_ft:.3f} best scalar {best} {ap_sc[best]:.3f} "
              f"lead {ap_ft-ap_sc[best]:+.3f} (vs verdict {ap_ft-ap_bin:+.3f}) rho {rho:.3f}",
              flush=True)
        json.dump(out, open(fp, "w"), indent=1, default=float)
    leads = [v["lead_over_best_scalar"] for v in out["curves"].values()]
    out["lead_range"] = [float(min(leads)), float(max(leads))]
    json.dump(out, open(fp, "w"), indent=1, default=float)
    print(f"lead over best telemetry scalar across {len(leads)} curves: "
          f"[{min(leads):+.3f}, {max(leads):+.3f}]  (paper: +0.099 to +0.111)\nsaved {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--subject", choices=C.SUBJECTS, default="openpilot")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.sweep:
        sweep(a.subject, a.reps, a.seeds)
    else:
        stage1()


if __name__ == "__main__":
    main()
