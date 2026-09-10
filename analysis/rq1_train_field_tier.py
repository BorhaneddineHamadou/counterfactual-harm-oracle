"""Train and evaluate the field tier out of fold (RQ1, Table 1 inputs).

Ten repeats of five-fold scenario-level cross-validation. For every fold the
estimator of proxima/field_tier.py is fitted on the training scenarios'
corpus (nominal executions + replay traces) and scored on the held-out
scenarios' run-0 traces. The out-of-fold scores, the detector probability
p and the survivor tail are saved so every RQ1 table can be recomputed
without retraining:

    results/rq1/<subject>_field_tier_oof.npz
        oofs      (10, 150)  structured score per repeat (0 = declared tie)
        oofs_esc  (10, 150)  with 10% escalation at 3 replays
        p, tail   (10, 150)  detector probability / survivor tail
        y         (150,)     Tier-1 labels
        taus      (10, 5)    tau actually used per fold

Two subject-specific settings are declared in analysis/common.py and
configs/field_tier.json:
  openpilot   6 seeds per readout, survivor = 1 + rank(p) + rank(z) + rank(raw),
              tau = 0.05 declared (the inner-CV pick is computed and logged
              but not used; `--tau cv` uses it, giving the Sec. 7 ablation);
  transfuser  3 seeds, survivor = 1 + rank(z) + rank(raw), tau selected per
              fold by maximum inner out-of-fold rank correlation.

Runtime on a 32-core node: openpilot ~1 h, TransFuser ~25 min (one process
per subject; sklearn uses all cores per fit). `--reps` shortens a check.

Usage:  python analysis/rq1_train_field_tier.py openpilot|transfuser
            [--tau declared|cv] [--reps 10] [--out results/rq1/<name>.npz]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima import field_tier as FT                  # noqa: E402
from proxima.metrics import ranks                     # noqa: E402


def run_rep(S, rep, tau_mode):
    n = len(S["y"])
    y = S["y"]
    oof = np.full(n, np.nan)
    oof_esc = np.full(n, np.nan)
    P = np.full(n, np.nan)
    TL = np.full(n, np.nan)
    taus = []
    for te in C.folds_for(rep, n):
        tri = np.setdiff1d(np.arange(n), te)
        # inner 4-group cross-validation on the training scenarios: the
        # tau-selection signal (used on TransFuser; logged on openpilot)
        rng = np.random.default_rng(rep * 7 + 1)
        inner = np.array_split(rng.permutation(tri), 4)
        agg_p = np.full(n, np.nan)
        agg_t = np.full(n, np.nan)
        for ite in inner:
            itr = np.setdiff1d(tri, ite)
            Xtr, ytr, w, _ = FT.corpus_ordered(S, itr)
            parts = FT.fit_readouts(Xtr, ytr, w, rep, S["inner_seeds"])
            p, tz, tr = FT.predict_readouts(parts, S["X2"][ite])
            agg_p[ite] = p
            agg_t[ite] = ranks(tz) + ranks(tr)
        if S["tau_rule"] == "per_fold_max":
            tau = FT.pick_tau_max(agg_p, agg_t, tri, y)
        else:
            tau_cv = FT.pick_tau_1se(agg_p, agg_t, tri, y, rep)
            tau = tau_cv if tau_mode == "cv" else FT.TAU
        taus.append(tau)
        # the fold's estimator
        Xtr, ytr, w, _ = FT.corpus_ordered(S, tri)
        parts = FT.fit_readouts(Xtr, ytr, w, rep, S["n_seeds"])
        p, tz, tr = FT.predict_readouts(parts, S["X2"][te])
        s = FT.structured_score(p, tz, tr, tau, p_in_tail=S["p_in_tail"])
        oof[te] = s
        P[te] = p
        TL[te] = ranks(tz) + ranks(tr)
        oof_esc[te] = FT.escalate(s, p, S["inj_by_ref"], te, frac=0.10, B=3)
    return oof, oof_esc, P, TL, taus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", choices=C.SUBJECTS)
    ap.add_argument("--tau", choices=("declared", "cv"), default="declared")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = C.load_subject(a.subject)
    tag = "field_tier_oof" + ("_cvtau" if a.tau == "cv" else "")
    out = a.out or C.results_path("rq1", f"{a.subject}_{tag}.npz")
    y, nz = S["y"], S["y"] > 0
    print(f"{a.subject}: n={len(y)} nonzero={int(nz.sum())} corpus "
          f"{len(S['aug_row'])} nominal + {len(S['rep_scn'])} replay rows, "
          f"{S['X2'].shape[1]} features, seeds={S['n_seeds']}, "
          f"tau rule={S['tau_rule'] if a.tau == 'declared' else 'inner-CV 1-SE'}",
          flush=True)
    t0 = time.time()
    R, RE, oofs, oofs_esc, Ps, TLs, TAUS = [], [], [], [], [], [], []
    for rep in range(a.reps):
        oof, oe, P, TL, taus = run_rep(S, rep, a.tau)
        oofs.append(oof); oofs_esc.append(oe); Ps.append(P); TLs.append(TL); TAUS.append(taus)
        r1 = spearmanr(oof, y).statistic
        r2 = spearmanr(oe, y).statistic
        zeros = oof == 0
        print(f"rep {rep}: rho {r1:.3f}  +escalation(10%x3) {r2:.3f}  "
              f"taus={np.round(taus, 2)} missed={int((zeros & nz).sum())} "
              f"kept={int((~zeros & ~nz).sum())} [{time.time()-t0:.0f}s]", flush=True)
        R.append(r1); RE.append(r2)
    np.savez_compressed(out, oofs=np.array(oofs), oofs_esc=np.array(oofs_esc),
                        p=np.array(Ps), tail=np.array(TLs), y=y,
                        taus=np.array(TAUS, float))
    print(f"\n{S['label']} field tier: rho {np.mean(R):.3f} +- {np.std(R):.3f}"
          f"   (paper: 0.729 +- 0.034 openpilot / 0.790 +- 0.015 TransFuser)")
    print(f"+ escalation 10% x 3 replays (1.3x): rho {np.mean(RE):.3f} +- {np.std(RE):.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
