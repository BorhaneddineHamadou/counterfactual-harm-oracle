"""RQ1 (fidelity): "Its confidence is honest" (Sec. 5.2).

The magnitude readout is the log-harm regressor of the field tier mapped
back to the harm scale, wrapped in a split-conformal 90% interval calibrated
on a held-out quarter of each training fold's scenarios. Out of fold, over
ten repeats of five-fold scenario-level CV, this script reports

  * coverage of the 90% intervals   (paper 93.3 +- 2.3 openpilot, 94.1 +- 1.7 TransFuser)
  * mean interval width
  * mean absolute error             (paper 0.007 and 0.003)
  * MAE / mean Monte-Carlo s.e. of the labels   (paper: labels 6.5x below model error)
  * split-half reliability of the M=100 labels  (paper 0.98)

Estimator details (identical to the reference runs): 71 single-trace
features, corpus = nominal rows (w 0.2) + replay rows (w 0.02) of the fit
scenarios, 3 seeds (random_state = seed*10+s, seed = rep*100 + fold),
calibration scenarios = first 25% of default_rng(seed+7).permutation(train).

--from-stored (default) reads results/rq1/<subject>_calibration.npz, the
per-repeat values of the reference run; --rerun retrains (openpilot ~30 min,
TransFuser ~20 min on a 32-core node) and writes <subject>_calibration_rerun.npz.

Usage:  python analysis/rq1_calibration.py [--rerun] [--reps 10] [--subject S]
Output: results/rq1/calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                       # noqa: E402
from proxima import field_tier as FT                   # noqa: E402
from proxima.metrics import split_half                 # noqa: E402

PAPER = {"openpilot": (93.3, 2.3, 0.007), "transfuser": (94.1, 1.7, 0.003)}
NF = 71    # single-trace features only


def rerun(S, reps):
    y, n = S["y"], len(S["y"])
    X2, Xa2, Xr2 = S["X2"][:, :NF], S["Xa2"][:, :NF], S["Xr2"][:, :NF]
    S71 = dict(S, X2=X2, Xa2=Xa2, Xr2=Xr2)
    cov, wid, mae = [], [], []
    t0 = time.time()
    for rep in range(reps):
        pt = np.full(n, np.nan); lo = np.full(n, np.nan); hi = np.full(n, np.nan)
        for fi, te in enumerate(C.folds_for(rep, n)):
            tr = np.setdiff1d(np.arange(n), te)
            seed = rep * 100 + fi
            perm = np.random.default_rng(seed + 7).permutation(tr)
            cal = perm[:int(0.25 * len(tr))]
            fit = perm[int(0.25 * len(tr)):]
            Xtr, ytr, w, _ = FT.corpus_ordered(S71, fit)
            ms = FT.fit_log_readout(Xtr, ytr, w, seed, 3)
            pt_cal = FT.to_harm(FT.predict_log(ms, X2[cal]))
            pt_te = FT.to_harm(FT.predict_log(ms, X2[te]))
            qhat = FT.conformal_qhat(pt_cal - y[cal], 0.9)
            pt[te] = pt_te
            lo[te] = np.clip(pt_te - qhat, 0, 1)
            hi[te] = np.clip(pt_te + qhat, 0, 1)
        cov.append(np.mean((y >= lo) & (y <= hi)))
        wid.append(np.mean(hi - lo))
        mae.append(np.mean(np.abs(pt - y)))
        print(f"  rep {rep}: cov90 {100 * cov[-1]:.1f}  width {wid[-1]:.4f}  "
              f"MAE {mae[-1]:.4f}  [{time.time() - t0:.0f}s]", flush=True)
    return np.array(cov), np.array(wid), np.array(mae)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--subject", choices=C.SUBJECTS, default=None)
    a = ap.parse_args()
    out = {}
    for subject in ([a.subject] if a.subject else C.SUBJECTS):
        S = C.load_subject(subject)
        if a.rerun:
            print(f"\n{S['label']}: retraining the magnitude readout ({a.reps} repeats)")
            cov, wid, mae = rerun(S, a.reps)
            np.savez(C.results_path("rq1", f"{subject}_calibration_rerun.npz"),
                     cov90=cov, width90=wid, mae=mae)
            src = "rerun"
        else:
            z = np.load(C.results_path("rq1", f"{subject}_calibration.npz"))
            cov, wid, mae = z["cov90"], z["width90"], z["mae"]
            src = "stored"
        mc = float(np.mean(S["se"]))
        sh = split_half(S["H_rep"])
        pc, ps, pm = PAPER[subject]
        print(f"\n{S['label']} ({src}, {len(cov)} repeats):")
        print(f"  coverage of 90% intervals  {100 * cov.mean():.1f} +- {100 * cov.std():.1f}   [paper {pc} +- {ps}]")
        print(f"  mean interval width        {wid.mean():.4f}")
        print(f"  mean absolute error        {mae.mean():.4f} +- {mae.std():.4f}   [paper {pm:.3f}]")
        print(f"  mean label MC s.e.         {mc:.4f}  ->  model error / MC error = {mae.mean() / mc:.1f}x"
              f"   [paper 6.5x]")
        print(f"  split-half reliability     {sh:.3f}   [paper 0.98]")
        out[subject] = {"source": src, "cov90": float(cov.mean()), "cov90_sd": float(cov.std()),
                        "width90": float(wid.mean()), "mae": float(mae.mean()),
                        "mae_sd": float(mae.std()), "label_mc_se": mc,
                        "mae_over_mc_se": float(mae.mean() / mc), "split_half": sh,
                        "per_repeat": {"cov90": cov.tolist(), "width90": wid.tolist(),
                                       "mae": mae.tolist()}}
    fp = C.results_path("rq1", "calibration" + ("_rerun" if a.rerun else "") + ".json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\nsaved {fp}")


if __name__ == "__main__":
    main()
