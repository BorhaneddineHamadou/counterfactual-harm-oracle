"""RQ3, sensitivity to the declared kernel scale (paper Sec. 5.4, Table 3).

The same 150 references of each subject were re-replayed under the kernel
rescaled to alpha in {0.5, 2} (M = 40 per scale on openpilot, 20 on
TransFuser), with common random numbers against the reference campaign
(the alpha = 1 arm is the first M reference replays, key H_1_crn).

Two levels:
  labels    Spearman agreement of the Tier-1 harm orderings between scales
            (all references / references with nonzero harm under either
            scale), what the scale moves (share of references with H > 0,
            mean / median H), and the paired per-reference direction.
  oracles   field oracles distilled independently under each scale's own
            labels and replay corpus (out of fold, 10 repeats), their
            pairwise agreement, and their agreement with the canonical
            M = 100 labels ("declaring the wrong scale").

Oracle scores are read from results/rq3/<subject>_kernel_oracle_scores.npz
(keys a05, crn, a20; (10, 150)). `--retrain` recomputes them with the
estimator of proxima/field_tier.py: corpus = 750 nominal rows + that arm's
replay traces (data/<subject>/rq2/features_a05.npz / a20; the tier-1
replays with rep_m < M for the crn arm), labels from
data/<subject>/rq2/H_by_scale.npz, 3 seeds per readout with
random_state = (rep*100 + fold)*10 + seed, tau = 0.05 with rank(p) in the
survivor band on both subjects (the retraining driver used the openpilot
composite for both).

Usage:  python analysis/rq3_kernel_rescaling.py [--retrain] [--reps 10]
                                                [--subject openpilot|transfuser]
Output: results/rq3/table3.json (and, with --retrain,
        results/rq3/<subject>_kernel_oracle_scores_retrained.npz)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima import field_tier as FT                  # noqa: E402
from proxima.metrics import ranks                     # noqa: E402

NT = 1e-4                      # non-trivial harm floor
M_RQ2 = {"openpilot": 40, "transfuser": 20}
ARMS = ("a05", "crn", "a20")
ALPHA = {"a05": 0.5, "crn": 1.0, "a20": 2.0}
PAIRS = (("a05", "crn"), ("crn", "a20"), ("a05", "a20"))


def load_labels(S):
    z = np.load(C.data_path(S["name"], "rq2", "H_by_scale.npz"))
    assert (z["sids"].astype(int) == S["sids"]).all()
    return {"a05": z["H_0.5"], "crn": z["H_1_crn"], "a20": z["H_2.0"],
            "full": z["H_1_full"]}


def label_level(S, Y):
    out = {"pairs": {}, "nonzero_share": {}, "mean_H": {}, "median_H": {}}
    print(f"\n=== {S['label']}: Tier-1 labels under alpha 0.5 | 1 | 2 "
          f"(M={M_RQ2[S['name']]} per scale, CRN) ===")
    for a, b in PAIRS:
        nt = (Y[a] > NT) | (Y[b] > NT)
        r_all = spearmanr(Y[a], Y[b]).statistic
        r_nt = spearmanr(Y[a][nt], Y[b][nt]).statistic
        out["pairs"][f"{ALPHA[a]}_vs_{ALPHA[b]}"] = {
            "rho_all": float(r_all), "rho_nontrivial": float(r_nt), "n_nontrivial": int(nt.sum())}
        print(f"  alpha {ALPHA[a]} vs {ALPHA[b]}: rho all {r_all:.3f} | "
              f"non-trivial (n={int(nt.sum())}) {r_nt:.3f}")
    out["min_rho_all"] = min(v["rho_all"] for v in out["pairs"].values())
    out["min_rho_nontrivial"] = min(v["rho_nontrivial"] for v in out["pairs"].values())
    for k in ARMS:
        out["nonzero_share"][str(ALPHA[k])] = float((Y[k] > 0).mean())
        out["mean_H"][str(ALPHA[k])] = float(Y[k].mean())
        out["median_H"][str(ALPHA[k])] = float(np.median(Y[k]))
    print(f"  min over pairs: all {out['min_rho_all']:.2f}, non-trivial "
          f"{out['min_rho_nontrivial']:.2f}")
    print("  refs with H>0 : " + " | ".join(f"{100*out['nonzero_share'][str(ALPHA[k])]:.0f}%" for k in ARMS))
    print("  mean H        : " + " | ".join(f"{out['mean_H'][str(ALPHA[k])]:.4f}" for k in ARMS))
    print("  median H      : " + " | ".join(f"{out['median_H'][str(ALPHA[k])]:.4f}" for k in ARMS))
    d = Y["a20"] - Y["crn"]
    up, down = int((d > 0).sum()), int((d < 0).sum())
    p = float(wilcoxon(Y["a20"], Y["crn"]).pvalue) if (d != 0).any() else np.nan
    out["paired_2_vs_1"] = {"rises": up, "falls": down, "wilcoxon_p": p}
    print(f"  paired per-reference harm, alpha 2 vs 1: {up} rise, {down} fall, "
          f"Wilcoxon p = {p:.1e}")
    return out


def oracle_level(S, Y, scores):
    out = {"pairs": {}, "vs_own_labels": {}, "vs_canonical_M100": {}}
    print(f"\n=== {S['label']}: field oracles distilled under each scale (OOF, "
          f"{scores['crn'].shape[0]} repeats) ===")
    for a, b in PAIRS:
        nt = (Y[a] > NT) | (Y[b] > NT)
        r_all = [spearmanr(sa, sb).statistic for sa, sb in zip(scores[a], scores[b])]
        r_nt = [spearmanr(sa[nt], sb[nt]).statistic for sa, sb in zip(scores[a], scores[b])]
        out["pairs"][f"{ALPHA[a]}_vs_{ALPHA[b]}"] = {
            "rho_all": float(np.mean(r_all)), "rho_all_sd": float(np.std(r_all)),
            "rho_nontrivial": float(np.mean(r_nt)), "rho_nontrivial_sd": float(np.std(r_nt)),
            "n_nontrivial": int(nt.sum())}
        print(f"  {ALPHA[a]} vs {ALPHA[b]}: all {np.mean(r_all):.3f}+-{np.std(r_all):.3f} | "
              f"non-trivial (n={int(nt.sum())}) {np.mean(r_nt):.3f}+-{np.std(r_nt):.3f}")
    out["min_rho_all"] = min(v["rho_all"] for v in out["pairs"].values())
    out["min_rho_nontrivial"] = min(v["rho_nontrivial"] for v in out["pairs"].values())
    print(f"  min over pairs: all {out['min_rho_all']:.2f}, non-trivial "
          f"{out['min_rho_nontrivial']:.2f}")
    for k in ARMS:
        r = [spearmanr(s, Y[k]).statistic for s in scores[k]]
        out["vs_own_labels"][str(ALPHA[k])] = float(np.mean(r))
        rc = [spearmanr(s, Y["full"]).statistic for s in scores[k]]
        out["vs_canonical_M100"][str(ALPHA[k])] = float(np.mean(rc))
    vc = out["vs_canonical_M100"]
    out["cost_halving"] = vc["0.5"] - vc["1.0"]
    out["cost_doubling"] = vc["2.0"] - vc["1.0"]
    print("  oracle r_s vs canonical M=100 labels, alpha 0.5 | 1 | 2: "
          + " | ".join(f"{vc[str(ALPHA[k])]:.2f}" for k in ARMS)
          + f"   (halving {out['cost_halving']:+.3f}, doubling {out['cost_doubling']:+.3f})")
    return out


# ------------------------------------------------------------- retrain
def arm_corpora(S, Y):
    """(replay features, replay scenario row) per arm."""
    corp = {}
    for tag in ("a05", "a20"):
        z = np.load(C.data_path(S["name"], "rq2", f"features_{tag}.npz"))
        corp[tag] = (np.hstack([z["X71"], z["Xtraj"]]),
                     np.array([S["sid_row"][int(s)] for s in z["rep_sid"]]))
    sel = S["rep_m"] < M_RQ2[S["name"]]
    corp["crn"] = (S["Xr2"][sel], S["rep_scn"][sel])
    return corp


def fit_arm(S, yk, Xr, rscn, tri, tei, seed):
    nrows = np.concatenate([np.where(S["aug_row"] == i)[0] for i in tri])
    rmask = np.isin(rscn, tri)
    Xtr = np.vstack([S["Xa2"][nrows], Xr[rmask]])
    ytr = np.concatenate([yk[S["aug_row"][nrows]], yk[rscn[rmask]]])
    w = np.concatenate([np.full(len(nrows), FT.W_NOMINAL),
                        np.full(rmask.sum(), FT.W_REPLAY)])
    parts = FT.fit_readouts(Xtr, ytr, w, seed, 3)
    return FT.predict_readouts(parts, S["X2"][tei])


def retrain(S, Y, reps):
    corp = arm_corpora(S, Y)
    n = len(S["y"])
    scores = {k: [] for k in ARMS}
    t0 = time.time()
    for rep in range(reps):
        parts = {k: [np.full(n, np.nan) for _ in range(3)] for k in ARMS}
        for fi, te in enumerate(C.folds_for(rep, n)):
            tr = np.setdiff1d(np.arange(n), te)
            for k in ARMS:
                p, tz, tr2 = fit_arm(S, Y[k], *corp[k], tr, te, rep * 100 + fi)
                parts[k][0][te] = p
                parts[k][1][te] = tz
                parts[k][2][te] = tr2
        for k in ARMS:
            p, tz, tr2 = parts[k]
            scores[k].append(np.where(p < FT.TAU, 0.0, 1.0 + ranks(p) + ranks(tz) + ranks(tr2)))
        print(f"  [{time.time()-t0:.0f}s] rep {rep} done", flush=True)
    return {k: np.array(v) for k, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--subject", choices=C.SUBJECTS, default=None)
    a = ap.parse_args()
    subjects = [a.subject] if a.subject else list(C.SUBJECTS)
    out = {}
    for s in subjects:
        S = C.load_subject(s)
        Y = load_labels(S)
        res = {"M_per_scale": M_RQ2[s], "labels": label_level(S, Y)}
        stored = np.load(C.results_path("rq3", f"{s}_kernel_oracle_scores.npz"))
        scores = {k: stored[k] for k in ARMS}
        if a.retrain:
            new = retrain(S, Y, a.reps)
            fp = C.results_path("rq3", f"{s}_kernel_oracle_scores_retrained.npz")
            np.savez_compressed(fp, **new, y_a05=Y["a05"], y_crn=Y["crn"],
                                y_a20=Y["a20"], y_full=Y["full"])
            for k in ARMS:
                r = min(a.reps, stored[k].shape[0])
                print(f"  {k}: max |retrained - stored| over {r} repeat(s) = "
                      f"{np.abs(new[k][:r] - stored[k][:r]).max():.2e}")
            print(f"  saved {fp}")
            scores = new
        res["oracles"] = oracle_level(S, Y, scores)
        out[s] = res
    fp = C.results_path("rq3", "table3.json")
    json.dump(out, open(fp, "w"), indent=1, default=float)
    print(f"\nsaved {fp}")
    print("\npaper Table 3: labels min (all | non-triv.) 0.64 | 0.34 openpilot, "
          "0.80 | 0.68 TransFuser; oracles 0.71 | 0.57 and 0.77 | 0.57;\n"
          "  refs with H>0 14|18|23% and 55|61|75%; oracle r_s vs canonical "
          ".60|.70|.63 and .72|.79|.76")


if __name__ == "__main__":
    main()
