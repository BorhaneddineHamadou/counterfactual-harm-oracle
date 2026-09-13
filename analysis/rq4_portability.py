"""RQ4 (Portability): point each subject's field oracle at the other subject.

Zero new simulation: the two reference stores of RQ1 are crossed in three
steps (paper Sec. 4.3 "RQ4 in detail", results Sec. 5.5, Fig. 3).

  zero-shot     the foreign oracle, distilled on ALL 150 scenarios of the
                source subject, scores the destination's run-0 traces; its
                ordering is judged against the destination's Tier-1 truth,
                and its conformal interval (calibrated on a seeded 75/25
                split of the SOURCE) is checked for coverage;
  self-awareness does the ensemble's member disagreement (variance of the
                log readout across seeds) rise on the foreign system, and
                what share of foreign runs exceeds the native escalation
                threshold (the source's 90th percentile of disagreement)?
  re-anchoring  keep the foreign ensemble frozen and refit only the
                conformal quantile on n in {10, 25, 50, 100} locally labelled
                runs (20 seeded draws), with labels at M=100 and at the cheap
                M=30 (the first 30 replays of each reference); coverage is
                measured on the remaining local references.

The local ceiling is the subject's own out-of-fold field tier
(results/rq1/<subject>_field_tier_oof.npz).

Usage:  python analysis/rq4_portability.py [--out results/rq4/portability.json]
Runtime: ~10 min on a 32-core node (two source ensembles x 3 seeds).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima import field_tier as FT                  # noqa: E402
from proxima.metrics import ranks, surfaced, zlog     # noqa: E402

N_GRID = (10, 25, 50, 100)
N_SEEDS = 20
M_CHEAP = 30
N_MODEL_SEEDS = 3
CAMPAIGN_REPLAYS = 15000


def train(S, tri, seed=0):
    """The four rank readouts of the shipped oracle, fitted on scenarios `tri`
    of subject S (log / raw / classifier on the full corpus, plus the
    nominal-only log readout), N_MODEL_SEEDS seeds each."""
    Xtr, ytr, w, nrows = FT.corpus_ordered(S, tri)
    parts = FT.fit_readouts(Xtr, ytr, w, seed, N_MODEL_SEEDS)
    aug = []
    from sklearn.ensemble import HistGradientBoostingRegressor
    for s in range(N_MODEL_SEEDS):
        m = HistGradientBoostingRegressor(random_state=seed * 10 + s, **FT.HGB)
        m.fit(S["Xa2"][nrows], zlog(S["y"][S["aug_row"][nrows]]))
        aug.append(m)
    return {"parts": parts, "aug": aug}


def score(models, X):
    zs = np.stack([rz.predict(X) for _, rz, _ in models["parts"]])
    z = zs.mean(0)
    raw = np.mean([rr.predict(X) for _, _, rr in models["parts"]], axis=0)
    p = np.mean([c.predict_proba(X)[:, 1] for c, _, _ in models["parts"]], axis=0)
    s4 = FT.structured_score(p, z, raw, FT.TAU, p_in_tail=True)
    pt = FT.to_harm(z)
    unc = zs.var(0)                      # member disagreement, log scale
    return s4, pt, unc


def direction(src, dst):
    y, n = dst["y"], len(dst["y"])
    all_src = np.arange(len(src["y"]))
    mfull = train(src, all_src)
    rank4, pt, unc = score(mfull, dst["X2"])
    perm = np.random.default_rng(0).permutation(all_src)
    cal = perm[:int(0.25 * len(perm))]
    m75 = train(src, perm[int(0.25 * len(perm)):])
    _, pt_cal, unc_cal = score(m75, src["X2"][cal])
    qhat_f = FT.conformal_qhat(pt_cal - src["y"][cal])
    _, pt75, unc75 = score(m75, dst["X2"])
    err = np.abs(pt75 - y)
    res = {
        "rho": float(spearmanr(rank4, y).statistic),
        "rho_nc": float(spearmanr(rank4[dst["NC"]], y[dst["NC"]]).statistic),
        "s10": surfaced(rank4, y, .10), "s20": surfaced(rank4, y, .20),
        "zero_shot_cov90": float(np.mean(err <= qhat_f)),
        "zero_shot_width90": float(2 * qhat_f),
        "foreign_qhat": float(qhat_f),
        "awareness_corr": float(spearmanr(unc75, err).statistic),
        "awareness_err_ratio_top10": float(err[np.argsort(-unc75)[:n // 10]].mean()
                                           / max(err.mean(), 1e-12)),
        "unc_native_mean": float(unc_cal.mean()),
        "unc_foreign_mean": float(unc75.mean()),
        "unc_rise_ratio": float(unc75.mean() / max(unc_cal.mean(), 1e-12)),
        "esc_rate_foreign_at_native_q90": float(np.mean(unc75 > np.quantile(unc_cal, 0.9))),
    }
    y30 = np.nanmean(dst["H_rep"][:, :M_CHEAP], axis=1)
    for label_tag, ylab in (("M100", y), ("M30", y30)):
        curve = {}
        for nn in N_GRID:
            covs, wids = [], []
            for s in range(N_SEEDS):
                rng = np.random.default_rng(1000 + s)
                pick = rng.choice(n, nn, replace=False)
                rest = np.setdiff1d(np.arange(n), pick)
                qh = FT.conformal_qhat(pt75[pick] - ylab[pick])
                covs.append(np.mean(np.abs(pt75[rest] - y[rest]) <= qh))
                wids.append(2 * qh)
            curve[str(nn)] = {"cov90": float(np.mean(covs)), "cov90_sd": float(np.std(covs)),
                              "width90": float(np.mean(wids))}
        res[f"reanchor_{label_tag}"] = curve
        # the frozen readouts are untouched by recalibration
        res[f"rho_after_reanchor_{label_tag}"] = res["rho"]
    return res, qhat_f


def native_qhat(S):
    """The subject's own conformal quantile on the same seeded 75/25 split."""
    perm = np.random.default_rng(0).permutation(len(S["y"]))
    cal = perm[:int(0.25 * len(perm))]
    m75 = train(S, perm[int(0.25 * len(perm)):])
    _, pt_cal, _ = score(m75, S["X2"][cal])
    return float(FT.conformal_qhat(pt_cal - S["y"][cal]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.results_path("rq4", "portability.json"))
    a = ap.parse_args()
    t0 = time.time()
    subjects = {s: C.load_subject(s) for s in C.SUBJECTS}
    out = {}
    nq = {}
    for src_n, dst_n in (("openpilot", "transfuser"), ("transfuser", "openpilot")):
        src, dst = subjects[src_n], subjects[dst_n]
        tag = f"{src_n}->{dst_n}"
        print(f"=== {tag} ===", flush=True)
        res, qf = direction(src, dst)
        oof = C.load_oof(dst_n)
        res["local_ceiling"] = {
            "rho": float(np.mean([spearmanr(sc, dst["y"]).statistic for sc in oof["oofs"]])),
            "s10": float(np.mean([surfaced(sc, dst["y"], .10) for sc in oof["oofs"]]))}
        if dst_n not in nq:
            nq[dst_n] = native_qhat(dst)
        res["native_qhat_of_destination"] = nq[dst_n]
        res["foreign_over_native_width"] = float(qf / max(nq[dst_n], 1e-12))
        out[tag] = res
        print(f"  zero-shot rho {res['rho']:.3f} (local ceiling {res['local_ceiling']['rho']:.3f})"
              f"  cov90 {100*res['zero_shot_cov90']:.0f}%  foreign qhat {qf:.4f} = "
              f"{res['foreign_over_native_width']:.2f}x native")
        print(f"  disagreement rise x{res['unc_rise_ratio']:.1f}, "
              f"{100*res['esc_rate_foreign_at_native_q90']:.0f}% of foreign runs above the native "
              f"escalation threshold")
        for tag2 in ("M100", "M30"):
            c = res[f"reanchor_{tag2}"]
            print(f"  re-anchor {tag2}: " + "  ".join(
                f"n={k}: {100*v['cov90']:.1f}±{100*v['cov90_sd']:.1f}" for k, v in c.items()))
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    # Fig. 3 data
    fp = C.results_path("rq4", "fig3_reanchoring.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "labels", "n", "M", "replays", "cov90", "cov90_sd", "width90"])
        for tag, res in out.items():
            w.writerow([tag, "zero-shot", 0, 0, 0, res["zero_shot_cov90"], 0.0, res["zero_shot_width90"]])
            for lt, M in (("M30", 30), ("M100", 100)):
                for nn, v in res[f"reanchor_{lt}"].items():
                    w.writerow([tag, lt, nn, M, int(nn) * M, v["cov90"], v["cov90_sd"], v["width90"]])
    print(f"saved {a.out}\nsaved {fp}")
    print("paper: zero-shot r_s 0.31 / 0.26 (ceilings 0.79 / 0.73); cov 78% / 91%; "
          "disagreement x16, 81% escalated; n=25: 96.8±4.5 / 97.5±2.8")
    # "intervals 2.5x over-wide" (Fig. 3 caption): the width the TransFuser
    # oracle carries into the foreign world (2 qhat from its 75/25 split)
    # against the width of its own out-of-fold conformal intervals at home
    # (results/rq1/<source>_calibration.npz, analysis/rq1_calibration.py).
    for src_n, dst_n in (("openpilot", "transfuser"), ("transfuser", "openpilot")):
        try:
            w_home = float(np.mean(np.load(C.results_path(
                "rq1", f"{src_n}_calibration.npz"))["width90"]))
        except FileNotFoundError:
            continue
        r = out[f"{src_n}->{dst_n}"]
        r["source_native_oof_width90"] = w_home
        r["foreign_width_over_source_native"] = r["zero_shot_width90"] / w_home
        print(f"  {src_n}->{dst_n}: foreign 90% width {r['zero_shot_width90']:.4f} = "
              f"{r['foreign_width_over_source_native']:.2f}x the source oracle's native "
              f"out-of-fold width {w_home:.4f}"
              + ("   [paper: 2.5x over-wide]" if src_n == "transfuser" else ""))
    json.dump(out, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
