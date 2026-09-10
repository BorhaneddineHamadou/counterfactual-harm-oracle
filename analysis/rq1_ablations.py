"""RQ1 ablations of the field tier (paper Sec. 5.2 and Sec. 7).

Four studies, all on the paper's protocol (10 repeats x 5 scenario-level
folds, test rows = run-0 traces against their own M=100 Tier-1 label):

  corpus    Table 1 row "ablation: no replay corpus": the same estimator
            trained on the 600 nominal rows only.
  sharing   what label sharing buys, and which part does the work
            (reference-only / k executions / replay rows / crashed replays).
  target    the gain comes from the target, not the model: same estimator,
            features, folds and seeds, only the training target (collision
            bit, realized impact speed) or the feature set (1 or 4 scalars)
            changes.
  tau       the tie-threshold rule of Sec. 7: fold-internal 1-SE selection
            against the declared tau = 0.05.

`--from-stored` (default) reads the precomputed out-of-fold arrays in
results/rq1/; `--rerun` retrains (corpus: TF ~10 min, openpilot ~30 min;
sharing/target: hours per subject) and saves results/rq1/<subject>_<study>_rerun.npz.
Every number is printed next to the paper's value in brackets, and a JSON
summary goes to results/rq1/ablations.json.

Usage:  python analysis/rq1_ablations.py corpus|sharing|target|tau|all
            [--subject openpilot|transfuser] [--rerun] [--arms a b ...]
            [--reps 10] [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima import field_tier as FT                  # noqa: E402
from proxima.injury import occupant                   # noqa: E402
from proxima.metrics import apfd_h, ranks, zlog       # noqa: E402

R = 10
SHARING_ARMS = ["ref_only", "nom_k5", "ref_rep", "full",
                "full_nocrash", "full_nocrash_rw"]
#              arm             corpus  features  target
TARGET_ARMS = [("H_full", "full", "f71", "H"),
               ("binary_full", "full", "f71", "binary"),
               ("sever_full", "full", "f71", "severity"),
               ("H_nom", "nom", "f71", "H"),
               ("binary_nom", "nom", "f71", "binary"),
               ("sever_nom", "nom", "f71", "severity"),
               ("H_clear", "full", "clear", "H"),
               ("H_scal4", "full", "scal4", "H")]
COL_TTC, COL_CLEAR, COL_VCRIT, COL_DV = 0, 1, 2, 3
FEATSETS = {"f71": None, "clear": [COL_CLEAR],
            "scal4": [COL_TTC, COL_CLEAR, COL_VCRIT, COL_DV]}

# paper values, for the side-by-side print
PAPER = {
    "corpus": {"openpilot": dict(apfd=.787, rho=.668, gain=.06),
               "transfuser": dict(apfd=.885, rho=.755, gain=.035)},
    "sharing": {"openpilot": dict(ref_only_vs_full=-.074, full_minus_ref_rep=+.012),
                "transfuser": dict(ref_only_vs_full=-.088, full_minus_ref_rep=.000,
                                   nocrash_vs_full=-.028, ref_only_rho=.692,
                                   clearance_rho=.704)},
    "target": {"openpilot": dict(H=.730, binary=.482, d_binary=-.248, d_binary_nom=-.187,
                                 scal4=.617, d_scal4=-.112, clear=.463,
                                 replay_gain_H=.062, replay_gain_free=.001,
                                 nc_free=(-.053, .074), nc_H=.500),
               "transfuser": dict(H=.780, binary=.553, d_binary=-.227, d_binary_nom=-.200,
                                  scal4=.710, d_scal4=-.070, clear=.664,
                                  replay_gain_H=.028, replay_gain_free=.001,
                                  nc_free=(-.053, .074), nc_H=.673, scal4_apfd=.895,
                                  H_apfd=.885)},
    "tau": dict(cv=(.714, .048), declared=(.729, .034)),
}


def stat(vals):
    v = np.asarray(vals, float)
    return float(np.nanmean(v)), float(np.nanstd(v))


def ms(vals, nd=3):
    m, s = stat(vals)
    return f"{m:.{nd}f} +- {s:.{nd}f}"


def summarize(oofs, y, NC):
    rho = np.array([spearmanr(s, y).statistic for s in oofs])
    with np.errstate(all="ignore"):
        nc = np.array([spearmanr(s[NC], y[NC]).statistic if np.std(s[NC]) > 0
                       else np.nan for s in oofs])
    ap = np.array([apfd_h(s, y) for s in oofs])
    return dict(rho=rho, rho_nc=nc, apfd=ap)


def paired_line(name, a, b, paper=None):
    d = np.asarray(a) - np.asarray(b)
    tag = f"  [paper {paper:+.3f}]" if paper is not None else ""
    return (f"  {name:26s} {d.mean():+.3f} +- {d.std():.3f}  "
            f"(min {d.min():+.3f}, max {d.max():+.3f}, all<0: {bool((d < 0).all())}){tag}")


# ================================================================ corpus
def corpus_rep(S, rep):
    """exp29/exp30: nominal rows only, unweighted, 3 seeds, seed = rep*100+fold*10+s."""
    y, n = S["y"], len(S["y"])
    oof = np.full(n, np.nan)
    for fi, te in enumerate(C.folds_for(rep, n)):
        tri = np.setdiff1d(np.arange(n), te)
        nrows = np.concatenate([np.where(S["aug_row"] == i)[0] for i in tri])
        Xtr, ytr = S["Xa2"][nrows], y[S["aug_row"][nrows]]
        Xq = S["X2"][te]
        p = np.zeros(len(te)); tz = np.zeros(len(te)); tr = np.zeros(len(te))
        for s in range(3):
            rs = rep * 100 + fi * 10 + s
            m = HistGradientBoostingClassifier(random_state=rs, **FT.HGB)
            m.fit(Xtr, ytr > 0); p += m.predict_proba(Xq)[:, 1]
            m = HistGradientBoostingRegressor(random_state=rs, **FT.HGB)
            m.fit(Xtr, zlog(ytr)); tz += m.predict(Xq)
            m = HistGradientBoostingRegressor(random_state=rs, **FT.HGB)
            m.fit(Xtr, ytr); tr += m.predict(Xq)
        p /= 3; tz /= 3; tr /= 3
        oof[te] = np.where(p < FT.TAU, 0.0, 1.0 + ranks(p) + ranks(tz) + ranks(tr))
    return oof


def run_corpus(S, a, out):
    path = C.results_path("rq1", f"{S['name']}_corpus_ablation.npz")
    if a.rerun or not os.path.exists(path):
        from joblib import Parallel, delayed
        t0 = time.time()
        oofs = np.array(Parallel(n_jobs=a.n_jobs, backend="loky")(
            delayed(corpus_rep)(S, rep) for rep in range(a.reps)))
        np.savez_compressed(path, oofs=oofs, y=S["y"])
        print(f"  retrained {a.reps} repeats in {time.time()-t0:.0f}s -> {path}")
    Z = np.load(path)
    y, NC = S["y"], S["NC"]
    ab = summarize(Z["oofs"], y, NC)
    ft = summarize(C.load_oof(S["name"])["oofs"], y, NC)
    P = PAPER["corpus"][S["name"]]
    print(f"\n=== {S['label']}: no-replay-corpus ablation (600 nominal rows) ===")
    print(f"  ablation   APFD_H {ms(ab['apfd'])}  [paper {P['apfd']:.3f}]   "
          f"r_s {ms(ab['rho'])}  [paper {P['rho']:.3f}]   r_s_nc {ms(ab['rho_nc'])}")
    print(f"  field tier APFD_H {ms(ft['apfd'])}   r_s {ms(ft['rho'])}")
    gain = ft["rho"].mean() - ab["rho"].mean()
    print(f"  replay corpus worth {gain:+.3f} of r_s  [paper +{P['gain']:.3f}]")
    out[f"corpus/{S['name']}"] = dict(apfd=stat(ab["apfd"]), rho=stat(ab["rho"]),
                                      rho_nc=stat(ab["rho_nc"]), gain_rho=gain)


# =============================================================== sharing
def sharing_corpus(S, arm, tri):
    flags = dict(
        ref_only=dict(nominal_only=True, run0_only=True),
        nom_k5=dict(nominal_only=True),
        ref_rep=dict(run0_only=True),
        full=dict(),
        full_nocrash=dict(drop_crashed_replays=True),
        full_nocrash_rw=dict(drop_crashed_replays=True, reweight=True))[arm]
    return FT.corpus(S, tri, **flags)


def fit_rep(S, rep, corpus_fn, cols=None):
    """exp31/exp32 protocol: seeds rep*10+s, n_seeds per subject, p in the
    survivor score on both subjects."""
    y, n = S["y"], len(S["y"])
    comp = np.full(n, np.nan); tzo = np.full(n, np.nan); po = np.full(n, np.nan)
    ntr = 0
    for te in C.folds_for(rep, n):
        tri = np.setdiff1d(np.arange(n), te)
        Xtr, ytr, w = corpus_fn(tri)
        ntr = len(ytr)
        Xq = S["X2"][te] if cols is None else S["X2"][te][:, cols]
        parts = FT.fit_readouts(Xtr, ytr, w, rep, S["n_seeds"])
        p, tz, tr = FT.predict_readouts(parts, Xq)
        comp[te] = FT.structured_score(p, tz, tr, FT.TAU, p_in_tail=True)
        tzo[te] = tz; po[te] = p
    return comp, tzo, po, ntr


def load_or_rerun(S, a, study, arms, corpus_of, cols_of=lambda arm: None):
    stored = C.results_path("rq1", f"{S['name']}_{study}.npz")
    Z = np.load(stored) if os.path.exists(stored) else None
    res = {}
    if a.rerun:
        from joblib import Parallel, delayed
        rerun_path = C.results_path("rq1", f"{S['name']}_{study}_rerun.npz")
        saved = {}
        for arm in arms:
            t0 = time.time()
            rr = Parallel(n_jobs=a.n_jobs, backend="loky")(
                delayed(fit_rep)(S, rep, corpus_of(arm), cols_of(arm))
                for rep in range(a.reps))
            res[arm] = dict(comp=np.array([r[0] for r in rr]), tz=np.array([r[1] for r in rr]),
                            p=np.array([r[2] for r in rr]), ntr=rr[0][3])
            print(f"  [{time.time()-t0:6.0f}s] rerun {arm:16s} rows={rr[0][3]:6d}  "
                  f"r_s {ms(summarize(res[arm]['comp'], S['y'], S['NC'])['rho'])}")
            if Z is not None and f"{arm}_comp" in Z.files:
                d = np.abs(res[arm]["comp"] - Z[f"{arm}_comp"][:a.reps]).max()
                print(f"      max |rerun - stored| over the first {a.reps} repeats: {d:.2e}")
            for k, v in res[arm].items():
                saved[f"{arm}_{k}"] = v
        np.savez_compressed(rerun_path, y=S["y"], NC=S["NC"], **saved)
        print(f"  saved {rerun_path}")
    if Z is not None:
        for arm in arms:
            if arm not in res and f"{arm}_comp" in Z.files:
                res[arm] = dict(comp=Z[f"{arm}_comp"], tz=Z[f"{arm}_tz"], p=Z[f"{arm}_p"],
                                ntr=int(Z[f"{arm}_ntr"]) if f"{arm}_ntr" in Z.files else -1)
    if not res:
        raise SystemExit(f"no stored {study} results for {S['name']}; use --rerun")
    return res


def run_sharing(S, a, out):
    arms = a.arms or SHARING_ARMS
    res = load_or_rerun(S, a, "label_sharing", arms,
                        lambda arm: (lambda tri: sharing_corpus(S, arm, tri)))
    y, NC = S["y"], S["NC"]
    P = PAPER["sharing"][S["name"]]
    print(f"\n=== {S['label']}: label-sharing ablation (training corpus only; "
          f"test rows = run-0 traces vs their own M=100 label) ===")
    print(f"  {'arm':16s} {'rows':>6s} {'r_s':>16s} {'r_s_nc':>16s} {'APFD_H':>16s}")
    summ = {arm: summarize(r["comp"], y, NC) for arm, r in res.items()}
    for arm in arms:
        if arm in summ:
            s = summ[arm]
            print(f"  {arm:16s} {res[arm]['ntr']:6d} {ms(s['rho']):>16s} "
                  f"{ms(s['rho_nc']):>16s} {ms(s['apfd']):>16s}")
    if "full" in summ:
        print("  paired deltas of r_s vs full (same folds, same seeds):")
        for arm in arms:
            if arm == "full" or arm not in summ:
                continue
            paper = {"ref_only": P.get("ref_only_vs_full"),
                     "ref_rep": -P["full_minus_ref_rep"],
                     "full_nocrash": P.get("nocrash_vs_full")}.get(arm)
            print(paired_line(f"{arm} - full", summ[arm]["rho"], summ["full"]["rho"], paper))
        if "ref_rep" in summ:
            d = summ["full"]["rho"] - summ["ref_rep"]["rho"]
            print(f"  sharing across the k executions (full - ref_rep): {d.mean():+.3f}  "
                  f"[paper {P['full_minus_ref_rep']:+.3f}]")
    if S["name"] == "transfuser" and "ref_only" in summ:
        clr = spearmanr(C.baselines(S)["min clearance"], y).statistic
        print(f"  ref_only r_s {summ['ref_only']['rho'].mean():.3f} vs min clearance "
              f"{clr:.3f}  [paper .692 vs .704]")
    out[f"sharing/{S['name']}"] = {arm: dict(rho=stat(s["rho"]), rho_nc=stat(s["rho_nc"]),
                                             apfd=stat(s["apfd"]))
                                   for arm, s in summ.items()}


# ================================================================ target
def target_corpus(S, spec, tri):
    """exp32: same rows as the shipped corpus, target and feature set swapped."""
    _, which, feats, target = spec
    y = S["y"]
    nom_contact_own = S["Xa2"][:, COL_CLEAR] == 0.0     # clearance 0 iff contact
    nom_dv_own = S["Xa2"][:, COL_DV]
    nrows = np.flatnonzero(np.isin(S["aug_row"], tri))
    Xn = S["Xa2"][nrows]
    if target == "H":
        yn = y[S["aug_row"][nrows]]
    elif target == "binary":
        yn = nom_contact_own[nrows].astype(float)
    else:
        yn = np.where(nom_contact_own[nrows], occupant(nom_dv_own[nrows]), 0.0)
    wn = np.full(len(nrows), FT.W_NOMINAL)
    if which == "nom":
        X, yy, w = Xn, yn, wn
    else:
        rmask = np.isin(S["rep_scn"], tri)
        if target == "H":
            yr = y[S["rep_scn"][rmask]]
        elif target == "binary":
            yr = S["rep_contact"][rmask].astype(float)
        else:
            yr = S["rep_injury"][rmask]
        X = np.vstack([Xn, S["Xr2"][rmask]])
        yy = np.concatenate([yn, yr])
        w = np.concatenate([wn, np.full(int(rmask.sum()), FT.W_REPLAY)])
    cols = FEATSETS[feats]
    return (X if cols is None else X[:, cols]), yy, w


def run_target(S, a, out):
    specs = {s[0]: s for s in TARGET_ARMS}
    arms = a.arms or [s[0] for s in TARGET_ARMS]
    res = load_or_rerun(S, a, "target_ablation", arms,
                        lambda arm: (lambda tri: target_corpus(S, specs[arm], tri)),
                        lambda arm: FEATSETS[specs[arm][2]])
    y, NC = S["y"], S["NC"]
    P = PAPER["target"][S["name"]]
    summ = {arm: summarize(r["comp"], y, NC) for arm, r in res.items()}
    print(f"\n=== {S['label']}: target and feature ablation (estimator, features, "
          f"folds, seeds, tie rule fixed) ===")
    print(f"  {'arm':12s} {'corpus':>6s} {'feat':>5s} {'target':>8s} {'r_s':>16s} "
          f"{'r_s_nc':>16s} {'APFD_H':>16s}")
    for arm in arms:
        if arm in summ:
            _, which, feats, target = specs[arm]
            s = summ[arm]
            print(f"  {arm:12s} {which:>6s} {feats:>5s} {target:>8s} {ms(s['rho']):>16s} "
                  f"{ms(s['rho_nc']):>16s} {ms(s['apfd']):>16s}")
    hb = C.baselines(S)["binary verdict"]
    print(f"  raw binary verdict: r_s {spearmanr(hb, y).statistic:.3f}  "
          f"APFD_H {apfd_h(hb, y):.3f}  (the binary-target GBM should reproduce it)")
    if "H_full" in summ:
        print("  paired deltas of r_s vs H_full:")
        for arm, paper in (("binary_full", P["d_binary"]), ("sever_full", None),
                           ("H_nom", -P["replay_gain_H"]), ("H_clear", None),
                           ("H_scal4", P["d_scal4"])):
            if arm in summ:
                print(paired_line(f"{arm} - H_full", summ[arm]["rho"], summ["H_full"]["rho"], paper))
    if "H_nom" in summ and "binary_nom" in summ:
        print(paired_line("binary_nom - H_nom", summ["binary_nom"]["rho"],
                          summ["H_nom"]["rho"], P["d_binary_nom"]))
    for tgt in ("H", "binary", "sever"):
        f, n_ = f"{tgt}_full", f"{tgt}_nom"
        if f in summ and n_ in summ:
            d = summ[f]["rho"] - summ[n_]["rho"]
            paper = P["replay_gain_H"] if tgt == "H" else P["replay_gain_free"]
            print(f"  replay rows worth to the {tgt:6s} target ({f} - {n_}): "
                  f"{d.mean():+.3f} +- {d.std():.3f}  [paper {paper:+.3f}]")
    # ungated readouts on the runs that never crash (no tie gate, no fusion)
    print(f"  ungated readouts on the {int(NC.sum())} non-collision runs "
          f"({int((y[NC] > 0).sum())} with nonzero harm), r_s vs Tier-1 harm:")
    with np.errstate(all="ignore"):
        for arm in arms:
            if arm not in res:
                continue
            rz = [spearmanr(t[NC], y[NC]).statistic for t in res[arm]["tz"]]
            rp = [spearmanr(t[NC], y[NC]).statistic for t in res[arm]["p"]]
            print(f"    {arm:12s} log readout {ms(rz):>16s}   classifier {ms(rp):>16s}")
    print(f"    [paper: free-label targets between -.053 and +.074; harm target "
          f"{P['nc_H']:.3f}]")
    if S["name"] == "transfuser" and "H_scal4" in summ and "H_full" in summ:
        print(f"  four scalars edge the full set on APFD_H: {summ['H_scal4']['apfd'].mean():.3f} "
              f"vs {summ['H_full']['apfd'].mean():.3f}  [paper .895 vs .885]")
    out[f"target/{S['name']}"] = {arm: dict(rho=stat(s["rho"]), rho_nc=stat(s["rho_nc"]),
                                            apfd=stat(s["apfd"]))
                                  for arm, s in summ.items()}


# =================================================================== tau
def run_tau(a, out):
    y = C.load_oof("openpilot")["y"]
    ft = summarize(C.load_oof("openpilot")["oofs"], y, ~np.zeros(len(y), bool))
    path = C.results_path("rq1", "openpilot_field_tier_oof_cvtau.npz")
    if not os.path.exists(path):
        raise SystemExit("run: python analysis/rq1_train_field_tier.py openpilot --tau cv")
    cv = summarize(np.load(path)["oofs"], y, ~np.zeros(len(y), bool))
    P = PAPER["tau"]
    print("\n=== openpilot: tie threshold, fold-internal 1-SE selection vs declared 0.05 ===")
    print(f"  inner-CV tau   r_s {ms(cv['rho'])}  [paper {P['cv'][0]:.3f} +- {P['cv'][1]:.3f}]")
    print(f"  declared 0.05  r_s {ms(ft['rho'])}  [paper {P['declared'][0]:.3f} +- {P['declared'][1]:.3f}]")
    print("  (rerun: python analysis/rq1_train_field_tier.py openpilot --tau cv)")
    out["tau/openpilot"] = dict(cv=stat(cv["rho"]), declared=stat(ft["rho"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=("corpus", "sharing", "target", "tau", "all"))
    ap.add_argument("--subject", choices=C.SUBJECTS, default=None)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--from-stored", action="store_true", help="(default)")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=R)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    subjects = [a.subject] if a.subject else list(C.SUBJECTS)
    jp = C.results_path("rq1", "ablations.json")
    out = json.load(open(jp)) if os.path.exists(jp) else {}
    for name in subjects:
        S = C.load_subject(name)
        if a.study in ("corpus", "all"):
            run_corpus(S, a, out)
        if a.study in ("sharing", "all"):
            run_sharing(S, a, out)
        if a.study in ("target", "all"):
            run_target(S, a, out)
    if a.study in ("tau", "all"):
        run_tau(a, out)
    json.dump(out, open(jp, "w"), indent=2, default=float)
    print(f"\nsaved {jp}")


if __name__ == "__main__":
    main()
