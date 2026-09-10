"""Tier 2, the field oracle (paper Sec. 3.2 and Sec. 3.4, "Field tier").

The estimator, as shipped for both subjects:

  corpus     every recorded trace of the training scenarios: the k=5
             nominal executions (weight 0.2) and the M=100 kernel replays
             (weight 0.02), each row labelled with its scenario's Tier-1
             harm ("learn from every trace the reference campaign paid for");
  inputs     71 single-trace descriptors + 71 checkpoint-trajectory
             descriptors (142 columns);
  readouts   gradient-boosted trees (sklearn HistGradientBoosting,
             400 iterations, learning rate 0.05), each averaged over
             `n_seeds` random seeds:
               p     H > 0 classifier  ("does the neighbourhood crash?")
               tz    log10(H + 1e-4) regressor (separates the small harms)
               tr    raw-H regressor          (separates the large harms)
  ordering   "say exactly zero when sure, order the rest":
               score = 0                       if p < tau
                     = 1 + rank(p) + rank(tz) + rank(tr)   otherwise
             tau = 0.05 is the declared conservative tie rule;
  magnitude  the log readout mapped back, 10**tz - 1e-4, wrapped in a
             split-conformal 90% interval calibrated on a held-out quarter
             of the training scenarios (`conformal_qhat`);
  escalation the runs the detector is least sure about, argmax p(1-p),
             go back to Tier 1 for b replays (`escalate`).

Every function here is deterministic given its seed arguments; the
cross-validation drivers in analysis/ pass the paper's seeds.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from .metrics import ranks, zlog

HGB = dict(max_iter=400, learning_rate=0.05)
W_REPLAY, W_NOMINAL = 0.02, 0.2
TAU = 0.05                                   # declared tie threshold
TAU_GRID = np.arange(0.05, 0.86, 0.05)       # for the tau-selection ablation
ZEPS = 1e-4


# ------------------------------------------------------------- corpus
def corpus(S, tri, nominal_only=False, run0_only=False,
           drop_crashed_replays=False, reweight=False):
    """Training rows for the scenarios `tri` (row indices into S['y']).

    S is the subject dict of analysis/common.py: Xa2 (nominal rows),
    aug_row (their scenario), Xr2 (replay rows), rep_scn (their scenario),
    nom_run, rep_contact. The default arms are the shipped estimator; the
    keyword arms are the ablations of Sec. 5.2.
    """
    y = S["y"]
    nsel = np.isin(S["aug_row"], tri)
    if run0_only:
        nsel &= S["nom_run"] == 0
    nrows = np.flatnonzero(nsel)
    Xn, yn = S["Xa2"][nrows], y[S["aug_row"][nrows]]
    wn = np.full(len(nrows), W_NOMINAL)
    if nominal_only:
        return Xn, yn, wn
    rmask = np.isin(S["rep_scn"], tri)
    if drop_crashed_replays:
        keep = rmask & ~S["rep_contact"]
        wr = np.full(int(keep.sum()), W_REPLAY)
        if reweight and keep.sum() > 0:
            wr *= rmask.sum() / keep.sum()
        rmask = keep
    else:
        wr = np.full(int(rmask.sum()), W_REPLAY)
    X = np.vstack([Xn, S["Xr2"][rmask]])
    yy = np.concatenate([yn, y[S["rep_scn"][rmask]]])
    w = np.concatenate([wn, wr])
    return X, yy, w


def corpus_ordered(S, tri):
    """The shipped corpus with nominal rows grouped by training scenario in
    `tri` order (the row order the reference runs used; HGB is order
    sensitive only through floating-point summation, but the drivers keep
    it so stored out-of-fold scores reproduce exactly)."""
    y = S["y"]
    nrows = np.concatenate([np.where(S["aug_row"] == i)[0] for i in tri])
    rmask = np.isin(S["rep_scn"], tri)
    Xtr = np.vstack([S["Xa2"][nrows], S["Xr2"][rmask]])
    ytr = np.concatenate([y[S["aug_row"][nrows]], y[S["rep_scn"][rmask]]])
    w = np.concatenate([np.full(len(nrows), W_NOMINAL),
                        np.full(rmask.sum(), W_REPLAY)])
    return Xtr, ytr, w, nrows


# ------------------------------------------------------------ readouts
def fit_readouts(Xtr, ytr, w, seed, n_seeds):
    """(classifier, log regressor, raw regressor) x n_seeds."""
    out = []
    for s in range(n_seeds):
        rs = seed * 10 + s
        c = HistGradientBoostingClassifier(random_state=rs, **HGB)
        c.fit(Xtr, ytr > 0, sample_weight=w)
        rz = HistGradientBoostingRegressor(random_state=rs, **HGB)
        rz.fit(Xtr, zlog(ytr), sample_weight=w)
        rr = HistGradientBoostingRegressor(random_state=rs, **HGB)
        rr.fit(Xtr, ytr, sample_weight=w)
        out.append((c, rz, rr))
    return out


def predict_readouts(parts, Xq):
    p = np.mean([c.predict_proba(Xq)[:, 1] for c, _, _ in parts], axis=0)
    tz = np.mean([r.predict(Xq) for _, r, _ in parts], axis=0)
    tr = np.mean([r.predict(Xq) for _, _, r in parts], axis=0)
    return p, tz, tr


def fit_log_readout(Xtr, ytr, w, seed, n_seeds=3):
    """The magnitude readout alone (log-harm regressor, seed-averaged)."""
    ms = []
    for s in range(n_seeds):
        m = HistGradientBoostingRegressor(random_state=seed * 10 + s, **HGB)
        m.fit(Xtr, zlog(ytr), sample_weight=w)
        ms.append(m)
    return ms


def predict_log(ms, Xq):
    return np.mean([m.predict(Xq) for m in ms], axis=0)


def to_harm(tz):
    """Map the log readout back to the harm scale."""
    return np.clip(10.0 ** tz - ZEPS, 0, 1)


# ------------------------------------------------------------- ordering
def structured_score(p, tz, tr, tau=TAU, p_in_tail=True):
    """The tie-gated rank fusion. `p_in_tail=False` is the TransFuser
    variant, whose survivor band is ordered by the two regressors only."""
    tail = ranks(tz) + ranks(tr)
    if p_in_tail:
        tail = ranks(p) + tail
    return np.where(p < tau, 0.0, 1.0 + tail)


def pick_tau_1se(agg_p, agg_tail, tri, y, rep):
    """Fold-internal tau selection, 1-SE rule over four inner groups
    (the 'fold-internal cross-validated choice' of Sec. 7)."""
    rng = np.random.default_rng(rep * 13 + 3)
    groups = np.array_split(rng.permutation(tri), 4)
    curves = []
    for g in groups:
        m = np.isin(np.arange(len(y)), g) & np.isfinite(agg_p)
        if m.sum() < 12:
            continue
        curves.append([spearmanr(np.where(agg_p[m] < t, 0.0,
                                          1.0 + ranks(agg_p[m]) + agg_tail[m]),
                                 y[m]).statistic for t in TAU_GRID])
    cu = np.nanmean(curves, axis=0)
    se = np.nanstd(curves, axis=0) / np.sqrt(max(len(curves), 1))
    k = int(np.nanargmax(cu))
    ok = np.flatnonzero(cu >= cu[k] - se[k])
    return float(TAU_GRID[int(ok.min())])


def pick_tau_max(agg_p, agg_tail, tri, y):
    """Fold-internal tau selection by maximum inner-OOF rank correlation
    (the TransFuser driver's rule)."""
    m = np.isin(np.arange(len(y)), tri)
    return float(max(TAU_GRID, key=lambda t: spearmanr(
        np.where(agg_p[m] < t, 0.0, 1.0 + agg_tail[m]), y[m]).statistic))


# ---------------------------------------------------------- escalation
def escalate(score, p, inj_by_ref, te, frac=0.10, B=3):
    """Send the `frac` most ambiguous runs of a test fold (argmax p(1-p)) to
    Tier 1 for B replays; a clean escalation scores 0.5 (between the tie
    and the survivors), any harm scores 3 + mean injury (above the field
    band, ordered by measured harm). `inj_by_ref[i]` = that reference's
    per-replay injuries in execution order (the first B are used)."""
    k = max(1, int(round(frac * len(te))))
    amb = np.argsort(-(p * (1 - p)))[:k]
    s = score.copy()
    for j in amb:
        hb = inj_by_ref[te[j]][:B].mean()
        s[j] = 0.5 if hb == 0 else 3.0 + hb
    return s


# ------------------------------------------------------------ conformal
def conformal_qhat(residuals, coverage=0.9):
    """Split-conformal quantile of |prediction - truth| on the calibration
    scenarios, with the finite-sample correction ceil((n+1)q)/n."""
    n = len(residuals)
    return float(np.quantile(np.abs(residuals),
                             min(1.0, np.ceil((n + 1) * coverage) / n),
                             method="higher"))
