"""Evaluation metrics used throughout the paper (Sec. 4.1).

  apfd_h      harm-weighted APFD, Eq. (3): 0.5 random, 1 harm-first
  auc         area under the ROC curve, mid-rank Mann-Whitney form
  surfaced    share of total harm in the top-k of an ordering
  ties_10x    fraction of reference pairs whose true harms differ by more
              than 10x that an oracle scores identically ("tie-blindness")
  split_half  split-half reliability of Monte-Carlo labels (Spearman-Brown)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

ZEPS = 1e-4


def zlog(v):
    """log10(H + 1e-4): the log-scale target of the magnitude readout."""
    return np.log10(np.clip(v, 0, 1) + ZEPS)


def ranks(v):
    """Ranks scaled to [0, 1] (argsort-of-argsort; no mid-ranks)."""
    return np.argsort(np.argsort(v)).astype(float) / max(len(v) - 1, 1)


def spearman(a, b):
    return float(spearmanr(a, b).statistic)


def apfd_h(score, harm):
    """Harm-weighted APFD: classical APFD with each run weighted by its
    Tier-1 harm; ties take mid-ranks (expectation over random tie-breaks).

        APFD_H = 1 - sum_i h_i r_i / (n sum_i h_i) + 1/(2n)
    """
    n = len(score)
    W = float(np.sum(harm))
    if W <= 0 or n == 0:
        return np.nan
    rank = rankdata(-np.asarray(score, float), method="average")
    return 1.0 - float(np.sum(harm * rank)) / (n * W) + 1.0 / (2 * n)


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any():
        return np.nan
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def surfaced(score, harm, budget):
    """Fraction of total harm found in the top `budget` share of the ordering."""
    k = max(1, int(round(budget * len(score))))
    tot = float(np.sum(harm))
    return np.nan if tot <= 0 else float(harm[np.argsort(-score)[:k]].sum() / tot)


def pairs_10x(y, ratio=10.0):
    """Index pairs (i, j) whose true harms differ by more than `ratio`x."""
    i, j = np.triu_indices(len(y), 1)
    hi = np.maximum(y[i], y[j])
    lo = np.maximum(np.minimum(y[i], y[j]), 1e-9)
    big = (hi / lo > ratio) & (hi > 0)
    return i[big], j[big]


def ties_10x(score, y, ratio=10.0):
    """Share of >10x-differing pairs the score cannot separate."""
    i, j = pairs_10x(y, ratio)
    s = np.asarray(score, float)
    return float(np.mean(s[i] == s[j]))


def split_half(H_rep, n_draws=200, seed=0):
    """Split-half reliability of the M-replay labels: Spearman correlation
    of labels recomputed from two disjoint halves of each reference's
    replays, Spearman-Brown corrected, averaged over random halvings."""
    rng = np.random.default_rng(seed)
    M = H_rep.shape[1]
    out = []
    for _ in range(n_draws):
        p = rng.permutation(M)
        a = np.nanmean(H_rep[:, p[:M // 2]], axis=1)
        b = np.nanmean(H_rep[:, p[M // 2:]], axis=1)
        r = spearman(a, b)
        out.append(2 * r / (1 + r))
    return float(np.mean(out))


def boot_ci(vals, lo=2.5, hi=97.5):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        return (np.nan, np.nan)
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))
