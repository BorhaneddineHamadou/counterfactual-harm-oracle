"""Tier 1: the reference oracle -- Monte-Carlo counterfactual harm.

H_kappa(e) = E_{eps~kappa}[ iota(dv(e (+) eps)) ], estimated by M perturbed
replays branched at t* = max(0, t_crit - T_h). This is the definition of
ground truth in every experiment; its MC standard error is reported and
propagated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .injury import get_curve
from .kernel import Kernel
from .trace import Trace

BRANCH_HORIZON_S = 4.0   # T_h: branch this many seconds before t_crit


@dataclass
class Tier1Label:
    H: float           # MC estimate of counterfactual harm
    se: float          # MC standard error
    crash_frac: float  # fraction of the neighborhood ending in contact
    M: int

    @property
    def ci(self):
        return (max(0.0, self.H - 1.96 * self.se),
                min(1.0, self.H + 1.96 * self.se))


def label(adapter, tr: Trace, kernel: Kernel, M: int,
          horizon_s: float = BRANCH_HORIZON_S) -> Tier1Label:
    t_star = max(0.0, tr.t_crit - horizon_s)
    contact, dv = adapter.branch(tr, t_star, kernel, M)
    iota = get_curve(tr.actor_type)
    h = np.where(contact, iota(dv), 0.0)
    return Tier1Label(H=float(np.mean(h)),
                      se=float(np.std(h, ddof=1) / np.sqrt(M)) if M > 1 else 0.0,
                      crash_frac=float(np.mean(contact)), M=M)


def label_all(adapter, traces: List[Trace], kernel: Kernel, M: int,
              horizon_s: float = BRANCH_HORIZON_S,
              progress_every: int = 0) -> List[Tier1Label]:
    out = []
    for i, tr in enumerate(traces):
        out.append(label(adapter, tr, kernel, M, horizon_s))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  tier1: {i+1}/{len(traces)} labeled", flush=True)
    return out
