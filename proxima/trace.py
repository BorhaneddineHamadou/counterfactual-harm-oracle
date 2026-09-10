"""Canonical Trace: one recorded execution of a concrete scenario.

One schema for every simulator adapter. A trace carries the time series
needed for (a) telemetry feature extraction and (b) locating the criticality
peak t_crit that anchors the counterfactual branch point t* = t_crit - T_h.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    template: str            # scenario template name
    sid: int                 # scenario id within the suite
    run_idx: int             # run index (seed) for this scenario
    global_seed: int         # campaign seed; (global_seed, sid, run_idx) -> noise
    params: dict             # concrete scenario parameters (scalars)
    actor_type: str          # 'occupant' | 'pedestrian'  (selects iota)
    dt: float
    # time series (T,)
    v_e: np.ndarray          # ego speed
    a_e: np.ndarray          # ego realized acceleration
    gap: np.ndarray          # bumper-to-bumper gap to threat
    closing: np.ndarray      # closing speed to threat
    y_rel: np.ndarray        # threat lateral offset from ego lane center
    overlap: np.ndarray      # bool: lateral overlap with ego corridor
    active: np.ndarray       # bool: threat considered active by geometry
    # outcome
    contact: bool
    t_contact: float         # np.inf if none
    dv: float                # relative impact speed (0 if no contact)
    _i_crit: int = field(default=-1, repr=False)

    @property
    def T(self) -> int:
        return len(self.v_e)

    def crit_index(self) -> int:
        """Index of the criticality peak: contact step if any, else min-TTC
        among active closing steps, else min gap."""
        if self._i_crit >= 0:
            return self._i_crit
        if self.contact and np.isfinite(self.t_contact):
            i = min(int(round(self.t_contact / self.dt)), self.T - 1)
        else:
            valid = self.active & (self.closing > 0.3)
            if valid.any():
                ttc = np.where(valid, self.gap / np.maximum(self.closing, 0.3),
                               np.inf)
                i = int(np.argmin(ttc))
            else:
                i = int(np.argmin(self.gap))
        object.__setattr__(self, "_i_crit", i)
        return i

    @property
    def t_crit(self) -> float:
        return self.crit_index() * self.dt

    def noise_key(self) -> list:
        return [self.global_seed, self.sid, self.run_idx]
