"""Telemetry feature basis f(e): deliberately minimal and interpretable.

Three established harm-relevant families (paper Sec. 'Tier 2'):
  proximity : TTC_min, minimum clearance
  energy    : speed at the critical moment, realized Delta-v
  control   : peak jerk, peak lateral rate (yaw-rate slot in real vehicles;
              the pilot ego does not steer, so the lateral rate of the
              conflict partner fills this slot)
"""
from __future__ import annotations

import numpy as np

from .trace import Trace

FEATURE_NAMES = ["ttc_min", "min_clearance", "speed_at_crit",
                 "dv_realized", "peak_jerk", "peak_lat_rate"]

TTC_CAP = 10.0
CLEAR_CAP = 50.0


def extract(tr: Trace) -> np.ndarray:
    valid = tr.active & (tr.closing > 0.3)
    if valid.any():
        ttc = np.where(valid, tr.gap / np.maximum(tr.closing, 0.3), np.inf)
        ttc_min = float(min(np.min(ttc), TTC_CAP))
    else:
        ttc_min = TTC_CAP
    if tr.contact:
        ttc_min = 0.0
        clearance = 0.0
    else:
        pool = tr.gap[tr.active] if tr.active.any() else tr.gap
        clearance = float(np.clip(np.min(pool), 0.0, CLEAR_CAP))
    i_c = tr.crit_index()
    speed_at_crit = float(tr.v_e[i_c])
    jerk = np.abs(np.diff(tr.a_e)) / tr.dt
    peak_jerk = float(np.max(jerk)) if len(jerk) else 0.0
    lat_rate = np.abs(np.diff(tr.y_rel)) / tr.dt
    peak_lat = float(np.max(lat_rate)) if len(lat_rate) else 0.0
    return np.array([ttc_min, clearance, speed_at_crit, float(tr.dv),
                     peak_jerk, peak_lat])


def extract_many(traces) -> np.ndarray:
    return np.stack([extract(t) for t in traces])
