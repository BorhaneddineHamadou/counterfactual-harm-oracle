"""The 41-feature single-trace descriptor block (families A, B, C).

This is the first block of the field tier's 71 per-trace features
(paper Sec. 3.4, "Field tier"). It extends the six interpretable base
features of `features.py` with

  B. criticality-profile statistics: time-to-collision minima and exposure
     times (TET/TIT), required deceleration (DRAC), the encounter state at
     the criticality peak and at the branch point t* = t_crit - 4 s;
  C. open-loop physics counterfactuals: the recorded kinematics replayed
     with delayed / weakened ego braking (the kernel's two ego axes),
     read off as predicted minimum gap and impact speed, plus `phys_H`,
     a kernel-averaged analytic harm estimate;
  D. one-hot template indicators (5).

Layout of the returned vector (41): 6 base | 30 expanded | 5 template.

`phys_H` is the only expensive feature (64 kinematic replays per trace).
The training corpus computes it for nominal rows and sets it to 0 on the
15,000 replay rows (`phys=False`), exactly as in the reference campaign.
"""
from __future__ import annotations

import numpy as np

from .features import extract as extract6
from .injury import occupant

DT = 0.05                 # trace cadence, 20 Hz (both adapters)
HORIZON = 4.0             # branch horizon T_h (s)
TEMPLATES = ["lead_decel", "lead_stopped", "cut_in",
             "oncoming_drift", "crossing_traffic"]
BASE_NAMES = ["ttc_min", "min_clearance", "speed_at_crit", "dv_realized",
              "peak_jerk", "peak_lat_rate"]


# --------------------------------------------------- physics counterfactual
def phys_counterfactual(tr, i0, d_lat, gain):
    """Open-loop kinematic replay from index i0 with ego braking delayed by
    d_lat seconds and scaled by `gain`. Threat motion and ego non-braking
    accel are taken from the recording. Returns (min_gap, dv_at_contact)."""
    n = tr.T
    if i0 >= n - 2:
        return float(tr.gap[-1]), 0.0
    shift = int(round(d_lat / DT))
    a = tr.a_e.copy()
    brake = np.minimum(a, 0.0)
    other = np.maximum(a, 0.0)
    if shift > 0:
        brake = np.concatenate([np.zeros(shift), brake])[:n]
    elif shift < 0:
        brake = np.concatenate([brake[-shift:], np.zeros(-shift)])
    a_cf = other + gain * brake
    v_t = tr.v_e - tr.closing            # threat speed along ego heading
    v_cf = np.empty(n)
    v_cf[:i0 + 1] = tr.v_e[:i0 + 1]
    for i in range(i0, n - 1):
        v_cf[i + 1] = max(v_cf[i] + a_cf[i] * DT, 0.0)
    closing_cf = v_cf - v_t
    gap_cf = np.empty(n)
    gap_cf[:i0 + 1] = tr.gap[:i0 + 1]
    gap_cf[i0 + 1:] = tr.gap[i0] - np.cumsum(closing_cf[i0:n - 1] * DT)
    ov = np.abs(tr.y_rel) < 2.6
    hit = np.flatnonzero((gap_cf <= 0.0) & ov & (closing_cf > 0.0))
    hit = hit[hit >= i0]
    if len(hit):
        i = int(hit[0])
        return 0.0, float(max(closing_cf[i], 0.0))
    pool = gap_cf[tr.active] if tr.active.any() else gap_cf
    return float(np.clip(np.min(pool), 0.0, 50.0)), 0.0


def phys_H(tr, i0, n_draws=64, seed=0):
    """Kernel-averaged analytic harm: E_eps[iota(dv_cf)] over 64 draws of the
    two ego kernel axes (latency, braking scale)."""
    rng = np.random.default_rng(seed)
    dlat = np.clip(rng.normal(0.0, 0.10, n_draws), -0.25, 0.25)
    fric = rng.uniform(0.80, 1.05, n_draws)
    h = 0.0
    for d, g in zip(dlat, fric):
        _, dv = phys_counterfactual(tr, i0, d, g)
        h += float(occupant(dv)) if dv > 0 else 0.0
    return h / n_draws


# ------------------------------------------------------- expanded features
def extract_expanded(tr, phys=True):
    """41 features for one Trace. `phys=False` stubs phys_H to 0 (replay
    rows of the training corpus)."""
    base = extract6(tr)
    valid = tr.active & (tr.closing > 0.3)
    ttc = np.where(valid, tr.gap / np.maximum(tr.closing, 0.3), np.inf)
    i_c = tr.crit_index()
    i0 = max(0, i_c - int(round(HORIZON / DT)))      # branch point t*

    ext, enames = [], []

    def add(nm, v):
        enames.append(nm)
        ext.append(float(v))

    add("log_ttc_min", np.log10(min(np.min(ttc), 30.0) + 0.1))
    for thr in (1.0, 2.0, 3.0):
        add(f"tet{int(thr)}", np.sum(ttc < thr) * DT)
    add("tit3", np.sum(np.clip(3.0 - ttc[np.isfinite(ttc)], 0, None)) * DT)
    drac = np.where(valid, tr.closing ** 2 /
                    (2.0 * np.maximum(tr.gap, 0.3)), 0.0)
    add("drac_max", np.clip(np.max(drac), 0, 30))
    add("closing_at_crit", tr.closing[i_c])
    add("gap_at_crit", np.clip(tr.gap[i_c], 0, 50))
    add("v_threat_at_crit", tr.v_e[i_c] - tr.closing[i_c])
    pool = tr.gap[tr.active] if tr.active.any() else tr.gap
    add("log_min_gap", np.log10(np.clip(np.min(pool), 0.0, 100.0) + 0.1))
    add("max_closing", np.max(tr.closing[tr.active])
        if tr.active.any() else 0.0)
    add("decel_peak", max(-np.min(tr.a_e), 0.0))
    j0 = max(0, i_c - int(round(2.0 / DT)))
    add("mean_decel_2s", max(-np.mean(tr.a_e[j0:i_c + 1]), 0.0))
    add("t_active", np.sum(tr.active) * DT)
    add("t_crit", tr.t_crit)
    add("gap_at_tstar", np.clip(tr.gap[i0], 0, 60))
    add("closing_at_tstar", tr.closing[i0])
    add("v_e_at_tstar", tr.v_e[i0])
    ttc_t0 = (tr.gap[i0] / max(tr.closing[i0], 0.3)
              if tr.closing[i0] > 0.3 else 30.0)
    add("ttc_at_tstar", min(ttc_t0, 30.0))
    add("ke_at_crit", tr.v_e[i_c] ** 2 / 100.0)
    add("jerk_rms", np.sqrt(np.mean((np.diff(tr.a_e) / DT) ** 2))
        if tr.T > 1 else 0.0)

    for (d, g) in ((0.25, 1.0), (0.0, 0.80), (0.25, 0.80), (0.10, 0.90)):
        mg, dv = phys_counterfactual(tr, i0, d, g)
        add(f"cf_mingap_d{int(d*100)}_g{int(g*100)}", mg)
        add(f"cf_dv_d{int(d*100)}_g{int(g*100)}", dv)
    add("phys_H", phys_H(tr, i0) if phys else 0.0)

    tpl = [1.0 if tr.template == t else 0.0 for t in TEMPLATES]
    names_all = BASE_NAMES + enames + [f"tpl_{t}" for t in TEMPLATES]
    return np.concatenate([base, np.array(ext), np.array(tpl)]), names_all
