"""The 30 extra single-trace features (second block of the 71).

Margin-trajectory shape, braking-response timing, stopping-energy margins,
lateral geometry and a richer open-loop counterfactual grid. Together with
`features_expanded.extract_expanded` (41) this gives the 71 per-trace
descriptors of the field tier (paper Sec. 3.4).

extract_extra(tr) -> (vec[30], names)
"""
from __future__ import annotations

import numpy as np

from .features_expanded import DT, phys_counterfactual

A_MAX = 7.0            # assumed available decel


def extract_extra(tr):
    n = tr.T
    i_c = tr.crit_index()
    i0 = max(0, i_c - int(round(4.0 / DT)))
    act = tr.active
    gap = np.clip(tr.gap, 0.0, 100.0)
    ext, names = [], []

    def add(nm, v):
        names.append(nm)
        ext.append(float(v) if np.isfinite(v) else 0.0)

    # margin-trajectory shape
    pool = gap[act] if act.any() else gap
    for q in (10, 25, 50):
        add(f"gap_q{q}", np.percentile(pool, q))
    for thr in (2.0, 5.0, 10.0):
        add(f"t_gap_lt{int(thr)}", np.sum((gap < thr) & act) * DT)
    add("gap_slope_2s", (gap[i_c] - gap[max(0, i_c - 40)]) / 2.0)
    # speed profile
    add("v_min_act", tr.v_e[act].min() if act.any() else tr.v_e.min())
    for off in (1.0, 2.0, 3.0):
        j = max(0, i_c - int(round(off / DT)))
        add(f"v_e_m{int(off)}s", tr.v_e[j])
    add("v_drop_frac", 1.0 - (tr.v_e[i_c] / max(tr.v_e[i0], 0.5)))
    # braking response timing
    first_act = int(np.argmax(act)) if act.any() else n - 1
    brk = np.flatnonzero(tr.a_e[first_act:] < -1.0)
    add("react_lag", (brk[0] * DT) if len(brk) else 10.0)
    a_sm = np.convolve(tr.a_e, np.ones(20) / 20, mode="same")
    add("max_sust_decel", max(-a_sm.min(), 0.0))
    taps = np.sum(np.diff((tr.a_e < -1.0).astype(int)) == 1)
    add("brake_taps", taps)
    # energy / stopping margin
    with np.errstate(divide="ignore", invalid="ignore"):
        a_req = np.where(act & (tr.closing > 0.3),
                         tr.closing ** 2 / (2 * np.maximum(gap, 0.3)), 0.0)
    add("t_areq_gt_amax", np.sum(a_req > A_MAX) * DT)
    add("areq_at_tstar", min(a_req[i0], 30.0))
    add("stop_margin_crit",
        gap[i_c] - tr.v_e[i_c] ** 2 / (2 * A_MAX))
    add("stop_margin_tstar",
        gap[i0] - tr.v_e[i0] ** 2 / (2 * A_MAX))
    # lateral geometry
    yr = np.abs(tr.y_rel)
    add("ymin_act", yr[act].min() if act.any() else yr.min())
    add("t_overlap", np.sum(tr.overlap & act) * DT)
    ov = tr.overlap & act
    add("closing_at_ov", tr.closing[ov].max() if ov.any() else 0.0)
    # richer counterfactual grid
    for (d, g) in ((0.10, 1.0), (0.40, 1.0), (0.25, 0.90), (0.40, 0.80)):
        mg, dv = phys_counterfactual(tr, i0, d, g)
        add(f"cf2_mingap_d{int(d*100)}_g{int(g*100)}", mg)
        add(f"cf2_dv_d{int(d*100)}_g{int(g*100)}", dv)
    return np.array(ext), names
