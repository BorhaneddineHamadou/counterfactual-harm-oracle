"""Unit tests for the core. Run: python -m pytest tests -q
(or python tests/test_core.py for a dependency-free run)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adapters.pilot import EgoParams, PilotAdapter, make_suite      # noqa: E402
from proxima.features import extract, extract_many                   # noqa: E402
from proxima.features_expanded import extract_expanded               # noqa: E402
from proxima.features_extra import extract_extra                     # noqa: E402
from proxima.features_traj import extract_traj                       # noqa: E402
from proxima.field_tier import conformal_qhat, structured_score      # noqa: E402
from proxima.injury import occupant, pedestrian, step                # noqa: E402
from proxima.injury_library import library                           # noqa: E402
from proxima.kernel import Kernel                                    # noqa: E402
from proxima.metrics import apfd_h, auc, split_half, ties_10x        # noqa: E402
from proxima.tier1 import label                                      # noqa: E402


def test_kernel_scaling():
    k = Kernel().scaled(2.0)
    assert np.isclose(k.latency_sigma, 0.20)
    assert np.isclose(k.friction_lo, 0.60)
    s = k.sample(1000, np.random.default_rng(0))
    assert np.all(np.abs(s["dlat"]) <= k.latency_bound + 1e-12)
    assert np.all((s["fric"] >= k.friction_lo) & (s["fric"] <= k.friction_hi))
    z = Kernel.zero()
    s0 = z.sample(10, np.random.default_rng(1))
    assert np.all(s0["dlat"] == 0) and np.all(s0["fric"] == 1.0)


def test_injury_curves_monotone():
    dv = np.linspace(0, 40, 100)
    for c in (occupant, pedestrian):
        v = c(dv)
        assert np.all(np.diff(v) > 0) and v[0] < 0.01
    assert step(0.0) == 0.0 and step(5.0) == 1.0
    # shipped coefficients: logit = -6.068 + 0.100 s[km/h]  ->  0.5 at 60.68 km/h
    assert abs(float(occupant(60.68 / 3.6)) - 0.5) < 1e-3
    assert abs(float(occupant(47 / 3.6)) - 0.20) < 0.01
    lib = library()
    assert sum(1 for _, pub, _ in lib.values() if pub) >= 14


def test_determinism_and_crn():
    """Same seeds -> identical traces; two adapters -> same noise (CRN)."""
    scn = make_suite(2, seed=7)
    a1 = PilotAdapter(EgoParams())
    a2 = PilotAdapter(EgoParams())
    t1 = a1.run_suite(scn, 2, 123)
    t2 = a2.run_suite(scn, 2, 123)
    for x, z in zip(t1, t2):
        assert np.allclose(x.v_e, z.v_e) and x.contact == z.contact


def test_binary_special_case():
    """Remark 1: kappa = delta_0 (no perturbation, no continuation left to
    re-randomize) + step iota recovers the collision bit B(e) exactly."""
    scn = make_suite(4, seed=11)
    ad = PilotAdapter(EgoParams())
    traces = ad.run_suite(scn, 2, 42)
    kz = Kernel.zero()
    for tr in traces:
        t_late = (ad.T - 1) * ad.dt
        contact, dv = ad.branch(tr, t_late, kz, M=3)
        H = float(np.mean(step(dv) * contact))
        assert H == float(tr.contact)


def test_branch_prefix_reproduction():
    scn = make_suite(3, seed=3)
    ad = PilotAdapter(EgoParams())
    traces = ad.run_suite(scn, 1, 99)
    kz = Kernel.zero()
    for tr in traces:
        t_late = (ad.T - 1) * ad.dt
        contact, dv = ad.branch(tr, t_late, kz, M=2)
        assert bool(contact[0]) == tr.contact
        if tr.contact:
            assert np.allclose(dv, tr.dv, atol=1e-9)


def test_tier1_label_bounds():
    scn = make_suite(2, seed=5)
    ad = PilotAdapter(EgoParams())
    tr = ad.run_suite(scn, 1, 7)[0]
    L = label(ad, tr, Kernel(), M=30)
    assert 0.0 <= L.H <= 1.0 and L.se >= 0 and 0 <= L.crash_frac <= 1
    lo, hi = L.ci
    assert lo <= L.H <= hi


def test_feature_blocks():
    scn = make_suite(2, seed=5)
    ad = PilotAdapter(EgoParams())
    trs = ad.run_suite(scn, 2, 7)
    F = extract_many(trs)
    assert F.shape == (len(trs), 6) and np.all(np.isfinite(F))
    for tr in trs:
        f41, n41 = extract_expanded(tr)
        f30, n30 = extract_extra(tr)
        f71, n71 = extract_traj(tr)
        assert len(f41) == len(n41) == 41 and np.all(np.isfinite(f41))
        assert len(f30) == len(n30) == 30 and np.all(np.isfinite(f30))
        assert len(f71) == len(n71) == 71 and np.all(np.isfinite(f71))
        assert np.allclose(f41[:6], extract(tr))
        f41s, _ = extract_expanded(tr, phys=False)
        assert f41s[n41.index("phys_H")] == 0.0


def test_field_tier_pieces():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40)
    tz = rng.normal(size=40)
    tr = rng.normal(size=40)
    s = structured_score(p, tz, tr, tau=0.5)
    assert np.all(s[p < 0.5] == 0.0) and np.all(s[p >= 0.5] >= 1.0)
    assert np.all(s[p >= 0.5] <= 4.0)
    s2 = structured_score(p, tz, tr, tau=0.5, p_in_tail=False)
    assert np.all(s2[p >= 0.5] <= 3.0)
    res = rng.normal(size=100)
    q = conformal_qhat(res, 0.9)
    assert np.mean(np.abs(res) <= q) >= 0.9
    assert conformal_qhat(np.array([0.1, -0.2, 0.3]), 0.9) == 0.3


def test_metrics():
    y = np.array([0.0, 0.0, 0.01, 0.1, 0.3])
    yy = np.concatenate([np.zeros(20), y])          # 25 runs, harm in the last 5
    assert apfd_h(np.arange(25, dtype=float), yy) > 0.9        # harm scored first
    assert apfd_h(-np.arange(25, dtype=float), yy) < 0.1       # harm scored last
    assert abs(apfd_h(np.zeros(25), yy) - 0.5) < 1e-9         # all tied = random
    assert abs(auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) - 0.75) < 1e-9
    assert ties_10x(np.zeros(5), y) == 1.0
    H = np.tile(y[:, None], (1, 100))
    assert split_half(H, n_draws=5) > 0.99


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as e:      # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {e!r}")
    sys.exit(1 if fails else 0)
