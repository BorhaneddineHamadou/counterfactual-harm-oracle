"""Synthetic pilot world: analytic closed-loop dynamics with exact Tier-1
ground truth. Fixes the protocol and statistics before any simulator campaign
(paper Sec. 'Study Design'); ships in the replication package.

Five scenario templates (NHTSA pre-crash typology):
  lead_decel      -- lead vehicle decelerates ahead of ego
  lead_stopped    -- stopped vehicle in lane (highest-frequency rear-end type)
  cut_in          -- adjacent vehicle cuts into ego lane, optionally braking
  oncoming_drift  -- oncoming vehicle drifts into ego lane, may return
  ped_crossing    -- pedestrian crosses; exercises the VRU injury curve

The ego is a reactive ACC/AEB-style controller with perception noise and
reaction latency; stochasticity per run comes from perception noise, a
per-run latency draw, and small actor jitter. The perturbation kernel enters
ONLY through the branch operation: extra latency, friction scaling of
achievable deceleration, and OU actor noise, all switched on from the branch
step i* onward, with the nominal noise stream reproduced exactly before it.

Everything is vectorized over a batch dimension B (scenarios x runs, or
replays), so Tier-1 labeling and power studies run in minutes on CPU.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from proxima.kernel import Kernel
from proxima.sim import SimulatorAdapter
from proxima.trace import Trace

DT = 0.05
T_MAX_S = 22.0
CAR_LEN = 4.5
LANE_W = 3.5
NOISE_CH = 5   # 0: gap percep, 1: closing percep, 2: actor jitter, 3: OU long, 4: OU lat

TEMPLATES = ["lead_decel", "lead_stopped", "cut_in", "oncoming_drift",
             "ped_crossing"]

PARAM_RANGES = {
    "lead_decel": {"v0": (18, 35), "headway": (0.8, 2.2),
                   "a_lead": (3.0, 8.0), "t_b": (1.0, 3.0)},
    "lead_stopped": {"v0": (15, 35), "d0": (50, 130)},
    "cut_in": {"v0": (18, 35), "gap0": (8, 28), "dv_rel": (-9, 2),
               "t_c": (1.0, 3.0), "t_lat": (1.0, 3.0), "a_cut": (0.0, 4.5)},
    "oncoming_drift": {"v0": (15, 30), "v_onc": (10, 25), "r0": (90, 200),
                       "vy_drift": (0.25, 1.2), "y_min": (0.0, 2.4),
                       "t_stay": (0.5, 2.5)},
    "ped_crossing": {"v0": (10, 22), "t_gap": (1.2, 3.5), "vp": (0.8, 2.5)},
}

ACTOR_TYPE = {t: ("pedestrian" if t == "ped_crossing" else "occupant")
              for t in TEMPLATES}


@dataclass(frozen=True)
class EgoParams:
    """The system under test. Seeded regressions are edits of these fields."""
    a_max: float = 6.5          # max commanded deceleration (m/s^2)
    tau0: float = 0.35          # nominal reaction latency (s)
    j_ramp: float = 12.0        # jerk limit toward brake target (m/s^3)
    a_trig: float = 3.2         # required-decel hazard trigger (m/s^2)
    ttc_trig: float = 2.2       # TTC hazard trigger (s)
    kp: float = 0.4             # cruise speed gain
    a_acc: float = 1.5          # max cruise acceleration
    eta0: float = 1.25          # nominal braking margin on required decel
    fade: float = 0.0           # brake fade under sustained peak demand
    sensor_range: float = 120.0
    tau_run_sigma: float = 0.04  # per-run latency variability (s)
    label: str = "V0"


@dataclass(frozen=True)
class Scenario:
    template: str
    sid: int
    params: dict


def make_suite(n_per_template: int, seed: int, templates=None):
    """Latin-hypercube instantiation of concrete scenarios per template."""
    from scipy.stats import qmc
    templates = templates or TEMPLATES
    out = []
    sid = 0
    for ti, tmpl in enumerate(templates):
        ranges = PARAM_RANGES[tmpl]
        names = list(ranges)
        sampler = qmc.LatinHypercube(d=len(names), seed=seed + 7919 * ti)
        u = sampler.random(n_per_template)
        lo = np.array([ranges[nm][0] for nm in names])
        hi = np.array([ranges[nm][1] for nm in names])
        vals = lo + u * (hi - lo)
        for row in vals:
            out.append(Scenario(tmpl, sid, dict(zip(names, map(float, row)))))
            sid += 1
    return out


def _noise(key, T):
    """Deterministic noise streams for one (scenario, run) or replay suffix."""
    rng = np.random.default_rng(np.random.SeedSequence(list(key)))
    per_run = rng.standard_normal(4)
    per_step = rng.standard_normal((T, NOISE_CH))
    return per_run, per_step


# ======================================================================
# core batched dynamics
# ======================================================================
def simulate(template, P, ego: EgoParams, T_steps, dt, noise, per_run,
             eps=None, i_star=0, record=True):
    """Simulate B rollouts of one template.

    P: dict of parameter arrays (B,). noise: (B, T, 5). per_run: (B, 4).
    eps: kernel disturbances {dlat (B,), fric (B,), ou_*} active from i_star.
    """
    B = len(next(iter(P.values())))
    v0 = P["v0"]
    x_e = np.zeros(B)
    v_e = v0.copy()
    a_e = np.zeros(B)
    v_des = v0.copy()
    tau_eff = np.clip(ego.tau0 + ego.tau_run_sigma * per_run[:, 0], 0.05, None)
    # per-run braking aggressiveness: comfort-optimized planners brake 'just
    # enough' (margin on required decel) -- this is what creates near-misses
    eta = np.clip(ego.eta0 + 0.1 * per_run[:, 1],
                  ego.eta0 - 0.2, ego.eta0 + 0.3)
    # stationary-target recognition range varies run to run (the classic AEB
    # 'stopped car problem'): late recognitions produce saturated braking
    d_detect = (np.clip(95.0 + 25.0 * per_run[:, 2], 45.0, 140.0)
                if template == "lead_stopped"
                else np.full(B, ego.sensor_range))
    fric = np.ones(B)
    trig = np.zeros(B, bool)
    t_trig = np.full(B, np.inf)
    brake_started = np.zeros(B, bool)
    hold = np.zeros(B, bool)
    done = np.zeros(B, bool)
    passed = np.zeros(B, bool)
    contact = np.zeros(B, bool)
    dv = np.zeros(B)
    t_contact = np.full(B, np.inf)
    i_hit = np.full(B, T_steps - 1, int)
    ou_l = np.zeros(B)
    ou_lat = np.zeros(B)
    y_off = np.zeros(B)
    eps_on = False
    vL_prev = None          # filtered lead-state estimator (lead-like only)
    aL_est = np.zeros(B)
    hd_time = np.zeros(B)   # cumulative sustained-peak-demand time (fade)

    # template-specific initial state
    if template == "lead_decel":
        x_t = P["headway"] * v0 + CAR_LEN
        v_t = v0.copy()
        jit = 0.3
    elif template == "lead_stopped":
        x_t = P["d0"] + CAR_LEN
        v_t = np.zeros(B)
        jit = 0.0
    elif template == "cut_in":
        x_t = P["gap0"] + CAR_LEN
        v_t = v0 + P["dv_rel"]
        jit = 0.3
    elif template == "oncoming_drift":
        x_t = P["r0"].copy()
        v_t = P["v_onc"].copy()
        jit = 0.2
    elif template == "ped_crossing":
        x_p = P["t_gap"] * v0 + CAR_LEN / 2
        y_p = np.full(B, 4.0)
        v_p = P["vp"].copy()
        jit = 0.15
    else:
        raise ValueError(template)

    lat_kernel = template in ("cut_in", "oncoming_drift")
    lead_like = template in ("lead_decel", "lead_stopped", "cut_in")

    if record:
        rec = {nm: np.zeros((T_steps, B)) for nm in
               ("v_e", "a_e", "gap", "closing", "y_rel")}
        rec["overlap"] = np.zeros((T_steps, B), bool)
        rec["active"] = np.zeros((T_steps, B), bool)

    sqdt = np.sqrt(dt)
    for i in range(T_steps):
        t = i * dt
        if eps is not None and i == i_star:
            eps_on = True
            fric = np.asarray(eps["fric"], float).copy()
            tau_eff = np.where(brake_started, tau_eff,
                               tau_eff + np.asarray(eps["dlat"], float))
        ns = noise[:, i, :]

        # ---- kernel OU noise (branch only) -------------------------------
        if eps_on:
            th = eps["ou_theta"]
            ou_l += -th * ou_l * dt + eps["ou_sigma_long"] * sqdt * ns[:, 3]
            if lat_kernel:
                ou_lat += (-th * ou_lat * dt
                           + eps["ou_sigma_lat"] * sqdt * ns[:, 4])
                y_off += ou_lat * dt

        # ---- threat dynamics --------------------------------------------
        if template == "lead_decel":
            a_t = np.where(t >= P["t_b"], -P["a_lead"], 0.0)
            v_t = np.maximum(v_t + (a_t + jit * ns[:, 2] + ou_l) * dt, 0.0)
            x_t = x_t + v_t * dt
            y_eff = np.zeros(B)
        elif template == "lead_stopped":
            y_eff = np.zeros(B)
        elif template == "cut_in":
            a_t = np.where((t >= P["t_c"]) & (t < P["t_c"] + 1.5),
                           -P["a_cut"], 0.0)
            v_t = np.maximum(v_t + (a_t + jit * ns[:, 2] + ou_l) * dt, 0.0)
            x_t = x_t + v_t * dt
            prog = np.clip((t - P["t_c"]) / P["t_lat"], 0.0, 1.0)
            y_eff = np.clip(LANE_W * (1.0 - prog) + y_off, -0.5, LANE_W + 1.0)
        elif template == "oncoming_drift":
            v_t = np.clip(v_t + (jit * ns[:, 2] + ou_l) * dt, 3.0, 40.0)
            x_t = x_t - v_t * dt
            t1 = (LANE_W - P["y_min"]) / P["vy_drift"]
            y_base = np.where(
                t < t1, LANE_W - P["vy_drift"] * t,
                np.where(t < t1 + P["t_stay"], P["y_min"],
                         np.minimum(LANE_W, P["y_min"]
                                    + P["vy_drift"] * (t - t1 - P["t_stay"]))))
            y_eff = np.clip(y_base + y_off, -0.5, LANE_W + 1.0)
        else:  # ped_crossing
            v_p = np.clip(v_p + (jit * ns[:, 2] + ou_l) * dt, 0.2, 4.0)
            y_p = y_p - v_p * dt

        # ---- geometry ----------------------------------------------------
        if template == "ped_crossing":
            gap = x_p - x_e - CAR_LEN / 2
            closing = v_e.copy()
            overlap = np.abs(y_p) < 1.0
            active = (y_p < 3.0) & (y_p > -1.5) & (gap > 0) & (gap < 60.0)
            y_rel = y_p
        else:
            gap = x_t - x_e - CAR_LEN
            if template == "oncoming_drift":
                closing = v_e + v_t
            else:
                closing = v_e - v_t
            overlap = (np.abs(y_eff) < 1.9) if template in (
                "cut_in", "oncoming_drift") else np.ones(B, bool)
            if template == "cut_in":
                # anticipate a cut in progress from the lateral motion
                cutting = (t >= P["t_c"]) & (t < P["t_c"] + P["t_lat"])
                active = (gap > 0) & ((np.abs(y_eff) < 2.6)
                                      | (cutting & (np.abs(y_eff) < 3.2)))
            else:
                active = (gap > 0) & (np.abs(y_eff) < 2.6)
            y_rel = y_eff
        active = active & ~passed

        # ---- ego perception + control -----------------------------------
        g_m = np.maximum(gap + 0.5 * ns[:, 0], 0.1)
        c_m = closing + 0.3 * ns[:, 1]
        ttc_m = g_m / np.maximum(c_m, 0.3)
        a_req_inst = np.where(c_m > 0,
                              c_m ** 2 / (2 * np.maximum(g_m, 0.5)), 0.0)
        if lead_like:
            # AEB-style stopping-distance criterion: estimate the lead's
            # deceleration (filtered differentiation of perceived lead
            # speed) and require ego to stop within gap + lead stop distance
            vL_m = v_e - c_m
            if vL_prev is None:
                vL_prev = vL_m.copy()
            aL_meas = (vL_prev - vL_m) / dt
            aL_est = aL_est + 0.1 * (aL_meas - aL_est)
            vL_prev = vL_m
            d_stop = np.where(
                aL_est > 0.8,
                np.maximum(vL_m, 0.0) ** 2 / (2 * np.clip(aL_est, 0.8, None)),
                1e9)
            a_req = np.where(
                d_stop < 1e8,
                v_e ** 2 / (2 * np.maximum(g_m + d_stop, 0.5)),
                a_req_inst)
        else:
            a_req = a_req_inst
        hazard = (active & (gap < d_detect) & (c_m > 0.3)
                  & ((ttc_m < ego.ttc_trig) | (a_req > ego.a_trig)))
        newly = hazard & ~trig
        t_trig = np.where(newly, t, t_trig)
        trig = (trig | newly) & active & (closing > 0.15)
        t_trig = np.where(trig, t_trig, np.inf)
        braking = trig & (t >= t_trig + tau_eff)
        brake_started = brake_started | braking
        if lead_like:
            # once braked to a near-stop close behind the threat, stay stopped
            hold = hold | (braking & (v_e < 0.5) & (gap < 12.0))
        v_des_eff = np.where(hold, 0.0, v_des)
        if lead_like:
            # ACC car-following: within following headway, adopt the lead's
            # speed as the cruise target instead of re-closing on it
            follow = active & (gap < 1.2 * v_e + 4.0)
            v_des_eff = np.where(follow,
                                 np.minimum(v_des_eff, np.clip(vL_m, 0, None)),
                                 v_des_eff)
        a_cruise = np.clip(ego.kp * (v_des_eff - v_e), -1.5, ego.a_acc)
        # proportional braking: just-enough + margin, capped by the
        # friction-limited maximum (friction only binds in hard braking)
        a_cmd = np.clip(eta * a_req, 2.0, ego.a_max)
        cap = ego.a_max * fric
        if ego.fade > 0.0:
            # actuator defect: after sustained peak demand, achievable
            # deceleration fades -- binds almost only in deep emergencies,
            # so it shifts conditional-on-crash severity, little else
            high = braking & (eta * a_req >= 0.95 * ego.a_max)
            hd_time = np.where(high, hd_time + dt, hd_time)
            cap = cap * np.where(hd_time > 0.5, 1.0 - ego.fade, 1.0)
        a_tgt = np.where(braking, -np.minimum(a_cmd, cap), a_cruise)
        a_e = a_e + np.clip(a_tgt - a_e, -ego.j_ramp * dt, ego.j_ramp * dt)
        v_new = np.maximum(v_e + a_e * dt, 0.0)
        a_real = (v_new - v_e) / dt
        x_e = x_e + v_new * dt
        v_e = v_new

        # ---- contact / pass detection -----------------------------------
        if template == "ped_crossing":
            gap_new = x_p - x_e - CAR_LEN / 2
            dv_now = v_e
        else:
            gap_new = x_t - x_e - CAR_LEN
            if template == "oncoming_drift":
                dv_now = v_e + v_t
            else:
                dv_now = np.maximum(v_e - v_t, 0.0)
        crossed = (gap_new <= 0) & ~done & ~passed
        hit = crossed & overlap
        miss = crossed & ~overlap
        contact = contact | hit
        dv = np.where(hit, dv_now, dv)
        t_contact = np.where(hit, t, t_contact)
        i_hit = np.where(hit, i, i_hit)
        done = done | hit
        passed = passed | miss

        if record:
            rec["v_e"][i] = v_e
            rec["a_e"][i] = a_real
            rec["gap"][i] = gap_new
            rec["closing"][i] = closing
            rec["y_rel"][i] = y_rel
            rec["overlap"][i] = overlap
            rec["active"][i] = active & ~done

    out = {"contact": contact, "dv": dv, "t_contact": t_contact,
           "i_hit": i_hit}
    if record:
        out["rec"] = rec
    return out


# ======================================================================
# adapter
# ======================================================================
class PilotAdapter(SimulatorAdapter):

    def __init__(self, ego: EgoParams = EgoParams(), dt: float = DT,
                 t_max_s: float = T_MAX_S):
        self.ego = ego
        self.dt = dt
        self.T = int(round(t_max_s / dt))

    def run_suite(self, scenarios, k, global_seed):
        traces = [None] * (len(scenarios) * k)
        groups = defaultdict(list)
        for pos, sc in enumerate(scenarios):
            groups[sc.template].append(pos)
        for tmpl, poss in groups.items():
            pnames = list(PARAM_RANGES[tmpl])
            pairs = [(pos, r) for pos in poss for r in range(k)]
            P = {nm: np.array([scenarios[pos].params[nm] for pos, _ in pairs])
                 for nm in pnames}
            prs, nss = [], []
            for pos, r in pairs:
                pr, nstep = _noise((global_seed, scenarios[pos].sid, r), self.T)
                prs.append(pr)
                nss.append(nstep)
            res = simulate(tmpl, P, self.ego, self.T, self.dt,
                           np.stack(nss), np.stack(prs))
            for b, (pos, r) in enumerate(pairs):
                traces[pos * k + r] = self._build_trace(
                    scenarios[pos], r, global_seed, res, b)
        return traces

    def _build_trace(self, sc: Scenario, run_idx, global_seed, res, b):
        end = res["i_hit"][b] + 1 if res["contact"][b] else self.T
        r = res["rec"]
        return Trace(
            template=sc.template, sid=sc.sid, run_idx=run_idx,
            global_seed=global_seed, params=sc.params,
            actor_type=ACTOR_TYPE[sc.template], dt=self.dt,
            v_e=r["v_e"][:end, b].copy(), a_e=r["a_e"][:end, b].copy(),
            gap=r["gap"][:end, b].copy(), closing=r["closing"][:end, b].copy(),
            y_rel=r["y_rel"][:end, b].copy(),
            overlap=r["overlap"][:end, b].copy(),
            active=r["active"][:end, b].copy(),
            contact=bool(res["contact"][b]),
            t_contact=float(res["t_contact"][b]), dv=float(res["dv"][b]))

    def branch(self, tr: Trace, t_star, kernel: Kernel, M):
        i_star = min(int(round(t_star / self.dt)), self.T - 1)
        pnames = list(PARAM_RANGES[tr.template])
        P = {nm: np.full(M, tr.params[nm]) for nm in pnames}
        key = tr.noise_key()
        nom_pr, nom_ns = _noise(key, self.T)
        noise = np.empty((M, self.T, NOISE_CH))
        noise[:, :i_star, :] = nom_ns[None, :i_star, :]
        for m in range(M):
            _, sfx = _noise(key + [1000 + m], self.T - i_star)
            noise[m, i_star:, :] = sfx
        per_run = np.tile(nom_pr, (M, 1))
        rng = np.random.default_rng(np.random.SeedSequence(key + [999]))
        eps = kernel.sample(M, rng)
        res = simulate(tr.template, P, self.ego, self.T, self.dt, noise,
                       per_run, eps=eps, i_star=i_star, record=False)
        return res["contact"], res["dv"]


# ======================================================================
# seeded regressions (RQ3)
# ======================================================================
def rate_regression(base: EgoParams, delta: float) -> EgoParams:
    """Weaker achievable braking -> collision rate rises."""
    return replace(base, a_max=base.a_max * (1 - delta),
                   label=f"rate-{delta:g}")


def severity_candidate(base: EgoParams, weak: float, earli: float,
                       eta_s: float = 1.0) -> EgoParams:
    """Comfort recalibration: react EARLIER (looser triggers, more braking
    margin) but with a GENTLER brake cap. Earlier/wider-margin reactions
    save the marginal cases (rate preserved); the deep-emergency crashes
    that remain shed less speed before impact (conditional severity up)."""
    return replace(base, a_max=base.a_max * (1 - weak),
                   a_trig=base.a_trig / earli,
                   ttc_trig=base.ttc_trig * earli,
                   eta0=base.eta0 * eta_s,
                   label=f"sev-w{weak:g}-e{earli:.3f}-m{eta_s:.3f}")


def crash_stats(ego: EgoParams, scenarios, k, seed, dt=DT, t_max_s=T_MAX_S):
    trs = PilotAdapter(ego, dt, t_max_s).run_suite(scenarios, k, seed)
    c = np.array([t.contact for t in trs])
    dvs = np.array([t.dv for t in trs])
    return float(c.mean()), float(dvs[c].mean()) if c.any() else 0.0


def fade_candidate(base: EgoParams, fade: float, earli: float,
                   eta_s: float = 1.0) -> EgoParams:
    """Emergency-actuation defect: brake fade under sustained peak demand.
    Binds almost exclusively in deep emergencies, so it shifts
    conditional-on-crash severity while leaving near-miss behavior nearly
    untouched -- the 'pure' severity regression."""
    return replace(base, fade=fade, a_trig=base.a_trig / earli,
                   ttc_trig=base.ttc_trig * earli, eta0=base.eta0 * eta_s,
                   label=f"fade-{fade:g}-e{earli:.3f}-m{eta_s:.3f}")


def _tune_matched(base, make, scenarios, k, seed, iters=12, earli_hi=2.2,
                  verbose=True, tag=""):
    """Bisect compensation (trigger earliness, then braking margin) until
    the candidate's collision rate matches baseline while Delta-v given
    collision shifts up (the bit-invisible regression)."""
    r0, dv0 = crash_stats(base, scenarios, k, seed)
    lo, hi = 1.0, earli_hi
    r_lo, _ = crash_stats(make(lo, 1.0), scenarios, k, seed)
    r_hi, _ = crash_stats(make(hi, 1.0), scenarios, k, seed)
    if r_lo <= r0:          # the defect alone did not raise the rate
        cand = make(1.0, 1.0)
    elif r_hi > r0:
        # earliness saturated: continue compensation on braking margin
        elo, ehi = 1.0, 1.7
        for _ in range(iters):
            mid = 0.5 * (elo + ehi)
            r_mid, _ = crash_stats(make(earli_hi, mid), scenarios, k, seed)
            if r_mid > r0:
                elo = mid
            else:
                ehi = mid
        cand = make(earli_hi, 0.5 * (elo + ehi))
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            r_mid, _ = crash_stats(make(mid, 1.0), scenarios, k, seed)
            if r_mid > r0:
                lo = mid
            else:
                hi = mid
        cand = make(0.5 * (lo + hi), 1.0)
    r1, dv1 = crash_stats(cand, scenarios, k, seed)
    if verbose:
        print(f"  {tag}: baseline rate={r0:.4f} dv|c={dv0:.2f} -> "
              f"tuned rate={r1:.4f} dv|c={dv1:.2f} ({cand.label})",
              flush=True)
    return cand, {"rate_base": r0, "rate_tuned": r1,
                  "dv_base": dv0, "dv_tuned": dv1}


def tune_severity_regression(base: EgoParams, weak: float, scenarios, k, seed,
                             **kw):
    """Recalibration arm (mixed-sign net harm; kept as a case study)."""
    return _tune_matched(base,
                         lambda e, m: severity_candidate(base, weak, e, m),
                         scenarios, k, seed, tag=f"severity weak={weak:g}",
                         **kw)


def tune_fade_regression(base: EgoParams, fade: float, scenarios, k, seed,
                         **kw):
    """Pure conditional-severity arm (brake fade in deep emergencies)."""
    return _tune_matched(base,
                         lambda e, m: fade_candidate(base, fade, e, m),
                         scenarios, k, seed, tag=f"fade={fade:g}", **kw)


# ======================================================================
# axiomatic battery (metamorphic properties, constructed trace pairs)
# ======================================================================
def _single(adapter, tmpl, params, seed, sid=10 ** 6, run=0):
    sc = Scenario(tmpl, sid, params)
    return adapter.run_suite([sc], 1, seed)[0]


def _H(adapter, tr, kernel, M):
    from proxima.tier1 import label
    return label(adapter, tr, kernel, M)


def axiom_battery(adapter, kernel, M, seed=0, n_mono=20, n_cont=6,
                  ensemble=None):
    """Returns violation counts and raw comparisons for A1-A3."""
    from proxima.features import extract
    rng = np.random.default_rng(seed)
    res = {"A1_monotonicity": [], "A2_invariance": [], "A3_continuity": []}

    # A1: monotonicity in closing speed at equal geometry (lead_stopped)
    for i in range(n_mono):
        v0 = rng.uniform(18, 30)
        d0 = rng.uniform(60, 110)
        t_lo = _single(adapter, "lead_stopped", {"v0": v0, "d0": d0},
                       seed + i)
        t_hi = _single(adapter, "lead_stopped", {"v0": v0 + 3.0, "d0": d0},
                       seed + i)
        L_lo = _H(adapter, t_lo, kernel, M)
        L_hi = _H(adapter, t_hi, kernel, M)
        tol = 1.96 * (L_lo.se + L_hi.se) + 0.01
        res["A1_monotonicity"].append(
            {"v0": v0, "d0": d0, "H_lo": L_lo.H, "H_hi": L_hi.H,
             "violated": bool(L_hi.H < L_lo.H - tol)})

    # A2: invariance under a harm-irrelevant transformation
    # (append benign post-encounter time: verdict must not change)
    long_adapter = PilotAdapter(adapter.ego, adapter.dt,
                                adapter.T * adapter.dt + 4.0)
    for i in range(n_mono // 2):
        v0 = rng.uniform(18, 30)
        d0 = rng.uniform(60, 110)
        t_a = _single(adapter, "lead_stopped", {"v0": v0, "d0": d0}, seed + i)
        t_b = _single(long_adapter, "lead_stopped", {"v0": v0, "d0": d0},
                      seed + i)
        L_a = _H(adapter, t_a, kernel, M)
        L_b = _H(long_adapter, t_b, kernel, M)
        tol = 1.96 * (L_a.se + L_b.se) + 0.02
        entry = {"v0": v0, "d0": d0, "H_a": L_a.H, "H_b": L_b.H,
                 "violated": bool(abs(L_a.H - L_b.H) > tol)}
        if ensemble is not None:
            fa = ensemble.predict_mean(extract(t_a)[None])[0]
            fb = ensemble.predict_mean(extract(t_b)[None])[0]
            entry["tier2_a"], entry["tier2_b"] = float(fa), float(fb)
            entry["tier2_violated"] = bool(abs(fa - fb) > 0.05)
        res["A2_invariance"].append(entry)

    # A3: continuity across the collision boundary (eps-graze vs eps-miss)
    for i in range(n_cont):
        v0 = float(rng.uniform(22, 34))
        lo_d, hi_d = 40.0, 130.0

        def _outcome(d0):
            return _single(adapter, "lead_stopped", {"v0": v0, "d0": d0},
                           seed + 500 + i)
        if not _outcome(lo_d).contact or _outcome(hi_d).contact:
            continue
        for _ in range(24):
            mid = 0.5 * (lo_d + hi_d)
            if _outcome(mid).contact:
                lo_d = mid
            else:
                hi_d = mid
        t_graze = _outcome(lo_d)
        t_miss = _outcome(hi_d)
        L_g = _H(adapter, t_graze, kernel, M)
        L_m = _H(adapter, t_miss, kernel, M)
        tol = 0.05 + 1.96 * (L_g.se + L_m.se)
        entry = {"v0": v0, "d_graze": lo_d, "d_miss": hi_d,
                 "dv_graze": t_graze.dv, "H_graze": L_g.H, "H_miss": L_m.H,
                 "binary_gap": 1.0, "violated": bool(abs(L_g.H - L_m.H) > tol)}
        if ensemble is not None:
            fg = ensemble.predict_mean(extract(t_graze)[None])[0]
            fm = ensemble.predict_mean(extract(t_miss)[None])[0]
            entry["tier2_graze"], entry["tier2_miss"] = float(fg), float(fm)
            entry["tier2_violated"] = bool(abs(fg - fm) > 0.15)
        res["A3_continuity"].append(entry)

    summary = {ax: {"n": len(v),
                    "violations": int(sum(e["violated"] for e in v))}
               for ax, v in res.items()}
    return {"summary": summary, "cases": res}
