"""Bridge: a stored Proxima execution trace -> a CommonRoad scenario.

Purpose. Proxima's telemetry baselines (TTC_min, clearance, ...) are our own
code. A reviewer cannot tell a weak baseline from a weak re-implementation of
a baseline. This module re-expresses each recorded execution in the CommonRoad
scenario format so that the criticality measures of CommonRoad-CriMe (Lin and
Althoff, IV 2023) -- 47 measures, implemented and sign-declared by a third
party -- can be computed on exactly the traces the reference oracle labelled.

No simulation is re-run: the adapters already stored the full planar state of
both actors (ego_x/y/vx/vy/heading, th_*), so the conversion is a change of
representation, not a new experiment.

Road reconstruction. The campaign templates place the ego on a straight road
(measured lateral excursion over the reference sets is <= 0.3 m on TransFuser
and <= 3.9 m on openpilot). The lanelet network is therefore reconstructed as
a straight corridor aligned with the ego's initial heading, centred on the
ego's initial position, with LANES parallel lanes of LANE_W. That is a faithful
reconstruction of the driven geometry for these templates and it is the input
CriMe needs for its curvilinear frame; it is *not* a claim to have recovered
the simulator's map, and metrics that depend on road boundaries inherit this
approximation. Reported as a threat to validity.
"""
from __future__ import annotations

import json

import numpy as np

CAR_LEN = 4.5          # matches the adapters' harvest convention
CAR_W = 1.9
LANE_W = 3.5
LANES = 5              # ego lane +/- 2, covers oncoming and cut-in partners
S_BACK = 60.0          # corridor extent behind the rearmost actor
S_AHEAD = 120.0        # ... and ahead of the foremost
VERT_DS = 2.0          # lanelet vertex spacing

EGO_ID = 1
THREAT_ID = 2


# --------------------------------------------------------------- trace io
def load_npz(path):
    """Raw column dict + metadata for one stored trace."""
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    d = z["data"]
    meta = json.loads(str(z["meta"]))
    return {c: d[:, i].astype(float) for i, c in enumerate(cols)}, meta


def trim_window(c):
    """The window the canonical Trace uses: up to and including first contact.

    Mirrors adapters/*/harvest.to_trace so the criticality measures see exactly
    the execution the reference oracle scored, not a longer or shorter one.
    """
    crash_idx = np.flatnonzero(c["crash"] > 0.5)
    if len(crash_idx):
        return 0, int(crash_idx[0]) + 1, True
    return 0, len(c["crash"]), False


def valid_mask(c, lo, hi):
    """Steps where both actors have a finite planar state."""
    keys = ["ego_x", "ego_y", "ego_vx", "ego_vy", "ego_heading",
            "th_x", "th_y", "th_vx", "th_vy"]
    m = np.ones(hi - lo, dtype=bool)
    for k in keys:
        m &= np.isfinite(c[k][lo:hi])
    return m


def longest_true_run(mask):
    """Start/stop of the longest contiguous True run (CommonRoad trajectories
    must be gap-free)."""
    if not mask.any():
        return 0, 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    starts, stops = idx[0::2], idx[1::2]
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k])


# ----------------------------------------------------------- lane geometry
def lane_frame(c, lo, hi):
    """Unit heading u and origin o defining the reconstructed corridor.

    The heading is taken as the circular mean of the ego heading over the
    window (robust when the ego barely moves, which happens in the
    lead-stopped template), not from a fit to the path.
    """
    h = c["ego_heading"][lo:hi]
    h = h[np.isfinite(h)]
    ang = np.arctan2(np.mean(np.sin(h)), np.mean(np.cos(h)))
    u = np.array([np.cos(ang), np.sin(ang)])
    o = np.array([c["ego_x"][lo], c["ego_y"][lo]])
    return u, o, ang


def build_lanelet_network(c, lo, hi, u, o):
    """Straight LANES-lane corridor spanning both actors' longitudinal extent."""
    from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

    n = np.array([-u[1], u[0]])
    pts = []
    for xk, yk in (("ego_x", "ego_y"), ("th_x", "th_y")):
        x, y = c[xk][lo:hi], c[yk][lo:hi]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            pts.append(np.stack([x[ok], y[ok]], axis=1))
    p = np.concatenate(pts, axis=0) - o
    s = p @ u
    s0, s1 = float(np.min(s)) - S_BACK, float(np.max(s)) + S_AHEAD
    ss = np.arange(s0, s1 + VERT_DS, VERT_DS)

    lanelets = []
    mid = LANES // 2
    for k in range(LANES):
        off = (k - mid) * LANE_W
        centre = o + np.outer(ss, u) + (off * n)
        left = centre + (LANE_W / 2) * n
        right = centre - (LANE_W / 2) * n
        ll = Lanelet(left_vertices=left, center_vertices=centre,
                     right_vertices=right, lanelet_id=100 + k)
        lanelets.append(ll)
    for k in range(LANES):
        if k + 1 < LANES:
            lanelets[k].adj_left = 100 + k + 1
            lanelets[k].adj_left_same_direction = True
        if k - 1 >= 0:
            lanelets[k].adj_right = 100 + k - 1
            lanelets[k].adj_right_same_direction = True
    return LaneletNetwork.create_from_lanelet_list(lanelets), 100 + mid


# ------------------------------------------------------------- conversion
def _states(c, lo, hi, prefix, dt):
    """CustomState list for one actor over [lo, hi)."""
    from commonroad.scenario.state import CustomState

    x, y = c[f"{prefix}_x"][lo:hi], c[f"{prefix}_y"][lo:hi]
    vx, vy = c[f"{prefix}_vx"][lo:hi], c[f"{prefix}_vy"][lo:hi]
    hd = c.get(f"{prefix}_heading")
    v = np.hypot(vx, vy)
    if hd is None:
        ori = np.arctan2(vy, vx)
    else:
        ori = np.unwrap(hd[lo:hi])
    out = []
    for i in range(hi - lo):
        out.append(CustomState(time_step=i,
                               position=np.array([x[i], y[i]]),
                               orientation=float(ori[i]),
                               velocity=float(v[i]),
                               yaw_rate=0.0, slip_angle=0.0))
    return out


def _obstacle(states, obs_id):
    from commonroad.geometry.shape import Rectangle
    from commonroad.prediction.prediction import TrajectoryPrediction
    from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
    from commonroad.scenario.trajectory import Trajectory

    from commonroad.scenario.state import InitialState

    shape = Rectangle(length=CAR_LEN, width=CAR_W)
    s0 = states[0]
    init = InitialState(time_step=0, position=s0.position,
                        orientation=s0.orientation, velocity=s0.velocity,
                        acceleration=0.0, yaw_rate=0.0, slip_angle=0.0)
    pred = TrajectoryPrediction(Trajectory(1, states[1:]), shape)
    return DynamicObstacle(obstacle_id=obs_id, obstacle_type=ObstacleType.CAR,
                           obstacle_shape=shape, initial_state=init,
                           prediction=pred)


def trace_to_scenario(path, dt=0.05):
    """Convert one stored trace into (scenario, info).

    info carries the bookkeeping the metric runner needs: the contact flag and
    impact speed as the adapters computed them, the criticality index used by
    Proxima's own t_crit anchoring, and the number of usable time steps.
    Returns (None, info) when the trace has no usable contiguous window.
    """
    from commonroad.scenario.scenario import Scenario, ScenarioID

    c, meta = load_npz(path)
    lo, hi, contact = trim_window(c)
    m = valid_mask(c, lo, hi)
    a, b = longest_true_run(m)
    lo2, hi2 = lo + a, lo + b
    info = {"path": path, "scenario": meta.get("scenario"),
            "contact": bool(contact), "n_steps": int(hi2 - lo2),
            "lo": int(lo2), "hi": int(hi2), "meta": meta}
    if hi2 - lo2 < 5:
        info["error"] = "no usable window"
        return None, info

    if contact:
        i = hi - 1
        info["dv"] = float(np.hypot(c["ego_vx"][i] - c["th_vx"][i],
                                    c["ego_vy"][i] - c["th_vy"][i]))
        if not np.isfinite(info["dv"]):
            info["dv"] = float(np.hypot(c["ego_vx"][i], c["ego_vy"][i]))
    else:
        info["dv"] = 0.0

    u, o, _ = lane_frame(c, lo2, hi2)
    net, ego_lane = build_lanelet_network(c, lo2, hi2, u, o)
    sc = Scenario(dt=dt, scenario_id=ScenarioID.from_benchmark_id(
        "ZAM_Proxima-1_1_T-1", "2020a"))
    # CriMe's intersection measures test `Tag.INTERSECTION not in sce.tags`;
    # an unset tags attribute raises there. The reconstructed corridor has no
    # intersection, so an empty tag set is both true and the value those
    # measures need to report themselves as not applicable.
    sc.tags = set()
    sc.add_objects(net)
    sc.add_objects(_obstacle(_states(c, lo2, hi2, "ego", dt), EGO_ID))
    sc.add_objects(_obstacle(_states(c, lo2, hi2, "th", dt), THREAT_ID))
    sc.assign_obstacles_to_lanelets()
    info["ego_lanelet"] = ego_lane
    return sc, info
