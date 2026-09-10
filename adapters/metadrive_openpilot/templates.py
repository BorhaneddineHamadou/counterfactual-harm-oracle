"""The five pre-crash templates for the openpilot+MetaDrive stack.

Five templates matching the pilot world, implemented against the existing
SceRTScenario interface (map_config / env_config / on_reset / on_step) so the
campaign bridge machinery runs them unchanged. Concrete parameters arrive via
the PROXIMA_SCENARIO_PARAMS env var (JSON) because the scenario instance is
re-created by name inside the MetaDrive subprocess.

Design notes (learned from the first smoke run):
  * The ego resets onto a SHORT spawn block, so threat placement must walk
    successor lanes by arc distance (_pos_ahead) -- naive spawn_longitude
    on the ego's current lane collapses the gap.
  * The ego starts from standstill and takes ~10 sim-s to reach speed, and
    the env runs ~5-10x slower than wall time -- so conflict triggers are
    STATE-based (ego-threat distance), not fixed times.

Kernel perturbations of the surrounding actor (OU noise, gated by
start_step) are applied here via PROXIMA_PERTURB; ego-side perturbations
(latency, brake effectiveness) live in the bridge.
"""
import json
import os
import sys

import numpy as np

OP_SIM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openpilot_sim")
if OP_SIM_ROOT not in sys.path:
    sys.path.insert(0, OP_SIM_ROOT)

from scenarios.base import SceRTScenario, straight, bidirection  # noqa: E402

ENV_STEP_HZ = 20.0


def _params():
    raw = os.environ.get("PROXIMA_SCENARIO_PARAMS", "")
    return json.loads(raw) if raw else {}


class _OUPerturb:
    """Ornstein-Uhlenbeck actor noise, active from start_step, seeded."""

    def __init__(self):
        raw = os.environ.get("PROXIMA_PERTURB", "")
        cfg = json.loads(raw) if raw else {}
        self.start = int(cfg.get("start_step", 0))
        self.sig_l = float(cfg.get("ou_sigma_long", 0.0))
        self.sig_lat = float(cfg.get("ou_sigma_lat", 0.0))
        self.theta = float(cfg.get("ou_theta", 1.0))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.ou_l = 0.0
        self.ou_lat = 0.0
        self.active = self.sig_l > 0 or self.sig_lat > 0

    def step(self, step, dt=1.0 / ENV_STEP_HZ):
        if not self.active or step < self.start:
            return 0.0, 0.0
        sq = np.sqrt(dt)
        self.ou_l += (-self.theta * self.ou_l * dt
                      + self.sig_l * sq * self.rng.standard_normal())
        self.ou_lat += (-self.theta * self.ou_lat * dt
                        + self.sig_lat * sq * self.rng.standard_normal())
        return self.ou_l, self.ou_lat


class _ProximaScenario(SceRTScenario):
    """Base: parameter loading, OU perturbation, robust geometry helpers."""

    def __init__(self):
        self.p = _params()
        self.ou = _OUPerturb()
        self._threat = None
        self.threat_kind = "vehicle"

    def threat_state(self, env):
        t = self._threat
        if t is None:
            return None
        try:
            return {"x": float(t.position[0]), "y": float(t.position[1]),
                    "vx": float(t.velocity[0]), "vy": float(t.velocity[1]),
                    "heading": float(t.heading_theta)}
        except Exception:
            return None

    def _ego_lane(self, env):
        return env.vehicle.navigation.current_lane

    def _set_speed_along_heading(self, obj, speed):
        h = obj.heading_theta
        obj.set_velocity([speed * np.cos(h), speed * np.sin(h)])

    def _ego_threat_dist(self, env):
        """Euclidean ego->threat distance (m)."""
        if self._threat is None:
            return 1e9
        d = (np.asarray(self._threat.position[:2])
             - np.asarray(env.vehicle.position[:2]))
        return float(np.linalg.norm(d))

    def _pos_ahead(self, env, dist):
        """(lane, lon) at arc distance `dist` ahead of the ego along its
        route, walking successor lanes past the short spawn block."""
        lane = self._ego_lane(env)
        lon = lane.local_coordinates(env.vehicle.position)[0] + dist
        net = env.current_map.road_network.graph
        li = lane.index
        for _ in range(8):
            if lon <= lane.length * 0.95:
                break
            succ = net.get(li[1])
            if not succ:
                break
            end = next(iter(succ))
            lanes = succ[end]
            idx = min(li[2] if len(li) > 2 else 0, len(lanes) - 1)
            lon -= lane.length
            lane = lanes[idx]
            li = lane.index
        return lane, float(min(max(lon, 1.0), lane.length * 0.95))

    def _spawn_vehicle_ahead(self, env, dist, static=False, lane_shift=0):
        from metadrive.component.vehicle.vehicle_type import (
            DefaultVehicle, StaticDefaultVehicle)
        lane, lon = self._pos_ahead(env, dist)
        if lane_shift:
            net = env.current_map.road_network.graph
            li = lane.index
            lanes = net[li[0]][li[1]]
            idx = li[2] if len(li) > 2 else 0
            j = min(max(idx + lane_shift, 0), len(lanes) - 1)
            lane = lanes[j]
        vc = self._vehicle_config(env)
        # video/demo runs: render the threat (campaign default keeps the
        # bridge's render_vehicle=False; needs full MetaDrive assets)
        if os.environ.get("PROXIMA_RENDER_THREATS") == "1":
            vc["render_vehicle"] = True
        vc["spawn_lane_index"] = lane.index
        vc["spawn_longitude"] = lon
        cls = StaticDefaultVehicle if static else DefaultVehicle
        v = env.engine.spawn_object(cls, vehicle_config=vc, random_seed=0)
        return v, lane, lon


class ProxLeadDecel(_ProximaScenario):
    """Lead ahead at gap0 cruising at v_lead; brakes to a stop at a_lead
    once the ego closes within g_trig. Params: gap0, v_lead, g_trig,
    a_lead."""
    NAME = "prox_lead_decel"

    def map_config(self):
        from metadrive.component.map.pg_map import MapGenerateMethod
        return {"type": MapGenerateMethod.PG_MAP_FILE, "lane_num": 2,
                "lane_width": 3.5,
                "config": [None, straight(250), straight(250)]}

    def on_reset(self, env):
        gap0 = float(self.p.get("gap0", 60.0))
        self._threat, _, _ = self._spawn_vehicle_ahead(env, gap0)
        self._v_cmd = float(self.p.get("v_lead", 10.0))
        self._braking = False

    def on_step(self, env, step):
        dt = 1.0 / ENV_STEP_HZ
        g_trig = float(self.p.get("g_trig", 30.0))
        a_lead = float(self.p.get("a_lead", 4.0))
        if not self._braking and self._ego_threat_dist(env) < g_trig:
            self._braking = True
        if self._braking:
            self._v_cmd = max(0.0, self._v_cmd - a_lead * dt)
        ou_l, _ = self.ou.step(step)
        if self._threat is not None:
            self._set_speed_along_heading(
                self._threat, max(0.0, self._v_cmd + ou_l))


class ProxLeadStopped(_ProximaScenario):
    """Stationary vehicle in the ego lane at arc distance d0. Params: d0."""
    NAME = "prox_lead_stopped"

    def map_config(self):
        from metadrive.component.map.pg_map import MapGenerateMethod
        return {"type": MapGenerateMethod.PG_MAP_FILE, "lane_num": 2,
                "lane_width": 3.5,
                "config": [None, straight(300), straight(200)]}

    def on_reset(self, env):
        d0 = float(self.p.get("d0", 140.0))
        self._threat, _, _ = self._spawn_vehicle_ahead(env, d0, static=True)

    def on_step(self, env, step):
        pass


class ProxCutIn(_ProximaScenario):
    """Adjacent-lane vehicle ahead cuts into the ego lane once the ego
    closes within d_trig, sliding over t_lat seconds, then optionally
    braking. Params: gap0, dv_rel, d_trig, t_lat, a_cut."""
    NAME = "prox_cut_in"

    def map_config(self):
        from metadrive.component.map.pg_map import MapGenerateMethod
        return {"type": MapGenerateMethod.PG_MAP_FILE, "lane_num": 2,
                "lane_width": 3.5,
                "config": [None, straight(280), straight(200)]}

    def on_reset(self, env):
        gap0 = float(self.p.get("gap0", 30.0))
        self._threat, lane, lon = self._spawn_vehicle_ahead(
            env, gap0, lane_shift=+1)
        self._lane_from = lane
        ego_lane, _ = self._pos_ahead(env, gap0)
        self._lane_to = ego_lane
        self._v_cmd = max(0.5, float(self.p.get("v_cut", 8.0)))
        self._cut_t0 = None
        self._cut_prog = 0.0

    def on_step(self, env, step):
        dt = 1.0 / ENV_STEP_HZ
        d_trig = float(self.p.get("d_trig", 25.0))
        t_lat = float(self.p.get("t_lat", 2.0))
        a_cut = float(self.p.get("a_cut", 0.0))
        ou_l, ou_lat = self.ou.step(step)
        th = self._threat
        if th is None:
            return
        if self._cut_t0 is None and self._ego_threat_dist(env) < d_trig:
            self._cut_t0 = step
        if self._cut_t0 is not None:
            tc = (step - self._cut_t0) * dt
            if a_cut > 0 and tc > t_lat * 0.5:
                self._v_cmd = max(0.5, self._v_cmd - a_cut * dt)
            prog = min(1.0, tc / max(t_lat, 0.1))
            self._cut_prog = prog
            # slide laterally from source lane centre toward ego lane centre
            lon_f = self._lane_from.local_coordinates(th.position)[0]
            p_from = np.asarray(self._lane_from.position(
                min(lon_f, self._lane_from.length * 0.98), 0))
            lon_t = self._lane_to.local_coordinates(th.position)[0]
            p_to = np.asarray(self._lane_to.position(
                min(lon_t, self._lane_to.length * 0.98), 0))
            tgt = (1 - prog) * p_from + prog * p_to
            h = th.heading_theta
            lat_dir = np.array([-np.sin(h), np.cos(h)])
            tgt = tgt + ou_lat * lat_dir
            cur = np.asarray(th.position[:2])
            delta = tgt - cur
            lat_delta = np.dot(delta, lat_dir) * lat_dir
            if os.environ.get("PROXIMA_STEERED_DRIFT") == "1":
                # video demos only: DRIVE a curved path toward the drift
                # target like a real lane change (pure-pursuit on an aim
                # point ahead, rate-limited steering). The campaign keeps
                # the kinematic lateral slide below.
                if not hasattr(self, "_h_lane"):
                    self._h_lane = float(h)
                fwd = np.array([np.cos(self._h_lane), np.sin(self._h_lane)])
                aim = tgt + fwd * 8.0
                des = float(np.arctan2(aim[1] - cur[1], aim[0] - cur[0]))
                dh = (des - h + np.pi) % (2 * np.pi) - np.pi
                th.set_heading_theta(h + float(np.clip(dh, -0.035, 0.035)))
            else:
                # apply only the lateral component; longitudinal from speed
                th.set_position([float(cur[0] + lat_delta[0]),
                                 float(cur[1] + lat_delta[1])])
        self._set_speed_along_heading(th, max(0.0, self._v_cmd + ou_l))


class ProxOncomingDrift(_ProximaScenario):
    """Oncoming vehicle drifts into the ego lane when the separation drops
    below d_trig, holds t_stay, then returns. Params: r0, v_onc, d_trig,
    vy_drift, y_frac, t_stay."""
    NAME = "prox_oncoming_drift"

    def map_config(self):
        from metadrive.component.map.pg_map import MapGenerateMethod
        return {"type": MapGenerateMethod.PG_MAP_FILE, "lane_num": 1,
                "lane_width": 3.8,
                "config": [None, bidirection(300), straight(100)]}

    def on_reset(self, env):
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle
        r0 = float(self.p.get("r0", 160.0))
        net = env.current_map.road_network.graph
        # parametrization-agnostic placement: scan EVERY opposing-direction
        # lane for the arc position whose Euclidean distance to the ego is
        # closest to r0, constrained to lie AHEAD of the ego (naive
        # projections can go negative and collapse onto the ego; the first
        # "-" node in dict order may be a short block behind the ego)
        ego_pos = np.asarray(env.vehicle.position[:2])
        h = env.vehicle.heading_theta
        u = np.array([np.cos(h), np.sin(h)])
        best = None      # (err, lane, lon)
        for s, ends in net.items():
            if not (isinstance(s, str) and s.startswith("-")):
                continue
            for e, lanes in ends.items():
                lane = lanes[0]
                cands = np.linspace(1.0, lane.length * 0.98,
                                    max(24, int(lane.length / 2)))
                pts = np.array([lane.position(float(c), 0) for c in cands])
                rel = pts - ego_pos[None, :]
                ahead = rel @ u
                dist = np.linalg.norm(rel, axis=1)
                ok = ahead > 15.0
                if not ok.any():
                    continue
                err = np.where(ok, np.abs(dist - r0), 1e9)
                i = int(np.argmin(err))
                if best is None or err[i] < best[0]:
                    best = (float(err[i]), lane, float(cands[i]))
        if best is None:
            self._opp_lane = None
            self._threat = None
            return
        _, opp, lon_opp = best
        self._opp_lane = opp
        vc = self._vehicle_config(env)
        if os.environ.get("PROXIMA_RENDER_THREATS") == "1":
            vc["render_vehicle"] = True
        vc["spawn_lane_index"] = opp.index
        vc["spawn_longitude"] = lon_opp
        self._threat = env.engine.spawn_object(
            DefaultVehicle, vehicle_config=vc, random_seed=0)
        self._v_cmd = float(self.p.get("v_onc", 10.0))
        self._drift_t0 = None

    def on_step(self, env, step):
        dt = 1.0 / ENV_STEP_HZ
        d_trig = float(self.p.get("d_trig", 90.0))
        vy = float(self.p.get("vy_drift", 0.7))
        y_peak = float(self.p.get("y_frac", 0.6)) * 3.8
        t_stay = float(self.p.get("t_stay", 1.5))
        ou_l, ou_lat = self.ou.step(step)
        th = self._threat
        if th is None or self._opp_lane is None:
            return
        self._set_speed_along_heading(th, max(1.0, self._v_cmd + ou_l))
        if self._drift_t0 is None and self._ego_threat_dist(env) < d_trig:
            self._drift_t0 = step
        if self._drift_t0 is None:
            off = 0.0
        else:
            t = (step - self._drift_t0) * dt
            t1 = y_peak / max(vy, 0.05)
            if t < t1:
                off = vy * t
            elif t < t1 + t_stay:
                off = y_peak
            else:
                off = max(0.0, y_peak - vy * (t - t1 - t_stay))
        lon = float(np.clip(self._opp_lane.local_coordinates(
            th.position)[0], 0.5, self._opp_lane.length * 0.99))
        base = np.asarray(self._opp_lane.position(lon, 0))
        h = th.heading_theta
        lat_dir = np.array([-np.sin(h), np.cos(h)])
        pos = base + (off + ou_lat) * lat_dir
        th.set_position([float(pos[0]), float(pos[1])])


class ProxCrossingTraffic(_ProximaScenario):
    """Vehicle crosses the ego's path perpendicularly (T-bone geometry).
    Substitutes the pilot's pedestrian template on this stack: the MetaDrive
    install ships no VRU assets, and the threat must render to be perceived
    by openpilot's camera. Placed d_place ahead, y0 off the road edge;
    starts crossing at v_cross when the ego closes within d_trig.
    Params: d_place, d_trig, v_cross, y0."""
    NAME = "prox_crossing_traffic"

    def map_config(self):
        from metadrive.component.map.pg_map import MapGenerateMethod
        return {"type": MapGenerateMethod.PG_MAP_FILE, "lane_num": 2,
                "lane_width": 3.5,
                "config": [None, straight(280), straight(150)]}

    def on_reset(self, env):
        d_place = float(self.p.get("d_place", 140.0))
        y0 = float(self.p.get("y0", 7.0))
        # spawn legally on a lane, then teleport to the perpendicular
        # roadside pose (vehicles render and are perceived; VRUs are not
        # available in this install)
        self._threat, lane, lon = self._spawn_vehicle_ahead(env, d_place)
        base = np.asarray(lane.position(lon, 0))
        h = lane.heading_theta_at(lon)
        lat_dir = np.array([-np.sin(h), np.cos(h)])
        pos = base - y0 * lat_dir
        self._threat.set_position([float(pos[0]), float(pos[1])])
        try:
            self._threat.set_heading_theta(float(h + np.pi / 2))
        except Exception:
            pass
        self._threat.set_velocity([0.0, 0.0])
        self._lat_dir = lat_dir
        self._started = False

    def on_step(self, env, step):
        d_trig = float(self.p.get("d_trig", 50.0))
        v_cross = float(self.p.get("v_cross", 3.0))
        ou_l, _ = self.ou.step(step)
        th = self._threat
        if th is None:
            return
        if not self._started and self._ego_threat_dist(env) < d_trig:
            self._started = True
        if self._started:
            speed = max(0.2, v_cross + ou_l)
            v = self._lat_dir * speed
            th.set_velocity([float(v[0]), float(v[1])])


PROXIMA_SCENARIOS = [ProxLeadDecel, ProxLeadStopped, ProxCutIn,
                     ProxOncomingDrift, ProxCrossingTraffic]
