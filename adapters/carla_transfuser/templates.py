"""The five pre-crash templates for the CARLA stack (used by TransFuser).

Same five templates and parameter names as the MetaDrive adapter, so the
calibrated ranges format and the campaign driver carry over. Runs inside the
Proxima runner process (a plain CARLA client, no ROS).

Design: everything is choreographed in the (s, lat) frame of the EGO lane
centerline (a dense polyline walked with the waypoint API from the hero's
spawn lane). Threats are physics-ON vehicles driven by set_target_velocity
(so the simulator reports their true velocity to the agent's sensors) plus lateral/heading correction via set_transform — the same
"longitudinal velocity, lateral teleport" semantics as the MetaDrive
templates. Conflict triggers are STATE-based (ego-threat Euclidean
distance), matching the MetaDrive adapter and pilot world.

Choreography starts when the runner arms the scenario (ego engaged and
moving); threats hold still before that. Kernel OU actor noise comes from
the job's perturb blob and is gated by the runner (active from t*).
"""
import math

import numpy as np

THREAT_BP = "vehicle.tesla.model3"


class LanePath:
    """Arc-length-parameterized centerline polyline of one lane, walked
    forward from a start waypoint. pose(s, lat) returns (x, y, heading_rad)
    with lat measured along the local +90-degree normal (left-handed CARLA
    frame; side signs are resolved empirically by the scenarios)."""

    def __init__(self, start_wp, length=500.0, step=0.5):
        self.lane_width = float(start_wp.lane_width) or 3.5
        pts, hs, ss = [], [], [0.0]
        cur = start_wp
        pts.append((cur.transform.location.x, cur.transform.location.y))
        hs.append(math.radians(cur.transform.rotation.yaw))
        dist = 0.0
        while dist < length:
            nxt = cur.next(step)
            if not nxt:
                break
            cur = nxt[0]
            x, y = cur.transform.location.x, cur.transform.location.y
            d = math.hypot(x - pts[-1][0], y - pts[-1][1])
            if d < 1e-3:
                continue
            dist += d
            pts.append((x, y))
            hs.append(math.radians(cur.transform.rotation.yaw))
            ss.append(dist)
        self.xy = np.asarray(pts, float)
        self.h = np.asarray(hs, float)
        self.s = np.asarray(ss, float)
        self.length = dist

    def pose(self, s, lat=0.0):
        s = float(np.clip(s, 0.0, self.length))
        x = float(np.interp(s, self.s, self.xy[:, 0]))
        y = float(np.interp(s, self.s, self.xy[:, 1]))
        i = min(int(np.searchsorted(self.s, s)), len(self.h) - 1)
        h = float(self.h[i])
        nx, ny = -math.sin(h), math.cos(h)
        return x + lat * nx, y + lat * ny, h

    def project(self, x, y, s_hint, window=40.0):
        """Arc position + signed lateral offset of (x, y), searched in a
        window around s_hint (avoids far-field aliasing on loops)."""
        lo = max(0.0, s_hint - window)
        hi = min(self.length, s_hint + window)
        cand = np.linspace(lo, hi, max(8, int((hi - lo) * 2)))
        px = np.interp(cand, self.s, self.xy[:, 0])
        py = np.interp(cand, self.s, self.xy[:, 1])
        d2 = (px - x) ** 2 + (py - y) ** 2
        i = int(np.argmin(d2))
        s = float(cand[i])
        _, _, h = self.pose(s)
        lat = -(x - px[i]) * math.sin(h) + (y - py[i]) * math.cos(h)
        return s, float(lat)


class _OUNoise:
    """Ornstein-Uhlenbeck actor noise; activated by the runner from t*."""

    def __init__(self, perturb, seed):
        cfg = perturb or {}
        self.sig_l = float(cfg.get("ou_sigma_long", 0.0))
        self.sig_lat = float(cfg.get("ou_sigma_lat", 0.0))
        self.theta = float(cfg.get("ou_theta", 1.0))
        self.rng = np.random.default_rng(int(cfg.get("seed", seed)))
        self.ou_l = 0.0
        self.ou_lat = 0.0

    def step(self, dt, active):
        if not active or (self.sig_l == 0.0 and self.sig_lat == 0.0):
            return 0.0, 0.0
        dt = max(min(dt, 0.25), 1e-3)
        sq = math.sqrt(dt)
        self.ou_l += (-self.theta * self.ou_l * dt
                      + self.sig_l * sq * self.rng.standard_normal())
        self.ou_lat += (-self.theta * self.ou_lat * dt
                        + self.sig_lat * sq * self.rng.standard_normal())
        return self.ou_l, self.ou_lat


class _ProximaScenario:
    """Base: ego-path frame, threat spawn/drive helpers, OU noise."""

    threat_kind = "vehicle"

    def __init__(self, carla_mod, world, hero, params, seed, perturb=None):
        self.carla = carla_mod
        self.world = world
        self.hero = hero
        self.p = params
        self.ou = _OUNoise(perturb, seed)
        self._threat = None
        wp = world.get_map().get_waypoint(hero.get_transform().location)
        self.path = LanePath(wp, length=500.0)
        # The ego is at the path start by construction (path is built from
        # its own waypoint) — keep the search window local. A full-length
        # window aliases to far-field s on looping town roads (Town01),
        # spawning threats hundreds of meters up the path.
        s0, _ = self.path.project(hero.get_transform().location.x,
                                  hero.get_transform().location.y, 0.0)
        self.ego_s0 = s0
        self.lane_w = self.path.lane_width
        # Side of the opposing lane: +lat or -lat, resolved from lane ids
        # (Town01 roads are one lane per direction; the adjacent lane with
        # flipped lane_id sign is the oncoming lane).
        self._opp_sign = self._resolve_opp_sign(wp)

    def _resolve_opp_sign(self, wp):
        m = self.world.get_map()
        for sign in (+1.0, -1.0):
            x, y, _ = self.path.pose(self.ego_s0 + 10.0, sign * self.lane_w)
            w = m.get_waypoint(self.carla.Location(x=x, y=y, z=0.3),
                               project_to_road=True)
            if w is not None and w.lane_id * wp.lane_id < 0:
                return sign
        return +1.0

    # ------------------------------------------------------------- helpers
    def _spawn_threat(self, s, lat=0.0, yaw_offset_deg=0.0):
        bp = self.world.get_blueprint_library().find(THREAT_BP)
        bp.set_attribute("role_name", "proxima_threat")
        x, y, h = self.path.pose(s, lat)
        yaw = math.degrees(h) + yaw_offset_deg
        # base spawn height on the road elevation at (x, y): a fixed world z
        # puts the threat under elevated roads/bridges, invisible to the SUT
        # (observed on the Town01 river road)
        wp = self.world.get_map().get_waypoint(
            self.carla.Location(x=x, y=y, z=0.0), project_to_road=True)
        base_z = wp.transform.location.z if wp is not None else 0.0
        for dz in (0.3, 0.8, 1.5):
            tf = self.carla.Transform(
                self.carla.Location(x=x, y=y, z=base_z + dz),
                self.carla.Rotation(yaw=yaw))
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is not None:
                self._threat = actor
                self._th_s = float(s)
                self._th_lat = float(lat)
                self._th_yaw = math.radians(yaw)
                return actor
        raise RuntimeError(f"threat spawn failed at s={s:.1f} lat={lat:.1f}")

    def _drive_threat(self, v, lat=None, dt=0.05, reverse=False,
                      yaw_rad=None):
        """Advance the threat kinematically along the path frame: physics
        velocity for longitudinal motion (visible to gt_objects), transform
        correction for lateral offset + heading."""
        th = self._threat
        if th is None:
            return
        loc = th.get_transform().location
        self._th_s, cur_lat = self.path.project(loc.x, loc.y, self._th_s)
        direction = -1.0 if reverse else 1.0
        # park at the path boundary: past it the projection clamps and the
        # lateral correction makes the actor bounce/teleport in place
        if (self._th_s <= 0.5 and direction < 0) or \
                (self._th_s >= self.path.length - 0.5 and direction > 0):
            th.set_target_velocity(self.carla.Vector3D())
            th.set_target_angular_velocity(self.carla.Vector3D())
            return
        x, y, h = self.path.pose(self._th_s, cur_lat if lat is None else lat)
        hdg = h + (math.pi if reverse else 0.0) if yaw_rad is None else yaw_rad
        vx, vy = v * direction * math.cos(h), v * direction * math.sin(h)
        th.set_target_velocity(self.carla.Vector3D(x=vx, y=vy, z=0.0))
        th.set_target_angular_velocity(self.carla.Vector3D())
        # lateral/heading correction only when choreography demands it
        if lat is not None and abs(cur_lat - lat) > 0.03:
            th.set_transform(self.carla.Transform(
                self.carla.Location(x=x, y=y, z=loc.z),
                self.carla.Rotation(yaw=math.degrees(hdg))))

    def ego_threat_dist(self):
        if self._threat is None:
            return 1e9
        a = self.hero.get_transform().location
        b = self._threat.get_transform().location
        return math.hypot(b.x - a.x, b.y - a.y)

    def threat_state(self):
        th = self._threat
        if th is None:
            return None
        tf = th.get_transform()
        v = th.get_velocity()
        return {"x": tf.location.x, "y": tf.location.y,
                "vx": v.x, "vy": v.y,
                "heading": math.radians(tf.rotation.yaw)}

    def destroy(self):
        if self._threat is not None:
            try:
                self._threat.destroy()
            except Exception:
                pass
            self._threat = None

    # ------------------------------------------------------ scenario hooks
    def on_spawn(self):
        raise NotImplementedError

    def on_step(self, t, dt, perturb_active):
        raise NotImplementedError


class ProxLeadDecel(_ProximaScenario):
    """Lead ahead at gap0 cruising at v_lead; brakes to a stop at a_lead
    once the ego closes within g_trig. Params: gap0, v_lead, g_trig,
    a_lead."""
    NAME = "prox_lead_decel"

    def on_spawn(self):
        self._spawn_threat(self.ego_s0 + float(self.p.get("gap0", 60.0)))
        self._v_cmd = float(self.p.get("v_lead", 10.0))
        self._braking = False

    def on_step(self, t, dt, perturb_active):
        if not self._braking and \
                self.ego_threat_dist() < float(self.p.get("g_trig", 30.0)):
            self._braking = True
        if self._braking:
            self._v_cmd = max(0.0, self._v_cmd
                              - float(self.p.get("a_lead", 4.0)) * dt)
        ou_l, _ = self.ou.step(dt, perturb_active)
        self._drive_threat(max(0.0, self._v_cmd + ou_l), lat=0.0, dt=dt)


class ProxLeadStopped(_ProximaScenario):
    """Stationary vehicle in the ego lane at arc distance d0. Params: d0."""
    NAME = "prox_lead_stopped"

    def on_spawn(self):
        self._spawn_threat(self.ego_s0 + float(self.p.get("d0", 140.0)))
        self._threat.set_target_velocity(self.carla.Vector3D())

    def on_step(self, t, dt, perturb_active):
        pass


class ProxCutIn(_ProximaScenario):
    """Vehicle ahead in the adjacent (oncoming, Town01 is 1+1) lane driving
    the EGO direction — an overtake-return geometry — cuts into the ego lane
    over t_lat seconds once the ego closes within d_trig, then optionally
    brakes at a_cut. Params: gap0, v_cut, d_trig, t_lat, a_cut."""
    NAME = "prox_cut_in"

    def on_spawn(self):
        self._spawn_threat(self.ego_s0 + float(self.p.get("gap0", 30.0)),
                           lat=self._opp_sign * self.lane_w)
        self._v_cmd = max(0.5, float(self.p.get("v_cut", 8.0)))
        self._cut_t0 = None

    def on_step(self, t, dt, perturb_active):
        t_lat = float(self.p.get("t_lat", 2.0))
        a_cut = float(self.p.get("a_cut", 0.0))
        ou_l, ou_lat = self.ou.step(dt, perturb_active)
        if self._cut_t0 is None and \
                self.ego_threat_dist() < float(self.p.get("d_trig", 25.0)):
            self._cut_t0 = t
        lat0 = self._opp_sign * self.lane_w
        if self._cut_t0 is None:
            lat = lat0
        else:
            tc = t - self._cut_t0
            if a_cut > 0 and tc > t_lat * 0.5:
                self._v_cmd = max(0.5, self._v_cmd - a_cut * dt)
            prog = min(1.0, tc / max(t_lat, 0.1))
            lat = (1.0 - prog) * lat0
        self._drive_threat(max(0.0, self._v_cmd + ou_l),
                           lat=lat + ou_lat, dt=dt)


class ProxOncomingDrift(_ProximaScenario):
    """Oncoming vehicle in the opposing lane at r0 drifts into the ego lane
    when separation drops below d_trig, holds t_stay, then returns.
    Params: r0, v_onc, d_trig, vy_drift, y_frac, t_stay."""
    NAME = "prox_oncoming_drift"

    def on_spawn(self):
        self._spawn_threat(self.ego_s0 + float(self.p.get("r0", 160.0)),
                           lat=self._opp_sign * self.lane_w,
                           yaw_offset_deg=180.0)
        self._v_cmd = float(self.p.get("v_onc", 10.0))
        self._drift_t0 = None

    def on_step(self, t, dt, perturb_active):
        vy = float(self.p.get("vy_drift", 0.7))
        y_peak = float(self.p.get("y_frac", 0.6)) * self.lane_w
        t_stay = float(self.p.get("t_stay", 1.5))
        ou_l, ou_lat = self.ou.step(dt, perturb_active)
        if self._drift_t0 is None and \
                self.ego_threat_dist() < float(self.p.get("d_trig", 90.0)):
            self._drift_t0 = t
        if self._drift_t0 is None:
            off = 0.0
        else:
            td = t - self._drift_t0
            t1 = y_peak / max(vy, 0.05)
            if td < t1:
                off = vy * td
            elif td < t1 + t_stay:
                off = y_peak
            else:
                off = max(0.0, y_peak - vy * (td - t1 - t_stay))
        lat = self._opp_sign * (self.lane_w - off)
        self._drive_threat(max(1.0, self._v_cmd + ou_l), lat=lat + ou_lat,
                           dt=dt, reverse=True)


class ProxCrossingTraffic(_ProximaScenario):
    """Vehicle crosses the ego's path perpendicularly (T-bone geometry),
    starting y0 beyond the ego-lane road edge, d_place ahead; begins
    crossing at v_cross when the ego closes within d_trig.
    Params: d_place, d_trig, v_cross, y0."""
    NAME = "prox_crossing_traffic"

    def on_spawn(self):
        d_place = float(self.p.get("d_place", 140.0))
        y0 = float(self.p.get("y0", 7.0))
        edge = -self._opp_sign          # road edge on the ego-lane side
        self._cross_dir = -edge          # crossing back toward the road
        self._lat = edge * (self.lane_w / 2.0 + y0)
        yaw_off = 90.0 if self._cross_dir > 0 else -90.0
        self._spawn_threat(self.ego_s0 + d_place, lat=self._lat,
                           yaw_offset_deg=yaw_off)
        self._threat.set_target_velocity(self.carla.Vector3D())
        self._started = False
        self._cross_s = self.ego_s0 + d_place

    def on_step(self, t, dt, perturb_active):
        v_cross = float(self.p.get("v_cross", 3.0))
        ou_l, _ = self.ou.step(dt, perturb_active)
        th = self._threat
        if th is None:
            return
        if not self._started and \
                self.ego_threat_dist() < float(self.p.get("d_trig", 50.0)):
            self._started = True
        if not self._started:
            return
        speed = max(0.2, v_cross + ou_l)
        self._lat += self._cross_dir * speed * dt
        x, y, h = self.path.pose(self._cross_s, self._lat)
        nx, ny = -math.sin(h), math.cos(h)
        th.set_target_velocity(self.carla.Vector3D(
            x=self._cross_dir * speed * nx,
            y=self._cross_dir * speed * ny, z=0.0))
        th.set_target_angular_velocity(self.carla.Vector3D())


PROXIMA_SCENARIOS = [ProxLeadDecel, ProxLeadStopped, ProxCutIn,
                     ProxOncomingDrift, ProxCrossingTraffic]
