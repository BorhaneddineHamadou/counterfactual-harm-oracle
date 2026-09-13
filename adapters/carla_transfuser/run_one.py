"""One TransFuser run: sync CARLA world + choreographed template + trace.

Usage: run_one.py '<job json>'   (or via worker.py)
Job fields (same contract as the other adapters): job_id, scenario_cls,
params, sid, run_idx, seed, duration, perturb|null, trace_out.

Writes the phase-2 npz trace schema (COLUMNS below) so
harvest.py converts it unchanged.
"""
import importlib.util
import json
import math
import os
import queue
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.expanduser(os.environ.get("TF_REPO", "~/transfuser_repo"))
CARLA_ROOT = os.path.expanduser(os.environ.get("CARLA_ROOT",
                                               "~/carla_09101"))
CKPT = os.path.expanduser(os.environ.get(
    "TF_CKPT", "~/transfuser_assets/model_ckpt/transfuser"))

import carla                                     # noqa: E402  (egg on path)
from agent_host import SensorRig, load_agent     # noqa: E402
from perturb import ControlPerturb               # noqa: E402

# scenario templates (same five templates and parameter names as the
# MetaDrive adapter)
import templates as scn_mod                      # noqa: E402

DT = 0.05
TOWN = os.environ.get("TF_TOWN", "Town01")
EGO_BP = "vehicle.lincoln.mkz2017"   # SMOKE: leaderboard hero bp in 0.9.10
COLUMNS = ["step", "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_heading",
           "ego_steering", "cmd_steer", "cmd_gas", "crash", "th_x", "th_y",
           "th_vx", "th_vy", "th_heading", "engaged", "t"]
ENGAGE_SPEED = 0.5           # m/s: hero moving == scenario clock armed
ROUTE_LEN = 400.0            # m of lane fed to the agent as global plan


def start_server(port, gpu=0):
    env = dict(os.environ,
               DISPLAY="", SDL_VIDEODRIVER="offscreen",
               SDL_HINT_CUDA_DEVICE=str(gpu))
    # SMOKE: 0.9.10.1 headless — '-opengl' path; try vulkan if EGL fails
    cmd = [os.path.join(CARLA_ROOT, "CarlaUE4.sh"), "-opengl", "-nosound",
           f"-carla-rpc-port={port}", "-quality-level=Epic"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)
    return proc


def connect(port, retries=150):   # UE4 first boot compiles shaders: be patient
    for _ in range(retries):
        try:
            client = carla.Client("localhost", port)
            client.set_timeout(10.0)
            client.get_server_version()
            return client
        except RuntimeError:
            time.sleep(2.0)
    raise RuntimeError("CARLA server did not come up")


def _straight_spawns(world):
    """Spawn points with >= ROUTE_LEN of lane whose first 150 m is straight.

    The straight prefix matters twice: the choreographed templates assume a
    mostly-straight interaction lane, and TransFuser brakes to a permanent
    stop at spawns that stare into a junction/boundary wall (Town01
    spawn 0 faces the map-edge barrier 45 m out).
    """
    m = world.get_map()
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location)
        path = scn_mod.LanePath(wp, length=ROUTE_LEN + 60.0)
        if path.length < ROUTE_LEN:
            continue
        h0 = path.pose(0.0)[2]
        dev = max(abs((path.pose(float(s))[2] - h0 + math.pi)
                      % (2.0 * math.pi) - math.pi)
                  for s in range(10, 151, 10))
        if dev < 0.09:
            yield sp, wp, path


def _corridor_clear(world, wp, path, d_place, y0):
    """True if the crossing corridor (road edge .. y0 beyond it, at arc
    distance d_place) is open AND at road level. Probe-spawns the threat
    blueprint at road elevation (spawn collision-checks static meshes),
    then ticks and requires the probe to settle near road height — a probe
    beyond a bridge/embankment edge spawns in mid-air and falls to the
    terrain below, which is exactly the invisible-crosser failure.
    Replicates _ProximaScenario's opposite-lane side resolution."""
    m = world.get_map()
    lane_w = path.lane_width
    opp = +1.0
    for sign in (+1.0, -1.0):
        x, y, _ = path.pose(10.0, sign * lane_w)
        w = m.get_waypoint(carla.Location(x=x, y=y, z=0.3),
                          project_to_road=True)
        if w is not None and w.lane_id * wp.lane_id < 0:
            opp = sign
            break
    edge = -opp
    bp = world.get_blueprint_library().find(scn_mod.THREAT_BP)
    for frac in (1.0, 0.6, 0.25):
        lat = edge * (lane_w / 2.0 + y0 * frac)
        x, y, h = path.pose(d_place, lat)
        rw = m.get_waypoint(carla.Location(x=x, y=y, z=0.0),
                            project_to_road=True)
        road_z = rw.transform.location.z if rw is not None else 0.0
        probe = world.try_spawn_actor(bp, carla.Transform(
            carla.Location(x=x, y=y, z=road_z + 0.4),
            carla.Rotation(yaw=math.degrees(h) + 90.0)))
        if probe is None:
            return False
        for _ in range(20):             # 1 s: an airborne probe falls ~5 m
            world.tick()
        settled = probe.get_transform().location.z
        probe.destroy()
        # must settle AT road level; over water/embankment it falls away
        # (a crosser spawned over the river beside the
        # bridge, fell to z=-6.4 during the pre-trigger wait, and crossed
        # invisibly underwater)
        if abs(settled - road_z) > 0.8:
            return False
    return True


def pick_spawn(world, job=None):
    """First straight spawn; for CrossingTraffic, first straight spawn whose
    crossing corridor is physically open (fences/walls block the crosser —
    seen on the Town01 riverside road)."""
    cands = list(_straight_spawns(world))
    if not cands:
        raise RuntimeError("no spawn point with a long straight lane")
    if job is not None and job.get("scenario_cls") == "ProxCrossingTraffic":
        p = job.get("params") or {}
        d_place = float(p.get("d_place", 140.0))
        y0 = float(p.get("y0", 7.0))
        for y0_eff in (y0, 4.0, 2.5):
            if y0_eff > y0:
                continue
            for sp, wp, path in cands:
                if _corridor_clear(world, wp, path, d_place, y0_eff):
                    if y0_eff != y0:
                        job["params"]["y0"] = y0_eff
                        print("[scn] crossing y0 clamped %.1f -> %.1f "
                              "(no road-level corridor at full offset)"
                              % (y0, y0_eff), flush=True)
                    return sp, wp
        print("[scn] WARNING: no clear crossing corridor; using first "
              "straight spawn (crosser may be blocked)", flush=True)
    return cands[0][0], cands[0][1]


def run(job, port=2000):
    t0 = time.time()
    client = connect(port)
    world = client.load_world(TOWN)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    # no TrafficManager: threats are choreographed, the ego is agent-driven.
    # A sync-mode TM left behind wedges the server's tick barrier and the
    # next episode's apply_settings blocks forever (campaign 665156 stall).

    rng_seed = int(job.get("seed", 0))
    # CARLA-side determinism levers (physics is deterministic in sync mode;
    # OU actor noise is seeded from the job, in the scenario)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # choreographed threat scenarios: a red light stopping the ego would
    # confound the threat response (same policy as the MetaDrive adapter)
    for tl in world.get_actors().filter("traffic.traffic_light"):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)

    sp, wp = pick_spawn(world, job)
    bp = world.get_blueprint_library().find(EGO_BP)
    bp.set_attribute("role_name", "hero")
    hero = world.spawn_actor(bp, sp)

    # route straight down the lane for the agent's global plan
    path = scn_mod.LanePath(wp, length=ROUTE_LEN + 60.0)
    locs = []
    for s in np.arange(0.0, ROUTE_LEN, 20.0):
        x, y, _ = path.pose(float(s))
        locs.append(carla.Location(x=x, y=y, z=sp.location.z))
    for p in (REPO + "/leaderboard", REPO + "/scenario_runner",
              REPO + "/team_code_transfuser"):
        if p not in sys.path:
            sys.path.insert(0, p)
    world.tick()
    agent, specs = load_agent(CKPT, world, locs)
    rig = SensorRig(world, hero, specs)

    # collision sensor -> crash flag
    col_q = queue.Queue()
    col_bp = world.get_blueprint_library().find("sensor.other.collision")
    col = world.spawn_actor(col_bp, carla.Transform(), attach_to=hero)
    col.listen(col_q.put)

    # optional video: chase camera -> JPEG frames (encode on host after)
    vid_dir = os.environ.get("CH_VIDEO_DIR")
    vid_cam, vid_q = None, None
    if vid_dir:
        vid_dir = os.path.join(vid_dir, job["job_id"])
        os.makedirs(vid_dir, exist_ok=True)
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "450")
        cam_bp.set_attribute("fov", "100")
        vid_q = queue.Queue()
        vid_cam = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=-7.5, z=5.5),
                            carla.Rotation(pitch=-18.0)),
            attach_to=hero)
        vid_cam.listen(vid_q.put)
        import cv2                                # in the venv

    scn_cls = getattr(scn_mod, job["scenario_cls"])
    scenario = scn_cls(carla, world, hero, job.get("params") or {},
                       rng_seed, perturb=job.get("perturb"))
    pert = ControlPerturb(dt=DT)

    rows = []
    meta = {"scenario": job["scenario_cls"], "seed": rng_seed,
            "params": json.dumps(job.get("params") or {}),
            "town": TOWN, "agent": "transfuser_2022"}
    duration = float(job.get("duration", 40.0))
    max_steps = int((duration + 20.0) / DT)      # +20 s pre-engage grace
    engaged_step = None
    crash = 0.0
    outcome = "completed"
    spawned = False
    try:
        # pre-roll: CARLA needs ~2 s of throttle to shift out of neutral,
        # and at v=0 the policy imitates "stopped" data (inertia problem;
        # its own creep fires too late/too briefly to beat the gear lag).
        # Hand control over with the ego already rolling.
        pre = carla.VehicleControl(throttle=0.5)
        for _ in range(100):
            world.tick()
            rig.tick(hero, 0)
            hero.apply_control(pre)
            v = hero.get_velocity()
            if math.hypot(v.x, v.y) >= 1.5:
                break
        if vid_q is not None:
            while not vid_q.empty():
                vid_q.get()
        for step in range(max_steps):
            snap = world.tick()
            frame = snap if isinstance(snap, int) else snap
            data = rig.tick(hero, frame)
            ts = step * DT
            control = agent.run_step(data, ts)
            control.manual_gear_shift = False
            control = pert.apply(control, step)
            hero.apply_control(control)

            v = hero.get_velocity()
            speed = math.hypot(v.x, v.y)
            if engaged_step is None and speed > ENGAGE_SPEED:
                engaged_step = step
                meta["engaged_step"] = step
                scenario.on_spawn()
                spawned = True
                th_loc = (scenario._threat.get_transform().location
                          if scenario._threat is not None else None)
                eg_loc = hero.get_transform().location
                print("[scn] ego_s0=%.1f path_len=%.1f th_s=%.1f "
                      "th_at=(%.1f, %.1f) ego_at=(%.1f, %.1f)" % (
                          scenario.ego_s0, scenario.path.length,
                          getattr(scenario, "_th_s", float("nan")),
                          th_loc.x if th_loc else float("nan"),
                          th_loc.y if th_loc else float("nan"),
                          eg_loc.x, eg_loc.y), flush=True)
            # skip the spawn tick itself: a just-spawned actor reports
            # transform (0,0,0) until the next world.tick(), which would
            # poison the path projection and teleport the threat
            if spawned and step > engaged_step:
                t_scn = (step - engaged_step) * DT
                p_active = (job.get("perturb") is not None
                            and step >= pert.start_step)
                scenario.on_step(t_scn, DT, p_active)
            while not col_q.empty():
                ev = col_q.get()
                if "proxima_threat" in (
                        ev.other_actor.attributes.get("role_name", "")) or \
                        ev.other_actor.type_id.startswith("vehicle"):
                    crash = 1.0
            tf = hero.get_transform()
            th_x = th_y = th_vx = th_vy = th_h = float("nan")
            if spawned and step > engaged_step \
                    and scenario._threat is not None:
                st = scenario.threat_state()   # dict, see templates.py
                if st is not None:
                    th_x, th_y = st["x"], st["y"]
                    th_vx, th_vy = st["vx"], st["vy"]
                    th_h = st["heading"]
            rows.append([step, tf.location.x, tf.location.y, v.x, v.y,
                         math.radians(tf.rotation.yaw),
                         hero.get_control().steer, control.steer,
                         control.throttle - control.brake, crash,
                         th_x, th_y, th_vx, th_vy, th_h,
                         1.0 if engaged_step is not None else 0.0, ts])
            if vid_cam is not None:
                try:
                    img = vid_q.get(timeout=2.0)
                    a = np.frombuffer(img.raw_data, dtype=np.uint8)
                    a = a.reshape((img.height, img.width, 4))[:, :, :3]
                    import cv2
                    cv2.imwrite(os.path.join(vid_dir, f"f{step:05d}.jpg"),
                                a)
                except queue.Empty:
                    pass
            if crash and step > (engaged_step or 0) + 20:
                outcome = "collision"
                break
            if engaged_step is not None and \
                    (step - engaged_step) * DT >= duration:
                break
        if engaged_step is None:
            outcome = "no_engage"
    finally:
        done_info = {"outcome": outcome, "crash": bool(crash),
                     "wall_s": round(time.time() - t0, 1),
                     "perturb_applied": pert.active_ever}
        meta["done_info"] = done_info
        os.makedirs(os.path.dirname(job["trace_out"]), exist_ok=True)
        np.savez_compressed(job["trace_out"],
                            data=np.array(rows, dtype=np.float64),
                            columns=np.array(COLUMNS),
                            meta=json.dumps(meta))
        try:
            agent.destroy()
        except Exception:
            pass
        rig.destroy()
        if vid_cam is not None:
            try:
                vid_cam.stop()
                vid_cam.destroy()
            except RuntimeError:
                pass
        for a in (col, getattr(scenario, "_threat", None), hero):
            if a is not None:
                try:
                    a.destroy()
                except RuntimeError:
                    pass
        settings.synchronous_mode = False
        world.apply_settings(settings)
    return outcome


if __name__ == "__main__":
    job = json.loads(sys.argv[1])
    port = int(os.environ.get("CARLA_PORT", "2000"))
    out = {"job_id": job["job_id"], "outcome": run(job, port)}
    print(json.dumps(out), flush=True)
    ofile = os.environ.get("CH_OUTCOME_OUT")
    if ofile:
        with open(ofile, "w") as f:
            json.dump(out, f)
