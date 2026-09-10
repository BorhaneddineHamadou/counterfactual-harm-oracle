"""ProximaMetaDriveBridge — non-invasive Proxima variant of the campaign
bridge. Imports the live `scenario_bridge` untouched (same pattern as
`scenario_bridge_inject.py`) and redefines only the MetaDrive subprocess to
add, relative to the campaign bridge:

  1. REAL SEEDING       — env.reset(seed=SEED) with start_seed pinned in the
                          config (the campaign bridge never seeds reset()).
  2. PER-FRAME TELEMETRY — ego + threat state, commands, crash flags and
                          instantaneous Delta-v recorded every env step and
                          written as .npz to PROXIMA_TRACE_OUT at episode end.
  3. EGO PERTURBATIONS  — gated by start_step (env steps, 20 Hz), from
                          PROXIMA_PERTURB (JSON):
        dlat_s     : extra actuation latency (ring buffer at 100 Hz)
        brake_gain : scale on the braking part of the command -- the
                     friction/achievable-decel analogue (the commaai
                     MetaDrive fork stubs true wheel friction)
        fade_frac      : RQ3 severity arm -- braking effectiveness drops by
                         this fraction once brake demand has stayed >=
                         fade_demand for fade_sustain_s (brake fade under
                         sustained peak demand)
        fade_demand    : demand threshold as |gas| in [0,1] (default 0.5)
        fade_sustain_s : sustain time before the fade engages (default 1.0)
        comp_boost     : compensation -- gain (>=1) on LIGHT braking
                         (|gas| < fade_demand): the earlier, more cautious
                         brake onset that is bisection-tuned until the
                         collision rate matches baseline
     Actor-side OU noise is applied inside the Proxima scenario classes
     (same PROXIMA_PERTURB blob, start-gated there too).

Environment contract (all set by the worker before the bridge starts):
  PROXIMA_SCENARIO_PARAMS  JSON of concrete scenario parameters
  PROXIMA_SEED             int env seed
  PROXIMA_PERTURB          JSON or unset (nominal run)
  PROXIMA_TRACE_OUT        path for the .npz trace
"""
import json
import math
import os
import sys
import time
import functools
from collections import deque
from unittest.mock import patch
from multiprocessing.connection import Connection

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "openpilot_sim"))
sys.path.insert(0, HERE)

import scenario_bridge as _b  # noqa: E402  (live campaign bridge, untouched)
from brake_shaping import shape_brake  # noqa: E402


def _perturb_cfg():
    raw = os.environ.get("PROXIMA_PERTURB", "")
    return json.loads(raw) if raw else {}


def proxima_metadrive_process(
    scenario_cls_name: str,
    dual_camera: bool,
    config: dict,
    camera_array,
    wide_camera_array,
    image_lock,
    controls_recv: Connection,
    simulation_state_send: Connection,
    vehicle_state_send: Connection,
    exit_event,
    op_engaged,
    test_duration: float,
    test_run: bool,
):
    from templates import PROXIMA_SCENARIOS
    scenario_cls = next(
        (c for c in PROXIMA_SCENARIOS if c.__name__ == scenario_cls_name), None)
    scenario = scenario_cls() if scenario_cls else None
    if scenario is None:
        print(f"[ERROR] Unknown Proxima scenario: {scenario_cls_name}")
        return

    seed = int(os.environ.get("PROXIMA_SEED", "0"))
    trace_out = os.environ.get("PROXIMA_TRACE_OUT", "")
    video_out = os.environ.get("PROXIMA_VIDEO_OUT", "")
    pert = _perturb_cfg()
    start_step = int(pert.get("start_step", 0))
    dlat_s = float(pert.get("dlat_s", 0.0))
    brake_gain = float(pert.get("brake_gain", 1.0))
    fade_frac = float(pert.get("fade_frac", 0.0))
    fade_demand = float(pert.get("fade_demand", 0.5))
    fade_sustain = max(1, int(round(float(pert.get("fade_sustain_s", 1.0))
                                    * 100)))       # frames at 100 Hz
    comp_boost = float(pert.get("comp_boost", 1.0))
    # fade defect may start earlier than the kernel keys (version defect
    # active whole-run, kernel perturbations branch-gated at t*); defaults
    # to start_step so every pre-existing job file behaves identically
    fade_start_step = int(pert.get("fade_start_step", start_step))
    k_lat = max(0, int(round(dlat_s * 100)))       # rk runs at 100 Hz
    start_frame = start_step * 5
    fade_start_frame = fade_start_step * 5

    arrive_dest_done = config.pop("arrive_dest_done", True)
    _b.apply_metadrive_patches(arrive_dest_done)

    H, W = _b.H, _b.W
    road_image = np.frombuffer(camera_array.get_obj(),
                               dtype=np.uint8).reshape((H, W, 3))
    if dual_camera:
        wide_road_image = np.frombuffer(wide_camera_array.get_obj(),
                                        dtype=np.uint8).reshape((H, W, 3))

    # optional per-run video capture. The sim loop must NOT stall (openpilot
    # runs in real time, so any tick lag acts as extra reaction latency and
    # perturbs the run): frames are 2x-downscaled cheaply in numpy and
    # handed to a background writer thread; if the writer falls behind,
    # frames are DROPPED, never awaited. mkv container so a killed process
    # still leaves a playable file.
    ffproc = None
    video_frames = []
    video_dropped = [0]
    vq = None
    if video_out:
        import queue as _q
        import subprocess as _sp
        import threading as _th
        os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)
        # absolute path: the openpilot venv ships a crippled ffmpeg that
        # shadows the system one on PATH and lacks the pipe protocol
        ffproc = _sp.Popen(
            ["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W//2}x{H//2}", "-r", "20",
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "23", video_out],
            stdin=_sp.PIPE)
        vq = _q.Queue(maxsize=64)

        def _vwriter():
            while True:
                buf = vq.get()
                if buf is None:
                    break
                try:
                    ffproc.stdin.write(buf)
                except Exception:
                    break

        _vthread = _th.Thread(target=_vwriter, daemon=True)
        _vthread.start()

    env = _b.MetaDriveEnv(config)

    def get_current_lane_info(vehicle):
        _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
        lane_idx = lane_info[2] if lane_info is not None else None
        return lane_idx, on_lane

    global_min_ttc = [float("inf")]
    rows = []                    # per-env-step telemetry
    meta = {"scenario": scenario_cls_name, "seed": seed,
            "params": os.environ.get("PROXIMA_SCENARIO_PARAMS", ""),
            "perturb": os.environ.get("PROXIMA_PERTURB", ""),
            "engaged_step": -1}

    def reset():
        env.reset(seed=seed)
        env.vehicle.config["max_speed_km_h"] = 1000
        if scenario is not None:
            try:
                scenario.on_reset(env)
            except Exception as e:
                print(f"[WARN] scenario.on_reset failed: {e}")
        lane_idx_prev, _ = get_current_lane_info(env.vehicle)
        simulation_state_send.send(_b.metadrive_simulation_state(
            running=True, done=False, done_info=None))
        return lane_idx_prev

    lane_idx_prev = reset()
    start_time = None
    env_step = 0

    def get_cam_as_rgb(cam):
        cam = env.engine.sensors[cam]
        cam.get_cam().reparentTo(env.vehicle.origin)
        cam.get_cam().setPos(_b.C3_POSITION)
        cam.get_cam().setHpr(_b.C3_HPR)
        img = cam.perceive(to_float=False)
        if not isinstance(img, np.ndarray):
            img = img.get()
        return img

    def record_row(vc_now):
        ego = env.vehicle
        crash = bool(getattr(ego, "crash_vehicle", False)
                     or getattr(ego, "crash_human", False)
                     or getattr(ego, "crash_object", False))
        ts = scenario.threat_state(env) if scenario is not None else None
        if ts is None:
            ts = {"x": np.nan, "y": np.nan, "vx": np.nan, "vy": np.nan,
                  "heading": np.nan}
        rows.append((
            env_step,
            float(ego.position[0]), float(ego.position[1]),
            float(ego.velocity[0]), float(ego.velocity[1]),
            float(ego.heading_theta), float(ego.steering),
            float(vc_now[0]), float(vc_now[1]),
            float(crash),
            ts["x"], ts["y"], ts["vx"], ts["vy"], ts["heading"],
            1.0 if op_engaged.is_set() else 0.0,
        ))

    def flush_trace(done_info):
        if not trace_out:
            return
        try:
            arr = np.asarray(rows, dtype=np.float64)
            np.savez_compressed(
                trace_out, data=arr,
                columns=np.array([
                    "step", "ego_x", "ego_y", "ego_vx", "ego_vy",
                    "ego_heading", "ego_steering", "cmd_steer", "cmd_gas",
                    "crash", "th_x", "th_y", "th_vx", "th_vy", "th_heading",
                    "engaged"]),
                meta=json.dumps({**meta,
                                 "done_info": {k: v for k, v in
                                               (done_info or {}).items()
                                               if isinstance(v, (int, float,
                                                                 str, bool))
                                               or v is None}}))
        except Exception as e:
            print(f"[WARN] trace flush failed: {e}")

    rk = _b.Ratekeeper(100, None)
    steer_ratio = 8
    vc = [0, 0]
    cmd_ring = deque(maxlen=600)   # raw (steer_angle, gas) at 100 Hz
    raw_cmd = (0.0, 0.0)
    peak_frames = 0                # consecutive frames at >= fade_demand
    flushed = False

    while not exit_event.is_set():
        vehicle_state_send.send(_b.metadrive_vehicle_state(
            velocity=_b.vec3(x=float(env.vehicle.velocity[0]),
                             y=float(env.vehicle.velocity[1]), z=0),
            position=env.vehicle.position,
            bearing=float(math.degrees(env.vehicle.heading_theta)),
            steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING,
        ))

        if controls_recv.poll(0):
            while controls_recv.poll(0):
                steer_angle, gas, should_reset = controls_recv.recv()
            raw_cmd = (steer_angle, gas)
            if should_reset:
                env_step = 0
                rows.clear()
                peak_frames = 0
                lane_idx_prev = reset()
                start_time = None

        # ── Proxima ego perturbations (gated by start_frame) ───────────────
        cmd_ring.append(raw_cmd)
        eff_steer, eff_gas = raw_cmd
        if rk.frame >= start_frame:
            if k_lat > 0 and len(cmd_ring) > k_lat:
                eff_steer, eff_gas = cmd_ring[-1 - k_lat]
            if brake_gain != 1.0 and eff_gas < 0:
                eff_gas = eff_gas * brake_gain
        if rk.frame >= fade_start_frame:
            eff_gas, peak_frames = shape_brake(
                eff_gas, peak_frames, fade_frac, fade_demand,
                fade_sustain, comp_boost)
        steer_md = np.clip(
            eff_steer / (env.vehicle.MAX_STEERING * steer_ratio), -1, 1)
        vc = [steer_md, eff_gas]

        is_engaged = op_engaged.is_set()
        if is_engaged and start_time is None:
            start_time = time.monotonic()
            meta["engaged_step"] = env_step

        if rk.frame % 5 == 0:
            if scenario is not None:
                try:
                    scenario.on_step(env, env_step)
                except Exception as e:
                    print(f"[WARN] scenario.on_step failed at {env_step}: {e}")

            _, _, terminated, _, _ = env.step(vc)
            env_step += 1
            record_row(vc)

            ttc = _b._min_ttc(env)
            if ttc < global_min_ttc[0]:
                global_min_ttc[0] = ttc

            timeout = (start_time is not None and
                       time.monotonic() - start_time >= test_duration)
            lane_idx_curr, on_lane = get_current_lane_info(env.vehicle)
            out_of_lane = (lane_idx_curr != lane_idx_prev) or not on_lane
            lane_idx_prev = lane_idx_curr

            if terminated or ((out_of_lane or timeout) and test_run):
                if terminated:
                    done_result = env.done_function("default_agent")
                elif out_of_lane:
                    done_result = (True, {"out_of_lane": True})
                else:
                    done_result = (True, {"timeout": True})
                info = dict(done_result[1]) if done_result[1] else {}
                min_ttc_val = (round(global_min_ttc[0], 2)
                               if global_min_ttc[0] < 999 else None)
                info["min_ttc"] = min_ttc_val
                info["proxima_seed"] = seed
                flush_trace(info)
                flushed = True
                simulation_state_send.send(_b.metadrive_simulation_state(
                    running=False, done=done_result[0], done_info=info))

            if dual_camera:
                wide_road_image[...] = get_cam_as_rgb("rgb_wide")
            road_image[...] = get_cam_as_rgb("rgb_road")
            image_lock.release()
            if vq is not None:
                try:
                    vq.put_nowait(road_image[::2, ::2].tobytes())
                    video_frames.append(env_step)
                except Exception:
                    video_dropped[0] += 1

        rk.keep_time()

    # exit_event can be set from outside the done path — openpilot's
    # metadrive_world watchdog kills the run ~29s after the ego stops
    # (e.g. post-crash with crash_*_done disabled), and without this the
    # trace of exactly those crash runs is lost (20/23 nominal collisions)
    if not flushed and rows:
        flush_trace({"aborted": True})
    if ffproc is not None:
        try:
            vq.put(None, timeout=10)
            _vthread.join(timeout=30)
            ffproc.stdin.close()
            ffproc.wait(timeout=60)
        except Exception:
            pass
        if video_dropped[0]:
            print(f"[video] dropped {video_dropped[0]} frames")
        # frame index -> env_step map (env_step resets at engagement)
        with open(video_out + ".frames.json", "w") as f:
            json.dump(video_frames, f)


class ProximaMetaDriveBridge(_b.ScenarioMetaDriveBridge):
    """Campaign bridge with the Proxima subprocess and real seeding."""

    def __init__(self, scenario_cls_name: str, dual_camera=False,
                 high_quality=False, test_duration=40, test_run=True):
        # bypass parent __init__'s ALL_SCENARIOS lookup: resolve from the
        # Proxima registry instead, then replicate the rest
        _b.MetaDriveBridge.__init__(self, dual_camera, high_quality)
        self.scenario_cls_name = scenario_cls_name
        self._test_duration = test_duration
        self._test_run = test_run
        from templates import PROXIMA_SCENARIOS
        cls = next((c for c in PROXIMA_SCENARIOS
                    if c.__name__ == scenario_cls_name), None)
        if cls is None:
            raise ValueError(f"Unknown Proxima scenario: {scenario_cls_name}")
        self._scenario_instance = cls()

    def spawn_world(self, queue):
        seed = int(os.environ.get("PROXIMA_SEED", "0"))
        sensors = {"rgb_road": (_b.RGBCameraRoad, _b.W, _b.H)}
        if self.dual_camera:
            sensors["rgb_wide"] = (_b.RGBCameraWide, _b.W, _b.H)
        config = dict(
            use_render=self.should_render,
            vehicle_config=dict(enable_reverse=False, render_vehicle=False,
                                image_source="rgb_road"),
            sensors=sensors,
            image_on_cuda=_b._cuda_enable,
            image_observation=True,
            interface_panel=[],
            out_of_route_done=False,
            on_continuous_line_done=False,
            crash_vehicle_done=False,
            crash_object_done=False,
            arrive_dest_done=False,
            traffic_density=0.0,
            map_config=self._scenario_instance.map_config(),
            decision_repeat=1,
            physics_world_step_size=self.TICKS_PER_FRAME / 100,
            preload_models=False,
            show_logo=False,
            anisotropic_filtering=False,
            num_scenarios=1,
            start_seed=seed,
        )
        config.update(self._scenario_instance.env_config())
        patched = functools.partial(proxima_metadrive_process,
                                    self.scenario_cls_name)
        with patch("openpilot.tools.sim.bridge.metadrive."
                   "metadrive_world.metadrive_process", patched):
            world = _b.MetaDriveWorld(queue, config, self._test_duration,
                                      self._test_run, self.dual_camera)
        return world
