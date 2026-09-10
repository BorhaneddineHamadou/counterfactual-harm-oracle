"""
ScenarioMetaDriveBridge — real openpilot bridge with scenario NPC scripting injected.

Uses unittest.mock.patch to replace `metadrive_process` in MetaDriveWorld's
module namespace with `scenario_metadrive_process`, which:
  1. Imports the scenario class by name inside the subprocess
  2. Calls scenario.on_reset(env) after env.reset()
  3. Calls scenario.on_step(env, step) before each env.step()
  4. Tracks min_ttc and appends it to done_info

This file must be run inside the openpilot Singularity environment:
  singularity exec --nv ~/openpilot_dev.sif bash -c "
    export PATH=~/.local/bin:~/openpilot/.venv/bin:$PATH
    cd ~/openpilot
    python3 ~/openpilot_sim/scenario_bridge.py --scenario highway_following
  "
"""
import math
import time
import functools
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import patch
from multiprocessing.connection import Connection

from panda3d.core import Vec3

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation
from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.component.sensors.base_camera import _cuda_enable

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_process import (
    apply_metadrive_patches, metadrive_simulation_state, metadrive_vehicle_state,
)
from openpilot.tools.sim.lib.camerad import W, H
from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.lib.common import vec3

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0, 0)

OP_SIM_ROOT = os.path.dirname(os.path.abspath(__file__))


def _min_ttc(env) -> float:
    """Compute minimum TTC between ego and any NPC vehicle in the scene."""
    try:
        ego = env.vehicle
        ep = np.array(ego.position[:2])
        ev = np.array(ego.velocity[:2])
        best = float("inf")
        for npc in env.engine.traffic_manager.vehicles:
            if npc is ego:
                continue
            np_ = np.array(npc.position[:2])
            nv = np.array(npc.velocity[:2])
            rel_p = np_ - ep
            dist = np.linalg.norm(rel_p)
            if dist < 0.01:
                return 0.0
            closing = np.dot(ev - nv, rel_p) / dist
            if closing > 0.1:
                best = min(best, dist / closing)
        return best
    except Exception:
        return float("inf")


def scenario_metadrive_process(
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
    """
    Mirror of openpilot's metadrive_process with scenario NPC hooks.
    scenario_cls_name is resolved to a SceRTScenario subclass inside this subprocess.
    """
    # ── Load scenario ──────────────────────────────────────────────────────
    if OP_SIM_ROOT not in sys.path:
        sys.path.insert(0, OP_SIM_ROOT)
    from scenarios import ALL_SCENARIOS
    scenario_cls = next((c for c in ALL_SCENARIOS if c.__name__ == scenario_cls_name), None)
    scenario = scenario_cls() if scenario_cls else None
    if scenario is None:
        print(f"[ERROR] Unknown scenario class: {scenario_cls_name}")
        return

    # ── Standard metadrive_process setup ──────────────────────────────────
    arrive_dest_done = config.pop("arrive_dest_done", True)
    apply_metadrive_patches(arrive_dest_done)

    road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
    if dual_camera:
        assert wide_camera_array is not None
        wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    env = MetaDriveEnv(config)

    def get_current_lane_info(vehicle):
        _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
        lane_idx = lane_info[2] if lane_info is not None else None
        return lane_idx, on_lane

    global_min_ttc = [float("inf")]  # mutable for closure

    def reset():
        env.reset()
        env.vehicle.config["max_speed_km_h"] = 1000
        # ── Scenario hook: spawn NPCs ──────────────────────────────────────
        if scenario is not None:
            try:
                scenario.on_reset(env)
            except Exception as e:
                print(f"[WARN] scenario.on_reset failed: {e}")
        lane_idx_prev, _ = get_current_lane_info(env.vehicle)
        simulation_state_send.send(metadrive_simulation_state(running=True, done=False, done_info=None))
        return lane_idx_prev

    lane_idx_prev = reset()
    start_time = None
    env_step = 0

    def get_cam_as_rgb(cam):
        cam = env.engine.sensors[cam]
        cam.get_cam().reparentTo(env.vehicle.origin)
        cam.get_cam().setPos(C3_POSITION)
        cam.get_cam().setHpr(C3_HPR)
        img = cam.perceive(to_float=False)
        if not isinstance(img, np.ndarray):
            img = img.get()
        return img

    rk = Ratekeeper(100, None)
    steer_ratio = 8
    vc = [0, 0]

    while not exit_event.is_set():
        # Send vehicle state
        vehicle_state_send.send(metadrive_vehicle_state(
            velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
            position=env.vehicle.position,
            bearing=float(math.degrees(env.vehicle.heading_theta)),
            steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING,
        ))

        # Receive controls
        if controls_recv.poll(0):
            while controls_recv.poll(0):
                steer_angle, gas, should_reset = controls_recv.recv()
            steer_md = np.clip(steer_angle / (env.vehicle.MAX_STEERING * steer_ratio), -1, 1)
            vc = [steer_md, gas]
            if should_reset:
                env_step = 0
                lane_idx_prev = reset()
                start_time = None

        is_engaged = op_engaged.is_set()
        if is_engaged and start_time is None:
            start_time = time.monotonic()

        if rk.frame % 5 == 0:
            # ── Scenario hook: timed NPC events ───────────────────────────
            if scenario is not None:
                try:
                    scenario.on_step(env, env_step)
                except Exception as e:
                    print(f"[WARN] scenario.on_step failed at {env_step}: {e}")

            _, _, terminated, _, _ = env.step(vc)
            env_step += 1

            # Track min TTC
            ttc = _min_ttc(env)
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

                # Attach min_ttc and scenario-specific metrics to done_info
                info = dict(done_result[1]) if done_result[1] else {}
                if scenario is not None and hasattr(scenario, 'get_extra_info'):
                    try:
                        info.update(scenario.get_extra_info(env))
                    except Exception:
                        pass
                min_ttc_val = round(global_min_ttc[0], 2) if global_min_ttc[0] < 999 else None
                info["min_ttc"] = min_ttc_val
                simulation_state_send.send(
                    metadrive_simulation_state(running=False, done=done_result[0], done_info=info)
                )

            if dual_camera:
                wide_road_image[...] = get_cam_as_rgb("rgb_wide")
            road_image[...] = get_cam_as_rgb("rgb_road")
            image_lock.release()

        rk.keep_time()


class ScenarioMetaDriveBridge(MetaDriveBridge):
    """
    MetaDriveBridge that injects scenario NPC scripting into the MetaDrive subprocess.
    Uses mock.patch to swap metadrive_process → scenario_metadrive_process.
    """

    def __init__(self, scenario_cls_name: str, dual_camera=False, high_quality=False,
                 test_duration=30, test_run=True):
        super().__init__(dual_camera, high_quality)
        self.scenario_cls_name = scenario_cls_name
        self._test_duration = test_duration
        self._test_run = test_run

        # Load scenario to get map_config and env_config
        if OP_SIM_ROOT not in sys.path:
            sys.path.insert(0, OP_SIM_ROOT)
        from scenarios import ALL_SCENARIOS
        cls = next((c for c in ALL_SCENARIOS if c.__name__ == scenario_cls_name), None)
        if cls is None:
            raise ValueError(f"Unknown scenario: {scenario_cls_name}")
        self._scenario_instance = cls()

    def spawn_world(self, queue):
        sensors = {"rgb_road": (RGBCameraRoad, W, H)}
        if self.dual_camera:
            sensors["rgb_wide"] = (RGBCameraWide, W, H)

        config = dict(
            use_render=self.should_render,
            vehicle_config=dict(enable_reverse=False, render_vehicle=False, image_source="rgb_road"),
            sensors=sensors,
            image_on_cuda=_cuda_enable,
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
            num_scenarios=10,
        )
        config.update(self._scenario_instance.env_config())

        # Patch MetaDriveWorld's module-level reference to metadrive_process
        patched = functools.partial(scenario_metadrive_process, self.scenario_cls_name)
        with patch("openpilot.tools.sim.bridge.metadrive.metadrive_world.metadrive_process", patched):
            world = MetaDriveWorld(queue, config, self._test_duration, self._test_run, self.dual_camera)

        return world
