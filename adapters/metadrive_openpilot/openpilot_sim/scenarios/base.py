"""
Base scenario class for SceRT openpilot+MetaDrive integration.
Each scenario subclass defines map geometry, NPC scripting, and metrics.
"""
import math
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.constants import MetaDriveType


# Valid block params from pg_space.py:
#   S/B: length
#   C:   length, radius, angle, dir
#   X:   radius, change_lane_num, decrease_increase
#   T:   radius, t_intersection_type, change_lane_num, decrease_increase
#   O:   radius_exit, radius_inner, angle
#   r/R: length
#   f/F: length, lane_num (fork only)

def straight(length=60):
    return {"id": "S", "pre_block_socket_index": 0, "length": length}

def curve(length=60, angle=90, direction=0):
    return {"id": "C", "pre_block_socket_index": 0, "length": length,
            "radius": length, "angle": angle, "dir": direction}

def intersection():
    return {"id": "X", "pre_block_socket_index": 0}

def t_intersection():
    return {"id": "T", "pre_block_socket_index": 0}

def roundabout():
    return {"id": "O", "pre_block_socket_index": 0}

def in_ramp():
    return {"id": "r", "pre_block_socket_index": 0}

def bidirection(length=80):
    return {"id": "B", "pre_block_socket_index": 0, "length": length}


class SceRTScenario:
    """
    Wraps MetaDriveEnv with scenario-specific NPC scripting and metric tracking.
    Subclasses override map_config(), on_reset(), on_step().
    """
    NAME = "base"
    DURATION_STEPS = 400   # max env.step() calls (~20s at 20Hz)

    # --- subclass interface ---

    def map_config(self) -> dict:
        raise NotImplementedError

    def env_config(self) -> dict:
        """Extra MetaDriveEnv config overrides."""
        return {}

    def on_reset(self, env):
        """Called after env.reset(). Spawn NPCs, traffic lights, etc."""
        pass

    def on_step(self, env, step: int):
        """Called each step. Trigger timed events (e.g. emergency brake at step 150)."""
        pass

    # --- internal machinery ---

    def _base_config(self):
        cfg = dict(
            use_render=False,
            image_observation=False,
            map_config=self.map_config(),
            num_scenarios=10,       # allow seeds 0-9
            traffic_density=0.0,    # we script NPCs manually
            out_of_route_done=False,
            on_continuous_line_done=False,
            crash_vehicle_done=True,
            crash_object_done=True,
            crash_human_done=True,
            decision_repeat=1,
            preload_models=False,
            show_logo=False,
        )
        cfg.update(self.env_config())
        return cfg

    def run(self, ego_policy=None, seed=0):
        """
        Run the scenario.  ego_policy(step, env) -> [steer, gas] or None (default straight 0.4).
        Returns dict: outcome, min_ttc, duration, collision_step.
        """
        env = MetaDriveEnv(self._base_config())
        try:
            env.reset(seed=seed)
            self.on_reset(env)
            return self._episode_loop(env, ego_policy, seed)
        finally:
            env.close()

    def _episode_loop(self, env, ego_policy, seed):
        min_ttc = float("inf")
        collision = False
        out_of_lane = False
        step = 0
        t0 = None

        import time
        t0 = time.monotonic()

        for step in range(self.DURATION_STEPS):
            # --- ego action ---
            if ego_policy is not None:
                action = ego_policy(step, env)
            else:
                action = [0.0, 0.4]   # go straight

            # --- scenario hook ---
            self.on_step(env, step)

            # --- physics step ---
            _, reward, terminated, truncated, info = env.step(action)

            # --- TTC ---
            ttc = self._compute_min_ttc(env)
            if ttc < min_ttc:
                min_ttc = ttc

            # --- termination ---
            if info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_human"):
                collision = True
                break
            if info.get("out_of_road"):
                out_of_lane = True
                break
            if terminated or truncated:
                break

        duration = time.monotonic() - t0

        outcome = "completed"
        if collision:
            outcome = "collision"
        elif out_of_lane:
            outcome = "out_of_lane"

        return {
            "scenario": self.NAME,
            "seed": seed,
            "outcome": outcome,
            "min_ttc": round(min_ttc, 2) if min_ttc < 999 else None,
            "duration_s": round(duration, 1),
            "steps": step + 1,
        }

    def _compute_min_ttc(self, env) -> float:
        """Minimum TTC between ego and any NPC vehicle."""
        try:
            ego = env.vehicle
            ego_pos = np.array(ego.position[:2])
            ego_vel = np.array(ego.velocity[:2])

            min_ttc = float("inf")
            for npc in env.engine.traffic_manager.vehicles:
                if npc is ego:
                    continue
                npc_pos = np.array(npc.position[:2])
                npc_vel = np.array(npc.velocity[:2])

                rel_pos = npc_pos - ego_pos
                dist = np.linalg.norm(rel_pos)
                if dist < 1e-2:
                    return 0.0

                rel_vel = ego_vel - npc_vel
                closing = np.dot(rel_vel, rel_pos) / dist
                if closing > 0.1:
                    ttc = dist / closing
                    if ttc < min_ttc:
                        min_ttc = ttc
            return min_ttc
        except Exception:
            return float("inf")

    def _get_lane(self, env, road_id, lane_idx=0):
        """Helper: get a lane object from the road network."""
        try:
            net = env.current_map.road_network
            return net.graph[road_id[0]][road_id[1]][lane_idx]
        except Exception:
            return None

    def _vehicle_config(self, env):
        """Return a minimal valid vehicle_config based on the env's global config."""
        try:
            return dict(env.engine.global_config["vehicle_config"])
        except Exception:
            return {}

    def _spawn_npc_on_lane(self, env, lane, longitude=10.0, seed=0):
        """
        Spawn a DefaultVehicle on a specific lane at a given longitudinal position.
        This is the reliable way to spawn NPCs — avoids NavigationError.
        Returns (vehicle, policy_handle) where policy_handle is None (caller adds policy).
        """
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle
        vc = self._vehicle_config(env)
        vc["spawn_lane_index"] = lane.index
        vc["spawn_longitude"] = min(longitude, lane.length * 0.9)
        v = env.engine.spawn_object(DefaultVehicle, vehicle_config=vc, random_seed=seed)
        return v

    def _get_lanes_by_road(self, env):
        """Return dict {road_key: [lane, ...]} for all roads in the map."""
        lanes_by_road = {}
        try:
            net = env.current_map.road_network
            for s, ends in net.graph.items():
                for e, lanes in ends.items():
                    lanes_by_road[(s, e)] = lanes
        except Exception:
            pass
        return lanes_by_road

    def _get_any_lane(self, env):
        """Return the first non-ego lane available."""
        try:
            net = env.current_map.road_network
            for s, ends in net.graph.items():
                for e, lanes in ends.items():
                    for lane in lanes:
                        return lane
        except Exception:
            return None
