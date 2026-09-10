"""Kernel ego-axes at the TransFuser control interface.

Sync mode makes this exact: perturbations activate at a step index, no
gate files. Semantics mirror the MetaDrive bridge:
  dlat_s     : reaction latency — the agent's BRAKE command reaches the
               vehicle dlat_s seconds late (ring buffer at 20 Hz); while
               a delayed brake is pending, the previous throttle is held.
               dlat_s < 0 is not realizable here (as in the aw_patch,
               negative draws clamp to 0 — noted in the kernel spec).
  brake_gain : multiplicative scale on the brake command (friction
               analogue at the actuation interface), clamped to [0, 1].
"""
import json
import os
from collections import deque


class ControlPerturb:
    def __init__(self, dt=0.05):
        self.dt = dt
        blob = os.environ.get("CH_PERTURB") or os.environ.get(
            "PROXIMA_PERTURB")
        cfg = json.loads(blob) if blob else {}
        self.dlat_s = max(float(cfg.get("dlat_s", 0.0)), 0.0)
        self.gain = float(cfg.get("brake_gain", 1.0))
        self.start_step = int(cfg.get("start_step", 0))
        self.shift = int(round(self.dlat_s / dt))
        self.buf = deque([0.0] * self.shift, maxlen=max(self.shift, 1))
        self.last_throttle = 0.0
        self.active_ever = False

    @property
    def enabled(self):
        return self.shift > 0 or self.gain != 1.0

    def apply(self, control, step):
        """Mutates and returns the carla.VehicleControl for this step."""
        if not self.enabled or step < self.start_step:
            self.last_throttle = control.throttle
            if self.shift:
                self.buf.append(control.brake)
            return control
        self.active_ever = True
        if self.shift:
            self.buf.append(control.brake)
            delayed = self.buf[0]
            if control.brake > 0.0 and delayed == 0.0:
                # brake requested but not yet arrived: hold prior throttle
                control.throttle = self.last_throttle
            control.brake = delayed
        control.brake = min(max(control.brake * self.gain, 0.0), 1.0)
        if control.brake > 0.0:
            control.throttle = 0.0
        else:
            self.last_throttle = control.throttle
        return control
