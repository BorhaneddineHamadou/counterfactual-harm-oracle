"""Perturbation kernel kappa: the declared, auditable disturbance model.

The kernel is part of the measure (H_kappa is kernel-indexed). It has three
components, each grounded in a citable disturbance family:

  1. reaction-latency jitter  -- extra actuation/reaction delay, truncated
     normal centered at 0 within a human-plausible band (ISO 26262
     controllability grounding);
  2. surface-friction scale   -- multiplicative scale on achievable
     deceleration, uniform within a weather band;
  3. actor noise              -- Ornstein-Uhlenbeck acceleration noise on the
     surrounding actor's longitudinal (and, where applicable, lateral) motion.

A single scalar `alpha` rescales the *spread* of every component around its
nominal center; RQ2 evaluates alpha in {0.5, 1, 2}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Kernel:
    # latency jitter (seconds): truncated N(0, latency_sigma) in +-latency_bound
    latency_sigma: float = 0.10
    latency_bound: float = 0.25
    # friction scale: Uniform(friction_lo, friction_hi), nominal center 1.0
    friction_lo: float = 0.80
    friction_hi: float = 1.05
    # OU acceleration noise on the threat actor (m/s^2), theta = mean reversion
    ou_sigma_long: float = 0.50
    ou_sigma_lat: float = 0.12   # lateral velocity noise (m/s), drift/cut-in only
    ou_theta: float = 1.0
    alpha: float = 1.0

    def scaled(self, alpha: float) -> "Kernel":
        """Rescale the spread of every component by `alpha` (center fixed)."""
        return Kernel(
            latency_sigma=self.latency_sigma * alpha,
            latency_bound=self.latency_bound * alpha,
            friction_lo=1.0 - (1.0 - self.friction_lo) * alpha,
            friction_hi=1.0 + (self.friction_hi - 1.0) * alpha,
            ou_sigma_long=self.ou_sigma_long * alpha,
            ou_sigma_lat=self.ou_sigma_lat * alpha,
            ou_theta=self.ou_theta,
            alpha=self.alpha * alpha,
        )

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        """Draw n disturbances. OU components are realized inside the
        simulator from the replay noise stream; here we return their sigmas."""
        dlat = rng.normal(0.0, max(self.latency_sigma, 1e-12), n)
        dlat = np.clip(dlat, -self.latency_bound, self.latency_bound)
        fric = rng.uniform(self.friction_lo, self.friction_hi, n)
        return {
            "dlat": dlat,
            "fric": np.clip(fric, 0.05, None),
            "ou_sigma_long": self.ou_sigma_long,
            "ou_sigma_lat": self.ou_sigma_lat,
            "ou_theta": self.ou_theta,
        }

    @staticmethod
    def zero() -> "Kernel":
        """Degenerate kernel delta_0 (Remark 'binary special case')."""
        return Kernel(latency_sigma=0.0, latency_bound=0.0,
                      friction_lo=1.0, friction_hi=1.0,
                      ou_sigma_long=0.0, ou_sigma_lat=0.0, alpha=0.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Kernel":
        return Kernel(**d)
