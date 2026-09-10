"""Injury-risk curves iota: impact speed (m/s) -> injury-risk weight.

The occupant curve is the MAIS2+ (moderate-to-fatal) logistic of Kusano and
Gabler, fitted on NASS/CDS rear-end collisions (Ann. Adv. Automotive Med.
54:203-214, 2010) and tabulated in IEEE T-ITS 13(4):1546-1555, 2012,
Table II. We adopt beta0/beta1 unchanged and DROP their belt-use covariate
(beta2 = -0.6234, coded +1 belted / -1 unbelted), which places this curve
between the two belt states. Note also that the curve was fitted against
crash-phase occupant delta-V, while we feed it relative speed at contact:
the output is a declared severity weight, not a calibrated injury
probability. The pedestrian curve is a steeper stand-in, not a fitted model. Coefficients are configurable
constants; iota = 0 for non-contact outcomes by construction (callers multiply
by the contact indicator). A step function recovers the binary oracle
(Remark: binary verdict is the degenerate special case).
"""
from __future__ import annotations

import numpy as np

# logit P = a + b * dv_kmh
OCCUPANT_A, OCCUPANT_B = -6.068, 0.100      # Kusano-Gabler MAIS2+; 50% @ 61 km/h
OCCUPANT_BELT = -0.6234                     # their beta2, unused (see module docstring)
PEDESTRIAN_A, PEDESTRIAN_B = -6.90, 0.160   # ~50% severe near 43 km/h impact


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def occupant(dv_ms):
    """MAIS2+ injury-risk weight for an occupant, given impact speed (m/s)."""
    return _sigmoid(OCCUPANT_A + OCCUPANT_B * np.asarray(dv_ms) * 3.6)


def pedestrian(dv_ms):
    """P(severe injury) for a struck pedestrian given impact speed (m/s)."""
    return _sigmoid(PEDESTRIAN_A + PEDESTRIAN_B * np.asarray(dv_ms) * 3.6)


def step(dv_ms):
    """Step curve: any contact = 1. With the zero kernel this recovers B(e)."""
    return (np.asarray(dv_ms) > 0).astype(float)


CURVES = {"occupant": occupant, "pedestrian": pedestrian, "step": step}


def get_curve(actor_type: str):
    return CURVES[actor_type]
