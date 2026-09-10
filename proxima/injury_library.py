"""Published occupant injury-risk curves, for the injury-curve robustness study.

The shipped curve (proxima/injury.py) is one published logistic. The worry a
reviewer raises is not that it is wrong but that it is *ours to choose*: a
declared parameter we could have tuned until the results came out. The answer
is to re-score the recorded impact speeds under curves nobody involved in this
paper fitted, and show the ordering does not care.

Every entry below is either (a) a logistic whose intercept and slope are quoted
from a peer-reviewed paper or a government report, or (b) an explicitly
synthetic shape, labelled as such, kept because earlier drafts reported it.

Sources
-------
kusano   Kusano and Gabler, MAIS2+ occupant logistic fitted on NASS/CDS
         rear-end collisions (Ann. Adv. Automotive Med. 54:203-214, 2010),
         tabulated in IEEE T-ITS 13(4):1546-1555, 2012, Table II.
         logit p = -6.068 + 0.100 * v[km/h], with belt covariate
         beta2 = -0.6234 coded +1 belted / -1 unbelted. The shipped curve
         drops the covariate, placing it between the two belt states.
nhtsa    Wang, J.-S. (2022, May). "MAIS(05/08) injury probability curves as
         functions of delta V" (Report No. DOT HS 813 219), NHTSA. Logistic
         regression on 2010-2015 NASS-CDS via PROC SURVEYLOGISTIC, delta V in
         MPH. Coefficients quoted verbatim from Tables A-3 (all crashes),
         A-6 (frontal) and A-9 (rear-end).

Speed convention
----------------
Proxima records the relative speed s at contact. Occupant delta-V is not s: for
a fully plastic impact between equal masses it is s/2. Both conventions are
offered (`half=True` applies s/2) so the study reports the ordering under each
rather than asserting one.
"""
from __future__ import annotations

import numpy as np

MS_TO_KMH = 3.6
MS_TO_MPH = 2.23693629

# --- Kusano and Gabler, IEEE T-ITS 2012 Table II (delta V in km/h) ----------
KUSANO = {"a": -6.068, "b": 0.100, "belt": -0.6234}

# --- NHTSA DOT HS 813 219, Tables A-3 / A-6 / A-9 (delta V in MPH) ----------
# (intercept, slope) per crash mode and MAIS level, quoted verbatim.
NHTSA = {
    "all": {"MAIS1+": (-1.3925, 0.0815), "MAIS2+": (-5.1331, 0.1479),
            "MAIS3+": (-6.9540, 0.1637), "MAIS4+": (-8.2070, 0.1564),
            "MAIS5+": (-8.7927, 0.1598), "fatality": (-8.9819, 0.1603)},
    "frontal": {"MAIS1+": (-1.4930, 0.0854), "MAIS2+": (-4.9429, 0.1425),
                "MAIS3+": (-6.9774, 0.1620), "MAIS4+": (-8.4254, 0.1586),
                "MAIS5+": (-8.8355, 0.1566), "fatality": (-9.0422, 0.1571)},
    "rear_end": {"MAIS1+": (-1.8199, 0.0671), "MAIS2+": (-6.1818, 0.1482),
                 "MAIS3+": (-8.0329, 0.1793), "MAIS4+": (-11.8787, 0.2210),
                 "MAIS5+": (-12.1944, 0.2276), "fatality": (-12.1982, 0.2255)},
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _speed(s_ms, half):
    s = np.asarray(s_ms, dtype=float)
    return 0.5 * s if half else s


def kusano_mais2(s_ms, belted=None, half=False):
    """Shipped curve; belted=True/False restores the published belt covariate."""
    v = _speed(s_ms, half) * MS_TO_KMH
    z = KUSANO["a"] + KUSANO["b"] * v
    if belted is not None:
        z = z + KUSANO["belt"] * (1.0 if belted else -1.0)
    return _sigmoid(z)


def nhtsa(s_ms, mode="all", level="MAIS3+", half=False):
    """NHTSA DOT HS 813 219 logistic for one crash mode and injury level."""
    a, b = NHTSA[mode][level]
    return _sigmoid(a + b * _speed(s_ms, half) * MS_TO_MPH)


# --- synthetic shapes, kept from the earlier draft and labelled as such -----
def step(s_ms):
    """Any contact counts 1: the binary verdict as a degenerate injury curve."""
    return (np.asarray(s_ms, dtype=float) > 0).astype(float)


def ramp(s_ms, sat_kmh):
    return np.minimum(np.asarray(s_ms, dtype=float) * MS_TO_KMH / sat_kmh, 1.0)


def quadratic(s_ms, ref_kmh=100.0):
    return np.minimum((np.asarray(s_ms, dtype=float) * MS_TO_KMH / ref_kmh) ** 2,
                      1.0)


def joksch(s_ms, ref_kmh=115.0):
    """Joksch's fourth-power fatality rule, normalised at ref_kmh."""
    return np.minimum((np.asarray(s_ms, dtype=float) * MS_TO_KMH / ref_kmh) ** 4,
                      1.0)


def pedestrian(s_ms):
    """Steeper stand-in used for struck pedestrians; not a fitted model."""
    return _sigmoid(-6.90 + 0.160 * np.asarray(s_ms, dtype=float) * MS_TO_KMH)


def library(include_half=True):
    """The curve family used by the robustness study.

    Returns an ordered dict name -> (callable, published?, source).
    """
    lib = {}

    def add(name, fn, published, source):
        lib[name] = (fn, published, source)

    add("Kusano-Gabler MAIS2+ (shipped)",
        lambda s: kusano_mais2(s), True, "kusano2012")
    add("Kusano-Gabler MAIS2+, belted",
        lambda s: kusano_mais2(s, belted=True), True, "kusano2012")
    add("Kusano-Gabler MAIS2+, unbelted",
        lambda s: kusano_mais2(s, belted=False), True, "kusano2012")
    for mode in ("all", "rear_end", "frontal"):
        for lvl in ("MAIS2+", "MAIS3+", "fatality"):
            add(f"NHTSA 813219 {mode}, {lvl}",
                (lambda m, l: (lambda s: nhtsa(s, m, l)))(mode, lvl),
                True, "nhtsa813219")
    if include_half:
        add("Kusano-Gabler MAIS2+, dv=s/2",
            lambda s: kusano_mais2(s, half=True), True, "kusano2012+half")
        add("NHTSA 813219 all, MAIS3+, dv=s/2",
            lambda s: nhtsa(s, "all", "MAIS3+", half=True),
            True, "nhtsa813219+half")
    add("pedestrian logistic (synthetic)", pedestrian, False, "-")
    add("saturating ramp, 30 km/h (synthetic)",
        lambda s: ramp(s, 30.0), False, "-")
    add("saturating ramp, 60 km/h (synthetic)",
        lambda s: ramp(s, 60.0), False, "-")
    add("quadratic (synthetic)", quadratic, False, "-")
    add("Joksch fourth power", joksch, True, "joksch1993")
    add("step: any contact (synthetic)", step, False, "-")
    return lib
