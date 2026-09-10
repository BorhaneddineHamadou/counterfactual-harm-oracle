"""Pure brake-command shaping used by the bridge's RQ3 version arms.

Called once per 100 Hz frame with the post-latency, post-brake_gain
command. Stateless except for peak_frames, which the caller threads
through (consecutive frames at or above fade_demand).

  fade: once demand has been >= fade_demand for fade_sustain frames,
        braking effectiveness drops by fade_frac (brake fade under
        sustained peak demand).
  compensation: light braking (|gas| < fade_demand) is scaled by
        comp_boost (>= 1) -- the earlier, more cautious onset -- and
        capped at full brake.
"""


def shape_brake(eff_gas, peak_frames, fade_frac, fade_demand,
                fade_sustain, comp_boost):
    """Returns (shaped_gas, new_peak_frames)."""
    if eff_gas >= 0:
        return eff_gas, 0
    if -eff_gas >= fade_demand:
        peak_frames += 1
        if fade_frac > 0.0 and peak_frames >= fade_sustain:
            eff_gas = eff_gas * (1.0 - fade_frac)
        return eff_gas, peak_frames
    if comp_boost != 1.0:
        eff_gas = max(-1.0, eff_gas * comp_boost)
    return eff_gas, 0
