# Field-tier features

Everything is computed from the judged run's own trace (`proxima/trace.py`
`Trace`: ego speed `v_e`, ego acceleration `a_e`, bumper gap `gap`, closing
speed `closing`, lateral offset `y_rel`, overlap and active masks, 20 Hz).
Column order is the order stored in `data/<subject>/features_*.npz`
(`names71`, `names_traj`). Units: m, s, m/s, m/s².

## Block 1: 41 single-trace descriptors (`proxima/features_expanded.py`)

| # | name | meaning |
|---|---|---|
| 0 | ttc_min | minimum time-to-collision over active closing steps (cap 10 s; 0 on contact) |
| 1 | min_clearance | minimum gap over active steps (cap 50 m; 0 on contact) |
| 2 | speed_at_crit | ego speed at the criticality peak |
| 3 | dv_realized | realized impact speed (0 without contact) |
| 4 | peak_jerk | max |Δa_e|/dt |
| 5 | peak_lat_rate | max lateral rate of the threat |
| 6 | log_ttc_min | log10(min(TTC, 30) + 0.1) |
| 7–9 | tet1, tet2, tet3 | time exposed below TTC 1, 2, 3 s |
| 10 | tit3 | time-integrated TTC deficit below 3 s |
| 11 | drac_max | max deceleration rate to avoid crash (cap 30) |
| 12 | closing_at_crit | closing speed at the criticality peak |
| 13 | gap_at_crit | gap at the criticality peak (cap 50) |
| 14 | v_threat_at_crit | threat speed at the peak |
| 15 | log_min_gap | log10(min gap + 0.1) |
| 16 | max_closing | maximum closing speed while active |
| 17 | decel_peak | peak ego deceleration |
| 18 | mean_decel_2s | mean deceleration over the 2 s before the peak |
| 19 | t_active | time the threat is active |
| 20 | t_crit | time of the criticality peak |
| 21–24 | gap_at_tstar, closing_at_tstar, v_e_at_tstar, ttc_at_tstar | encounter state at the branch point t* |
| 25 | ke_at_crit | v_e² / 100 at the peak |
| 26 | jerk_rms | RMS jerk |
| 27–34 | cf_mingap_d{25,0,25,10}_g{100,80,80,90}, cf_dv_… | open-loop counterfactual replays of the recorded kinematics with delayed (d, in 0.01 s) and scaled (g, in %) ego braking: predicted min gap and impact speed |
| 35 | phys_H | kernel-averaged analytic harm of the open-loop replay (64 draws); 0 on replay rows of the training corpus |
| 36–40 | tpl_* | one-hot template indicator |

## Block 2: 30 extra descriptors (`proxima/features_extra.py`)

| name | meaning |
|---|---|
| gap_q10, gap_q25, gap_q50 | gap percentiles over active steps |
| t_gap_lt2, t_gap_lt5, t_gap_lt10 | time with gap below 2, 5, 10 m |
| gap_slope_2s | gap change over the 2 s before the peak |
| v_min_act | minimum ego speed while active |
| v_e_m1s, v_e_m2s, v_e_m3s | ego speed 1, 2, 3 s before the peak |
| v_drop_frac | 1 − v_e(peak) / v_e(t*) |
| react_lag | time from first activity to the first braking step (< −1 m/s²) |
| max_sust_decel | maximum 1 s moving-average deceleration |
| brake_taps | number of braking onsets |
| t_areq_gt_amax | time the required deceleration exceeds 7 m/s² |
| areq_at_tstar | required deceleration at t* (cap 30) |
| stop_margin_crit, stop_margin_tstar | gap minus stopping distance at 7 m/s² |
| ymin_act | minimum lateral offset while active |
| t_overlap | time in lateral overlap |
| closing_at_ov | max closing speed while overlapping |
| cf2_mingap_…, cf2_dv_… | four further open-loop counterfactual settings (d 0.10/0.40/0.25/0.40 s, g 1.0/1.0/0.9/0.8) |

## Block 3: 71 checkpoint-trajectory descriptors (`proxima/features_traj.py`)

The approach sampled at fixed gap checkpoints 30, 20, 15, 10, 7, 5, 3, 2 m
(first crossing while closing): closing speed `cp<g>_clos`, lateral offset
`cp<g>_yr`, lateral rate `cp<g>_lr`, ego speed `cp<g>_v`, ego acceleration
`cp<g>_a`, TTC `cp<g>_ttc` and brake/gas command `cp<g>_cmd` (56 columns;
sentinel values when the checkpoint is never reached). Then the race margin
between time-to-evade and time-to-contact (`race_min`, `race_end`,
`race_frac_bad`), the commitment state at minimum TTC (`yr_at_minttc`,
`lr_at_minttc`, `commit`), the end state (`t_contact_end`, `closing_end`,
`gap_end`, `v_end`, `yr_end`, `lr_end`) and the command channel
(`cmd_min`, `cmd_brk_frac`, `cmd_osc`).

The four families the paper names (margins and their duration, speed
when it mattered, timing and strength of the reaction, encounter state at
fixed distances) map onto blocks 1–2 (families 1–3) and block 3 (family 4).
