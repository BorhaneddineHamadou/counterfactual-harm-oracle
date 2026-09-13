# Data dictionary

Two reference campaigns, one per subject, in the same layout. Every file
below is derived from the raw traces by `campaign/build_features.py`
(`docs/TRACES.md` for the traces themselves; `docs/CAMPAIGN.md` for
regenerating them). Scenario ids `sid` run 0–149 (30 per template, in
template order lead_decel, lead_stopped, cut_in, oncoming_drift,
crossing_traffic); replay ids `m` run 0–99.

## `<subject>/reference_labels.npz` — the reference set

| key | shape | meaning |
|---|---|---|
| `sids` | 150 | scenario id of each reference (run 0 of the scenario) |
| `y` | 150 | Tier-1 harm `H`: mean over the 100 replays of `iota(impact speed)` (0 without contact) |
| `se` | 150 | Monte-Carlo standard error of `y` |
| `crash_frac` | 150 | fraction of replays ending in contact |
| `contact_nom` | 150 | the run-0 binary verdict (trace-level contact) |
| `template` | 150 | template name |
| `X`, `feature_names` | 150 × 41 | the 41 single-trace descriptors of run 0 (`docs/FEATURES.md`) |
| `H_rep` | 150 × 100 | per-replay injury `iota(s_m)`; `y = nanmean(H_rep, 1)` exactly |
| `rep_sid`, `rep_m` | 15000 | replay table keys |
| `rep_dlat`, `rep_gain` | 15000 | the kernel draws of that replay (latency jitter in s, brake gain) |
| `rep_contact`, `rep_dv`, `rep_injury` | 15000 | replay outcome: contact, impact speed (m/s), injury weight |
| `global_seed` | scalar | the campaign's published seed |

## `<subject>/features_nominal.npz` — all 750 nominal executions

`sid`, `run` (0–4), `contact`, `X71` (750 × 71: blocks 1–2 of
`docs/FEATURES.md`, `names71`), `Xtraj` (750 × 71: block 3, `names_traj`).
Run-0 rows equal `reference_labels.X` on the first 41 columns.

## `<subject>/features_replays.npz` — all 15,000 reference replays

`rep_sid`, `rep_m` (same order as the replay table above), `X71`, `Xtraj`.
The expensive `phys_H` column is 0 on replay rows, as in the campaign.

## `<subject>/nominal_runs.csv` — one row per nominal execution

`sid, run, template, seed, outcome (worker's done reason), contact,
impact_speed_ms, ttc_min_s, min_clearance_m, speed_at_crit_ms,
ego_vmax_ms, t_crit_s, n_steps, in_paper_heldout`. Runs 1–4 are the
held-out criterion of RQ2. `in_paper_heldout` is 0 for one openpilot run
(s50_r2) whose first attempt crashed the bridge and which was re-executed
by a top-up job. The paper's criterion is all 600 runs; the RQ2 script's
`--exclude-reexecuted` reproduces the 599-run subset an earlier draft
used (every reported number is identical on both).

## `<subject>/replay_outcomes.npz`

Per reference replay: `sid`, `m`, `contact`, `dv` (impact speed), `v_ego`
(ego speed at contact), `ref_contact`. Used by the injury-curve
re-scoring (RQ3) and the campaign summary.

## `<subject>/rq2/` — the kernel-rescaling arms (paper RQ3)

`features_a05.npz`, `features_a20.npz` (replays under alpha 0.5 and 2:
`rep_sid`, `rep_m`, `X71`, `Xtraj`), `H_by_scale.npz` (per reference:
`H_0.5`, `H_1_crn` = the M-replay subset of the reference campaign that
shares the arms' draws, `H_1_full` = the M=100 label, `H_2.0`) and
`labels.json` (the label-level rank agreements). M = 40 on openpilot,
20 on TransFuser.

## Campaign record

`campaign_scenarios.json` (the concrete suite), `tier1_refs.json`
(`t_crit`, `t*`, contact per reference), `tier1_manifest.json` and
`rq2_manifest.json` (the kernels the replays embody), `jobs/*.jsonl` (the
exact job lists: scenario class, parameters, seed, perturbation draws,
relative trace path).

## `crime/`

`<subject>_measures.json`: every implemented CommonRoad-CriMe 0.4.5
measure on the 150 reference executions, two foldings each (`_worst`
over the run in the toolbox's declared direction, `_crit` at the
criticality instant), computed by `campaign/run_crime_baselines.py`;
`measure_monotone.json`: the toolbox's declared direction per measure.

## Validity notes

- Contact is taken from the trace's crash flag, never from the simulator's
  outcome field.
- `y` is in the declared severity unit of `iota`; orderings are what the
  paper reports, and Sec. 5.4 re-scores them under fourteen other curves.
- Labels are Monte-Carlo estimates: mean `se` 0.0011 (openpilot) and
  0.0003 (TransFuser), 6.0× and 9.2× below field-tier model error;
  split-half reliability 0.98 and 0.96.
