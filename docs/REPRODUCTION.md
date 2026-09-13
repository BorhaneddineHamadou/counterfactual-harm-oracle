# Claim-to-command map

Every quantitative statement in the paper, the script that recomputes it,
and where the value lands. All commands run from the repository root with
`python` = Python 3.9+ with `requirements.txt`. "stored" = the default path
reads the precomputed model outputs under `results/`; "retrain" = the flag
that recomputes them from `data/` (runtimes are for a 32-core node).

## Section 5.1, Fig. 1, abstract — `python analysis/campaign_summary.py`

| claim | paper |
|---|---|
| campaigns complete: 750 nominal + 15,000 replays per subject | both |
| mean Monte-Carlo s.e. of the labels | 0.0011 openpilot, 0.0003 TransFuser |
| openpilot mean H, references with nonzero harm | 0.008, 28 of 150 |
| openpilot run-0 contact rate by template | 27% (cut-in) … 0% (crossing) |
| top 15 openpilot references hold … of suite harm; the top one | 88%; 11% |
| TransFuser oncoming-drift contact rate; cut-in; lead templates | 52%; a fifth; 0 |
| oncoming template's share of TransFuser harm; ego still moving at impact | 89%; 99% of contacts |
| Fig. 1 run (sid 40): approach speed, stop short, replays crashing, impact speed, iota, H, share of suite harm | 46 km/h, 4.4 m, 37/100, ~47 km/h, 0.20, 0.07, 6% |
| abstract pair: sid 55 (29 m, 2.2 s → 23/100 crash at 44 km/h) vs sid 16 (5.4 m, 0.4 s → 0/100) | as stated |
| GPU-hours of the reference campaigns | 800 and 1,400 |

Output: `results/summary/campaign_summary.json`.

## RQ1, Table 1 and Sec. 5.2 — `python analysis/rq1_report.py`

Stored: `results/rq1/<subject>_field_tier_oof.npz`. Retrain:
`python analysis/rq1_train_field_tier.py openpilot|transfuser` (~1 h / ~30 min;
reproduces the stored scores to floating-point precision on scikit-learn 1.8).

| row (openpilot APFD_H / r_s; TransFuser APFD_H / r_s) | paper |
|---|---|
| binary collision verdict | .714 / .481; .841 / .555 |
| minimum time-to-collision | .801 / .417; .881 / .609 |
| minimum clearance | .820 / .498; .878 / .704 |
| realized impact speed | .721 / .481; .867 / .560 |
| best of 35 CriMe measures | .828 / .646; .908 / .666 |
| field tier | .864 [.76,.94] / .729 [.59,.85]; .913 [.88,.93] / .790 [.69,.86] |
| + escalation, 15% at 3 replays (1.45×) | .864 / .748 (openpilot) |
| ablation: no replay corpus | .787 ± .04 / .668 ± .05; .885 ± .02 / .755 ± .02 (`rq1_ablations.py corpus`) |
| Tier-1, 1 replay (2×) | .744 / .610; .877 / .594 |
| Tier-1, 3 replays (4×) | .865 / .782; .913 / .710 |
| true harm, M = 100 (101×) | .954 / 1; .940 / 1 |

Prose: r_s on non-collision runs 0.68 / 0.70; r_s excluding oncoming drift
on TransFuser 0.73 ± 0.03; paired r_s margin over the best CriMe measure
+.082 [−.098, .264] / +.124 [.011, .239]; APFD gap over the verdict
+.153 [.039, .281] / +.073 [.032, .130]; TransFuser APFD gap over TTC
+.032 [.007, .053] and clearance +.034 [.008, .061]; ties on pairs whose
harms differ by more than 10×: verdict 63% (openpilot) and 67% (TransFuser),
TTC flag 36% (openpilot), field tier 8% and 2%; escalation 10% at 3 replays
0.737 (1.3×).
Output: `results/rq1/table1.json`.

## RQ1, calibration — `python analysis/rq1_calibration.py` (`--rerun` ~20–40 min per subject)

| claim | paper |
|---|---|
| coverage of the 90% intervals, out of fold | 92.7% ± 1.9; 93.3% ± 2.2 (abstract and RQ1 answer: 93%) |
| mean absolute error | 0.006; 0.003 |
| Monte-Carlo error of the labels below model error | 6.0× (openpilot) |
| split-half reliability of the labels | 0.98 (openpilot; TransFuser 0.96) |
| axiomatic battery | 219 cases, 0 violations (`analysis/axioms.py`) |

Output: `results/rq1/calibration.json`.

## RQ1, ablations — `python analysis/rq1_ablations.py [corpus|sharing|target|tau]`

| claim | paper |
|---|---|
| no sharing at all (150 references only): Δr_s | −0.074; −0.088, negative in all ten repeats |
| sharing across the k executions once replays are present | +0.012; 0.000 |
| dropping crashed replay rows | −0.028 (TransFuser) |
| at 150 references TransFuser falls to the best single-run baseline | .692 vs clearance .704 |
| binary target vs harm target, same estimator | .482 / .553 vs .730 / .780; paired −.248 ± .034, −.227 ± .020 |
| the same at the nominal-only corpus | −.187 ± .044, −.200 ± .014 |
| four scalars instead of 71 features | .617 / .710, paired −.112 / −.070 |
| replay rows worth to the harm target vs the free-label targets | +.062 / +.028 vs +.001 |
| ungated correlations on the non-collision runs, free labels vs harm | −.053 … +.074 vs .500 / .673 |
| recalibrated minimum clearance | .463 / .664 |
| four scalars on TransFuser APFD_H | .895 vs .885 |
| tie threshold: fold-internal choice vs declared 0.05 | 0.714 ± 0.048 vs 0.729 ± 0.034 |

Stored: `results/rq1/<subject>_{corpus_ablation,label_sharing,target_ablation}.npz`,
`results/rq1/openpilot_field_tier_oof_cvtau.npz`. Output: `results/rq1/ablations.json`.

## RQ2, Table 2 and Sec. 5.3 — `python analysis/rq2_predictive_validity.py`

| row (openpilot AUC / AUC_pass; TransFuser AUC / AUC_pass) | paper |
|---|---|
| binary verdict | .731 / —; .818 / — |
| realized impact speed | .731 / .500; .820 / .500 |
| −TTC_min | .742 / .522 [.31,.74]; .893 / .711 [.51,.89] |
| −min clearance | .893 / .815 [.72,.90]; .927 / .815 [.68,.93] |
| field tier | .953 [.89,.99] / .923 [.80,.99]; .961 [.92,.99] / .901 [.80,.99] |
| Ĥ (M=1), 2× | .820 / .764; .885 / .764 |
| Ĥ (M=3), 4× | .819 / .764; .900 / .733 |
| H (M=100), 101× | .859 / .840; .902 / .742 |

Counts: 600 held-out executions per subject; 23 / 31 scenarios crash again while
run 0 called 13 / 21; 137 / 129 passing run-0 scenarios of which 12 / 11
crash again, holding 48% / 12% of held-out harm. Paired on the pass
subset: field tier − TTC +0.372 [0.135, 0.598] / +0.171 [0.038, 0.332];
− clearance +0.084 [−0.079, 0.225] / +0.066 [−0.046, 0.172]. All 150:
gap over the verdict +0.200 [0.094, 0.310] / +0.126 [0.043, 0.213], over
TTC on openpilot +0.188 [0.044, 0.344]. Severity among scenarios that
crash again, TransFuser (n = 31): field tier 0.87 [0.74, 0.93], impact
speed 0.81, three replays 0.82, M = 100 0.82, verdict 0.57 with 94% ties;
openpilot (n = 23) every interval crosses zero. Point estimates reproduce
exactly; the bootstrap interval endpoints and paired means depend on the
resampling stream at the third decimal (e.g. +0.376 [0.119, 0.608] for the
paper's +0.372 [0.135, 0.598]). Output: `results/rq2/table2.json`
(`--exclude-reexecuted` reproduces the 599-run subset of an earlier draft,
see `data/README.md`; every number is identical).

## RQ3, Table 3 and Sec. 5.4 — `python analysis/rq3_kernel_rescaling.py` (`--retrain` ~35 min per subject)

| claim | paper |
|---|---|
| label agreement across scales, minimum over pairs, all / non-trivial | 0.64 / 0.34; 0.80 / 0.68 |
| distilled-oracle agreement, minimum over pairs, all / non-trivial | 0.71 / 0.57; 0.77 / 0.57 |
| references with H > 0 at α = 0.5 / 1 / 2 | 14 / 18 / 23%; 55 / 61 / 75% |
| openpilot mean H; TransFuser median H | .0073 / .0078 / .0150; .0002 / .0004 / .0010 |
| oracle r_s vs canonical labels at 0.5 / 1 / 2 | .60 / .70 / .63; .72 / .79 / .76 |
| TransFuser paired per-reference harm, α = 2 vs 1: rises vs falls | 82 vs 30, Wilcoxon p < 10⁻⁵ |
| cost of halving vs doubling | −0.095 vs −0.062; −0.070 vs −0.035 |

Stored: `results/rq3/<subject>_kernel_oracle_scores.npz`. Output: `results/rq3/table3.json`.

## RQ3, injury curves — `python analysis/rq3_injury_curves.py` (`--sweep` re-distills, hours)

| claim | paper |
|---|---|
| fourteen published curves move mean harm by a factor of | 25 (openpilot) to 220 (TransFuser) |
| training ordering stays at r_s ≥ … with the shipped curve | 0.998; 0.922 |
| comparable pairs that reorder | 1.3%; 9.4% (the fourth-power rule alone) |
| field tier's r_s lead over the best telemetry scalar across all fourteen | +0.23 (openpilot); +0.08 to +0.09 (TransFuser) |

Output: `results/rq3/injury_curves.json` (`--sweep`, not reported in the paper:
re-distilling the tier under each curve at a reduced protocol,
`results/rq3/injury_sweep_<subject>.json`).

## RQ4, Sec. 5.5 and Fig. 3 — `python analysis/rq4_portability.py` (~10 min)

| claim | paper |
|---|---|
| zero-shot r_s, openpilot oracle on TransFuser / reverse (local ceilings) | 0.31 (0.79); 0.26 (0.73) |
| zero-shot coverage at a claimed 90% | 78%; 91% (the TransFuser oracle's foreign interval is 2.5× its native out-of-fold width) |
| member disagreement into the crash-rich world; foreign runs above the native escalation threshold | 16×; 81% |
| re-anchoring with n = 25 local labels | 96.8% ± 4.5; 97.5% ± 2.8 |
| M = 30 labels leave every point unchanged to three decimals; n = 25 × M = 30 = 750 replays = 5% of a campaign | as stated |

Output: `results/rq4/portability_rerun.json`, `results/rq4/fig3_reanchoring.csv`.

## External baselines and suite validity

- `python analysis/crime_baselines.py`: 35 CommonRoad-CriMe measures at
  their stronger folding; best 0.646 (CPI) on openpilot, inverting to
  −0.23 on TransFuser; 0.666 (WTTC) on TransFuser, 0.44 on openpilot;
  non-collision slice 0.48 / 0.53. Output `results/crime/crime_table.json`.
- `python analysis/regulatory_mapping.py`: openpilot stopped-lead suite
  93% inside the Euro NCAP CCRs speed range, cut-ins entirely inside the
  UN R157 lateral-velocity and distance boxes; TransFuser below grids
  starting at 20–50 km/h. Output `results/regulatory/regulatory_mapping_rerun.json`.
- `python analysis/axioms.py`: A1 monotonicity, A2 invariance, A3
  continuity on the pilot world at M = 100, seeds 20260722 / 1 / 987654:
  219 cases, 0 violations. Output `results/axioms/`.

## Figures

`python analysis/make_figures.py` → `results/figures/fig1_severity_curve.*`
(the curve in Fig. 1, right) and `fig3_reanchoring.*` (Fig. 3). Fig. 2 is
a schematic.

## Reproduction status

Every script above was run while assembling this package (scikit-learn
1.8, numpy 1.26), and again after the paper's numbers were revised on
2026-09-13. Everything reproduces to the printed precision: all of Sec. 5.1 and the worked
  examples; every row and interval of Table 1; the paired margins, ties,
  escalation rows and per-template correlations of RQ1; coverage, MAE,
  split-half reliability; every ablation of Sec. 5.2 and the
  tie-threshold ablation of Sec. 7; all of Table 2 and the RQ2 prose;
  Table 3 and the RQ3 prose; the injury-curve re-scoring (span, r_s,
  reordered pairs, r_s lead over the telemetry scalars); every RQ4 number
  including the 2.5× width (foreign interval width over the TransFuser
  oracle's native out-of-fold width, 0.0562 / 0.0226); the CriMe ranking
  (0.646 / 0.666, non-collision 0.484 / 0.527); the regulatory fractions;
  the axiom battery (219 / 0). Retraining the field tier reproduces the
  stored out-of-fold scores to floating-point precision on both subjects
  (all ten repeats, maximum difference 4e-16;
  `results/rq1/<subject>_field_tier_oof_rerun.npz`); retraining the
  conformal magnitude readout reproduces the stored per-repeat coverage,
  width and MAE exactly (`results/rq1/<subject>_calibration_rerun.npz`);
  rebuilding `data/` from the raw traces reproduces every derived file
  to 1 ulp.
