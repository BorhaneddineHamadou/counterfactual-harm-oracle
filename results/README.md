# Precomputed results

Outputs of the analysis scripts as they were run for the paper, so every
table can be regenerated in seconds without retraining. Each analysis
script reads these by default and has a `--rerun` / `--retrain` path that
recomputes them from `data/`.

| path | produced by | contents |
|---|---|---|
| `rq1/<subject>_field_tier_oof.npz` | `analysis/rq1_train_field_tier.py` | out-of-fold field-tier scores, 10 CV repeats (`oofs`), with 10% escalation (`oofs_esc`), detector `p`, survivor `tail`, labels `y` |
| `rq1/openpilot_field_tier_oof_cvtau.npz` | `rq1_train_field_tier.py openpilot --tau cv` | the tie-threshold ablation of Sec. 7 |
| `rq1/<subject>_calibration.npz` | `analysis/rq1_calibration.py` | conformal coverage, width, MAE per repeat |
| `rq1/<subject>_corpus_ablation.npz` | `analysis/rq1_ablations.py corpus` | Table 1 row "no replay corpus" |
| `rq1/<subject>_label_sharing.npz` | `analysis/rq1_ablations.py sharing` | label-sharing arms (Sec. 5.2) |
| `rq1/<subject>_target_ablation.npz` | `analysis/rq1_ablations.py target` | target / feature arms (Sec. 5.2) |
| `rq1/table1.json`, `rq1/calibration.json`, `rq1/ablations.json` | the RQ1 scripts | the printed tables |
| `rq2/table2.json`, `rq2/table2_all_runs.json`, `rq2/predictive_validity.json` | `analysis/rq2_predictive_validity.py` | Table 2 and its paired tests (599-run criterion as in the paper; all 600 runs) |
| `rq3/<subject>_kernel_oracle_scores.npz`, `rq3/table3.json` | `analysis/rq3_kernel_rescaling.py` | field tiers distilled under each kernel scale; Table 3 (`_retrained` = fresh recomputation, bit-identical) |
| `rq3/injury_curves*.json` | `analysis/rq3_injury_curves.py` | re-scoring under fourteen published curves; the re-distillation sweep |
| `rq4/portability*.json`, `rq4/fig3_reanchoring.csv` | `analysis/rq4_portability.py` | zero-shot transfer, self-awareness, re-anchoring (Fig. 3) |
| `crime/` | `analysis/crime_baselines.py` | the 35 CommonRoad-CriMe measures ranked against harm |
| `regulatory/` | `analysis/regulatory_mapping.py` | the scenario suite against UN R157 / Euro NCAP grids |
| `axioms/` | `analysis/axioms.py` | the metamorphic battery (219 cases, 0 violations) |
| `summary/` | `analysis/campaign_summary.py` | Sec. 5.1 statistics, Fig. 1 example, abstract pair |
| `figures/` | `analysis/make_figures.py` | Fig. 1 (right) and Fig. 3 |

Files ending in `_rerun` are fresh recomputations produced while
assembling this package, kept next to the originals as evidence that the
pipeline reproduces.
