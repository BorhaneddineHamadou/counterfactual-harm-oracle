# Precomputed results

Outputs of the analysis scripts, so every table regenerates in seconds
without retraining. Each analysis script reads these by default; its
`--rerun` / `--retrain` path recomputes them from `data/` and writes the
result next to the stored file with a `_rerun` / `_retrained` suffix for
comparison.

| path | produced by | contents |
|---|---|---|
| `rq1/<subject>_field_tier_oof.npz` | `analysis/rq1_train_field_tier.py` | out-of-fold field-tier scores, 10 CV repeats (`oofs`), with 10% escalation (`oofs_esc`), detector `p`, survivor `tail`, labels `y` |
| `rq1/openpilot_field_tier_oof_cvtau.npz` | `rq1_train_field_tier.py openpilot --tau cv` | the tie-threshold ablation of Sec. 7 |
| `rq1/<subject>_calibration.npz` | `analysis/rq1_calibration.py --rerun` | conformal coverage, width, MAE per repeat of the shipped log-harm readout |
| `rq1/<subject>_corpus_ablation.npz` | `analysis/rq1_ablations.py corpus` | Table 1 row "no replay corpus" |
| `rq1/<subject>_label_sharing.npz` | `analysis/rq1_ablations.py sharing` | label-sharing arms (Sec. 5.2) |
| `rq1/<subject>_target_ablation.npz` | `analysis/rq1_ablations.py target` | target / feature arms (Sec. 5.2) |
| `rq1/table1.json`, `rq1/calibration.json`, `rq1/ablations.json` | the RQ1 scripts | the printed tables |
| `rq2/table2.json` | `analysis/rq2_predictive_validity.py` | Table 2 and its paired tests (600 held-out runs per subject) |
| `rq3/<subject>_kernel_oracle_scores.npz`, `rq3/table3.json` | `analysis/rq3_kernel_rescaling.py` | field tiers distilled under each kernel scale; Table 3 |
| `rq3/injury_curves.json` | `analysis/rq3_injury_curves.py` | re-scoring under fourteen published curves (ordering stability, r_s lead over the telemetry scalars) |
| `rq4/portability.json`, `rq4/fig3_reanchoring.csv` | `analysis/rq4_portability.py` | zero-shot transfer, self-awareness, re-anchoring (Fig. 3) |
| `crime/crime_table.json` | `analysis/crime_baselines.py` | the 35 CommonRoad-CriMe measures ranked against harm |
| `regulatory/regulatory_mapping.json` | `analysis/regulatory_mapping.py` | the scenario suite against UN R157 / Euro NCAP grids |
| `axioms/` | `analysis/axioms.py` | the metamorphic battery (219 cases, 0 violations) |
| `summary/campaign_summary.json` | `analysis/campaign_summary.py` | Sec. 5.1 statistics, Fig. 1 example, abstract pair |
| `figures/` | `analysis/make_figures.py` | Fig. 1 (right) and Fig. 3 |
