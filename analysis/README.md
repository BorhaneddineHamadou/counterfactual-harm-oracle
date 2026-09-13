# Analysis scripts

One script per result in the paper. All read the derived campaign record
through `common.py` (`load_subject`), print each number next to the
paper's value in brackets, and write a JSON under `results/`. The default
path uses the stored model outputs in `results/`; `--rerun` / `--retrain`
recomputes them. `docs/REPRODUCTION.md` lists every claim with its command
and expected value.

| script | paper | what it does |
|---|---|---|
| `campaign_summary.py` | Sec. 5.1, Fig. 1, abstract | campaign statistics, the worked example, the inversion pair |
| `rq1_train_field_tier.py` | Sec. 3.2, 3.4 | trains the field tier out of fold (10 × 5-fold) and stores scores, detector and escalation |
| `rq1_report.py` | Table 1, Sec. 5.2 | APFD_H and r_s with bootstrap intervals, paired margins, ties, escalation |
| `rq1_calibration.py` | Sec. 5.2 | conformal coverage, width, MAE, label reliability |
| `rq1_ablations.py` | Table 1 row, Sec. 5.2, Sec. 7 | corpus, label-sharing, target / feature and tie-threshold ablations |
| `rq2_predictive_validity.py` | Table 2, Sec. 5.3 | the outside check: run 0 predicts runs 1–4 |
| `rq3_kernel_rescaling.py` | Table 3, Sec. 5.4 | rank stability under α ∈ {0.5, 1, 2} at both tiers |
| `rq3_injury_curves.py` | Sec. 5.4 | fourteen published injury curves |
| `rq4_portability.py` | Sec. 5.5, Fig. 3 | zero-shot transfer, self-awareness, re-anchoring |
| `crime_baselines.py` | Table 1 row, Sec. 5.2 | the 35 CommonRoad-CriMe measures against harm |
| `regulatory_mapping.py` | Sec. 4.2 | the suites against UN R157 / Euro NCAP parameter grids |
| `axioms.py` | Sec. 4.1, 5.2 | the metamorphic battery on the pilot world |
| `make_figures.py` | Fig. 1, Fig. 3 | figure panels from the stored results |

Conventions shared by every script (`common.py`, `proxima/metrics.py`):
scenario-level folds `np.array_split(default_rng(rep).permutation(150), 5)`
for repeats 0–9; bootstrap over scenarios with 4,000 draws, rows that
depend on a fit or a replay draw also drawing one repeat per sample;
harm-weighted APFD with mid-ranks for ties; AUC in its mid-rank form.
