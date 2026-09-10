# Campaign layer

The scripts that generate, run and harvest the two reference campaigns
(paper Sec. 4.2). See `docs/CAMPAIGN.md` for the end-to-end recipe and
budget; nothing here is needed to reproduce the paper's numbers from
`data/`.

| script | stage |
|---|---|
| `openpilot_calibrate.py`, `transfuser_calibrate.py` | range calibration rounds (`generate <round>` → SLURM → `analyze <round>`); adopted ranges in `configs/` |
| `openpilot_campaign.py`, `transfuser_campaign.py` | `gen-nominal` (Latin-hypercube suite, 150 × 5 runs), `gen-tier1` (t* per reference, 100 replay jobs each with kernel draws), `gen-rq2` (alpha 0.5 / 2 arms), `status` |
| `build_features.py` | traces → every derived file in `data/<subject>/` (labels, features, held-out outcomes, RQ3 arms) |
| `run_crime_baselines.py` | CommonRoad-CriMe 0.4.5 measures on the reference executions → `data/crime/` (needs the toolbox and the nominal traces) |

Store layout and environment variables: `docs/CAMPAIGN.md`,
`adapters/<stack>/README.md`.
