# Raw simulation traces

Every analysis in this repository runs from the derived record in `data/`
(about 50 MB). The raw traces behind it, one `.npz` per execution, are
larger (openpilot 278 MB, TransFuser 581 MB) and are distributed as
archived tarballs rather than in git:

| tarball | contents | sha256 |
|---|---|---|
| `counterfactual-harm-oracle-traces-openpilot.tar` | 750 nominal + 15,000 reference replays + 2 × 6,000 rescaled-kernel replays (27,750 traces) | `ba92f88083510d858ee20c69552d1200c284f488026641368e1bb04ff693206d` |
| `counterfactual-harm-oracle-traces-transfuser.tar` | 750 nominal + 15,000 reference replays + 2 × 3,000 rescaled-kernel replays (21,750 traces) | `b535ea82cb31d8d4fc3c9a28ce2b842a71b330ff59383a5c4c6c038df82e902f` |

Unpack into `data/<subject>/traces/` (`tools/fetch_traces.sh` does this
given `TRACES_URL`), then `campaign/build_features.py <subject>`
regenerates every file in `data/<subject>/` and `campaign/run_crime_baselines.py`
recomputes the third-party criticality measures.

## Trace format

`numpy.savez` archives with three entries:

- `columns`: the channel names;
- `data` (T × columns): one row per simulation step at 20 Hz. Both
  adapters record `step, ego_x, ego_y, ego_vx, ego_vy, ego_heading,
  ego_steering, cmd_steer, cmd_gas, crash, th_x, th_y, th_vx, th_vy,
  th_heading, engaged` (threat state is NaN while no threat exists); the
  CARLA adapter adds the actual simulation time `t` of each row and
  `harvest.py` resamples to the uniform grid;
- `meta` (JSON string): scenario class, concrete parameters, seed,
  perturbation blob (MetaDrive) or town and agent (CARLA), engagement step,
  and the simulator's done reason.

`adapters/<stack>/harvest.py::to_trace` converts a file into the canonical
`proxima.trace.Trace` (gap, closing speed, lateral offset, overlap and
active masks, contact, impact speed) that every feature extractor and the
Tier-1 labeller consume.

File names: `s<sid>_r<run>.npz` (nominal), `s<sid>_r0_b<m>.npz` (replay m
of the reference run).
