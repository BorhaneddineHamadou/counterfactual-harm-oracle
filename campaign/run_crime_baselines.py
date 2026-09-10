"""Third-party criticality baselines (CommonRoad-CriMe) on the reference executions.

Computes every implemented measure of CommonRoad-CriMe 0.4.5 (Lin and Althoff,
IV 2023) on the same reference executions the Tier-1 oracle labelled, so that
H_kappa can be compared against 30+ danger scores that we did not implement and
whose direction of criticality was declared by their authors, not by us.

Nothing is re-simulated: each measure reads a CommonRoad scenario reconstructed
from the stored trace (see proxima/crime_bridge.py).

Per trace and measure we report two scalars:
  *_worst : the measure reduced over the execution using the *toolbox's own*
            monotone declaration (POS -> max, NEG -> min). This is the
            trace-level danger score, the analogue of our TTC_min.
  *_crit  : the measure evaluated at Proxima's criticality index t_crit, the
            same instant the counterfactual branch is anchored to.

Usage:
  python campaign/run_crime_baselines.py <subject> [--shard i/n] [--limit N]
Output: data/crime/<subject>_measures.json (one shard: --shard 0/1); the
shipped files were computed in 20 shards and merged. Needs commonroad-crime
0.4.5 and the nominal traces (docs/TRACES.md).
  subject in {openpilot, transfuser}
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import commonroad_crime.measure as M                      # noqa: E402
from commonroad_crime.data_structure.base import CriMeBase  # noqa: E402
from commonroad_crime.data_structure.configuration import (  # noqa: E402
    CriMeConfiguration)
from commonroad_crime.data_structure.type import TypeMonotone  # noqa: E402

from proxima.crime_bridge import (EGO_ID, THREAT_ID,  # noqa: E402
                                  trace_to_scenario)

SUBJECTS = {
    "openpilot": {
        "traces": "data/openpilot/traces/nominal",
        "refs": "data/openpilot/tier1_refs.json",
    },
    "transfuser": {
        "traces": "data/transfuser/traces/nominal",
        "refs": "data/transfuser/tier1_refs.json",
    },
}

# Measures that are stubs ("coming soon") in CriMe 0.4.5: they return None for
# every input. Excluded by inspection of the installed source, not by outcome.
STUBS = {"ACI", "AGS", "CS", "PRI", "P_SMH", "P_SRS", "PTTC", "RSS", "SP",
         "TC", "TV"}
# TTM is the generic time-to-manoeuvre base and needs a manoeuvre argument;
# its instantiations TTB/TTK/TTS/TTR are all included separately.
SKIP = STUBS | {"TTM"}

# Measures that aggregate over the whole execution internally: one call, at the
# first step, is the trace-level value; sweeping them would be meaningless.
WHOLE_TRACE = {"TET", "TIT", "CPI", "ET", "PET"}

MAX_GRID = 40          # evaluation points per trace for per-step measures
DEFAULT_DT = 0.05


def measure_classes():
    out = []
    for n in sorted(dir(M)):
        o = getattr(M, n)
        if (inspect.isclass(o) and issubclass(o, CriMeBase)
                and o is not CriMeBase and n not in SKIP):
            out.append((n, o))
    return out


def call(obj, cls, vehicle_id, time_step):
    params = list(inspect.signature(cls.compute).parameters)
    kw = {}
    if "vehicle_id" in params:
        kw["vehicle_id"] = vehicle_id
    if "time_step" in params:
        kw["time_step"] = time_step
    if "verbose" in params:
        kw["verbose"] = False
    return obj.compute(**kw)


def reduce_by_monotone(values, monotone):
    """Trace-level score, folded the way the toolbox declares the measure.

    Infinities are kept, not filtered: for a NEG measure such as THW or TTC,
    inf is the toolbox's way of saying 'no conflict at this step', and an
    execution whose every step is inf genuinely scores inf. Dropping them
    would turn 'never in conflict' into missing data.
    """
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)],
                   dtype=float)
    if not len(v):
        return np.nan
    return float(np.max(v)) if monotone is TypeMonotone.POS else float(np.min(v))


def grid(n_steps, i_crit):
    """Evaluation steps: an even sweep plus the criticality instant."""
    hi = max(1, n_steps - 2)
    g = np.unique(np.clip(np.linspace(0, hi, min(MAX_GRID, hi + 1)).astype(int),
                          0, hi))
    return np.unique(np.concatenate([g, [int(np.clip(i_crit, 0, hi))]]))


def crit_index(path, t_crit, dt, lo):
    """t_crit (seconds, campaign clock) -> index inside the converted window."""
    return int(round(t_crit / dt)) - lo


def stage_scenario(sc, workdir):
    """Write the reconstructed scenario to disk.

    The reachable-set measures (DA, WTTR) re-load the scenario from
    config.general.path_scenario rather than from the object they were handed,
    so the converted scenario has to exist as a CommonRoad XML for them to run.
    """
    from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
    from commonroad.common.util import Interval
    from commonroad.geometry.shape import Rectangle
    from commonroad.planning.goal import GoalRegion
    from commonroad.planning.planning_problem import (PlanningProblem,
                                                      PlanningProblemSet)
    from commonroad.scenario.state import CustomState

    os.makedirs(workdir, exist_ok=True)
    name = str(sc.scenario_id)
    # commonroad-reach (used by DA and WTTR) resolves a planning problem from
    # the written file, so the ego gets one: start at its initial state, finish
    # where the recorded execution actually finished.
    ego = sc.obstacle_by_id(EGO_ID)
    last = ego.prediction.trajectory.state_list[-1]
    goal = GoalRegion([CustomState(
        time_step=Interval(max(0, last.time_step - 10), last.time_step + 10),
        position=Rectangle(length=12.0, width=6.0, center=last.position,
                           orientation=float(last.orientation)))])
    pp = PlanningProblem(planning_problem_id=999,
                         initial_state=ego.initial_state, goal_region=goal)
    pps = PlanningProblemSet([pp])
    w = CommonRoadFileWriter(sc, pps, author="proxima",
                             affiliation="", source="proxima-crime-bridge")
    w.write_to_file(os.path.join(workdir, name + ".xml"),
                    OverwriteExistingFile.ALWAYS)
    return name


def run_trace(path, t_crit, classes, dt=DEFAULT_DT, workdir=None):
    sc, info = trace_to_scenario(path, dt=dt)
    row = {"path": os.path.basename(path), "n_steps": info["n_steps"],
           "contact": info["contact"], "dv": info.get("dv", 0.0),
           "template": info.get("scenario")}
    if sc is None:
        row["error"] = info.get("error", "conversion failed")
        return row
    cfg = CriMeConfiguration()
    if workdir:
        sc_name = stage_scenario(sc, workdir)
        cfg.general.path_scenarios = workdir + os.sep
        cfg.general.path_output_abs = os.path.join(workdir, "output") + os.sep
        cfg.general.path_logs = os.path.join(workdir, "output", "logs") + os.sep
        cfg.general.set_scenario_name(sc_name)
        os.makedirs(cfg.general.path_output, exist_ok=True)
    cfg.update(ego_id=EGO_ID, sce=sc)

    i_crit = int(np.clip(crit_index(path, t_crit, dt, info["lo"]),
                         0, max(0, info["n_steps"] - 2)))
    row["i_crit"] = i_crit
    steps = grid(info["n_steps"], i_crit)

    for name, cls in classes:
        t0 = time.time()
        try:
            obj = cls(cfg)
        except Exception as e:
            row[f"{name}_worst"] = np.nan
            row[f"{name}_crit"] = np.nan
            row[f"{name}_err"] = f"init:{type(e).__name__}"
            continue
        vals, errs = [], 0
        use = [0] if name in WHOLE_TRACE else steps
        for ts in use:
            try:
                v = call(obj, cls, THREAT_ID, int(ts))
                vals.append(float(v) if v is not None else None)
            except Exception:
                vals.append(None)
                errs += 1
        row[f"{name}_worst"] = reduce_by_monotone(vals, cls.monotone)
        if name in WHOLE_TRACE:
            row[f"{name}_crit"] = row[f"{name}_worst"]
        else:
            k = int(np.flatnonzero(np.asarray(use) == i_crit)[0]) \
                if i_crit in list(use) else 0
            v = vals[k] if k < len(vals) else None
            row[f"{name}_crit"] = (float(v) if v is not None
                                   and not np.isnan(v) else np.nan)
        row[f"{name}_nfail"] = errs
        row[f"{name}_secs"] = round(time.time() - t0, 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", choices=sorted(SUBJECTS))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = SUBJECTS[a.subject]
    refs = json.load(open(os.path.join(ROOT, cfg["refs"])))
    if a.limit:
        refs = refs[:a.limit]
    i, n = (int(x) for x in a.shard.split("/"))
    refs = [r for k, r in enumerate(refs) if k % n == i]

    classes = measure_classes()
    print(f"{a.subject}: {len(refs)} reference traces, "
          f"{len(classes)} third-party measures, shard {a.shard}", flush=True)

    workdir = os.path.join(os.environ.get("CH_TMP", "/tmp"),
                           f"crime_{a.subject}_{i}of{n}_{os.getpid()}")
    rows, t0 = [], time.time()
    for k, r in enumerate(refs):
        p = os.path.join(ROOT, cfg["traces"], f"{r['job_id']}.npz")
        try:
            row = run_trace(p, r["t_crit"], classes, workdir=workdir)
        except Exception as e:
            row = {"path": os.path.basename(p), "error":
                   f"{type(e).__name__}: {e}"}
        row["sid"] = r["sid"]
        row["t_crit"] = r["t_crit"]
        row["ref_contact"] = r["contact"]
        rows.append(row)
        print(f"  [{k+1}/{len(refs)}] sid={r['sid']} "
              f"{time.time()-t0:.0f}s", flush=True)

    out = a.out or os.path.join(
        ROOT, "data/crime", f"{a.subject}_measures_{i}of{n}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1, default=float)
    print(f"wrote {out} ({len(rows)} rows, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
