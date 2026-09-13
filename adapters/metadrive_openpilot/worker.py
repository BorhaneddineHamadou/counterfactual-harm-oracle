#!/usr/bin/env python3
"""Proxima batch worker — runs inside the openpilot Singularity image on a
GPU node. Consumes a JSONL job list, executes each job with a fresh openpilot
manager (mirroring campaign_worker's lifecycle), and appends result records.

Job record:
  {"job_id": str, "scenario_cls": str, "params": {...}, "seed": int,
   "duration": int, "perturb": {...}|null, "trace_out": str}

Perturb blob (shared by bridge and scenario classes via PROXIMA_PERTURB):
  {"start_step": int, "dlat_s": float, "brake_gain": float,
   "ou_sigma_long": float, "ou_sigma_lat": float, "ou_theta": float,
   "seed": int}

Idempotent: a job is skipped if its trace_out already exists.
Only our own manager's process group is killed between jobs (no broad pkill:
other campaigns may share the node).

Env: JOBS_FILE, RESULT_FILE, MANAGER_LOG, DISPLAY_NUM.
"""
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
OPENPILOT_ROOT = os.path.expanduser(os.environ.get("OPENPILOT_ROOT", "~/openpilot"))
sys.path.insert(0, os.path.join(HERE, "openpilot_sim"))
sys.path.insert(0, HERE)
sys.path.insert(0, OPENPILOT_ROOT)

import numpy as np  # noqa: E402

import run_openpilot_smoke as smoke  # noqa: E402
import scenario_bridge  # noqa: E402
import bridge  # noqa: E402

# run_one() resolves ScenarioMetaDriveBridge from the scenario_bridge module
# at call time -- swap in the Proxima bridge (same trick as the injection
# campaign; the file on disk is untouched).
scenario_bridge.ScenarioMetaDriveBridge = bridge.ProximaMetaDriveBridge
smoke.ScenarioMetaDriveBridge = bridge.ProximaMetaDriveBridge


PARAMS_D = Path.home() / ".comma" / "params" / "d"
# Latching offroad alerts live in the SHARED persistent params dir, so one
# tripped guard blocks openpilot startup for every later run on every node
# (hardwared: startup_conditions["no_excessive_actuation"]) -> silent 100%
# no_engage. Cleared before every attempt.
STICKY_OFFROAD_ALERTS = ("Offroad_ExcessiveActuation",)


def clear_sticky_alerts():
    """Returns the alerts that were set (i.e. tripped by the previous run)."""
    tripped = []
    for k in STICKY_OFFROAD_ALERTS:
        try:
            (PARAMS_D / k).unlink()
            tripped.append(k)
            print(f"      [guard] cleared stale {k}", flush=True)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"      [guard] could not clear {k}: {e}", flush=True)
    return tripped


def kill_manager(manager):
    # NEVER killpg here: the manager shares our process group, so killpg
    # terminates the whole SLURM step (learned the hard way in job 627813).
    # openpilot's manager tears down its children on SIGTERM.
    try:
        manager.terminate()
        manager.wait(15)
    except Exception:
        try:
            manager.kill()
        except Exception:
            pass


def trace_contact(path):
    """Authoritative contact signal from the trace itself: the env done
    reason can mask a crash (calib-3 crossing run 0 hit at dv=10.2 m/s but
    was reported 'out_of_lane'). Returns (contact, dv_at_first_crash)."""
    try:
        z = np.load(path, allow_pickle=True)
        cols = [str(c) for c in z["columns"]]
        d = z["data"]
        col = {k: d[:, i] for i, k in enumerate(cols)}
        idx = np.flatnonzero(col["crash"] > 0.5)
        if not len(idx):
            return False, 0.0
        i = int(idx[0])
        dv = float(np.hypot(col["ego_vx"][i] - col["th_vx"][i],
                            col["ego_vy"][i] - col["th_vy"][i]))
        if not np.isfinite(dv):
            dv = float(np.hypot(col["ego_vx"][i], col["ego_vy"][i]))
        return True, round(dv, 2)
    except Exception:
        return None, None


def trace_ego_vmax(path):
    """Peak |ego v_x| in a trace, or nan if unreadable.

    A reference run whose ego never moves is 'safe' only because nothing
    happened: its telemetry sits at the measurement caps while its kernel
    replays drive normally, so the replays diverge from the run they are
    meant to neighbour by far more than the kernel accounts for. Those
    runs are rejected and re-executed, in the same spirit as the existing
    no_engage retry.
    """
    try:
        import numpy as _np
        d = _np.load(path, allow_pickle=True)
        cols = [str(c) for c in d["columns"]]
        return float(_np.nanmax(_np.abs(d["data"][:, cols.index("ego_vx")])))
    except Exception:
        return float("nan")


def main():
    jobs_file = os.environ["JOBS_FILE"]
    result_file = os.environ.get(
        "RESULT_FILE", str(Path(jobs_file).with_suffix(".results.jsonl")))
    jobs = [json.loads(l) for l in open(jobs_file) if l.strip()]
    lo = int(os.environ.get("JOB_LO", 0))
    hi = int(os.environ.get("JOB_HI", len(jobs)))
    jobs = jobs[lo:hi]
    print(f"[proxima-worker] {len(jobs)} jobs ({lo}:{hi}) -> {result_file}",
          flush=True)

    display = int(os.environ.get("DISPLAY_NUM", "97"))
    xvfb = smoke.start_xvfb(display)
    print(f"[proxima-worker] Xvfb :{display} up", flush=True)

    for i, job in enumerate(jobs):
        trace_out = job["trace_out"]
        if os.path.exists(trace_out) or os.path.exists(trace_out + ".npz"):
            print(f"  [{i}] {job['job_id']} SKIP (trace exists)", flush=True)
            continue
        Path(trace_out).parent.mkdir(parents=True, exist_ok=True)

        os.environ["PROXIMA_SCENARIO_PARAMS"] = json.dumps(job["params"])
        os.environ["PROXIMA_SEED"] = str(job["seed"])
        os.environ["PROXIMA_TRACE_OUT"] = trace_out
        vdir = os.environ.get("PROXIMA_VIDEO_DIR", "")
        if vdir:
            os.environ["PROXIMA_VIDEO_OUT"] = os.path.join(
                vdir, f"{job['job_id']}.mkv")
        else:
            os.environ.pop("PROXIMA_VIDEO_OUT", None)
        if job.get("perturb"):
            os.environ["PROXIMA_PERTURB"] = json.dumps(job["perturb"])
        else:
            os.environ.pop("PROXIMA_PERTURB", None)

        print(f"  [{i}] {job['job_id']} {job['scenario_cls']} "
              f"seed={job['seed']} perturb={bool(job.get('perturb'))}",
              flush=True)
        t0 = time.time()
        retries = int(os.environ.get("PROXIMA_RETRIES", "3"))
        tripped = []
        for attempt in range(retries):
            tripped += clear_sticky_alerts()
            manager = smoke.start_manager()
            time.sleep(8)
            try:
                r = smoke.run_one(job["scenario_cls"], job["job_id"],
                                  int(job.get("duration", 40)))
            except Exception as e:
                r = {"scenario": job["scenario_cls"], "outcome": f"error:{e}"}
                traceback.print_exc()
            finally:
                kill_manager(manager)
            bad = r.get("outcome") in ("no_engage", "bridge_start_timeout",
                                        "run_timeout")
            # validity gate: the ego must actually drive for the run to be a
            # usable reference. Off by default (0.0) so existing campaigns
            # are unchanged; set PROXIMA_MIN_EGO_VMAX to enable.
            vmin = float(os.environ.get("PROXIMA_MIN_EGO_VMAX", "0"))
            if not bad and vmin > 0:
                tp = (trace_out if os.path.exists(trace_out)
                      else trace_out + ".npz")
                if os.path.exists(tp):
                    v = trace_ego_vmax(tp)
                    if v == v and v < vmin:
                        bad = True
                        r["outcome"] = f"degenerate_ego_vmax_{v:.2f}"
                        os.remove(tp)          # else the next attempt SKIPs
            if not bad:
                break
            print(f"      retry {attempt+1}/{retries} "
                  f"({r.get('outcome')})", flush=True)
            # re-execute as an independent draw, deterministically
            os.environ["PROXIMA_SEED"] = str(int(job["seed"])
                                             + 1000000 * (attempt + 1))
        # a trip seen on the NEXT attempt's clear was caused by this run
        tripped += clear_sticky_alerts()
        r["attempts"] = attempt + 1
        r["sticky_alerts"] = sorted(set(tripped))
        r["job_id"] = job["job_id"]
        r["seed"] = job["seed"]
        r["perturb"] = job.get("perturb")
        r["wall_s"] = round(time.time() - t0, 1)
        r["trace_exists"] = os.path.exists(trace_out) or os.path.exists(
            trace_out + ".npz")
        if r["trace_exists"]:
            p = trace_out if os.path.exists(trace_out) else trace_out + ".npz"
            r["contact"], r["dv"] = trace_contact(p)
        with open(result_file, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"      -> {r['outcome']} engaged={r.get('engaged')} "
              f"contact={r.get('contact')} trace={r['trace_exists']} "
              f"wall={r['wall_s']}s", flush=True)

    xvfb.terminate()
    print("[proxima-worker] done", flush=True)


if __name__ == "__main__":
    main()
