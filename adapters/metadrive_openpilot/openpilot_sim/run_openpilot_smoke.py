#!/usr/bin/env python3
"""
Real openpilot + MetaDrive smoke test.

Architecture:
  manager.py (subprocess) ← publishes managerState, selfdriveState, etc.
  ScenarioMetaDriveBridge (subprocess) → publishes pandaStates, CAN, camera
         └── scenario_metadrive_process (subprocess) → MetaDrive env + NPC scripting
  This process → SubMaster monitoring selfdriveState + queue for done_info

Engagement chain:
  bridge publishes pandaStates
  → hardwared sets deviceState.started=True
  → manager starts all on-road processes (modeld, controlsd, selfdrived…)
  → bridge sends CruiseButtons when selfdriveState.engageable=True
  → selfdriveState.active=True (openpilot engaged)

Two commits tested by default:
  HEAD  492ed7312  with FirstOrderFilter on lead prob (stable lead detection)
  OLD   534fb1971  before filter (raw lead prob — can flicker)
  Diff: only selfdrive/controls/radard.py — no scons rebuild needed.

Per-commit artifact injection:
  For commits with compiled .so artifacts in ~/openpilot_builds/<sha>/compiled/,
  the runner injects them via PYTHONPATH and LD_LIBRARY_PATH before starting
  the manager, so per-commit binaries override base SIF equivalents.

Usage (inside Singularity):
  singularity exec --nv --bind $HOME:$HOME ~/openpilot_dev.sif bash -c "
    export PATH=~/.local/bin:~/openpilot/.venv/bin:\\$PATH
    cd ~/openpilot
    python3 ~/openpilot_sim/run_openpilot_smoke.py
  "
  # Run all scenarios for a specific commit:
  python3 ~/openpilot_sim/run_openpilot_smoke.py --sha 64b0ede9
  # Run from manifest (all usable commits, one scenario each):
  python3 ~/openpilot_sim/run_openpilot_smoke.py --from-manifest --scenarios HighwayFollowing
"""
import os, sys, json, time, argparse, subprocess, shutil, traceback, uuid
from pathlib import Path
from multiprocessing import Queue

OP_ROOT      = Path(os.path.expanduser(os.environ.get("OPENPILOT_ROOT", "~/openpilot")))
SIM_DIR      = OP_ROOT / "openpilot/tools/sim"
SCRIPT_DIR   = Path(__file__).resolve().parent
BUILDS_DIR   = Path(os.path.expanduser(os.environ.get("OPENPILOT_BUILDS", "~/openpilot_builds")))
MANIFEST     = BUILDS_DIR / "manifest.jsonl"
RESULTS      = SCRIPT_DIR / "openpilot_smoke_results.jsonl"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(OP_ROOT))

COMMITS = [
    {
        "sha":   "492ed7312",
        "label": "HEAD: radard filtered-lead-prob",
        "files": [],
    },
    {
        "sha":   "534fb1971",
        "label": "534fb197: pre-radard-filter (raw prob)",
        "files": ["selfdrive/controls/radard.py"],
    },
]

ALL_SCENARIOS = [
    "HighwayFollowing", "OncomingVehicle", "EmergencyBraking",
    "UnprotectedLeftTurn", "PedestrianCrossing", "CyclistInteraction",
    "EgoLaneChange", "MergingOntoHighway", "RoundaboutEntryExit",
    "TrafficLightCompliance", "NightDriving", "LocalizationDegradation",
    "StaticObstacle",
]
# Only run radard-sensitive scenarios for the second commit
RADARD_SENSITIVE = ["HighwayFollowing", "EmergencyBraking", "OncomingVehicle"]


def load_manifest() -> dict:
    """Returns {sha: entry} for all manifest entries."""
    if not MANIFEST.exists():
        return {}
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            entries[e["sha"]] = e
    return entries


def resolve_sha(sha_prefix: str) -> str | None:
    """Resolve a short SHA prefix to a full SHA via the manifest or git."""
    manifest = load_manifest()
    matches = [s for s in manifest if s.startswith(sha_prefix)]
    if len(matches) == 1:
        return matches[0]
    r = subprocess.run(["git", "rev-parse", sha_prefix],
                       cwd=str(OP_ROOT), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def artifact_env_for_commit(sha: str) -> dict:
    """
    Returns extra env vars to inject per-commit compiled artifacts.
    Prepends compiled/ to PYTHONPATH and adds .so lib dirs to LD_LIBRARY_PATH.
    Returns {} if no compiled artifacts exist for this sha.
    """
    # Accept both short and full SHAs
    candidates = list(BUILDS_DIR.glob(f"{sha}*"))
    if not candidates:
        return {}
    build_dir = candidates[0]
    compiled = build_dir / "compiled"
    if not compiled.is_dir():
        return {}

    # Build PYTHONPATH: prepend compiled/ so per-commit .so files shadow base SIF ones.
    # The compiled/ tree mirrors the openpilot repo structure, so Python extension
    # imports (e.g. "from msgq.visionipc import ...") find the right version first.
    existing_pypath = os.environ.get("PYTHONPATH", "")
    new_pypath = str(compiled)
    if existing_pypath:
        new_pypath = f"{new_pypath}:{existing_pypath}"

    # Build LD_LIBRARY_PATH: add dirs containing native .so libraries.
    lib_dirs = [
        compiled / "third_party/acados/x86_64/lib",
        compiled / "selfdrive/locationd/models/generated",
        compiled / "opendbc_repo/opendbc/can",
        compiled / "msgq_repo/msgq",
        compiled / "msgq_repo/msgq/visionipc",
    ]
    existing_ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    new_ldpath = ":".join(str(d) for d in lib_dirs if d.is_dir())
    if existing_ldpath:
        new_ldpath = f"{new_ldpath}:{existing_ldpath}"

    env = {}
    if new_pypath:
        env["PYTHONPATH"] = new_pypath
    if new_ldpath:
        env["LD_LIBRARY_PATH"] = new_ldpath
    return env


def start_xvfb(display_num=97):
    xvfb_bin = str(OP_ROOT / ".venv/bin/Xvfb")
    p = subprocess.Popen(
        [xvfb_bin, f":{display_num}", "-screen", "0", "1280x720x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = f":{display_num}"
    time.sleep(1.5)
    return p


def start_manager(params_root: str = "", extra_env: dict | None = None):
    """Start openpilot manager in simulation mode.
    Uses DEFAULT params path (~/.comma/params/d/) — already properly initialised.
    soundd blocked: requires audio hardware; its absence blocks deviceState.started.
    extra_env: additional env vars (e.g. PYTHONPATH/LD_LIBRARY_PATH for per-commit artifacts).
    """
    env = os.environ.copy()
    env.update({
        "PASSIVE":       "0",
        "NOBOARD":       "1",
        "SIMULATION":    "1",
        "SKIP_FW_QUERY": "1",
        "FINGERPRINT":   "HONDA_CIVIC_2022",
        # Block hardware-dependent or unneeded processes
        "BLOCK": "camerad,loggerd,encoderd,micd,logmessaged,manage_athenad,soundd,ui",
        "DISPLAY": os.environ.get("DISPLAY", ":97"),
    })
    # Use default PARAMS_ROOT (~/.comma/params/d/) — already has valid
    # HasAcceptedTerms, CompletedTrainingVersion, CalibrationParams etc.
    # DO NOT set PARAMS_ROOT to an isolated path; isolated paths lack these keys.
    if params_root:
        env["PARAMS_ROOT"] = params_root
    if extra_env:
        env.update(extra_env)
    log_path = os.environ.get("MANAGER_LOG", "")
    if log_path:
        log_fh = open(log_path, "a")
        stdout, stderr = log_fh, log_fh
    else:
        log_fh = None
        stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
    proc = subprocess.Popen(
        ["bash", str(SIM_DIR / "launch_openpilot.sh")],
        cwd=str(SIM_DIR),
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    if log_fh:
        proc._log_fh = log_fh  # keep reference so it stays open
    return proc


def run_one(scenario_cls_name: str, commit_label: str, duration: int,
            params_root: str = "", extra_env: dict | None = None) -> dict:
    """
    Run one scenario with real openpilot.
    Manager must already be running.
    camerad.py timestamp fix required: eof = time.monotonic_ns() (not frame_id*50ms).
    Engagement takes ~90-100s from bridge start; allow 130s timeout.
    extra_env: per-commit artifact env vars (PYTHONPATH, LD_LIBRARY_PATH).
    """
    import cereal.messaging as messaging
    from scenario_bridge import ScenarioMetaDriveBridge
    from openpilot.tools.sim.bridge.common import QueueMessageType

    if params_root:
        os.environ["PARAMS_ROOT"] = params_root
    if extra_env:
        os.environ.update(extra_env)

    result = {
        "scenario": scenario_cls_name,
        "commit":   commit_label,
        "outcome":  "no_engage",
        "min_ttc":  None,
        "duration_s": 0.0,
        "engaged":  False,
        "time_to_engage_s": None,
    }
    t_start = time.monotonic()

    try:
        bridge = ScenarioMetaDriveBridge(
            scenario_cls_name=scenario_cls_name,
            dual_camera=False,
            high_quality=False,
            test_duration=duration,
            test_run=True,
        )
        q = Queue()
        p_bridge = bridge.run(q, retries=10)

        # ── 1. Wait for MetaDrive world to start (up to 60s) ─────────────
        # bridge.started.value becomes True when MetaDriveWorld's first vehicle
        # state is received — this is also when SimulatedCar begins publishing
        # pandaStates, which triggers hardwared to set deviceState.started=True.
        t_bridge_start = time.monotonic()
        bridge_start_deadline = t_bridge_start + 70
        while not bridge.started.value and time.monotonic() < bridge_start_deadline:
            if p_bridge.exitcode is not None:
                result["outcome"] = f"bridge_crash_exit{p_bridge.exitcode}"
                result["duration_s"] = round(time.monotonic() - t_start, 1)
                return result
            time.sleep(0.3)

        if not bridge.started.value:
            result["outcome"] = "bridge_start_timeout"
            result["duration_s"] = round(time.monotonic() - t_start, 1)
            p_bridge.terminate(); p_bridge.join(5)
            return result

        bridge_up_time = time.monotonic() - t_bridge_start
        print(f"      MetaDrive started in {bridge_up_time:.1f}s", flush=True)

        # ── 2. Monitor engagement via SubMaster ───────────────────────────
        # Full chain: pandaStates → deviceState.started → on-road processes
        # → calibration + locationd stabilise (~90s) → engageable=True
        # → bridge sends CruiseButtons → selfdriveState.active=True (~94s)
        sm = messaging.SubMaster(['selfdriveState', 'managerState'])
        engage_deadline = time.monotonic() + 130
        engaged_ever = False
        t_engaged = None

        while time.monotonic() < engage_deadline:
            sm.update(500)
            if sm['selfdriveState'].active and not engaged_ever:
                t_engaged = time.monotonic()
                engaged_ever = True
                tte = round(t_engaged - t_bridge_start, 1)
                print(f"      ENGAGED at {tte}s after bridge start", flush=True)
                result["time_to_engage_s"] = tte
                result["engaged"] = True
                break
            if not bridge.started.value:
                # Scenario ended before engagement
                break
            time.sleep(0.5)

        if not engaged_ever:
            # check why
            nr = [p.name for p in sm['managerState'].processes
                  if not p.running and p.shouldBeRunning]
            print(f"      no engagement after 120s. not_running={nr[:5]}", flush=True)
            result["outcome"] = "no_engage"
            result["duration_s"] = round(time.monotonic() - t_start, 1)
            p_bridge.terminate(); p_bridge.join(5)
            return result

        # ── 3. Wait for scenario to complete ─────────────────────────────
        # bridge.started.value becomes False when metadrive_process sends
        # a done simulation_state (timeout, out_of_lane, or collision).
        scenario_deadline = time.monotonic() + duration + 45
        while bridge.started.value and time.monotonic() < scenario_deadline:
            time.sleep(0.5)

        # ── 4. Collect outcome from queue ─────────────────────────────────
        outcome_set = False
        while not q.empty():
            try:
                msg = q.get_nowait()
                if msg.type == QueueMessageType.TERMINATION_INFO:
                    info = msg.info or {}
                    result["min_ttc"] = info.get("min_ttc")
                    result["done_info"] = {k: v for k, v in info.items() if k != "min_ttc"}
                    if info.get("timeout"):
                        result["outcome"] = "completed"
                    elif info.get("out_of_lane"):
                        result["outcome"] = "out_of_lane"
                    elif any(v for k, v in info.items()
                             if k not in ("min_ttc", "timeout", "out_of_lane") and v):
                        result["outcome"] = "collision"
                    else:
                        result["outcome"] = "completed"
                    outcome_set = True
            except Exception:
                break

        if not outcome_set:
            result["outcome"] = "run_timeout"

        if p_bridge.is_alive():
            p_bridge.terminate()
        p_bridge.join(timeout=10)

    except Exception as e:
        result["outcome"] = f"error: {str(e)[:100]}"
        traceback.print_exc()

    result["duration_s"] = round(time.monotonic() - t_start, 1)
    return result


def patch_files_to_commit(commit_sha: str, files: list):
    if not files:
        return
    for f in files:
        r = subprocess.run(["git", "checkout", commit_sha, "--", f],
                           cwd=str(OP_ROOT), capture_output=True)
        status = "ok" if r.returncode == 0 else f"FAILED: {r.stderr.decode()[:60]}"
        print(f"  [git] {f} → {commit_sha[:8]}  {status}")


def restore_to_head(files: list):
    if not files:
        return
    for f in files:
        subprocess.run(["git", "checkout", "HEAD", "--", f],
                       cwd=str(OP_ROOT), capture_output=True)
    print("  [git] restored HEAD")


def print_table(results: list):
    W = 80
    print(f"\n{'='*W}")
    print("  OPENPILOT SMOKE TEST — RESULTS")
    print(f"{'='*W}")
    hdr = f"{'Scenario':<32} {'Commit':<10} {'Outcome':<14} {'TTC':>7} {'Dur':>7} {'TTE':>6}"
    print(hdr)
    print("-" * W)
    for r in results:
        sha = r["commit"][:8]
        ttc = f"{r['min_ttc']:.2f}s" if r.get("min_ttc") else "  —  "
        tte = f"{r['time_to_engage_s']:.0f}s" if r.get("time_to_engage_s") else "  —"
        print(f"{r['scenario']:<32} {sha:<10} {r['outcome']:<14} "
              f"{ttc:>7} {r['duration_s']:>5.0f}s {tte:>6}")
    print(f"{'='*W}")
    ok  = sum(1 for r in results if r["outcome"] in ("completed","out_of_lane","collision"))
    err = sum(1 for r in results if "error" in r["outcome"] or "timeout" in r["outcome"] or r["outcome"]=="no_engage")
    col = sum(1 for r in results if r["outcome"] == "collision")
    print(f"Total: {len(results)}  ok: {ok}  collision: {col}  no-engage/error: {err}")
    print("TTE = time-to-engage (seconds from bridge start to openpilot active)")


def build_commit_list(args) -> list[dict]:
    """
    Return a list of commit dicts: {sha, label, files, artifact_env}.
    Sources (in priority order):
      1. --sha <prefix>          single commit by SHA prefix
      2. --from-manifest         all usable commits from manifest
      3. default                 the two hardcoded COMMITS
    """
    if args.sha:
        full_sha = resolve_sha(args.sha)
        if not full_sha:
            print(f"ERROR: cannot resolve SHA '{args.sha}'")
            sys.exit(1)
        manifest = load_manifest()
        entry = manifest.get(full_sha, {})
        status = entry.get("status", "unknown")
        if status == "build_partial_no_acados":
            print(f"WARNING: {full_sha[:12]} is missing acados MPC libs — MPC control will not work")
        artifact_env = artifact_env_for_commit(full_sha)
        return [{"sha": full_sha, "label": full_sha[:12], "files": [],
                 "artifact_env": artifact_env}]

    if args.from_manifest:
        manifest = load_manifest()
        usable_statuses = {"python_only", "build_partial"}
        commits = []
        for sha, entry in manifest.items():
            if entry.get("status") in usable_statuses or entry.get("usable", False):
                artifact_env = artifact_env_for_commit(sha) if entry.get("status") == "build_partial" else {}
                commits.append({"sha": sha, "label": sha[:12], "files": [],
                                 "artifact_env": artifact_env})
        print(f"  Loaded {len(commits)} usable commits from manifest")
        return commits

    # Default: two hardcoded commits
    result = []
    for c in COMMITS:
        artifact_env = artifact_env_for_commit(c["sha"])
        result.append({**c, "artifact_env": artifact_env})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--scenarios", default=None,
                    help="Comma-separated class names (default: all 13 for HEAD, 3 for old)")
    ap.add_argument("--commits", default="both", choices=["head","old","both"],
                    help="Which hardcoded commits to run (ignored with --sha/--from-manifest)")
    ap.add_argument("--sha", default=None,
                    help="Run all scenarios for a single commit by SHA prefix")
    ap.add_argument("--from-manifest", action="store_true",
                    help="Run scenarios for all usable commits in the manifest")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    scenarios_override = None
    if args.scenarios:
        scenarios_override = [s.strip() for s in args.scenarios.split(",")]

    # Build commit list
    if not args.sha and not args.from_manifest:
        commits_to_run_base = COMMITS[:]
        if args.commits == "head":
            commits_to_run_base = [COMMITS[0]]
        elif args.commits == "old":
            commits_to_run_base = [COMMITS[1]]
        commits_to_run = []
        for c in commits_to_run_base:
            artifact_env = artifact_env_for_commit(c["sha"])
            commits_to_run.append({**c, "artifact_env": artifact_env})
    else:
        commits_to_run = build_commit_list(args)

    all_results = []

    for c in commits_to_run:
        sha = c["sha"]
        label = c.get("label", sha[:12])
        files = c.get("files", [])
        artifact_env = c.get("artifact_env", {})
        is_head = (sha.startswith("492ed7312"))

        if scenarios_override:
            scenarios = scenarios_override
        elif args.sha or args.from_manifest:
            scenarios = ALL_SCENARIOS
        else:
            scenarios = ALL_SCENARIOS if is_head else RADARD_SENSITIVE

        print(f"\n{'='*80}")
        print(f"  COMMIT: {sha[:12]}  {label}")
        if artifact_env:
            print(f"  Artifacts: injecting {len(artifact_env)} env vars (PYTHONPATH+LD_LIBRARY_PATH)")
        print(f"  Scenarios ({len(scenarios)}): {', '.join(s[:12] for s in scenarios)}")
        print(f"{'='*80}")

        patch_files_to_commit(sha if not is_head else "HEAD", files)

        xvfb = start_xvfb(97)
        print(f"  Xvfb :97 started (pid={xvfb.pid})")

        for sname in scenarios:
            print(f"\n  ▶ {sname} [{sha[:8]}]  (fresh manager)", flush=True)

            # Fresh manager per scenario — guarantees clean process state and
            # avoids locationd/hardwared getting stuck after previous bridge disconnect.
            manager = start_manager(extra_env=artifact_env)
            print(f"    Manager started (pid={manager.pid}), warming up 8s…")
            time.sleep(8)

            try:
                r = run_one(sname, sha[:12], args.duration, extra_env=artifact_env)
            except Exception as e:
                r = {"scenario": sname, "commit": sha[:12], "outcome": f"error:{e}",
                     "min_ttc": None, "duration_s": 0.0, "engaged": False,
                     "time_to_engage_s": None}
                traceback.print_exc()
            finally:
                manager.terminate()
                try: manager.wait(10)
                except: manager.kill()

            ttc = f"{r['min_ttc']:.2f}s" if r.get("min_ttc") else "—"
            tte = f"{r.get('time_to_engage_s','—')}s" if r.get("time_to_engage_s") else "—"
            print(f"     → outcome={r['outcome']}  ttc={ttc}  "
                  f"dur={r['duration_s']:.0f}s  tte={tte}", flush=True)
            all_results.append(r)
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")

        xvfb.terminate()
        restore_to_head(files)
        print(f"\n  Commit {sha[:8]} done.", flush=True)

    print_table(all_results)
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
