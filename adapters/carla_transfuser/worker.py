"""Jobs-file worker: runs its slice of jobs, each in a fresh run_one.py
subprocess with a hard wall-clock cap; on a stall or error the CARLA server
is recycled and the job retried once.

The subprocess isolation exists because CARLA RPCs can block past the client
timeout (campaign 665156: every task eventually froze in apply_settings) and
because the long-lived client+torch process leaks (task 7 OOM at 32G). A
fresh process per job caps both failure modes at one job.

Usage: worker.py JOBS_FILE [TASK_ID NUM_TASKS]
Env: CARLA_PORT (default 2000+task*50); CARLA_EXTERNAL=1 means the server
is managed outside (slurm restart loop) and recycling = pkill by port;
TF_JOB_TIMEOUT seconds per attempt (default 1500).
"""
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _kill_external(port):
    subprocess.call(["pkill", "-u", os.environ.get("USER", ""),
                     "-f", f"carla-rpc-port={port}"])


def _attempt(job, port, timeout):
    env = dict(os.environ)
    if job.get("perturb"):
        env["CH_PERTURB"] = json.dumps(job["perturb"])
    else:
        env.pop("CH_PERTURB", None)
    ofile = job["trace_out"] + ".outcome.json"
    env["CH_OUTCOME_OUT"] = ofile
    if os.path.exists(ofile):
        os.remove(ofile)
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "run_one.py"), json.dumps(job)],
        env=env, preexec_fn=os.setsid)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return None                              # stall
    if os.path.exists(ofile):
        with open(ofile) as f:
            outcome = json.load(f)["outcome"]
        os.remove(ofile)
        return outcome
    return f"error:exit{proc.returncode}"


def main():
    jobs_file = sys.argv[1]
    task = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ntask = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    port = int(os.environ.get("CARLA_PORT", 2000 + task * 50))
    timeout = float(os.environ.get("TF_JOB_TIMEOUT", 1500))
    jobs = [json.loads(l) for l in open(jobs_file)]
    mine = jobs[task::ntask]
    res_path = jobs_file.replace(".jsonl", f"_task{task}.results.jsonl")

    external = bool(os.environ.get("CARLA_EXTERNAL"))
    server = None
    if not external:
        import run_one
        server = run_one.start_server(port)
        time.sleep(25.0)

    def recycle():
        nonlocal server
        if external:
            _kill_external(port)     # slurm loop reboots it in ~5s
            time.sleep(60.0)         # UE4 boot ~45s
        else:
            import run_one
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            server = run_one.start_server(port)
            time.sleep(25.0)

    try:
        for job in mine:
            if os.path.exists(job["trace_out"]):
                continue
            t0 = time.time()
            outcome = _attempt(job, port, timeout)
            if outcome is None or str(outcome).startswith("error:"):
                print(f"[worker] {job['job_id']} "
                      f"{'stalled' if outcome is None else outcome}; "
                      "recycling server", flush=True)
                recycle()
                retry = _attempt(job, port, timeout)
                if retry is not None:
                    outcome = retry
                elif outcome is None:
                    outcome = "error:stall"
                    recycle()        # don't hand a wedged server to next job
            with open(res_path, "a") as f:
                f.write(json.dumps({
                    "job_id": job["job_id"], "sid": job.get("sid"),
                    "outcome": str(outcome),
                    "wall_s": round(time.time() - t0, 1),
                    "trace_exists": os.path.exists(job["trace_out"])})
                    + "\n")
            print(f"[worker] {job['job_id']}: {outcome} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        if server is not None:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
