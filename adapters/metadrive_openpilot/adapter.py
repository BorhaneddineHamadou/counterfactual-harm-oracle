"""File-backed SimulatorAdapter over the SLURM trace store.

Execution on this stack is asynchronous (SLURM array workers write .npz
traces); this adapter closes the loop by presenting the finished store
through the synchronous SimulatorAdapter interface, so tier1.label_all, the
ensemble and the validation stack run unchanged on real openpilot+MetaDrive
data. Runs on the login node (no MetaDrive/openpilot imports).

Contract with campaign/openpilot_campaign.py (store = data/openpilot):
  gen-nominal writes jobs/nominal_jobs.jsonl (job_id = s{sid}_r{run})
              and campaign_scenarios.json (the concrete suite);
  gen-tier1   writes jobs/tier1_jobs.jsonl (job_id = s{sid}_r0_b{m}),
              tier1_refs.json (t_star per reference, computed at gen time)
              and tier1_manifest.json (kernel/M/horizon actually sampled).

branch() does NOT sample the kernel: the replays on disk already embody eps
drawn at gen-tier1 time (harvest.branch_jobs seeds its rng with
[seed, 999]). It instead validates that the requested (kernel, M, t_star)
match the manifest/refs -- a mismatch means the store was generated for a
different measure and the labels would be silently wrong -- then harvests
(contact, dv) from the replay traces.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from proxima.kernel import Kernel  # noqa: E402
from proxima.sim import SimulatorAdapter  # noqa: E402
from proxima.trace import Trace  # noqa: E402

from . import harvest as hv  # noqa: E402

T_STAR_TOL_S = 2 * hv.DT


class StoreIncompleteError(RuntimeError):
    """Raised in strict mode when required traces are missing on disk."""

    def __init__(self, what, missing):
        self.missing = missing
        preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        super().__init__(f"{what}: {len(missing)} trace(s) missing "
                         f"({preview})")


class StoreMismatchError(RuntimeError):
    """Raised when the store was generated for a different suite/kernel."""


def load_suite(out_dir):
    """The concrete suite the store was generated for:
    list of (scenario_cls, params)."""
    rows = json.load(open(os.path.join(out_dir, "campaign_scenarios.json")))
    return [(r["scls"], r["params"]) for r in rows]


class MetaDriveBatchAdapter(SimulatorAdapter):

    def __init__(self, out_dir, strict=True):
        self.out_dir = out_dir
        self.strict = strict
        self.tr_tier1 = os.path.join(out_dir, "traces", "tier1")
        self._nominal = [json.loads(l) for l in open(
            os.path.join(out_dir, "jobs", "nominal_jobs.jsonl"))]
        self._refs = None      # job_id -> ref record (t_star, ...)
        self._manifest = None

    # ------------------------------------------------------------- nominal
    def run_suite(self, scenarios, k, global_seed):
        jobs = self._nominal
        if len(jobs) != len(scenarios) * k:
            raise StoreMismatchError(
                f"store has {len(jobs)} nominal jobs, caller asked for "
                f"{len(scenarios)} scenarios x {k} runs")
        for job in jobs:
            sid, r = job["sid"], job["run_idx"]
            scls, params = scenarios[sid]
            if job["scenario_cls"] != scls:
                raise StoreMismatchError(
                    f"sid {sid}: store has {job['scenario_cls']}, "
                    f"caller has {scls}")
            if any(abs(job["params"][p] - params[p]) > 1e-9 for p in params):
                raise StoreMismatchError(f"sid {sid}: params differ")
            if job["seed"] != global_seed + sid * 1000 + r:
                raise StoreMismatchError(
                    f"{job['job_id']}: seed {job['seed']} does not follow "
                    f"from global_seed {global_seed}")
        traces, missing = [], []
        for job in jobs:
            if not os.path.exists(job["trace_out"]):
                missing.append(job["job_id"])
                continue
            traces.append(hv.to_trace(self._abs(job["trace_out"]), sid=job["sid"],
                                      run_idx=job["run_idx"],
                                      global_seed=global_seed))
        if missing:
            if self.strict:
                raise StoreIncompleteError("run_suite", missing)
            print(f"run_suite: skipping {len(missing)} missing trace(s)",
                  file=sys.stderr)
        return traces

    # -------------------------------------------------------------- tier 1
    def _abs(self, p):
        return p if os.path.isabs(p) else os.path.join(self.out_dir, p)

    def _load_tier1_meta(self):
        if self._refs is None:
            refs = json.load(open(os.path.join(self.out_dir,
                                               "tier1_refs.json")))
            self._refs = {r["job_id"]: r for r in refs}
            self._manifest = json.load(open(os.path.join(
                self.out_dir, "tier1_manifest.json")))

    def branch(self, trace: Trace, t_star, kernel: Kernel, M):
        self._load_tier1_meta()
        want, have = asdict(kernel), self._manifest["kernel"]
        if any(abs(want[f] - have[f]) > 1e-9 for f in want):
            raise StoreMismatchError(
                "replays were generated under a different kernel; "
                "regenerate gen-tier1 for this measure")
        if M != self._manifest["M"]:
            raise StoreMismatchError(
                f"replays were generated with M={self._manifest['M']}, "
                f"caller asked for M={M} (se would be miscomputed)")
        job_id = f"s{trace.sid}_r{trace.run_idx}"
        ref = self._refs.get(job_id)
        if ref is None:
            raise StoreMismatchError(
                f"{job_id} is not a tier-1 reference (only run_idx 0 traces "
                "are branched)")
        if abs(t_star - ref["t_star"]) > T_STAR_TOL_S:
            raise StoreMismatchError(
                f"{job_id}: caller t_star={t_star:.2f}s but replays were "
                f"branched at {ref['t_star']:.2f}s (nominal trace changed "
                "since gen-tier1?)")
        contact, dv, missing = [], [], []
        for m in range(M):
            p = os.path.join(self.tr_tier1, f"{job_id}_b{m}.npz")
            if not os.path.exists(p):
                missing.append(f"{job_id}_b{m}")
                continue
            tr = hv.to_trace(p, sid=trace.sid)
            contact.append(tr.contact)
            dv.append(tr.dv)
        if missing:
            if self.strict:
                raise StoreIncompleteError(f"branch({job_id})", missing)
            print(f"branch({job_id}): {len(missing)}/{M} replays missing",
                  file=sys.stderr)
        return np.asarray(contact, bool), np.asarray(dv, float)

    # -------------------------------------------------------------- status
    def missing(self):
        """Missing trace files, for progress reporting / gating."""
        nom = [j["job_id"] for j in self._nominal
               if not os.path.exists(j["trace_out"])]
        t1_path = os.path.join(self.out_dir, "campaign_tier1_jobs.jsonl")
        t1 = []
        if os.path.exists(t1_path):
            for l in open(t1_path):
                j = json.loads(l)
                if not os.path.exists(j["trace_out"]):
                    t1.append(j["job_id"])
        return {"nominal": nom, "tier1": t1}
