"""The axiomatic falsification battery (paper Sec. 4.1, result in Sec. 5.2).

Metamorphic properties any harm measure must satisfy, tested on
constructed trace pairs in the synthetic pilot world (adapters/pilot.py),
under the declared kernel and the campaign's M = 100:

  A1 monotonicity  faster closing at equal geometry never scores lower;
  A2 invariance    appending benign post-encounter time changes nothing;
  A3 continuity    an eps-graze and an eps-miss receive adjacent scores
                   (the pair is found by bisection on the initial gap).

Violation tolerances include the Monte-Carlo error of both labels, so a
violation is a real effect, not sampling noise. The paper reports the
battery over three seeds: 219 cases, zero violations.

Usage:  python analysis/axioms.py [--seeds 20260722,1,987654] [--M 100]
            [--n-mono 40] [--n-cont 20] [--tag _rerun]
Runtime: a few minutes per seed on one core.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.pilot import EgoParams, PilotAdapter, axiom_battery   # noqa: E402
from analysis import common as C                                     # noqa: E402
from proxima.kernel import Kernel                                     # noqa: E402

PAPER_SEEDS = (20260722, 1, 987654)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in PAPER_SEEDS))
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--n-mono", type=int, default=40,
                    help="A1 cases (A2 runs n_mono/2)")
    ap.add_argument("--n-cont", type=int, default=20,
                    help="A3 boundary searches (some do not survive the "
                         "bisection precondition)")
    ap.add_argument("--tag", default="_rerun")
    a = ap.parse_args()
    kernel = Kernel()
    total_n = total_v = 0
    for seed in (int(s) for s in a.seeds.split(",")):
        t0 = time.time()
        ax = axiom_battery(PilotAdapter(EgoParams()), kernel, a.M, seed=seed,
                           n_mono=a.n_mono, n_cont=a.n_cont)
        n = sum(v["n"] for v in ax["summary"].values())
        v = sum(v["violations"] for v in ax["summary"].values())
        total_n += n
        total_v += v
        for name, s in ax["summary"].items():
            print(f"  seed {seed} {name:18s}: {s['violations']}/{s['n']} violations")
        print(f"seed {seed}: {v}/{n} violations ({time.time()-t0:.0f}s)", flush=True)
        out = C.results_path("axioms", f"axioms_M{a.M}_s{seed}{a.tag}.json")
        json.dump({"M": a.M, "seed": seed, "kernel": kernel.__dict__, **ax},
                  open(out, "w"), indent=1, default=float)
        print(f"  -> {out}")
    print(f"\nTOTAL: {total_v}/{total_n} violations   (paper: 0/219 over the three seeds)")


if __name__ == "__main__":
    main()
