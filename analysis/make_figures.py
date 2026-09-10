"""Figures: the severity curve of Fig. 1 (right) and the re-anchoring curve
of Fig. 3, from proxima/injury.py and results/rq4/fig3_reanchoring.csv.

Usage:  python analysis/make_figures.py [--rq4 results/rq4/fig3_reanchoring.csv]
Writes results/figures/fig1_severity_curve.{png,pdf} and fig3_reanchoring.{png,pdf}.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                      # noqa: E402
from proxima.injury import occupant                   # noqa: E402

CAMPAIGN_REPLAYS = 15000


def fig1(outdir, plt):
    s_kmh = np.linspace(0, 100, 401)
    iota = occupant(s_kmh / 3.6)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(s_kmh[1:], iota[1:], color="black", lw=1.5)
    ax.plot([0], [0], "o", color="black", ms=4)
    ax.annotate(r"$\iota(0)=0$", (0, 0), xytext=(4, 0.08), fontsize=8)
    v47 = float(occupant(47 / 3.6))
    ax.plot([47], [v47], "o", color="tab:red", ms=4)
    ax.annotate(f"47 km/h $\\to$ $\\iota$ = {v47:.2f}", (47, v47), xytext=(50, 0.12),
                fontsize=8, color="tab:red")
    s50 = 60.68            # logit = 0 at s = 6.068/0.100 km/h
    ax.plot([s50], [0.5], "s", color="tab:blue", ms=4)
    ax.annotate(f"$\\iota$ = 0.5 near {s50:.0f} km/h", (s50, 0.5), xytext=(20, 0.6),
                fontsize=8, color="tab:blue")
    ax.set_xlabel("impact speed at contact (km/h)")
    ax.set_ylabel(r"severity $\iota$")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_title("Kusano-Gabler MAIS2+ curve (shipped $\\iota$)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig1_severity_curve.{ext}"), dpi=200)
    plt.close(fig)


def fig3(path, outdir, plt):
    rows = list(csv.DictReader(open(path)))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    colors = {"openpilot->transfuser": "tab:blue", "transfuser->openpilot": "tab:orange"}
    pretty = {"openpilot->transfuser": r"openpilot$\to$TransFuser",
              "transfuser->openpilot": r"TransFuser$\to$openpilot"}
    for d, col in colors.items():
        for lt, M, ls, fill in (("M30", 30, "-", True), ("M100", 100, "--", False)):
            sel = [r for r in rows if r["direction"] == d and r["labels"] == lt]
            x = [int(r["replays"]) for r in sel]
            y = [float(r["cov90"]) for r in sel]
            e = [float(r["cov90_sd"]) for r in sel]
            ax.errorbar(x, y, yerr=e, color=col, ls=ls, marker="o",
                        mfc=col if fill else "white", ms=5, capsize=2,
                        label=f"{pretty[d]}, M = {M}")
        z = [r for r in rows if r["direction"] == d and r["labels"] == "zero-shot"]
        if z:
            ax.plot([1e2 * 0.7], [float(z[0]["cov90"])], "*", color=col, ms=11,
                    label=f"{pretty[d]} zero-shot")
    ax.axhline(0.9, color="gray", lw=0.8)
    ax.text(150, 0.905, "target 90%", fontsize=7, color="gray")
    ax.axvline(CAMPAIGN_REPLAYS, color="gray", lw=0.8, ls="-.")
    ax.text(CAMPAIGN_REPLAYS * 0.55, 0.62, "full reference\ncampaign", fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set_xlim(50, 3e4)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel(r"calibration cost (replays = $n \times M$)")
    ax.set_ylabel("coverage of 90% intervals")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig3_reanchoring.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq4", default=os.path.join(C.RESULTS, "rq4", "fig3_reanchoring.csv"))
    ap.add_argument("--outdir", default=os.path.join(C.RESULTS, "figures"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.outdir, exist_ok=True)
    fig1(a.outdir, plt)
    print(f"wrote {a.outdir}/fig1_severity_curve.{{png,pdf}}")
    if os.path.exists(a.rq4):
        fig3(a.rq4, a.outdir, plt)
        print(f"wrote {a.outdir}/fig3_reanchoring.{{png,pdf}}")
    else:
        print(f"skip Fig. 3: {a.rq4} not found (run analysis/rq4_portability.py first)")


if __name__ == "__main__":
    main()
