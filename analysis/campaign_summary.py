"""Sec. 5.1 ("Two Regimes, One Oracle"), the worked examples and the
campaign counts, straight from the campaign record. No training.

Reproduces:
  * both campaigns: 750 nominal runs and 15,000 reference replays each,
    every label on its full M=100 replays, mean Monte-Carlo s.e. 0.0011
    (openpilot) and 0.0003 (TransFuser);
  * openpilot: mean H = 0.008, 28 of 150 references nonzero, run-0 contact
    rates by template from 27% (cut-in) to 0% (crossing), the top 15 runs
    hold 88% of the suite's harm and the top one 11%;
  * TransFuser: oncoming drift 52% contact over all 750 nominal runs and
    89% of the suite's harm, cut-in 20%, the two lead templates zero
    contacts; in 99% of the oncoming replay contacts the ego is still
    moving (footnote 1);
  * Figure 1 (sid 40): stopped car approached at 46 km/h, halts 4.4 m
    short, 37 of 100 replays rear-end at ~47 km/h, iota ~ 0.20, H = 0.07,
    6% of the suite's harm;
  * the abstract's inversion pair: sid 55 (29 m, 2.2 s; 23/100 replays
    crash at 44 km/h) vs sid 16 (5.4 m, 0.4 s; 0/100);
  * RQ2 counts: run-0 verdict calls 13 / 21 collisions, 23 / 31 scenarios
    crash at least once more in runs 1-4, 600 held-out executions per subject;
  * reference-campaign cost, 800 and 1,400 GPU-hours (configs/measure.json).

Usage:  python analysis/campaign_summary.py
Output: results/summary/campaign_summary.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C            # noqa: E402

KMH = 3.6


def example(S, sid):
    i = S["sid_row"][sid]
    fn = S["names41"]
    m = S["rep_sid"] == sid
    c = S["rep_contact"][m]
    dv = S["rep_dv"][m][c]
    return {"sid": sid, "template": str(S["template"][i]),
            "ttc_min_s": float(S["X41"][i, fn.index("ttc_min")]),
            "min_clearance_m": float(S["X41"][i, fn.index("min_clearance")]),
            "speed_at_crit_kmh": float(S["X41"][i, fn.index("speed_at_crit")] * KMH),
            "run0_contact": bool(~S["NC"][i]),
            "replays": int(m.sum()), "replays_crashing": int(c.sum()),
            "median_impact_speed_kmh": float(np.median(dv) * KMH) if c.any() else 0.0,
            "mean_iota_of_crashes": float(S["rep_injury"][m][c].mean()) if c.any() else 0.0,
            "H": float(S["y"][i]), "share_of_suite_harm": float(S["y"][i] / S["y"].sum())}


def main():
    cfg = json.load(open(os.path.join(C.ROOT, "configs", "measure.json")))
    out = {}
    for subject in C.SUBJECTS:
        S = C.load_subject(subject)
        y, tmpl = S["y"], S["template"]
        nr = S["nominal_runs"]
        r0 = {t: [] for t in np.unique(tmpl)}
        allr = {t: [] for t in np.unique(tmpl)}
        for r in nr:
            allr[r["template"]].append(r["contact"])
            if r["run"] == 0:
                r0[r["template"]].append(r["contact"])
        ys = np.sort(y)[::-1]
        d = {
            "nominal_runs": len(nr), "reference_replays": int(len(S["rep_scn"])),
            "replays_per_reference_min": int(np.min(np.bincount(S["rep_scn"]))),
            "mean_label_mc_se": float(S["se"].mean()), "mean_H": float(y.mean()),
            "median_H": float(np.median(y)), "nonzero_references": int((y > 0).sum()),
            "top15_harm_share": float(ys[:15].sum() / y.sum()),
            "top1_harm_share": float(ys[0] / y.sum()),
            "run0_contact_rate_by_template": {t: float(np.mean(v)) for t, v in r0.items()},
            "all_runs_contact_rate_by_template": {t: float(np.mean(v)) for t, v in allr.items()},
            "harm_share_by_template": {t: float(y[tmpl == t].sum() / y.sum()) for t in np.unique(tmpl)},
            "run0_collisions": int((~S["NC"]).sum()),
            "scenarios_crashing_again": int(sum(any(c for c, _ in v) for v in S["hold"].values())),
            "held_out_runs": int(sum(len(v) for v in S["hold"].values())),
            "held_out_runs_excl_reexecuted": int(sum(len(v) for v in S["hold_paper"].values())),
            "gpu_hours_reference_campaign": cfg["subjects"][subject]["gpu_hours_reference_campaign"],
        }
        if subject == "transfuser":
            R = np.load(C.data_path(subject, "replay_outcomes.npz"))
            st = np.array([tmpl[S["sid_row"][int(s)]] for s in R["sid"]])
            m = (st == "oncoming_drift") & R["contact"].astype(bool)
            d["oncoming_replay_contacts"] = int(m.sum())
            d["oncoming_replay_contacts_ego_moving"] = float((R["v_ego"][m] > 0).mean())
        else:
            d["examples"] = {"figure1_sid40": example(S, 40), "abstract_sid55": example(S, 55),
                             "abstract_sid16": example(S, 16)}
        out[subject] = d

        print(f"\n================ {S['label']}")
        print(f"  {d['nominal_runs']} nominal runs, {d['reference_replays']} reference replays "
              f"(min {d['replays_per_reference_min']}/100 per reference)   [paper 750 / 15,000, all full]")
        print(f"  mean label MC s.e. {d['mean_label_mc_se']:.4f}   [paper "
              f"{'0.0011' if subject == 'openpilot' else '0.0003'}]")
        print(f"  mean H {d['mean_H']:.4f}, nonzero references {d['nonzero_references']}/150"
              + ("   [paper 0.008, 28]" if subject == "openpilot" else ""))
        print(f"  top-15 references hold {100 * d['top15_harm_share']:.0f}% of suite harm, top one "
              f"{100 * d['top1_harm_share']:.0f}%" + ("   [paper 88%, 11%]" if subject == "openpilot" else ""))
        print("  run-0 contact rate by template: " + ", ".join(
            f"{t} {100 * v:.0f}%" for t, v in d["run0_contact_rate_by_template"].items())
              + ("   [paper cut-in 27% ... crossing 0%]" if subject == "openpilot" else ""))
        if subject == "transfuser":
            print("  contact rate over all 750 runs: " + ", ".join(
                f"{t} {100 * v:.0f}%" for t, v in d["all_runs_contact_rate_by_template"].items())
                  + "   [paper oncoming 52%, cut-in 20%, lead templates 0]")
            print("  harm share by template: " + ", ".join(
                f"{t} {100 * v:.0f}%" for t, v in d["harm_share_by_template"].items())
                  + "   [paper oncoming 89%]")
            print(f"  oncoming replay contacts with the ego still moving: "
                  f"{100 * d['oncoming_replay_contacts_ego_moving']:.0f}% of "
                  f"{d['oncoming_replay_contacts']}   [paper 99%]")
        print(f"  run-0 verdict collisions {d['run0_collisions']}, scenarios crashing again in runs 1-4 "
              f"{d['scenarios_crashing_again']}, held-out executions {d['held_out_runs']} "
              f"({d['held_out_runs_excl_reexecuted']} without the re-executed run)   [paper "
              f"{'13, 23, 600' if subject == 'openpilot' else '21, 31, 600'}]")
        print(f"  reference campaign: {d['gpu_hours_reference_campaign']} GPU-hours   [paper "
              f"{'800' if subject == 'openpilot' else '1,400'}]")
        if subject == "openpilot":
            for k, e in d["examples"].items():
                print(f"  {k}: {e['template']}, TTC {e['ttc_min_s']:.2f} s, clearance "
                      f"{e['min_clearance_m']:.1f} m, speed at criticality {e['speed_at_crit_kmh']:.0f} km/h, "
                      f"run-0 contact {e['run0_contact']}, {e['replays_crashing']}/{e['replays']} replays crash, "
                      f"median impact {e['median_impact_speed_kmh']:.0f} km/h, iota|crash "
                      f"{e['mean_iota_of_crashes']:.2f}, H {e['H']:.3f} = {100 * e['share_of_suite_harm']:.0f}% "
                      f"of suite harm")
            print("   [paper: sid 40 = 46 km/h, 4.4 m, 37/100 at ~47 km/h, iota 0.20, H 0.07, 6%;"
                  " sid 55 = 29 m, 2.2 s, 23/100 at 44 km/h; sid 16 = 5.4 m, 0.4 s, 0/100]")
    fp = C.results_path("summary", "campaign_summary.json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\nsaved {fp}")


if __name__ == "__main__":
    main()
