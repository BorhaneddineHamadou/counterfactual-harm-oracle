"""Shared data access for every analysis script.

`load_subject("openpilot" | "transfuser")` returns one dict `S` holding the
whole derived campaign record of that subject in the layout the field-tier
drivers use. Every script in analysis/ goes through this module, so the
data files are read in exactly one place.

Keys of S (n = 150 references, sid order):
  sids, y, se, crash_frac, template, NC (non-collision mask of run 0),
  X41, names41           reference features of run 0 (41)
  X2       (150, 142)    run-0 features 71 + 71 traj, the test inputs
  Xa2      (750, 142)    all nominal executions; aug_row / nom_run / nom_contact
  Xr2      (15000, 142)  all reference replays; rep_scn (scenario row), rep_m,
                         rep_contact, rep_dv, rep_injury, rep_dlat, rep_gain
  H_rep    (150, 100)    per-replay injury matrix (nan where a replay is absent)
  inj_by_ref             {row: injuries in replay order} (escalation)
  nominal_runs           list of dict rows of nominal_runs.csv (750)
  hold                   {sid: [(contact, impact_speed)]} for runs 1..4 (600)
  hold_paper             same, without the one openpilot run (s50_r2) that was
                         re-executed by a top-up job (the 599-run subset an
                         earlier draft of the paper used; all numbers identical)
  n_seeds, p_in_tail, tau_rule   the subject's field-tier settings
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SUBJECTS = ("openpilot", "transfuser")
LABEL = {"openpilot": "openpilot", "transfuser": "TransFuser"}

# field-tier settings that differ between the two subjects (configs/field_tier.json)
SETTINGS = {
    "openpilot": dict(n_seeds=6, inner_seeds=2, p_in_tail=True, tau_rule="declared"),
    "transfuser": dict(n_seeds=3, inner_seeds=1, p_in_tail=False, tau_rule="per_fold_max"),
}


def data_path(subject, *parts):
    return os.path.join(DATA, subject, *parts)


def results_path(*parts):
    p = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def folds_for(rep, n=150):
    """The paper's scenario-level folds for CV repeat `rep`."""
    return np.array_split(np.random.default_rng(rep).permutation(n), 5)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("template", "outcome") or v == "":
                continue
            r[k] = float(v) if "." in v or "e" in v else int(v)
    return rows


def load_subject(subject):
    L = np.load(data_path(subject, "reference_labels.npz"), allow_pickle=True)
    N = np.load(data_path(subject, "features_nominal.npz"), allow_pickle=True)
    R = np.load(data_path(subject, "features_replays.npz"))
    sids = L["sids"].astype(int)
    sid_row = {int(s): i for i, s in enumerate(sids)}
    y = L["y"].astype(float)

    Xa2 = np.hstack([N["X71"], N["Xtraj"]])
    aug_row = np.array([sid_row[int(s)] for s in N["sid"]])
    nom_run = N["run"].astype(int)
    r0 = nom_run == 0
    order = np.argsort(aug_row[r0])
    X2 = Xa2[r0][order]
    assert (aug_row[r0][order] == np.arange(len(sids))).all()

    rep_sid = L["rep_sid"].astype(int)
    rep_m = L["rep_m"].astype(int)
    assert (R["rep_sid"].astype(int) == rep_sid).all() and (R["rep_m"].astype(int) == rep_m).all()
    Xr2 = np.hstack([R["X71"], R["Xtraj"]])
    rep_scn = np.array([sid_row[s] for s in rep_sid])
    rep_injury = L["rep_injury"].astype(float)
    inj_by_ref = {}
    for i in range(len(y)):
        m = rep_scn == i
        inj_by_ref[i] = rep_injury[m][np.argsort(rep_m[m])]

    nominal_runs = read_csv(data_path(subject, "nominal_runs.csv"))
    hold = {int(s): [] for s in sids}
    hold_paper = {int(s): [] for s in sids}       # 599/600: re-executed run excluded
    for r in nominal_runs:
        if r["run"] != 0:
            hold[r["sid"]].append((bool(r["contact"]), float(r["impact_speed_ms"])))
            if r.get("in_paper_heldout", 1) == 1:
                hold_paper[r["sid"]].append((bool(r["contact"]), float(r["impact_speed_ms"])))

    S = dict(name=subject, label=LABEL[subject], sids=sids, sid_row=sid_row,
             y=y, se=L["se"].astype(float), crash_frac=L["crash_frac"].astype(float),
             template=L["template"].astype(str), NC=~L["contact_nom"].astype(bool),
             X41=L["X"], names41=[str(n) for n in L["feature_names"]],
             names71=[str(n) for n in N["names71"]], names_traj=[str(n) for n in N["names_traj"]],
             X2=X2, Xa2=Xa2, aug_row=aug_row, nom_run=nom_run,
             nom_contact=N["contact"].astype(bool),
             Xr2=Xr2, rep_scn=rep_scn, rep_m=rep_m, rep_sid=rep_sid,
             rep_contact=L["rep_contact"].astype(bool), rep_dv=L["rep_dv"].astype(float),
             rep_injury=rep_injury, rep_dlat=L["rep_dlat"].astype(float),
             rep_gain=L["rep_gain"].astype(float), H_rep=L["H_rep"].astype(float),
             inj_by_ref=inj_by_ref, nominal_runs=nominal_runs, hold=hold,
             hold_paper=hold_paper,
             global_seed=int(L["global_seed"]))
    S.update(SETTINGS[subject])
    return S


def baselines(S):
    """The single-run baseline orderings of Table 1 / Table 2, higher = more
    dangerous, from the run-0 features."""
    X, fn = S["X41"], S["names41"]
    return {
        "binary verdict": (~S["NC"]).astype(float),
        "min TTC": -X[:, fn.index("ttc_min")],
        "min clearance": -X[:, fn.index("min_clearance")],
        "realized impact speed": X[:, fn.index("dv_realized")],
    }


def crime_best(S):
    """The strongest third-party CriMe measure per subject (Table 1 row
    'Best of 35 CriMe measures'), as scored in results/crime/crime_vs_harm.json:
    CPI (worst-of-run folding) on openpilot, WTTC at the criticality instant
    on TransFuser, each in its declared criticality direction."""
    rows = json.load(open(os.path.join(DATA, "crime", f"{S['name']}_measures.json")))
    bysid = {r["sid"]: r for r in rows if "sid" in r}
    key, sign = (("CPI_worst", +1) if S["name"] == "openpilot" else ("WTTC_crit", -1))
    v = np.array([bysid[int(s)].get(key, np.nan) if int(s) in bysid else np.nan
                  for s in S["sids"]], float)
    fin = np.isfinite(v)
    v = np.where(np.isposinf(v), np.nanmax(v[fin]) + 1, v)
    v = np.where(np.isneginf(v), np.nanmin(v[fin]) - 1, v)
    return sign * v, key


def load_oof(subject, tag="field_tier_oof"):
    """Stored out-of-fold field-tier scores (results/rq1/<subject>_<tag>.npz)."""
    return np.load(results_path("rq1", f"{subject}_{tag}.npz"))


def fmt(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
