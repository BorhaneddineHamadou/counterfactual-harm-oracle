"""Build the derived campaign record of one subject from its raw traces.

This is the bridge between the simulation layer (adapters/, campaign
drivers) and the analysis layer (analysis/). Given a campaign store

    <store>/campaign_scenarios.json, tier1_refs.json, tier1_manifest.json
    <store>/jobs/{nominal,tier1,rq2_a05,rq2_a20}_jobs.jsonl
    <store>/traces/{nominal,tier1,rq2_a05,rq2_a20}/*.npz

it writes the files every analysis script reads (data/<subject>/):

    reference_labels.npz   Tier-1 labels of the 150 references: y = mean
                           injury over the M=100 replays, MC standard error,
                           crash fraction; run-0 features (41); the full
                           replay table (sid, m, kernel draws, contact,
                           impact speed, injury) and the (150, 100) injury
                           matrix H_rep
    features_nominal.npz   750 nominal executions x (71 + 71 traj) features
    features_replays.npz   15,000 replays x (71 + 71 traj); phys_H stubbed
    nominal_runs.csv       one row per nominal execution (outcome, contact,
                           impact speed, TTC, clearance, ego speed, ...)
    replay_outcomes.npz    (sid, m, contact, dv, v_ego at contact) per replay
    rq2/features_a05.npz, rq2/features_a20.npz, rq2/H_by_scale.npz,
    rq2/labels.json        the kernel-rescaling arms of RQ3

Stages: labels | features | rq2 | all   (each idempotent)

Usage:
  python campaign/build_features.py openpilot|transfuser [--store DIR]
                                    [--stage all] [--jobs 16]
Runtime with 16 workers: ~10 min per subject (phys_H on nominal rows and
the 71+71 features on 15,000 replays dominate).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from proxima.features import extract as extract6                 # noqa: E402
from proxima.features_expanded import extract_expanded            # noqa: E402
from proxima.features_extra import extract_extra                  # noqa: E402
from proxima.features_traj import extract_traj                    # noqa: E402
from proxima.injury import get_curve                              # noqa: E402

HARVEST = {"openpilot": "adapters.metadrive_openpilot.harvest",
           "transfuser": "adapters.carla_transfuser.harvest"}
M_RQ2 = {"openpilot": 40, "transfuser": 20}
MIN_FRAC = 0.9
NT = 1e-4


def _hv(subject):
    import importlib
    return importlib.import_module(HARVEST[subject])


def _jobs(store, name):
    p = os.path.join(store, "jobs", f"{name}_jobs.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]


def _trace_path(store, job):
    p = job["trace_out"]
    return p if os.path.isabs(p) else os.path.join(store, p)


def _cmd_gas(path, T):
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    if "cmd_gas" not in cols:
        return None
    return z["data"][:, cols.index("cmd_gas")].astype(float)[:T]


def _ego_vmax(path):
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    d = z["data"]
    v = np.hypot(d[:, cols.index("ego_vx")].astype(float),
                 d[:, cols.index("ego_vy")].astype(float))
    return float(np.nanmax(v))


def _v_ego_at_contact(path):
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    d = z["data"]
    crash = np.flatnonzero(d[:, cols.index("crash")].astype(float) > 0.5)
    if not len(crash):
        return 0.0
    i = int(crash[0])
    return float(np.hypot(d[i, cols.index("ego_vx")], d[i, cols.index("ego_vy")]))


# --------------------------------------------------------------- workers
def _nominal_row(subject, store, job):
    hv = _hv(subject)
    p = _trace_path(store, job)
    tr = hv.to_trace(p, sid=job["sid"], run_idx=job["run_idx"], global_seed=job["seed"])
    f41, names41 = extract_expanded(tr, phys=True)
    f30, names30 = extract_extra(tr)
    ft, namest = extract_traj(tr, _cmd_gas(p, tr.T))
    f6 = extract6(tr)
    row = dict(sid=job["sid"], run=job["run_idx"], template=tr.template, seed=job["seed"],
               contact=int(tr.contact), impact_speed_ms=round(float(tr.dv), 4),
               ttc_min_s=round(float(f6[0]), 4), min_clearance_m=round(float(f6[1]), 4),
               speed_at_crit_ms=round(float(f6[2]), 4), ego_vmax_ms=round(_ego_vmax(p), 4),
               t_crit_s=round(float(tr.t_crit), 3), n_steps=int(tr.T))
    return (np.concatenate([f41, f30]), ft, names41 + names30, namest, row)


def _replay_row(subject, store, job):
    hv = _hv(subject)
    p = _trace_path(store, job)
    tr = hv.to_trace(p, sid=job["sid"])
    f41, _ = extract_expanded(tr, phys=False)       # phys_H stubbed, as shipped
    f30, _ = extract_extra(tr)
    ft, _ = extract_traj(tr, _cmd_gas(p, tr.T))
    m = int(job["job_id"].rsplit("_b", 1)[1])
    inj = float(get_curve(tr.actor_type)(tr.dv)) if tr.contact else 0.0
    return (job["sid"], m, np.concatenate([f41, f30]), ft, bool(tr.contact),
            float(tr.dv), inj, _v_ego_at_contact(p))


def _pmap(fn, items, n_jobs):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(*it) for it in items)


# ---------------------------------------------------------------- stages
def stage_features(subject, store, out, n_jobs):
    t0 = time.time()
    nom = _jobs(store, "nominal")
    nom = [j for j in nom if os.path.exists(_trace_path(store, j))]
    res = _pmap(_nominal_row, [(subject, store, j) for j in nom], n_jobs)
    X71 = np.stack([r[0] for r in res])
    Xt = np.stack([r[1] for r in res])
    names71, namest = res[0][2], res[0][3]
    rows = [r[4] for r in res]
    # outcomes recorded by the worker, if present
    outc = {}
    for f in glob.glob(os.path.join(store, "jobs", "nominal_jobs_task*.results.jsonl")) + \
            glob.glob(os.path.join(store, "*nominal*results.jsonl")):
        for line in open(f):
            r = json.loads(line)
            outc[r["job_id"]] = r.get("outcome", "")
    for r in rows:
        r["outcome"] = outc.get(f"s{r['sid']}_r{r['run']}", "")
        r["in_paper_heldout"] = 1 if r["run"] > 0 else ""
    with open(os.path.join(out, "nominal_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(out, "features_nominal.npz"),
                        sid=np.array([j["sid"] for j in nom]),
                        run=np.array([j["run_idx"] for j in nom]),
                        contact=np.array([bool(r["contact"]) for r in rows]),
                        X71=X71, Xtraj=Xt, names71=np.array(names71),
                        names_traj=np.array(namest))
    print(f"[{time.time()-t0:.0f}s] nominal features: {len(nom)} runs", flush=True)

    t1 = _jobs(store, "tier1")
    t1 = [j for j in t1 if os.path.exists(_trace_path(store, j))]
    res = _pmap(_replay_row, [(subject, store, j) for j in t1], n_jobs)
    rep_sid = np.array([r[0] for r in res])
    rep_m = np.array([r[1] for r in res])
    np.savez_compressed(os.path.join(out, "features_replays.npz"),
                        rep_sid=rep_sid, rep_m=rep_m,
                        X71=np.stack([r[2] for r in res]), Xtraj=np.stack([r[3] for r in res]))
    o = np.lexsort((rep_m, rep_sid))              # (sid, m) order
    refc = {r["sid"]: bool(r["contact"]) for r in json.load(open(os.path.join(store, "tier1_refs.json")))}
    np.savez_compressed(os.path.join(out, "replay_outcomes.npz"),
                        sid=rep_sid[o], m=rep_m[o], contact=np.array([r[4] for r in res])[o],
                        dv=np.array([r[5] for r in res])[o], v_ego=np.array([r[7] for r in res])[o],
                        ref_contact=np.array([refc.get(int(s), False) for s in rep_sid[o]]))
    # labels ride along with the replay pass
    perturb = {(j["sid"], int(j["job_id"].rsplit("_b", 1)[1])): j["perturb"] for j in t1}
    _write_labels(subject, store, out, nom, rows, X71, rep_sid, rep_m,
                  np.array([r[4] for r in res]), np.array([r[5] for r in res]),
                  np.array([r[6] for r in res]), perturb)
    print(f"[{time.time()-t0:.0f}s] replay features + labels: {len(t1)} replays", flush=True)


def _write_labels(subject, store, out, nom, rows, X71, rep_sid, rep_m, rep_contact,
                  rep_dv, rep_injury, perturb):
    refs = json.load(open(os.path.join(store, "tier1_refs.json")))
    M = json.load(open(os.path.join(store, "tier1_manifest.json")))["M"]
    sids = np.array([r["sid"] for r in refs])
    sid_row = {int(s): i for i, s in enumerate(sids)}
    H_rep = np.full((len(sids), M), np.nan)
    for s, m, h in zip(rep_sid, rep_m, rep_injury):
        H_rep[sid_row[int(s)], m] = h
    n_rep = np.sum(np.isfinite(H_rep), axis=1)
    keep = n_rep >= MIN_FRAC * M
    if not keep.all():
        print(f"  WARNING {int((~keep).sum())} references below {MIN_FRAC:.0%} replay "
              f"completeness are kept with nan labels")
    y = np.nanmean(H_rep, axis=1)
    se = np.nanstd(H_rep, axis=1, ddof=1) / np.sqrt(n_rep)
    crash_frac = np.array([np.mean(rep_contact[rep_sid == s]) for s in sids])
    r0 = {(r["sid"], r["run"]): i for i, r in enumerate(rows)}
    X = np.stack([X71[r0[(int(s), 0)], :41] for s in sids])
    contact_nom = np.array([bool(rows[r0[(int(s), 0)]]["contact"]) for s in sids])
    template = np.array([rows[r0[(int(s), 0)]]["template"] for s in sids])
    names41 = [str(n) for n in np.load(os.path.join(out, "features_nominal.npz"))["names71"][:41]]
    seed0 = next(j["seed"] for j in nom if j["sid"] == 0 and j["run_idx"] == 0)
    np.savez_compressed(os.path.join(out, "reference_labels.npz"),
                        sids=sids, y=y, se=se, crash_frac=crash_frac, contact_nom=contact_nom,
                        template=template, X=X, feature_names=np.array(names41), H_rep=H_rep,
                        rep_sid=rep_sid, rep_m=rep_m,
                        rep_dlat=np.array([perturb[(int(s), int(m))]["dlat_s"] for s, m in zip(rep_sid, rep_m)]),
                        rep_gain=np.array([perturb[(int(s), int(m))]["brake_gain"] for s, m in zip(rep_sid, rep_m)]),
                        rep_contact=rep_contact, rep_dv=rep_dv, rep_injury=rep_injury,
                        global_seed=np.int64(seed0))


def stage_rq2(subject, store, out, n_jobs):
    """Kernel-rescaling arms: features of the alpha replays and the labels
    H_by_scale (alpha 0.5 | 1 CRN subset | 1 full | 2) with the paper's
    completeness rule."""
    from scipy.stats import spearmanr
    t0 = time.time()
    os.makedirs(os.path.join(out, "rq2"), exist_ok=True)
    L = np.load(os.path.join(out, "reference_labels.npz"), allow_pickle=True)
    sids = L["sids"].astype(int)
    M = M_RQ2[subject]
    H = {}
    Hfull = dict(zip(sids, L["y"]))
    crn = {}
    for s, m, h in zip(L["rep_sid"], L["rep_m"], L["rep_injury"]):
        if m < M:
            crn.setdefault(int(s), []).append(h)
    H["1_crn"] = crn
    counts = {}
    for tag, alpha in (("a05", "0.5"), ("a20", "2.0")):
        jobs = _jobs(store, f"rq2_{tag}")
        jobs = [j for j in jobs if os.path.exists(_trace_path(store, j))]
        res = _pmap(_replay_row, [(subject, store, j) for j in jobs], n_jobs)
        np.savez_compressed(os.path.join(out, "rq2", f"features_{tag}.npz"),
                            rep_sid=np.array([r[0] for r in res]), rep_m=np.array([r[1] for r in res]),
                            X71=np.stack([r[2] for r in res]), Xtraj=np.stack([r[3] for r in res]))
        d = {}
        for r in res:
            d.setdefault(int(r[0]), []).append(r[6])
        H[alpha] = d
        print(f"[{time.time()-t0:.0f}s] rq2 {tag}: {len(res)} replays", flush=True)
    keep, cols = [], {"1_crn": [], "1_full": [], "0.5": [], "2.0": []}
    dropped = {k: 0 for k in cols}
    for s in sids:
        ok = True
        for k in ("1_crn", "0.5", "2.0"):
            if len(H[k].get(int(s), [])) < MIN_FRAC * M:
                dropped[k] += 1
                ok = False
                break
        if not ok:
            continue
        keep.append(int(s))
        cols["1_crn"].append(np.mean(H["1_crn"][int(s)]))
        cols["1_full"].append(Hfull[int(s)])
        cols["0.5"].append(np.mean(H["0.5"][int(s)]))
        cols["2.0"].append(np.mean(H["2.0"][int(s)]))
    cols = {k: np.array(v) for k, v in cols.items()}

    def rank(a, b):
        nt = (cols[a] > NT) | (cols[b] > NT)
        return {"spearman_all": float(spearmanr(cols[a], cols[b]).statistic),
                "spearman_nontrivial": float(spearmanr(cols[a][nt], cols[b][nt]).statistic),
                "n_nontrivial": int(nt.sum())}
    res = {"n_refs": len(keep), "M_rq2": M, "dropped": dropped,
           "mean_H": {k: float(v.mean()) for k, v in cols.items()},
           "nonzero_frac": {k: float((v > NT).mean()) for k, v in cols.items()},
           "rank_crn": {"0.5_vs_1": rank("0.5", "1_crn"), "1_vs_2": rank("1_crn", "2.0"),
                        "0.5_vs_2": rank("0.5", "2.0")},
           "rank_vs_full_ref": {"0.5_vs_1full": rank("0.5", "1_full"),
                                "2.0_vs_1full": rank("2.0", "1_full")}}
    json.dump(res, open(os.path.join(out, "rq2", "labels.json"), "w"), indent=2)
    np.savez(os.path.join(out, "rq2", "H_by_scale.npz"), sids=np.array(keep),
             **{f"H_{k}": v for k, v in cols.items()})
    print(f"[{time.time()-t0:.0f}s] rq2 labels: {len(keep)}/{len(sids)} refs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", choices=("openpilot", "transfuser"))
    ap.add_argument("--store", default=None, help="campaign store (default data/<subject>)")
    ap.add_argument("--out", default=None, help="output dir (default = store)")
    ap.add_argument("--stage", default="all", choices=("all", "features", "rq2"))
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    store = a.store or os.path.join(BASE, "data", a.subject)
    out = a.out or store
    os.makedirs(out, exist_ok=True)
    if a.stage in ("all", "features"):
        stage_features(a.subject, store, out, a.jobs)
    if a.stage in ("all", "rq2"):
        stage_rq2(a.subject, store, out, a.jobs)


if __name__ == "__main__":
    main()
