"""The 71 checkpoint-trajectory features (third block of the field-tier input).

The per-ref forensics (2026-08-24) showed openpilot's fragile runs differ from
robust ones not in any end-state scalar but in WHEN the evasion/braking
commitment happens relative to the closing gap: nominal traces of 38%-crash
and 0%-crash scenarios have near-identical minima, end states and 2D miss
distances. The discriminating information lives along the approach.

So: sample the trace at fixed gap checkpoints (30..2 m) and fixed times before
the criticality peak, recording at each the channels a planner's commitment
shows up in -- closing speed, lateral offset and rate, ego accel, brake
command, TTC -- plus race features (time-to-contact vs time-to-evade) at every
checkpoint, reduced to their min/last. Purely single-trace, contract-clean.
"""
import numpy as np

DT=0.05; CAR_W=1.9
GAPS=(30.,20.,15.,10.,7.,5.,3.,2.)
NCH=7

def _cols(tr):
    return tr.gap, tr.closing, np.abs(tr.y_rel), tr.v_e, tr.a_e

def extract_traj(tr, cmd_gas=None):
    gap,clos,yr,v,a=_cols(tr)
    n=len(gap)
    lr=np.abs(np.gradient(yr,DT)) if n>2 else np.zeros(n)
    ttc=np.where((clos>0.3), gap/np.maximum(clos,0.3), 30.0)
    ttc=np.clip(ttc,0,30)
    cg=cmd_gas if cmd_gas is not None else np.zeros(n)
    feats=[]; names=[]
    def add(nm,val): names.append(nm); feats.append(float(val) if np.isfinite(val) else 0.0)
    # checkpoint matrix: first time gap drops below g (approach ordering)
    for g in GAPS:
        idx=np.flatnonzero((gap<g)&(clos>0.0))
        if len(idx):
            i=int(idx[0])
            add(f"cp{int(g)}_clos",clos[i]); add(f"cp{int(g)}_yr",yr[i])
            add(f"cp{int(g)}_lr",min(lr[i],8)); add(f"cp{int(g)}_v",v[i])
            add(f"cp{int(g)}_a",np.clip(a[i],-8,8)); add(f"cp{int(g)}_ttc",ttc[i])
            add(f"cp{int(g)}_cmd",np.clip(cg[i],-1,1))
        else:
            add(f"cp{int(g)}_clos",-1); add(f"cp{int(g)}_yr",9); add(f"cp{int(g)}_lr",0)
            add(f"cp{int(g)}_v",-1); add(f"cp{int(g)}_a",0); add(f"cp{int(g)}_ttc",30)
            add(f"cp{int(g)}_cmd",0)
    # race margin along the approach: t_evade - t_contact at each step, min over run
    with np.errstate(divide='ignore',invalid='ignore'):
        t_con=np.where(clos>0.3,gap/np.maximum(clos,0.3),99.)
        lat_def=np.maximum(CAR_W-yr,0.0)
        t_ev=lat_def/np.maximum(lr,0.05)
    margin=np.where(gap>0,np.clip(t_ev,0,40)-np.clip(t_con,0,40),0.0)
    act=gap<35
    add("race_min", margin[act].min() if act.any() else 0.0)
    add("race_end", margin[-1])
    add("race_frac_bad", np.mean((margin>0)&act&(t_con<3)) if act.any() else 0.0)
    # commitment state at min-TTC moment
    i_c=int(np.argmin(ttc))
    add("yr_at_minttc",yr[i_c]); add("lr_at_minttc",min(lr[i_c],8))
    add("commit", yr[i_c]+2.0*min(lr[i_c],8))
    # end-state (truncation view)
    add("t_contact_end", min(t_con[-1],30)); add("closing_end",clos[-1])
    add("gap_end",np.clip(gap[-1],-5,60)); add("v_end",v[-1])
    add("yr_end",min(yr[-1],9)); add("lr_end",min(lr[-1],8))
    # command channel
    add("cmd_min",np.min(cg) if len(cg) else 0)
    add("cmd_brk_frac",np.mean(cg<-0.05) if len(cg) else 0)
    add("cmd_osc",np.sum(np.abs(np.diff((cg<0).astype(int)))) if len(cg)>1 else 0)
    return np.array(feats),names
