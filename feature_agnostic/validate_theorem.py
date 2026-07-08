#!/usr/bin/env python3
"""
validate_theorem.py -- numerically check the DCI 1-D sufficiency theorem
(DCI_SUFFICIENCY_THEOREM_DRAFT.md) against the canonical kernel.

Per (workload, config), over seeds, from the real kernel features:
  DCI_raw  = participation ratio of the drift covariance C          (paper's DCI)
  DCI_w    = participation ratio of whitened C~ = S^-1/2 C S^-1/2    (theory)
  p1_w     = dominant whitened-mode share = R (Theorem 1 ratio)
  auc_1d, auc_5d, gap = measured detection AUCs (cost_benefit.analyse)
  sig_offdiag, sig_cond = how white the steady covariance S is (R1 proxy check)

Checks: (i) DCI_w ~ DCI_raw?  (ii) low p1_w  <=>  large AUC gap?  (iii) S near white?
Run from anywhere; uses the newest canonical run implicitly via the same harness.
"""
import os, sys, collections
import numpy as np
from pathlib import Path
_HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(REPO,"geometry_E0")); sys.path.insert(0, REPO)
import cost_benefit as cb

NSEED=int(os.environ.get("NSEED","50"))
pools={"tpch":cb.tpch_pool(),"job":cb.job_pool(Path(REPO)/"data"/"job"/"queries"),"pgbench":cb.pgbench_pool()}
configs=["template_only","volume_only","mixed"]

def pr(M):
    ev=np.clip(np.linalg.eigvalsh(M),0,None); t=ev.sum()
    return float(t*t/np.sum(ev*ev)) if t>0 else float("nan")

def whitened(fv, drift_idx):
    n=len(fv); lab=np.array([1 if (i+1) in drift_idx else 0 for i in range(n)])
    if lab.sum()<2 or (lab==0).sum()<2: return None
    nd=lab==0; d=fv-fv[nd].mean(0)
    Sig=np.cov(d[nd].T)+1e-9*np.eye(d.shape[1])
    driftd=d[lab==1]; C=(driftd.T@driftd)/len(driftd)
    w,V=np.linalg.eigh(Sig); w=np.clip(w,1e-12,None)
    Sisq=V@np.diag(1/np.sqrt(w))@V.T
    Ct=Sisq@C@Sisq
    evt=np.clip(np.linalg.eigvalsh(Ct),0,None)
    p1w=float(evt.max()/evt.sum()) if evt.sum()>0 else float("nan")
    evC=np.clip(np.linalg.eigvalsh(C),0,None)
    p1_raw=float(evC.max()/evC.sum()) if evC.sum()>0 else float("nan")  # R under white noise (Thm 1)
    Dg=np.sqrt(np.diag(Sig)); Corr=Sig/np.outer(Dg,Dg)
    off=float(np.abs(Corr[np.triu_indices_from(Corr,1)]).mean())
    return dict(dci_raw=pr(C), dci_w=pr(Ct), p1_w=p1w, p1_raw=p1_raw, sig_off=off, sig_cond=float(w.max()/w.min()))

agg=collections.defaultdict(lambda: collections.defaultdict(list))
for wl,pool in pools.items():
    for cfg in configs:
        for s in range(NSEED):
            windows,drift=cb.build_trajectory(pool,cfg,s)
            fv,sc=cb.kernel_adjacent(windows)
            r=cb.analyse(fv,sc,drift); w=whitened(fv,drift)
            if r and w:
                for k in ("dci_raw","dci_w","p1_w","p1_raw","sig_off","sig_cond"): agg[(wl,cfg)][k].append(w[k])
                agg[(wl,cfg)]["auc1"].append(r["auc_1d"]); agg[(wl,cfg)]["auc5"].append(r["auc_5d"])
                agg[(wl,cfg)]["dci_paper"].append(r["dci"])

def m(v): v=[x for x in v if not(isinstance(x,float) and np.isnan(x))]; return float(np.mean(v)) if v else float("nan")
print(f"=== Theorem validation (canonical kernel, {NSEED} seeds) ===\n")
print(f"{'workload':8}{'config':14}{'DCI_raw':>8}{'DCI_w':>8}{'p1_raw=R':>9}{'p1_w':>7}{'auc1':>7}{'auc5':>7}{'gap':>7}{'S_off':>7}")
for wl in pools:
    for cfg in configs:
        a=agg[(wl,cfg)]
        g=m(a["auc5"])-m(a["auc1"])
        print(f"{wl:8}{cfg:14}{m(a['dci_raw']):8.2f}{m(a['dci_w']):8.2f}{m(a['p1_raw']):9.2f}{m(a['p1_w']):7.2f}{m(a['auc1']):7.2f}{m(a['auc5']):7.2f}{g:7.2f}{m(a['sig_off']):7.2f}")
print("\nCHECK (i)  DCI_w vs DCI_raw close?  -> raw DCI is a valid proxy (R1)")
print("CHECK (ii)  p1_raw = R (Thm 1, white-noise); low p1_raw <=> large gap -> predicts AUC regime")
print("CHECK (ii') p1_w (whitened) for contrast: on JOB it misleads (says sufficient) -> R3/future work")
print("CHECK (iii) S_off (mean |corr| off-diag): near 0 = white; larger = whitening matters")
