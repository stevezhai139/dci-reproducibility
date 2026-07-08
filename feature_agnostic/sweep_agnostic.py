#!/usr/bin/env python3
"""
sweep_agnostic.py -- Paper 3C: DCI predicts 1-D sufficiency ACROSS feature
representations (the feature-representation robustness experiment).

Runs on the SAME drift windows/seeds as the main regime map, re-featurised
four ways, and reports DCI + 1-D/multi-D detection AUC per cell:
  HSM-5D        -- the HSM-tradition instance used in the main eval (baseline)
  generic-sim3  -- kernel-free 3-axis similarity (cosine, count-ratio, Jaccard)
  raw-freq      -- kernel-free raw template-frequency vector
  table-bag     -- kernel-free table-incidence vector

Result: DCI-routing (low DCI -> 1-D suffices; high DCI -> multi-D needed)
holds in EVERY representation; the bounded-similarity reps (HSM, generic-sim)
put ordinary drift at low DCI, raw reps push it high -- same router, different
operating point. No path setup needed; run from anywhere.
"""
import os, sys, csv, collections
import numpy as np
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))       # repro_3c/feature_agnostic
REPO  = os.path.dirname(_HERE)                            # repro_3c
sys.path.insert(0, os.path.join(REPO, "geometry_E0"))    # cost_benefit, delong
sys.path.insert(0, REPO)                                  # kernel.*
sys.path.insert(0, _HERE)                                 # features_alt
import cost_benefit as cb
import features_alt as fa

NSEED = int(os.environ.get("NSEED", "50"))
OUT = os.path.join(_HERE, "out"); os.makedirs(OUT, exist_ok=True)
pools = {"tpch": cb.tpch_pool(),
         "job": cb.job_pool(Path(REPO) / "data" / "job" / "queries"),
         "pgbench": cb.pgbench_pool()}
configs = ["template_only", "volume_only", "mixed"]
rows = collections.defaultdict(lambda: collections.defaultdict(list))

for wl, pool in pools.items():
    for cfg in configs:
        for s in range(NSEED):
            windows, drift = cb.build_trajectory(pool, cfg, s)
            fv, sc = cb.kernel_adjacent(windows)          # HSM-tradition instance
            r = cb.analyse(fv, sc, drift)
            if r:
                rows[(wl,cfg,"HSM-5D")]["dci"].append(r["dci"])
                rows[(wl,cfg,"HSM-5D")]["a1"].append(r["auc_1d"])
                rows[(wl,cfg,"HSM-5D")]["am"].append(r["auc_5d"])
            for rep, res in [("generic-sim3", fa.analyse_sim(fa.sim_matrix(windows), drift)),
                             ("raw-freq",     fa.analyse_alt(fa.freq_matrix(windows), drift)),
                             ("table-bag",    fa.analyse_alt(fa.table_matrix(windows), drift))]:
                if res:
                    rows[(wl,cfg,rep)]["dci"].append(res["dci"])
                    rows[(wl,cfg,rep)]["a1"].append(res["auc_1d"])
                    rows[(wl,cfg,rep)]["am"].append(res["auc_md"])

def ms(v):
    v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

with open(os.path.join(OUT, "feature_representation_percell.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["workload","config","representation","dci_mean","dci_sd","auc_1d","auc_multiD","delta_auc","n"])
    for (wl,cfg,rep), d in sorted(rows.items()):
        dm,ds = ms(d["dci"]); a1,_ = ms(d["a1"]); am,_ = ms(d["am"])
        w.writerow([wl,cfg,rep,f"{dm:.3f}",f"{ds:.3f}",f"{a1:.3f}",f"{am:.3f}",f"{am-a1:.3f}",len(d["dci"])])

print(f"=== DCI + 1-D/multi-D AUC across representations (pooled 3 workloads, {NSEED} seeds) ===")
print(f"{'representation':16}{'template':>20}{'volume':>20}{'mixed':>20}   (DCI 1D/mD)")
for rep in ["HSM-5D","generic-sim3","raw-freq","table-bag"]:
    line = f"{rep:16}"
    for cfg in configs:
        ad=[]; a1=[]; am=[]
        for wl in pools: ad+=rows[(wl,cfg,rep)]["dci"]; a1+=rows[(wl,cfg,rep)]["a1"]; am+=rows[(wl,cfg,rep)]["am"]
        line += f"{ms(ad)[0]:5.2f} {ms(a1)[0]:.2f}/{ms(am)[0]:.2f}".rjust(20)
    print(line)
print(f"\nper-cell CSV -> {OUT}/feature_representation_percell.csv")
