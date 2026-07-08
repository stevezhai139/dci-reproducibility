import os,sys,csv,collections
import numpy as np
from pathlib import Path
_HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(_HERE)
sys.path.insert(0,os.path.join(REPO,"geometry_E0")); sys.path.insert(0,REPO); _HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
import cost_benefit as cb, features_alt as fa
NSEED=int(os.environ.get("NSEED","50"))
pools={"tpch":cb.tpch_pool(),"job":cb.job_pool(Path(REPO)/"data"/"job"/"queries"),"pgbench":cb.pgbench_pool()}
configs=["template_only","volume_only","mixed"]
rows=collections.defaultdict(lambda:collections.defaultdict(list))
for wl,pool in pools.items():
    for cfg in configs:
        for s in range(NSEED):
            windows,drift=cb.build_trajectory(pool,cfg,s)
            for rep,res in [("generic-sim3",fa.analyse_sim(fa.sim_matrix(windows),drift)),
                            ("raw-freq",fa.analyse_alt(fa.freq_matrix(windows),drift)),
                            ("table-bag",fa.analyse_alt(fa.table_matrix(windows),drift))]:
                if res:
                    rows[(wl,cfg,rep)]["dci"].append(res["dci"]); rows[(wl,cfg,rep)]["a1"].append(res["auc_1d"]); rows[(wl,cfg,rep)]["am"].append(res["auc_md"])
def ms(v): v=[x for x in v if not(isinstance(x,float) and np.isnan(x))]; return float(np.mean(v)) if v else float("nan")
os.makedirs("/tmp/fa_build/out",exist_ok=True)
with open("/tmp/fa_build/out/alt_percell.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["workload","config","representation","dci","auc_1d","auc_multiD","delta"])
    for (wl,cfg,rep),d in sorted(rows.items()):
        w.writerow([wl,cfg,rep,f"{ms(d['dci']):.3f}",f"{ms(d['a1']):.3f}",f"{ms(d['am']):.3f}",f"{ms(d['am'])-ms(d['a1']):.3f}"])
print("=== ALT reps: DCI (pooled 3 workloads, %d seeds) ==="%NSEED)
print(f"{'rep':14}{'template':>10}{'volume':>10}{'mixed':>10}")
for rep in ["generic-sim3","raw-freq","table-bag"]:
    line=f"{rep:14}"
    for cfg in configs:
        allv=[]; 
        for wl in pools: allv+=rows[(wl,cfg,rep)]["dci"]
        line+=f"{ms(allv):10.2f}"
    print(line)
print("done -> out/alt_percell.csv")
