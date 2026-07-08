#!/usr/bin/env python3
"""
Paper 3C eval expansion -- ABLATION: why the participation ratio (DCI)?
Reproduces each cell/seed trajectory via the harness (deterministic stable_seed),
recovers the drift-covariance spectrum, and compares DCI against three alternative
spectrum functionals as routing statistics:

  DCI (participation ratio, Hill-2) = tr(C)^2 / ||C||_F^2   [NO eigendecomposition, O(D^2)]
  effective rank (Roy-Vetterli)     = exp(-sum p_i ln p_i)  [needs full spectrum, O(D^3)]
  stable rank                       = tr(C) / lambda_max     [needs top eigenvalue]
  p1 (dominant-mode share)          = lambda_max / tr(C)     [needs top eigenvalue]

Validation gate: recomputed DCI must match the harness cost_benefit_raw.csv to ~1e-6.

HOW TO RUN (Steve's Mac -- gives the full 9-cell / 50-seed result incl JOB):
    cd "<...>/Paper 3/Paper 3C/geometry_E0"
    pip install fastdtw pywavelets numpy scipy pandas    # if not already in dbexp_venv
    cp "<...>/ablation_complexity.py" .                  # drop this script into geometry_E0/
    python3 ablation_complexity.py                        # ABL_SEEDS defaults to 50
Outputs -> geometry_E0/out/ablation/{ablation_per_seed.csv, ablation_per_cell.csv}
Send me ablation_per_cell.csv and I fold the table + subsection into the DASFAA paper.

Env knobs:  ABL_SEEDS (default 50) | ABL_WL=tpch,pgbench (subset) |
            GEO_DIR / P3C_DIR / REF_CSV (override auto-detected paths)
"""
import sys, os, glob, numpy as np, pandas as pd
from pathlib import Path

NSEED = int(os.environ.get("ABL_SEEDS", "50"))
WLSET = os.environ["ABL_WL"].split(",") if os.environ.get("ABL_WL") else None
GEO = Path(os.environ.get("GEO_DIR", Path(__file__).resolve().parent))   # geometry_E0/
P3C = Path(os.environ.get("P3C_DIR", GEO.parent))                         # Paper 3C/
sys.path.insert(0, str(GEO)); sys.path.insert(0, str(P3C))
import cost_benefit as cb

OUT = GEO / "out" / "ablation"; OUT.mkdir(parents=True, exist_ok=True)
ref_csv = os.environ.get("REF_CSV") or next(
    iter(sorted(glob.glob(str(GEO / "out" / "*" / "cost_benefit_raw.csv")))[-1:]), None)

pools = {"tpch": cb.tpch_pool()}
pp = cb.pgbench_pool()
if pp: pools["pgbench"] = pp
jobdir = P3C / "data" / "job" / "queries"  # vendored in-repo (self-contained)
try:
    jp = cb.job_pool(jobdir)
    if jp: pools["job"] = jp
except Exception as e:
    print("[job] skipped:", e)
if WLSET:
    pools = {k: v for k, v in pools.items() if k in WLSET}
print("pools loaded:", {k: len(v) for k, v in pools.items()}, "| seeds:", NSEED)

rows = []
for wl, pool in pools.items():
    for cfg in cb.CONFIGS:
        for s in range(NSEED):
            seed = cb.stable_seed(wl, cfg, s)
            windows, didx = cb.build_trajectory(pool, cfg, seed)
            fv, sc = cb.kernel_adjacent(windows)
            n = len(fv)
            lab = np.array([1 if (i + 1) in didx else 0 for i in range(n)])
            if lab.sum() < 2 or (lab == 0).sum() < 2:
                continue
            nd = lab == 0
            d = fv - fv[nd].mean(axis=0)
            driftd = d[lab == 1]
            C = (driftd.T @ driftd) / len(driftd)
            ev = np.clip(np.linalg.eigvalsh(C), 0, None)
            tot = ev.sum()
            if tot <= 0:
                continue
            p = ev / tot; pnz = p[p > 0]
            rows.append(dict(workload=wl, config=cfg, seed=s,
                             dci=(tot ** 2) / np.sum(ev ** 2),
                             eff_rank=float(np.exp(-np.sum(pnz * np.log(pnz)))),
                             stable_rank=float(tot / ev.max()),
                             p1=float(ev.max() / tot)))
A = pd.DataFrame(rows)

if ref_csv and Path(ref_csv).exists():
    ref = pd.read_csv(ref_csv)[["workload", "config", "seed", "dci"]]
    chk = A.merge(ref, on=["workload", "config", "seed"], suffixes=("_new", "_ref"))
    md = (chk.dci_new - chk.dci_ref).abs().max() if len(chk) else float("nan")
    print(f"[VALIDATION] DCI vs {Path(ref_csv).name}: n={len(chk)} max|diff|={md:.2e} "
          f"-> {'MATCH' if md < 1e-6 else 'CHECK'}")
else:
    print("[VALIDATION] no reference cost_benefit_raw.csv found -- skipping")

cell = A.groupby(["workload", "config"]).agg(
    dci=("dci", "mean"), eff_rank=("eff_rank", "mean"),
    stable_rank=("stable_rank", "mean"), p1=("p1", "mean")).reset_index()
cell["regime"] = np.where(cell.config == "mixed", "HIGH", "LOW")
print("\n[PER-CELL MEANS]\n" + cell.to_string(index=False))
print("\n[SEPARATION] empty band between LOW (template/volume) and HIGH (mixed):")
for m in ["dci", "eff_rank", "stable_rank", "p1"]:
    lo = cell[cell.regime == "LOW"][m]; hi = cell[cell.regime == "HIGH"][m]
    if m == "p1":
        band = (round(hi.max(), 3), round(lo.min(), 3)); ok = lo.min() > hi.max()
    else:
        band = (round(lo.max(), 3), round(hi.min(), 3)); ok = hi.min() > lo.max()
    print(f"  {m:12s}: band {band}  {'SEPARATES' if ok else 'OVERLAPS'}")

A.to_csv(OUT / "ablation_per_seed.csv", index=False)
cell.to_csv(OUT / "ablation_per_cell.csv", index=False)
print(f"\nsaved -> {OUT}/ablation_per_cell.csv  (send me this)")
print("Point: all separating measures give the same verdict, but only DCI needs no "
      "eigendecomposition; the others require the spectrum / top eigenvalue.")
