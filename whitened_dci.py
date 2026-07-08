#!/usr/bin/env python3
"""whitened_dci.py -- S1 robustness check for Paper 3C.

Recomputes, per (workload, config, seed) cell of the regime map, BOTH the
raw DCI (must reproduce the paper's Table 2 to the printed digit -- the
fidelity check) and the *whitened* DCI: the participation ratio of
Sigma^{-1/2} C Sigma^{-1/2}, where Sigma is the steady-window noise
covariance exactly as built in cost_benefit.analyse() and C is the
drift-window deviation covariance exactly as used for the raw DCI.

Mirrors cost_benefit.main()'s seed loop verbatim: stable_seed(wl, cfg, s),
build_trajectory, kernel_adjacent, and analyse()'s d/nd/Sigma/C algebra.
"""
import sys, json, csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = Path(sys.argv[1]).resolve()          # .../repro_3c
sys.path.insert(0, str(REPRO))               # kernel.*
sys.path.insert(0, str(REPRO / "geometry_E0"))

import cost_benefit as cb                    # reuse the paper's own code

SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
ONLY_WL = sys.argv[3] if len(sys.argv) > 3 else None    # run a single cell
ONLY_CFG = sys.argv[4] if len(sys.argv) > 4 else None
OUT = HERE / "whitened_dci_results_v2.csv"


def pr(M: np.ndarray) -> float:
    ev = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    tot = float(ev.sum()); f2 = float((ev ** 2).sum())
    return (tot * tot) / f2 if tot > 0 and f2 > 0 else float("nan")


def whiten_inv_sqrt(S: np.ndarray) -> np.ndarray:
    ev, V = np.linalg.eigh(S)
    ev = np.clip(ev, 1e-12, None)
    return V @ np.diag(ev ** -0.5) @ V.T


def main() -> int:
    makers = {"tpch": cb.tpch_pool,
              "job": lambda: cb.job_pool(REPRO / "data" / "job" / "queries"),
              "pgbench": cb.pgbench_pool}
    if ONLY_WL:
        makers = {ONLY_WL: makers[ONLY_WL]}
    rows = []
    for wl, mk in makers.items():
        pool = mk()
        for cfg in ([ONLY_CFG] if ONLY_CFG else cb.CONFIGS):
            for s in range(SEEDS):
                seed = cb.stable_seed(wl, cfg, s)
                windows, didx = cb.build_trajectory(pool, cfg, seed)
                fv, sc = cb.kernel_adjacent(windows)
                n = len(fv)
                lab = np.array([1 if (i + 1) in didx else 0 for i in range(n)])
                if lab.sum() < 2 or (lab == 0).sum() < 2:
                    continue
                nd = lab == 0
                d = fv - fv[nd].mean(axis=0)              # as analyse()
                Sig = np.cov(d[nd].T) + 1e-6 * np.eye(d.shape[1])  # as analyse()
                driftd = d[lab == 1]
                C = (driftd.T @ driftd) / len(driftd)      # as analyse()
                W = whiten_inv_sqrt(Sig)
                Cw = W @ C @ W
                rows.append({"workload": wl, "config": cfg, "seed": s,
                             "dci_raw": pr(C), "dci_white": pr(Cw)})
        print(f"[done] {wl}", flush=True)

    new_file = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new_file:
            w.writeheader()
        w.writerows(rows)

    # per-cell aggregation over EVERYTHING accumulated so far
    allrows = list(csv.DictReader(open(OUT)))
    for r in allrows:
        r["dci_raw"] = float(r["dci_raw"]); r["dci_white"] = float(r["dci_white"])
    agg = {}
    for r in allrows:
        agg.setdefault((r["workload"], r["config"]), []).append(r)
    print(f"{'cell':26s} {'rawDCI':>7s} {'whiteDCI':>9s} {'w_lo95':>7s} {'w_hi95':>7s}")
    summ = []
    for (wl, cfg), rs in sorted(agg.items()):
        raw = np.array([x["dci_raw"] for x in rs])
        wh = np.array([x["dci_white"] for x in rs])
        ci = 1.96 * wh.std(ddof=1) / np.sqrt(len(wh))
        print(f"{wl:8s}{cfg:18s} {raw.mean():7.3f} {wh.mean():9.3f} "
              f"{wh.mean()-ci:7.3f} {wh.mean()+ci:7.3f}")
        summ.append({"workload": wl, "config": cfg, "n": len(rs),
                     "dci_raw_mean": float(raw.mean()),
                     "dci_white_mean": float(wh.mean()),
                     "dci_white_ci95": float(ci)})
    json.dump(summ, open(HERE / "whitened_dci_summary_v2.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
