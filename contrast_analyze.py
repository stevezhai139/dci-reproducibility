#!/usr/bin/env python3
"""contrast_analyze.py — Paper 3C live contrast canary: 3-arm analysis.

Reads end_to_end/postgres/out_CONTRAST_PG_<lam>_{afull,gated,acheap}/
breakdown_per_window CSVs. Onset per block = first drift_truth==1 window.

Frontier statistics (the correct ones for a PERSISTENT change — see
contrast_preflight.py: any nonzero-mean axis makes the union consistent
via its noise tail, so binary recall is meaningless; rate and delay are
the statistics):
  per arm: net-of-floor detection rate at k in {1,3,6,12} post windows
           (floor = that arm's own steady FA rate -> 1-(1-p)^k),
           median delay, steady FA/window, post esc%% (non-audit),
           R4s at trajectory end, mean det_ms steady/post.

Usage: python3 contrast_analyze.py <repro_root> [--lam l06]
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ARMS = ["afull", "gated", "acheap"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=Path)
    ap.add_argument("--lam", default=None, help="e.g. l06 (default: autodetect)")
    ap.add_argument("--out", default="contrast_summary.csv")
    a = ap.parse_args()
    pgdir = a.repro / "end_to_end" / "postgres"
    if a.lam is None:
        hits = sorted(glob.glob(str(pgdir / "out_CONTRAST_PG_*_gated")))
        if not hits:
            print("no out_CONTRAST_PG_*_gated found"); return 1
        a.lam = Path(hits[-1]).name.split("_")[3]
    print(f"[contrast] lam tag = {a.lam}")

    rows = []
    for arm in ARMS:
        d = pgdir / f"out_CONTRAST_PG_{a.lam}_{arm}"
        fs = sorted(glob.glob(str(d / "breakdown_per_window*.csv")))
        if not fs:
            print(f"  !! {d} missing"); continue
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        det_k = {k: [] for k in (1, 3, 6, 12)}
        delays, fa_rates, escs, r_ends = [], [], [], []
        dms_s, dms_p = [], []
        for blk, g in df.groupby("block"):
            g = g.sort_values("window")
            on = g.loc[g["drift_truth"] == 1, "window"]
            if on.empty:
                continue
            on = int(on.iloc[0])
            steady = g[(g["window"] >= 1) & (g["window"] < on)]
            post = g[g["window"] >= on]
            p_fa = float(steady["invoked"].mean()) if len(steady) else 0.0
            fa_rates.append(p_fa)
            fired = post.loc[post["invoked"] == 1, "window"]
            dly = (int(fired.iloc[0]) - on) if len(fired) else None
            delays.append(dly if dly is not None else np.inf)
            for k in det_k:
                det_k[k].append(int(dly is not None and dly < k))
            if "gate_regime" in g:
                escs.append(float(((post["gate_regime"] == "full")
                                   & (post["gate_audit"] == 0)).mean()))
            if "gate_R4s" in g:
                r_ends.append(float(g["gate_R4s"].iloc[-1]))
            if "det_ms" in g:
                dms_s.append(float(steady["det_ms"].mean()))
                dms_p.append(float(post["det_ms"].mean()))
        p_fa = float(np.mean(fa_rates)) if fa_rates else 0.0
        row = {"arm": arm, "blocks": len(fa_rates),
               "fa_per_win": round(p_fa, 4),
               "dly_med": (float(np.median([d for d in delays]))
                           if delays else float("nan")),
               "esc_post": round(float(np.mean(escs)), 3) if escs else None,
               "R_end_med": round(float(np.median(r_ends)), 3) if r_ends else None,
               "det_ms_steady": round(float(np.mean(dms_s)), 3) if dms_s else None,
               "det_ms_post": round(float(np.mean(dms_p)), 3) if dms_p else None}
        for k, v in det_k.items():
            raw = float(np.mean(v)) if v else 0.0
            floor = 1.0 - (1.0 - p_fa) ** k
            row[f"det{k}"] = round(raw, 3)
            row[f"net{k}"] = round(raw - floor, 3)
        rows.append(row)

    cols = ["arm", "blocks", "det1", "net1", "det3", "net3", "det6", "net6",
            "det12", "net12", "dly_med", "fa_per_win", "esc_post",
            "R_end_med", "det_ms_steady", "det_ms_post"]
    out = pd.DataFrame(rows)[[c for c in cols if c in rows[0]]] if rows else None
    if out is None:
        return 1
    print(out.to_string(index=False))
    out.to_csv(a.out, index=False)
    print(f"[out] {a.out}")
    g = {r["arm"]: r for r in rows}
    if "gated" in g and "acheap" in g:
        ng, nc = g["gated"]["net3"], max(g["acheap"]["net3"], 0.04)
        print(f"\n[frontier] net3 gated/cheap = {ng:.3f}/{nc:.3f} "
              f"= {ng/nc:.1f}x | dly med {g['gated']['dly_med']} vs "
              f"{g['acheap']['dly_med']} | esc {g['gated']['esc_post']} "
              f"| R_end {g['gated']['R_end_med']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
