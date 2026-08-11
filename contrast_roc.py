#!/usr/bin/env python3
"""contrast_roc.py — matched-FA (ROC) + attribution analysis of the live
contrast run, replayed offline from the LOGGED per-window feature stream.

Motivation (Steve, 2026-08-11): a union's tail fires are real alarms paid
for out of its false-alarm budget — same statistic, no margin. The fair
comparison is therefore detection at MATCHED false-alarm rate, not at
matched nominal alpha. All inputs are the live run's logged S_R..S_P
(deterministic post-processing; no new machine time).

Per arm-agnostic replay (features are paired across arms; we use the gated
arm's stream, which carries S_P):
  1. Pool steady windows (pre-onset) across blocks -> mu0/Sigma0 (declared:
     replay calibration differs from the live gate's 64-window pass).
  2. Detector statistics per window:  union = max_j z_j^2 (4 cheap axes);
     full = 5-D Mahalanobis;  routed = full's stat on windows the LIVE
     router escalated (logged gate_regime), union's stat elsewhere —
     i.e., the live routing decisions, rethresholded.
  3. Sweep the threshold scale; report det@3 (first 3 post windows) at
     matched FA/window in {0.01, 0.02, 0.05} + attribution: fraction of
     routed first-detections landing on an escalated window.

Usage: python3 contrast_roc.py <repro_root> [--lam l06]
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd

AX = ["S_R", "S_V", "S_T", "S_A", "S_P"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=Path)
    ap.add_argument("--lam", default=None)
    ap.add_argument("--arm", default="gated")
    ap.add_argument("--out", default="contrast_roc.csv")
    a = ap.parse_args()
    pgdir = a.repro / "end_to_end" / "postgres"
    if a.lam is None:
        hits = sorted(glob.glob(str(pgdir / "out_CONTRAST_PG_*_gated")))
        if not hits:
            print("no contrast output found"); return 1
        a.lam = Path(hits[-1]).name.split("_")[3]
    fs = sorted(glob.glob(str(pgdir / f"out_CONTRAST_PG_{a.lam}_{a.arm}" /
                              "breakdown_per_window*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    print(f"[roc] lam={a.lam} arm={a.arm} rows={len(df)}")

    # onset per block from ground truth
    onsets = {b: int(g.loc[g["drift_truth"] == 1, "window"].iloc[0])
              for b, g in df.groupby("block")}
    df["post"] = df.apply(lambda r: r["window"] >= onsets[r["block"]], axis=1)

    # replay calibration on pooled steady windows (window >= 2 to skip
    # the init-adjacent pair)
    cal = df[(~df["post"]) & (df["window"] >= 2)][AX].to_numpy(float)
    mu, sd = cal.mean(0), cal.std(0, ddof=1)
    live = sd > 1e-9          # S_V is constant under fixed-count windows:
    sd_s = np.where(live, sd, 1.0)   # zero-variance axes carry no noise and
    print(f"[cal] pooled steady m={len(cal)}; live axes = "
          f"{[a for a, m in zip(AX, live) if m]}")
    C = np.corrcoef(((cal - mu) / sd_s)[:, live].T)
    Ci = np.linalg.pinv(C)

    Z = (df[AX].to_numpy(float) - mu) / sd_s
    cheap_live = live.copy(); cheap_live[4] = False
    stat_union = (Z[:, cheap_live] ** 2).max(1)
    Zl = Z[:, live]
    stat_full = np.einsum("ij,jk,ik->i", Zl, Ci, Zl)
    esc = (df["gate_regime"].astype(str).to_numpy() == "full")
    # routed statistic: live routing decisions, rethresholded. Scale the
    # two statistics by their own steady quantiles so one threshold works.
    df["_su"], df["_sf"] = stat_union, stat_full

    steady_mask = (~df["post"]) & (df["window"] >= 2)

    def det_at_fa(stat, fa_target, use=None):
        st = df.loc[steady_mask].index
        base = stat[st] if use is None else stat[st]
        thr = float(np.quantile(base, 1.0 - fa_target))
        det, dly, attr = [], [], []
        for b, g in df.groupby("block"):
            on = onsets[b]
            post = g[g["window"] >= on]
            fired = post.index[stat[post.index] > thr]
            first = int(post.loc[fired[0], "window"]) - on if len(fired) else None
            det.append(int(first is not None and first < 3))
            if first is not None:
                dly.append(first)
                attr.append(bool(esc[fired[0]]))
        return (float(np.mean(det)), float(np.median(dly)) if dly else np.nan,
                float(np.mean(attr)) if attr else np.nan, thr)

    rows = []
    stat_routed = np.where(esc, stat_full / np.quantile(stat_full[df.loc[steady_mask].index], 0.95),
                           stat_union / np.quantile(stat_union[df.loc[steady_mask].index], 0.95))
    for fa in (0.01, 0.02, 0.05):
        for nm, stt in (("union", stat_union), ("full", stat_full),
                        ("routed", stat_routed)):
            d, dl, at, thr = det_at_fa(stt, fa)
            rows.append({"fa_per_win": fa, "detector": nm, "det3": round(d, 3),
                         "dly_med": dl, "attr_esc": (round(at, 3) if at == at else None)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(a.out, index=False)
    print(f"[out] {a.out}")
    print("\n[read] attr_esc for 'routed' = fraction of first-detections on an")
    print("       escalated window (geometry-driven, not tail-driven).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
