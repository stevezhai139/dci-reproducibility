#!/usr/bin/env python3
"""
energy_gap.py
=============

Paper 3C -- measure the energy-gap function g and close the loop
between DCI_THEORY_v1.md and experiment.

THE THEORY (DCI_THEORY_v1.md, section 3-4)
    A k-dimensional drift detector captures a fraction E_k = sum of the
    top-k normalised eigenvalues of the drift-motion covariance. The
    theory claims:
      (i)  detection AUC is monotone in captured energy E_k, so the
           gap  AUC(full) - AUC(k-D)  is controlled by the uncaptured
           tail energy  tail_k = 1 - E_k;
      (ii) the map  g : tail energy -> AUC gap  is increasing, g(0)=0;
      (iii) the routing thresholds are derived: tau = 1/(1 - t*) where
            g(t*) = epsilon*.
    The DCI-validation run measured an empirical 1-D routing threshold
    of DCI ~ 1.20, which implies t* ~ 0.167; the theory therefore
    PREDICTS  g(0.167) ~ 0.02.  This script measures g and checks it.

WHAT IT DOES
    Generates many workload cells (the harness, coverage dial swept so
    DCI / tail energy spans a wide range). For each cell:
      - drift-motion covariance C  ->  eigen-spectrum p_i, DCI;
      - for k = 1..5: a top-k PCA-subspace motion detector; its AUC,
        the captured energy E_k, the tail tail_k;
      - the cell's drift SNR proxy.
    Every (tail_k, gap_k) pair is one point on g. Pooled, they ARE the
    measured g. The script then checks monotonicity, the SNR
    dependence (g may be g(tail, SNR), refining the theory), and the
    g(0.167) ~ 0.02 tie-point.

OUTPUT (timestamped; every CSV row carries a UTC timestamp)
    energy_gap_points.csv     one row per (cell, k): tail, gap, AUC...
    energy_gap_run.json       monotonicity, tie-point, verdict
    energy_gap_fig.png        the measured g (if matplotlib present)

RUN
    pip install pandas numpy --break-system-packages          # mpl optional
    python energy_gap.py
    # drift_harness.py must sit in the same directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_harness as H            # noqa: E402

FEATURES = H.FEATURES
TRIAL_KEYS = H.TRIAL_KEYS

# theory tie-point (DCI_THEORY_v1.md section 4)
TIE_TAIL = 0.167
TIE_GAP = 0.02
TIE_BAND = 0.04                      # +/- band around TIE_TAIL for the check


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def cell_motions(df: pd.DataFrame, lag: int):
    """Return (all_motions, labels, drift_motions) for one cell.

    all_motions : every window's lag-difference Df, pooled over trials.
    labels      : 1 if that window is a drift window.
    drift_motions : the subset where label == 1.
    """
    allm, lab = [], []
    for _, g in df.groupby(TRIAL_KEYS, sort=False):
        g = g.sort_values("window").reset_index(drop=True)
        f = g[FEATURES].to_numpy(float)
        dt = g["drift_truth"].to_numpy()
        for t in range(lag, len(f)):
            allm.append(f[t] - f[t - lag])
            lab.append(int(dt[t]))
    allm = np.asarray(allm, float)
    lab = np.asarray(lab, int)
    return allm, lab, allm[lab == 1]


def analyse_cell(allm, lab, driftm) -> dict | None:
    """Eigen-spectrum + per-k PCA-subspace detector AUC for one cell."""
    if driftm.shape[0] < 3 or (lab == 1).sum() < 2 or (lab == 0).sum() < 2:
        return None
    # drift-motion second-moment matrix and its spectrum
    C = (driftm.T @ driftm) / driftm.shape[0]
    evals, evecs = np.linalg.eigh(C)              # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    tot = evals.sum()
    if tot <= 0:
        return None
    p = evals / tot
    dci = float(1.0 / np.sum(p ** 2))
    # drift SNR proxy: drift-window motion energy / non-drift motion energy
    e_drift = float(np.mean(np.sum(allm[lab == 1] ** 2, axis=1)))
    e_noise = float(np.mean(np.sum(allm[lab == 0] ** 2, axis=1)))
    snr = e_drift / e_noise if e_noise > 0 else float("inf")
    # per-k: top-k PCA-subspace motion detector
    rows = []
    auc_full = None
    for k in range(1, len(FEATURES) + 1):
        Vk = evecs[:, :k]                          # D x k
        score = np.linalg.norm(allm @ Vk, axis=1)  # ||P_k Df||
        a = H.auc(lab, score)
        if k == len(FEATURES):
            auc_full = a
        rows.append({"k": k, "E_k": float(p[:k].sum()),
                     "tail": float(1.0 - p[:k].sum()), "auc": a})
    for r in rows:
        r["gap"] = (auc_full - r["auc"]) if auc_full is not None else float("nan")
    return {"dci": dci, "snr": snr, "per_k": rows, "auc_full": auc_full}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                    help="E0 breakdown_per_window.csv")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--c-values", type=str,
                    default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--r-values", type=str, default="1,3")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=10)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    in_path = (Path(args.input).expanduser().resolve() if args.input
               else H.default_input_path(script_dir))
    out_root = (Path(args.outdir).expanduser().resolve() if args.outdir
                else script_dir / "out")
    run_ts = utc_now_iso()
    run_id = run_ts.replace("-", "").replace(":", "")
    outdir = out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[run ] id={run_id}")
    if not in_path.exists():
        print(f"[ERR ] E0 input not found: {in_path}", file=sys.stderr)
        return 2
    cal = H.extract_anchors(pd.read_csv(in_path))

    c_vals = [float(x) for x in args.c_values.split(",")]
    r_vals = [int(x) for x in args.r_values.split(",")]
    print(f"[grid] {len(c_vals)} c x {len(r_vals)} r x {args.batches} batches")

    # ---- sweep: every (cell, k) is a point on g -------------------------
    points = []
    cidx = 0
    for r in r_vals:
        for c in c_vals:
            for b in range(args.batches):
                seed = 50000 + cidx * 1000 + b
                df = H.make_cell(cal, c, r, 0.0, 1, args.n_trials, seed)
                allm, lab, driftm = cell_motions(df, lag=r)
                res = analyse_cell(allm, lab, driftm)
                if res is None:
                    continue
                for rk in res["per_k"]:
                    points.append({
                        "analysis_timestamp_utc": run_ts, "run_id": run_id,
                        "c_coverage": c, "r_speed": r, "batch": b,
                        "dci": res["dci"], "snr_proxy": res["snr"],
                        "k_axes": rk["k"], "captured_energy": rk["E_k"],
                        "tail_energy": rk["tail"], "auc_k": rk["auc"],
                        "auc_full": res["auc_full"], "gap": rk["gap"],
                    })
            cidx += 1
    P = pd.DataFrame(points)
    P.to_csv(outdir / "energy_gap_points.csv", index=False)

    # ---- analysis: the measured g ---------------------------------------
    # g points: exclude k=5 (tail=0, gap=0 by construction)
    G = P[P.k_axes < len(FEATURES)].copy()
    rho_tail_gap = spearman(G["tail_energy"], G["gap"])
    rho_dci_gap1 = spearman(P.loc[P.k_axes == 1, "dci"],
                            P.loc[P.k_axes == 1, "gap"])

    # tie-point: mean gap among points with tail near TIE_TAIL
    near = G[(G.tail_energy >= TIE_TAIL - TIE_BAND) &
             (G.tail_energy <= TIE_TAIL + TIE_BAND)]
    tie_gap_measured = float(near["gap"].mean()) if len(near) else float("nan")
    tie_gap_std = float(near["gap"].std(ddof=1)) if len(near) > 1 else float("nan")

    # SNR dependence: split g into low/high SNR halves, compare gap at
    # matched tail. If g is truly g(tail) alone the two agree.
    snr_med = float(G["snr_proxy"].replace(np.inf, np.nan).median())
    lo = near[near.snr_proxy <= snr_med]
    hi = near[near.snr_proxy > snr_med]
    snr_split = {
        "snr_median": snr_med,
        "gap_low_snr": float(lo["gap"].mean()) if len(lo) else float("nan"),
        "gap_high_snr": float(hi["gap"].mean()) if len(hi) else float("nan"),
    }

    # binned g curve for reporting
    bins = np.linspace(0, max(0.6, G["tail_energy"].max()), 13)
    G["tail_bin"] = pd.cut(G["tail_energy"], bins)
    curve = (G.groupby("tail_bin", observed=True)["gap"]
             .agg(["mean", "std", "count"]).reset_index())
    curve["tail_mid"] = curve["tail_bin"].apply(lambda iv: iv.mid)

    monotone_ok = rho_tail_gap >= 0.6
    tie_ok = (np.isfinite(tie_gap_measured) and
              abs(tie_gap_measured - TIE_GAP) <= 0.02)
    snr_gap = abs(snr_split["gap_low_snr"] - snr_split["gap_high_snr"])
    snr_matters = np.isfinite(snr_gap) and snr_gap > 0.02

    if monotone_ok and tie_ok:
        verdict = ("Loop CLOSED: g is monotone in tail energy and the "
                   f"measured g({TIE_TAIL}) = {tie_gap_measured:.3f} "
                   f"matches the theory prediction {TIE_GAP}. DCI_THEORY "
                   "section 3-4 is empirically supported.")
    elif monotone_ok:
        verdict = ("g is monotone in tail energy (theory step (i)/(ii) "
                   f"holds, Spearman {rho_tail_gap:.2f}) but the tie-point "
                   f"g({TIE_TAIL}) = {tie_gap_measured:.3f} differs from "
                   f"the predicted {TIE_GAP}: the threshold derivation in "
                   "section 4 needs the measured g, not the assumed one.")
    else:
        verdict = (f"g is NOT cleanly monotone in tail energy "
                   f"(Spearman {rho_tail_gap:.2f}); the theory's "
                   "'gap controlled by tail energy' claim does not hold "
                   "as stated -- likely the SNR dependence below.")
    if snr_matters:
        verdict += (f" NOTE: g depends on SNR -- at tail~{TIE_TAIL} the "
                    f"AUC gap differs by {snr_gap:.3f} between low- and "
                    "high-SNR cells, so g = g(tail, SNR); the theory "
                    "should be refined to two arguments.")

    run_json = {
        "run": {"analysis_timestamp_utc": run_ts, "run_id": run_id,
                "script": Path(__file__).name, "n_points": int(len(P)),
                "grid": {"c": c_vals, "r": r_vals, "batches": args.batches}},
        "measured_g": curve[["tail_mid", "mean", "std", "count"]]
        .to_dict(orient="records"),
        "monotonicity_spearman_tail_gap": rho_tail_gap,
        "spearman_dci_vs_1d_gap": rho_dci_gap1,
        "tie_point": {"predicted_tail": TIE_TAIL, "predicted_gap": TIE_GAP,
                      "measured_gap": tie_gap_measured,
                      "measured_gap_std": tie_gap_std,
                      "n_points_in_band": int(len(near)), "match": bool(tie_ok)},
        "snr_dependence": snr_split,
        "verdict": verdict,
    }
    (outdir / "energy_gap_run.json").write_text(json.dumps(run_json, indent=2))

    # ---- figure ----------------------------------------------------------
    fig_path = outdir / "energy_gap_fig.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        sc = ax[0].scatter(G["tail_energy"], G["gap"], c=G["snr_proxy"],
                           cmap="viridis", s=18, alpha=0.7,
                           vmax=np.nanpercentile(
                               G["snr_proxy"].replace(np.inf, np.nan), 95))
        ax[0].plot(curve["tail_mid"], curve["mean"], color="#d62728",
                   lw=2, marker="o", label="binned mean g")
        ax[0].scatter([TIE_TAIL], [TIE_GAP], s=180, marker="*",
                      color="#d62728", edgecolor="black", zorder=5,
                      label=f"theory tie-point ({TIE_TAIL}, {TIE_GAP})")
        ax[0].set_xlabel("tail energy  1 - E_k")
        ax[0].set_ylabel("AUC gap   auc_full - auc_k")
        ax[0].set_title(f"A  measured energy-gap g   "
                        f"(Spearman={rho_tail_gap:.2f})", fontsize=10)
        ax[0].legend(fontsize=8)
        fig.colorbar(sc, ax=ax[0], label="drift SNR proxy")
        d1 = P[P.k_axes == 1]
        ax[1].scatter(d1["dci"], d1["gap"], s=18, alpha=0.7, color="#0a3d62")
        ax[1].set_xlabel("DCI"); ax[1].set_ylabel("1-D detector AUC gap")
        ax[1].set_title(f"B  DCI vs 1-D gap   "
                        f"(Spearman={rho_dci_gap1:.2f})", fontsize=10)
        fig.suptitle(f"Paper 3C  -  energy-gap function g   run {run_id}",
                     fontsize=11, weight="bold")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
    except Exception as exc:                                   # noqa: BLE001
        fig_path = None
        print(f"[warn] figure skipped ({exc})")

    # ---- console --------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  ENERGY-GAP MEASUREMENT   run {run_id}")
    print("=" * 70)
    print(f"  points: {len(P)}  ({len(G)} on the g curve)")
    print(f"  monotonicity  Spearman(tail, gap)   : {rho_tail_gap:.3f}")
    print(f"  Spearman(DCI, 1-D gap)              : {rho_dci_gap1:.3f}")
    print()
    print("  measured g  (tail energy -> AUC gap):")
    for _, r in curve.iterrows():
        if r["count"] > 0:
            print(f"    tail~{r['tail_mid']:.3f}  gap={r['mean']:.3f} "
                  f"+/-{(r['std'] if np.isfinite(r['std']) else 0):.3f}  "
                  f"(n={int(r['count'])})")
    print()
    print(f"  THEORY TIE-POINT  g({TIE_TAIL}) predicted = {TIE_GAP}")
    print(f"    measured g({TIE_TAIL}) = {tie_gap_measured:.3f} "
          f"+/- {tie_gap_std:.3f}  (n={len(near)})  -> match: {tie_ok}")
    print(f"  SNR dependence at tail~{TIE_TAIL}: "
          f"low-SNR gap {snr_split['gap_low_snr']:.3f}  "
          f"high-SNR gap {snr_split['gap_high_snr']:.3f}")
    print()
    print(f"  VERDICT: {verdict}")
    print()
    for f in ("energy_gap_points.csv", "energy_gap_run.json"):
        print(f"[out ] {outdir / f}")
    if fig_path:
        print(f"[out ] {fig_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
