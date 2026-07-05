#!/usr/bin/env python3
"""
visualize_e0_drift.py
=====================

Paper 3C -- visual companion to decompose_e0_geometry.py.

PURPOSE
    Make 5-D workload drift *visible*.  The decomposition script gives
    numbers (PC1 share, angular fraction, Gram matrix); this script
    turns the same raw data into pictures, because a 5-D trajectory is
    easier to reason about when you can see it move.

INPUT
    breakdown_per_window.csv  -- the per-window HSM feature log.
    Columns required: block, block_seed, strategy, window, phase,
    drift_truth, and the 5 features S_R, S_V, S_T, S_A, S_P.
    The script is ENGINE-AGNOSTIC: point it at a MongoDB log now, at a
    Postgres `rq4a` log later (E1), or at synthetic drift later (E4) --
    same five views, directly comparable.

WHAT IT DRAWS  (one static figure, five panels + one animation)
    A  per-axis time series  -- f(t) on each of the 5 axes, drift
                                windows + phase bands marked.  The
                                literal "drift along all 5 axes".
    B  radar / polar plot    -- the 5-D similarity profile as a
                                pentagon, one per workload phase;
                                shows how the *shape* deforms.
    C  PCA trajectory        -- the 24-window path projected onto its
                                top-2 principal components.  If drift
                                is rank-1 the path moves along PC1.
    D  informative subspace  -- 3-D path over (S_R, S_T, S_A), the
                                three axes that actually carry signal.
    E  radius from pole      -- r(t) = ||f(t) - 1||, the scalar a
                                1-D detector would watch.
    + animated radar GIF     -- the pentagon evolving window 1 -> 24
                                (falls back to a frame contact-sheet
                                if no GIF writer is available).

OUTPUT  (timestamped folder; every CSV row carries a UTC timestamp)
    <outdir>/<run_id>/
        e0_drift_fiveaxis.png       the 5-panel static figure
        e0_drift_radar.gif          animation (or *_frames.png fallback)
        e0_drift_plotted_series.csv per-window values behind the plots
        e0_drift_viz_run.json       run metadata

RUN
    pip install pandas numpy matplotlib --break-system-packages
    python visualize_e0_drift.py
    python visualize_e0_drift.py --input /path/to/breakdown_per_window.csv
    python visualize_e0_drift.py --strategy all      # pool all strategies
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D proj)

# --------------------------------------------------------------------------
FEATURES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
TRIAL_KEYS = ["strategy", "block", "block_seed"]
AXIS_COLOR = {
    "S_R": "#1f77b4", "S_V": "#9467bd", "S_T": "#d62728",
    "S_A": "#2ca02c", "S_P": "#ff7f0e",
}


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_input_path(script_dir: Path) -> Path:
    rel = ("../../Paper 3B/HSM_gated_3B_for_paper3b_v2/code/results/"
           "cross_engine/mongo/adaptation/20260430_144825/"
           "breakdown_per_window.csv")
    return (script_dir / rel).resolve()


def pca_2d(points: np.ndarray):
    """Center + SVD; return (scores Nx2, var_fraction len-k, components 2xD)."""
    mu = points.mean(axis=0)
    X = points - mu
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    var = s ** 2
    frac = var / var.sum() if var.sum() > 0 else var
    scores = X @ vt[:2].T
    return scores, frac, vt[:2], mu


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                    help="path to breakdown_per_window.csv")
    ap.add_argument("--outdir", type=str, default=None,
                    help="output directory (default: <script_dir>/out)")
    ap.add_argument("--strategy", type=str, default="no_advisor",
                    help="strategy to plot, or 'all' to pool (default: no_advisor)")
    ap.add_argument("--tag", type=str, default="mongo_e0",
                    help="provenance label stored in outputs")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    in_path = (Path(args.input).expanduser().resolve() if args.input
               else default_input_path(script_dir))
    out_root = (Path(args.outdir).expanduser().resolve() if args.outdir
                else script_dir / "out")

    run_ts = utc_now_iso()
    run_id = run_ts.replace("-", "").replace(":", "")
    outdir = out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[run ] id={run_id}  tag={args.tag}  strategy={args.strategy}")
    print(f"[in  ] {in_path}")
    if not in_path.exists():
        print(f"[ERR ] input not found: {in_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(in_path)
    need = TRIAL_KEYS + ["window", "phase", "drift_truth"] + FEATURES
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"[ERR ] missing columns: {miss}", file=sys.stderr)
        return 2

    if args.strategy != "all":
        if args.strategy in set(df["strategy"].unique()):
            df = df[df.strategy == args.strategy].copy()
        else:
            print(f"[warn] strategy '{args.strategy}' not in data; "
                  f"using all rows instead")
            args.strategy = "all"

    drift_windows = sorted(int(w) for w in
                           df.loc[df.drift_truth == 1, "window"].unique())

    # ---- per-window aggregate across trials ------------------------------
    grp = df.groupby("window")
    mean_f = grp[FEATURES].mean()
    std_f = grp[FEATURES].std().fillna(0.0)
    windows = mean_f.index.to_numpy()
    M = mean_f.to_numpy(dtype=float)                       # (n_win, 5)

    # phase label per window (mode)
    phase_of = grp["phase"].agg(lambda s: s.mode().iloc[0])
    # drift flag per window
    drift_of = grp["drift_truth"].max()

    pole = np.ones(len(FEATURES))
    radius = np.linalg.norm(M - pole, axis=1)

    print(f"[data] strategy rows={len(df)}  windows={windows.min()}-{windows.max()}"
          f"  drift windows={drift_windows}")

    # ---- write the plotted series (datetime-stamped records) -------------
    series = mean_f.copy()
    series.columns = [f"mean_{c}" for c in FEATURES]
    for c in FEATURES:
        series[f"std_{c}"] = std_f[c]
    series["radius_from_pole"] = radius
    series["phase"] = phase_of
    series["drift_truth"] = drift_of
    series.insert(0, "analysis_timestamp_utc", run_ts)
    series.insert(1, "run_id", run_id)
    series.insert(2, "tag", args.tag)
    series.insert(3, "strategy", args.strategy)
    series = series.reset_index().rename(columns={"index": "window"})
    csv_path = outdir / "e0_drift_plotted_series.csv"
    series.to_csv(csv_path, index=False)

    # ======================================================================
    # STATIC FIGURE -- 5 panels
    # ======================================================================
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    def mark_drift(ax):
        for w in drift_windows:
            ax.axvline(w, color="#d62728", ls="--", lw=1, alpha=0.7)

    # ---- Panel A: per-axis time series -----------------------------------
    axA = fig.add_subplot(gs[0, 0])
    for c in FEATURES:
        axA.plot(windows, mean_f[c], color=AXIS_COLOR[c], lw=2, label=c)
        axA.fill_between(windows, mean_f[c] - std_f[c], mean_f[c] + std_f[c],
                         color=AXIS_COLOR[c], alpha=0.12)
    mark_drift(axA)
    axA.set_title("A  Per-axis similarity over time", fontsize=11, weight="bold")
    axA.set_xlabel("window"); axA.set_ylabel("similarity  S_*(t)")
    axA.set_ylim(-0.05, 1.08)
    axA.legend(ncol=5, fontsize=8, loc="lower center")

    # ---- Panel B: radar / polar, one pentagon per phase ------------------
    axB = fig.add_subplot(gs[0, 1], projection="polar")
    n = len(FEATURES)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang_closed = np.concatenate([ang, ang[:1]])
    phases = list(dict.fromkeys(phase_of.tolist()))         # preserve order
    phase_colors = plt.cm.viridis(np.linspace(0, 0.85, len(phases)))
    for ph, col in zip(phases, phase_colors):
        wsel = phase_of.index[(phase_of == ph) & (drift_of == 0)]
        if len(wsel) == 0:
            wsel = phase_of.index[phase_of == ph]
        prof = mean_f.loc[wsel, FEATURES].mean().to_numpy()
        prof_closed = np.concatenate([prof, prof[:1]])
        axB.plot(ang_closed, prof_closed, color=col, lw=2, label=ph)
        axB.fill(ang_closed, prof_closed, color=col, alpha=0.10)
    axB.set_xticks(ang)
    axB.set_xticklabels(FEATURES, fontsize=9)
    axB.set_ylim(0, 1.05)
    axB.set_title("B  Similarity profile per phase (steady state)",
                  fontsize=11, weight="bold", pad=18)
    axB.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8)

    # ---- Panel C: PCA trajectory projection ------------------------------
    axC = fig.add_subplot(gs[0, 2])
    scores, frac, comps, _ = pca_2d(M)
    seg = np.stack([scores[:-1], scores[1:]], axis=1)
    lc = LineCollection(seg, cmap="plasma", lw=2)
    lc.set_array(windows[:-1])
    axC.add_collection(lc)
    axC.scatter(scores[:, 0], scores[:, 1], c=windows, cmap="plasma",
                s=28, zorder=3, edgecolor="white", linewidth=0.5)
    dmask = np.isin(windows, drift_windows)
    axC.scatter(scores[dmask, 0], scores[dmask, 1], s=150,
                facecolors="none", edgecolors="#d62728", linewidths=2,
                zorder=4, label="drift window")
    for i, w in enumerate(windows):
        if w in drift_windows or w == windows[0]:
            axC.annotate(f"w{w}", (scores[i, 0], scores[i, 1]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7)
    axC.set_title(f"C  Workload path in PCA space "
                  f"(PC1 {frac[0]*100:.1f}%, PC2 {frac[1]*100:.1f}%)",
                  fontsize=11, weight="bold")
    axC.set_xlabel("PC1"); axC.set_ylabel("PC2")
    axC.legend(fontsize=8, loc="best")
    axC.autoscale()

    # ---- Panel D: 3-D informative subspace (S_R, S_T, S_A) ---------------
    axD = fig.add_subplot(gs[1, 0], projection="3d")
    sr, st, sa = M[:, 0], M[:, 2], M[:, 3]
    axD.plot(sr, st, sa, color="#888888", lw=1.5)
    p = axD.scatter(sr, st, sa, c=windows, cmap="plasma", s=34,
                    edgecolor="white", linewidth=0.4)
    axD.scatter(sr[dmask], st[dmask], sa[dmask], s=160, facecolors="none",
                edgecolors="#d62728", linewidths=2)
    axD.set_xlabel("S_R"); axD.set_ylabel("S_T"); axD.set_zlabel("S_A")
    axD.set_title("D  Path in the 3 informative axes", fontsize=11,
                  weight="bold")

    # ---- Panel E: radius from pole ---------------------------------------
    axE = fig.add_subplot(gs[1, 1])
    axE.plot(windows, radius, color="#0a3d62", lw=2, marker="o", ms=4)
    axE.fill_between(windows, 0, radius, color="#0a3d62", alpha=0.12)
    mark_drift(axE)
    axE.set_title("E  Radius from no-drift pole  r(t)=||f(t)-1||",
                  fontsize=11, weight="bold")
    axE.set_xlabel("window"); axE.set_ylabel("radius")

    # ---- Panel F: text notes ---------------------------------------------
    axF = fig.add_subplot(gs[1, 2])
    axF.axis("off")
    pc1 = comps[0]
    notes = [
        f"run {run_id}",
        f"input: {in_path.name}",
        f"strategy: {args.strategy}    trials: "
        f"{df.groupby(TRIAL_KEYS).ngroups}",
        f"drift windows: {drift_windows}",
        "",
        "PC1 loadings (drift direction):",
        "   " + "  ".join(f"{c}={v:+.2f}" for c, v in zip(FEATURES, pc1)),
        "",
        "Reading guide:",
        " A  S_T plunges at every drift window;",
        "    S_V is flat (dead axis), S_P is noise.",
        " B  pentagon dents inward on the S_T spoke",
        "    as the workload changes phase.",
        " C  if the path moves mostly along PC1,",
        "    drift is ~rank-1 (one direction).",
        " D  the 3 signal axes trace a near-line",
        "    => they are highly correlated.",
        " E  radius spikes = a 1-D detector's signal.",
    ]
    axF.text(0.0, 1.0, "\n".join(notes), va="top", ha="left",
             fontsize=9, family="monospace", transform=axF.transAxes)

    fig.suptitle(
        f"Paper 3C  -  E0 workload drift across 5 HSM axes   "
        f"[{args.tag} / {args.strategy}]",
        fontsize=13, weight="bold")
    png_path = outdir / "e0_drift_fiveaxis.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[out ] {png_path}")

    # ======================================================================
    # ANIMATED RADAR  (window 1 -> last)
    # ======================================================================
    gif_path = outdir / "e0_drift_radar.gif"
    made_gif = False
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        figR, axR = plt.subplots(figsize=(5.6, 5.6),
                                 subplot_kw={"projection": "polar"})

        def draw(i):
            axR.clear()
            w = windows[i]
            prof = M[i]
            prof_closed = np.concatenate([prof, prof[:1]])
            is_drift = w in drift_windows
            col = "#d62728" if is_drift else "#0a3d62"
            axR.plot(ang_closed, prof_closed, color=col, lw=2.5)
            axR.fill(ang_closed, prof_closed, color=col, alpha=0.20)
            axR.set_xticks(ang)
            axR.set_xticklabels(FEATURES, fontsize=10)
            axR.set_ylim(0, 1.05)
            tag = "  <-- DRIFT" if is_drift else ""
            axR.set_title(f"window {w}  ({phase_of.loc[w]}){tag}",
                          fontsize=12, weight="bold",
                          color=col, pad=16)

        anim = FuncAnimation(figR, draw, frames=len(windows), interval=420)
        anim.save(gif_path, writer=PillowWriter(fps=2.4))
        plt.close(figR)
        made_gif = True
        print(f"[out ] {gif_path}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[warn] GIF writer unavailable ({exc}); writing frame sheet")
        key = [windows[0]] + drift_windows + [windows[-1]]
        key = sorted(set(key))
        cols = len(key)
        figR, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.2),
                                  subplot_kw={"projection": "polar"})
        if cols == 1:
            axes = [axes]
        for ax, w in zip(axes, key):
            i = int(np.where(windows == w)[0][0])
            prof = M[i]
            prof_closed = np.concatenate([prof, prof[:1]])
            is_drift = w in drift_windows
            col = "#d62728" if is_drift else "#0a3d62"
            ax.plot(ang_closed, prof_closed, color=col, lw=2)
            ax.fill(ang_closed, prof_closed, color=col, alpha=0.2)
            ax.set_xticks(ang); ax.set_xticklabels(FEATURES, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"w{w}", fontsize=10, color=col)
        gif_path = outdir / "e0_drift_radar_frames.png"
        figR.savefig(gif_path, dpi=140, bbox_inches="tight")
        plt.close(figR)
        print(f"[out ] {gif_path}")

    # ---- run metadata ----------------------------------------------------
    (outdir / "e0_drift_viz_run.json").write_text(json.dumps({
        "analysis_timestamp_utc": run_ts,
        "run_id": run_id,
        "tag": args.tag,
        "script": Path(__file__).name,
        "input_path": str(in_path),
        "strategy": args.strategy,
        "n_trials": int(df.groupby(TRIAL_KEYS).ngroups),
        "windows": [int(w) for w in windows],
        "drift_windows": drift_windows,
        "pca_var_fraction": [float(x) for x in frac],
        "pc1_loadings": {c: float(v) for c, v in zip(FEATURES, pc1)},
        "outputs": {
            "figure": png_path.name,
            "animation": gif_path.name,
            "series_csv": csv_path.name,
        },
    }, indent=2))

    print(f"[out ] {csv_path}")
    print(f"[out ] {outdir / 'e0_drift_viz_run.json'}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
