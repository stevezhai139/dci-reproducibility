#!/usr/bin/env python3
"""
drift_harness.py
================

Paper 3C -- E0-anchored graded drift harness  (the "caveat solver").

WHY THIS EXISTS
    Every Paper 3C empirical finding so far comes from ONE benchmark:
    the MongoDB E0 baseline (abrupt, stable-reference, full-coverage,
    one-way phase march). "Does it generalise?" cannot be answered by
    one benchmark -- but it CAN be answered by *mapping the boundary*.

    This harness takes the REAL MongoDB E0 phase profiles + noise as
    anchors and synthesises drift trajectories under four dials:

      c   coverage      -- per axis, probability it joins each drift
                           event.  c=1 -> every drift event moves the
                           same axes (rank-1, like E0); c<1 -> events
                           move varying subsets -> drift rank rises.
      r   speed         -- a drift event is spread over r windows as a
                           triangular onset/recovery.  r=1 = abrupt
                           (E0); r>1 = gradual.
      rho reference     -- the reference baseline performs a random
                           walk (isotropic Gaussian step, std rho, per
                           window): a fixed-reference detector sees the
                           steady baseline itself wander.  rho=0 = fixed
                           reference (E0).
      w   cyclicity     -- the phase order repeats w times.  w=1 = the
                           one-way march (E0); w>=2 = a cyclic workload
                           that loops (enables genuine winding).

    At the default cell (c=1, r=1, rho=0, w=1) the harness reproduces
    E0 statistically -- the sanity anchor.  Dial away and watch the
    Drift Complexity Index (DCI) and the 1-D-vs-5-D detection gap move.
    The output is a regime map: the MongoDB caveat turned into data.

NON-CIRCULARITY
    The harness CAN produce high-rank drift (dial c down).  Rank is an
    emergent consequence of coverage structure, never imposed.  If it
    only ever made rank-1 drift, "rank-1 holds" would prove nothing.

TWO MODES
    single  -- generate one cell, write a file in the exact schema of
               breakdown_per_window.csv so decompose_e0_geometry.py and
               visualize_e0_drift.py run on it UNCHANGED.
    sweep   -- sweep the dial grid; per cell compute DCI + 1-D/5-D
               detection AUC; write regime_map.csv (the caveat map).

OUTPUT  (timestamped folder; every CSV row carries a UTC timestamp)
    single :  synthetic_breakdown_<cell>.csv , harness_run.json
    sweep  :  regime_map.csv , regime_map_run.json , regime_map_fig.png

RUN
    pip install pandas numpy --break-system-packages        # mpl optional
    python drift_harness.py                                 # default sweep
    python drift_harness.py --mode single --c 0.6 --r 3
    python drift_harness.py --mode sweep --c-values 0.2,0.4,0.6,0.8,1.0 \
                            --r-values 1,2,3,5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
TRIAL_KEYS = ["strategy", "block", "block_seed"]
BASE_PHASES = ["edge", "geo", "text", "review"]
WIN_PER_PHASE = 6                       # E0 layout: 6 windows per phase
# S_V is the dead axis in E0 (constant) -- it never moves under any
# dial, including the reference random walk.
LIVE_AXES = np.array([1.0, 0.0, 1.0, 1.0, 1.0])   # S_R,S_V,S_T,S_A,S_P


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_input_path(script_dir: Path) -> Path:
    rel = ("../../Paper 3B/HSM_gated_3B_for_paper3b_v2/code/results/"
           "cross_engine/mongo/adaptation/20260430_144825/"
           "breakdown_per_window.csv")
    return (script_dir / rel).resolve()


# --------------------------------------------------------------------------
# 1. Extract anchors from the real E0 data
# --------------------------------------------------------------------------
def extract_anchors(df: pd.DataFrame) -> dict:
    """Steady profile + noise per phase, and the drift 'kick' vector.

    Uses the no_advisor strategy = clean signal, no advisor side effects.
    steady profile  : mean f over a phase's NON-drift windows
    noise (std)     : std  f over the same windows
    kick            : mean f at drift windows  minus  mean f at
                      non-drift windows  -- the transient similarity
                      collapse that IS the E0 drift signal.
    """
    d = df[df.strategy == "no_advisor"] if "no_advisor" in set(df.strategy) else df
    anchors, stds = {}, {}
    for ph in BASE_PHASES:
        sub = d[(d.phase == ph) & (d.drift_truth == 0)]
        anchors[ph] = sub[FEATURES].mean().to_numpy(dtype=float)
        stds[ph] = sub[FEATURES].std().fillna(0.0).to_numpy(dtype=float)
    drift_mean = d.loc[d.drift_truth == 1, FEATURES].mean().to_numpy(dtype=float)
    nondrift_mean = d.loc[d.drift_truth == 0, FEATURES].mean().to_numpy(dtype=float)
    kick = drift_mean - nondrift_mean          # negative on the live axes
    return {"anchors": anchors, "stds": stds, "kick": kick}


# --------------------------------------------------------------------------
# 2. Synthesise one trajectory
# --------------------------------------------------------------------------
def make_trajectory(cal: dict, c: float, r: int, rho: float, w: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    anchors, stds, kick = cal["anchors"], cal["stds"], cal["kick"]
    phases = BASE_PHASES * int(w)
    n_win = len(phases) * WIN_PER_PHASE
    # Reference drift: the reference baseline performs a random walk.
    # Each window it takes an isotropic Gaussian step (std rho) on the
    # four live axes; it wanders ~rho*sqrt(t), so a fixed-reference
    # detector sees the steady baseline itself move. rho=0 -> no walk.
    ref_walk = np.cumsum(
        rng.normal(0.0, rho, size=(n_win, len(FEATURES))) * LIVE_AXES, axis=0)
    rows = []
    t = 0
    for pi, ph in enumerate(phases):
        is_transition = pi > 0
        # coverage mask: decided ONCE per transition (Bernoulli c per axis)
        cov_mask = (rng.random(len(FEATURES)) < c).astype(float) if is_transition \
            else np.zeros(len(FEATURES))
        for wi in range(WIN_PER_PHASE):
            t += 1
            prof = anchors[ph].copy()
            prof = prof + ref_walk[t - 1]                    # drifting reference
            drift_flag = 0
            if is_transition and wi < r:                     # drift event
                mid = (r - 1) / 2.0
                shape = 1.0 if r == 1 else 1.0 - abs(wi - mid) / (mid + 1.0)
                prof = prof + kick * cov_mask * shape
                drift_flag = 1
            prof = prof + rng.normal(0.0, stds[ph])          # phase noise
            prof = np.clip(prof, 0.0, 1.0)
            rows.append((t, ph, drift_flag, *prof))
    out = pd.DataFrame(rows, columns=["window", "phase", "drift_truth"] + FEATURES)
    out["HSM"] = out[FEATURES].mean(axis=1)                  # 5-D aggregate
    return out


def make_cell(cal: dict, c: float, r: int, rho: float, w: int,
              n_trials: int, seed_base: int) -> pd.DataFrame:
    parts = []
    for k in range(n_trials):
        rng = np.random.default_rng(seed_base + k)
        tr = make_trajectory(cal, c, r, rho, w, rng)
        tr["strategy"] = "synthetic"
        tr["block"] = k
        tr["block_seed"] = seed_base + k
        parts.append(tr)
    df = pd.concat(parts, ignore_index=True)
    return df[TRIAL_KEYS + ["window", "phase", "drift_truth"] + FEATURES + ["HSM"]]


# --------------------------------------------------------------------------
# 3. Measuring sticks: DCI + detection AUC
# --------------------------------------------------------------------------
def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U). higher score => label 1."""
    labels = np.asarray(labels)
    m = np.isfinite(scores)
    labels, scores = labels[m], np.asarray(scores)[m]
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def drift_motion_dci(df: pd.DataFrame, lag: int = 1) -> dict:
    """DCI = participation ratio of the drift-window motion spectrum.

    Motion is f(t)-f(t-lag).  For gradual drift the per-window (lag=1)
    difference is noise-dominated -- the drift signal is spread thin
    across many windows -- so 'lag' should match the drift onset
    duration.  The sweep auto-sets lag = r for exactly this reason;
    without it DCI inflates from low SNR, not a genuine rank rise.
    """
    motions = []
    for _, g in df.groupby(TRIAL_KEYS, sort=False):
        g = g.sort_values("window").reset_index(drop=True)
        f = g[FEATURES].to_numpy(dtype=float)
        dt = g["drift_truth"].to_numpy()
        for t in range(lag, len(f)):
            if dt[t] == 1:
                motions.append(f[t] - f[t - lag])
    M = np.asarray(motions, dtype=float)
    if M.shape[0] < 2:
        return {"dci": float("nan"), "pc1_fraction": float("nan"),
                "effective_rank_95": None, "n_motions": int(M.shape[0])}
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    e = s ** 2
    p = e / e.sum() if e.sum() > 0 else e
    inv_simpson = float(np.sum(p ** 2))
    cum = np.cumsum(p)
    return {
        "dci": float(1.0 / inv_simpson) if inv_simpson > 0 else float("nan"),
        "pc1_fraction": float(p[0]),
        "effective_rank_95": int(np.searchsorted(cum, 0.95) + 1),
        "n_motions": int(M.shape[0]),
    }


def cell_metrics(df: pd.DataFrame, lag: int = 1) -> dict:
    g = drift_motion_dci(df, lag)
    g["diff_lag"] = int(lag)
    y = df["drift_truth"].to_numpy()
    # The three v3 detectors, all scored on the same windows:
    #   1-D static     : 1 - S_T            (simple/abrupt regime)
    #   rate-of-change : ||f(t)-f(t-lag)||   (gradual regime)
    #   5-D static     : 1 - HSM            (complex regime / full kernel)
    g["auc_1d_S_T"] = auc(y, 1.0 - df["S_T"].to_numpy())
    g["auc_5d_HSM"] = auc(y, 1.0 - df["HSM"].to_numpy())
    roc_scores, roc_labels = [], []
    for _, gg in df.groupby(TRIAL_KEYS, sort=False):
        gg = gg.sort_values("window").reset_index(drop=True)
        ff = gg[FEATURES].to_numpy(dtype=float)
        dd = gg["drift_truth"].to_numpy()
        for t in range(lag, len(ff)):
            roc_scores.append(float(np.linalg.norm(ff[t] - ff[t - lag])))
            roc_labels.append(int(dd[t]))
    g["auc_rate"] = (auc(np.array(roc_labels), np.array(roc_scores))
                     if roc_scores else float("nan"))
    g["auc_gap"] = g["auc_5d_HSM"] - g["auc_1d_S_T"]
    return g


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def parse_list(s: str, cast):
    return [cast(x) for x in str(s).split(",") if x != ""]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                    help="path to the real E0 breakdown_per_window.csv")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--mode", choices=["single", "sweep"], default="sweep")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=9000)
    # single-cell dial values
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=1)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--w", type=int, default=1)
    # sweep dial grids
    ap.add_argument("--c-values", type=str, default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--r-values", type=str, default="1,2,3,5")
    ap.add_argument("--rho-values", type=str, default="0.0")
    ap.add_argument("--w-values", type=str, default="1")
    ap.add_argument("--diff-lag", type=int, default=0,
                    help="DCI motion lag. 0 = auto (lag=r per cell, the "
                         "timescale-matched choice that removes the "
                         "speed/SNR confound); >0 = a fixed lag everywhere.")
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

    print(f"[run ] id={run_id}  mode={args.mode}")
    print(f"[in  ] {in_path}")
    if not in_path.exists():
        print(f"[ERR ] E0 input not found: {in_path}", file=sys.stderr)
        return 2
    df0 = pd.read_csv(in_path)
    cal = extract_anchors(df0)
    print(f"[anch] kick vector (drift transient) = "
          + ", ".join(f"{a}={v:+.3f}" for a, v in zip(FEATURES, cal["kick"])))

    if args.mode == "single":
        cell = make_cell(cal, args.c, args.r, args.rho, args.w,
                         args.n_trials, args.seed)
        cell.insert(0, "gen_timestamp_utc", run_ts)
        cell.insert(1, "run_id", run_id)
        lag = args.diff_lag if args.diff_lag > 0 else args.r
        m = cell_metrics(cell, lag)
        tag = f"c{args.c}_r{args.r}_rho{args.rho}_w{args.w}"
        path = outdir / f"synthetic_breakdown_{tag}.csv"
        cell.to_csv(path, index=False)
        (outdir / "harness_run.json").write_text(json.dumps({
            "analysis_timestamp_utc": run_ts, "run_id": run_id,
            "mode": "single", "input_path": str(in_path),
            "dials": {"c": args.c, "r": args.r, "rho": args.rho, "w": args.w},
            "n_trials": args.n_trials, "metrics": m,
            "output": path.name,
        }, indent=2))
        print(f"[cell] c={args.c} r={args.r} rho={args.rho} w={args.w}  "
              f"DCI={m['dci']:.3f}  auc_1d={m['auc_1d_S_T']:.3f}  "
              f"auc_5d={m['auc_5d_HSM']:.3f}")
        print(f"[out ] {path}")
        print(f"[out ] {outdir / 'harness_run.json'}")
        print("       -> feed this file to decompose_e0_geometry.py "
              "/ visualize_e0_drift.py via --input")
        print("[done]")
        return 0

    # ---- sweep mode ------------------------------------------------------
    c_vals = parse_list(args.c_values, float)
    r_vals = parse_list(args.r_values, int)
    rho_vals = parse_list(args.rho_values, float)
    w_vals = parse_list(args.w_values, int)
    print(f"[grid] c={c_vals}  r={r_vals}  rho={rho_vals}  w={w_vals}  "
          f"-> {len(c_vals)*len(r_vals)*len(rho_vals)*len(w_vals)} cells")

    rows = []
    cell_id = 0
    for w in w_vals:
        for rho in rho_vals:
            for r in r_vals:
                for c in c_vals:
                    cell = make_cell(cal, c, r, rho, w,
                                     args.n_trials, args.seed + cell_id * 1000)
                    lag = args.diff_lag if args.diff_lag > 0 else r
                    m = cell_metrics(cell, lag)
                    rows.append({
                        "analysis_timestamp_utc": run_ts, "run_id": run_id,
                        "cell_id": cell_id,
                        "c_coverage": c, "r_speed": r,
                        "rho_reference_drift": rho, "w_cyclicity": w,
                        "n_trials": args.n_trials, "diff_lag": m["diff_lag"],
                        "dci": m["dci"], "pc1_fraction": m["pc1_fraction"],
                        "effective_rank_95": m["effective_rank_95"],
                        "auc_1d_S_T": m["auc_1d_S_T"],
                        "auc_rate_of_change": m["auc_rate"],
                        "auc_5d_HSM": m["auc_5d_HSM"],
                        "auc_gap_5d_minus_1d": m["auc_gap"],
                    })
                    cell_id += 1
    regime = pd.DataFrame(rows)
    map_path = outdir / "regime_map.csv"
    regime.to_csv(map_path, index=False)

    # default-cell sanity check
    base = regime[(regime.c_coverage == 1.0) & (regime.r_speed == 1)
                  & (regime.rho_reference_drift == 0.0)
                  & (regime.w_cyclicity == 1)]
    (outdir / "regime_map_run.json").write_text(json.dumps({
        "analysis_timestamp_utc": run_ts, "run_id": run_id, "mode": "sweep",
        "input_path": str(in_path),
        "grid": {"c": c_vals, "r": r_vals, "rho": rho_vals, "w": w_vals},
        "n_trials": args.n_trials, "n_cells": len(regime),
        "kick_vector": {a: float(v) for a, v in zip(FEATURES, cal["kick"])},
        "default_cell_sanity": (base.iloc[0].to_dict() if len(base) else None),
        "output": map_path.name,
    }, indent=2, default=str))

    # optional heatmap if exactly two dials vary
    fig_path = None
    varying = [("c_coverage", c_vals), ("r_speed", r_vals),
               ("rho_reference_drift", rho_vals), ("w_cyclicity", w_vals)]
    varying = [(k, v) for k, v in varying if len(v) > 1]
    if len(varying) == 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            (xk, xv), (yk, yv) = varying
            piv = regime.pivot_table(index=yk, columns=xk, values="dci")
            fig, ax = plt.subplots(figsize=(7, 5.2))
            im = ax.imshow(piv.values, origin="lower", aspect="auto",
                           cmap="viridis")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(piv.columns)
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(piv.index)
            ax.set_xlabel(xk); ax.set_ylabel(yk)
            for (i, j), val in np.ndenumerate(piv.values):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
            fig.colorbar(im, label="Drift Complexity Index (DCI)")
            ax.set_title(f"Paper 3C regime map -- DCI over ({xk}, {yk})\n"
                         f"run {run_id}", fontsize=10)
            fig.tight_layout()
            fig_path = outdir / "regime_map_fig.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
        except Exception as exc:                              # noqa: BLE001
            print(f"[warn] heatmap skipped ({exc})")

    # console summary
    print()
    print("=" * 70)
    print(f"  REGIME MAP   ({len(regime)} cells, run {run_id})")
    print("=" * 70)
    if len(base):
        b = base.iloc[0]
        print(f"  default cell (c=1,r=1,rho=0,w=1)  -- E0 sanity anchor:")
        print(f"     DCI={b['dci']:.3f}   auc_1d={b['auc_1d_S_T']:.3f}   "
              f"auc_5d={b['auc_5d_HSM']:.3f}   (expect DCI~1.1, auc~1.0)")
    print()
    print(f"  {'c':>5} {'r':>3} {'rho':>6} {'w':>3} | {'DCI':>6} "
          f"{'auc_1d':>7} {'auc_roc':>7} {'auc_5d':>7}")
    print("  " + "-" * 58)
    for _, x in regime.iterrows():
        print(f"  {x['c_coverage']:>5.2f} {x['r_speed']:>3d} "
              f"{x['rho_reference_drift']:>6.3f} {x['w_cyclicity']:>3d} | "
              f"{x['dci']:>6.3f} "
              f"{x['auc_1d_S_T']:>7.3f} {x['auc_rate_of_change']:>7.3f} "
              f"{x['auc_5d_HSM']:>7.3f}")
    print()
    print(f"[out ] {map_path}")
    print(f"[out ] {outdir / 'regime_map_run.json'}")
    if fig_path:
        print(f"[out ] {fig_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
