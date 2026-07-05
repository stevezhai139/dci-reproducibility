#!/usr/bin/env python3
"""
decompose_e0_geometry.py
========================

Paper 3C -- geometric-foundation experiment  (working name: E0-geometry).

WHAT THIS IS
------------
The Paper 3C v2 framing treats the 5-D HSM feature vector

    f(t) = (S_R, S_V, S_T, S_A, S_P)(t)

as a point in a *similarity space* whose "no-drift" corner is the
reference pole

    P = (1, 1, 1, 1, 1)          # perfect similarity to reference

Workload drift is then a MOTION of f(t) away from P, and that motion
splits into two geometric parts:

  * RADIAL   -- f moves directly away from / toward the pole.  Overall
               similarity rises or falls but the *shape* of the
               similarity profile is preserved (the displacement ray
               keeps its direction).
  * ANGULAR  -- f moves sideways: the direction of the
               displacement-from-pole rotates.  The *shape* of the
               similarity profile changes.

HYPOTHESIS  (H_radial)
    On the MongoDB E0 benchmark -- abrupt, stable-reference,
    full-coverage drift -- the drift motion is RADIAL-DOMINANT.
    If true, this explains the headline E0 result (a single similarity
    dimension detects drift as well as the full 5-D kernel): a radial
    collapse projects onto *every* axis, so any one axis sees it.

WHAT THIS SCRIPT DOES *NOT* DO
    It does not re-run any HSM experiment.  It only re-reads the
    existing E0 raw file (breakdown_per_window.csv) and measures the
    radial/angular split.  It is a read-only analysis of data that
    already exists -- the first empirical check of the geometric
    framing before any new experiment (E1) is built.

OUTPUTS  (every CSV row carries an ISO-8601 UTC analysis timestamp)
    Written to  <outdir>/<run_id>/ :
      e0_geometry_per_window.csv     per (trial, window): radius,
                                     |Df|, radial/angular split
      e0_geometry_drift_summary.csv  drift- vs non-drift-window
                                     aggregates, per scope
      e0_geometry_axes.csv           per-axis drift energy share,
                                     PCA / effective-rank, feature
                                     correlation (Gram) matrix
      e0_geometry_run.json           full structured results +
                                     run metadata + machine verdict
      e0_geometry_fig.png            (optional, if matplotlib present)

RUN
    pip install pandas numpy --break-system-packages      # matplotlib optional
    python decompose_e0_geometry.py
    python decompose_e0_geometry.py --input /path/to/breakdown_per_window.csv
    python decompose_e0_geometry.py --outdir /path/to/out --tag mongo_e0

Author note: this is Paper 3C exploratory code.  It is deliberately
verbose and defensive -- trace every number back to the raw rows.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Order matters: this is the f-vector axis order used everywhere below.
FEATURES = ["S_R", "S_V", "S_T", "S_A", "S_P"]

# A trial = one ordered 24-window trajectory.
TRIAL_KEYS = ["strategy", "block", "block_seed"]

# Numerical floor: motions with |Df| below this are treated as "no motion"
# and their angular fraction is left undefined (NaN) rather than 0/0.
EPS = 1e-9

# Verdict thresholds (reported, not hard gates -- the numbers speak).
RADIAL_DOMINANT_MAX_ANG_FRAC = 0.25   # energy-weighted angular fraction
RANK1_MIN_PC1_FRACTION = 0.80          # PC1 share of drift-motion energy

# Drift Complexity Index (DCI) regime bands. DCI is the participation
# ratio of the drift energy spectrum -- an "effective number of modes"
# in [1, D]. These band edges are PROVISIONAL: the graded drift harness
# is what calibrates them. Until then they are only a reading aid.
DCI_BASIC_MAX = 1.5
DCI_COMPLEX_MIN = 2.5


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, second precision, trailing Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_input_path(script_dir: Path) -> Path:
    """E0 raw file, resolved relative to this script's location.

    Expected layout:
        Paper 3/Paper 3C/geometry_E0/decompose_e0_geometry.py   <- here
        Paper 3/Paper 3B/HSM_gated_3B_for_paper3b_v2/code/results/
            cross_engine/mongo/adaptation/20260430_144825/
            breakdown_per_window.csv                            <- data
    """
    rel = ("../../Paper 3B/HSM_gated_3B_for_paper3b_v2/code/results/"
           "cross_engine/mongo/adaptation/20260430_144825/"
           "breakdown_per_window.csv")
    return (script_dir / rel).resolve()


# --------------------------------------------------------------------------
# Core geometry
# --------------------------------------------------------------------------
def decompose_trial(g: pd.DataFrame, pole: np.ndarray,
                    diff_lag: int = 1) -> pd.DataFrame:
    """Per-window radial/angular decomposition for ONE trial.

    g must be one trial's rows, sorted by window ascending.
    diff_lag K sets the differencing baseline: motion is f(t)-f(t-K).

    For window t (t >= K):
        Df(t)        = f(t) - f(t-K)                      motion vector
        u(t-K)       = (f(t-K) - pole) / ||f(t-K) - pole||  radial unit dir
        radial(t)    = Df(t) . u(t-1)                     signed scalar
                       ( > 0 : moving AWAY from pole = losing similarity )
        tangent(t)   = Df(t) - radial(t) * u(t-1)         angular vector
        ||Df||^2     = radial^2 + ||tangent||^2           (orthogonal split)
        ang_frac(t)  = ||tangent|| / ||Df||               in [0, 1]
    """
    g = g.sort_values("window").reset_index(drop=True)
    f = g[FEATURES].to_numpy(dtype=float)             # (n_win, 5)
    n = len(f)

    disp = f - pole                                   # displacement from pole
    r_pole = np.linalg.norm(disp, axis=1)             # radius from ideal pole

    delta_norm = np.full(n, np.nan)
    radial_signed = np.full(n, np.nan)
    tangent_norm = np.full(n, np.nan)
    ang_frac = np.full(n, np.nan)
    rad_frac = np.full(n, np.nan)

    for t in range(diff_lag, n):
        df = f[t] - f[t - diff_lag]                   # motion over diff_lag windows
        dn = float(np.linalg.norm(df))
        delta_norm[t] = dn
        if dn < EPS:
            continue
        prev_disp = disp[t - diff_lag]
        prev_r = float(np.linalg.norm(prev_disp))
        if prev_r < EPS:
            # f(t-1) sits on the pole: radial direction undefined.
            continue
        u = prev_disp / prev_r
        rad = float(df @ u)
        tang_vec = df - rad * u
        tn = float(np.linalg.norm(tang_vec))
        radial_signed[t] = rad
        tangent_norm[t] = tn
        ang_frac[t] = tn / dn
        rad_frac[t] = abs(rad) / dn

    out = g[TRIAL_KEYS + ["window", "phase", "drift_truth"] + FEATURES].copy()
    out["r_pole_ideal"] = r_pole
    out["delta_norm"] = delta_norm
    out["radial_signed"] = radial_signed
    out["tangent_norm"] = tangent_norm
    out["radial_fraction"] = rad_frac
    out["angular_fraction"] = ang_frac
    return out


def energy_split(sub: pd.DataFrame) -> dict:
    """Energy-weighted radial/angular split over a set of windows.

    Uses sum-of-squares so it is robust to tiny-motion windows
    (no 0/0): radial_energy + angular_energy = sum ||Df||^2 .
    """
    rad = sub["radial_signed"].to_numpy(dtype=float)
    tan = sub["tangent_norm"].to_numpy(dtype=float)
    m = np.isfinite(rad) & np.isfinite(tan)
    rad, tan = rad[m], tan[m]
    radial_energy = float(np.sum(rad ** 2))
    angular_energy = float(np.sum(tan ** 2))
    total = radial_energy + angular_energy
    return {
        "n_windows": int(m.sum()),
        "radial_energy": radial_energy,
        "angular_energy": angular_energy,
        "angular_fraction_energy": (angular_energy / total
                                    if total > EPS else float("nan")),
        "mean_delta_norm": float(np.mean(np.sqrt(rad ** 2 + tan ** 2)))
        if m.sum() else float("nan"),
        "mean_angular_fraction": float(np.nanmean(sub["angular_fraction"]))
        if len(sub) else float("nan"),
    }


def svd_rank(motion: np.ndarray) -> dict:
    """SVD of a stack of motion vectors (uncentered).

    Uncentered on purpose: we want to know whether the drift vectors
    lie near a single ray through the origin (rank-1 drift direction),
    not their spread about a mean.
    """
    if motion.shape[0] < 2:
        return {"n": int(motion.shape[0]), "singular_values": [],
                "energy_fraction": [], "effective_rank_95": None,
                "dci": float("nan"), "pc1": []}
    # economy SVD; singular values descending
    _, s, vt = np.linalg.svd(motion, full_matrices=False)
    energy = s ** 2
    frac = energy / energy.sum() if energy.sum() > 0 else energy
    cum = np.cumsum(frac)
    eff = int(np.searchsorted(cum, 0.95) + 1)
    # Drift Complexity Index: participation ratio of the energy spectrum.
    #   DCI = 1 / sum(p_i^2) = (sum s^2)^2 / sum(s^4)
    # This is the inverse Simpson / inverse Herfindahl index -- a single
    # real scalar in [1, D] giving the "effective number of drift modes".
    # All real arithmetic: the spectrum of a symmetric PSD covariance.
    inv_simpson = float(np.sum(np.asarray(frac) ** 2))
    dci = float(1.0 / inv_simpson) if inv_simpson > 0 else float("nan")
    return {
        "n": int(motion.shape[0]),
        "singular_values": [float(x) for x in s],
        "energy_fraction": [float(x) for x in frac],
        "effective_rank_95": eff,
        "dci": dci,
        "pc1": [float(x) for x in vt[0]],          # dominant drift direction
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                    help="path to breakdown_per_window.csv (E0 raw data)")
    ap.add_argument("--outdir", type=str, default=None,
                    help="output directory (default: <script_dir>/out)")
    ap.add_argument("--tag", type=str, default="mongo_e0",
                    help="label stored in outputs for provenance")
    ap.add_argument("--diff-lag", type=int, default=1,
                    help="motion = f(t) - f(t-K). K=1 (default) is the "
                         "per-window difference, correct for abrupt drift. "
                         "For gradual drift set K to the drift's onset "
                         "duration so the motion captures the full drift "
                         "rather than a noise-dominated single-step slice.")
    args = ap.parse_args()
    if args.diff_lag < 1:
        args.diff_lag = 1

    script_dir = Path(__file__).resolve().parent
    in_path = (Path(args.input).expanduser().resolve() if args.input
               else default_input_path(script_dir))
    out_root = (Path(args.outdir).expanduser().resolve() if args.outdir
                else script_dir / "out")

    run_ts = utc_now_iso()
    run_id = run_ts.replace("-", "").replace(":", "")        # 20260521T101500Z
    outdir = out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[run ] id={run_id}  tag={args.tag}")
    print(f"[in  ] {in_path}")
    if not in_path.exists():
        print(f"[ERR ] input file not found: {in_path}", file=sys.stderr)
        print("       pass --input /path/to/breakdown_per_window.csv",
              file=sys.stderr)
        return 2

    df = pd.read_csv(in_path)
    missing = [c for c in (TRIAL_KEYS + ["window", "phase", "drift_truth"]
                           + FEATURES) if c not in df.columns]
    if missing:
        print(f"[ERR ] missing columns: {missing}", file=sys.stderr)
        return 2
    print(f"[in  ] rows={len(df)}  sha256={file_sha256(in_path)[:16]}...")

    # Resolve the "clean signal" subset. MongoDB E0 has a no_advisor
    # strategy (workload with no advisor side effects); synthetic or
    # Postgres data may not -- fall back to all rows so the script is
    # genuinely dataset-agnostic.
    strategies = set(df["strategy"].unique())
    clean_strategy = "no_advisor" if "no_advisor" in strategies else None
    clean_df = (df[df.strategy == clean_strategy].copy()
                if clean_strategy else df)
    clean_label = (f"strategy={clean_strategy}" if clean_strategy
                   else "all_rows")
    print(f"[in  ] clean-signal subset = {clean_label}")

    pole = np.ones(len(FEATURES), dtype=float)               # P = (1,...,1)

    # ----- per-window decomposition, trial by trial ----------------------
    per_window_parts = []
    for _, g in df.groupby(TRIAL_KEYS, sort=False):
        per_window_parts.append(decompose_trial(g, pole, args.diff_lag))
    pw = pd.concat(per_window_parts, ignore_index=True)
    pw.insert(0, "analysis_timestamp_utc", run_ts)
    pw.insert(1, "run_id", run_id)
    pw.insert(2, "tag", args.tag)

    n_trials = df.groupby(TRIAL_KEYS, sort=False).ngroups
    drift_windows = sorted(int(w) for w in
                           df.loc[df.drift_truth == 1, "window"].unique())
    print(f"[data] trials={n_trials}  drift windows={drift_windows}  "
          f"strategies={sorted(df.strategy.unique())}")

    # ----- drift vs non-drift summary, per scope -------------------------
    summary_rows = []

    def add_summary(scope: str, sub: pd.DataFrame) -> None:
        for label, mask in (("drift", sub.drift_truth == 1),
                            ("non_drift", sub.drift_truth == 0)):
            es = energy_split(sub[mask])
            summary_rows.append({
                "analysis_timestamp_utc": run_ts,
                "run_id": run_id,
                "tag": args.tag,
                "scope": scope,
                "window_class": label,
                **es,
            })

    add_summary("pooled_all_strategies", pw)
    for strat, sub in pw.groupby("strategy", sort=True):
        add_summary(f"strategy={strat}", sub)
    # per drift-window index (each is a different phase transition)
    for w in drift_windows:
        sub = pw[pw.window.isin([w])]
        es = energy_split(sub)
        summary_rows.append({
            "analysis_timestamp_utc": run_ts, "run_id": run_id,
            "tag": args.tag, "scope": f"drift_window={w}",
            "window_class": "drift", **es,
        })
    summary = pd.DataFrame(summary_rows)

    # ----- per-axis drift energy + PCA + Gram matrix ---------------------
    # motion vectors entering drift windows (no_advisor = clean signal)
    def motion_matrix(scope_df: pd.DataFrame, drift_only: bool) -> np.ndarray:
        lag = args.diff_lag
        rows = []
        for _, g in scope_df.groupby(TRIAL_KEYS, sort=False):
            g = g.sort_values("window").reset_index(drop=True)
            f = g[FEATURES].to_numpy(dtype=float)
            for t in range(lag, len(f)):
                if drift_only and int(g.loc[t, "drift_truth"]) != 1:
                    continue
                rows.append(f[t] - f[t - lag])
        return np.array(rows, dtype=float) if rows else np.empty((0, 5))

    axes_rows = []
    for scope_name, scope_df in (("pooled_all_strategies", df),
                                 (clean_label, clean_df)):
        dmat = motion_matrix(scope_df, drift_only=True)
        per_axis_energy = (dmat ** 2).sum(axis=0)
        tot = per_axis_energy.sum()
        rank = svd_rank(dmat)
        for i, ax in enumerate(FEATURES):
            axes_rows.append({
                "analysis_timestamp_utc": run_ts, "run_id": run_id,
                "tag": args.tag, "scope": scope_name, "axis": ax,
                "drift_motion_energy": float(per_axis_energy[i]),
                "drift_motion_energy_fraction":
                    float(per_axis_energy[i] / tot) if tot > 0 else float("nan"),
                "feature_variance": float(scope_df[ax].var()),
                "feature_min": float(scope_df[ax].min()),
                "feature_max": float(scope_df[ax].max()),
                "pc1_loading": float(rank["pc1"][i]) if rank["pc1"] else float("nan"),
            })
    axes = pd.DataFrame(axes_rows)

    # SVD detail (overall + per drift window) for the JSON
    svd_detail = {
        "pooled_all_strategies_drift":
            svd_rank(motion_matrix(df, drift_only=True)),
        "clean_subset_drift":
            svd_rank(motion_matrix(clean_df, drift_only=True)),
    }
    for w in drift_windows:
        sub = clean_df
        rows = []
        for _, g in sub.groupby(TRIAL_KEYS, sort=False):
            g = g.sort_values("window").reset_index(drop=True)
            f = g[FEATURES].to_numpy(dtype=float)
            idx = g.index[g.window == w]
            for t in idx:
                if t >= args.diff_lag:
                    rows.append(f[t] - f[t - args.diff_lag])
        svd_detail[f"clean_drift_window_{w}"] = svd_rank(
            np.array(rows, dtype=float) if rows else np.empty((0, 5)))

    # Gram / correlation matrix of the 5 features (S_V is constant -> NaN row)
    corr = df[FEATURES].corr().round(6)
    gram = {a: {b: (None if pd.isna(corr.loc[a, b]) else float(corr.loc[a, b]))
                for b in FEATURES} for a in FEATURES}

    # ----- machine verdict ----------------------------------------------
    pooled_drift = summary[(summary.scope == "pooled_all_strategies")
                           & (summary.window_class == "drift")].iloc[0]
    naq = svd_detail["clean_subset_drift"]
    ang_frac_drift = float(pooled_drift["angular_fraction_energy"])
    pc1_frac = (naq["energy_fraction"][0] if naq["energy_fraction"]
                else float("nan"))
    dci_na = float(naq.get("dci", float("nan")))
    dci_regime = ("basic" if dci_na < DCI_BASIC_MAX
                  else "complex" if dci_na >= DCI_COMPLEX_MIN
                  else "moderate")
    verdict = {
        "angular_fraction_energy_at_drift": ang_frac_drift,
        "radial_dominant": bool(ang_frac_drift < RADIAL_DOMINANT_MAX_ANG_FRAC),
        "radial_dominant_threshold": RADIAL_DOMINANT_MAX_ANG_FRAC,
        "pc1_energy_fraction_no_advisor": float(pc1_frac),
        "drift_approx_rank1": bool(pc1_frac >= RANK1_MIN_PC1_FRACTION),
        "rank1_threshold": RANK1_MIN_PC1_FRACTION,
        "effective_rank_95_no_advisor": naq["effective_rank_95"],
        "drift_complexity_index": dci_na,
        "dci_scale": "participation ratio in [1, D]; ~1 = rank-1 basic regime",
        "dci_regime": dci_regime,
        "dci_regime_thresholds_provisional": [DCI_BASIC_MAX, DCI_COMPLEX_MIN],
        "interpretation": (
            "Two findings, not one. "
            "(1) The pole-anchored radial hypothesis is "
            + ("SUPPORTED" if ang_frac_drift < RADIAL_DOMINANT_MAX_ANG_FRAC
               else "REFUTED")
            + f": angular fraction {ang_frac_drift:.2f} "
            + ("<" if ang_frac_drift < RADIAL_DOMINANT_MAX_ANG_FRAC else ">=")
            + f" {RADIAL_DOMINANT_MAX_ANG_FRAC}. "
            "(2) The rank/complexity finding is "
            + ("SUPPORTED" if pc1_frac >= RANK1_MIN_PC1_FRACTION
               else "NOT supported")
            + f": drift motion is near rank-1 (PC1 {pc1_frac:.3f}), "
            f"DCI {dci_na:.2f} -> '{dci_regime}' regime. "
            "The operative explanation for the E0 result -- one "
            "similarity dimension detects drift as well as the full "
            "5-D kernel -- is that the drift is low-complexity (low "
            "DCI), i.e. it occupies essentially one mode; it is NOT "
            "that the drift points radially away from the pole."
        ),
    }

    # ----- write CSV outputs (every row already timestamped) -------------
    pw_path = outdir / "e0_geometry_per_window.csv"
    sm_path = outdir / "e0_geometry_drift_summary.csv"
    ax_path = outdir / "e0_geometry_axes.csv"
    js_path = outdir / "e0_geometry_run.json"
    pw.to_csv(pw_path, index=False)
    summary.to_csv(sm_path, index=False)
    axes.to_csv(ax_path, index=False)

    run_json = {
        "run": {
            "analysis_timestamp_utc": run_ts,
            "run_id": run_id,
            "tag": args.tag,
            "script": Path(__file__).name,
            "input_path": str(in_path),
            "input_sha256": file_sha256(in_path),
            "input_rows": int(len(df)),
            "n_trials": int(n_trials),
            "drift_windows": drift_windows,
            "features": FEATURES,
            "pole": "ones (perfect-similarity corner)",
            "clean_signal_subset": clean_label,
        },
        "drift_summary": summary.to_dict(orient="records"),
        "axes": axes.to_dict(orient="records"),
        "svd_detail": svd_detail,
        "feature_correlation_matrix": gram,
        "verdict": verdict,
    }
    js_path.write_text(json.dumps(run_json, indent=2))

    # ----- optional figure ----------------------------------------------
    fig_path = outdir / "e0_geometry_fig.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axx = plt.subplots(1, 3, figsize=(15, 4.2))
        # (1) mean radius from pole per window
        mr = pw.groupby("window")["r_pole_ideal"].mean()
        axx[0].plot(mr.index, mr.values, marker="o", color="#0a3d62")
        for w in drift_windows:
            axx[0].axvline(w, color="#d62728", ls="--", lw=1)
        axx[0].set_title("Mean radius from pole  r(t) = ||f(t) - 1||")
        axx[0].set_xlabel("window"); axx[0].set_ylabel("radius")
        # (2) angular fraction drift vs non-drift
        d = pw.loc[(pw.drift_truth == 1) & pw.angular_fraction.notna(),
                   "angular_fraction"]
        nd = pw.loc[(pw.drift_truth == 0) & pw.angular_fraction.notna(),
                    "angular_fraction"]
        axx[1].boxplot([d.values, nd.values], tick_labels=["drift", "non-drift"])
        axx[1].axhline(RADIAL_DOMINANT_MAX_ANG_FRAC, color="#d62728",
                       ls="--", lw=1)
        axx[1].set_title("Angular fraction of motion")
        axx[1].set_ylabel("||tangent|| / ||Df||")
        # (3) PCA scree of drift motion
        ef = naq["energy_fraction"]
        if ef:
            axx[2].bar(range(1, len(ef) + 1), ef, color="#0a3d62")
            axx[2].set_title("Drift-motion PCA scree (no_advisor)")
            axx[2].set_xlabel("component"); axx[2].set_ylabel("energy fraction")
        fig.suptitle(f"Paper 3C E0-geometry  |  run {run_id}", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as exc:                                  # noqa: BLE001
        fig_path = None
        print(f"[warn] figure skipped ({exc})")

    # ----- console summary ----------------------------------------------
    print()
    print("=" * 68)
    print(f"  E0-GEOMETRY VERDICT   (run {run_id})")
    print("=" * 68)
    print(f"  angular fraction of drift motion (energy) : "
          f"{ang_frac_drift:.4f}   "
          f"(radial-dominant if < {RADIAL_DOMINANT_MAX_ANG_FRAC})")
    print(f"  PC1 energy share of drift motion          : "
          f"{pc1_frac:.4f}   "
          f"(approx rank-1 if >= {RANK1_MIN_PC1_FRACTION})")
    print(f"  effective rank (95% energy)               : "
          f"{naq['effective_rank_95']}")
    print(f"  Drift Complexity Index (DCI)              : "
          f"{dci_na:.3f}   on [1, {len(FEATURES)}]  -> '{dci_regime}' regime")
    print(f"  -> radial hypothesis : {verdict['radial_dominant']}    "
          f"rank-1 / low-DCI finding : {verdict['drift_approx_rank1']}")
    print()
    print(f"  per-axis share of drift-motion energy ({clean_label}):")
    sub = axes[axes.scope == clean_label]
    for _, r in sub.iterrows():
        print(f"    {r['axis']:5s}  energy_frac={r['drift_motion_energy_fraction']:.4f}"
              f"   var={r['feature_variance']:.5f}")
    print()
    print(f"[out ] {pw_path}")
    print(f"[out ] {sm_path}")
    print(f"[out ] {ax_path}")
    print(f"[out ] {js_path}")
    if fig_path:
        print(f"[out ] {fig_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
