#!/usr/bin/env python3
"""
hardness_router.py
==================

Paper 3C -- test whether routing should use the 2-vector (DCI, SNR)
rather than DCI alone, and find the operating point of the 1-D
detector. Runs BEFORE any DCI_THEORY v2 rewrite -- the theory is
refined only if the experiment says the 2-vector wins.

EXPERIMENT A -- does (DCI, SNR) beat DCI alone?
    Per workload cell measure two coordinates -- DCI (drift rank /
    participation ratio) and SNR (drift signal strength) -- and the
    ground truth: is the 1-D detector sufficient, i.e. the 5-D-over-
    1-D AUC gap is within the per-cell DeLong epsilon (delong.py).
    Then compare, as predictors of that label:
      - DCI alone        : ROC AUC of (-DCI) vs the label;
      - (DCI, SNR) joint : ROC AUC of a cross-validated logistic
                           regression on the 2-vector.
    If the joint ROC AUC clearly exceeds DCI-alone, the 2-vector wins.
    The fitted decision boundary defines a single principled HARDNESS
    scalar h (the boundary normal) -- the "metric vector reduced to a
    value", derived not guessed.

EXPERIMENT B -- the 1-D operating point.
    Stratify cells by hardness h; at each level compare the 1-D and
    5-D detectors. The 1-D-sufficiency FRONTIER is the hardness h* at
    which the 5-D-over-1-D gap first exceeds the per-cell DeLong
    epsilon -- the optimal point up to which 1-D may be deployed.

Detectors (matching cost_benefit.py / sketch v5): both read the
position deviation d(t) = f(t) - f_steady, single-edged on a
transient-dip drift (no false alarm on recovery, unlike ||Df||).
f_steady is the per-trial mean over non-drift windows.
  1-D : the single most-informative similarity axis, score -d_j.
  5-D : the Mahalanobis distance d^T Sigma^-1 d under the steady-
        window covariance -- the v5 canonical 5-D detector.
The sufficiency tolerance is the per-cell DeLong (1988) 95% noise
floor of the 5-D-over-1-D AUC gap -- data-derived, no hand-set
constant.

OUTPUT (timestamped; every CSV row carries a UTC timestamp)
    hardness_router_cells.csv   per cell: DCI, SNR, auc_1d/5d, h, label
    hardness_router_run.json    Experiment A + B results + verdict
    hardness_router_fig.png     the (DCI,SNR) plane + the frontier

RUN
    pip install pandas numpy --break-system-packages          # mpl optional
    python hardness_router.py
    # drift_harness.py and energy_gap.py must sit alongside.
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
from delong import delong_epsilon    # noqa: E402


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Tiny logistic regression (numpy only -- no sklearn dependency)
# --------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logistic_fit(X: np.ndarray, y: np.ndarray,
                 iters: int = 4000, lr: float = 0.3) -> np.ndarray:
    """Fit logistic regression by gradient descent. X assumed standardised.
    Returns weights [intercept, w_1, ...]."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        grad = Xb.T @ (_sigmoid(Xb @ w) - y) / len(y)
        w -= lr * grad
    return w


def logistic_prob(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ w)


def kfold_cv_prob(X: np.ndarray, y: np.ndarray, k: int = 5,
                  seed: int = 0) -> np.ndarray:
    """Cross-validated P(y=1) -- honest, no train-on-test."""
    n = len(y)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    pred = np.zeros(n)
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        w = logistic_fit(X[tr], y[tr])
        pred[te] = logistic_prob(X[te], w)
    return pred


# --------------------------------------------------------------------------
# Static similarity-deviation detectors (v5: 1-D best axis, 5-D Mahalanobis)
# --------------------------------------------------------------------------
def static_cell_analysis(df: pd.DataFrame) -> dict | None:
    """Per-cell DCI, SNR, and the v5 detector AUCs + DeLong epsilon.

    d(t) = f(t) - f_steady, f_steady the per-trial mean over non-drift
    windows; both detectors read the *position* deviation, single-edged
    on a transient dip. DCI = participation ratio of the drift-window
    deviation covariance. The detectors match cost_benefit.py:
      1-D : the single most-informative similarity axis, score -d_j;
      5-D : the Mahalanobis distance d^T Sigma^-1 d under the steady-
            window covariance -- the v5 canonical 5-D detector.
    epsilon is the per-cell DeLong (1988) 95% noise floor of the 5-D-
    over-1-D AUC gap (delong.py) -- data-derived, no hand-set constant.
    """
    alld, lab = [], []
    for _, g in df.groupby(H.TRIAL_KEYS, sort=False):
        g = g.sort_values("window").reset_index(drop=True)
        f = g[H.FEATURES].to_numpy(float)
        dt = g["drift_truth"].to_numpy()
        nd = dt == 0
        if nd.sum() < 2:
            continue
        f_steady = f[nd].mean(axis=0)
        for t in range(len(f)):
            alld.append(f[t] - f_steady)
            lab.append(int(dt[t]))
    alld = np.asarray(alld, float)
    lab = np.asarray(lab, int)
    if (lab == 1).sum() < 3 or (lab == 0).sum() < 3:
        return None
    # DCI: participation ratio of the drift-window deviation covariance
    driftd = alld[lab == 1]
    C = (driftd.T @ driftd) / driftd.shape[0]
    evals = np.clip(np.linalg.eigvalsh(C), 0, None)
    tot = float(evals.sum())
    if tot <= 0:
        return None
    p = evals / tot
    dci = float(1.0 / np.sum(p ** 2))
    # SNR: drift vs steady deviation energy
    e_d = float(np.mean(np.sum(alld[lab == 1] ** 2, axis=1)))
    e_n = float(np.mean(np.sum(alld[lab == 0] ** 2, axis=1)))
    snr = e_d / e_n if e_n > 0 else float("nan")
    # 1-D detector: the single most-informative similarity axis
    # (drift lowers similarity, so -d_j rises on drift)
    per_dim = {H.FEATURES[j]: H.auc(lab, -alld[:, j])
               for j in range(len(H.FEATURES))}
    best_dim = max(per_dim, key=lambda d: per_dim[d])
    bj = H.FEATURES.index(best_dim)
    auc_1d = per_dim[best_dim]
    # 5-D detector: Mahalanobis distance under the steady-window covariance
    nd = lab == 0
    Sig = np.cov(alld[nd].T) + 1e-6 * np.eye(alld.shape[1])
    Pinv = np.linalg.pinv(Sig)
    maha = np.einsum("ij,jk,ik->i", alld, Pinv, alld)
    auc_5d = H.auc(lab, maha)
    # per-cell DeLong epsilon of the 5-D-over-1-D AUC gap
    de = delong_epsilon(lab, -alld[:, bj], maha)
    return {"dci": dci, "snr": snr, "auc_1d": auc_1d, "auc_5d": auc_5d,
            "best_dim": best_dim, "gap": auc_5d - auc_1d,
            "epsilon": de["epsilon"]}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--c-values", type=str,
                    default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--r-values", type=str, default="1,2,3,5")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.03,
                    help="(DCI,SNR) 'wins' if its ROC AUC beats "
                         "DCI-alone by at least this margin")
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

    # ---- generate cells -> per-cell (DCI, SNR, auc_1d, auc_5d) ----------
    rows = []
    cidx = 0
    for r in r_vals:
        for c in c_vals:
            for b in range(args.batches):
                seed = 70000 + cidx * 1000 + b
                df = H.make_cell(cal, c, r, 0.0, 1, args.n_trials, seed)
                res = static_cell_analysis(df)
                if res is None or not np.isfinite(res["snr"]):
                    continue
                auc_1d, auc_5d = res["auc_1d"], res["auc_5d"]
                if not (np.isfinite(auc_1d) and np.isfinite(auc_5d)):
                    continue
                rows.append({
                    "analysis_timestamp_utc": run_ts, "run_id": run_id,
                    "c_coverage": c, "r_speed": r, "batch": b,
                    "dci": res["dci"], "snr": res["snr"],
                    "auc_1d": auc_1d, "auc_5d": auc_5d,
                    "best_dim": res["best_dim"],
                    "gap": res["gap"], "epsilon": res["epsilon"],
                    "sufficient_1d": int(res["gap"] <= res["epsilon"]),
                })
            cidx += 1
    C = pd.DataFrame(rows)
    if len(C) < 30:
        print(f"[ERR ] only {len(C)} usable cells", file=sys.stderr)
        return 2

    # ====================================================================
    # EXPERIMENT A
    # ====================================================================
    y = C["sufficient_1d"].to_numpy()
    dci = C["dci"].to_numpy()
    snr = C["snr"].to_numpy()
    # DCI alone: lower DCI -> more likely sufficient
    dci_roc = H.auc(y, -dci)
    # (DCI, SNR) joint, cross-validated logistic regression
    Xraw = np.column_stack([dci, snr])
    mu, sd = Xraw.mean(0), Xraw.std(0)
    sd[sd == 0] = 1.0
    Xstd = (Xraw - mu) / sd
    cv_p = kfold_cv_prob(Xstd, y.astype(float), k=5, seed=1)
    joint_roc = H.auc(y, cv_p)
    # full-data fit -> decision boundary -> hardness scalar
    w = logistic_fit(Xstd, y.astype(float))
    # hardness h = negative log-odds of sufficiency (high h = hard);
    # expressed back in raw (DCI, SNR) units.
    w_dci_raw = -w[1] / sd[0]
    w_snr_raw = -w[2] / sd[1]
    b_raw = -w[0] + w[1] * mu[0] / sd[0] + w[2] * mu[1] / sd[1]
    C["hardness"] = b_raw + w_dci_raw * dci + w_snr_raw * snr
    a_wins = bool(joint_roc - dci_roc >= args.margin)

    # ====================================================================
    # EXPERIMENT B -- 1-D operating point along hardness
    # ====================================================================
    C = C.sort_values("hardness").reset_index(drop=True)
    qs = np.quantile(C["hardness"], np.linspace(0, 1, 9))
    qs[-1] += 1e-9
    C["h_level"] = pd.cut(C["hardness"], np.unique(qs),
                          include_lowest=True, duplicates="drop")
    strata = (C.groupby("h_level", observed=True)
              .agg(h_mid=("hardness", "mean"),
                   auc_1d=("auc_1d", "mean"),
                   auc_5d=("auc_5d", "mean"),
                   gap=("gap", "mean"),
                   epsilon=("epsilon", "mean"),
                   frac_suff=("sufficient_1d", "mean"),
                   n=("hardness", "size")).reset_index())
    # frontier h*: highest hardness whose stratum is still 1-D-sufficient
    # (mean gap within the stratum's mean per-cell DeLong epsilon)
    suff_strata = strata[strata["gap"] <= strata["epsilon"]]
    h_star = float(suff_strata["h_mid"].max()) if len(suff_strata) else float("nan")
    one_d_usable_frac = float((C["hardness"] <= h_star).mean()) \
        if np.isfinite(h_star) else 0.0

    # ---- verdict --------------------------------------------------------
    if a_wins:
        verdictA = (f"(DCI, SNR) WINS: joint ROC AUC {joint_roc:.3f} beats "
                    f"DCI-alone {dci_roc:.3f} by {joint_roc-dci_roc:+.3f} "
                    f">= margin {args.margin}. Routing should use the "
                    f"2-vector; DCI_THEORY v2 should adopt (DCI, SNR). "
                    f"Hardness h = {b_raw:.3f} + {w_dci_raw:.3f}*DCI + "
                    f"{w_snr_raw:.3f}*SNR.")
    else:
        verdictA = (f"(DCI, SNR) does NOT clearly beat DCI alone: joint "
                    f"ROC AUC {joint_roc:.3f} vs DCI-alone {dci_roc:.3f} "
                    f"(+{joint_roc-dci_roc:.3f} < margin {args.margin}). "
                    f"DCI alone is an adequate router; keep it 1-D.")
    verdictB = (f"1-D operating point: the 5-D-over-1-D gap stays within "
                f"the per-cell DeLong epsilon up to hardness h* = "
                f"{h_star:.3f}, which covers {one_d_usable_frac:.0%} of "
                f"cells on this grid."
                if np.isfinite(h_star) else
                "No hardness stratum is 1-D-sufficient (gap exceeds the "
                "DeLong epsilon everywhere): 1-D never matches 5-D on "
                "this grid.")

    run_json = {
        "run": {"analysis_timestamp_utc": run_ts, "run_id": run_id,
                "script": Path(__file__).name, "n_cells": int(len(C)),
                "epsilon": "per-cell DeLong 95% noise floor (delong.py)",
                "margin": args.margin,
                "grid": {"c": c_vals, "r": r_vals, "batches": args.batches}},
        "experiment_A": {
            "dci_alone_roc_auc": float(dci_roc),
            "dci_snr_joint_roc_auc": float(joint_roc),
            "improvement": float(joint_roc - dci_roc),
            "two_vector_wins": a_wins,
            "hardness_weights_raw": {"intercept": float(b_raw),
                                     "w_dci": float(w_dci_raw),
                                     "w_snr": float(w_snr_raw)},
            "base_rate_1d_sufficient": float(y.mean()),
        },
        "experiment_B": {
            "strata": strata.assign(
                h_level=strata["h_level"].astype(str)).to_dict("records"),
            "frontier_h_star": h_star,
            "one_d_usable_fraction": one_d_usable_frac,
        },
        "verdict_A": verdictA,
        "verdict_B": verdictB,
    }
    C.assign(h_level=C["h_level"].astype(str)).to_csv(
        outdir / "hardness_router_cells.csv", index=False)
    (outdir / "hardness_router_run.json").write_text(
        json.dumps(run_json, indent=2, default=str))

    # ---- figure ---------------------------------------------------------
    fig_path = outdir / "hardness_router_fig.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
        # Panel A: the (DCI, SNR) plane
        suff = y == 1
        ax[0].scatter(dci[suff], snr[suff], s=24, c="#1d9e75",
                      label="1-D sufficient", alpha=0.8)
        ax[0].scatter(dci[~suff], snr[~suff], s=24, c="#d62728",
                      label="needs 5-D", alpha=0.8)
        gx = np.linspace(dci.min(), dci.max(), 100)
        if abs(w_snr_raw) > 1e-9:                       # boundary line h=0
            gy = -(b_raw + w_dci_raw * gx) / w_snr_raw
            ax[0].plot(gx, gy, color="#0a3d62", lw=2,
                       label="decision boundary")
        ax[0].set_xlabel("DCI"); ax[0].set_ylabel("drift SNR")
        ax[0].set_ylim(snr.min() - 0.2, np.percentile(snr, 97))
        ax[0].set_title(f"A  (DCI,SNR) plane   DCI-alone ROC "
                        f"{dci_roc:.3f}  ->  joint ROC {joint_roc:.3f}",
                        fontsize=10)
        ax[0].legend(fontsize=8)
        # Panel B: the 1-D frontier along hardness
        ax[1].plot(strata["h_mid"], strata["auc_5d"], marker="o",
                   color="#0a3d62", label="5-D detector")
        ax[1].plot(strata["h_mid"], strata["auc_1d"], marker="o",
                   color="#d62728", label="1-D detector")
        ax[1].fill_between(strata["h_mid"], strata["auc_1d"],
                           strata["auc_5d"], color="#d62728", alpha=0.12)
        if np.isfinite(h_star):
            ax[1].axvline(h_star, color="#1d9e75", ls="--",
                          label=f"1-D operating point h*={h_star:.2f}")
        ax[1].set_xlabel("hardness  h"); ax[1].set_ylabel("detection AUC")
        ax[1].set_title("B  1-D-sufficiency frontier", fontsize=10)
        ax[1].legend(fontsize=8)
        fig.suptitle(f"Paper 3C  -  hardness router   run {run_id}",
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
    print(f"  HARDNESS ROUTER   run {run_id}")
    print("=" * 70)
    print(f"  cells: {len(C)}   base rate 1-D-sufficient: {y.mean():.3f}")
    print()
    print("  EXPERIMENT A  -- (DCI, SNR)  vs  DCI alone")
    print(f"    DCI alone        ROC AUC : {dci_roc:.3f}")
    print(f"    (DCI, SNR) joint ROC AUC : {joint_roc:.3f}   "
          f"(+{joint_roc-dci_roc:.3f})")
    print(f"    -> {verdictA}")
    print()
    print("  EXPERIMENT B  -- 1-D vs 5-D along hardness")
    print(f"    {'h_mid':>7} {'auc_1d':>7} {'auc_5d':>7} {'gap':>7} "
          f"{'frac_suff':>10} {'n':>4}")
    for _, s in strata.iterrows():
        print(f"    {s['h_mid']:>7.3f} {s['auc_1d']:>7.3f} "
              f"{s['auc_5d']:>7.3f} {s['gap']:>7.3f} "
              f"{s['frac_suff']:>10.2f} {int(s['n']):>4}")
    print(f"    -> {verdictB}")
    print()
    for f in ("hardness_router_cells.csv", "hardness_router_run.json"):
        print(f"[out ] {outdir / f}")
    if fig_path:
        print(f"[out ] {fig_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
