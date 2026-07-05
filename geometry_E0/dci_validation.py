#!/usr/bin/env python3
"""
dci_validation.py
=================

Paper 3C -- validate the Drift Complexity Index (DCI) before it is used
to route anything.  RQ2.  This experiment gates RQ1 (the regime map)
and RQ3 (the selector): if DCI is not a reliable and valid router,
everything built on it is built on sand.

Two questions, matching "DCI yang naen-mae" (solid / accurate):

  RELIABILITY ("naen" -- solid).  Is the DCI *measurement* stable?
    For each cell, estimate DCI from B independent seed-batches and
    measure its spread (std, coefficient of variation). Separately,
    sweep n_trials to see how fast the spread shrinks -- this tells us
    how much data a trustworthy DCI reading needs.

  VALIDITY ("mae" -- accurate).  Does DCI predict the thing it must?
    The selector uses DCI to decide "is the cheap 1-D detector
    sufficient here?".  Ground truth, defined NON-circularly from
    detection performance (not from DCI): a cell is "1-D sufficient"
    iff  auc_1d  >=  auc_5d - eps.  We then ask how well DCI separates
    1-D-sufficient cells from the rest:
      - DCI-as-router ROC AUC  (threshold-free quality of DCI as a
        binary router; ~1.0 = perfect, ~0.5 = useless);
      - the routing threshold that keeps mis-routing below a safety
        bound, and the resulting 1-D-usage / cost saving;
      - Spearman rank-correlation of DCI vs the 1-D-vs-5-D AUC gap.
    Plus a construct check: at cells where rank-1 drift was *injected*
    (coverage c = 1), does DCI actually read ~1?

It reuses drift_harness.py for trajectory generation and metrics, so
the DCI under test is exactly the DCI that would be deployed.

OUTPUT (timestamped folder; every CSV row carries a UTC timestamp):
    dci_validation_samples.csv   one row per (cell, batch): DCI + AUCs
    dci_reliability.csv          per cell: DCI mean/std/CV across batches
    dci_ntrials_sensitivity.csv  DCI spread vs n_trials
    dci_validation_run.json      headline verdict + metadata
    dci_validation_fig.png       (if matplotlib present)

RUN
    pip install pandas numpy --break-system-packages        # mpl optional
    python dci_validation.py
    python dci_validation.py --batches 8 --eps 0.02
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

# import the harness machinery (must be in the same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_harness as H            # noqa: E402


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks). No scipy needed."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def parse_list(s: str, cast):
    return [cast(v) for v in str(s).split(",") if v != ""]


# --------------------------------------------------------------------------
def run_cell(cal: dict, c: float, r: int, n_trials: int,
             seed_base: int) -> dict:
    """Generate one cell and return its DCI + 3 detector AUCs.

    diff-lag is set to r (timescale-matched), exactly as the harness
    sweep does -- so the DCI under test is the deployed DCI.
    """
    df = H.make_cell(cal, c, r, 0.0, 1, n_trials, seed_base)
    m = H.cell_metrics(df, lag=r)
    return {
        "dci": m["dci"],
        "pc1_fraction": m["pc1_fraction"],
        "effective_rank_95": m["effective_rank_95"],
        "auc_1d": m["auc_1d_S_T"],
        "auc_rate": m["auc_rate"],
        "auc_5d": m["auc_5d_HSM"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                    help="path to the real E0 breakdown_per_window.csv")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--c-values", type=str,
                    default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--r-values", type=str, default="1,2,3,5")
    ap.add_argument("--batches", type=int, default=8,
                    help="independent seed-batches per cell (reliability)")
    ap.add_argument("--n-trials", type=int, default=10,
                    help="trials per cell-batch")
    ap.add_argument("--ntrials-values", type=str, default="5,10,20,40",
                    help="n_trials values for the sensitivity sweep")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="sufficiency tolerance: 1-D sufficient iff "
                         "auc_1d >= auc_5d - eps")
    ap.add_argument("--safety", type=float, default=0.95,
                    help="min routing precision for the reported threshold")
    ap.add_argument("--seed", type=int, default=20260,
                    help="base seed")
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
    print(f"[in  ] {in_path}")
    if not in_path.exists():
        print(f"[ERR ] E0 input not found: {in_path}", file=sys.stderr)
        return 2
    df0 = pd.read_csv(in_path)
    cal = H.extract_anchors(df0)

    c_vals = parse_list(args.c_values, float)
    r_vals = parse_list(args.r_values, int)
    B = args.batches
    print(f"[grid] {len(c_vals)} c x {len(r_vals)} r x {B} batches "
          f"= {len(c_vals)*len(r_vals)*B} cell-runs  (n_trials={args.n_trials})")

    # ---- PART A: main sweep -- each (cell, batch) is one sample ----------
    samples = []
    cell_idx = 0
    for r in r_vals:
        for c in c_vals:
            for b in range(B):
                seed_base = args.seed + cell_idx * 10000 + b * 1000
                res = run_cell(cal, c, r, args.n_trials, seed_base)
                samples.append({
                    "analysis_timestamp_utc": run_ts, "run_id": run_id,
                    "c_coverage": c, "r_speed": r, "batch": b,
                    "n_trials": args.n_trials, **res,
                })
            cell_idx += 1
    S = pd.DataFrame(samples)
    S["gap"] = S["auc_5d"] - S["auc_1d"]
    S["sufficient_1d"] = (S["auc_1d"] >= S["auc_5d"] - args.eps).astype(int)
    S.to_csv(outdir / "dci_validation_samples.csv", index=False)

    # ---- PART B: reliability -- DCI spread per cell across batches -------
    rel_rows = []
    for (c, r), g in S.groupby(["c_coverage", "r_speed"]):
        d = g["dci"].to_numpy()
        rel_rows.append({
            "analysis_timestamp_utc": run_ts, "run_id": run_id,
            "c_coverage": c, "r_speed": r, "n_batches": len(d),
            "dci_mean": float(np.mean(d)), "dci_std": float(np.std(d, ddof=1)),
            "dci_cv": float(np.std(d, ddof=1) / np.mean(d))
            if np.mean(d) > 0 else float("nan"),
            "dci_min": float(np.min(d)), "dci_max": float(np.max(d)),
            "auc_1d_std": float(np.std(g["auc_1d"], ddof=1)),
            "auc_5d_std": float(np.std(g["auc_5d"], ddof=1)),
        })
    REL = pd.DataFrame(rel_rows)
    REL.to_csv(outdir / "dci_reliability.csv", index=False)
    rel_cv_median = float(np.median(REL["dci_cv"]))

    # ---- PART C: n_trials sensitivity -----------------------------------
    nt_vals = parse_list(args.ntrials_values, int)
    sens_cells = [(1.0, 1), (0.6, 1), (0.3, 1)]
    sens_rows = []
    for (c, r) in sens_cells:
        for nt in nt_vals:
            d = []
            for b in range(B):
                seed_base = args.seed + 900000 + b * 1000 + nt
                d.append(run_cell(cal, c, r, nt, seed_base)["dci"])
            d = np.array(d)
            sens_rows.append({
                "analysis_timestamp_utc": run_ts, "run_id": run_id,
                "c_coverage": c, "r_speed": r, "n_trials": nt,
                "dci_mean": float(np.mean(d)),
                "dci_std": float(np.std(d, ddof=1)),
            })
    SENS = pd.DataFrame(sens_rows)
    SENS.to_csv(outdir / "dci_ntrials_sensitivity.csv", index=False)

    # ---- PART D: validity ------------------------------------------------
    y = S["sufficient_1d"].to_numpy()
    dci = S["dci"].to_numpy()
    # DCI-as-router ROC AUC: low DCI should predict 1-D-sufficient,
    # so the score that should rank with the positive label is -DCI.
    router_auc = H.auc(y, -dci)
    rho_dci_gap = spearman(dci, S["gap"].to_numpy())

    # threshold scan: route DCI < theta -> 1-D, else -> 5-D.
    # precision = of cells routed to 1-D, fraction actually sufficient.
    # usage     = fraction of cells routed to 1-D (the cost saving).
    thetas = np.unique(np.round(dci, 4))
    scan = []
    for th in thetas:
        routed_1d = dci < th
        n_routed = int(routed_1d.sum())
        prec = float(y[routed_1d].mean()) if n_routed else float("nan")
        scan.append((float(th), n_routed / len(dci), prec, n_routed))
    # the largest threshold whose precision still meets the safety bound
    safe = [s for s in scan if s[3] >= 5 and np.isfinite(s[2])
            and s[2] >= args.safety]
    theta_safe = max(safe, key=lambda s: s[0]) if safe else None
    # also the plain best-accuracy threshold
    def routing_acc(th):
        r1 = dci < th
        return float(((r1 & (y == 1)) | (~r1)).mean())
    best_acc_theta = max(thetas, key=routing_acc) if len(thetas) else float("nan")

    base_rate = float(y.mean())          # share of cells that ARE 1-D-suff.
    dci_at_c1 = float(S.loc[S.c_coverage == 1.0, "dci"].mean())
    dci_at_clow = float(S.loc[S.c_coverage <= 0.4, "dci"].mean())
    rate_ever_best = bool(
        ((S["auc_rate"] >= S["auc_5d"] - args.eps) &
         (S["auc_1d"] < S["auc_5d"] - args.eps)).any())

    if theta_safe is not None:
        usage = theta_safe[1]
        mean_cost = usage * 1.0 + (1.0 - usage) * 50.0
        cost_vs_full = 50.0 / mean_cost
    else:
        usage = mean_cost = cost_vs_full = float("nan")

    reliable = rel_cv_median <= 0.12
    has_headroom = bool(np.isfinite(usage) and usage >= 0.15)
    if router_auc >= 0.85 and reliable and has_headroom:
        verdict = ("DCI VALIDATED: it separates 1-D-sufficient cells "
                   "well, is a stable measurement, and a safety-bounded "
                   "threshold still routes a useful share of cells to "
                   "the cheap detector.")
    elif router_auc >= 0.78 and reliable:
        verdict = (f"DCI carries real routing signal (router ROC AUC "
                   f"{router_auc:.2f}) and is a stable measurement (CV "
                   f"{rel_cv_median:.2f}) -- BUT on this uniform grid the "
                   f"1-D-sufficient region is a thin slice: a safety-"
                   f"bounded threshold routes only {usage:.0%} of cells "
                   f"to the cheap detector, so there is little cost-"
                   f"saving headroom here. Whether DCI-routing pays off "
                   f"depends on the production workload distribution "
                   f"(which a uniform regime sweep cannot establish) "
                   f"and/or on a richer minimal detector than fixed-1-D. "
                   f"Resolve before RQ1/RQ3.")
    else:
        verdict = (f"DCI is a WEAK or UNRELIABLE router on this evidence "
                   f"(router ROC AUC {router_auc:.2f}, DCI CV "
                   f"{rel_cv_median:.2f}); the selector premise needs "
                   f"rethinking before proceeding.")

    run_json = {
        "run": {
            "analysis_timestamp_utc": run_ts, "run_id": run_id,
            "script": Path(__file__).name, "input_path": str(in_path),
            "grid": {"c": c_vals, "r": r_vals, "batches": B,
                     "n_trials": args.n_trials},
            "eps_sufficiency": args.eps, "safety_precision": args.safety,
            "n_samples": int(len(S)),
        },
        "reliability": {
            "dci_cv_median": rel_cv_median,
            "dci_cv_worst": float(np.max(REL["dci_cv"])),
            "interpretation": "coefficient of variation of DCI across "
                              f"{B} seed-batches per cell; lower = more "
                              "reliable.",
        },
        "validity": {
            "dci_router_roc_auc": float(router_auc),
            "spearman_dci_vs_gap": float(rho_dci_gap),
            "base_rate_1d_sufficient": base_rate,
            "routing_threshold_safe": (theta_safe[0] if theta_safe
                                       else None),
            "routing_precision_target": args.safety,
            "one_d_usage_at_safe_threshold": usage,
            "best_accuracy_threshold": float(best_acc_theta),
            "mean_cost_at_safe_threshold": mean_cost,
            "cost_reduction_vs_full_kernel": cost_vs_full,
        },
        "construct_check": {
            "dci_mean_at_c1_injected_rank1": dci_at_c1,
            "dci_mean_at_c_le_0p4": dci_at_clow,
            "ordering_ok": bool(dci_at_c1 < dci_at_clow),
        },
        "rate_of_change_detector": {
            "ever_uniquely_sufficient": rate_ever_best,
            "note": "if false, the rate-of-change detector never wins a "
                    "cell the 1-D detector loses -- evidence on the "
                    "3-detector framing (A/B question).",
        },
        "verdict": verdict,
    }
    (outdir / "dci_validation_run.json").write_text(json.dumps(run_json, indent=2))

    # ---- figure ----------------------------------------------------------
    fig_path = outdir / "dci_validation_fig.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        suff = y == 1
        ax[0].scatter(dci[suff], S["gap"].to_numpy()[suff], s=22,
                      c="#1d9e75", label="1-D sufficient", alpha=0.8)
        ax[0].scatter(dci[~suff], S["gap"].to_numpy()[~suff], s=22,
                      c="#d62728", label="needs 5-D", alpha=0.8)
        if theta_safe is not None:
            ax[0].axvline(theta_safe[0], color="#0a3d62", ls="--",
                          label=f"route θ={theta_safe[0]:.2f}")
        ax[0].axhline(args.eps, color="#888", ls=":", lw=1)
        ax[0].set_xlabel("DCI"); ax[0].set_ylabel("auc_5d - auc_1d (gap)")
        ax[0].set_title(f"A  DCI vs sufficiency gap "
                        f"(router ROC AUC={router_auc:.3f})", fontsize=10)
        ax[0].legend(fontsize=8)
        ax[1].hist([dci[suff], dci[~suff]], bins=18, stacked=True,
                   color=["#1d9e75", "#d62728"],
                   label=["1-D sufficient", "needs 5-D"])
        ax[1].set_xlabel("DCI"); ax[1].set_ylabel("# cells")
        ax[1].set_title("B  DCI distribution by class", fontsize=10)
        ax[1].legend(fontsize=8)
        for (c, r) in sens_cells:
            sub = SENS[(SENS.c_coverage == c) & (SENS.r_speed == r)]
            ax[2].plot(sub["n_trials"], sub["dci_std"], marker="o",
                       label=f"c={c}, r={r}")
        ax[2].set_xlabel("n_trials"); ax[2].set_ylabel("DCI std across batches")
        ax[2].set_title("C  DCI reliability vs n_trials", fontsize=10)
        ax[2].legend(fontsize=8)
        fig.suptitle(f"Paper 3C  -  DCI validation (RQ2)   run {run_id}",
                     fontsize=11, weight="bold")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
    except Exception as exc:                                  # noqa: BLE001
        fig_path = None
        print(f"[warn] figure skipped ({exc})")

    # ---- console summary -------------------------------------------------
    print()
    print("=" * 70)
    print(f"  DCI VALIDATION (RQ2)   run {run_id}")
    print("=" * 70)
    print(f"  samples: {len(S)}   ({len(c_vals)}x{len(r_vals)} cells x {B} batches)")
    print(f"  base rate (cells that are 1-D-sufficient) : {base_rate:.3f}")
    print()
    print("  RELIABILITY (naen):")
    print(f"    DCI coeff. of variation, median over cells : {rel_cv_median:.3f}")
    print(f"    DCI coeff. of variation, worst cell        : "
          f"{np.max(REL['dci_cv']):.3f}")
    print()
    print("  VALIDITY (mae):")
    print(f"    DCI-as-router ROC AUC                      : {router_auc:.3f}")
    print(f"    Spearman(DCI, gap)                         : {rho_dci_gap:.3f}")
    if theta_safe is not None:
        print(f"    routing threshold @ >={args.safety:.0%} precision     "
              f"     : DCI < {theta_safe[0]:.3f}")
        print(f"    -> 1-D usage {usage:.0%}  ->  mean cost {mean_cost:.1f} "
              f"vs 50  ({cost_vs_full:.1f}x cheaper)")
    else:
        print(f"    no threshold reaches {args.safety:.0%} precision")
    print()
    print("  CONSTRUCT CHECK:")
    print(f"    DCI at c=1 (injected rank-1) : {dci_at_c1:.3f}   "
          f"DCI at c<=0.4 : {dci_at_clow:.3f}")
    print()
    print(f"  rate-of-change detector ever uniquely sufficient: {rate_ever_best}")
    print()
    print(f"  VERDICT: {verdict}")
    print()
    for f in ("dci_validation_samples.csv", "dci_reliability.csv",
              "dci_ntrials_sensitivity.csv", "dci_validation_run.json"):
        print(f"[out ] {outdir / f}")
    if fig_path:
        print(f"[out ] {fig_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
