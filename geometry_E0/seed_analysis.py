#!/usr/bin/env python3
"""
seed_analysis.py
================

Paper 3C -- how many seeds must cost_benefit RUN?  Computed from an
existing cost_benefit_raw.csv; no official re-run needed.

THE RULE THIS ANALYSES
    Paper 3C's sufficiency verdict is the n-STABLE practical floor:
    axis d is 1-D-sufficient iff  mean_gap_d <= delta_d, with
        delta_d = epsilon_d(N_REF) = t_{.975,N_REF-1} * sqrt(sigma2_d/N_REF)
    evaluated at a FIXED reference seed count N_REF (= 15, matching
    cost_benefit.py).  Because delta_d is fixed at N_REF it does not
    shrink with the run's seed count -- the verdict cannot flip as
    seeds are added.  (An un-floored run-n epsilon WOULD shrink as
    1/sqrt(n) and flip borderline verdicts; see the `floor
    justification` block in the output.  The floor removes that.)

    So the only seed-count dependence left is ESTIMATION noise: with
    few seeds, sigma2_d and mean_gap_d are noisy estimates, so the
    estimated regime map can differ from the full-data map.

WHAT IT COMPUTES
    n_min -- the smallest run size at which the regime map is reliably
    estimated.  Subsampling: draw size-n subsamples of the seeds,
    recompute every (cell, axis) verdict under the delta rule, and
    measure the fraction that match the full-data verdict.  n_min is
    the smallest n on the grid with match rate >= 99%.  The metric is
    per (cell, axis) -- 9 cells x 5 axes = 45 verdicts -- so it scales
    with the cell count (a single all-cells-must-match metric caps
    near 0.99^(#cells) and is unreachable for 9 cells).

RUN
    python seed_analysis.py [--raw <cost_benefit_raw.csv>] [--n-ref 15]
    # default raw: the newest out/*/cost_benefit_raw.csv beside this file
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from delong import Z_975, combine_across_seed            # noqa: E402

FEATURES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
SUBSAMPLE_GRID = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45]
N_SUBSAMPLE = 600
STABILITY_TARGET = 0.99
N_REF_DEFAULT = 15            # must match cost_benefit.N_REF


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epsilon_at(n: int, sigma2: float) -> float:
    """The across-seed 95% tolerance for a floored variance sigma2 at
    n seeds: t_{.975,n-1} * sqrt(sigma2/n).  (Same formula as
    delong.epsilon_from_sigma2; kept local so this script is
    self-contained for the subsampling inner loop.)"""
    if n < 2 or not np.isfinite(sigma2) or sigma2 <= 0:
        return 0.0
    return float(t_dist.ppf(0.975, n - 1) * np.sqrt(sigma2 / n))


def cell_verdicts(g: pd.DataFrame, n_ref: int) -> dict:
    """The delta-rule sufficiency verdict for every axis of one cell.

    Returns {axis: (mean_gap, sigma2, delta_ref, sufficient_bool)}.
    delta_ref = epsilon(n_ref, sigma2) -- the n-stable practical floor.
    """
    out = {}
    for d in FEATURES:
        cc = combine_across_seed(list(g[f"gap_{d}"]), list(g[f"vargap_{d}"]))
        gap = cc["delta_combined"]
        sig2 = cc["sigma2"]
        delta = epsilon_at(n_ref, sig2)
        suff = bool(np.isfinite(gap) and gap <= delta)
        out[d] = (gap, sig2, delta, suff)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=str, default=None)
    ap.add_argument("--n-ref", type=int, default=N_REF_DEFAULT,
                    help="reference seed count for the practical floor "
                         "delta_d = epsilon_d(N_REF); must match "
                         "cost_benefit.py")
    args = ap.parse_args()
    n_ref = args.n_ref

    if args.raw:
        raw_path = Path(args.raw).expanduser().resolve()
    else:
        cands = sorted((_HERE / "out").glob("*/cost_benefit_raw.csv"))
        if not cands:
            print("[ERR ] no out/*/cost_benefit_raw.csv found; pass --raw",
                  file=sys.stderr)
            return 2
        raw_path = cands[-1]
    R = pd.read_csv(raw_path)
    run_ts = utc_now_iso()
    n_full = int(R.groupby(["workload", "config"]).size().min())
    cells = list(R.groupby(["workload", "config"]))
    print(f"[seed_analysis] {run_ts}")
    print(f"[in  ] {raw_path}")
    print(f"[cfg ] {len(cells)} cells x {n_full} seeds   N_REF = {n_ref}\n")

    # ---- full-data regime map (the delta rule) --------------------------
    ref = {}                                   # (wl,cfg) -> {axis: verdict}
    A = []
    for (wl, cfg), g in cells:
        v = cell_verdicts(g, n_ref)
        ref[(wl, cfg)] = {d: v[d][3] for d in FEATURES}
        for d in FEATURES:
            gap, sig2, delta, suff = v[d]
            # without the floor the run-n epsilon would cross the gap at
            # n_flip = z^2 sigma2 / gap^2 -- the justification for the floor
            n_flip = ((Z_975 ** 2) * sig2 / (gap ** 2)
                      if gap > 1e-9 and sig2 > 0 else float("inf"))
            A.append({"workload": wl, "config": cfg, "dim": d,
                      "mean_gap": gap, "sigma2": sig2, "delta_ref": delta,
                      "sufficient": suff, "n_flip_unfloored": n_flip})
    A = pd.DataFrame(A)
    n_verdicts = len(A)

    # ---- subsampling: how many seeds to estimate that map? --------------
    rng = np.random.default_rng(20260522)
    stab = {}
    for n in SUBSAMPLE_GRID:
        if n >= n_full:
            break
        match = 0
        for _ in range(N_SUBSAMPLE):
            for (wl, cfg), g in cells:
                sub = g.sample(n=n, replace=False,
                               random_state=int(rng.integers(0, 2 ** 31)))
                v = cell_verdicts(sub, n_ref)
                for d in FEATURES:
                    match += (v[d][3] == ref[(wl, cfg)][d])
        stab[n] = match / (N_SUBSAMPLE * n_verdicts)
    n_min = next((n for n in SUBSAMPLE_GRID
                  if stab.get(n, 0.0) >= STABILITY_TARGET), None)

    # ---- console --------------------------------------------------------
    print("=" * 72)
    print("  REGIME MAP  (delta rule: axis sufficient iff "
          f"mean_gap <= epsilon(N_REF={n_ref}))")
    print("=" * 72)
    for (wl, cfg) in sorted(ref):
        suf = sorted(d for d in FEATURES if ref[(wl, cfg)][d])
        print(f"  {wl + '/' + cfg:24} sufficient axes: "
              f"{suf if suf else '(none)'}")
    print()

    print("  SEEDS TO RUN  -- subsampling: fraction of the "
          f"{n_verdicts} (cell,axis)")
    print("  verdicts that match the full-data regime map:")
    for n in SUBSAMPLE_GRID:
        if n in stab:
            bar = "#" * int(round(stab[n] * 40))
            mark = "  <- n_min" if n == n_min else ""
            print(f"    n={n:3d}  {stab[n]:6.1%}  {bar}{mark}")
    if n_min is not None:
        print(f"\n  n_min = {n_min}  (>= {STABILITY_TARGET:.0%} of verdicts "
              f"match from this run size up)")
        print(f"  current run uses n = {n_full}  -- "
              f"{'OK' if n_full >= n_min else 'BELOW n_min'}")
    else:
        best = max(stab, key=stab.get) if stab else None
        print(f"\n  n_min = not reached on the grid; best is "
              f"{stab.get(best, 0):.1%} at n={best}. The regime map is "
              f"recovered well but not at the {STABILITY_TARGET:.0%} bar.")
    print()

    # ---- floor justification: what the un-floored rule would do ---------
    bl = A[(A.sufficient) & np.isfinite(A.n_flip_unfloored)]
    if len(bl):
        print("  FLOOR JUSTIFICATION  -- without the practical floor, the "
              "run-n epsilon")
        print("  shrinks as 1/sqrt(n) and these sufficient axes would flip "
              "to NOT:")
        for _, r in bl.sort_values("n_flip_unfloored").iterrows():
            print(f"    {r.workload}/{r.config} {r.dim}  gap "
                  f"{r.mean_gap:.4f}  would flip at n ~ "
                  f"{r.n_flip_unfloored:.0f}")
        print("  the delta_d = epsilon_d(N_REF) floor fixes N_REF, so none "
              "of these flip.\n")

    # ---- outputs --------------------------------------------------------
    out_dir = raw_path.parent
    out = {"analysis_timestamp_utc": run_ts, "raw_csv": str(raw_path),
           "n_full": n_full, "n_ref": n_ref, "n_min": n_min,
           "n_verdicts": n_verdicts, "stability_curve": stab,
           "regime_map": {f"{wl}/{cfg}":
                          sorted(d for d in FEATURES if ref[(wl, cfg)][d])
                          for (wl, cfg) in ref},
           "per_axis": A.to_dict(orient="records")}
    (out_dir / "seed_analysis.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"[out ] {out_dir / 'seed_analysis.json'}")

    fig_path = out_dir / "seed_analysis_fig.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        gs = sorted(stab)
        ax[0].plot(gs, [stab[n] for n in gs], "o-", color="#1d9e75")
        ax[0].axhline(STABILITY_TARGET, ls="--", color="#888")
        if n_min:
            ax[0].axvline(n_min, ls="--", color="#1d9e75",
                          label=f"n_min={n_min}")
        ax[0].axvline(n_full, color="#333", alpha=0.4, lw=2,
                      label=f"run n={n_full}")
        ax[0].set_ylim(0, 1.03)
        ax[0].set_xlabel("seeds run (subsample size)")
        ax[0].set_ylabel("fraction of verdicts matching full map")
        ax[0].set_title("A  seeds to estimate the regime map", fontsize=10)
        ax[0].legend(fontsize=8)
        col = ["#1d9e75" if s else "#d62728" for s in A["sufficient"]]
        ax[1].scatter(A["mean_gap"], A["delta_ref"], c=col, s=36,
                      edgecolor="black", linewidth=0.4)
        lim = float(max(A["mean_gap"].max(), A["delta_ref"].max()) * 1.1)
        ax[1].plot([0, lim], [0, lim], ls=":", color="#888")
        ax[1].set_xlabel("mean AUC gap (5-D - axis)")
        ax[1].set_ylabel(f"delta_ref = epsilon(N_REF={n_ref})")
        ax[1].set_title("B  gap vs floor  (green = sufficient)", fontsize=10)
        fig.suptitle("Paper 3C  -  seeds to estimate the regime map",
                     fontsize=11, weight="bold")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        print(f"[out ] {fig_path}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[warn] figure skipped ({exc})")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
