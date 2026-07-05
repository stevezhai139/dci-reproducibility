#!/usr/bin/env python3
"""
compare_to_3c.py
================

Paper 3D -- Phase 1 T2 verification.

Compares an official `cost_benefit.py` run against the locked Paper 3C
reference run, to confirm the Paper 3D code tree reproduces Paper 3C's
signal-layer numbers (DCI, regime map, routing).

Two classes of quantity are treated differently:

  * SEED-DETERMINISTIC -- DCI, the AUCs, delta-AUC, the routed-to-1D
    fraction, the 5-D-gain-retained ratio. Paper 3C's methodology
    guarantees these are byte-identical for a given (workload, config,
    seed) on any machine. They MUST match to floating-point machine
    precision; the verdict fails otherwise.

  * MACHINE-DEPENDENT -- wall-clock overheads, the 5-D/1-D cost ratio,
    cost_paid_vs_5D, per-policy mean_cost_ms. These legitimately differ
    by machine. They are reported for information only and NEVER cause
    the verdict to fail.

Usage
-----
    python3 compare_to_3c.py --run out/<your_run_id>/ \\
                             --ref out/20260522T145920Z/

Writes `comparison_<utc>.json` to the current directory and prints a
human-readable verdict. Exit code 0 = reproduction confirmed, 1 = not.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# A seed-deterministic quantity must agree at least this tightly.
SEED_TOL = 1e-9

RAW_NUMERIC = ["dci", "auc_1d", "auc_5d", "auc_5d_l2",
               "auc_5d_composite", "delta_auc"]
SUMMARY_NUMERIC = ["dci_mean", "auc_1d_mean", "auc_5d_mean",
                   "auc_5d_composite_mean", "delta_auc_mean"]


def _utc_now() -> str:
    """ISO-8601 UTC timestamp, matching the project's record discipline."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    """Timestamped log line, flushed immediately (project logging convention)."""
    print(f"[{_utc_now()}] {msg}", flush=True)


def _load_csv(path: Path, key_cols: list[str]) -> dict:
    rows: dict = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[tuple(r[c] for c in key_cols)] = r
    return rows


def compare_raw(run_dir: Path, ref_dir: Path) -> dict:
    """Per-(workload, config, seed) check of the seed-deterministic columns."""
    run = _load_csv(run_dir / "cost_benefit_raw.csv", ["workload", "config", "seed"])
    ref = _load_csv(ref_dir / "cost_benefit_raw.csv", ["workload", "config", "seed"])
    shared = sorted(set(run) & set(ref))
    max_diff = 0.0
    worst = None
    numeric_mismatches = 0
    dim_mismatches = 0
    for k in shared:
        for c in RAW_NUMERIC:
            d = abs(float(run[k][c]) - float(ref[k][c]))
            if d > max_diff:
                max_diff, worst = d, (k, c)
            if d > SEED_TOL:
                numeric_mismatches += 1
        if run[k].get("best_1d_dim") != ref[k].get("best_1d_dim"):
            dim_mismatches += 1
    return {
        "cells_in_run": len(run),
        "cells_in_ref": len(ref),
        "cells_compared": len(shared),
        "max_abs_diff": max_diff,
        "worst_cell": worst,
        "numeric_mismatches_over_tol": numeric_mismatches,
        "best_1d_dim_mismatches": dim_mismatches,
        "ok": numeric_mismatches == 0 and dim_mismatches == 0
              and len(shared) == len(ref),
    }


def compare_summary(run_dir: Path, ref_dir: Path) -> dict:
    """Per-(workload, config) check of the aggregated mean columns."""
    run = _load_csv(run_dir / "cost_benefit_summary.csv", ["workload", "config"])
    ref = _load_csv(ref_dir / "cost_benefit_summary.csv", ["workload", "config"])
    shared = sorted(set(run) & set(ref))
    max_diff = 0.0
    worst = None
    mismatches = 0
    for k in shared:
        for c in SUMMARY_NUMERIC:
            d = abs(float(run[k][c]) - float(ref[k][c]))
            if d > max_diff:
                max_diff, worst = d, (k, c)
            if d > SEED_TOL:
                mismatches += 1
    return {
        "cells_compared": len(shared),
        "max_abs_diff": max_diff,
        "worst_cell": worst,
        "mismatches_over_tol": mismatches,
        "ok": mismatches == 0 and len(shared) == len(ref),
    }


def compare_runjson(run_dir: Path, ref_dir: Path) -> dict:
    """Headline check: the seed-deterministic parts of the run JSON."""
    run = json.loads((run_dir / "cost_benefit_run.json").read_text())
    ref = json.loads((ref_dir / "cost_benefit_run.json").read_text())

    det = {}  # seed-deterministic headline numbers
    for name, getter in [
        ("selector_accuracy_of_5D_gain_retained",
         lambda d: d["selector_value"]["accuracy_of_5D_gain_retained"]),
        ("selector_frac_routed_to_1D",
         lambda d: d["policies"]["DCI_selector"]["frac_routed_to_1D"]),
        ("always_1D_mean_auc", lambda d: d["policies"]["always_1D"]["mean_auc"]),
        ("always_5D_mean_auc", lambda d: d["policies"]["always_5D"]["mean_auc"]),
        ("DCI_selector_mean_auc",
         lambda d: d["policies"]["DCI_selector"]["mean_auc"]),
    ]:
        rv, fv = getter(run), getter(ref)
        det[name] = {"run": rv, "ref": fv, "abs_diff": abs(rv - fv),
                     "ok": abs(rv - fv) <= SEED_TOL}

    mach = {}  # machine-dependent -- reported only, never fails the verdict
    for name, getter in [
        ("cost_ratio_5d_over_1d",
         lambda d: d["overhead"]["cost_ratio_5d_over_1d"]),
        ("selector_cost_paid_vs_5D",
         lambda d: d["selector_value"]["cost_paid_vs_5D"]),
    ]:
        try:
            mach[name] = {"run": getter(run), "ref": getter(ref)}
        except KeyError:
            pass

    return {
        "seed_deterministic": det,
        "machine_dependent_info_only": mach,
        "ok": all(v["ok"] for v in det.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="your official run's timestamped output folder")
    ap.add_argument("--ref", type=Path,
                    default=Path("out/20260522T145920Z"),
                    help="locked Paper 3C reference run (default: the vendored one)")
    args = ap.parse_args()

    for d in (args.run, args.ref):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 2

    t0 = time.perf_counter()
    _log("compare_to_3c.py - start")
    _log(f"  run = {args.run}")
    _log(f"  ref = {args.ref}")

    raw = compare_raw(args.run, args.ref)
    summ = compare_summary(args.run, args.ref)
    rj = compare_runjson(args.run, args.ref)
    reproduced = raw["ok"] and summ["ok"] and rj["ok"]

    verdict = {
        "analysis_timestamp_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(args.run),
        "ref_dir": str(args.ref),
        "seed_deterministic_tolerance": SEED_TOL,
        "raw_per_seed": raw,
        "summary_means": summ,
        "run_json_headline": rj,
        "verdict": "REPRODUCED" if reproduced else "DIVERGENCE",
    }
    ts = verdict["analysis_timestamp_utc"].replace("-", "").replace(":", "")
    out = Path("comparison_" + ts + ".json")
    out.write_text(json.dumps(verdict, indent=2))

    print("=" * 64)
    print("Paper 3D -- T2 reproduction check vs locked Paper 3C run")
    print("=" * 64)
    print(f"  raw per-seed : {raw['cells_compared']} cells compared, "
          f"max abs diff {raw['max_abs_diff']:.3e}, "
          f"mismatches {raw['numeric_mismatches_over_tol']}")
    print(f"  summary means: {summ['cells_compared']} cells, "
          f"max abs diff {summ['max_abs_diff']:.3e}, "
          f"mismatches {summ['mismatches_over_tol']}")
    print(f"  headline     : "
          f"{'all match' if rj['ok'] else 'MISMATCH'} "
          f"(seed-deterministic numbers)")
    for n, v in rj["seed_deterministic"].items():
        print(f"      {n:42s} run={v['run']:.6f} ref={v['ref']:.6f} "
              f"d={v['abs_diff']:.2e} {'ok' if v['ok'] else 'MISMATCH'}")
    if rj["machine_dependent_info_only"]:
        print("  machine-dependent (cost; info only, never fails verdict):")
        for n, v in rj["machine_dependent_info_only"].items():
            print(f"      {n:42s} run={v['run']:.4f} ref={v['ref']:.4f}")
    print("-" * 64)
    print(f"  VERDICT: {verdict['verdict']}")
    print(f"  written: {out}")
    print("=" * 64)
    _log(f"compare_to_3c.py - done | verdict={verdict['verdict']} | "
         f"elapsed {time.perf_counter() - t0:.2f}s")
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
