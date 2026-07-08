#!/usr/bin/env python3
"""sdss_routing_cost.py -- provenance for the SkyServer routing sentence (Paper 3C, Sec 6.2).

Reads the per-block DCI of the real SkyServer trace (sdss_adapter.py output) and the
locked-env per-window detector timings, and prints the numbers used in the paper:
fraction of blocks below tau, the selector's cost per window on the real trace, and
that cost as a fraction of always-multi-D. No labels are used (the trace has no
ground-truth onsets); this is a cost/routing statement only.

Usage:
    python3 sdss_routing_cost.py <path-to-repro_3c> [--run RUN_ID] [--tau 1.5]
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=str)
    ap.add_argument("--run", type=str, default=None,
                    help="feature_agnostic/out run id (default: latest)")
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--timing-run", type=str, default="20260705T135856Z")
    args = ap.parse_args()

    repro = Path(args.repro).resolve()
    outroot = repro / "feature_agnostic" / "out"
    run = args.run or sorted(p.name for p in outroot.iterdir() if p.is_dir())[-1]
    dci_csv = outroot / run / "sdss_dci.csv"

    ov = json.load(open(repro / "geometry_E0" / "out" / args.timing_run /
                        "cost_benefit_run.json"))["overhead"]
    c1, c5 = ov["overhead_1d_s"] * 1e3, ov["overhead_5d_s"] * 1e3

    d = [float(r["dci"]) for r in csv.DictReader(open(dci_csv))]
    n = len(d)
    low = sum(1 for x in d if x < args.tau)
    sel = (low * c1 + (n - low) * c5) / n
    print(f"run={run}  blocks={n}  tau={args.tau}")
    print(f"low-DCI blocks : {low}  ({low / n:.1%})   high: {n - low}  ({(n - low) / n:.1%})")
    print(f"mean DCI       : {sum(d) / n:.3f}   range [{min(d):.3f}, {max(d):.3f}]")
    print(f"selector cost  : {sel:.3f} ms/window  =  {sel / c5:.1%} of always-multi-D "
          f"(1-D {c1:.4f} ms, 5-D {c5:.4f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
