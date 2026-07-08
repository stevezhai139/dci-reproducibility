#!/usr/bin/env python3
"""sdss_detector_replay.py -- label-free detector-resolution replay on the real SkyServer trace.

Runs three DCIGate configurations (tau = 0 always-multi-D / 1.5 DCI-gated / 1e9 always-1-D)
over the SAME logged per-window feature stream of the real SkyServer workload
(feature_agnostic/sdss_adapter.py output). The trace has no ground-truth onsets, so no
recall is claimed; the metrics are label-free:

  - monitoring cost per window and %5-D windows, per policy
  - decision fidelity: agreement of firing decisions with always-multi-D,
    reported OVERALL and WITHIN the windows the gated policy ran 1-D
    (the only windows where fidelity is actually at stake on a 90%-high-DCI trace)

Calibration: the first --cal windows are used as the steady sample (disclosed; a natural
trace has no clean steady segment). Gate trajectory resets every --block windows to match
the paper's block structure.

Usage:
    python3 sdss_detector_replay.py <path-to-repro_3c> [--run RUN_ID] [--cal 64] [--block 50]

Axis note (3C vs 3A/3D): sweeps the DETECTOR only; no advisor, no policy sweep, no
wall_qps/economics; the kernel is a frozen black box.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

AXES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
TAUS = {"always_5D": 0.0, "dci_gated": 1.5, "always_1D": 1e9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=str)
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--cal", type=int, default=64)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--timing-run", type=str, default="20260705T135856Z")
    args = ap.parse_args()

    repro = Path(args.repro).resolve()
    spec = importlib.util.spec_from_file_location(
        "dci_gate", repro / "end_to_end" / "dci_gate.py")
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)

    outroot = repro / "feature_agnostic" / "out"
    run = args.run or sorted(p.name for p in outroot.iterdir() if p.is_dir())[-1]
    rows = list(csv.DictReader(open(outroot / run / "sdss_windows.csv")))
    cols = [c for c in rows[0] if c in AXES]
    if len(cols) != 5:                      # fall back: last 6 numeric cols = 5 axes + HSM
        keys = list(rows[0].keys())
        cols = keys[-6:-1]
        print(f"[warn] axis columns inferred as {cols}")
    F = np.array([[float(r[c]) for c in cols] for r in rows])
    n = len(F)

    ov = json.load(open(repro / "geometry_E0" / "out" / args.timing_run /
                        "cost_benefit_run.json"))["overhead"]
    c1, c5 = ov["overhead_1d_s"] * 1e3, ov["overhead_5d_s"] * 1e3
    print(f"[in  ] run={run}  windows={n}  cal=first {args.cal}  block={args.block}")

    cal = F[: args.cal]
    eval_idx = range(args.cal, n)

    results = {}
    for policy, tau in TAUS.items():
        gate = dg.DCIGate(tau=tau, alpha=0.05).fit(cal)
        fired = np.zeros(n, dtype=int)
        regime5 = np.zeros(n, dtype=bool)
        for t in eval_idx:
            if (t - args.cal) % args.block == 0:
                gate.reset_trajectory()
            fired[t] = gate.decide(F[t])
            regime5[t] = gate.last["regime"] == "5-D"
        ne = len(list(eval_idx))
        cost = (regime5[args.cal:].sum() * c5 +
                (ne - regime5[args.cal:].sum()) * c1) / ne
        results[policy] = {"fired": fired[args.cal:], "r5": regime5[args.cal:],
                           "cost": cost, "frac5": regime5[args.cal:].mean(),
                           "n_fire": int(fired[args.cal:].sum())}

    ref = results["always_5D"]["fired"]
    print(f"\n{'policy':12s}{'%5D':>8s}{'ms/win':>9s}{'%5Dcost':>9s}"
          f"{'firings':>9s}{'agree':>8s}{'agree@1D':>10s}")
    for pol, r in results.items():
        agree = float((r["fired"] == ref).mean())
        m1 = ~r["r5"]                            # windows this policy ran 1-D
        agree1 = float((r["fired"][m1] == ref[m1]).mean()) if m1.any() else float("nan")
        print(f"{pol:12s}{r['frac5']:8.1%}{r['cost']:9.3f}{r['cost'] / c5:9.1%}"
              f"{r['n_fire']:9d}{agree:8.3f}{agree1:10.3f}")
    print("\n[note] agree@1D = firing agreement with always-5D restricted to the windows "
          "the row's policy ran the 1-D detector (n=%d for dci_gated)."
          % int((~results['dci_gated']['r5']).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
