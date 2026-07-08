#!/usr/bin/env python3
"""escalation_replay.py -- S2: escalation frequency of the DCI gate (Paper 3C, Sec 6.8).

Replays the LOGGED per-window feature stream of the live end-to-end runs through
the frozen DCIGate (tau=1.5, alpha=0.05, reset per block) and reports the fraction
of deployment windows routed to the multi-dimensional detector.

No database, no kernel, no RNG: pure deterministic arithmetic over the recorded
CSVs, so the result is machine-independent. Steady-state calibration (mu0/Sigma0)
is reconstructed from the logged steady-phase windows; the paper's numbers were
verified insensitive to this choice (identical to 4 decimals across three
disjoint calibration sets).

Usage:
    python3 escalation_replay.py <path-to-repro_3c>

Expected output (source of the Sec 6.8 sentence):
    PG : all=0.1458  beyond-warmup=0.0682
    MG : all=0.2083  beyond-warmup=0.1364
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

AX = ["S_R", "S_V", "S_T", "S_A", "S_P"]


def load_gate_module(repro: Path):
    p = repro / "end_to_end" / "dci_gate.py"
    spec = importlib.util.spec_from_file_location("dci_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_blocks(csv_path: Path):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["strategy"] == "dci_gated"]
    blocks = {}
    for r in rows:
        blocks.setdefault(int(r["block"]), []).append(r)
    for b in blocks:
        blocks[b].sort(key=lambda r: int(r["window"]))
    return blocks


def replay(dg, blocks, steady_rows):
    X = np.array([[float(r[a]) for a in AX] for r in steady_rows])
    gate = dg.DCIGate(tau=1.5, alpha=0.05).fit(X)
    tot = esc = totw = escw = 0
    for _, rows in sorted(blocks.items()):
        gate.reset_trajectory()
        for r in rows:
            gate.decide([float(r[a]) for a in AX])
            is5 = gate.last["regime"] == "5-D"
            tot += 1
            esc += is5
            if int(r["window"]) > 2:          # beyond the 2-window warm-up
                totw += 1
                escw += is5
    return esc / tot, escw / totw


def main() -> int:
    repro = Path(sys.argv[1]).resolve()
    dg = load_gate_module(repro)
    e2e = repro / "end_to_end"

    runs = [
        ("PG", e2e / "postgres" / "out_PG_gated_tau1p5" / "breakdown_per_window_v2.csv",
         "Reporting"),
        ("MG", sorted((e2e / "mongo" / "out" / "MG_gated_tau1p5").glob(
            "*/breakdown_per_window.csv"))[-1],
         "edge"),
    ]
    for label, path, steady_phase in runs:
        blocks = load_blocks(path)
        steady = [r for b in blocks.values() for r in b if r["phase"] == steady_phase]
        a, w = replay(dg, blocks, steady)
        print(f"{label} : all={a:.4f}  beyond-warmup={w:.4f}  "
              f"(steady n={len(steady)}, file={path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
