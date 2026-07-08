#!/usr/bin/env python3
"""d_scaling_bench.py -- the O(D^2)-vs-O(D^3) claim, measured (Paper 3C, Sec 4).

At the paper's D=5 the closed-form DCI (one trace + one Frobenius norm) and a full
eigendecomposition both cost microseconds; the closed form's point is how the costs
SCALE once the feature vector grows (raw template-frequency vectors, learned
query/plan embeddings: D in the hundreds). This microbenchmark times, per D:

    dci_closed : tr(C)^2 / ||C||_F^2          -- O(D^2), what the paper deploys
    dci_eig    : 1 / sum(p_i^2) via eigvalsh  -- O(D^3), what it avoids

on random Wishart-type PSD covariances (C = A A^T / m, A ~ N(0,1)^{D x 2D}),
median over repeats, identical matrices for both paths. Pure NumPy; no database.

Usage:  python3 d_scaling_bench.py [--dims 5,20,50,200,500,1000] [--repeats 200]
Writes: d_scaling_bench.csv (+ prints the paper-ready line)
"""
import argparse, csv, time
from pathlib import Path

import numpy as np


def dci_closed(C):
    tr = np.trace(C)
    f2 = float(np.sum(C * C))          # ||C||_F^2 without forming C^T C
    return (tr * tr) / f2


def dci_eig(C):
    ev = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    p = ev / ev.sum()
    return 1.0 / float(np.sum(p * p))


def bench(fn, mats, repeats):
    # warm-up
    for M in mats[:3]:
        fn(M)
    t0 = time.perf_counter()
    for _ in range(repeats):
        for M in mats:
            fn(M)
    return (time.perf_counter() - t0) / (repeats * len(mats))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="5,20,50,200,500,1000")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--mats", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(20260708)
    rows = []
    print(f"{'D':>6s}{'closed (µs)':>14s}{'eig (µs)':>12s}{'ratio':>9s}   agree")
    for D in [int(x) for x in args.dims.split(",")]:
        mats = []
        for _ in range(args.mats):
            A = rng.standard_normal((D, 2 * D))
            mats.append((A @ A.T) / (2 * D))
        reps = max(3, args.repeats // max(1, D // 50))   # keep large-D runs short
        t_c = bench(dci_closed, mats, reps) * 1e6
        t_e = bench(dci_eig, mats, reps) * 1e6
        agree = max(abs(dci_closed(M) - dci_eig(M)) for M in mats)
        print(f"{D:>6d}{t_c:>14.2f}{t_e:>12.1f}{t_e / t_c:>9.1f}x   |diff|<={agree:.2e}")
        rows.append({"D": D, "closed_us": round(t_c, 3), "eig_us": round(t_e, 3),
                     "ratio": round(t_e / t_c, 2), "max_abs_diff": agree,
                     "repeats": reps, "mats": args.mats})
    out = Path(__file__).resolve().parent / "d_scaling_bench.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[out] {out.name}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
