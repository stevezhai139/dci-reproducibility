#!/usr/bin/env python3
"""
verify_kernel_equivalence.py
============================

Paper 3D -- Phase 2 T2.1.

Numerically checks whether the **canonical kernel** (`code/kernel/
hsm_v2_kernel.py`) computes the same 5-D HSM scores as the **V3 kernel**
(`cross_engine/_v3_hsm/hsm_v2_core.py`) that the MongoDB cross-engine
path currently imports.

Both kernels' `hsm_v2(...)` take the same 12 positional arguments. The
known packaging difference: V3 `return`s values `round(., 4)`; the
canonical kernel returns full-precision `float()`. This script feeds an
identical battery of window-feature pairs through both and reports, per
axis, the maximum |canonical - V3| difference.

Expected: every difference is within V3's 4-decimal rounding
granularity (<= 5e-5). That would confirm the canonical kernel is V3's
maths at full precision -- so harmonising the MongoDB path onto it is a
safe re-point, and the only numerical effect is the (correct) removal
of V3's cosmetic rounding.

Usage:
    python3 verify_kernel_equivalence.py --v3 <path> --canon <path>
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from datetime import datetime, timezone

import numpy as np

AXES = ["S_R", "S_V", "S_T", "S_A", "S_P", "HSM"]
V3_ROUNDING = 5e-5   # half of V3's 1e-4 rounding step


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def _load(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_window(rng: np.random.Generator, n_qids: int = 24):
    """A random WindowFeatures-shaped tuple of kernel inputs."""
    k = int(rng.integers(1, 8))                       # distinct templates
    ids = rng.integers(0, n_qids, size=int(rng.integers(12, 32)))
    freq = np.zeros(n_qids, dtype=float)
    for i in ids:
        freq[i] += 1.0
    if freq.sum() > 0:
        freq /= freq.sum()
    n = int(len(ids))
    tables = {f"t{i}" for i in set(ids[:k])}
    cols = {f"c{i}" for i in set(ids)}
    m = int(rng.integers(20, 60))
    base = rng.uniform(50, 150)
    times = (base + rng.normal(0, 8, m)
             + 30 * np.sin(np.linspace(0, rng.uniform(2, 9), m))).astype(float)
    qset = {f"Q{i}" for i in set(ids)}
    return freq, n, tables, cols, times, qset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3", required=True, help="path to V3 hsm_v2_core.py")
    ap.add_argument("--canon", required=True, help="path to canonical hsm_v2_kernel.py")
    ap.add_argument("--pairs", type=int, default=300, help="number of window pairs")
    ap.add_argument("--seed", type=int, default=20260524)
    args = ap.parse_args()

    t0 = time.perf_counter()
    _log("verify_kernel_equivalence.py - start")
    _log(f"  v3    = {args.v3}")
    _log(f"  canon = {args.canon}")

    v3 = _load(args.v3, "v3_kernel")
    canon = _load(args.canon, "canon_kernel")
    _log("  both kernels imported OK")

    rng = np.random.default_rng(args.seed)
    max_diff = {ax: 0.0 for ax in AXES}
    over = {ax: 0 for ax in AXES}
    sp_diverge = []   # (pair, len_a, len_b, sp_v3, sp_canon, diff)

    for p in range(args.pairs):
        wa = _make_window(rng)
        wb = _make_window(rng)
        args12 = (wa[0], wb[0], wa[1], wb[1],
                  wa[2], wb[2], wa[3], wb[3],
                  wa[4], wb[4], wa[5], wb[5])
        rv3 = v3.hsm_v2(*args12)
        rcn = canon.hsm_v2(*args12)
        for ax in AXES:
            d = abs(float(rcn[ax]) - float(rv3[ax]))
            if d > max_diff[ax]:
                max_diff[ax] = d
            if d > V3_ROUNDING:
                over[ax] += 1
        d_sp = abs(float(rcn["S_P"]) - float(rv3["S_P"]))
        if d_sp > V3_ROUNDING:
            sp_diverge.append((p, len(wa[4]), len(wb[4]),
                               float(rv3["S_P"]), float(rcn["S_P"]), d_sp))
        if (p + 1) % 100 == 0:
            _log(f"  {p + 1}/{args.pairs} pairs compared "
                 f"(elapsed {time.perf_counter() - t0:.1f}s)")

    print("=" * 60)
    print("canonical hsm_v2  vs  V3 hsm_v2   (per-axis max |diff|)")
    print("=" * 60)
    all_within = True
    for ax in AXES:
        flag = "ok (within V3 rounding)" if over[ax] == 0 else f"OVER in {over[ax]} pairs"
        if over[ax] != 0:
            all_within = False
        print(f"  {ax:5s}  max|diff| = {max_diff[ax]:.3e}   {flag}")
    if sp_diverge:
        print("-" * 60)
        print(f"  S_P divergence detail ({len(sp_diverge)} pairs):")
        print(f"    {'pair':>5} {'len_a':>6} {'len_b':>6} "
              f"{'S_P(v3)':>10} {'S_P(canon)':>11} {'diff':>10}")
        for row in sp_diverge:
            print(f"    {row[0]:5d} {row[1]:6d} {row[2]:6d} "
                  f"{row[3]:10.6f} {row[4]:11.6f} {row[5]:10.3e}")
    print("-" * 60)
    verdict = ("EQUIVALENT (canonical = V3 maths, full precision)"
               if all_within else
               "DIVERGENCE beyond rounding - investigate the flagged axis")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    _log(f"verify_kernel_equivalence.py - done | {verdict} | "
         f"elapsed {time.perf_counter() - t0:.2f}s")
    return 0 if all_within else 1


if __name__ == "__main__":
    raise SystemExit(main())
