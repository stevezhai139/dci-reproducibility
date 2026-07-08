#!/usr/bin/env python3
"""dci_resolution_e2e.py -- S5 Part 1: the detector-resolution sweep, offline (Paper 3C).

Per 3C_END_TO_END_ARC_AND_DESIGN.md: hold the advisor policy fixed (gated) and sweep the
DETECTOR by running three DCIGate instances -- tau = 0.0 (always-multi-D), 1.5 (DCI-gated),
1e9 (always-1-D once estimable) -- over the SAME per-window feature stream, including the
heterogeneous (mixed) regime the live runs never exercised. Logs, per config: firing
decisions vs ground-truth onsets (recall / false alarms) and detector monitoring cost.

Axis discipline (3C vs 3D): metrics here are detection fidelity + firings + monitoring
cost ONLY -- no wall_qps, no advisor economics (those belong to the 3D/journal axis).

Feature streams come from the paper's own harness (geometry_E0/cost_benefit.py:
build_trajectory + kernel_adjacent, deterministic stable_seed), so Part 1 runs with no
database. Part 2 (live PG/Mongo, Steve's machine) reuses the same three-tau loop in front
of the real advisor.

Usage:
    python3 dci_resolution_e2e.py <path-to-repro_3c> [--seeds N] [--configs mixed,template_only,volume_only]

Cost constants are taken from the locked-env timing run
repro_3c/geometry_E0/out/20260705T135856Z/cost_benefit_run.json (overridable via CLI).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

AXES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
TAUS = {"always_5D": 0.0, "dci_gated": 1.5, "always_1D": 1e9}
WARMUP = 2                       # min_dci_windows-1: first 2 windows default to 5-D


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def steady_calibration(cb, pool, n_needed: int = 64, base_tag: str = "cal"):
    """Harvest >= n_needed steady feature vectors from phase-0 (pre-transition) windows
    of dedicated trajectories (template_only; only indices strictly before the first
    onset are used, so the drift type is irrelevant)."""
    feats = []
    s = 0
    while len(feats) < n_needed and s < 200:
        seed = cb.stable_seed(base_tag, "template_only", s)
        windows, didx = cb.build_trajectory(pool, "template_only", seed)
        first_onset = min(didx)                     # window index of first transition
        fv, _ = cb.kernel_adjacent(windows)
        # f index t corresponds to window t+1; steady pairs are windows within phase 0
        steady = fv[: max(0, first_onset - 1)]
        feats.extend(list(steady))
        s += 1
    return np.asarray(feats[:n_needed], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=str)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--configs", type=str, default="mixed,template_only,volume_only")
    ap.add_argument("--workloads", type=str, default="tpch,job,pgbench")
    ap.add_argument("--cost-1d-ms", type=float, default=None)
    ap.add_argument("--cost-5d-ms", type=float, default=None)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    repro = Path(args.repro).resolve()
    sys.path.insert(0, str(repro))
    sys.path.insert(0, str(repro / "geometry_E0"))
    cb = load_module(repro / "geometry_E0" / "cost_benefit.py", "cost_benefit")
    dg = load_module(repro / "end_to_end" / "dci_gate.py", "dci_gate")

    # cost constants: locked-env timing run unless overridden
    c1, c5 = args.cost_1d_ms, args.cost_5d_ms
    if c1 is None or c5 is None:
        run_json = repro / "geometry_E0" / "out" / "20260705T135856Z" / "cost_benefit_run.json"
        ov = json.load(open(run_json))["overhead"]
        c1 = c1 or ov["overhead_1d_s"] * 1e3
        c5 = c5 or ov["overhead_5d_s"] * 1e3
    print(f"[cost] 1-D {c1:.4f} ms  5-D {c5:.4f} ms (per window)")

    pools = {}
    for wl in args.workloads.split(","):
        if wl == "tpch":
            pools[wl] = cb.tpch_pool()
        elif wl == "job":
            pools[wl] = cb.job_pool(repro / "data" / "job" / "queries")
        elif wl == "pgbench":
            pools[wl] = cb.pgbench_pool()

    rows = []
    for wl, pool in pools.items():
        cal = steady_calibration(cb, pool)
        print(f"[cal ] {wl}: {len(cal)} steady vectors")
        for cfg in args.configs.split(","):
            for s in range(args.seeds):
                seed = cb.stable_seed(wl, cfg, s)
                windows, didx = cb.build_trajectory(pool, cfg, seed)
                fv, _ = cb.kernel_adjacent(windows)
                n = len(fv)
                truth = np.array([1 if (i + 1) in didx else 0 for i in range(n)])
                for policy, tau in TAUS.items():
                    gate = dg.DCIGate(tau=tau, alpha=0.05).fit(cal)
                    gate.reset_trajectory()
                    fired = np.zeros(n, dtype=int)
                    cost = 0.0
                    n5 = 0
                    for t in range(n):
                        fired[t] = gate.decide(fv[t])
                        if gate.last["regime"] == "5-D":
                            n5 += 1
                            cost += c5
                        else:
                            cost += c1
                    # STRICT onset scoring (aligned with the adjacent-window feature
                    # convention): the transient dip occupies exactly one f-index --- the
                    # transition pair --- which is the labeled index. Hit = fired AT the
                    # onset index; FA = any fire at a non-onset index. No tolerance
                    # window, so a spurious fire adjacent to an onset counts as FA, and
                    # alpha=0.05 implies a per-run FA floor of ~alpha*(n - onsets).
                    onsets = np.where(truth == 1)[0]
                    hits = int(fired[onsets].sum()) if len(onsets) else 0
                    fa = int(fired[truth == 0].sum())
                    rows.append({
                        "workload": wl, "config": cfg, "seed": s, "policy": policy,
                        "n_windows": n, "n_onsets": len(onsets), "onsets_hit": hits,
                        "recall": hits / len(onsets) if len(onsets) else float("nan"),
                        "mean_lag": 0.0,
                        "false_alarms": fa, "advisor_calls": int(fired.sum()),
                        "frac_5d_windows": n5 / n,
                        "monitor_cost_ms_per_window": cost / n,
                    })
        print(f"[done] {wl}")

    out = Path(args.outdir) / "dci_resolution_e2e_results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summary: pooled over workloads+seeds, per (config, policy) ------------
    key = {}
    for r in rows:
        key.setdefault((r["config"], r["policy"]), []).append(r)
    print(f"\n{'config':15s}{'policy':12s}{'recall':>8s}{'lag':>6s}{'FA/run':>8s}"
          f"{'calls/run':>10s}{'%5D':>7s}{'ms/win':>9s}{'%of5Dcost':>10s}")
    for (cfg, pol), rs in sorted(key.items()):
        rec = np.nanmean([x["recall"] for x in rs])
        lag = np.nanmean([x["mean_lag"] for x in rs])
        fa = np.mean([x["false_alarms"] for x in rs])
        calls = np.mean([x["advisor_calls"] for x in rs])
        f5 = np.mean([x["frac_5d_windows"] for x in rs])
        ms = np.mean([x["monitor_cost_ms_per_window"] for x in rs])
        print(f"{cfg:15s}{pol:12s}{rec:8.3f}{lag:6.2f}{fa:8.2f}{calls:10.2f}"
              f"{f5:7.1%}{ms:9.4f}{ms / c5:10.1%}")
    print(f"\n[out ] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
