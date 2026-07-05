#!/usr/bin/env python3
"""
mongo_signal.py
===============
Paper 3D -- Phase 2, task T2.3 (MongoDB signal-layer harness).

Computes the signal-layer DCI for the MongoDB workload under the SAME
synthetic drift schedules (template / volume / mixed) that Paper 3C's
cost_benefit.py drives on the relational workloads -- so MongoDB DCI is
directly comparable to TPC-H / JOB DCI for the cross-data-model
engine-free test (RQ1 / RQ2).

Design (PHASE2_PLAN.md Sec. 4; HARNESS_CONTRACT.md):
  3C's `build_trajectory(pool, config, seed)` is workload-agnostic --
  it drives template/volume drift on ANY pool of template ids. This
  harness imports it UNCHANGED, feeds it a MongoDB template pool, and
  runs the MongoDB feature path (canonical kernel via
  make_window_features + hsm_score_from_features) in place of the SQL
  path. DCI is then computed by 3C's `analyse()` -- the same function,
  unchanged -- so MongoDB DCI and relational DCI come from identical
  drift + DCI code. No database is touched (the kernel is pre-execution).

Outputs (out/<run_id>/; every row carries run_id + analysis_timestamp_utc):
  mongo_signal_raw.csv      one row per (config, seed): DCI + AUCs
  mongo_signal_summary.csv  per config: mean / sd / 95% CI
  mongo_signal_run.json     configuration + summary

Usage:
  python3 mongo_signal.py [--seeds 50] [--outdir out]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# ── locate the shared code on the path ────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))            # code/cross_engine/
_CODE = os.path.abspath(os.path.join(_HERE, ".."))            # code/
for p in (_CODE,                                              # for `kernel.*`
          os.path.join(_CODE, "geometry_E0"),                 # cost_benefit, delong
          os.path.join(_CODE, "kernel"),                      # hsm_v2_kernel
          os.path.join(_HERE, "common"),                      # window_features
          os.path.join(_HERE, "mongo", "workload")):          # templates
    if p not in sys.path:
        sys.path.insert(0, p)

# 3C drift generator + DCI analysis -- imported UNCHANGED.
from cost_benefit import build_trajectory, analyse, FEATURES   # noqa: E402
# MongoDB feature path.
from window_features import make_window_features              # noqa: E402
from hsm_v2_kernel import hsm_score_from_features             # noqa: E402
from templates import ALL_QIDS_SORTED                         # noqa: E402

CONFIGS = ["template_only", "volume_only", "mixed"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_utc_now()}] {msg}", flush=True)


def _mean_sd_ci(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {"mean": None, "sd": None, "ci95": None, "n": 0}
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    # t-based 95% CI half-width; t approx 1.96 for the seed counts here
    ci = 1.96 * sd / np.sqrt(a.size) if a.size > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95": float(ci), "n": int(a.size)}


def mongo_dci(config: str, seed: int):
    """One MongoDB drift trajectory -> 3C's analyse() result dict (or None)."""
    # Pool entry = the template id itself: build_trajectory's _make_window
    # picks rng.choice(pool[qid]) == qid, so each window is a list of
    # MongoDB template ids -- exactly the `qnames` make_window_features
    # wants. The drift schedule (template swaps, volume changes) is
    # identical to the relational path.
    pool = {qid: [qid] for qid in ALL_QIDS_SORTED}
    windows, drift_idx = build_trajectory(pool, config, seed)

    wfs = [make_window_features(qnames, ts, all_qids=ALL_QIDS_SORTED)
           for qnames, ts in windows]
    fv, sc = [], []
    for t in range(1, len(wfs)):
        hsm, dims = hsm_score_from_features(wfs[t - 1], wfs[t])
        fv.append([dims[k] for k in FEATURES])
        sc.append(float(hsm))
    return analyse(np.asarray(fv, float), np.asarray(sc, float), drift_idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=50,
                    help="seeds per config (official run: 50, matching 3C)")
    ap.add_argument("--outdir", default=os.path.join(_HERE, "out"))
    args = ap.parse_args()

    run_id = _utc_now().replace("-", "").replace(":", "")
    t0 = time.perf_counter()
    _log("mongo_signal.py - start")
    _log(f"  workload=mongodb  seeds={args.seeds}  "
         f"templates={len(ALL_QIDS_SORTED)}  configs={CONFIGS}")

    rows = []
    for config in CONFIGS:
        for seed in range(args.seeds):
            res = mongo_dci(config, seed)
            if res is None:
                continue
            rows.append({"workload": "mongodb", "config": config, "seed": seed,
                         **{k: res[k] for k in
                            ("dci", "auc_1d", "auc_5d", "auc_5d_l2",
                             "auc_5d_composite", "best_1d_dim", "delta_auc")}})
        done = sum(1 for r in rows if r["config"] == config)
        _log(f"  {config}: {done}/{args.seeds} seeds "
             f"(elapsed {time.perf_counter() - t0:.1f}s)")

    # ── outputs ──────────────────────────────────────────────────────
    out = os.path.join(args.outdir, run_id)
    os.makedirs(out, exist_ok=True)
    ts = _utc_now()
    raw_cols = ["analysis_timestamp_utc", "run_id", "workload", "config",
                "seed", "dci", "auc_1d", "auc_5d", "auc_5d_l2",
                "auc_5d_composite", "best_1d_dim", "delta_auc"]
    with open(os.path.join(out, "mongo_signal_raw.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(raw_cols)
        for r in rows:
            wr.writerow([ts, run_id] + [r[c] for c in raw_cols[2:]])

    summary = []
    with open(os.path.join(out, "mongo_signal_summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["analysis_timestamp_utc", "run_id", "workload", "config",
                     "n_seeds", "dci_mean", "dci_sd", "dci_ci95",
                     "auc_1d_mean", "auc_5d_mean"])
        for config in CONFIGS:
            sub = [r for r in rows if r["config"] == config]
            dci = _mean_sd_ci([r["dci"] for r in sub])
            a1 = _mean_sd_ci([r["auc_1d"] for r in sub])
            a5 = _mean_sd_ci([r["auc_5d"] for r in sub])
            summary.append({"config": config, "n": dci["n"],
                            "dci_mean": dci["mean"], "dci_sd": dci["sd"],
                            "dci_ci95": dci["ci95"],
                            "auc_1d_mean": a1["mean"], "auc_5d_mean": a5["mean"]})
            wr.writerow([ts, run_id, "mongodb", config, dci["n"],
                         dci["mean"], dci["sd"], dci["ci95"],
                         a1["mean"], a5["mean"]])

    with open(os.path.join(out, "mongo_signal_run.json"), "w") as f:
        json.dump({"analysis_timestamp_utc": ts, "run_id": run_id,
                   "script": "mongo_signal.py", "workload": "mongodb",
                   "seeds": args.seeds, "configs": CONFIGS,
                   "n_templates": len(ALL_QIDS_SORTED),
                   "n_rows": len(rows), "summary": summary,
                   "wall_seconds": round(time.perf_counter() - t0, 3)}, f, indent=2)

    for s in summary:
        if s["dci_mean"] is not None:
            _log(f"  {s['config']:14s} DCI={s['dci_mean']:.3f}"
                 f"+/-{s['dci_ci95']:.3f}  "
                 f"auc_1d={s['auc_1d_mean']:.3f} auc_5d={s['auc_5d_mean']:.3f}")
    _log(f"  written: {out}/")
    _log(f"mongo_signal.py - done | {len(rows)} cells | "
         f"elapsed {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
