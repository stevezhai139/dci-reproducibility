#!/usr/bin/env python3
"""
sdss_adapter.py
===============
Paper 3D -- Phase 2, task T2.2.

SDSS real-world-workload adapter. Turns the SDSS SkyServer query log
(a real, fixed query trace) into per-window HSM 5-D feature vectors and
a per-block DCI summary -- Paper 3D's *real-world robustness anchor*
(ENGINE_FREE_SCOPING.md Decision 3; PHASE2_PLAN.md Sec. 4).

SDSS is used as a TEXT + TIMESTAMP TRACE only: the HSM kernel is
pre-execution, so NO database is touched. The log's *natural* drift --
the query mix evolving over real time -- is the signal; there is no
synthetic drift schedule and no seeds.

Pipeline (HARNESS_CONTRACT.md Sec. 3-4):
  CSV -> drop malformed rows -> sort by theTime
      -> classify each statement into 10 subject categories
      -> windows of 20 queries (chronological)
      -> WindowFeatures (the canonical 6-key dict)
      -> canonical kernel -> adjacent-window f = (S_R,S_V,S_T,S_A,S_P)
      -> DCI per block of windows (participation ratio, Paper 3C Sec. 4)

Outputs (out/<run_id>/; every row carries run_id + analysis_timestamp_utc):
  sdss_windows.csv  one row per window -- the 5-D f vector + HSM
  sdss_dci.csv      one row per block  -- DCI
  sdss_run.json     configuration + summary

Usage:
  python3 sdss_adapter.py [--csv PATH] [--window 20] [--block 50]
                          [--max-rows N]   (--max-rows: smoke tests only)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np

# ── canonical kernel + canonical WindowFeatures builder ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "kernel"))
sys.path.insert(0, os.path.join(_HERE, "..", "cross_engine", "common"))
from hsm_v2_kernel import hsm_score_from_features      # noqa: E402
from window_features import make_window_features        # noqa: E402

csv.field_size_limit(10_000_000)   # SDSS statements can be very long

# ── SDSS query classifier ─────────────────────────────────────────────
# Vendored verbatim from Paper 3A's sdss_workload_analyzer.py. This is
# SDSS's templatisation: each statement -> one of 10 subject categories.
SDSS_QUERY_PATTERNS = {
    "PhotoObj":   r"\bPhotoObj\b|\bPhotoObjAll\b|\bPhotoTag\b",
    "SpecObj":    r"\bSpecObj\b|\bSpecObjAll\b|\bSpecLine\b",
    "Galaxy":     r"\bGalaxy\b|\bgalaxy\b",
    "Star":       r"\bStar\b|\bstar\b",
    "Quasar":     r"\bQuasar\b|\bQSO\b|\bqso\b",
    "Field":      r"\bField\b|\bRun\b|\bCamcol\b",
    "Coordinate": r"\bra\b|\bdec\b|\bRA\b|\bDEC\b|fGetNearestObjEq",
    "Redshift":   r"\bredshift\b|\bz\b.*\bFROM\b",
    "Metadata":   r"\bDBObjects\b|\bDBColumns\b|\bHistory\b|sys\.",
}
SDSS_CATEGORIES = list(SDSS_QUERY_PATTERNS.keys()) + ["Other"]   # 10 ids
# category is its own table/field surrogate for the S_A axis
_TABLES_MAP = {c: {c} for c in SDSS_CATEGORIES}
_FIELDS_MAP = {c: {c} for c in SDSS_CATEGORIES}


def classify_query(sql: str) -> str:
    """Classify an SDSS statement by its primary subject (10 categories)."""
    for category, pattern in SDSS_QUERY_PATTERNS.items():
        if re.search(pattern, sql, re.IGNORECASE):
            return category
    return "Other"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_utc_now()}] {msg}", flush=True)


# A complete SDSS record ends with the fixed 6-field tail
#   ,theTime,elapsed,busy,rows,dbname,error
# theTime is US-format  M/D/YYYY h:mm:ss AM/PM.
_TAIL_RE = re.compile(
    r",(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M),"
    r"([\d.eE+-]*),([\d.eE+-]*),(\d*),([^,\n]*),([^,\n]*)\s*$")


def load_sdss(csv_path: str, max_records):
    """Parse the SDSS log into chronologically-ordered query records.

    The SkyServer export is NOT a well-formed CSV: the `statement` field
    spans multiple physical lines and carries unquoted commas, so a
    standard csv reader mis-parses it. Instead we accumulate physical
    lines into a record buffer and close a record when the buffer ends
    with the fixed 6-field tail (`_TAIL_RE`). Records are then sorted by
    theTime (the log's real, if compressed, arrival order).

    Returns (records, n_malformed): each record is
    {category, elapsed_ms, t}.
    """
    records = []
    skipped = 0
    buf = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        f.readline()                                   # skip header line
        for line in f:
            buf.append(line.rstrip("\n"))
            joined = "\n".join(buf)
            m = _TAIL_RE.search(joined)
            if m is None:
                continue
            statement = joined[:m.start()].strip()
            buf = []
            if max_records is not None and len(records) >= max_records:
                break
            if len(statement) < 20:
                skipped += 1
                continue
            try:
                t = datetime.strptime(m.group(1), "%m/%d/%Y %I:%M:%S %p")
            except ValueError:
                t = datetime.min
            try:
                elapsed = max(float(m.group(2) or 0.0), 0.0)
            except ValueError:
                elapsed = 0.0
            records.append({
                "category": classify_query(statement),
                "elapsed_ms": elapsed,
                "t": t,
            })
    records.sort(key=lambda r: r["t"])
    return records, skipped


def compute_dci(feature_block: np.ndarray) -> float:
    """DCI = participation ratio of the drift-motion covariance
    (Paper 3C Sec. 4): DCI = trace(C)^2 / ||C||_F^2, computed on the
    per-window feature deviations d = f - mean(f over the block).
    Returned in [1, 5]; reused, unchanged, by the T2.3 harness.
    """
    if feature_block.shape[0] < 2:
        return float("nan")
    d = feature_block - feature_block.mean(axis=0, keepdims=True)
    C = (d.T @ d) / d.shape[0]
    fro2 = float(np.sum(C * C))
    if fro2 <= 0.0:
        return 1.0
    tr = float(np.trace(C))
    return float(tr * tr / fro2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_csv = os.path.join(_HERE, "..", "data", "sdss", "SkyLog_Workload.csv")
    ap.add_argument("--csv", default=default_csv)
    ap.add_argument("--window", type=int, default=20, help="queries per window")
    ap.add_argument("--block", type=int, default=50, help="windows per DCI block")
    ap.add_argument("--max-rows", type=int, default=None, dest="max_rows",
                    help="cap on records parsed (smoke tests only; "
                         "default = the whole log)")
    ap.add_argument("--outdir", default=os.path.join(_HERE, "out"))
    args = ap.parse_args()

    run_id = _utc_now().replace("-", "").replace(":", "")
    t0 = time.perf_counter()
    _log("sdss_adapter.py - start")
    _log(f"  csv={args.csv}  window={args.window}  block={args.block}")

    rows, skipped = load_sdss(args.csv, args.max_rows)
    _log(f"  loaded {len(rows):,} valid queries ({skipped:,} malformed rows dropped)")
    n_windows = len(rows) // args.window
    if n_windows < 2:
        _log("ERROR: not enough data for two windows")
        return 2
    _log(f"  {n_windows:,} windows of {args.window} queries")

    # Build per-window WindowFeatures, then adjacent-window f vectors.
    wfeats = []
    for w in range(n_windows):
        seg = rows[w * args.window:(w + 1) * args.window]
        wfeats.append(make_window_features(
            [r["category"] for r in seg],
            [r["elapsed_ms"] for r in seg],
            all_qids=SDSS_CATEGORIES,
            tables_map=_TABLES_MAP, fields_map=_FIELDS_MAP))

    feats = []          # f(t) = 5-D similarity of window t to window t-1
    for t in range(1, n_windows):
        hsm, dims = hsm_score_from_features(wfeats[t], wfeats[t - 1])
        feats.append((t, dims["S_R"], dims["S_V"], dims["S_T"],
                      dims["S_A"], dims["S_P"], hsm))
        if t % 2000 == 0:
            _log(f"  {t:,}/{n_windows - 1} windows scored "
                 f"(elapsed {time.perf_counter() - t0:.1f}s)")

    F = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in feats], dtype=float)

    # DCI per block of windows over the log's natural drift.
    dci_rows = []
    for b in range(0, len(F), args.block):
        chunk = F[b:b + args.block]
        if chunk.shape[0] >= 2:
            dci_rows.append((b // args.block, b, b + chunk.shape[0],
                             compute_dci(chunk)))

    # ── write outputs ────────────────────────────────────────────────
    out = os.path.join(args.outdir, run_id)
    os.makedirs(out, exist_ok=True)
    ts = _utc_now()
    with open(os.path.join(out, "sdss_windows.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["analysis_timestamp_utc", "run_id", "window",
                     "S_R", "S_V", "S_T", "S_A", "S_P", "HSM"])
        for r in feats:
            wr.writerow([ts, run_id] + [r[0]] + [f"{x:.10f}" for x in r[1:]])
    with open(os.path.join(out, "sdss_dci.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["analysis_timestamp_utc", "run_id", "block",
                     "window_start", "window_end", "dci"])
        for r in dci_rows:
            wr.writerow([ts, run_id, r[0], r[1], r[2], f"{r[3]:.10f}"])
    dci_vals = [r[3] for r in dci_rows]
    summary = {
        "analysis_timestamp_utc": ts, "run_id": run_id, "script": "sdss_adapter.py",
        "csv": os.path.abspath(args.csv), "window": args.window, "block": args.block,
        "max_rows": args.max_rows,
        "n_queries": len(rows), "n_malformed_dropped": skipped,
        "n_windows": n_windows, "n_feature_vectors": len(feats),
        "n_dci_blocks": len(dci_rows),
        "dci_mean": float(np.mean(dci_vals)) if dci_vals else None,
        "dci_sd": float(np.std(dci_vals, ddof=1)) if len(dci_vals) > 1 else None,
        "dci_min": float(np.min(dci_vals)) if dci_vals else None,
        "dci_max": float(np.max(dci_vals)) if dci_vals else None,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }
    with open(os.path.join(out, "sdss_run.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"  DCI over {len(dci_rows)} blocks: "
         f"mean={summary['dci_mean']:.4f} "
         f"range=[{summary['dci_min']:.4f}, {summary['dci_max']:.4f}]"
         if dci_vals else "  (no DCI blocks)")
    _log(f"  written: {out}/")
    _log(f"sdss_adapter.py - done | {len(feats)} windows | "
         f"elapsed {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
