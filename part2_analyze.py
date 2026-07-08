#!/usr/bin/env python3
"""part2_analyze.py -- S5 Part 2 result reader (Paper 3C, detector-resolution axis).

Reads the six RUN_PART2.sh outputs (PG/MG x tau in {0, 1.5, 1e9}) and reports,
per engine and per config, ONLY the Part 2 metrics (PART2_BUILD_SPEC.md):

  1. detection fidelity  : recall / false alarms per block against the mixed
                           schedule's ground-truth onsets (drift_truth column);
                           recall(+1) also shown (fire at onset or next window).
  2. firings             : advisor invocations per block.
  3. monitoring cost     : %5-D-routed windows (gate_regime column) x the
                           locked-env per-window detector costs -> ms/window
                           and % of the always-5-D detector cost.
  4. latency fidelity    : per-query exec-time ratio vs the always-5-D
                           reference, paired per (block, window) -- same seeds
                           => identical query draws across tau configs
                           (workload_fp equality is asserted). Reported on
                           steady windows, on post-onset windows
                           (onset..onset+2, where stale-index damage shows),
                           and as the DRIFT-CORRECTED contrast post/steady:
                           the tau legs run sequentially, so machine-state
                           drift (cache/thermal) inflates ALL windows of a
                           leg uniformly; the post/steady ratio cancels that
                           first-order and is the number to read as effect.
  + onset-type recall    : onsets are typed from the log itself (phase change
                           = template, n_queries change = volume) so the
                           Prop-2 story is decomposable: WHICH drift type
                           does the 1-D detector miss?
  + fire-set fidelity    : % windows where the config's firing decision equals
                           always-5-D's, and Jaccard of fired window sets.

NO wall_qps, NO advisor economics, NO sign-flip/regret -- the 3D/journal axis.

Usage:  python3 part2_analyze.py <path-to-repro_3c>
Writes: part2_summary.csv (one row per engine x config)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

TAGS = ["tau0", "tau1p5", "tau1e9"]
NICE = {"tau0": "always_5D", "tau1p5": "dci_gated", "tau1e9": "always_1D"}
REF = "tau0"


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def pg_paths(root: Path, tag: str):
    d = root / "end_to_end" / "postgres" / f"out_PART2_PG_mixed_{tag}"
    return d / "breakdown_per_window_v2.csv", d / "adaptation_comparison_v2.csv"


def mg_paths(root: Path, tag: str):
    base = root / "end_to_end" / "mongo" / "out" / f"PART2_MG_mixed_{tag}"
    runs = sorted([p for p in base.glob("*") if p.is_dir()])
    if not runs:
        return None, None
    if len(runs) > 1:
        print(f"  [warn] {base.name}: {len(runs)} runs, using latest {runs[-1].name}")
    return runs[-1] / "breakdown_per_window.csv", runs[-1] / "block_metrics.csv"


def load_engine(root: Path, engine: str):
    data = {}
    for tag in TAGS:
        bpath, mpath = (pg_paths if engine == "PG" else mg_paths)(root, tag)
        if bpath is None or not bpath.exists():
            print(f"  [miss] {engine} {tag}: no breakdown CSV -- skipped")
            continue
        rows = [r for r in read_csv(bpath) if r["strategy"] == "dci_gated"]
        blocks = {}
        for r in rows:
            blocks.setdefault(int(r["block"]), []).append(r)
        for b in blocks:
            blocks[b].sort(key=lambda r: int(r["window"]))
        data[tag] = {"blocks": blocks, "metrics": read_csv(mpath) if mpath.exists() else []}
    return data


def fp_check(engine: str, data: dict):
    """Paired-RCB sanity: same block => same workload fingerprint across taus."""
    fps = {}
    for tag, d in data.items():
        for m in d["metrics"]:
            if m.get("strategy") != "dci_gated":
                continue
            fps.setdefault(int(m["block"]), {})[tag] = m.get("workload_fp", "?")
    bad = {b: v for b, v in fps.items() if len(set(v.values())) > 1}
    if bad:
        print(f"  [FATAL?] {engine}: workload fingerprints differ across tau configs "
              f"in blocks {sorted(bad)} -- pairing broken, latency ratios invalid:")
        for b, v in sorted(bad.items()):
            print(f"      block {b}: {v}")
    else:
        print(f"  [ok] {engine}: workload fingerprints identical across tau configs "
              f"({len(fps)} blocks)")
    return not bad


def analyse(engine: str, data: dict, c1: float, c5: float, out_rows: list):
    print(f"\n════════ {engine} — live mixed-drift, 3-τ detector-resolution sweep ════════")
    if REF not in data:
        print("  [miss] always-5D reference absent; latency/fire fidelity skipped")
    fp_ok = fp_check(engine, data)

    # reference structures
    ref_fire, ref_lat = {}, {}
    if REF in data:
        for b, rows in data[REF]["blocks"].items():
            for r in rows:
                w = int(r["window"])
                ref_fire[(b, w)] = int(r["invoked"])
                nq = float(r["n_queries"])
                ref_lat[(b, w)] = float(r["exec_ms_window_sum"]) / nq if nq else np.nan

    hdr = (f"  {'config':11s}{'recall':>8s}{'rec(+1)':>9s}{'FA/blk':>8s}{'fires/blk':>10s}"
           f"{'%5D':>7s}{'ms/win':>8s}{'%5Dcost':>9s}{'fire==5D':>10s}{'lat.steady':>11s}"
           f"{'lat.post':>9s}{'post/std':>9s}{'fails':>7s}")
    print(hdr); print("  " + "─" * (len(hdr) - 2))

    for tag in TAGS:
        if tag not in data:
            continue
        blocks = data[tag]["blocks"]
        nb = len(blocks)
        hits = hits1 = onsets = fa = fires = n5 = nw = fails = agree = 0
        ratios, ratios_post, ratios_steady = [], [], []
        ty_n, ty_h = {}, {}
        for b, rows in blocks.items():
            onset_ws = {int(r["window"]) for r in rows if r["drift_truth"] == "1"}
            post_ws = {w + k for w in onset_ws for k in (0, 1, 2)}
            fired_ws = {int(r["window"]) for r in rows if r["invoked"] == "1"}
            # type each onset from the log itself: modality/template move
            # changes `phase`; a pure volume move changes only `n_queries`
            otype = {}
            prev_by_w = {int(r["window"]): r for r in rows}
            for w in onset_ws:
                r, q = prev_by_w.get(w), prev_by_w.get(w - 1)
                # volume first: PG names its volume phases (MixSurge1/MixCalm),
                # so "phase changed" alone would mistype pure count moves.
                # Every mixed-schedule onset changes exactly one of the two.
                if r is None or q is None:
                    otype[w] = "?"
                elif r["n_queries"] != q["n_queries"]:
                    otype[w] = "V"
                elif r["phase"] != q["phase"]:
                    otype[w] = "T"
                else:
                    otype[w] = "?"
            for r in rows:
                w = int(r["window"])
                inv = int(r["invoked"])
                fires += inv
                truth = r["drift_truth"] == "1"
                if truth:
                    onsets += 1
                    hits += inv
                    hits1 += int(w in fired_ws or (w + 1) in fired_ws)
                    t = otype.get(w, "?")
                    ty_n[t] = ty_n.get(t, 0) + 1
                    ty_h[t] = ty_h.get(t, 0) + int(w in fired_ws or (w + 1) in fired_ws)
                elif inv and (w - 1) not in onset_ws:
                    fa += 1          # fire not at an onset nor the +1 slot
                reg = r.get("gate_regime", "")
                if reg:
                    nw += 1
                    n5 += (reg == "5-D")
                fails += int(float(r.get("n_failed", 0) or 0))
                if REF in data and tag != REF and fp_ok:
                    agree += int(inv == ref_fire.get((b, w), -1))
                    nq = float(r["n_queries"])
                    lat = float(r["exec_ms_window_sum"]) / nq if nq else np.nan
                    ref = ref_lat.get((b, w), np.nan)
                    if lat > 0 and ref > 0:
                        ratios.append(lat / ref)
                        (ratios_post if w in post_ws else ratios_steady).append(lat / ref)
        rec = hits / onsets if onsets else float("nan")
        rec1 = hits1 / onsets if onsets else float("nan")
        f5 = n5 / nw if nw else float("nan")
        msw = (n5 * c5 + (nw - n5) * c1) / nw if nw else float("nan")
        n_win_total = sum(len(v) for v in blocks.values())
        agree_pc = agree / n_win_total if (REF in data and tag != REF and fp_ok) else float("nan")
        gm = math.exp(np.mean(np.log(ratios))) if ratios else float("nan")
        gms = math.exp(np.mean(np.log(ratios_steady))) if ratios_steady else float("nan")
        gmp = math.exp(np.mean(np.log(ratios_post))) if ratios_post else float("nan")
        did = gmp / gms if (gmp == gmp and gms == gms and gms > 0) else float("nan")
        ty_str = "  ".join(f"{t}:{ty_h.get(t,0)}/{ty_n[t]}" for t in sorted(ty_n))
        print(f"  {NICE[tag]:11s}{rec:8.3f}{rec1:9.3f}{fa / nb:8.2f}{fires / nb:10.2f}"
              f"{f5:7.1%}{msw:8.4f}{msw / c5:9.1%}"
              + (f"{agree_pc:10.1%}" if not math.isnan(agree_pc) else f"{'—':>10s}")
              + (f"{gms:11.3f}" if not math.isnan(gms) else f"{'1.000*':>11s}")
              + (f"{gmp:9.3f}" if not math.isnan(gmp) else f"{'1.000*':>9s}")
              + (f"{did:9.3f}" if not math.isnan(did) else f"{'—':>9s}")
              + f"{fails:7d}")
        print(f"  {'':11s}└ onset-type recall(+1): {ty_str}")
        out_rows.append({
            "engine": engine, "config": NICE[tag], "tau": {"tau0": 0.0, "tau1p5": 1.5, "tau1e9": 1e9}[tag],
            "blocks": nb, "onsets_total": onsets, "recall": round(rec, 4),
            "recall_lag1": round(rec1, 4), "fa_per_block": round(fa / nb, 3),
            "fires_per_block": round(fires / nb, 3), "frac_5d_windows": round(f5, 4),
            "monitor_ms_per_window": round(msw, 4), "pct_of_5d_cost": round(msw / c5, 4),
            "fire_agree_vs_5d": (round(agree_pc, 4) if not math.isnan(agree_pc) else ""),
            "latency_ratio_vs_5d_geomean": (round(gm, 4) if not math.isnan(gm) else 1.0),
            "latency_ratio_steady": (round(gms, 4) if not math.isnan(gms) else 1.0),
            "latency_ratio_post_onset": (round(gmp, 4) if not math.isnan(gmp) else 1.0),
            "latency_post_over_steady": (round(did, 4) if not math.isnan(did) else 1.0),
            "onset_type_recall": " ".join(f"{t}:{ty_h.get(t,0)}/{ty_n[t]}" for t in sorted(ty_n)),
            "queries_failed": fails,
        })
    print("  (*) reference config. ratios are vs always-5D, paired per (block,window); "
          "geometric means.")
    print("      lat.steady = non-post-onset windows (pure session-drift baseline); "
          "lat.post = onset..onset+2;")
    print("      post/std = drift-corrected contrast -- READ THIS as the stale-index "
          "effect, not the raw ratios.")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    run_json = root / "geometry_E0" / "out" / "20260705T135856Z" / "cost_benefit_run.json"
    ov = json.load(open(run_json))["overhead"]
    c1, c5 = ov["overhead_1d_s"] * 1e3, ov["overhead_5d_s"] * 1e3
    print(f"[cost] per-window detector cost: 1-D {c1:.4f} ms, 5-D {c5:.4f} ms "
          f"(locked-env {run_json.parent.name})")
    out_rows: list[dict] = []
    for engine in ("PG", "MG"):
        data = load_engine(root, engine)
        if data:
            analyse(engine, data, c1, c5, out_rows)
        else:
            print(f"\n[miss] no {engine} Part 2 outputs found")
    if out_rows:
        out = root / "part2_summary.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader(); w.writerows(out_rows)
        print(f"\n[out ] {out}")
    print("[note] Axis discipline: no wall_qps / economics here (3D axis). "
          "Paper numbers only from the locked-env run + provenance comment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
