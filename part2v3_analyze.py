#!/usr/bin/env python3
"""part2v3_analyze.py -- Part 2 v3 (gate v3, decision A) three-arm analyzer.

Faithful adaptation of part2_analyze.py (which stays untouched so the official
v1 Part 2 analysis reproduces). Differences:

  arms    : afull (always-full reference) / gated (DCIGateV3, rho=0.35)
            / acheap (always-cheap union) -- via DCI_FORCE, RUN_PART2_V3.sh
  cost    : MEASURED per-window det_ms from the log (decision A), not modeled
            constants. Reported per arm + ratio vs the always-full arm.
  regime  : gate_regime is 'cheap'|'full'; escalation fraction = %full windows
            in the gated arm (with gate_audit and sp_out_of_band accounting).

Scoring conventions identical to v1: strict recall on drift_truth windows,
recall(+1), FA excluding the onset+1 slot, onset typing from the log itself
(n_queries move => V, else phase move => T), paired latency ratios vs the
reference arm (geomean; steady vs post-onset), workload-fp pairing check.

Usage: python3 part2v3_analyze.py [root]
Writes: part2v3_summary.csv
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

TAGS = ["afull", "gated", "acheap"]
NICE = {"afull": "always_full", "gated": "dci_gated_v3", "acheap": "always_cheap"}
REF = "afull"


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def pg_paths(root: Path, tag: str):
    d = root / "end_to_end" / "postgres" / f"out_PART2V3_PG_mixed_{tag}"
    return d / "breakdown_per_window_v2.csv", d / "adaptation_comparison_v2.csv"


def mg_paths(root: Path, tag: str):
    base = root / "end_to_end" / "mongo" / "out" / f"PART2V3_MG_mixed_{tag}"
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
    fps = {}
    for tag, d in data.items():
        for m in d["metrics"]:
            if m.get("strategy") != "dci_gated":
                continue
            fps.setdefault(int(m["block"]), {})[tag] = m.get("workload_fp", "?")
    bad = {b: v for b, v in fps.items() if len(set(v.values())) > 1}
    if bad:
        print(f"  [FATAL?] {engine}: workload fingerprints differ across arms "
              f"in blocks {sorted(bad)} -- pairing broken, latency ratios invalid:")
        for b, v in sorted(bad.items()):
            print(f"      block {b}: {v}")
    else:
        print(f"  [ok] {engine}: workload fingerprints identical across arms "
              f"({len(fps)} blocks)")
    return not bad


def analyse(engine: str, data: dict, out_rows: list):
    print(f"\n════════ {engine} — live mixed-drift, gate-v3 three-arm study ════════")
    if REF not in data:
        print("  [miss] always-full reference absent; agreement/latency skipped")
    fp_ok = fp_check(engine, data)

    ref_fire, ref_lat, ref_det = {}, {}, []
    if REF in data:
        for b, rows in data[REF]["blocks"].items():
            for r in rows:
                w = int(r["window"])
                ref_fire[(b, w)] = int(r["invoked"])
                nq = float(r["n_queries"])
                ref_lat[(b, w)] = float(r["exec_ms_window_sum"]) / nq if nq else np.nan
                if r.get("det_ms"):
                    ref_det.append(float(r["det_ms"]))
    ref_det_mean = float(np.mean(ref_det)) if ref_det else float("nan")

    hdr = (f"  {'arm':13s}{'recall':>8s}{'rec(+1)':>9s}{'FA/blk':>8s}{'fires/blk':>10s}"
           f"{'%full':>7s}{'det ms/win':>11s}{'vs afull':>9s}{'fire==ref':>10s}"
           f"{'lat.steady':>11s}{'lat.post':>9s}{'post/std':>9s}{'fails':>7s}")
    print(hdr); print("  " + "─" * (len(hdr) - 2))

    for tag in TAGS:
        if tag not in data:
            continue
        blocks = data[tag]["blocks"]
        nb = len(blocks)
        hits = hits1 = onsets = fa = fires = nfull = nw = fails = agree = 0
        audits = sp_oob = 0
        det = []
        ratios, ratios_post, ratios_steady = [], [], []
        ty_n, ty_h = {}, {}
        for b, rows in blocks.items():
            onset_ws = {int(r["window"]) for r in rows if r["drift_truth"] == "1"}
            post_ws = {w + k for w in onset_ws for k in (0, 1, 2)}
            fired_ws = {int(r["window"]) for r in rows if r["invoked"] == "1"}
            otype = {}
            prev_by_w = {int(r["window"]): r for r in rows}
            for w in onset_ws:
                r, q = prev_by_w.get(w), prev_by_w.get(w - 1)
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
                    fa += 1
                reg = r.get("gate_regime", "")
                if reg:
                    nw += 1
                    nfull += (reg == "full")
                audits += int((r.get("gate_audit") or "0") == "1")
                sp_oob += int((r.get("sp_out_of_band") or "0") == "1")
                if r.get("det_ms"):
                    det.append(float(r["det_ms"]))
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
        ffull = nfull / nw if nw else float("nan")
        det_mean = float(np.mean(det)) if det else float("nan")
        det_vs_ref = det_mean / ref_det_mean if ref_det_mean == ref_det_mean else float("nan")
        n_win_total = sum(len(v) for v in blocks.values())
        agree_pc = agree / n_win_total if (REF in data and tag != REF and fp_ok) else float("nan")
        gm = math.exp(np.mean(np.log(ratios))) if ratios else float("nan")
        gms = math.exp(np.mean(np.log(ratios_steady))) if ratios_steady else float("nan")
        gmp = math.exp(np.mean(np.log(ratios_post))) if ratios_post else float("nan")
        did = gmp / gms if (gmp == gmp and gms == gms and gms > 0) else float("nan")
        ty_str = "  ".join(f"{t}:{ty_h.get(t,0)}/{ty_n[t]}" for t in sorted(ty_n))
        print(f"  {NICE[tag]:13s}{rec:8.3f}{rec1:9.3f}{fa / nb:8.2f}{fires / nb:10.2f}"
              f"{ffull:7.1%}{det_mean:11.3f}"
              + (f"{det_vs_ref:9.1%}" if not math.isnan(det_vs_ref) else f"{'—':>9s}")
              + (f"{agree_pc:10.1%}" if not math.isnan(agree_pc) else f"{'—':>10s}")
              + (f"{gms:11.3f}" if not math.isnan(gms) else f"{'1.000*':>11s}")
              + (f"{gmp:9.3f}" if not math.isnan(gmp) else f"{'1.000*':>9s}")
              + (f"{did:9.3f}" if not math.isnan(did) else f"{'—':>9s}")
              + f"{fails:7d}")
        print(f"  {'':13s}└ onset-type recall(+1): {ty_str}"
              + (f"   audits={audits} sp_oob={sp_oob}" if tag == "gated" else ""))
        out_rows.append({
            "engine": engine, "arm": NICE[tag], "blocks": nb,
            "onsets_total": onsets, "recall": round(rec, 4),
            "recall_lag1": round(rec1, 4), "fa_per_block": round(fa / nb, 3),
            "fires_per_block": round(fires / nb, 3),
            "frac_full_windows": round(ffull, 4),
            "det_ms_per_window_measured": round(det_mean, 4),
            "det_cost_vs_always_full": (round(det_vs_ref, 4)
                                        if not math.isnan(det_vs_ref) else ""),
            "fire_agree_vs_ref": (round(agree_pc, 4) if not math.isnan(agree_pc) else ""),
            "latency_ratio_steady": (round(gms, 4) if not math.isnan(gms) else 1.0),
            "latency_ratio_post_onset": (round(gmp, 4) if not math.isnan(gmp) else 1.0),
            "latency_post_over_steady": (round(did, 4) if not math.isnan(did) else 1.0),
            "onset_type_recall": " ".join(f"{t}:{ty_h.get(t,0)}/{ty_n[t]}"
                                          for t in sorted(ty_n)),
            "audit_windows": audits, "sp_out_of_band_windows": sp_oob,
            "queries_failed": fails,
        })
    print("  (*) reference arm. ratios vs always-full, paired per (block,window); "
          "geometric means.")
    print("      det ms/win = MEASURED conditional detector path (decision A): cheap "
          "sims + lazy S_P when escalated.")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out_rows: list[dict] = []
    for engine in ("PG", "MG"):
        data = load_engine(root, engine)
        if data:
            analyse(engine, data, out_rows)
        else:
            print(f"\n[miss] no {engine} Part 2 v3 outputs found")
    if out_rows:
        out = root / "part2v3_summary.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader(); w.writerows(out_rows)
        print(f"\n[out ] {out}")
    print("[note] Axis discipline unchanged: no wall_qps / economics here (3D axis). "
          "Paper numbers only from the locked-env run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
