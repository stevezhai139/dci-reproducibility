#!/usr/bin/env python3
"""part2_validate_offline.py -- S5 Part 2 pre-flight: DCI of the MIXED live schedules (Paper 3C).

PART2_BUILD_SPEC.md requires validating OFFLINE -- through the paper's own kernel --
that the new mixed-drift phase schedules (PG: tpch_mixed in pg_workloads.py;
MG: --schedule mixed via season_schedule.make_mixed_schedule) actually produce
multi-axis drift (block DCI in the mixed band, target ~1.7-2.0 like the Sec 6.2
mixed cells) BEFORE burning live compute on Steve's machine.

What it does, per engine (no database, no mongod -- pure Python):
  1. Rebuilds the exact per-window template/op draws of the live harness
     (same block seeds, same np.random / random.Random usage) with SYNTHETIC
     per-template exec times (deterministic hash-based base + small noise;
     conservative: no advisor feedback, no cache effects).
  2. Runs the canonical kernel adjacent-window feature path (the same
     make_window_features / hsm breakdown the harness logs).
  3. Reports:
     a. onset DCI (Sec 6.2 convention: participation ratio of the deviation
        covariance at ground-truth onset windows, deviations from the steady
        calibration mean) -- pooled across blocks + per-block;
     b. the gated DCIGate's ROLLING dci trace (what tau routing actually sees),
        segment medians and %windows >= tau;
     c. the 3-tau firing/cost table (always_5D / dci_gated / always_1D),
        recall & FA vs the schedule's ground-truth onsets.
  4. Prints PASS/CHECK verdicts against the build-spec band.

Sandbox previews are DEVELOPMENT data; paper numbers come only from the
locked-env run (provenance rule, RUN_HANDOFF).

Usage:
    python3 part2_validate_offline.py <path-to-repro_3c> [--engine pg|mg|both]
                                      [--blocks 10] [--seeds-cal 64]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

AXES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
TAUS = {"always_5D": 0.0, "dci_gated": 1.5, "always_1D": 1e9}
TARGET_BAND = (1.5, 2.6)          # acceptance; spec target ~1.7-2.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def synth_ms(key: str, rng: random.Random, lo=20.0, hi=200.0, sigma=0.20) -> float:
    """Deterministic per-template base exec time (hash -> log-uniform in
    [lo, hi] ms) + multiplicative lognormal noise. The base depends only on
    the template id, so S_P responds to template moves (as live) while
    volume moves stay pure count moves."""
    u = int(sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    base = lo * (hi / lo) ** u
    return base * math.exp(rng.gauss(0.0, sigma))


# ---------------------------------------------------------------- PG stream
def pg_streams(repro: Path, n_blocks: int, n_cal: int):
    """Yield (cal_features, blocks) for the tpch_mixed live schedule.
    Mirrors pg_adaptation.py calibrate_dci_gate()/run_block() draw logic
    exactly (np.random for template draws, seeded per block)."""
    pgdir = repro / "end_to_end" / "postgres"
    sys.path.insert(0, str(pgdir))
    # import pg_adaptation configured for tpch_mixed (uses its exact
    # make_window_features / compute_hsm_breakdown / PHASES / constants)
    argv_save = sys.argv
    sys.argv = ["pg_adaptation.py", "--workload", "tpch_mixed"]
    try:
        pga = load_module(pgdir / "pg_adaptation.py", "pga_mixed")
    finally:
        sys.argv = argv_save
    PH = pga.PHASES
    QPW = pga.QUERIES_PW

    def draw(ph, trng):
        wts = np.array(ph["w"], float); wts /= wts.sum()
        qn = list(np.random.choice(ph["qs"], size=ph.get("qpw", QPW),
                                   p=wts, replace=True))
        ex = [synth_ms("pg:" + q, trng) for q in qn]
        return pga.make_window_features(qn, ex)

    # calibration: steady phase-0 chain, n_cal adjacent pairs
    np.random.seed(20260525)
    trng = random.Random(20260525)
    prev = draw(PH[0], trng)
    cal = []
    for _ in range(n_cal):
        cur = draw(PH[0], trng)
        b = pga.compute_hsm_breakdown(prev, cur)
        cal.append([b[a] for a in AXES])
        prev = cur
    # blocks: base_seed 7100 (sf=1.0), block_seed = base + block*100
    blocks = []
    for blk in range(1, n_blocks + 1):
        seed = 7100 + blk * 100
        np.random.seed(seed)
        trng = random.Random(seed ^ 0x5EED)
        prev = draw(PH[0], trng)                      # window 0 init
        feats, truth = [], []
        for win in range(1, pga.N_WINDOWS + 1):
            ph_idx = min((win - 1) // pga.WIN_PER_PH, len(PH) - 1)
            cur = draw(PH[ph_idx], trng)
            b = pga.compute_hsm_breakdown(prev, cur)
            feats.append([b[a] for a in AXES])
            if win == 1:
                truth.append(0)
            else:
                prev_idx = min((win - 2) // pga.WIN_PER_PH, len(PH) - 1)
                truth.append(int(ph_idx != prev_idx))
            prev = cur
        blocks.append((np.asarray(feats), np.asarray(truth)))
    # onset map from the config geometry: phase p starts at window
    # p*win_per_ph+1; boundary type alternates template/volume (§6.2 rotation)
    onset_types = {ph * pga.WIN_PER_PH + 1: ("template" if ph % 2 == 1 else "volume")
                   for ph in range(1, len(PH))}
    return np.asarray(cal), blocks, onset_types


# ---------------------------------------------------------------- MG stream
def mg_streams(repro: Path, n_blocks: int, n_cal: int):
    """Same for the Mongo --schedule mixed path (season_schedule.
    make_mixed_schedule + templates.py phase mixes + window_features)."""
    sys.path.insert(0, str(repro / "cross_engine" / "common"))
    sys.path.insert(0, str(repro / "cross_engine" / "mongo" / "workload"))
    sys.path.insert(0, str(repro / "end_to_end" / "mongo"))
    import templates as T                      # noqa
    from window_features import make_window_features   # noqa
    from hsm_bridge import compute_window_hsm_breakdown  # noqa
    season = load_module(repro / "end_to_end" / "mongo" / "season_schedule.py",
                         "season_mixed")
    pw, cw, btypes = season.make_mixed_schedule(24)   # builder defaults = live

    def gen(phase_name, n, rng):
        phase = T.ALL_PHASES[phase_name]
        qids = list(phase["mix"].keys())
        weights = [phase["mix"][q] for q in qids]
        return rng.choices(qids, weights=weights, k=n)

    def feats_of(qids, trng):
        ex = [synth_ms("mg:" + q, trng) for q in qids]
        return make_window_features(qids, ex)

    # calibration: steady edge@20 chain (mirrors calibrate_dci_gate)
    rng = random.Random(20260525)
    trng = random.Random(20260525 ^ 0x5EED)
    prev = feats_of(gen(pw[0], 20, rng), trng)
    cal = []
    for _ in range(n_cal):
        cur = feats_of(gen(pw[0], 20, rng), trng)
        b = compute_window_hsm_breakdown(prev, cur)
        cal.append([b[a] for a in AXES])
        prev = cur
    # blocks: BASE_SEED 9000 + b*100, window0 init + 24 measured
    blocks = []
    for b_i in range(n_blocks):
        seed = 9000 + b_i * 100
        rng = random.Random(seed)
        trng = random.Random(seed ^ 0x5EED)
        prev = feats_of(gen(pw[0], cw[0], rng), trng)     # window 0
        feats, truth = [], []
        for w in range(1, 25):
            qids = gen(pw[w - 1], cw[w - 1], rng)
            cur = feats_of(qids, trng)
            bd = compute_window_hsm_breakdown(prev, cur)
            feats.append([bd[a] for a in AXES])
            if w == 1:
                truth.append(0)
            else:
                truth.append(int(pw[w - 1] != pw[w - 2] or cw[w - 1] != cw[w - 2]))
            prev = cur
        blocks.append((np.asarray(feats), np.asarray(truth)))
    onset_types = {k + 1: v for k, v in btypes.items()}
    return np.asarray(cal), blocks, onset_types


# ---------------------------------------------------------------- analysis
def participation_ratio(C):
    ev = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    tot, fro2 = float(ev.sum()), float((ev ** 2).sum())
    return (tot * tot) / fro2 if tot > 0 and fro2 > 0 else 1.0


def analyse_engine(tag, cal, blocks, onset_types, dg, c1, c5, csv_rows):
    mu0 = cal.mean(axis=0)
    print(f"\n================ {tag} : mixed schedule ================")
    print(f"[cal ] {len(cal)} steady vectors  mu0 = "
          + " ".join(f"{a}={m:.3f}" for a, m in zip(AXES, mu0)))
    onsets_1b = sorted(onset_types)

    # (a) Sec 6.2-convention onset DCI
    per_block_dci, pooled = [], []
    for feats, truth in blocks:
        d = feats[[w - 1 for w in onsets_1b]] - mu0
        pooled.extend(list(d))
        per_block_dci.append(participation_ratio((d.T @ d) / len(d)))
    D = np.asarray(pooled)
    dci_pooled = participation_ratio((D.T @ D) / len(D))
    pb = np.asarray(per_block_dci)
    print(f"[DCI ] onset-deviation DCI (Sec 6.2 convention, {len(onsets_1b)} onsets/block):")
    print(f"        pooled over {len(blocks)} blocks = {dci_pooled:.3f}")
    print(f"        per-block  mean={pb.mean():.3f}  min={pb.min():.3f}  "
          f"max={pb.max():.3f}")
    # per-axis mean deviation at each onset type (diagnostic)
    for w in onsets_1b:
        dv = np.asarray([f[w - 1] - mu0 for f, _ in blocks])
        print(f"        onset win{w:>3d} ({onset_types[w]:8s}) mean |dev| = "
              + " ".join(f"{a}={abs(m):.3f}" for a, m in zip(AXES, dv.mean(axis=0))))

    # (b) rolling gate DCI + (c) 3-tau table
    print(f"\n{'config':12s}{'recall':>8s}{'FA/blk':>8s}{'fires/blk':>10s}"
          f"{'%5D':>8s}{'ms/win':>9s}{'%of5D':>8s}   rolling-DCI med [>=2nd onset]")
    verdicts = {}
    for pol, tau in TAUS.items():
        rec_n = rec_d = fa = fires = n5 = nw = 0
        cost = 0.0
        roll_last = []
        onsets_1b_local = onsets_1b
        for feats, truth in blocks:
            gate = dg.DCIGate(tau=tau, alpha=0.05).fit(cal)
            gate.reset_trajectory()
            for t in range(len(feats)):
                f = gate.decide(feats[t])
                is5 = gate.last["regime"] == "5-D"
                n5 += is5; nw += 1
                cost += c5 if is5 else c1
                if truth[t]:
                    rec_d += 1; rec_n += f
                elif f:
                    fa += 1
                fires += f
                dci_t = gate.last["dci"]
                if pol == "dci_gated":
                    csv_rows.append({"engine": tag, "block": len(roll_last),
                                     "window": t + 1,
                                     "dci_roll": ("" if math.isnan(dci_t) else round(dci_t, 4)),
                                     "regime": gate.last["regime"],
                                     "fired": f, "truth": int(truth[t])})
                if t + 1 >= onsets_1b[1]:      # from the 2nd onset on
                    roll_last.append(dci_t)
        nb = len(blocks)
        med = float(np.nanmedian(roll_last)) if roll_last else float("nan")
        f5 = n5 / nw
        msw = cost / nw
        print(f"{pol:12s}{rec_n / rec_d:8.3f}{fa / nb:8.2f}{fires / nb:10.2f}"
              f"{f5:8.1%}{msw:9.4f}{msw / c5:8.1%}   "
              + (f"{med:.3f}" if pol == "dci_gated" else "-"))
        verdicts[pol] = dict(recall=rec_n / max(rec_d, 1), f5=f5, med=med)

    ok_band = TARGET_BAND[0] <= dci_pooled <= TARGET_BAND[1]
    g = verdicts["dci_gated"]
    ok_roll = g["med"] >= 1.5 if not math.isnan(g["med"]) else False
    ok_route = verdicts["always_1D"]["f5"] < 0.2 < g["f5"] < verdicts["always_5D"]["f5"]
    print(f"\n[verdict:{tag}]")
    print(f"  {'PASS' if ok_band else 'CHECK'}  pooled onset DCI {dci_pooled:.3f} "
          f"in target band [{TARGET_BAND[0]}, {TARGET_BAND[1]}] (spec ~1.7-2.0)")
    print(f"  {'PASS' if ok_roll else 'CHECK'}  gated rolling DCI median (win>={onsets_1b[1]}) "
          f"= {g['med']:.3f} >= tau 1.5 -> block reaches the multi-axis regime live")
    print(f"  {'PASS' if ok_route else 'CHECK'}  routing separates: "
          f"always_1D %5D={verdicts['always_1D']['f5']:.1%} < gated "
          f"%5D={g['f5']:.1%} < always_5D={verdicts['always_5D']['f5']:.1%}")
    return dci_pooled, ok_band and ok_roll and ok_route


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=str)
    ap.add_argument("--engine", choices=["pg", "mg", "both"], default="both")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--seeds-cal", type=int, default=64)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    repro = Path(args.repro).resolve()

    dg = load_module(repro / "end_to_end" / "dci_gate.py", "dci_gate_p2v")
    run_json = repro / "geometry_E0" / "out" / "20260705T135856Z" / "cost_benefit_run.json"
    ov = json.load(open(run_json))["overhead"]
    c1, c5 = ov["overhead_1d_s"] * 1e3, ov["overhead_5d_s"] * 1e3
    print(f"[cost] 1-D {c1:.4f} ms  5-D {c5:.4f} ms per window "
          f"(locked-env {run_json.parent.name})")

    csv_rows: list[dict] = []
    results = {}
    if args.engine in ("pg", "both"):
        cal, blocks, ot = pg_streams(repro, args.blocks, args.seeds_cal)
        results["PG"] = analyse_engine("PG", cal, blocks, ot, dg, c1, c5, csv_rows)
    if args.engine in ("mg", "both"):
        cal, blocks, ot = mg_streams(repro, args.blocks, args.seeds_cal)
        results["MG"] = analyse_engine("MG", cal, blocks, ot, dg, c1, c5, csv_rows)

    out = Path(args.outdir) / "part2_validate_offline_trace.csv"
    if csv_rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader(); w.writerows(csv_rows)
        print(f"\n[out ] rolling-DCI trace -> {out}")
    all_ok = all(ok for _, ok in results.values())
    print("[note] synthetic exec times, no advisor feedback -- structural "
          "pre-validation only; paper numbers come from the live locked-env run.")
    print(f"[exit] {'ALL PASS' if all_ok else 'CHECK ITEMS ABOVE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
