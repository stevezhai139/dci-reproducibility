#!/usr/bin/env python3
"""redset_gonogo.py -- Day-1/2 go--no-go for the Redset splice study (Paper 3C).

Four tests on the official 1% sample (provisioned fleet), per the study spec:
  T1  fingerprint reuse      : do top-50 fingerprints cover enough of the load?
                               (S_R and S_T both live on fingerprint frequency
                               vectors; a flat fingerprint distribution kills both)
  T2  stability across time  : cosine of adjacent fixed-count windows' frequency
                               vectors within a cluster (steady signal vs noise)
  T3  cross-tenant contrast  : chosen clusters must differ visibly (freq cosine
                               across clusters low; table-vocabulary Jaccard low)
                               while within-cluster stays high
  T4  timestamp granularity  : arrival_timestamp resolution + window time-span
                               => is an S_P arrival-QPS series viable at W=1000?

Redset schema (amazon-science/redset): instance_id, user_id, database_id,
query_id, arrival_timestamp, compile/queue/execution_duration_ms,
feature_fingerprint, was_aborted, was_cached, cache_source_query_id,
query_type, num_*_tables_accessed, read_table_ids, write_table_ids,
mbytes_scanned, mbytes_spilled, num_joins, num_scans, num_aggregations.
NOTE (honesty, from the README): feature_fingerprint is a hash proxy for
query-likeness, not text, and WILL OVERESTIMATE repetition -- must be declared
in the paper if the study proceeds.

Usage:
  python3 redset_gonogo.py data/sample_0.01.parquet [--clusters 3] [--win 1000]
Writes: redset_gonogo_report.json (+ console verdicts)
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
import numpy as np
import pandas as pd

NEED = ["instance_id", "arrival_timestamp", "feature_fingerprint", "query_type",
        "read_table_ids", "was_aborted", "was_cached", "user_id"]


def freq_vector(fps, vocab):
    c = Counter(fps)
    v = np.array([c.get(f, 0) for f in vocab], float)
    s = v.sum()
    return v / s if s > 0 else v


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def tables_of(series):
    out = set()
    for x in series.dropna():
        for t in str(x).split(","):
            t = t.strip()
            if t:
                out.add(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--win", type=int, default=1000)
    ap.add_argument("--min-span-days", type=int, default=56)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=None)
    missing = [c for c in NEED if c not in df.columns]
    if missing:
        print(f"[FATAL] missing columns {missing}; have: {list(df.columns)}")
        return 2
    df = df[NEED].copy()
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    n_total = len(df)
    n_aborted = int(df["was_aborted"].sum()) if df["was_aborted"].notna().any() else 0
    df = df[df["was_aborted"] != True]  # noqa: E712  (keep cached: they are real arrivals)
    print(f"[load] {n_total:,} rows; dropped aborted {n_aborted:,}; kept {len(df):,}")
    print(f"[load] query_type mix (top6): {df['query_type'].value_counts().head(6).to_dict()}")

    # ── cluster selection: volume + span ─────────────────────────────
    g = df.groupby("instance_id").agg(
        n=("feature_fingerprint", "size"),
        t0=("arrival_timestamp", "min"),
        t1=("arrival_timestamp", "max"))
    g["span_days"] = (g["t1"] - g["t0"]).dt.total_seconds() / 86400
    ok = g[g["span_days"] >= args.min_span_days].sort_values("n", ascending=False)
    chosen = list(ok.head(args.clusters).index)
    print(f"\n[clusters] chosen (of {len(g)} with span>={args.min_span_days}d: {len(ok)}):")
    for cid in chosen:
        r = g.loc[cid]
        print(f"  instance {cid}: n={int(r['n']):,} (sample) span={r['span_days']:.0f}d")

    report = {"file": args.parquet, "win": args.win, "chosen": [str(c) for c in chosen],
              "tests": {}}
    verdicts = {}

    # ── T1 fingerprint reuse ─────────────────────────────────────────
    t1 = {}
    print("\n══ T1 fingerprint reuse (need Zipf-like, not flat) ══")
    for cid in chosen:
        fps = df.loc[df.instance_id == cid, "feature_fingerprint"]
        c = Counter(fps); n = len(fps); d = len(c)
        top = c.most_common(100)
        cov = lambda k: sum(v for _, v in top[:k]) / n * 100
        row = {"n": n, "distinct": d, "distinct_pct": round(d / n * 100, 2),
               "top1": round(cov(1), 1), "top10": round(cov(10), 1),
               "top50": round(cov(50), 1), "top100": round(cov(100), 1)}
        t1[str(cid)] = row
        print(f"  {cid}: distinct {d:,}/{n:,} ({row['distinct_pct']}%)  "
              f"coverage top1/10/50/100 = {row['top1']}/{row['top10']}/"
              f"{row['top50']}/{row['top100']}%")
    worst50 = min(r["top50"] for r in t1.values())
    verdicts["T1"] = "PASS" if worst50 >= 40 else ("MARGINAL" if worst50 >= 20 else "FAIL")
    report["tests"]["T1"] = t1

    # ── T2 adjacent-window stability ─────────────────────────────────
    t2 = {}
    print(f"\n══ T2 adjacent-window stability (W={args.win} queries, fingerprint freq cosine) ══")
    for cid in chosen:
        sub = df[df.instance_id == cid].sort_values("arrival_timestamp")
        fps = sub["feature_fingerprint"].to_numpy()
        W = args.win
        nw = len(fps) // W
        cosines = []
        for i in range(nw - 1):
            a, b = fps[i*W:(i+1)*W], fps[(i+1)*W:(i+2)*W]
            vocab = sorted(set(a) | set(b))
            cosines.append(cosine(freq_vector(a, vocab), freq_vector(b, vocab)))
        arr = np.array([c for c in cosines if c == c])
        row = {"windows": nw, "pairs": len(arr),
               "median": round(float(np.median(arr)), 3) if len(arr) else None,
               "p10": round(float(np.quantile(arr, .1)), 3) if len(arr) else None,
               "p90": round(float(np.quantile(arr, .9)), 3) if len(arr) else None}
        t2[str(cid)] = row
        print(f"  {cid}: {nw} windows, adjacent cosine median={row['median']} "
              f"p10={row['p10']} p90={row['p90']}")
    med_min = min((r["median"] or 0) for r in t2.values())
    verdicts["T2"] = "PASS" if med_min >= 0.6 else ("MARGINAL" if med_min >= 0.4 else "FAIL")
    report["tests"]["T2"] = t2

    # ── T3 cross-tenant contrast ─────────────────────────────────────
    print("\n══ T3 cross-tenant contrast (across should be << within) ══")
    mean_vecs, tabsets, within = {}, {}, {}
    for cid in chosen:
        sub = df[df.instance_id == cid]
        fps = sub["feature_fingerprint"]
        vocab = [f for f, _ in Counter(fps).most_common(2000)]
        mean_vecs[cid] = (vocab, freq_vector(fps, vocab))
        tabsets[cid] = tables_of(sub["read_table_ids"])
        within[cid] = t2[str(cid)]["median"]
    t3 = {"pairs": {}}
    for i, a in enumerate(chosen):
        for b in chosen[i+1:]:
            va, vb = mean_vecs[a], mean_vecs[b]
            vocab = sorted(set(va[0]) | set(vb[0]))
            ca = freq_vector(df.loc[df.instance_id == a, "feature_fingerprint"], vocab)
            cb = freq_vector(df.loc[df.instance_id == b, "feature_fingerprint"], vocab)
            cos_ab = round(cosine(ca, cb), 3)
            ja, jb = tabsets[a], tabsets[b]
            jac = round(len(ja & jb) / len(ja | jb), 3) if (ja | jb) else None
            t3["pairs"][f"{a}x{b}"] = {"freq_cosine": cos_ab, "table_jaccard": jac}
            print(f"  {a} x {b}: cross freq-cosine={cos_ab} (within medians "
                  f"{within[a]}/{within[b]}), table-Jaccard={jac}")
    max_cross = max(p["freq_cosine"] for p in t3["pairs"].values())
    verdicts["T3"] = "PASS" if max_cross <= 0.5 else ("MARGINAL" if max_cross <= 0.8 else "FAIL")
    report["tests"]["T3"] = t3

    # ── T4 timestamp granularity / S_P viability ─────────────────────
    print("\n══ T4 arrival_timestamp granularity (S_P viability) ══")
    t4 = {}
    for cid in chosen:
        ts = df.loc[df.instance_id == cid, "arrival_timestamp"].sort_values()
        deltas = ts.diff().dropna().dt.total_seconds()
        pos = deltas[deltas > 0]
        subsec = float((ts.dt.microsecond > 0).mean() + (ts.dt.nanosecond > 0).mean())
        span_s = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
        qps_sample = len(ts) / span_s if span_s > 0 else float("nan")
        # full-data window span estimate: sample is 1% uniform -> x100 rate
        est_win_span_min = args.win / (qps_sample * 100) / 60 if qps_sample > 0 else None
        row = {"min_pos_delta_s": round(float(pos.min()), 4) if len(pos) else None,
               "subsec_frac": round(subsec, 3),
               "sample_qps": round(qps_sample, 4),
               "est_full_win_span_min": round(est_win_span_min, 1) if est_win_span_min else None}
        t4[str(cid)] = row
        print(f"  {cid}: min +delta={row['min_pos_delta_s']}s subsec_frac={row['subsec_frac']} "
              f"est. full-data W={args.win} span ≈ {row['est_full_win_span_min']} min")
    spans = [r["est_full_win_span_min"] for r in t4.values() if r["est_full_win_span_min"]]
    verdicts["T4"] = "PASS" if spans and max(spans) <= 24*60 else ("MARGINAL" if spans else "FAIL")
    report["tests"]["T4"] = t4

    # ── verdicts ─────────────────────────────────────────────────────
    print("\n══════ VERDICTS ══════")
    for k in ("T1", "T2", "T3", "T4"):
        print(f"  {k}: {verdicts[k]}")
    go = all(v != "FAIL" for v in verdicts.values()) and verdicts["T1"] != "MARGINAL"
    print(f"\n  GO/NO-GO suggestion: {'GO' if go else 'NO-GO (or switch fleet/SnowSet per spec)'}")
    print("  (T1 is the hard gate per the spec; judge MARGINALs with eyes on the numbers)")
    report["verdicts"] = verdicts
    json.dump(report, open("redset_gonogo_report.json", "w"), indent=1, default=str)
    print("[out] redset_gonogo_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
