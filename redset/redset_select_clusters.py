#!/usr/bin/env python3
"""redset_select_clusters.py -- rank Redset clusters for the splice-study pool.

Criteria (from validation lessons, clusters 96/100/79):
  c1  span >= 56 days
  c2  fingerprint top-50 coverage >= 40%   (S_R/S_T viability; go/no-go T1)
  c3  no-fingerprint drop rate <= 10%      (cluster-79 lesson: 41.9% dropped)
Ranked by kept-volume. Prints the qualified pool + download commands.

Usage: python3 redset_select_clusters.py data/sample_0.01.parquet [--top 12]
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-span", type=int, default=56)
    ap.add_argument("--min-top50", type=float, default=40.0)
    ap.add_argument("--max-nofp", type=float, default=10.0)
    a = ap.parse_args()
    df = pd.read_parquet(a.parquet, columns=["instance_id", "arrival_timestamp",
                                             "feature_fingerprint", "was_aborted"])
    df = df[df["was_aborted"] != True]  # noqa: E712
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    rows = []
    for cid, sub in df.groupby("instance_id"):
        n = len(sub)
        nofp = float(sub["feature_fingerprint"].isna().mean() * 100)
        fp = sub["feature_fingerprint"].dropna()
        span = (sub["arrival_timestamp"].max() - sub["arrival_timestamp"].min()).days
        if len(fp) == 0:
            continue
        vc = fp.value_counts()
        top50 = float(vc.head(50).sum() / len(fp) * 100)
        rows.append((cid, n, span, top50, nofp))
    t = pd.DataFrame(rows, columns=["cluster", "n_sample", "span_d", "top50_pct", "nofp_pct"])
    q = t[(t.span_d >= a.min_span) & (t.top50_pct >= a.min_top50) & (t.nofp_pct <= a.max_nofp)]
    q = q.sort_values("n_sample", ascending=False).head(a.top)
    print(f"[pool] eligible {len(t)} clusters; qualified (span>={a.min_span}d, "
          f"top50>={a.min_top50}%, nofp<={a.max_nofp}%): {len(q)} shown top {a.top}\n")
    print(q.to_string(index=False, formatters={
        "n_sample": "{:,}".format, "top50_pct": "{:.1f}".format, "nofp_pct": "{:.2f}".format}))
    ids = " ".join(str(c) for c in q.cluster)
    print(f"\n# download:\nfor id in {ids}; do curl -L -o data/cluster_$id.parquet "
          f"https://s3.amazonaws.com/redshift-downloads/redset/provisioned/parts/$id.parquet; done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
