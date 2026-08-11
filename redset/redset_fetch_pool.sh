#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  redset_fetch_pool.sh — fetch + prepare the Redset cluster pool
#  (Paper 3C splice study; pool = redset_select_clusters.py output,
#   criteria c1-c3, run of 2026-08-11 on provisioned/sample_0.01)
#
#  Idempotent: existing files are skipped. Override the pool with
#      POOL="96 100" ./redset_fetch_pool.sh
#  Data license: Redset (c) 2024 Amazon, CC BY-NC 4.0 — data stays in
#  redset/data/ which is .gitignored; never redistributed in this repo.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"
POOL="${POOL:-96 100 31 77 178 33 3 11 56 16 53 14}"
BASE=https://s3.amazonaws.com/redshift-downloads/redset
mkdir -p data

[ -f data/LICENSE ] || curl -fL -o data/LICENSE "$BASE/LICENSE"
[ -f data/sample_0.01.parquet ] || curl -fL -o data/sample_0.01.parquet "$BASE/provisioned/sample_0.01.parquet"

for id in $POOL; do
  if [ ! -f "data/cluster_${id}.parquet" ]; then
    echo "── fetch cluster $id"
    curl -fL -o "data/cluster_${id}.parquet" "$BASE/provisioned/parts/${id}.parquet"
  fi
done

for id in $POOL; do
  if [ ! -f "data/store_${id}.npz" ]; then
    echo "── build store $id"
    python3 redset_features.py build-store "data/cluster_${id}.parquet" --out "data/store_${id}.npz"
  fi
done

echo "═══ pool ready: $POOL ═══"
