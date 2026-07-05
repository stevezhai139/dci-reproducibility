#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Paper 3D — Under-indexed (PK-only) probe. Tests whether an
# advisor-VALUABLE regime exists when the FK backbone is removed
# (answers the "just never call the advisor" reviewer objection).
#
# Cells: PG TPC-H SF=1.0 (tpch_sf1) and PG JOB (imdb).
# Mechanism: PROBE_PK_ONLY=1 drops the FK backbone (PK-only baseline);
#            PROBE_TAG=_pkonly writes results to *_sf1_0_pkonly.* /
#            *_job_pkonly.* so the full-FK results are NOT touched.
#
# ⚠️ This DROPS the FK backbone indexes on tpch_sf1 and imdb.
#    Run  ./RUN_PROBE_pkonly.sh restore   afterwards to rebuild them
#    (or any later normal full-FK run recreates them via CREATE IF NOT EXISTS).
#
# Usage:
#   ./RUN_PROBE_pkonly.sh tpch     # PK-only PG TPC-H SF=1.0
#   ./RUN_PROBE_pkonly.sh job      # PK-only PG JOB
#   ./RUN_PROBE_pkonly.sh restore  # rebuild FK backbone on tpch_sf1 + imdb
#
# Do NOT use the machine during a run (wall-QPS is the measured outcome).
# Do NOT run while the Monday TNI demo needs the machine.
# ════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "../.venv/bin/activate" ] && source ../.venv/bin/activate

MODE="${1:-}"
BLOCKS="${2:-10}"   # optional: ./RUN_PROBE_pkonly.sh job 3
PGUSER_ARG="${PGUSER:-postgres}"

restore_backbone () {
  echo "── Restoring FK backbone on tpch_sf1 + imdb ──"
  python3 - <<'PY'
import sys; sys.path.insert(0, 'postgres')
import psycopg2, pg_workloads
cfg = pg_workloads.WORKLOAD_CONFIGS
for wl, db in [('tpch','tpch_sf1'), ('job','imdb')]:
    conn = psycopg2.connect(host='localhost', port=5432,
        user=__import__('os').environ.get('PGUSER','postgres'), password='', dbname=db)
    conn.autocommit = True; cur = conn.cursor()
    for ddl in cfg[wl]['backbone_indexes']:
        cur.execute(ddl)
    cur.execute('ANALYZE'); cur.close(); conn.close()
    print(f'  ✓ backbone restored on {db}')
PY
}

[ "$MODE" = "restore" ] && { restore_backbone; exit $?; }
if [ "$MODE" != "tpch" ] && [ "$MODE" != "job" ]; then
  echo "usage: ./RUN_PROBE_pkonly.sh [tpch | job | restore]"; exit 2
fi

echo "── Pre-flight ──"
python3 -c "import psycopg2" 2>/dev/null || { echo "✗ psycopg2 missing"; exit 1; }
python3 - <<PY || { echo "✗ cannot connect to PostgreSQL"; exit 1; }
import psycopg2; psycopg2.connect(host="localhost",port=5432,user="$PGUSER_ARG",password="",dbname="postgres").close()
PY
echo "✓ PostgreSQL reachable"

TS=$(date -u +%Y%m%dT%H%M%SZ)
if [ "$MODE" = "tpch" ]; then
  WL=tpch; SFARGS="--sf 1.0"; TAGFILE=sf1_0_pkonly; CELL="PG TPC-H SF=1.0 (PK-only)"
  [ "$BLOCKS" != "10" ] && { TAGFILE="sf1_0_pkonly_b${BLOCKS}"; CELL="$CELL [${BLOCKS}-block preview]"; }
else
  WL=job;  SFARGS="";        TAGFILE=job_pkonly;   CELL="PG JOB (PK-only)"
  [ "$BLOCKS" != "10" ] && { TAGFILE="job_pkonly_b${BLOCKS}"; CELL="$CELL [${BLOCKS}-block preview]"; }
fi

if [ -s "postgres/out/adaptation_comparison_v2_${TAGFILE}.csv" ]; then
  echo "⚠ ${TAGFILE} already exists — move aside to re-run. Aborting."; exit 1
fi

LOG="postgres/out/probe_${TAGFILE}_${TS}.log"
echo "── Launch: ${CELL} ──  log → ${LOG}"
echo "   (drops FK backbone on the target DB for the PK-only baseline)"
PROBE_PK_ONLY=1 PROBE_TAG=_pkonly caffeinate -dimsu \
  python3 postgres/pg_adaptation.py --workload "$WL" $SFARGS --blocks "$BLOCKS" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { echo "✗ rc=$rc — see $LOG"; exit "$rc"; }

if python3 -c "import scipy, statsmodels" 2>/dev/null; then
  python3 paired_comparison_analysis.py \
    --block-metrics-csv "postgres/out/adaptation_comparison_v2_${TAGFILE}.csv" \
    --output-dir        "postgres/out/paired_comparison_${TAGFILE}/" \
    --cell-label        "$CELL"
  python3 divergence_analysis.py \
    --breakdown-csv "postgres/out/breakdown_per_window_v2_${TAGFILE}.csv" \
    --output-dir    "postgres/out/divergence_analysis_${TAGFILE}/"
else
  echo "⚠ scipy/statsmodels missing — run analysis later"
fi
echo "── DONE: ${CELL}. Remember: ./RUN_PROBE_pkonly.sh restore  when both probes are done."
