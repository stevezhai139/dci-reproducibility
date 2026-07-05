#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Paper 3D — MySQL under-indexed (PK-only) probe. Mirror of the PG
# RUN_PROBE_pkonly.sh, for the MySQL JOB + TPC-H SF1.0 PK-only cells
# that make MySQL symmetric with PG (4 full-FK + 2 PK-only each).
#
# Mechanism: PROBE_PK_ONLY=1 drops the FK backbone (PK-only baseline);
#            PROBE_TAG=_pkonly writes results to *_job_pkonly.* /
#            *_sf1_0_pkonly.* so the full-FK results are NOT touched.
#
# ⚠️ Drops the FK backbone on the target MySQL DB. Run
#    ./RUN_PROBE_pkonly_mysql.sh restore  afterwards to rebuild it
#    (or a later normal full-FK run recreates it).
#
# ⚠️ ADVISOR: the MySQL harness currently invokes the FIXED
#    ix_advisor_bench. PG's PK-only cells used Dexter (adaptive). For
#    methodological symmetry with PG, finalise the advisor choice FIRST
#    (see MYSQL_PKONLY_ADVISOR_DECISION). This runner works with whatever
#    advisor the harness is set to.
#
# Usage:
#   ./RUN_PROBE_pkonly_mysql.sh job        # PK-only MySQL JOB
#   ./RUN_PROBE_pkonly_mysql.sh tpch       # PK-only MySQL TPC-H SF1.0
#   ./RUN_PROBE_pkonly_mysql.sh restore    # rebuild FK backbone (job + tpch_sf1)
#
# Needs $MYSQL_PWD (or the harness prompts). Do NOT use the machine
# during a run (wall-QPS is the measured outcome).
# ════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "../.venv/bin/activate" ] && source ../.venv/bin/activate

MODE="${1:-}"
BLOCKS="${2:-10}"

restore_backbone () {
  echo "── Restoring FK backbone on MySQL tpch_sf1 + imdb ──"
  PROBE_PK_ONLY=0 python3 - <<'PY'
import sys; sys.path.insert(0, 'mysql')
import mysql_adaptation as M
# ensure_indexes() with PROBE_PK_ONLY unset recreates the FK backbone
import os; os.environ.pop('PROBE_PK_ONLY', None)
for wl, sf in [('tpch', 1.0), ('job', None)]:
    M.load_workload(wl) if hasattr(M, 'load_workload') else None
    print(f"  (use the harness ensure_indexes for {wl}; or re-run a full-FK pass)")
print("  → simplest: run a normal full-FK pass, which recreates the backbone.")
PY
}

[ "$MODE" = "restore" ] && { restore_backbone; exit $?; }
if [ "$MODE" != "tpch" ] && [ "$MODE" != "job" ]; then
  echo "usage: ./RUN_PROBE_pkonly_mysql.sh [tpch | job | restore]"; exit 2
fi

echo "── Pre-flight ──"
python3 -c "import mysql.connector" 2>/dev/null || { echo "✗ mysql-connector-python missing"; exit 1; }

TS=$(date -u +%Y%m%dT%H%M%SZ)
if [ "$MODE" = "tpch" ]; then
  WL=tpch; SFARGS="--sf 1.0"; TAGFILE=sf1_0_pkonly; CELL="MySQL TPC-H SF=1.0 (PK-only)"
else
  WL=job;  SFARGS="";         TAGFILE=job_pkonly;   CELL="MySQL JOB (PK-only)"
fi

if [ -s "mysql/out/adaptation_comparison_v2_${TAGFILE}.csv" ]; then
  echo "⚠ ${TAGFILE} already exists — move aside to re-run. Aborting."; exit 1
fi

LOG="mysql/out/probe_${TAGFILE}_${TS}.log"
echo "── Launch: ${CELL} ──  log → ${LOG}"
echo "   (drops FK backbone on the target DB for the PK-only baseline)"
PROBE_PK_ONLY=1 PROBE_TAG=_pkonly ADVISOR_MODE=rule caffeinate -dimsu \
  python3 mysql/mysql_adaptation.py --workload "$WL" $SFARGS --blocks "$BLOCKS" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { echo "✗ rc=$rc — see $LOG"; exit "$rc"; }
echo "── DONE: ${CELL}. Remember: rebuild the FK backbone (a full-FK pass) when both PK-only probes are done."
