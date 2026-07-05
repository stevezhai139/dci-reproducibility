#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Paper 3D — T3.7: PostgreSQL RQ5 runs (frozen order steps 2–3,
# FRAMING_v3_DRAFT Addendum 3). Run ONE cell per invocation / per
# night; harness resumes per-SF so each night is a clean checkpoint.
#
# Usage:
#   ./RUN_T37_PG.sh smoke              # 1-block smoke on tpch sf0.2 (ALWAYS run first)
#   ./RUN_T37_PG.sh tpch 0.2           # full 10-block cell, TPC-H SF=0.2
#   ./RUN_T37_PG.sh tpch 1.0
#   ./RUN_T37_PG.sh tpch 3.0
#   ./RUN_T37_PG.sh job                # full 10-block cell, JOB on imdb
#
# Do NOT use the machine during a run: wall-QPS is the measured outcome.
# ════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"

# Paper 3D venv (psycopg2 lives here, not system python)
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "../.venv/bin/activate" ]; then
  source ../.venv/bin/activate
  echo "✓ activated Paper 3D venv ($(python3 --version))"
fi

MODE="${1:-}"; ARG="${2:-}"
if [ -z "$MODE" ]; then
  echo "usage: ./RUN_T37_PG.sh [smoke | tpch <0.2|1.0|3.0> | job]"; exit 2
fi

echo "── Pre-flight ──────────────────────────────────────────────"

# 1. psycopg2 present
python3 -c "import psycopg2" 2>/dev/null \
  || { echo "✗ psycopg2 missing (pip install psycopg2-binary into the venv)"; exit 1; }
echo "✓ python3 + psycopg2"

# 2. PG server reachable + user OK (connects to 'postgres' maintenance db)
python3 - <<'PY' || { echo "✗ cannot connect to PostgreSQL — is the server running?"; exit 1; }
import psycopg2
psycopg2.connect(host="localhost", port=5432,
                 user=__import__('os').environ.get('PGUSER','postgres'), password="", dbname="postgres").close()
PY
echo "✓ PostgreSQL reachable (localhost:5432, user ${PGUSER:-postgres})"

# 3. Disk floor (hard 10 GB; warn < 20)
FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [ "${FREE_GB:-0}" -lt 10 ]; then
  echo "✗ only ${FREE_GB} GB free (<10)"; exit 1
elif [ "${FREE_GB}" -lt 20 ]; then
  echo "⚠ disk: ${FREE_GB} GB free (low but ok)"
else
  echo "✓ disk: ${FREE_GB} GB free"
fi

# 4. scipy/statsmodels for the post-run analysis (warn only)
ANALYSIS_OK=1
python3 -c "import scipy, statsmodels" 2>/dev/null || ANALYSIS_OK=0

TS=$(date -u +%Y%m%dT%H%M%SZ)
run () {  # $1=workload  $2=sf-or-empty  $3=blocks  $4=logtag
  local wl="$1" sf="$2" blocks="$3" tag="$4"
  local log="postgres/out/${tag}_${TS}.log"
  echo "── Launch: $tag (blocks=$blocks) ───────────────────────────"
  echo "log → ${log}"
  caffeinate -dimsu python3 postgres/pg_adaptation.py \
      --workload "$wl" ${sf:+--sf "$sf"} --blocks "$blocks" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

analyze () {  # $1=sf_tag (e.g. sf0_2 / sf1_0 / sf3_0 / job)
  local tag="$1"
  [ "$ANALYSIS_OK" -eq 1 ] || { echo "⚠ scipy/statsmodels missing — run analysis later (pip install scipy statsmodels)"; return; }
  local bm="postgres/out/adaptation_comparison_v2_${tag}.csv"
  local bk="postgres/out/breakdown_per_window_v2_${tag}.csv"
  [ -s "$bm" ] && python3 paired_comparison_analysis.py \
      --block-metrics-csv "$bm" --output-dir "postgres/out/paired_comparison_${tag}/" \
      --cell-label "PostgreSQL ${tag}"
  [ -s "$bk" ] && python3 divergence_analysis.py \
      --breakdown-csv "$bk" --output-dir "postgres/out/divergence_analysis_${tag}/"
}

case "$MODE" in
  smoke)
    echo "Smoke: 1 block, TPC-H SF=0.2 (validation only — output NOT used in paper)"
    run tpch 0.2 1 "smoke_pg_tpch_sf0_2_blocks1"
    rc=$?
    echo "── Smoke rc=$rc. If 0, the harness + DB + advisor path all work."
    echo "   NOTE: this wrote sf0_2 CSVs from a 1-block run. Move them aside"
    echo "   before the real 10-block sf0.2 cell:"
    echo "     mkdir -p postgres/out/_smoke_archive && mv postgres/out/*_v2_sf0_2.* postgres/out/_smoke_archive/ 2>/dev/null"
    exit $rc ;;
  tpch)
    case "$ARG" in 0.2) tag=sf0_2;; 1.0) tag=sf1_0;; 3.0) tag=sf3_0;; *) echo "✗ sf must be 0.2|1.0|3.0"; exit 2;; esac
    if [ -s "postgres/out/adaptation_comparison_v2_${tag}.csv" ]; then
      echo "⚠ ${tag} CSV already exists — harness will resume-skip it. Move aside if re-running. Aborting."; exit 1
    fi
    run tpch "$ARG" 10 "full_pg_tpch_${tag}_blocks10"; rc=$?
    [ "$rc" -eq 0 ] && analyze "$tag"
    exit $rc ;;
  job)
    if [ -s "postgres/out/adaptation_comparison_v2_job.csv" ]; then
      echo "⚠ job CSV already exists — harness will resume-skip it. Move aside if re-running. Aborting."; exit 1
    fi
    run job "" 10 "full_pg_job_blocks10"; rc=$?
    [ "$rc" -eq 0 ] && analyze job
    exit $rc ;;
  *) echo "usage: ./RUN_T37_PG.sh [smoke | tpch <0.2|1.0|3.0> | job]"; exit 2 ;;
esac
