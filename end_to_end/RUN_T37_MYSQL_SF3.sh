#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Paper 3D — T3.7: MySQL × TPC-H SF=3.0 (close out MySQL × TPC-H)
# Created 2026-06-12 (assistant + Steve). Frozen order step 1
# (FRAMING_v3_DRAFT Addendum 3).
#
# Usage:   ./RUN_T37_MYSQL_SF3.sh
#          (run interactively; you will be prompted ONCE for the
#           paper3d MySQL password up-front, or export MYSQL_PWD
#           first. Then you can walk away — caffeinate keeps the
#           Mac awake. Do NOT use the machine during the run:
#           wall-QPS is the measured outcome.)
# ════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"

# 0. Paper 3D venv (code/.venv) — mysql-connector lives here, not in
#    system python (MYSQL_SETUP.md). Auto-activate if not already in a venv.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "../.venv/bin/activate" ]; then
  source ../.venv/bin/activate
  echo "✓ activated Paper 3D venv ($(python3 --version))"
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="mysql/out/full_tpch_sf3_blocks10_${TS}.log"

echo "── Pre-flight ──────────────────────────────────────────────"

# 1. Python + connector
python3 -c "import mysql.connector" 2>/dev/null \
  || { echo "✗ mysql-connector-python missing (pip install mysql-connector-python)"; exit 1; }
echo "✓ python3 + mysql.connector"

# 2. Resume CSVs intact → harness will SKIP sf0.2/sf1.0 (we pass --sf 3.0
#    explicitly anyway, but verify nothing clobbers the finished cells)
for f in mysql/out/adaptation_comparison_v2_sf0_2.csv \
         mysql/out/breakdown_per_window_v2_sf0_2.csv \
         mysql/out/adaptation_comparison_v2_sf1_0.csv \
         mysql/out/breakdown_per_window_v2_sf1_0.csv; do
  [ -s "$f" ] || { echo "✗ missing finished-cell file: $f (check un-archive!)"; exit 1; }
done
echo "✓ sf0.2 / sf1.0 outputs present (will not be touched)"

# 3. SF=3.0 must NOT already have outputs (avoid silent resume-skip)
if [ -s mysql/out/adaptation_comparison_v2_sf3_0.csv ]; then
  echo "⚠ sf3_0 CSV already exists — harness would resume-skip it."
  echo "  If this is a stale partial file, move it aside first. Aborting."
  exit 1
fi
echo "✓ no stale sf3_0 outputs"

# 4. Disk space (run writes only logs+CSVs; MySQL temp is modest.
#    Hard floor 10 GB; warn below 20.)
FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [ "${FREE_GB:-0}" -lt 10 ]; then
  echo "✗ only ${FREE_GB} GB free (<10) — free up space first"; exit 1
elif [ "${FREE_GB}" -lt 20 ]; then
  echo "⚠ disk: ${FREE_GB} GB free (low but sufficient — consider cleanup later)"
else
  echo "✓ disk: ${FREE_GB} GB free"
fi

# 5. tpch_sf3 reachability is verified by the harness itself
#    (ensure_database raises before any block runs).

echo "── Launch (≈3–4 h) ─────────────────────────────────────────"
echo "log → ${LOG}"
caffeinate -dimsu python3 mysql/mysql_adaptation.py \
    --workload tpch --sf 3.0 --blocks 10 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
[ "$RC" -eq 0 ] || { echo "✗ harness exited rc=$RC — see $LOG"; exit "$RC"; }

echo "── Post-run analysis (T3.8 per-cell) ───────────────────────"
if python3 -c "import scipy, statsmodels" 2>/dev/null; then
  python3 paired_comparison_analysis.py \
      --block-metrics-csv mysql/out/adaptation_comparison_v2_sf3_0.csv \
      --output-dir         mysql/out/paired_comparison_sf3_0/ \
      --cell-label         "MySQL × TPC-H SF=3.0"
  python3 divergence_analysis.py \
      --breakdown-csv mysql/out/breakdown_per_window_v2_sf3_0.csv \
      --output-dir    mysql/out/divergence_analysis_sf3_0/
else
  echo "⚠ scipy/statsmodels not installed — run analysis later:"
  echo "  pip install scipy statsmodels"
fi

echo "── DONE. Next: PG pre-flight + smoke (frozen order step 2) ──"
