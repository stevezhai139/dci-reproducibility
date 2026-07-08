#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  part2_preflight.sh — Paper 3C S5 Part 2: environment readiness check
#  (read-only; no writes to any database; safe to run repeatedly)
#
#  Verifies everything RUN_PART2.sh needs BEFORE any live compute:
#    Python deps · PostgreSQL(tpch_sf1 + HypoPG) · Dexter · MongoDB
#    (mydb_p3a.combined_clean) · locked-env cost artifact · Part 2 files
#    · the tpch_mixed / --schedule mixed variants load correctly.
#
#  Usage:   ./part2_preflight.sh          (from repro_3c/)
#  Exit 0 = all checks passed.
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"
[ -f config.local.sh ] && source config.local.sh
export PGUSER="${PGUSER:-postgres}"
export DEXTER_BIN="${DEXTER_BIN:-dexter}"

PASS=0; FAIL=0
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

echo "── Python deps ──"
if python3 - <<'PY'
import numpy, scipy, pandas, fastdtw, pywt, pymongo, psycopg2  # noqa
PY
then ok "numpy/scipy/pandas/fastdtw/pywt/pymongo/psycopg2 importable"
else bad "python deps missing (pip install -r requirements.txt + pymongo psycopg2-binary)"; fi

echo "── PostgreSQL (tpch_sf1, user=$PGUSER) ──"
if python3 - <<PY
import psycopg2
c = psycopg2.connect(host="localhost", port=5432, user="$PGUSER", dbname="tpch_sf1",
                     connect_timeout=3)
cur = c.cursor()
cur.execute("SELECT count(*) FROM lineitem")
n = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM pg_extension WHERE extname='hypopg'")
h = cur.fetchone()[0]
print(f"    lineitem rows = {n:,}   hypopg = {h}")
assert n > 0, "lineitem empty"
assert h == 1, "hypopg extension missing (Dexter would be a silent no-op)"
PY
then ok "tpch_sf1 reachable, data loaded, HypoPG installed"
else bad "tpch_sf1 / HypoPG check failed (is postgres running? hypopg created?)"; fi

echo "── Dexter (\$DEXTER_BIN=$DEXTER_BIN) ──"
if command -v "$DEXTER_BIN" >/dev/null 2>&1
then ok "dexter found: $(command -v "$DEXTER_BIN")"
else bad "dexter not on PATH (set DEXTER_BIN in config.local.sh)"; fi

echo "── MongoDB (mydb_p3a.combined_clean) ──"
if python3 - <<'PY'
import pymongo
c = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
c.admin.command("ping")
n = c["mydb_p3a"]["combined_clean"].estimated_document_count()
print(f"    combined_clean docs ≈ {n:,}")
assert n > 0, "collection empty"
PY
then ok "mongod reachable, source collection present"
else bad "mongod / mydb_p3a.combined_clean check failed (is mongod running?)"; fi

echo "── Part 2 artifacts ──"
J="geometry_E0/out/20260705T135856Z/cost_benefit_run.json"
[ -f "$J" ] && ok "locked-env cost artifact: $J" || bad "missing $J"
for f in RUN_PART2.sh part2_validate_offline.py part2_analyze.py; do
  [ -f "$f" ] && ok "$f present" || bad "$f missing"
done
if python3 - <<'PY'
import importlib.util, sys
sys.argv = ["x", "--workload", "tpch_mixed"]; sys.path.insert(0, "end_to_end/postgres")
spec = importlib.util.spec_from_file_location("pga", "end_to_end/postgres/pg_adaptation.py")
m = importlib.util.module_from_spec(spec); sys.modules["pga"] = m; spec.loader.exec_module(m)
assert m.WIN_PER_PH == 4 and len(m.PHASES) == 6
assert [p.get("qpw") for p in m.PHASES] == [20, 20, 32, 32, 20, 20]
sys.path.insert(0, "end_to_end/mongo")
spec = importlib.util.spec_from_file_location("ss", "end_to_end/mongo/season_schedule.py")
s = importlib.util.module_from_spec(spec); sys.modules["ss"] = s; spec.loader.exec_module(s)
pw, cw, bt = s.make_mixed_schedule(24)
assert len(pw) == 24 and sorted(bt) == [4, 8, 12, 16, 20]
assert (min(cw), max(cw)) == (20, 48)
print("    tpch_mixed: 6 phases × 4 win, qpw 20↔32 | mg mixed: onsets 5,9,13,17,21, n 20↔48")
PY
then ok "mixed schedule variants load with the validated geometry"
else bad "mixed variant load failed — re-check patched harness files"; fi

echo "── Engine/driver versions (paper §6 provenance) ──"
python3 - <<PY
import psycopg2, pymongo
c = psycopg2.connect(host="localhost", port=5432, user="$PGUSER", dbname="tpch_sf1", connect_timeout=3)
cur = c.cursor(); cur.execute("SHOW server_version"); print("  PostgreSQL :", cur.fetchone()[0])
cur.execute("SELECT extversion FROM pg_extension WHERE extname='hypopg'")
r = cur.fetchone(); print("  HypoPG     :", r[0] if r else "absent")
print("  psycopg2   :", psycopg2.__version__.split()[0])
try:
    m = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    print("  MongoDB    :", m.server_info()["version"])
except Exception:
    import subprocess
    try:
        v = subprocess.check_output(["mongod", "--version"], text=True).splitlines()[0]
        print("  MongoDB    :", v.replace("db version ", ""), "(from binary; server not running)")
    except Exception:
        print("  MongoDB    : unknown (mongod not on PATH and server not running)")
print("  pymongo    :", pymongo.version)
PY
"$DEXTER_BIN" --version 2>/dev/null | sed 's/^/  Dexter     : /' || echo "  Dexter     : (no --version flag; note gem version manually)"
python3 -c "import platform; print('  Python     :', platform.python_version())"

echo
echo "═══ preflight: $PASS passed, $FAIL failed ═══"
[ "$FAIL" -eq 0 ] && echo "READY → next: PART2_BLOCKS=1 ./RUN_PART2.sh (smoke)" || echo "fix the ✗ items before RUN_PART2.sh"
exit "$FAIL"
