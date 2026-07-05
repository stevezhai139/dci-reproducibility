#!/bin/bash
# Paper 3D — validate the real Dexter recommender BEFORE the full run.
# For each JOB phase, drops to PK-only and asks Dexter what indexes it
# recommends for that phase's queries. Compare to Addendum 15.6 oracle map.
# Leaves DB PK-only. Run RUN_PROBE_pkonly.sh restore to revert.
set -uo pipefail
cd "$(dirname "$0")"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "../.venv/bin/activate" ] && source ../.venv/bin/activate
DEXTER_BIN="${DEXTER_BIN:-dexter}"

echo "── pre-flight ──"
command -v "$DEXTER_BIN" >/dev/null || { echo "✗ dexter not found (brew install dexter / gem install pgdexter)"; exit 1; }
echo "✓ dexter: $($DEXTER_BIN --version 2>&1 | head -1)"
psql -d imdb -tAc "SELECT 1 FROM pg_extension WHERE extname='hypopg'" | grep -q 1 \
  || { echo "✗ HypoPG not enabled in imdb → run: psql -d imdb -c 'CREATE EXTENSION hypopg;'"; exit 1; }
echo "✓ HypoPG enabled in imdb"

python3 - "$DEXTER_BIN" <<'PY'
import sys, subprocess, tempfile, os, re
sys.path.insert(0,'postgres')
import psycopg2, pg_workloads
DEX=sys.argv[1]; cfg=pg_workloads.WORKLOAD_CONFIGS['job']
exec(open('postgres/job_queries.py').read())  # defines QUERIES
phases=cfg['phases']
# PK-only
c=psycopg2.connect(host='localhost',port=5432,user=__import__('os').environ.get('PGUSER','postgres'),password='',dbname='imdb'); c.autocommit=True; cur=c.cursor()
for ddl in cfg['backbone_indexes']:
    t=ddl.split(); nm=t[t.index('EXISTS')+1] if 'EXISTS' in t else t[2]; cur.execute(f'DROP INDEX IF EXISTS {nm}')
cur.execute("SELECT indexname FROM pg_indexes WHERE indexname LIKE 'ix_dexter_%'")
for (n,) in cur.fetchall(): cur.execute(f'DROP INDEX IF EXISTS {n}')
cur.execute('ANALYZE')
print("PK-only baseline set.\n")
for ph in phases:
    qs=[QUERIES[q] for q in ph['qs']]
    with tempfile.NamedTemporaryFile('w',suffix='.sql',delete=False) as fh:
        for q in qs: fh.write(q.strip().rstrip(';')+';\n')
        path=fh.name
    out=subprocess.run([DEX,'-h','localhost','-p','5432','-U',__import__('os').environ.get('PGUSER','postgres'),
                        '-d','imdb',path,'--min-calls','1'],capture_output=True,text=True,timeout=600)
    os.unlink(path)
    txt=(out.stdout or '')+(out.stderr or '')
    found=re.findall(r'Index found:\s*([^\s(]+)\s*\(([^)]+)\)',txt)
    print(f"=== {ph['name']}  (qs={ph['qs']}) ===")
    if found:
        for rel,cols in found: print(f"   + {rel} ({cols})")
    else:
        print("   (no recommendation)\n   raw:", txt.strip()[:300])
    print()
cur.close(); c.close()
print("DB left PK-only. Compare ↑ to Addendum 15.6. Run ./RUN_PROBE_pkonly.sh restore to revert.")
PY
