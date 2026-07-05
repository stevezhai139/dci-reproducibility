#!/bin/bash
# Paper 3D — direct cache/stat instrumentation (3C-style plan analysis).
# Answers: (a) does cache warm across repeated exec? (shared hit vs read)
#          (b) does warm NO-INDEX approach WITH-INDEX? (Steve's erosion worry)
#          (c) how much of the 5x is STRUCTURAL vs cache?
#          (d) stat quality: planner estimated rows vs actual rows on cast_info.
# Leaves DB PK-only (advisor index dropped). ~2-4 min.
set -uo pipefail
cd "$(dirname "$0")"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "../.venv/bin/activate" ] && source ../.venv/bin/activate
N="${1:-12}"   # repetitions per condition
python3 - "$N" <<'PY'
import sys, re; sys.path.insert(0,'postgres')
import psycopg2, pg_workloads
N=int(sys.argv[1]); cfg=pg_workloads.WORKLOAD_CONFIGS['job']
Q=("SELECT MIN(k.keyword), MIN(n.name), MIN(t.title) "
   "FROM cast_info AS ci, keyword AS k, movie_keyword AS mk, name AS n, title AS t "
   "WHERE k.keyword='marvel-cinematic-universe' AND n.name LIKE '%Downey%Robert%' "
   "AND t.production_year>2010 AND k.id=mk.keyword_id AND t.id=mk.movie_id "
   "AND t.id=ci.movie_id AND ci.movie_id=mk.movie_id AND n.id=ci.person_id")
c=psycopg2.connect(host='localhost',port=5432,user=__import__('os').environ.get('PGUSER','postgres'),password='',dbname='imdb'); c.autocommit=True; cur=c.cursor()
for ddl in cfg['backbone_indexes']:
    t=ddl.split(); nm=t[t.index('EXISTS')+1] if 'EXISTS' in t else t[2]; cur.execute(f'DROP INDEX IF EXISTS {nm}')
cur.execute('DROP INDEX IF EXISTS ix_advisor_bench'); cur.execute('ANALYZE cast_info')
def run_once():
    cur.execute('EXPLAIN (ANALYZE, BUFFERS, COSTS) '+Q); txt='\n'.join(r[0] for r in cur.fetchall())
    ms=float(re.search(r'Execution Time: ([\d.]+)',txt).group(1))
    hit=sum(int(x) for x in re.findall(r'shared hit=(\d+)',txt))
    rd =sum(int(x) for x in re.findall(r'shared.*?read=(\d+)',txt))
    # cast_info node: estimated vs actual rows
    m=re.search(r'cast_info ci\s+\(cost=[^)]*rows=(\d+)[^)]*\)\s+\(actual[^)]*rows=(\d+)',txt)
    est,act=(int(m.group(1)),int(m.group(2))) if m else (None,None)
    return ms,hit,rd,est,act
def phase(label):
    print(f"\n===== {label} ({N} reps) =====")
    print(f"  {'rep':>3} {'exec_ms':>9} {'shared_hit':>11} {'shared_read':>12} {'hit_ratio':>9}  est/act rows")
    first=last=None
    for i in range(1,N+1):
        ms,hit,rd,est,act=run_once()
        ratio=hit/(hit+rd) if (hit+rd) else 1.0
        er=f"{est}/{act}" if est is not None else "n/a"
        print(f"  {i:>3} {ms:>9.1f} {hit:>11} {rd:>12} {ratio:>9.3f}  {er}")
        if i==1: first=ms
        last=ms
    return first,last
nf,nl=phase("NO INDEX (PK-only)")
print("\nCreating ix_advisor_bench..."); cur.execute(cfg['advisor_create_ddl']); cur.execute('ANALYZE cast_info')
wf,wl=phase("WITH INDEX (PK-only)")
cur.execute('DROP INDEX IF EXISTS ix_advisor_bench')
print("\n================= INTERPRETATION =================")
print(f"  NO-INDEX: cold={nf:.1f}ms  warm={nl:.1f}ms  (self-warm {nf/nl:.2f}x)")
print(f"  WITH-INDEX: cold={wf:.1f}ms  warm={wl:.1f}ms")
print(f"  WARM gap (no-index warm / with-index warm): {nl/wl:.2f}x  <- structural floor that cache CANNOT close")
if nl/wl>1.5: print("  => structural advantage PERSISTS warm: missing index can't be cached away. ✅ thesis-safe")
else:         print("  => warm cache LARGELY erodes the gap. ⚠ Steve's erosion worry confirmed for this query")
print("  (NOTE: PG cannot truly flush cache w/o restart; 'cold' = first touch this session.)")
print("  DB left PK-only. Run ./RUN_PROBE_pkonly.sh restore to revert.")
cur.close(); c.close()
PY
