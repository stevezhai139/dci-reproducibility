#!/bin/bash
# Paper 3D — EXPLAIN gate-condition test for JOB (before committing 10-block probe).
# Tests Steve's 4 conditions for an advisor-valuable regime on imdb:
#   (1) no usable index  (2) out-of-cache  (3) plan switches to index  (4) faster.
# Procedure: PK-only baseline -> EXPLAIN ANALYZE a cast_info-join query
# WITHOUT then WITH ix_advisor_bench; compares plan + actual time.
# Leaves DB PK-only (advisor index dropped). Run RUN_PROBE_pkonly.sh restore to revert.
set -uo pipefail
cd "$(dirname "$0")"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "../.venv/bin/activate" ] && source ../.venv/bin/activate
python3 - <<'PY'
import sys; sys.path.insert(0,'postgres')
import psycopg2, pg_workloads
cfg = pg_workloads.WORKLOAD_CONFIGS['job']
DB='imdb'
# Representative cast_info-join query (JOB 6a): filters a person, joins cast_info on person_id -> movie_id
Q = ("SELECT MIN(k.keyword), MIN(n.name), MIN(t.title) "
     "FROM cast_info AS ci, keyword AS k, movie_keyword AS mk, name AS n, title AS t "
     "WHERE k.keyword='marvel-cinematic-universe' AND n.name LIKE '%Downey%Robert%' "
     "AND t.production_year>2010 AND k.id=mk.keyword_id AND t.id=mk.movie_id "
     "AND t.id=ci.movie_id AND ci.movie_id=mk.movie_id AND n.id=ci.person_id")
def conn():
    c=psycopg2.connect(host='localhost',port=5432,user=__import__('os').environ.get('PGUSER','postgres'),password='',dbname=DB)
    c.autocommit=True; return c
def drop_backbone(cur):
    for ddl in cfg['backbone_indexes']:
        t=ddl.split(); name=t[t.index('EXISTS')+1] if 'EXISTS' in t else t[2]
        cur.execute(f'DROP INDEX IF EXISTS {name}')
def explain(cur, label):
    cur.execute("SET max_parallel_workers_per_gather=2")
    cur.execute("DISCARD ALL") if False else None
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, COSTS) "+Q)
    rows=[r[0] for r in cur.fetchall()]
    plan="\n".join(rows)
    # extract exec time + whether cast_info uses index
    import re
    t=re.search(r'Execution Time: ([\d.]+) ms', plan)
    exec_ms=float(t.group(1)) if t else None
    uses_idx = 'ix_advisor_bench' in plan
    ci_scan = [l.strip() for l in rows if 'cast_info' in l]
    print(f"\n===== {label} =====")
    print(f"  Execution Time: {exec_ms} ms")
    print(f"  uses ix_advisor_bench: {uses_idx}")
    for l in ci_scan: print(f"  cast_info access: {l}")
    return exec_ms, uses_idx

c=conn(); cur=c.cursor()
print("Dropping FK backbone (PK-only baseline)..."); drop_backbone(cur); cur.execute("ANALYZE cast_info")
cur.execute("DROP INDEX IF EXISTS ix_advisor_bench")
# cold-ish: can't truly flush PG cache without restart; note this caveat
t0,_ = explain(cur, "WITHOUT advisor index (PK-only)")
print("\nCreating ix_advisor_bench ON cast_info(person_id, movie_id)...")
cur.execute(cfg['advisor_create_ddl']); cur.execute("ANALYZE cast_info")
t1,used = explain(cur, "WITH advisor index (PK-only)")
cur.execute("DROP INDEX IF EXISTS ix_advisor_bench")
print("\n================= VERDICT =================")
if t0 and t1:
    sp = t0/t1 if t1>0 else 0
    print(f"  without: {t0:.1f} ms   with: {t1:.1f} ms   speedup: {sp:.2f}x   planner used index: {used}")
    if used and sp>1.2:
        print("  ✅ advisor-valuable regime PLAUSIBLE on JOB → 10-block probe worth running")
    elif used and sp<=1.2:
        print("  ⚠ index used but speedup small → marginal; probe may show weak/no contrast")
    else:
        print("  ❌ planner did NOT use the index → JOB likely neutral like TPC-H (don't waste 8h)")
print("\n  NOTE: PG cache not flushed (condition 2 'out-of-cache' only partially tested).")
print("  DB left in PK-only state. Run ./RUN_PROBE_pkonly.sh restore to revert, or")
print("  ./RUN_PROBE_pkonly.sh job to proceed with the full probe.")
cur.close(); c.close()
PY
