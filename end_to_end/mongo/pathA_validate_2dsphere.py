#!/usr/bin/env python3
"""
pathA_validate_2dsphere.py (v2, fast) — prove 2dsphere is the right, beneficial
index for the rewritten geospatial Q10. Builds the index ONCE (not per seed),
streams output, then drops it so --baseline under starts clean.
Run AFTER pathA_geo_migrate.py --apply.
"""
import sys, os, time, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cross_engine", "mongo", "workload"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cross_engine", "common"))
from pymongo import MongoClient
from templates import ALL_TEMPLATES
from param_sampler import materialize_pipeline

DB, COLL, IX = "mydb_p3a", "combined_clean", "_pathA_test_2dsphere"
def out(*a): print(*a, flush=True)

def timed(coll, pl, hint=None):
    t0 = time.perf_counter()
    res = list(coll.aggregate(pl, allowDiskUse=True, **({"hint": hint} if hint else {}))) if hint \
          else list(coll.aggregate(pl, allowDiskUse=True))
    return (time.perf_counter()-t0)*1000, (res[0]["n"] if res else 0)

def plan_of(coll, pl):
    try:
        s = str(coll.database.command("aggregate", COLL, pipeline=pl, explain=True))
        for t in ("IXSCAN","GEO_NEAR_2DSPHERE","FETCH","COLLSCAN"):
            if t in s: return t
        return "?"
    except Exception as e:
        return f"err:{str(e)[:30]}"

def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else "mongodb://localhost:27017"
    coll = MongoClient(uri, serverSelectionTimeoutMS=5000)[DB][COLL]
    out("connecting...")
    ng = coll.count_documents({"label":"thailand_osm","geometry":{"$exists":True}})
    out(f"thailand_osm docs with geometry: {ng:,}")
    if ng == 0:
        out("!! run pathA_geo_migrate.py --apply first."); sys.exit(1)
    if IX in coll.index_information(): coll.drop_index(IX)

    seeds = (11, 37, 59, 99)
    pls = {s: materialize_pipeline(ALL_TEMPLATES["Q10"], random.Random(s)) for s in seeds}

    out("\n[phase 1] baseline (no geo index, uses bb_label + geo filter)")
    base = {}
    for s in seeds:
        ms, n = timed(coll, pls[s]); base[s] = (ms, n)
        out(f"   seed {s:>3}: {ms:>9.1f} ms   matched {n:,}")

    out("\n[phase 2] building (label, geometry:2dsphere) once ...")
    t0 = time.perf_counter()
    coll.create_index([("label",1),("geometry","2dsphere")], name=IX)
    out(f"   build took {(time.perf_counter()-t0):.1f}s")
    out("[phase 2] with 2dsphere:")
    sp = []
    for s in seeds:
        plan = plan_of(coll, pls[s])
        ms, n = timed(coll, pls[s])
        b = base[s][0]; speed = b/ms if ms>0 else float("inf"); sp.append(speed)
        flag = "" if n==base[s][1] else "  !!COUNT MISMATCH"
        out(f"   seed {s:>3}: {ms:>9.1f} ms   {speed:>6.2f}x   plan={plan}{flag}")

    coll.drop_index(IX)
    out(f"\nmedian speedup: {st.median(sp):.2f}x")
    out("verdict:", "2dsphere IS the right, beneficial index ✓" if st.median(sp)>1.3
        else "weak — check selectivity")
    assert IX not in coll.index_information()
    out("cleanup: test index dropped — collection clean for the run.")

if __name__ == "__main__":
    main()
