#!/usr/bin/env python3
"""
pathA_geo_migrate.py — PATH A step 1: turn the Mongo "geo" modality into a
REAL geospatial workload.

thailand_osm docs currently carry only scalar (lat, lon) — point coordinates.
Q10 was written as a scalar lat-range AND lon-range query, which MongoDB has
NO good index for (btree uses only the first range; the second becomes a
fetch-filter -> toxic at low selectivity). The correct MongoDB form is a
GeoJSON `geometry` Point + a $geoWithin query served by a 2dsphere index.

This script:
  1. INSPECTS thailand_osm: sample doc, lat/lon type, empirical min/max/count,
     and how many docs have out-of-range / non-numeric / missing coordinates.
  2. (only with --apply) adds  geometry:{type:"Point",coordinates:[lon,lat]}
     to every thailand_osm doc whose lat/lon are valid and that does not yet
     have a geometry field.  Idempotent.  Does NOT build any index (the
     advisor builds the 2dsphere index during the experiment; the under-
     provisioned baseline must start with no advisor index).

Run inspection first (safe, read-only):
    python3 pathA_geo_migrate.py
Then, after reviewing the bounds it prints:
    python3 pathA_geo_migrate.py --apply
"""
import sys, argparse
from pymongo import MongoClient

DB, COLL, LABEL = "mydb_p3a", "combined_clean", "thailand_osm"
LAT_OK = (-90.0, 90.0)
LON_OK = (-180.0, 180.0)


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="mongodb://localhost:27017")
    ap.add_argument("--apply", action="store_true",
                    help="actually add the geometry field (default: inspect only)")
    a = ap.parse_args()
    coll = MongoClient(a.uri, serverSelectionTimeoutMS=5000)[DB][COLL]

    n = coll.count_documents({"label": LABEL})
    print(f"=== {DB}.{COLL}  label={LABEL} : {n:,} docs ===")
    if n == 0:
        print("!! no thailand_osm docs — wrong db/coll? aborting."); sys.exit(1)

    sample = coll.find_one({"label": LABEL})
    print("\n-- sample doc (keys) --")
    for k, v in sample.items():
        print(f"   {k:14s} = {repr(v)[:70]}  ({type(v).__name__})")

    have_geom = coll.count_documents({"label": LABEL, "geometry": {"$exists": True}})
    lat_num = coll.count_documents({"label": LABEL, "lat": {"$type": ["double", "int", "long"]}})
    lon_num = coll.count_documents({"label": LABEL, "lon": {"$type": ["double", "int", "long"]}})
    print(f"\n-- coordinate health --")
    print(f"   lat numeric : {lat_num:,}/{n:,}")
    print(f"   lon numeric : {lon_num:,}/{n:,}")
    print(f"   already have geometry : {have_geom:,}")

    # empirical bounds (numeric only)
    pipe = [{"$match": {"label": LABEL, "lat": {"$type": ["double", "int", "long"]},
                        "lon": {"$type": ["double", "int", "long"]}}},
            {"$group": {"_id": None,
                        "lat_min": {"$min": "$lat"}, "lat_max": {"$max": "$lat"},
                        "lon_min": {"$min": "$lon"}, "lon_max": {"$max": "$lon"},
                        "c": {"$sum": 1}}}]
    g = list(coll.aggregate(pipe))
    if g:
        d = g[0]
        print(f"\n-- empirical bounds (n={d['c']:,}) --")
        print(f"   lat : [{d['lat_min']}, {d['lat_max']}]")
        print(f"   lon : [{d['lon_min']}, {d['lon_max']}]")
        print(f"   >>> COPY THESE BOUNDS BACK so the box sampler is calibrated <<<")
        # out-of-GeoJSON-range count
        bad = coll.count_documents({"label": LABEL, "$or": [
            {"lat": {"$lt": LAT_OK[0]}}, {"lat": {"$gt": LAT_OK[1]}},
            {"lon": {"$lt": LON_OK[0]}}, {"lon": {"$gt": LON_OK[1]}}]})
        print(f"   out-of-GeoJSON-range docs (skipped on migrate): {bad:,}")

    if not a.apply:
        print("\n(inspection only — re-run with --apply to add the geometry field)")
        return

    print("\n=== APPLYING geometry migration ===")
    flt = {"label": LABEL,
           "geometry": {"$exists": False},
           "lat": {"$type": ["double", "int", "long"], "$gte": LAT_OK[0], "$lte": LAT_OK[1]},
           "lon": {"$type": ["double", "int", "long"], "$gte": LON_OK[0], "$lte": LON_OK[1]}}
    todo = coll.count_documents(flt)
    print(f"   docs to update : {todo:,}")
    # GeoJSON order is [lon, lat]
    res = coll.update_many(flt, [{"$set": {"geometry": {
        "type": "Point", "coordinates": ["$lon", "$lat"]}}}])
    print(f"   modified       : {res.modified_count:,}")
    after = coll.find_one({"label": LABEL, "geometry": {"$exists": True}})
    print(f"   verify sample geometry : {after.get('geometry')}")
    print("   done. (no index built — advisor builds the 2dsphere during the run)")


if __name__ == "__main__":
    main()
