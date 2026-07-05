"""
esr_recommender.py — workload-driven MongoDB index recommender following the
official ESR rule (Equality, Sort, Range). Paper 3D Mongo arm (Addendum 21).

This is the REAL recommender (the MongoDB analogue of Dexter for Postgres):
it derives index recommendations from the *query shapes* seen in a window, NOT
from the templates' declared `candidate_index` (that is used only as a VALIDATION
yardstick). Rule-based and citable (MongoDB index-design guidance), not oracle.

Per query ($match [+ $sort]):
  classify predicate fields ->
     equality : field:value | {$eq} | {$in}            (ESR: Equality)
     range    : {$gt|$gte|$lt|$lte} | {$regex}         (ESR: Range)
     text     : {$text:{$search:..}}                   -> text index
     geo      : {$near|$geoWithin|$geoIntersects}      -> 2dsphere
  sort fields from $sort                               (ESR: Sort)
  key = [equality (discriminator first)] + [sort] + [range]   (ESR order)
  TYPE: text -> {disc:1, <textfield>:"text"};
        geo  -> {disc:1, <geofield>:"2dsphere"};
        else -> compound btree on the ESR key.
Aggregate over the window: dedup, drop prefix-subsumed specs.

$text does not name its field, so we consult light SCHEMA metadata
(text/geo fields per discriminator value) — legitimate schema knowledge a
DBA/advisor has, not the answer key.
"""
from __future__ import annotations

DISCRIMINATORS = ("type", "label")            # equality fields that lead every key
RANGE_OPS = {"$gt", "$gte", "$lt", "$lte"}
GEO_OPS = {"$near", "$nearSphere", "$geoWithin", "$geoIntersects"}

# schema metadata: which field carries searchable text / geometry, keyed by the
# discriminator VALUE present in the query (e.g. label == "arxiv").
TEXT_FIELDS = {"arxiv": "abstract"}
GEO_FIELDS = {"thailand_osm": "geometry"}      # only used if a $geo op appears


def _match_stage(pipeline):
    for st in pipeline:
        if "$match" in st:
            return st["$match"]
    return {}


def _sort_stage(pipeline):
    for st in pipeline:
        if "$sort" in st:
            return list(st["$sort"].items())   # [(field, dir), ...]
    return []

def _group_output_aliases(pipeline):
    """Computed $group output field names (everything in $group except _id).
    These are NOT document fields and must never enter an index key."""
    for st in pipeline:
        if "$group" in st:
            return {k for k in st["$group"].keys() if k != "_id"}
    return set()


def _group_id_fields(pipeline):
    """Real document fields used as the $group _id key (ESR group/equality).
    Handles  _id: "$field"  and  _id: {alias: "$field", ...}.
    Computed group OUTPUT fields (e.g. {n: {$sum: 1}}) are never indexable and
    are intentionally ignored. Addendum 21 / TOMORROW step 1."""
    for st in pipeline:
        if "$group" in st:
            gid = st["$group"].get("_id")
            out = []
            if isinstance(gid, str) and gid.startswith("$"):
                out.append(gid[1:])
            elif isinstance(gid, dict):
                for v in gid.values():
                    if isinstance(v, str) and v.startswith("$"):
                        out.append(v[1:])
            return out
    return []


def recommend_for_query(pipeline):
    """Return one index spec (list of (field, kind)) for a single query, or None.
    kind is 1 (asc btree), -1 (desc btree), 'text', or '2dsphere'."""
    match = _match_stage(pipeline)
    _computed = _group_output_aliases(pipeline)
    eq, rng = [], []
    sort = [(f, d) for f, d in _sort_stage(pipeline) if f not in _computed]
    disc_val, text, geo_field = None, False, None

    for field, cond in match.items():
        if field == "$text":
            text = True
            continue
        if isinstance(cond, dict):
            ops = set(cond.keys())
            if ops & GEO_OPS:
                geo_field = field
            elif "$in" in ops or "$eq" in ops:
                eq.append(field)
            elif ops & RANGE_OPS or "$regex" in ops:
                rng.append(field)
            else:
                rng.append(field)              # conservative default
        else:
            eq.append(field)
            if field in DISCRIMINATORS:
                disc_val = cond

    # discriminator(s) lead the key (ESR equality), in DISCRIMINATORS order
    disc = [f for f in DISCRIMINATORS if f in eq]
    eq_rest = [f for f in eq if f not in DISCRIMINATORS]

    if text:
        tf = TEXT_FIELDS.get(disc_val)
        key = [(f, 1) for f in (disc or eq[:1])]
        key.append((tf or "_text", "text"))
        return key
    if geo_field:
        gf = GEO_FIELDS.get(disc_val, geo_field)
        key = [(f, 1) for f in (disc or eq[:1])]
        key.append((gf, "2dsphere"))
        return key

    # ESR: Equality (disc first, then other eq) -> Group keys -> Sort -> Range
    seen, key = set(), []
    for f in disc + eq_rest:
        if f not in seen:
            key.append((f, 1)); seen.add(f)
    # $group _id fields are grouping keys (equality-like), but they are NOT
    # equality PREDICATES: never let them displace a selective range filter.
    # Inject only when the match already leads with equality, or there is no
    # range field at all (pure group-by / equality+group). ESR-faithful;
    # fixes the Q24 range-only regression while covering Q22/Q23.
    if eq or not rng:
        for f in _group_id_fields(pipeline):
            if f not in seen:
                key.append((f, 1)); seen.add(f)
    for f, d in sort:
        if f not in seen:
            key.append((f, d)); seen.add(f)
    for f in rng:
        if f not in seen:
            key.append((f, 1)); seen.add(f)
    return key or None


def _subsumed(a, b):
    """True if key `a` is a prefix of key `b` (so `a` is redundant given `b`)."""
    return len(a) < len(b) and b[:len(a)] == a


def recommend(pipelines):
    """Workload recommender: given the window's query pipelines, return a deduped,
    prefix-pruned list of index specs. Each spec is a list of (field, kind)."""
    specs = []
    for p in pipelines:
        k = recommend_for_query(p)
        if k and k not in specs:
            specs.append(k)
    # drop specs that are a prefix of a longer recommended spec
    pruned = [a for a in specs if not any(a != b and _subsumed(a, b) for b in specs)]
    return pruned


def spec_name(key):
    parts = []
    for f, kind in key:
        suf = {1: "", -1: "_d", "text": "_txt", "2dsphere": "_geo"}.get(kind, f"_{kind}")
        parts.append(f.replace(".", "_") + suf)
    return ("ix_esr_" + "_".join(parts))[:120]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "..", "cross_engine", "mongo", "workload"))
    import templates as T
    print("=== ESR recommendation vs declared candidate_index (validation) ===")
    ok = 0
    for qid, tpl in sorted(T.ALL_TEMPLATES.items(), key=lambda kv: int(kv[0][1:])):
        rec = recommend_for_query(tpl.pipeline)
        rec_fields = tuple(f for f, _ in (rec or []))
        cand_fields = tuple(f for f, _ in tpl.candidate_index)
        match = "OK " if rec_fields == cand_fields else "DIFF"
        if rec_fields == cand_fields:
            ok += 1
        print(f"  {qid:4s} {tpl.shape:9s} {match}  ESR={rec}  cand={list(tpl.candidate_index)}")
    print(f"\nfield-order exact match: {ok}/{len(T.ALL_TEMPLATES)}")

    print("\n=== window-level recommend() per phase (first 6 qids of each mix) ===")
    for name, ph in T.ALL_PHASES.items():
        qids = list(ph["mix"].keys())
        pls = [T.ALL_TEMPLATES[q].pipeline for q in qids]
        recs = recommend(pls)
        print(f"\n  [{name}] qids={qids}")
        for k in recs:
            print(f"     {spec_name(k)}  {k}")
