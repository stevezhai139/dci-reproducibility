"""
param_sampler.py — Deterministic parameter sampler for workload templates.

Templates in `mongo/workload/templates.py` use `$$PARAM_name` placeholders
inside aggregation pipelines. This module walks a pipeline recursively,
replaces each placeholder with a value drawn from the template's
`params` dict, and returns a concrete executable pipeline.

Design:
  • Pure Python — no mongod contact.
  • Deterministic given a seed (for paired-RCB reproducibility).
  • Placeholder syntax: a dict value that is the string "$$NAME" is
    replaced by `sample(NAME)`. List/dict values are walked recursively.
  • Supported parameter spec syntaxes (from Template.params):
      list of strings      → rng.choice
      "int_range(lo, hi)"  → rng.randint
      "sample(int, k=K, lo, hi)" → K random ints
      "iso_date_range"     → ISO date between 2010 and 2024
      "iso_date"           → single ISO date
      "lat" / "lon"        → realistic lat/lon value for Europe/Asia
      "sample_user_id"     → random uid int
      "sample_asin"        → random ASIN-like string
      "sample_user"        → random user_id string
      "epoch_ms_range"     → random epoch ms since 2015
      "random_label_subset" → random subset of known labels

All samplers accept a `random.Random` instance so the block seed from
paired-RCB fully determines the realized workload.
"""
from __future__ import annotations

import random
import re
from datetime import datetime, timedelta
from typing import Any


# ───────────────────────────────────────────────────────────────────────
# Primitive samplers
# ───────────────────────────────────────────────────────────────────────

_KNOWN_LABELS = [
    "snap_edges", "livejournal_edges", "osm_changeset",
    "arxiv_metadata", "amazon_reviews_patio",
]

_LAT_BOUNDS = (-60.0, 70.0)       # realistic inhabited-area band
_LON_BOUNDS = (-170.0, 180.0)


def _sample_int_range(spec: str, rng: random.Random) -> int:
    """Parse 'int_range(lo, hi)' and return a single int."""
    m = re.match(r"int_range\((\d[\d_]*),\s*(\d[\d_]*)\)", spec)
    if not m:
        raise ValueError(f"unparseable int_range spec: {spec}")
    lo = int(m.group(1).replace("_", ""))
    hi = int(m.group(2).replace("_", ""))
    return rng.randint(lo, hi)


def _sample_int_batch(spec: str, rng: random.Random) -> list[int]:
    """Parse 'sample(int, k=K, lo, hi)' and return a list of K random ints."""
    m = re.match(r"sample\(int,\s*k=(\d+),\s*(\d[\d_]*),\s*(\d[\d_]*)\)", spec)
    if not m:
        raise ValueError(f"unparseable sample(int,...) spec: {spec}")
    k = int(m.group(1))
    lo = int(m.group(2).replace("_", ""))
    hi = int(m.group(3).replace("_", ""))
    return [rng.randint(lo, hi) for _ in range(k)]


def _sample_iso_date(rng: random.Random) -> str:
    start = datetime(2010, 1, 1)
    end = datetime(2024, 12, 31)
    delta = end - start
    off = rng.randint(0, delta.days)
    return (start + timedelta(days=off)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample_lat(rng: random.Random) -> float:
    return round(rng.uniform(*_LAT_BOUNDS), 6)


def _sample_lon(rng: random.Random) -> float:
    return round(rng.uniform(*_LON_BOUNDS), 6)


# Thailand empirical bbox (from mydb_p3a.combined_clean thailand_osm, 850k docs:
# lat[5.5683282,20.619053] lon[97.3380445,105.728935]).  PATH A: Q10 is a real
# geospatial $geoWithin query; we sample a sub-box (0.5deg-4deg per side) wholly
# inside Thailand so selectivity is meaningful (not the old global all-or-nothing
# box).  Returns a GeoJSON Polygon ring ([lon,lat], closed) for $geoWithin.
_THAI_LAT = (5.5683282, 20.619053)
_THAI_LON = (97.3380445, 105.728935)


def _sample_thai_box(rng: random.Random) -> dict:
    h_lat = rng.uniform(0.5, 4.0)
    h_lon = rng.uniform(0.5, 4.0)
    lat_lo = rng.uniform(_THAI_LAT[0], _THAI_LAT[1] - h_lat)
    lon_lo = rng.uniform(_THAI_LON[0], _THAI_LON[1] - h_lon)
    lat_hi = lat_lo + h_lat
    lon_hi = lon_lo + h_lon
    r = lambda x: round(x, 6)
    return {"type": "Polygon", "coordinates": [[
        [r(lon_lo), r(lat_lo)], [r(lon_hi), r(lat_lo)],
        [r(lon_hi), r(lat_hi)], [r(lon_lo), r(lat_hi)],
        [r(lon_lo), r(lat_lo)]]]}


def _sample_epoch_ms(rng: random.Random) -> int:
    lo = int(datetime(2015, 1, 1).timestamp() * 1000)
    hi = int(datetime(2024, 12, 31).timestamp() * 1000)
    return rng.randint(lo, hi)


def _sample_asin(rng: random.Random) -> str:
    # ASIN format: B + 9 alphanumeric
    chars = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "B" + "".join(rng.choice(chars) for _ in range(9))


def _sample_user_token(rng: random.Random, prefix: str = "AG", length: int = 26) -> str:
    chars = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    return prefix + "".join(rng.choice(chars) for _ in range(length))


def _sample_label_subset(rng: random.Random) -> list[str]:
    k = rng.randint(2, len(_KNOWN_LABELS))
    return rng.sample(_KNOWN_LABELS, k)


# ───────────────────────────────────────────────────────────────────────
# Param spec resolver
# ───────────────────────────────────────────────────────────────────────

def _resolve_param(spec: Any, rng: random.Random) -> Any:
    """Given a param spec value from Template.params, return a sampled value."""
    if isinstance(spec, list):
        return rng.choice(spec)
    if not isinstance(spec, str):
        return spec
    s = spec.strip()
    if s.startswith("int_range("):
        return _sample_int_range(s, rng)
    if s.startswith("sample(int"):
        return _sample_int_batch(s, rng)
    if s == "iso_date_range" or s == "iso_date":
        return _sample_iso_date(rng)
    if s == "lat":
        return _sample_lat(rng)
    if s == "lon":
        return _sample_lon(rng)
    if s == "thai_box_polygon":
        return _sample_thai_box(rng)
    if s == "epoch_ms_range":
        return _sample_epoch_ms(rng)
    if s == "sample_asin":
        return _sample_asin(rng)
    if s in ("sample_user_id",):
        return rng.randint(1, 10_000_000)
    if s in ("sample_user",):
        return _sample_user_token(rng)
    if s == "random_label_subset":
        return _sample_label_subset(rng)
    return s  # fallback: treat as literal


# ───────────────────────────────────────────────────────────────────────
# Pipeline walker
# ───────────────────────────────────────────────────────────────────────

_PLACEHOLDER_FULL_RE = re.compile(r"^\$\$(\w+)$")
_PLACEHOLDER_SUB_RE = re.compile(r"\$\$(\w+)")


def _get_or_sample(name: str, params: dict, rng: random.Random, values: dict) -> Any:
    if name not in values:
        if name not in params:
            raise KeyError(f"placeholder $${name} not in template.params")
        values[name] = _resolve_param(params[name], rng)
    return values[name]


def _materialize(node: Any, params: dict, rng: random.Random, values: dict) -> Any:
    """Recursively walk a BSON-like structure, replacing $$NAME strings
    with a sampled value looked up in `params`.

    Two replacement modes:
      1. Whole-string match ($$NAME only)     → value replaces the node
         (preserves non-string types like int/list/float).
      2. Substring match ("prefix$$NAME…")    → value is str()'d and
         spliced inside the host string (e.g. regex fragments).

    Uses `values` as a memo so the same placeholder resolves to the same
    value within a single pipeline (e.g. $$LAT_LO used twice stays
    consistent with itself after resolution).
    """
    if isinstance(node, dict):
        return {k: _materialize(v, params, rng, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_materialize(v, params, rng, values) for v in node]
    if isinstance(node, str):
        m_full = _PLACEHOLDER_FULL_RE.match(node)
        if m_full:
            return _get_or_sample(m_full.group(1), params, rng, values)
        # substring mode: replace every $$NAME with str(value)
        def _sub(match):
            return str(_get_or_sample(match.group(1), params, rng, values))
        if "$$" in node:
            return _PLACEHOLDER_SUB_RE.sub(_sub, node)
    return node


def _fix_degenerate_ranges(node: Any) -> Any:
    """Walk the pipeline and swap $gte/$lt (or $gt/$lte) pairs whose
    bounds are reversed — can happen when lo and hi are sampled
    independently from a symmetric range (e.g. lat, lon).

    A valid range must satisfy: lower ≤ upper. If the sampler produced
    lower > upper we swap, so the query still selects a non-empty region.
    """
    if isinstance(node, dict):
        # fix this level first
        lo_key = next((k for k in ("$gte", "$gt") if k in node), None)
        hi_key = next((k for k in ("$lt", "$lte") if k in node), None)
        if lo_key and hi_key:
            lo, hi = node[lo_key], node[hi_key]
            try:
                if lo > hi:
                    node[lo_key], node[hi_key] = hi, lo
            except TypeError:
                pass  # non-comparable (e.g. str vs int), leave as is
        return {k: _fix_degenerate_ranges(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_fix_degenerate_ranges(v) for v in node]
    return node


def materialize_pipeline(template, rng: random.Random) -> list:
    """Convert a Template to an executable aggregation pipeline.

    Returns a deep copy of `template.pipeline` with every $$PARAM replaced.
    Raises KeyError if a placeholder is missing from template.params.
    After materialization, degenerate ranges (lo > hi) are auto-swapped.
    """
    values: dict = {}
    p = _materialize(template.pipeline, template.params or {}, rng, values)
    return _fix_degenerate_ranges(p)


# ───────────────────────────────────────────────────────────────────────
# Self-check
# ───────────────────────────────────────────────────────────────────────

def _self_check() -> None:
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "mongo", "workload"))
    from templates import ALL_TEMPLATES

    rng = random.Random(42)
    ok = 0
    skipped = 0
    fails = []
    for qid, t in sorted(ALL_TEMPLATES.items()):
        try:
            p = materialize_pipeline(t, rng)
            # ensure no $$ placeholder leaked
            s = repr(p)
            if "$$" in s:
                fails.append((qid, "leak", s[:120]))
            else:
                ok += 1
        except KeyError as e:
            skipped += 1
            fails.append((qid, "missing", str(e)))
        except Exception as e:
            fails.append((qid, type(e).__name__, str(e)))

    print(f"materialize OK   : {ok}")
    print(f"missing params   : {skipped}")
    print(f"failures         : {len(fails) - skipped}")
    for qid, kind, msg in fails[:20]:
        print(f"  {qid}  {kind}  {msg}")

    # also smoke-test determinism
    rng1 = random.Random(7)
    rng2 = random.Random(7)
    p1 = materialize_pipeline(ALL_TEMPLATES["Q10"], rng1)
    p2 = materialize_pipeline(ALL_TEMPLATES["Q10"], rng2)
    assert p1 == p2, "non-deterministic materialization!"
    print("determinism     : OK")

    # show one realized pipeline
    print("\n— example realised pipeline (Q10, seed=99) —")
    rng = random.Random(99)
    p = materialize_pipeline(ALL_TEMPLATES["Q10"], rng)
    for stage in p:
        print(f"  {stage}")


if __name__ == "__main__":
    _self_check()
