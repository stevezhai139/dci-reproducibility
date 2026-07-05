"""
templates.py — Q1–Q24 workload templates for Paper 3A V2 MongoDB cross-engine.

Design:
  • 5 document types × ~5 templates each = 24 total (plus QX spillover).
  • Each template is a pure-Python dict describing:
      - qid      : stable identifier (Q1..Q24)
      - type     : target document type the query primarily hits
      - dim      : T2 dimension primarily witnessed {S_E, S_P, S_V, S_G, S_T}
      - shape    : "equality" | "range" | "geo" | "text" | "graph" | "sort"
      - candidate_index : the single-field or compound index that would
                          accelerate this query (cost model fit target)
      - pipeline : list of aggregation stages (parameterised with $placeholder)
      - params   : dict of sampleable parameter ranges (for workload generator)
      - weight   : relative frequency in phase mix

Phase design:
  Phases cycle through 4 type-dominant mixes to exercise T5 phase shift:
    PHASE_EDGE     : 80% edge queries + 20% noise
    PHASE_GEO      : 80% OSM geo (thailand_osm) + 20% noise
    PHASE_TEXT     : 80% arxiv textual + 20% noise
    PHASE_REVIEW   : 80% patio_review + 20% noise
    (spatial/ghs is a 5th latent phase but smaller sample — used for S_E witness only)

Actual data label mapping (mydb.combined_data):
  spatial   → label="ghs_built_2018"  (fields: coordinate_system, is_spatial_raster, mean_value)
  edge      → type="edge"             (fields: from_node, to_node; label="livejournal_edge")
  changeset → label="thailand_osm"    (fields: lat, lon, timestamp, id, version)
  textual   → label="arxiv"           (fields: abstract, categories, doi, title, update_date)
  review    → label="patio_review"    (fields: parent_asin, rating, helpful_vote, timestamp, user_id)

This file is pure data — no mongod I/O. Safe to import in any context.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Template:
    qid: str
    doc_type: str           # target type discriminator (spatial, edge, changeset, textual, review)
    dim: str                # S_E | S_P | S_V | S_G | S_T
    shape: str              # equality | range | geo | text | graph | sort | compound
    candidate_index: tuple  # (("field", ASC|DESC), ...)
    pipeline: list          # aggregation stages (use $$PARAM_name placeholders)
    params: dict            # parameter ranges for the generator
    weight: float = 1.0
    description: str = ""


# ───────────────────────────────────────────────────────────────────────
# TYPE 1 — spatial (GHS raster windows, label="ghs_built_2018")
# ───────────────────────────────────────────────────────────────────────

Q1 = Template(
    qid="Q1",
    doc_type="spatial",
    dim="S_E",
    shape="equality",
    candidate_index=(("label", 1),),
    pipeline=[
        {"$match": {"label": "ghs_built_2018"}},
        {"$count": "n"},
    ],
    params={},
    description="Count all spatial docs — S_E witness: purely type-discrimination",
)

Q2 = Template(
    qid="Q2",
    doc_type="spatial",
    dim="S_V",
    shape="equality",
    candidate_index=(("label", 1), ("coordinate_system", 1)),
    pipeline=[
        {"$match": {"label": "ghs_built_2018", "coordinate_system": "$$CRS"}},
        {"$group": {"_id": None, "avg_mean": {"$avg": "$mean_value"}}},
    ],
    params={"CRS": ["ESRI:54009", "EPSG:4326"]},
    description="Spatial aggregation by CRS — S_V witness: volume scaling",
)

Q3 = Template(
    qid="Q3",
    doc_type="spatial",
    dim="S_P",
    shape="compound",
    candidate_index=(("label", 1), ("is_spatial_raster", 1), ("mean_value", 1)),
    pipeline=[
        {"$match": {"label": "ghs_built_2018", "is_spatial_raster": True, "mean_value": {"$gte": "$$V_MIN"}}},
        {"$limit": 1000},
    ],
    params={"V_MIN": [0.0, 50.0, 100.0, 200.0]},
    description="Spatial filter with bool+range — S_P pattern witness",
)

# ───────────────────────────────────────────────────────────────────────
# TYPE 2 — edge (SNAP / LiveJournal graph, type="edge")
# ───────────────────────────────────────────────────────────────────────

Q4 = Template(
    qid="Q4",
    doc_type="edge",
    dim="S_E",
    shape="equality",
    candidate_index=(("type", 1),),
    pipeline=[
        {"$match": {"type": "edge"}},
        {"$count": "n"},
    ],
    params={},
    description="Edge count — S_E witness for edge type",
)

Q5 = Template(
    qid="Q5",
    doc_type="edge",
    dim="S_P",
    shape="graph",
    candidate_index=(("from_node", 1),),
    pipeline=[
        {"$match": {"type": "edge", "from_node": "$$NODE_ID"}},
        {"$project": {"to_node": 1, "label": 1}},
    ],
    params={"NODE_ID": "int_range(1, 4_000_000)"},
    description="1-hop neighbourhood of a node — single-field index benefit",
)

Q6 = Template(
    qid="Q6",
    doc_type="edge",
    dim="S_P",
    shape="graph",
    candidate_index=(("from_node", 1), ("to_node", 1)),
    pipeline=[
        {"$match": {"type": "edge", "from_node": {"$in": "$$NODE_BATCH"}}},
        {"$group": {"_id": "$to_node", "in_deg": {"$sum": 1}}},
        {"$sort": {"in_deg": -1}},
        {"$limit": 20},
    ],
    params={"NODE_BATCH": "sample(int, k=50, 1, 4_000_000)"},
    description="Multi-source BFS probe — compound edge index benefit",
)

Q7 = Template(
    qid="Q7",
    doc_type="edge",
    dim="S_V",
    shape="equality",
    candidate_index=(("label", 1),),
    pipeline=[
        {"$match": {"label": "$$LBL"}},
        {"$count": "n"},
    ],
    params={"LBL": ["livejournal_edge", "livejournal_node"]},
    description="Label partition count — S_V witness across sub-datasets",
)

# ───────────────────────────────────────────────────────────────────────
# TYPE 3 — changeset (OSM Thailand, label="thailand_osm")
# Fields: lat, lon, timestamp, id, version
# ───────────────────────────────────────────────────────────────────────

Q8 = Template(
    qid="Q8",
    doc_type="changeset",
    dim="S_E",
    shape="equality",
    candidate_index=(("label", 1),),
    pipeline=[
        {"$match": {"label": "thailand_osm"}},
        {"$count": "n"},
    ],
    params={},
    description="Changeset count — S_E witness",
)

Q9 = Template(
    qid="Q9",
    doc_type="changeset",
    dim="S_T",
    shape="range",
    candidate_index=(("label", 1), ("timestamp", 1)),
    pipeline=[
        {"$match": {"label": "thailand_osm", "timestamp": {"$gte": "$$T0", "$lt": "$$T1"}}},
        {"$count": "n"},
    ],
    params={"T0": "iso_date_range", "T1": "iso_date_range"},
    description="Temporal range on timestamp — S_T witness",
)

Q10 = Template(
    qid="Q10",
    doc_type="changeset",
    dim="S_G",
    shape="geo",
    candidate_index=(("label", 1), ("geometry", "2dsphere")),
    pipeline=[
        {"$match": {
            "label": "thailand_osm",
            "geometry": {"$geoWithin": {"$geometry": "$$BOX"}},
        }},
        {"$count": "n"},
    ],
    params={"BOX": "thai_box_polygon"},
    description="Geo bounding-box on thailand_osm points — real $geoWithin "
                "served by a 2dsphere index (PATH A). S_G witness.",
)

Q11 = Template(
    qid="Q11",
    doc_type="changeset",
    dim="S_P",
    shape="compound",
    candidate_index=(("label", 1), ("id", 1), ("timestamp", -1)),
    pipeline=[
        {"$match": {"label": "thailand_osm", "id": {"$gte": "$$ID_LO", "$lt": "$$ID_HI"}}},
        {"$sort": {"timestamp": -1}},
        {"$limit": 100},
    ],
    params={"ID_LO": "int_range(1, 10_000_000)", "ID_HI": "int_range(1, 10_000_000)"},
    description="OSM id range with sort — S_P witness (compound + sort)",
)

Q12 = Template(
    qid="Q12",
    doc_type="changeset",
    dim="S_V",
    shape="range",
    candidate_index=(("label", 1), ("version", 1)),
    pipeline=[
        {"$match": {"label": "thailand_osm", "version": {"$gte": "$$V_MIN"}}},
        {"$count": "n"},
    ],
    params={"V_MIN": [1, 2, 5, 10]},
    description="High-version changeset filter — S_V witness",
)

# ───────────────────────────────────────────────────────────────────────
# TYPE 4 — textual (arXiv, label="arxiv")
# Fields: abstract, categories, doi, title, update_date, authors, id
# ───────────────────────────────────────────────────────────────────────

Q13 = Template(
    qid="Q13",
    doc_type="textual",
    dim="S_E",
    shape="equality",
    candidate_index=(("label", 1),),
    pipeline=[
        {"$match": {"label": "arxiv"}},
        {"$count": "n"},
    ],
    params={},
    description="Textual count — S_E witness",
)

Q14 = Template(
    qid="Q14",
    doc_type="textual",
    dim="S_P",
    shape="equality",
    candidate_index=(("label", 1), ("categories", 1)),
    pipeline=[
        {"$match": {"label": "arxiv", "categories": {"$regex": "^$$CAT"}}},
        {"$project": {"title": 1, "doi": 1}},
        {"$limit": 100},
    ],
    params={"CAT": ["hep-ph", "cs.CG", "math.CO", "cond-mat"]},
    description="Category prefix filter — S_P witness (regex range)",
)

Q15 = Template(
    qid="Q15",
    doc_type="textual",
    dim="S_T",
    shape="range",
    candidate_index=(("label", 1), ("update_date", 1)),
    pipeline=[
        {"$match": {"label": "arxiv", "update_date": {"$gte": "$$YEAR"}}},
        {"$count": "n"},
    ],
    params={"YEAR": ["2015", "2018", "2021"]},
    description="Recent arxiv papers — S_T witness on update_date",
)

Q16 = Template(
    qid="Q16",
    doc_type="textual",
    dim="S_P",
    shape="text",
    candidate_index=(("label", 1), ("abstract", "text")),
    pipeline=[
        {"$match": {"label": "arxiv", "$text": {"$search": "$$TERM"}}},
        {"$limit": 20},
    ],
    params={"TERM": ["quantum", "neural", "dark matter", "turbulence"]},
    description="Text search on abstract — high-cost text index witness",
)

# ───────────────────────────────────────────────────────────────────────
# TYPE 5 — review (Amazon Patio reviews, label="patio_review")
# Fields: parent_asin, rating, helpful_vote, timestamp, user_id, text, title, asin
# ───────────────────────────────────────────────────────────────────────

Q17 = Template(
    qid="Q17",
    doc_type="review",
    dim="S_E",
    shape="equality",
    candidate_index=(("label", 1),),
    pipeline=[
        {"$match": {"label": "patio_review"}},
        {"$count": "n"},
    ],
    params={},
    description="Review count — S_E witness via label",
)

Q18 = Template(
    qid="Q18",
    doc_type="review",
    dim="S_P",
    shape="equality",
    candidate_index=(("label", 1), ("parent_asin", 1)),
    pipeline=[
        {"$match": {"label": "patio_review", "parent_asin": "$$ASIN"}},
        {"$project": {"rating": 1, "title": 1, "timestamp": 1}},
    ],
    params={"ASIN": "sample_asin"},
    description="Reviews for a product — compound index witness",
)

Q19 = Template(
    qid="Q19",
    doc_type="review",
    dim="S_T",
    shape="range",
    candidate_index=(("label", 1), ("timestamp", 1)),
    pipeline=[
        {"$match": {"label": "patio_review", "timestamp": {"$gte": "$$T_EPOCH"}}},
        {"$count": "n"},
    ],
    params={"T_EPOCH": "epoch_ms_range"},
    description="Recent reviews — S_T witness via epoch ms",
)

Q20 = Template(
    qid="Q20",
    doc_type="review",
    dim="S_V",
    shape="range",
    candidate_index=(("label", 1), ("helpful_vote", -1)),
    pipeline=[
        {"$match": {"label": "patio_review", "helpful_vote": {"$gte": "$$HV"}}},
        {"$sort": {"helpful_vote": -1}},
        {"$limit": 50},
    ],
    params={"HV": [1, 5, 10, 50]},
    description="Most helpful reviews — S_V witness + sort",
)

Q21 = Template(
    qid="Q21",
    doc_type="review",
    dim="S_P",
    shape="equality",
    candidate_index=(("user_id", 1),),
    pipeline=[
        {"$match": {"user_id": "$$UID"}},
        {"$project": {"rating": 1, "asin": 1, "timestamp": 1}},
    ],
    params={"UID": "sample_user"},
    description="User's reviews — single-field index witness",
)

# ───────────────────────────────────────────────────────────────────────
# MIXED-TYPE (cross-phase probe queries)
# ───────────────────────────────────────────────────────────────────────

Q22 = Template(
    qid="Q22",
    doc_type="MIXED",
    dim="S_E",
    shape="equality",
    candidate_index=(("label", 1),),
    pipeline=[
        {"$group": {"_id": "$label", "n": {"$sum": 1}}},
    ],
    params={},
    description="Global label histogram — worst-case COLLSCAN, probes backbone index",
)

Q23 = Template(
    qid="Q23",
    doc_type="MIXED",
    dim="S_V",
    shape="compound",
    candidate_index=(("label", 1), ("type", 1)),
    pipeline=[
        {"$match": {"label": {"$in": "$$LBLS"}}},
        {"$group": {"_id": {"lbl": "$label", "t": "$type"}, "n": {"$sum": 1}}},
    ],
    params={"LBLS": "random_label_subset"},
    description="Cross-label join-like — tests compound index value",
)

Q24 = Template(
    qid="Q24",
    doc_type="MIXED",
    dim="S_T",
    shape="range",
    candidate_index=(("timestamp", 1),),
    pipeline=[
        {"$match": {"timestamp": {"$gte": "$$T0"}}},
        {"$group": {"_id": "$label", "n": {"$sum": 1}}},
    ],
    params={"T0": "iso_date"},
    description="Temporal range across all types — probes shared time index",
)


ALL_TEMPLATES: dict[str, Template] = {
    t.qid: t
    for t in [
        Q1, Q2, Q3,
        Q4, Q5, Q6, Q7,
        Q8, Q9, Q10, Q11, Q12,
        Q13, Q14, Q15, Q16,
        Q17, Q18, Q19, Q20, Q21,
        Q22, Q23, Q24,
    ]
}


# ───────────────────────────────────────────────────────────────────────
# Phase mixes (feed into paired-RCB workload generator)
# ───────────────────────────────────────────────────────────────────────

PHASE_EDGE = {
    "name": "edge",
    "dominant_type": "edge",
    "mix": {
        "Q4": 0.15, "Q5": 0.40, "Q6": 0.25,           # 80% edge
        "Q1": 0.05, "Q8": 0.05, "Q13": 0.05, "Q17": 0.05,  # 20% noise (one per other type)
    },
    "ideal_index": [("from_node", 1), ("from_node", 1), ("to_node", 1)],
}

PHASE_GEO = {
    "name": "geo",
    "dominant_type": "changeset",
    "mix": {
        "Q9": 0.20, "Q10": 0.35, "Q11": 0.15, "Q12": 0.10,  # 80% changeset
        "Q1": 0.05, "Q4": 0.05, "Q13": 0.05, "Q17": 0.05,
    },
    "ideal_index": [("label", 1), ("geometry", "2dsphere")],
}

PHASE_TEXT = {
    "name": "text",
    "dominant_type": "textual",
    "mix": {
        "Q13": 0.10, "Q14": 0.25, "Q15": 0.20, "Q16": 0.25,  # 80% textual
        "Q1": 0.05, "Q4": 0.05, "Q8": 0.05, "Q17": 0.05,
    },
    "ideal_index": [("categories", 1), ("update_date", 1), ("abstract", "text")],
}

PHASE_REVIEW = {
    "name": "review",
    "dominant_type": "review",
    "mix": {
        "Q17": 0.10, "Q18": 0.25, "Q19": 0.20, "Q20": 0.15, "Q21": 0.10,  # 80% review
        "Q1": 0.05, "Q4": 0.05, "Q8": 0.05, "Q13": 0.05,
    },
    "ideal_index": [("parent_asin", 1), ("timestamp", 1), ("helpful_vote", -1)],
}


ALL_PHASES = {p["name"]: p for p in [PHASE_EDGE, PHASE_GEO, PHASE_TEXT, PHASE_REVIEW]}


# ───────────────────────────────────────────────────────────────────────
# Schema usage maps (for S_A Jaccard in hsm_v2)
# ───────────────────────────────────────────────────────────────────────
# Postgres analogue uses {qid: set(table_name)} and {qid: set(column_name)}.
# Mongo equivalents:
#   QUERY_TABLES = {qid: set(doc_type)}    — surrogate "tables" = label-based type
#   QUERY_FIELDS = {qid: set(field_path)}  — every field path the query
#                  references in $match / $project / $group / $sort.
#
# For MIXED probe queries (Q22, Q23, Q24) that span all 5 doc-types we use
# the full type set so S_A correctly reflects the cross-type footprint.

_ALL_DOC_TYPES = {"spatial", "edge", "changeset", "textual", "review"}

QUERY_TABLES: dict[str, set[str]] = {
    "Q1":  {"spatial"},
    "Q2":  {"spatial"},
    "Q3":  {"spatial"},
    "Q4":  {"edge"},
    "Q5":  {"edge"},
    "Q6":  {"edge"},
    "Q7":  {"edge"},
    "Q8":  {"changeset"},
    "Q9":  {"changeset"},
    "Q10": {"changeset"},
    "Q11": {"changeset"},
    "Q12": {"changeset"},
    "Q13": {"textual"},
    "Q14": {"textual"},
    "Q15": {"textual"},
    "Q16": {"textual"},
    "Q17": {"review"},
    "Q18": {"review"},
    "Q19": {"review"},
    "Q20": {"review"},
    "Q21": {"review"},
    "Q22": set(_ALL_DOC_TYPES),
    "Q23": set(_ALL_DOC_TYPES),
    "Q24": set(_ALL_DOC_TYPES),
}

QUERY_FIELDS: dict[str, set[str]] = {
    "Q1":  {"label"},
    "Q2":  {"label", "coordinate_system", "mean_value"},
    "Q3":  {"label", "is_spatial_raster", "mean_value"},
    "Q4":  {"type"},
    "Q5":  {"type", "from_node", "to_node", "label"},
    "Q6":  {"type", "from_node", "to_node"},
    "Q7":  {"label"},
    "Q8":  {"label"},
    "Q9":  {"label", "timestamp"},
    "Q10": {"label", "geometry"},
    "Q11": {"label", "id", "timestamp"},
    "Q12": {"label", "version"},
    "Q13": {"label"},
    "Q14": {"label", "categories", "title", "doi"},
    "Q15": {"label", "update_date"},
    "Q16": {"label", "abstract"},
    "Q17": {"label"},
    "Q18": {"label", "parent_asin", "rating", "title", "timestamp"},
    "Q19": {"label", "timestamp"},
    "Q20": {"label", "helpful_vote"},
    "Q21": {"user_id", "rating", "asin", "timestamp"},
    "Q22": {"label"},
    "Q23": {"label", "type"},
    "Q24": {"timestamp", "label"},
}

assert set(QUERY_TABLES.keys()) == set(ALL_TEMPLATES.keys()), \
    f"QUERY_TABLES key mismatch: {set(QUERY_TABLES.keys()) ^ set(ALL_TEMPLATES.keys())}"
assert set(QUERY_FIELDS.keys()) == set(ALL_TEMPLATES.keys()), \
    f"QUERY_FIELDS key mismatch: {set(QUERY_FIELDS.keys()) ^ set(ALL_TEMPLATES.keys())}"


# Sorted list of all template ids — needed by make_window_features for the
# fixed-length frequency vector (S_R / S_T require positionally identical
# vectors across windows).
ALL_QIDS_SORTED: list[str] = sorted(ALL_TEMPLATES.keys())


# ───────────────────────────────────────────────────────────────────────
# Self-check when run directly
# ───────────────────────────────────────────────────────────────────────

def _self_check() -> None:
    print(f"Templates defined : {len(ALL_TEMPLATES)}")
    print(f"Phases defined    : {len(ALL_PHASES)}")
    dims = {}
    for t in ALL_TEMPLATES.values():
        dims.setdefault(t.dim, []).append(t.qid)
    print("\nWitness dimension coverage:")
    for d in sorted(dims.keys()):
        print(f"  {d}: {dims[d]}")
    print("\nPhase mix sanity (sums should ≈ 1.0):")
    for name, phase in ALL_PHASES.items():
        s = sum(phase["mix"].values())
        print(f"  {name:8s} sum={s:.3f}  qids={list(phase['mix'].keys())}")
    # check all phase qids exist
    for name, phase in ALL_PHASES.items():
        for qid in phase["mix"]:
            assert qid in ALL_TEMPLATES, f"phase {name} references unknown {qid}"
    print("\nOK all phase references resolve.")
    print("\nSchema map sanity:")
    print(f"  QUERY_TABLES: {len(QUERY_TABLES)} entries  "
          f"(distinct types: {sorted({t for s in QUERY_TABLES.values() for t in s})})")
    avg_fields = sum(len(v) for v in QUERY_FIELDS.values()) / len(QUERY_FIELDS)
    print(f"  QUERY_FIELDS: {len(QUERY_FIELDS)} entries  avg_fields_per_q={avg_fields:.2f}")
    distinct_fields = sorted({f for s in QUERY_FIELDS.values() for f in s})
    print(f"  distinct fields ({len(distinct_fields)}): {distinct_fields}")


if __name__ == "__main__":
    _self_check()
