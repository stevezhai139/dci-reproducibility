"""
pg_workloads.py
================
Paper 3D — Phase 3 (task T3.4b).  Workload configurations for the
PostgreSQL end-to-end harness `pg_adaptation.py`.

Each entry of `WORKLOAD_CONFIGS` specifies the workload-specific
constants the harness needs:
  - queries_module      Python module exporting QUERIES, QUERY_TABLES, QUERY_COLS
  - sf_db_map           scale-factor → database name
  - has_sf_axis         True if the workload supports multiple SFs (TPC-H) or
                        is single-DB (JOB on imdb)
  - phases              4-phase drift schedule (qid + weight per phase)
  - lambda_map          per-phase Poisson arrival rate (queries/sec)
  - advisor_index_name  the supplementary index the advisor creates/drops
  - advisor_create_ddl  CREATE INDEX statement (PG dialect, can use IF NOT EXISTS)
  - advisor_drop_ddl    DROP INDEX statement (PG dialect, IF EXISTS)
  - backbone_indexes    list of CREATE INDEX statements run once per SF in
                        ensure_indexes() — the "PK + FK structural" baseline
                        every strategy starts with (PHASE3_PLAN §5)
  - verify_table        table the harness queries to verify the database has
                        data (replaces TPC-H's hard-coded `lineitem` check)
  - warmup_qids         3 fast qids used to warm the buffer cache

DCI integration (`dci_gated`, `calibrate_dci_gate`, `DCIGate.decide`,
per-window throughput, paired-RCB design) is workload-independent and
lives in the harness directly — it does not appear in this module.
"""
from __future__ import annotations


WORKLOAD_CONFIGS: dict[str, dict] = {
    # ────────────────────────────────────────────────────────────────
    #  TPC-H (default; the original 07_adaptation_comparison.py workload)
    # ────────────────────────────────────────────────────────────────
    "tpch": {
        "queries_module": "tpch_queries",
        "sf_db_map": {0.2: "tpch_sf0_2", 1.0: "tpch_sf1",
                      3.0: "tpch_sf3", 7.0: "tpch_sf7", 10.0: "tpch_sf10"},
        "has_sf_axis":   True,
        "phases": [
            {"name": "Reporting",   "qs": ["Q1", "Q6", "Q14"],          "w": [4, 3, 3]},
            {"name": "JoinHeavy",   "qs": ["Q3", "Q5", "Q7", "Q10"],   "w": [3, 3, 2, 2]},
            {"name": "Aggregation", "qs": ["Q1", "Q4", "Q11", "Q12"],  "w": [2, 3, 2, 3]},
            {"name": "MultiJoin",   "qs": ["Q5", "Q7", "Q17", "Q18"],  "w": [2, 3, 3, 2]},
        ],
        "lambda_map": {"Reporting": 50.0, "JoinHeavy": 30.0,
                       "Aggregation": 40.0, "MultiJoin": 25.0},
        "advisor_index_name": "ix_advisor_bench",
        "advisor_create_ddl":
            "CREATE INDEX ix_advisor_bench ON lineitem(l_shipdate, l_discount)",
        "advisor_drop_ddl":
            "DROP INDEX IF EXISTS ix_advisor_bench",
        "backbone_indexes": [
            "CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey ON lineitem(l_orderkey)",
            "CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey)",
            "CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey)",
            "CREATE INDEX IF NOT EXISTS idx_lineitem_l_discount ON lineitem(l_discount)",
            "CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey)",
            "CREATE INDEX IF NOT EXISTS idx_orders_o_orderdate ON orders(o_orderdate)",
            "CREATE INDEX IF NOT EXISTS idx_partsupp_partkey ON partsupp(ps_partkey)",
            "CREATE INDEX IF NOT EXISTS idx_partsupp_suppkey ON partsupp(ps_suppkey)",
            "CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey)",
            "CREATE INDEX IF NOT EXISTS idx_nation_n_name ON nation(n_name)",
            "CREATE INDEX IF NOT EXISTS idx_nation_regionkey ON nation(n_regionkey)",
            "CREATE INDEX IF NOT EXISTS idx_part_p_type ON part(p_type)",
            "CREATE INDEX IF NOT EXISTS idx_region_r_name ON region(r_name)",
            "CREATE INDEX IF NOT EXISTS idx_supplier_nationkey ON supplier(s_nationkey)",
        
        ],
        "verify_table":  "lineitem",
        "warmup_qids":   ["Q1", "Q3", "Q6"],
    },

    # ────────────────────────────────────────────────────────────────
    #  JOB (Join Order Benchmark) — 113 queries on IMDB (added T3.4b)
    # ────────────────────────────────────────────────────────────────
    "job": {
        "queries_module": "job_queries",
        # JOB has no scale-factor axis — only one IMDB instance (74M rows
        # total per MYSQL_SETUP.md §3).  We use sf=1.0 as a stable lookup
        # key so the per-SF CSV/loop machinery still works unchanged.
        "sf_db_map":     {1.0: "imdb"},
        "has_sf_axis":   False,
        # 4-phase drift schedule.  Each phase is dominated by a different
        # table cluster (mirrors TPC-H's Reporting/JoinHeavy/.../MultiJoin
        # pattern).  Representative qids picked from JOB families 1-33 —
        # 3 queries per phase × 4 phases = 12 queries used out of the 113
        # vendored, matching the TPC-H harness's ~12-of-22 coverage.
        "phases": [
            {"name": "CompanyProduction", "qs": ["1a", "2a", "4a"],   "w": [4, 3, 3]},
            {"name": "CastKeyword",       "qs": ["6a", "8a", "11a"],  "w": [3, 4, 3]},
            {"name": "MovieInfo",         "qs": ["13a", "17a", "22a"], "w": [3, 4, 3]},
            {"name": "ComplexLink",       "qs": ["25a", "29a", "33a"], "w": [3, 4, 3]},
        ],
        # Lower λ values than TPC-H because JOB queries are heavier
        # (multi-table joins on 74M-row IMDB → typical exec_ms is higher).
        "lambda_map": {"CompanyProduction": 30.0, "CastKeyword": 25.0,
                       "MovieInfo": 25.0, "ComplexLink": 20.0},
        "advisor_index_name": "ix_advisor_bench",
        # Advisor adds a COMPOUND index on the largest table (cast_info,
        # 36M rows).  Single-column indexes on movie_id and person_id are
        # already in the backbone — the advisor's (person_id, movie_id)
        # compound is a meaningful addition that backbone does not cover.
        "advisor_create_ddl":
            "CREATE INDEX ix_advisor_bench ON cast_info(person_id, movie_id)",
        "advisor_drop_ddl":
            "DROP INDEX IF EXISTS ix_advisor_bench",
        # Backbone = FK indexes on the most-joined-on column of each
        # large table (movie_id is the central join key across cast_info,
        # movie_info, movie_info_idx, movie_companies, movie_keyword,
        # movie_link).  Plus a couple of selective lookups (cast_info by
        # person_id, title by production_year).  The PK on each table
        # remains the default; these add secondary lookup paths.
        # Comprehensive FK backbone — 24 indexes covering every join key
        # JOB queries traverse (T3.4b smoke-debug fix 2026-05-28: original
        # 8-index backbone left 16 FK columns unindexed → many queries
        # were forced into full table scans on cast_info 36M rows
        # → wall_QPS 0.29 = ~20× slower than TPC-H).
        "backbone_indexes": [
            # cast_info (36M rows — most-joined-on)
            "CREATE INDEX IF NOT EXISTS idx_ci_movie       ON cast_info(movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_ci_person      ON cast_info(person_id)",
            "CREATE INDEX IF NOT EXISTS idx_ci_role        ON cast_info(role_id)",
            # movie_companies (2.6M rows)
            "CREATE INDEX IF NOT EXISTS idx_mc_movie       ON movie_companies(movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_mc_company     ON movie_companies(company_id)",
            "CREATE INDEX IF NOT EXISTS idx_mc_comptype    ON movie_companies(company_type_id)",
            # movie_info (14.8M rows)
            "CREATE INDEX IF NOT EXISTS idx_mi_movie       ON movie_info(movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_mi_infotype    ON movie_info(info_type_id)",
            # movie_info_idx (1.4M rows)
            "CREATE INDEX IF NOT EXISTS idx_mii_movie      ON movie_info_idx(movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_mii_infotype   ON movie_info_idx(info_type_id)",
            # movie_keyword (4.5M rows)
            "CREATE INDEX IF NOT EXISTS idx_mk_movie       ON movie_keyword(movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_mk_keyword     ON movie_keyword(keyword_id)",
            # movie_link (30K rows)
            "CREATE INDEX IF NOT EXISTS idx_ml_movie       ON movie_link(movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_ml_linkedmovie ON movie_link(linked_movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_ml_linktype    ON movie_link(link_type_id)",
            # title (2.5M rows)
            "CREATE INDEX IF NOT EXISTS idx_t_kind         ON title(kind_id)",
            "CREATE INDEX IF NOT EXISTS idx_t_year         ON title(production_year)",
            # aka_name (901K rows)
            "CREATE INDEX IF NOT EXISTS idx_an_person      ON aka_name(person_id)",
            # aka_title (361K rows)
            "CREATE INDEX IF NOT EXISTS idx_at_movie       ON aka_title(movie_id)",
            # person_info (2.96M rows)
            "CREATE INDEX IF NOT EXISTS idx_pi_person      ON person_info(person_id)",
            "CREATE INDEX IF NOT EXISTS idx_pi_infotype    ON person_info(info_type_id)",
            # complete_cast (135K rows)
            "CREATE INDEX IF NOT EXISTS idx_cc_movie       ON complete_cast(movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_cc_subject     ON complete_cast(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_cc_status      ON complete_cast(status_id)",
            # ── Selective predicate indexes on lookup tables (T3.4b v2, 2026-05-28) ──
            # JOB queries filter heavily on these string columns of small lookup
            # tables; without secondary indexes the optimizer full-scans the
            # lookup table and picks it as the driving table.
            # EXPLAIN on query 6a after v1 (24 FK indexes only) confirmed:
            # keyword full-scanned 130K rows -> wall_QPS 0.327 (no improvement
            # over 8-FK baseline 0.291).
            "CREATE INDEX IF NOT EXISTS idx_k_keyword   ON keyword(keyword)",
            "CREATE INDEX IF NOT EXISTS idx_cn_country  ON company_name(country_code)",
            "CREATE INDEX IF NOT EXISTS idx_it_info     ON info_type(info)",
            "CREATE INDEX IF NOT EXISTS idx_ct_kind     ON company_type(kind)",
            "CREATE INDEX IF NOT EXISTS idx_rt_role     ON role_type(role)",
            "CREATE INDEX IF NOT EXISTS idx_lt_link     ON link_type(link)",
        ],
        # cast_info has 36M rows in IMDB — the canonical "data is loaded" check.
        "verify_table":  "cast_info",
        # Light queries to warm the buffer cache (one per main phase).
        "warmup_qids":   ["1a", "6a", "13a"],
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Paper 3C — S5 Part 2 (added 2026-07-07): "tpch_mixed" schedule variant
# ══════════════════════════════════════════════════════════════════════
# Live MIXED-drift schedule for the detector-resolution sweep
# (PART2_BUILD_SPEC.md). The legacy TPC-H schedule moves ONLY template
# structure between DISJOINT query sets, so every block is a single-cause
# cell (and the onset deviation is so template-dominant that the block DCI
# stays ~1.2). This variant reproduces the OFFLINE mixed cells of §6.2
# (cost_benefit.build_trajectory) live:
#
#   • 6 phases × 4 windows (win_per_ph=4, N_WINDOWS still 24), onsets at
#     windows 5, 9, 13, 17, 21 alternating template / volume / template /
#     volume / template — the §6.2 rotation.
#   • TEMPLATE moves are HALF-SWAPS (keep 3 of 5 active templates, swap 2),
#     matching build_trajectory's n_swap = k//2 — NOT full phase swaps.
#   • VOLUME moves toggle queries-per-window 20 ↔ 32 (the exact offline
#     VOLUME_LEVELS pair of §6.2; S_V dip 0.375) with
#     the template mix HELD, a pure S_V move.
#
# Offline pre-validation (part2_validate_offline.py) confirms this layout
# puts the pooled onset DCI in the §6.2 mixed band and drives the gated
# rolling DCI above τ=1.5 in-block. `qpw` = per-phase queries-per-window
# override; `win_per_ph` = per-workload override of WIN_PER_PH (legacy
# configs untouched: both default in the harness).
WORKLOAD_CONFIGS["tpch_mixed"] = {
    **WORKLOAD_CONFIGS["tpch"],
    # Part 2 is defined on the SF=1 instance only (parity with the
    # 2026-07-02 live tau runs); other SFs are intentionally unmapped.
    "sf_db_map": {1.0: "tpch_sf1"},
    "win_per_ph": 4,
    "phases": [
        # k=5 active templates; half-swap chain (swapped-in marked ←)
        {"name": "MixBase",   "qs": ["Q1", "Q6", "Q14", "Q3", "Q12"],  "w": [3, 2, 2, 2, 1], "qpw": 20},
        {"name": "MixShift1", "qs": ["Q1", "Q6", "Q3", "Q5", "Q10"],   "w": [3, 2, 2, 2, 1], "qpw": 20},  # ← Q5,Q10 (drop Q14,Q12)  TEMPLATE
        {"name": "MixSurge1", "qs": ["Q1", "Q6", "Q3", "Q5", "Q10"],   "w": [3, 2, 2, 2, 1], "qpw": 32},  # same mix, 20→32           VOLUME
        {"name": "MixShift2", "qs": ["Q3", "Q5", "Q10", "Q17", "Q18"], "w": [3, 2, 2, 2, 1], "qpw": 32},  # ← Q17,Q18 (drop Q1,Q6)    TEMPLATE
        {"name": "MixCalm",   "qs": ["Q3", "Q5", "Q10", "Q17", "Q18"], "w": [3, 2, 2, 2, 1], "qpw": 20},  # same mix, 32→20           VOLUME
        {"name": "MixShift3", "qs": ["Q10", "Q17", "Q18", "Q4", "Q7"], "w": [3, 2, 2, 2, 1], "qpw": 20},  # ← Q4,Q7 (drop Q3,Q5)      TEMPLATE
    ],
    # λ HELD CONSTANT across every boundary: the volume move is a
    # query-COUNT move (S_V axis), not an arrival-rate move — the
    # arrival-drift config was dropped for the honest reason recorded
    # in §6.1 (S_P does not respond when templates are held).
    "lambda_map": {"MixBase": 40.0, "MixShift1": 40.0, "MixSurge1": 40.0,
                   "MixShift2": 40.0, "MixCalm": 40.0, "MixShift3": 40.0},
}
