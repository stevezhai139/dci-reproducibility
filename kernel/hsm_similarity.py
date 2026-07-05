"""
hsm_similarity.py  --  Backward-compat wrapper around the canonical v2 kernel
=============================================================================
This module previously contained a parallel v2 implementation of the HSM
kernel.  An audit on 2026-04-14 found two drifts vs paper §III:

    1. S_V used queries-per-second (qps) instead of "Volume Q = query
       count per window" (paper §III-B Relational extractor).
    2. test_metric_properties.py imported from this module, so Lemma 1 was
       being verified against the legacy code -- not against the kernel
       used by the seven validation scripts.

To enforce a single source of truth, this module is now a *thin wrapper*
that delegates every scoring call to ``hsm_v2_kernel`` (which is the
verbatim port of Version 3's HSM core).  Public API names are preserved so
``experiment_runner.py``, ``sdss_workload_analyzer.py`` and
``test_metric_properties.py`` keep working unchanged.

Strict per-paper formulae (paper §III, Eqs. 1--6) -- delegated:

    S_R = 1 - arccos(rho_s) / pi                       [Eq. 1]
    S_V = min(Q_a, Q_b) / max(Q_a, Q_b)                [Eq. 2, count-based]
    S_T = 1 - (2/pi) * arccos(v_hat_A . v_hat_B)       [Eq. 3]
    S_A = 0.5*J(T_A, T_B) + 0.5*J(C_A, C_B)            [Eq. 4]
    S_P = sum_b w_b * sc_b   (DWT db4, L=3 -> SAX -> FastDTW)  [Eq. 5]
    HSM = 0.25*S_R + 0.20*S_V + 0.20*S_T + 0.20*S_A + 0.15*S_P [Eq. 6]
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Single source of truth for all five HSM dimensions and the composite
# score.  Every callable here ultimately delegates to functions in
# ``hsm_v2_kernel`` so there is only one kernel in the codebase.
#
# VENDORING NOTE (Paper 3B-Cal): Original file imports
# ``from hsm_v2_kernel import ...``. We rewrite to a package-relative
# import so the vendored copy is self-contained and does NOT accidentally
# resolve to Paper 3A's live ``code/experiments/hsm_v2_kernel.py`` (which
# may drift under Paper 3A revision). This is the ONLY semantic edit to
# the vendored file; everything else is byte-identical to v5.0.0.
from .hsm_v2_kernel import (
    W0 as _W0,
    sa_v2,
    sp_v2,
    sr_v2,
    st_v2,
    sv_v2,
)


# ---------------------------------------------------------------------------
# Public constants (kept in legacy "w_R/w_V/..." form for backward compat)
# ---------------------------------------------------------------------------
DWT_WAVELET     = "db4"
DWT_LEVEL       = 3
SAX_ALPHABET    = 4
FASTDTW_RADIUS  = 3

DWT_BAND_WEIGHTS = {"cA3": 0.40, "cD3": 0.20, "cD2": 0.20, "cD1": 0.20}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_R": _W0["R"],
    "w_V": _W0["V"],
    "w_T": _W0["T"],
    "w_A": _W0["A"],
    "w_P": _W0["P"],
}

DEFAULT_THETA = 0.75


# ---------------------------------------------------------------------------
# Data structures (unchanged from the legacy module so callers don't break)
# ---------------------------------------------------------------------------
@dataclass
class QueryFeatures:
    """Features extracted from a single SQL query."""
    raw_sql:   str
    template:  str             = ""
    tables:    set             = field(default_factory=set)
    columns:   set             = field(default_factory=set)
    timestamp: Optional[float] = None


@dataclass
class WorkloadWindow:
    """A sliding window of queries representing a workload phase.

    Aggregates per-query features into the constructs S_R/S_V/S_T/S_A/S_P
    operate on:
        * frequency Counter over distinct templates  (S_R, S_T)
        * raw query count                            (S_V, paper-compliant)
        * aggregated table / column sets             (S_A)
        * per-bucket arrival series q(t)             (S_P, paper §III-B)
    """
    queries:    List[QueryFeatures] = field(default_factory=list)
    window_id:  int = 0
    duration_s: float = 1.0

    # ----- aggregated views ------------------------------------------------
    @property
    def template_freq(self) -> Counter:
        return Counter(q.template for q in self.queries)

    @property
    def count(self) -> int:
        """Query count per window (paper §III-B "Volume Q")."""
        return len(self.queries)

    @property
    def qps(self) -> float:
        """Legacy queries-per-second view; retained for diagnostics only."""
        if self.duration_s <= 0:
            return 0.0
        return len(self.queries) / float(self.duration_s)

    @property
    def tables(self) -> set:
        return set().union(*(q.tables for q in self.queries)) if self.queries else set()

    @property
    def columns(self) -> set:
        return set().union(*(q.columns for q in self.queries)) if self.queries else set()

    def arrival_series(self, n_buckets: int = 64) -> np.ndarray:
        """Bucketed arrival count series q(t) for S_P (paper line 291).

        Equal-width buckets across the window (need >= 2**DWT_LEVEL = 8 to
        admit a level-3 db4 decomposition; default 64 matches Lemma 4).
        """
        if not self.queries or self.duration_s <= 0:
            return np.zeros(n_buckets, dtype=float)
        timestamps = np.array(
            [q.timestamp if q.timestamp is not None else 0.0 for q in self.queries],
            dtype=float,
        )
        t_min, t_max = timestamps.min(), timestamps.max()
        if t_max == t_min:
            arr = np.zeros(n_buckets, dtype=float)
            arr[0] = float(len(self.queries))
            return arr
        edges = np.linspace(t_min, t_max + 1e-9, n_buckets + 1)
        counts, _ = np.histogram(timestamps, bins=edges)
        return counts.astype(float)


# ---------------------------------------------------------------------------
# Feature extraction (PostgreSQL extractor; paper §III-B)
# ---------------------------------------------------------------------------
TPCH_TABLES = {
    "lineitem", "orders", "customer", "part", "partsupp",
    "supplier", "nation", "region",
}
JOB_TABLES = {
    "title", "movie_info", "movie_keyword", "movie_companies",
    "cast_info", "company_name", "info_type", "keyword",
    "kind_type", "name", "person_info", "role_type",
    "movie_link", "link_type", "char_name", "complete_cast",
    "comp_cast_type", "movie_info_idx", "aka_name", "aka_title",
    "company_type",
}
# pgbench TPC-B-style schema (used by Tier-2 OLTP / Burst).  The schema
# is fixed by ``pgbench -i`` so the table+column vocabularies are known
# a priori; enumerating them here keeps the relational extractor faithful
# to paper §III-B's "{table.column} sets derived from the query access
# map" without breaking the existing TPC-H/JOB detection.
PGBENCH_TABLES = {
    "pgbench_accounts", "pgbench_branches",
    "pgbench_tellers",  "pgbench_history",
}
PGBENCH_COLUMNS = {
    "aid", "bid", "tid", "abalance", "bbalance", "tbalance",
    "delta", "mtime", "filler",
}
KNOWN_TABLES = TPCH_TABLES | JOB_TABLES | PGBENCH_TABLES
KNOWN_COLUMNS = PGBENCH_COLUMNS  # extended in extract_features below


def canonicalize(sql: str) -> str:
    """Normalize literals to '?' to extract a query template.

    Recognises optional leading sign on numeric literals (``-1000`` →
    ``?``, not ``-?``) so canonicalised templates are stable under sign
    changes in WHERE clauses.  This matches what a real query planner
    does when extracting prepared-statement parameters.
    """
    s = sql.lower()
    s = re.sub(r"'[^']*'",                    "?", s)
    s = re.sub(r"(?<![\w.])-?\b\d+\.?\d*\b",  "?", s)
    s = re.sub(r"\s+",                        " ", s).strip()
    return s


def extract_features(sql: str, timestamp: Optional[float] = None) -> QueryFeatures:
    """Extract HSM-relevant features from a SQL string.

    Column extraction happens in three layers (paper §III-B "{table.column}
    sets derived from the query access map"):

      (1) qualified ``table.column`` references — works for any schema.
      (2) TPC-H / JOB style prefix-keyed columns (``l_``, ``o_``, ``c_``,
          ``p_``, ``ps_``, ``s_``, ``n_``, ``r_``).
      (3) Schemas whose column names have no consistent prefix (pgbench
          TPC-B: ``aid``, ``bid``, ``abalance``, …) are enumerated in
          ``KNOWN_COLUMNS`` and matched as whole words.

    Layer 3 was added on 2026-04-15 to close an extractor drift that left
    pgbench windows with ∅ table+column sets — Jaccard(∅,∅)=1 collapsed
    S_A to a constant and prevented HSM from sensing phase changes in
    Tier-2 OLTP/Burst experiments.
    """
    sql_lower = sql.lower()
    qf = QueryFeatures(raw_sql=sql, timestamp=timestamp)
    qf.template = canonicalize(sql)
    for tbl in KNOWN_TABLES:
        if re.search(r"\b" + re.escape(tbl) + r"\b", sql_lower):
            qf.tables.add(tbl)
    # Layer 1+2: qualified refs OR TPC-H/JOB prefix patterns.
    for m in re.findall(
        r"\b([a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*)\b|"
        r"\b(l_\w+|o_\w+|c_\w+|p_\w+|ps_\w+|s_\w+|n_\w+|r_\w+)\b",
        sql_lower,
    ):
        col = m[0] or m[1]
        if col:
            qf.columns.add(col)
    # Layer 3: enumerated pgbench columns (whole-word match so we don't
    # accidentally pick up substrings inside other identifiers).
    for col in KNOWN_COLUMNS:
        if re.search(r"\b" + re.escape(col) + r"\b", sql_lower):
            qf.columns.add(col)
    return qf


def build_window(
    sql_list: List[str],
    timestamps: Optional[List[float]] = None,
    window_id: int = 0,
    duration_s: float = 1.0,
) -> WorkloadWindow:
    """Construct a WorkloadWindow from a list of SQL strings."""
    if timestamps is None:
        timestamps = list(range(len(sql_list)))
    feats = [extract_features(sql, ts) for sql, ts in zip(sql_list, timestamps)]
    return WorkloadWindow(queries=feats, window_id=window_id, duration_s=duration_s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _aligned_freq_vectors(
    f_a: Counter, f_b: Counter
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two frequency counters on their union vocabulary."""
    vocab = sorted(set(f_a) | set(f_b))
    if not vocab:
        return np.zeros(1), np.zeros(1)
    va = np.array([f_a.get(t, 0) for t in vocab], dtype=float)
    vb = np.array([f_b.get(t, 0) for t in vocab], dtype=float)
    return va, vb


# ---------------------------------------------------------------------------
# Five HSM dimensions  (delegated to hsm_v2_kernel; Lemma 1 holds there)
# ---------------------------------------------------------------------------
def s_r(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_R = 1 - arccos(rho_s) / pi  (paper Eq. 1, delegated)."""
    va, vb = _aligned_freq_vectors(w_a.template_freq, w_b.template_freq)
    return float(sr_v2(va, vb))


def s_v(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_V = min(Q_a, Q_b) / max(Q_a, Q_b)  (paper Eq. 2, count-based).

    Volume Q is the **query count per window** per paper §III-B Relational
    extractor.  The legacy qps-based path is preserved on
    ``WorkloadWindow.qps`` for diagnostics but is no longer the S_V input.
    """
    return float(sv_v2(w_a.count, w_b.count))


def s_t(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_T = 1 - (2/pi) arccos(v_hat_A . v_hat_B)  (paper Eq. 3, delegated).

    For relational extractors, ``v`` is the template-frequency vector (per
    paper §III-B "set of SQL query templates").  Aligning on the union
    vocabulary keeps the angular distance on a shared subspace.
    """
    va, vb = _aligned_freq_vectors(w_a.template_freq, w_b.template_freq)
    return float(st_v2(va, vb))


def s_a(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_A = 0.5 J(T_A, T_B) + 0.5 J(C_A, C_B)  (paper Eq. 4, delegated)."""
    return float(sa_v2(w_a.tables, w_b.tables, w_a.columns, w_b.columns))


def s_p(
    w_a: WorkloadWindow,
    w_b: WorkloadWindow,
    n_buckets: int = 64,
) -> float:
    """S_P  via DWT(db4, L=3) + SAX(alpha=4) + FastDTW(r=3)  (paper Eq. 5).

    Input is the bucketed arrival series q(t) per paper line 291 ("a QPS
    time series q(t) at 1-second resolution").  Delegates to ``sp_v2``.
    """
    q_a = w_a.arrival_series(n_buckets=n_buckets)
    q_b = w_b.arrival_series(n_buckets=n_buckets)
    qset_a = set(w_a.template_freq.keys())
    qset_b = set(w_b.template_freq.keys())
    return float(sp_v2(q_a, q_b, qset_a=qset_a, qset_b=qset_b))


# ---------------------------------------------------------------------------
# Composite score (paper Eq. 6, delegated)
# ---------------------------------------------------------------------------
def hsm_score(
    w_a: WorkloadWindow,
    w_b: WorkloadWindow,
    weights: Optional[Dict[str, float]] = None,
    n_buckets: int = 64,
) -> Tuple[float, Dict[str, float]]:
    """Compute composite HSM = sum_i w_i * S_i (paper Eq. 6).

    Calls each dimension separately so the legacy WorkloadWindow API stays
    semantic (template_freq, count, tables, ...) while the actual scoring
    is done by ``hsm_v2_kernel``.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    dims = {
        "S_R": s_r(w_a, w_b),
        "S_V": s_v(w_a, w_b),
        "S_T": s_t(w_a, w_b),
        "S_A": s_a(w_a, w_b),
        "S_P": s_p(w_a, w_b, n_buckets=n_buckets),
    }
    score = (
        w["w_R"] * dims["S_R"]
      + w["w_V"] * dims["S_V"]
      + w["w_T"] * dims["S_T"]
      + w["w_A"] * dims["S_A"]
      + w["w_P"] * dims["S_P"]
    )
    return float(score), dims


def compute_q_min(
    a: float,
    b: float,
    f: float,
    g: float,
    N: int,
) -> float:
    """Break-even query volume per paper Theorem 3.

    Q_min(N) = (a·N·log N + b) / (f·N − g·log N)

    where the cost-model constants come from Table (Cost Model):
        a : per-tuple index creation cost      (aN log N total)
        b : fixed index drop cost              (constant)
        f : per-query full-scan cost            (fN per query)
        g : per-query indexed-lookup cost       (g·log N per query)
        N : table cardinality (rows)

    All costs share the same time unit (seconds or ms); the ratio is
    dimensionless in Q. Q_min ≥ 1 is required for Theorem 3 to apply;
    otherwise the advisor path is unconditionally dominant.

    Returns +inf if the denominator is non-positive (degenerate cost
    model where full-scan is cheaper than indexed lookup — e.g. when
    f·N ≤ g·log N, meaning the workload is too small to benefit from
    any index).
    """
    if N <= 1:
        return float("inf")
    log_N = math.log(N)
    num = a * N * log_N + b
    den = f * N - g * log_N
    if den <= 0:
        return float("inf")
    return num / den


def optimal_theta(N: int, Q: float,
                  a: float, b: float, f: float, g: float) -> float:
    """Closed-form θ*(N,Q) per paper Theorem 3 (ii):

        θ*(N, Q) = 1 − Q_min(N) / Q

    Clipped to [0, 1]. Used for post-hoc analysis of whether HSM gating
    decisions match the economic optimum at the measured (N, Q) of each
    window. The runtime path uses DEFAULT_THETA = 0.75 as a workload-
    agnostic default; see paper §III Definition (Decision Threshold) and
    the Tier-2 C2 framing for why a static θ is retained in experiments.
    """
    q_min = compute_q_min(a, b, f, g, N)
    if Q <= 0:
        return 0.0
    theta_star = 1.0 - q_min / float(Q)
    return float(max(0.0, min(1.0, theta_star)))


def should_trigger_advisor(
    w_prev: WorkloadWindow,
    w_curr: WorkloadWindow,
    theta: float = DEFAULT_THETA,
    weights: Optional[Dict[str, float]] = None,
    n_buckets: int = 64,
) -> Tuple[bool, float, Dict[str, float]]:
    """HSM gating decision (Theorem 4).

    trigger=True  -> workload drift detected; rerun the index advisor.
    trigger=False -> workload stable; reuse current indexes.

    First window has no predecessor: drift is undefined, so the gate
    returns trigger=False with the self-similarity sentinel score=1.0,
    consistent with the Theorem 4 assumption that at least one reference
    window W_{i-1} exists before any gating decision is made.
    """
    if w_prev is None or not w_prev.queries:
        return False, 1.0, {}
    score, dims = hsm_score(w_prev, w_curr, weights=weights, n_buckets=n_buckets)
    return score < theta, score, dims


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sql_a = [
        "SELECT l_returnflag, SUM(l_quantity) FROM lineitem WHERE l_shipdate <= '1998-09-01' GROUP BY l_returnflag",
        "SELECT o_orderpriority, COUNT(*) FROM orders WHERE o_orderdate BETWEEN '1993-07-01' AND '1993-10-01' GROUP BY o_orderpriority",
    ] * 16
    sql_b = [
        "SELECT c_custkey, SUM(l_extendedprice * (1 - l_discount)) FROM customer, orders, lineitem WHERE c_custkey = o_custkey GROUP BY c_custkey",
        "SELECT n_name, SUM(l_extendedprice) FROM nation, supplier, lineitem WHERE s_nationkey = n_nationkey GROUP BY n_name",
    ] * 16
    ts_a = list(np.linspace(0.0, 60.0, len(sql_a)))
    ts_b = list(np.linspace(0.0, 30.0, len(sql_b)))
    w_a = build_window(sql_a, timestamps=ts_a, window_id=1, duration_s=60.0)
    w_b = build_window(sql_b, timestamps=ts_b, window_id=2, duration_s=30.0)
    score, dims = hsm_score(w_a, w_b)
    print(f"HSM score   : {score:.4f}")
    for k, v in dims.items():
        print(f"  {k} = {v:.4f}")
    trig, _, _ = should_trigger_advisor(w_a, w_b)
    print(f"trigger?    : {trig}")
