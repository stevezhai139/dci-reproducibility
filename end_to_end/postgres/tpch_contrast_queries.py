"""tpch_contrast_queries.py — B-population for the tpch_contrast live schedule.

X0r design (contrast_preflight, 2026-08-12): the canary population is a set of
PLAN-EQUIVALENT REWRITES of the five steady templates. New template identities
move S_R/S_T at the mixture fraction; the table/column footprint (S_A) and the
plans/exec times (S_P) are identical by construction. Live semantics: an
ORM/framework upgrade rolls out rewritten SQL for the same application logic
to a small canary share of traffic.

Rewrite classes used (all PostgreSQL-plan-equivalent):
  conjunct reordering | BETWEEN <-> paired inequalities | IN-list <-> OR chain
  comma-join <-> explicit JOIN | ORDER BY position <-> name | DATE literal cast
"""
from tpch_queries import QUERIES as _Q0, QUERY_TABLES as _T0, QUERY_COLS as _C0

QUERIES = dict(_Q0)
QUERY_TABLES = {k: set(v) for k, v in _T0.items()}
QUERY_COLS = {k: set(v) for k, v in _C0.items()}

_REWRITES = {
    "Q1v2": """
SELECT l_returnflag, l_linestatus,
  SUM(l_quantity), SUM(l_extendedprice*(1-l_discount)), COUNT(*)
FROM lineitem WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus
""",
    "Q6v2": """
SELECT SUM(l_extendedprice*l_discount) AS rev
FROM lineitem
WHERE l_quantity < 24
  AND l_discount >= 0.05 AND l_discount <= 0.07
  AND l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'
""",
    "Q14v2": """
SELECT 100.0*SUM(CASE WHEN p_type LIKE 'PROMO%'
  THEN l_extendedprice*(1-l_discount) ELSE 0 END)
  /SUM(l_extendedprice*(1-l_discount))
FROM part JOIN lineitem ON p_partkey = l_partkey
WHERE l_shipdate < '1995-10-01' AND l_shipdate >= '1995-09-01'
""",
    "Q3v2": """
SELECT l_orderkey,
  SUM(l_extendedprice*(1-l_discount)) AS rev,
  o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_custkey = o_custkey AND o_orderkey = l_orderkey
  AND l_shipdate > '1995-03-15' AND o_orderdate < '1995-03-15'
  AND c_mktsegment = 'BUILDING'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY rev DESC LIMIT 10
""",
    "Q12v2": """
SELECT l_shipmode,
  SUM(CASE WHEN o_orderpriority IN ('2-HIGH','1-URGENT') THEN 1 ELSE 0 END),
  SUM(CASE WHEN o_orderpriority NOT IN ('2-HIGH','1-URGENT') THEN 1 ELSE 0 END)
FROM orders JOIN lineitem ON l_orderkey = o_orderkey
WHERE l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01'
  AND l_shipdate < l_commitdate AND l_commitdate < l_receiptdate
  AND (l_shipmode = 'MAIL' OR l_shipmode = 'SHIP')
GROUP BY l_shipmode ORDER BY l_shipmode
""",
}
for _k, _sql in _REWRITES.items():
    _b = _k[:-2]
    QUERIES[_k] = _sql
    QUERY_TABLES[_k] = set(_T0[_b])
    QUERY_COLS[_k] = set(_C0[_b])
