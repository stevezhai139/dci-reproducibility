"""
tpch_queries.py
================
Paper 3D -- Phase 3.  TPC-H query templates + table/column metadata.
Vendored verbatim from Paper 3A v2_10seed/01_run_tpch_10seeds.py
(QUERIES / QUERY_TABLES / QUERY_COLS, lines 57-202) -- the per-engine
adaptation harness imports these as the TPC-H workload definition.
"""
from __future__ import annotations

QUERIES = {
"Q1": """SELECT l_returnflag, l_linestatus,
  SUM(l_quantity), SUM(l_extendedprice*(1-l_discount)), COUNT(*)
FROM lineitem WHERE l_shipdate <= '1998-09-02'
GROUP BY l_returnflag, l_linestatus ORDER BY 1,2""",

"Q3": """SELECT l_orderkey,
  SUM(l_extendedprice*(1-l_discount)) AS rev,
  o_orderdate, o_shippriority
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON l_orderkey=o_orderkey
WHERE c_mktsegment='BUILDING'
  AND o_orderdate<'1995-03-15' AND l_shipdate>'1995-03-15'
GROUP BY l_orderkey,o_orderdate,o_shippriority
ORDER BY rev DESC LIMIT 10""",

"Q4": """SELECT o_orderpriority, COUNT(*) AS cnt
FROM orders WHERE o_orderdate>='1993-07-01' AND o_orderdate<'1993-10-01'
  AND EXISTS(SELECT 1 FROM lineitem
             WHERE l_orderkey=o_orderkey AND l_commitdate<l_receiptdate)
GROUP BY o_orderpriority ORDER BY 1""",

"Q5": """SELECT n_name, SUM(l_extendedprice*(1-l_discount)) AS rev
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON l_orderkey=o_orderkey
JOIN supplier ON l_suppkey=s_suppkey AND s_nationkey=c_nationkey
JOIN nation ON n_nationkey=s_nationkey
JOIN region ON r_regionkey=n_regionkey
WHERE r_name='ASIA' AND o_orderdate>='1994-01-01' AND o_orderdate<'1995-01-01'
GROUP BY n_name ORDER BY rev DESC""",

"Q6": """SELECT SUM(l_extendedprice*l_discount) AS rev
FROM lineitem
WHERE l_shipdate>='1994-01-01' AND l_shipdate<'1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity<24""",

"Q7": """SELECT n1.n_name, n2.n_name, EXTRACT(year FROM l_shipdate),
  SUM(l_extendedprice*(1-l_discount))
FROM supplier JOIN lineitem ON s_suppkey=l_suppkey
JOIN orders ON o_orderkey=l_orderkey
JOIN customer ON c_custkey=o_custkey
JOIN nation n1 ON s_nationkey=n1.n_nationkey
JOIN nation n2 ON c_nationkey=n2.n_nationkey
WHERE ((n1.n_name='FRANCE' AND n2.n_name='GERMANY')
    OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))
  AND l_shipdate BETWEEN '1995-01-01' AND '1996-12-31'
GROUP BY 1,2,3 ORDER BY 1,2,3""",

"Q10": """SELECT c_custkey, c_name,
  SUM(l_extendedprice*(1-l_discount)), c_acctbal, n_name
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON l_orderkey=o_orderkey
JOIN nation ON c_nationkey=n_nationkey
WHERE o_orderdate>='1993-10-01' AND o_orderdate<'1994-01-01'
  AND l_returnflag='R'
GROUP BY c_custkey,c_name,c_acctbal,c_phone,n_name,c_address,c_comment
ORDER BY 3 DESC LIMIT 20""",

"Q11": """SELECT ps_partkey, SUM(ps_supplycost*ps_availqty) AS val
FROM partsupp JOIN supplier ON ps_suppkey=s_suppkey
JOIN nation ON s_nationkey=n_nationkey
WHERE n_name='GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost*ps_availqty)>(
  SELECT SUM(ps_supplycost*ps_availqty)*0.0001
  FROM partsupp JOIN supplier ON ps_suppkey=s_suppkey
  JOIN nation ON s_nationkey=n_nationkey WHERE n_name='GERMANY')
ORDER BY val DESC LIMIT 20""",

"Q12": """SELECT l_shipmode,
  SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END),
  SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END)
FROM orders JOIN lineitem ON o_orderkey=l_orderkey
WHERE l_shipmode IN ('MAIL','SHIP') AND l_commitdate<l_receiptdate
  AND l_shipdate<l_commitdate AND l_receiptdate>='1994-01-01'
  AND l_receiptdate<'1995-01-01'
GROUP BY l_shipmode ORDER BY l_shipmode""",

"Q14": """SELECT 100.0*SUM(CASE WHEN p_type LIKE 'PROMO%'
  THEN l_extendedprice*(1-l_discount) ELSE 0 END)
  /SUM(l_extendedprice*(1-l_discount))
FROM lineitem JOIN part ON l_partkey=p_partkey
WHERE l_shipdate>='1995-09-01' AND l_shipdate<'1995-10-01'""",

"Q17": """SELECT SUM(l_extendedprice)/7.0
FROM lineitem JOIN part ON p_partkey=l_partkey
WHERE p_brand='Brand#23' AND p_container='MED BOX'
  AND l_quantity<(SELECT 0.2*AVG(l_quantity)
                  FROM lineitem WHERE l_partkey=p_partkey)""",

"Q18": """SELECT c_name,c_custkey,o_orderkey,o_orderdate,
  o_totalprice, SUM(l_quantity)
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON o_orderkey=l_orderkey
WHERE o_orderkey IN(SELECT l_orderkey FROM lineitem
  GROUP BY l_orderkey HAVING SUM(l_quantity)>300)
GROUP BY 1,2,3,4,5 ORDER BY o_totalprice DESC LIMIT 10""",
}

QUERY_TABLES = {
    'Q1':  {'lineitem'},
    'Q3':  {'customer','orders','lineitem'},
    'Q4':  {'orders','lineitem'},
    'Q5':  {'customer','orders','lineitem','supplier','nation','region'},
    'Q6':  {'lineitem'},
    'Q7':  {'supplier','lineitem','orders','customer','nation'},
    'Q10': {'customer','orders','lineitem','nation'},
    'Q11': {'partsupp','supplier','nation'},
    'Q12': {'orders','lineitem'},
    'Q14': {'lineitem','part'},
    'Q17': {'lineitem','part'},
    'Q18': {'customer','orders','lineitem'},
}

QUERY_COLS = {
    'Q1':  {'lineitem.l_shipdate','lineitem.l_returnflag','lineitem.l_linestatus',
            'lineitem.l_quantity','lineitem.l_extendedprice','lineitem.l_discount','lineitem.l_tax'},
    'Q3':  {'customer.c_mktsegment','customer.c_custkey','orders.o_custkey',
            'orders.o_orderdate','orders.o_shippriority','lineitem.l_orderkey',
            'lineitem.l_shipdate','lineitem.l_extendedprice','lineitem.l_discount'},
    'Q4':  {'orders.o_orderdate','orders.o_orderpriority','orders.o_orderkey',
            'lineitem.l_orderkey','lineitem.l_commitdate','lineitem.l_receiptdate'},
    'Q5':  {'customer.c_custkey','customer.c_nationkey','orders.o_custkey','orders.o_orderdate',
            'lineitem.l_orderkey','lineitem.l_suppkey','lineitem.l_extendedprice','lineitem.l_discount',
            'supplier.s_suppkey','supplier.s_nationkey','nation.n_nationkey','nation.n_name',
            'region.r_regionkey','region.r_name'},
    'Q6':  {'lineitem.l_shipdate','lineitem.l_discount','lineitem.l_quantity','lineitem.l_extendedprice'},
    'Q7':  {'supplier.s_suppkey','supplier.s_nationkey','lineitem.l_suppkey','lineitem.l_orderkey',
            'lineitem.l_shipdate','lineitem.l_extendedprice','lineitem.l_discount',
            'orders.o_orderkey','orders.o_custkey','customer.c_custkey','customer.c_nationkey',
            'nation.n_nationkey','nation.n_name'},
    'Q10': {'customer.c_custkey','customer.c_name','customer.c_acctbal','customer.c_nationkey',
            'orders.o_custkey','orders.o_orderdate','orders.o_orderkey',
            'lineitem.l_orderkey','lineitem.l_returnflag','lineitem.l_extendedprice','lineitem.l_discount',
            'nation.n_nationkey','nation.n_name'},
    'Q11': {'partsupp.ps_suppkey','partsupp.ps_supplycost','partsupp.ps_availqty','partsupp.ps_partkey',
            'supplier.s_suppkey','supplier.s_nationkey','nation.n_nationkey','nation.n_name'},
    'Q12': {'orders.o_orderkey','orders.o_orderpriority','lineitem.l_orderkey','lineitem.l_shipmode',
            'lineitem.l_commitdate','lineitem.l_receiptdate','lineitem.l_shipdate'},
    'Q14': {'lineitem.l_partkey','lineitem.l_shipdate','lineitem.l_extendedprice','lineitem.l_discount',
            'part.p_partkey','part.p_type'},
    'Q17': {'lineitem.l_partkey','lineitem.l_quantity','lineitem.l_extendedprice',
            'part.p_partkey','part.p_brand','part.p_container'},
    'Q18': {'customer.c_name','customer.c_custkey','orders.o_orderkey','orders.o_orderdate',
            'orders.o_totalprice','lineitem.l_orderkey','lineitem.l_quantity'},
}
