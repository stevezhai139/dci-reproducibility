"""
window_features.py — Cross-engine window-feature builder for HSM v2.

Mirrors `make_window_features` in
`experiments/v2_10seed/07_adaptation_comparison.py:202`, but pulls schema
maps (QUERY_TABLES, QUERY_FIELDS, ALL_QIDS_SORTED) from the cross-engine
Mongo workload module rather than from `01_run_tpch_10seeds`. The output
dict is the SAME shape so `common.hsm_bridge.compute_window_hsm[_breakdown]`
can consume it without engine-specific code.

A MySQL fallback runner can re-use this builder unchanged provided it
defines its own QUERY_TABLES / QUERY_FIELDS map and passes them in via
the optional `tables_map` and `fields_map` arguments.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional, Set

import numpy as np

# Default schema maps come from the Mongo workload package; engines that
# need a different mapping pass their own dicts in.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MONGO_WORKLOAD = os.path.abspath(
    os.path.join(_HERE, "..", "mongo", "workload")
)
if _MONGO_WORKLOAD not in sys.path:
    sys.path.insert(0, _MONGO_WORKLOAD)

from templates import (  # noqa: E402
    QUERY_TABLES as _DEFAULT_TABLES,
    QUERY_FIELDS as _DEFAULT_FIELDS,
    ALL_QIDS_SORTED as _DEFAULT_QIDS,
)


def make_window_features(
    qnames: Iterable[str],
    exec_times_ms: Iterable[float],
    *,
    all_qids: Optional[List[str]] = None,
    tables_map: Optional[dict] = None,
    fields_map: Optional[dict] = None,
) -> dict:
    """Build the canonical 6-key window-feature dict for hsm_v2.

    Args
    ----
    qnames : sequence of qid strings, IN EXECUTION ORDER. Length defines
             the window size; duplicates contribute to frequency mass.
    exec_times_ms : sequence of per-query wall-clock times in milliseconds.
                    Must be 1:1 with `qnames`.
    all_qids : optional override for the canonical sorted qid universe.
               Required when running on a non-Mongo template set so the
               frequency vector has a stable position-to-qid mapping.
    tables_map / fields_map : optional override of the per-qid schema maps.
                              Each is `Dict[qid, Set[str]]`.

    Returns
    -------
    A dict with keys exactly matching the contract in
    `common/hsm_bridge.py::compute_window_hsm_breakdown`:

        freq   : np.ndarray, shape (len(all_qids),), L1-normalised
        n      : int — total queries in the window
        tables : Set[str] — union of per-qid table sets
        cols   : Set[str] — union of per-qid field sets
        times  : np.ndarray of float64 — per-query exec_ms
        qset   : Set[str] — distinct qids that appeared in the window
    """
    qids = list(qnames)
    times = np.asarray(list(exec_times_ms), dtype=float)
    if len(qids) != len(times):
        raise ValueError(
            f"qnames/exec_times length mismatch: {len(qids)} vs {len(times)}"
        )

    universe = list(all_qids) if all_qids is not None else _DEFAULT_QIDS
    tmap = tables_map if tables_map is not None else _DEFAULT_TABLES
    fmap = fields_map if fields_map is not None else _DEFAULT_FIELDS

    freq = np.array(
        [float(sum(1 for q in qids if q == t)) for t in universe],
        dtype=float,
    )
    total = freq.sum()
    if total > 0:
        freq /= total

    tables: Set[str] = set()
    cols: Set[str] = set()
    for q in set(qids):
        tables |= tmap.get(q, set())
        cols |= fmap.get(q, set())

    return {
        "freq": freq,
        "n": len(qids),
        "tables": tables,
        "cols": cols,
        "times": times,
        "qset": set(qids),
    }


# ── Self-check ────────────────────────────────────────────────────────

def _self_check() -> None:
    print("window_features._self_check")
    rng = np.random.default_rng(0)
    qnames = ["Q1", "Q5", "Q5", "Q14", "Q22"]
    times = rng.uniform(5, 50, size=len(qnames))
    w = make_window_features(qnames, times)
    print(f"  n         = {w['n']}")
    print(f"  qset      = {sorted(w['qset'])}")
    print(f"  tables    = {sorted(w['tables'])}")
    print(f"  cols      = {sorted(w['cols'])}")
    print(f"  freq.sum  = {w['freq'].sum():.6f}")
    print(f"  freq.shape= {w['freq'].shape}")
    assert w["n"] == 5
    assert abs(w["freq"].sum() - 1.0) < 1e-9
    assert w["freq"].shape[0] == len(_DEFAULT_QIDS)
    # Q1+Q5 → spatial,edge ; Q14 → textual ; Q22 → all 5 types
    assert w["tables"] == {"spatial", "edge", "changeset", "textual", "review"}
    print("OK")


if __name__ == "__main__":
    _self_check()
