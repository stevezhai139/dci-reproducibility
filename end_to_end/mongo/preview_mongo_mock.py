#!/usr/bin/env python3
"""
PRE-RUN PREVIEW for the Paper 3D MongoDB arm — runs the REAL harness
(mongo_adaptation.run_block, the DCI/HSM gate, the ESR advisor, the error
handling, the CSV outputs) against a MOCK MongoDB engine with a synthetic but
realistic cost model. NO mongod needed.

Purpose (what Steve asked):
  (1) prove the wiring works end-to-end before the real run,
  (2) show the DIRECTION of results under a plausible cost model,
  (3) confirm alignment with the research framework (5 strategies, paired-RCB,
      drift_truth, advisor-necessity under --baseline under).

IMPORTANT: numbers here come from the SYNTHETIC cost model below, NOT real
MongoDB. They are for wiring + direction only. Real magnitudes come from the
live run on the Mac.

Cost model (scaled down so a 2-block preview runs in seconds; RATIOS realistic):
  - covered query (right index exists)        : 1 ms
  - uncovered query (collection scan)         : 8 ms   (~8x, cf JOB EXPLAIN ~5x)
  - $text query with NO text index            : ERROR  (MongoDB requires one)
  - btree index build                         : 20 ms
  - text index build                          : 120 ms (expensive → churn cost)
"""
import os, sys, time, random

HARNESS_DIR = "/sessions/blissful-awesome-carson/mnt/Research Papers/Paper 3 /Paper 3D/code/end_to_end/mongo"
WORKLOAD    = "/sessions/blissful-awesome-carson/mnt/Research Papers/Paper 3 /Paper 3D/code/cross_engine/mongo/workload"
COMMON      = "/sessions/blissful-awesome-carson/mnt/Research Papers/Paper 3 /Paper 3D/code/cross_engine/common"
for p in (HARNESS_DIR, WORKLOAD, COMMON):
    sys.path.insert(0, p)

import esr_recommender as esr

# ── synthetic cost model (ms) ──
FAST_MS, SCAN_MS = 1.0, 8.0
BTREE_BUILD_MS, TEXT_BUILD_MS = 20.0, 120.0
MS = 1e-3

def _needed_index(pipeline):
    """The per-query ideal index (reuse the ESR logic the advisor uses)."""
    return esr.recommend_for_query(pipeline)

def _is_text_query(pipeline):
    for st in pipeline:
        m = st.get("$match", {})
        if isinstance(m, dict) and "$text" in m:
            return True
    return False

def _covered(needed, existing_keys):
    """needed is a prefix of some existing index key (text matched by kind)."""
    if needed is None:
        return True
    if any(k in ("text", "2dsphere") for _, k in needed):
        want = [k for _, k in needed if k in ("text", "2dsphere")][0]
        return any(any(kk == want for _, kk in key) for key in existing_keys)
    n = [(f, k) for f, k in needed]
    for key in existing_keys:
        if [(f, k) for f, k in key][:len(n)] == n:
            return True
    return False


class FakeColl:
    def __init__(self):
        self._idx = {"_id_": {"key": [("_id", 1)]}}
    # index admin --------------------------------------------------------
    def create_index(self, keys, name=None):
        key = [(f, k) for f, k in keys]
        nm = name or "idx_" + "_".join(f"{f}{k}" for f, k in key)
        is_text = any(k == "text" for _, k in key)
        time.sleep((TEXT_BUILD_MS if is_text else BTREE_BUILD_MS) * MS)  # build cost
        self._idx[nm] = {"key": key}
        return nm
    def drop_index(self, name):
        self._idx.pop(name, None)
    def index_information(self):
        return {n: {"key": list(v["key"])} for n, v in self._idx.items()}
    def estimated_document_count(self):
        return 12_000_000
    # query --------------------------------------------------------------
    def aggregate(self, pipeline, allowDiskUse=False):
        existing = [v["key"] for v in self._idx.values()]
        if _is_text_query(pipeline) and not _covered([("_", "text")], existing):
            raise Exception("planner returned error :: caused by :: "
                            "need a text index to satisfy a $text query")
        needed = _needed_index(pipeline)
        time.sleep((FAST_MS if _covered(needed, existing) else SCAN_MS) * MS)
        return []


class FakeDB:
    def __init__(self): self._c = FakeColl()
    def __getitem__(self, _): return self._c

class FakeClient:
    def __init__(self): self._db = FakeDB()
    def __getitem__(self, _): return self._db
    class _Admin:
        def command(self, *a, **k): return {"ok": 1}
    admin = _Admin()
    def close(self): pass


def run_preview(argv, label):
    import importlib
    import mongo_adaptation as M
    importlib.reload(M)
    M.connect = lambda uri: FakeClient()          # monkeypatch the engine
    sys.argv = ["mongo_adaptation.py"] + argv
    print("\n" + "#"*78 + f"\n# PREVIEW: {label}\n# argv: {' '.join(argv)}\n" + "#"*78)
    rc = M.main()
    return rc

def summarize(subdir_glob):
    import glob, csv, statistics as st
    base = os.path.join(HARNESS_DIR, "out", subdir_glob)
    dirs = sorted(glob.glob(base + "/*"))
    if not dirs:
        print("  (no output dir found)"); return
    d = dirs[-1]
    bm = os.path.join(d, "block_metrics.csv")
    rows = list(csv.DictReader(open(bm)))
    by = {}
    for r in rows:
        by.setdefault(r["strategy"], []).append(r)
    order = ["no_advisor", "always_on", "periodic", "hsm_gated", "dci_gated"]
    print(f"\n  results dir: ...{d[-40:]}")
    print(f"  {'strategy':12} {'wall_qps':>9} {'adv_calls':>9} {'q_failed':>9} {'T_A_ms':>9}")
    for s in order:
        rs = by.get(s, [])
        if not rs: continue
        qps = st.mean(float(r["wall_qps"]) for r in rs)
        adv = st.mean(float(r["advisor_calls"]) for r in rs)
        qf  = st.mean(float(r.get("queries_failed", 0)) for r in rs)
        ta  = st.mean(float(r["T_A_total_ms"]) for r in rs)
        print(f"  {s:12} {qps:9.3f} {adv:9.1f} {qf:9.1f} {ta:9.1f}")
    qe = os.path.join(d, "query_errors.csv")
    if os.path.exists(qe):
        ers = list(csv.DictReader(open(qe)))
        tot = sum(int(r["count"]) for r in ers)
        cls = {}
        for r in ers: cls[r["error_class"]] = cls.get(r["error_class"], 0) + int(r["count"])
        print(f"  query_errors.csv: {tot} failed queries, by class {cls}")
    else:
        print("  query_errors.csv: none (no failures)")


if __name__ == "__main__":
    t0 = time.time()
    # headline: under-provisioned + irregular long seasons + ESR advisor
    run_preview(["--schedule", "irregular", "--advisor", "esr", "--baseline", "under",
                 "--n-windows", "24", "--blocks", "2",
                 "--results-subdir", "preview_under"], "B = --baseline under (headline)")
    summarize("preview_under")
    # contrast: well-provisioned baseline (A) — for comparison only
    run_preview(["--schedule", "irregular", "--advisor", "esr", "--baseline", "full",
                 "--n-windows", "24", "--blocks", "2",
                 "--results-subdir", "preview_full"], "A = --baseline full (contrast)")
    summarize("preview_full")
    print(f"\n[preview finished in {time.time()-t0:.1f}s — SYNTHETIC cost model, wiring+direction only]")
