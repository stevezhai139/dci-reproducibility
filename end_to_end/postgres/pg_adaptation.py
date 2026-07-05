"""
07_adaptation_comparison.py
===========================
4-way adaptation strategy comparison — RCB design (paper A13, Table throughput).

Theory mapping (Option D — 5 Theorems / 4 Lemmas):
  - T3 : `wall_qps`, `advisor_calls`, `T_A_total_ms` are the empirical
         counterparts of the cost-model quantities {a,b,f,g,c}. θ=0.75 is
         hardcoded (T3 G2 — known gap; θ*(N,Q) supplementary check pending).
  - T4 : `hsm_series` + `phase_series` + `precision`/`recall` per block-strategy
         provide the stratified detector-quality inputs for the post-processor
         `report_detector_quality.py` (joint Hoeffding band over (TPR̂, TNR̂)).
  - T5 : `wall_qps` for `hsm_gated` and `always_on` and the empirical
         `p_advisor = advisor_calls / N_WINDOWS` feed the deployment
         certificate Speedup_∞ ≈ 1/(1 − p̂_stable).
  - L4 : per-call HSM cost is independent of database cardinality; this
         script measures total wall-clock so L4 is implicit, not asserted.

Strategies:
  no_advisor : Run queries without index management (baseline)
  always_on  : Invoke index advisor every window
  periodic   : Invoke advisor every K windows (K=3)
  hsm_gated  : Invoke advisor only when HSM(w_i, w_{i-1}) < θ (θ=0.75)

Design:
  10-block paired Randomised Complete Block (RCB) per SF.
  Within each block, all four strategies run on the SAME workload
  realisation (identical init queries, identical per-window query draws,
  identical Poisson inter-arrival delays); the strategy *order* is the
  only thing the Latin-square shuffle counterbalances. This is the
  paired-RCB pairing fixed on 2026-04-09 (see PAIRED RCB FIX in main()).
  Each block = 1 complete workload sequence (4 phases × 6 windows = 24 windows).

Window 0 provisioning (design — fixed 2026-04-09):
  Window 0 is an **initialisation** window: the same set of init queries
  runs under every strategy (deterministic via the paired-RCB seed) so
  that the buffer cache reaches a comparable warm state before
  measurement starts. Window 0 query latencies are NOT counted in
  `wall_qps` — measurement begins at window 1.

  At the END of window 0, the strategy decides whether to provision the
  test index `ix_advisor_bench`:
    - no_advisor : NO call to invoke_advisor → `ix_advisor_bench`
                   does not exist for windows 1..24. This is the true
                   index-free baseline.
    - always_on  : invoke_advisor → index exists
    - periodic   : invoke_advisor → index exists
    - hsm_gated  : invoke_advisor → index exists

  The pre-existing realistic indexes (idx_li_shipdate, idx_li_orderkey,
  …) are created once per SF by `ensure_indexes()` and exist for ALL
  strategies. Only `ix_advisor_bench` is gated by strategy.

  At end of block, `cleanup_advisor_index()` drops `ix_advisor_bench`
  unconditionally so the next strategy starts from a clean slate.

Metrics per (SF, block, strategy):
  wall_qps        : total_queries / wall_time  (including advisor overhead)
  advisor_calls   : number of advisor invocations
  T_A_total_ms    : cumulative advisor overhead
  queries_total   : total queries executed
  wall_time_s     : total wall-clock time

Poisson arrival:
  Inter-query delay ~ Exp(1/λ) where λ varies by phase to create realistic
  workload variance.  Steady phases: λ=50 q/s, Heavy phases: λ=30 q/s.
  This adds ~0.02s/query average delay (realistic for OLTP/OLAP mix).

Usage:
  cd "Version 2/code"
  python experiments/v2_10seed/07_adaptation_comparison.py

  # Test SF=0.2 only:
  python experiments/v2_10seed/07_adaptation_comparison.py --sf 0.2

  # All SFs:
  python experiments/v2_10seed/07_adaptation_comparison.py --sf 0.2 1.0 3.0 10.0
"""

import sys, os, time, random, math, argparse, hashlib, json, importlib
from datetime import datetime, timezone
# ── Paper 3D vendoring: re-point onto the Paper 3D code tree ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))     # code/end_to_end/postgres/
sys.path.insert(0, _HERE)                              # tpch_queries.py, job_queries.py, pg_workloads.py
sys.path.insert(0, os.path.join(_HERE, '..'))          # dci_gate.py  (code/end_to_end/)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'kernel'))   # canonical HSM kernel

# ── Workload selector (T3.4b — multi-workload support) ────────────────
# Pre-parse --workload so the rest of the module can load the right
# QUERIES / PHASES / ADVISOR config.  `parse_known_args` is used so the
# remaining args (--sf, --blocks, --no-poisson) still flow through to
# main()'s parser later.  Default = 'tpch' (the original Paper 3A workload).
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--workload', choices=['tpch', 'job'], default='tpch')
_pre_args, _ = _pre_parser.parse_known_args()
WORKLOAD = _pre_args.workload

from pg_workloads import WORKLOAD_CONFIGS                  # noqa: E402
_WC = WORKLOAD_CONFIGS[WORKLOAD]

# Dynamic workload-specific queries import (tpch_queries OR job_queries).
# Both modules export the same triple: QUERIES, QUERY_TABLES, QUERY_COLS.
_qmod = importlib.import_module(_WC['queries_module'])
QUERIES      = _qmod.QUERIES
QUERY_TABLES = _qmod.QUERY_TABLES
QUERY_COLS   = _qmod.QUERY_COLS

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  (workload-independent + workload dispatch)
# ══════════════════════════════════════════════════════════════
PG_BASE = dict(
    host='localhost', port=5432,
    user=__import__('os').environ.get('PGUSER','postgres'),
    password='', dbname='postgres',
)

# Workload-specific (looked up from pg_workloads.WORKLOAD_CONFIGS).
SF_DB_MAP      = _WC['sf_db_map']
PHASES         = _WC['phases']
LAMBDA_MAP     = _WC['lambda_map']
ADVISOR_INDEX  = _WC['advisor_index_name']
ADVISOR_DDL_CR = _WC['advisor_create_ddl']
ADVISOR_DDL_DR = _WC['advisor_drop_ddl']
VERIFY_TABLE   = _WC['verify_table']
WARMUP_QIDS    = _WC['warmup_qids']
HAS_SF_AXIS    = _WC['has_sf_axis']

# Workload-independent constants.
N_BLOCKS   = 10                    # RCB blocks
N_WINDOWS  = 24                    # 4 phases × 6 windows per phase
WIN_PER_PH = 6
QUERIES_PW = 20                    # queries per window
THETA      = 0.75                  # HSM gating threshold (placeholder; T3.6 calibrates)
K_PERIODIC = 3                     # periodic interval
STRATEGIES = ['dci_gated']  # Paper 3C repo: DCI-routed gate only

# DCI gate (Paper 3D RQ5) -- the engine-free gate, head-to-head against
# the HSM-composite theta-gate (`hsm_gated`). One run yields both, paired.
DCI_TAU   = float(os.environ.get('DCI_TAU', 1.5))  # 3C: env-overridable (RawDCIGate)
DCI_ALPHA = 0.05    # gate false-positive rate (RQ5 sweeps {0.01,0.05,0.10})
DCI_N_CAL = 64      # steady windows in the calibration pass (m ~ 64)

RESULTS_DIR = Path(__file__).parent / 'out'
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  Kernel + DCI gate imports  (Paper 3D vendoring)
# ══════════════════════════════════════════════════════════════
# HSM kernel -- the single canonical kernel (code/kernel/), NOT the V3
# `hsm_v2_core`. `hsm_v2` has the identical 12-arg signature, so the
# compute_hsm_breakdown() call site is unchanged; only S_P differs (the
# canonical kernel's symmetrised FastDTW -- see cross_engine/PROVENANCE.md).
from hsm_v2_kernel import hsm_v2, sr_v2, st_v2, sv_v2, sa_v2, sp_v2, W0

# DCI gate -- the engine-agnostic gate under test (Paper 3D RQ5).
from dci_gate import DCIGate  # Paper 3C: DCI-routed gate (routes on the raw participation ratio)


def ts() -> str:
    """ISO-8601 UTC timestamp for per-window progress logs (T3.4c, Phase 1 §5)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ══════════════════════════════════════════════════════════════
#  Database helpers
# ══════════════════════════════════════════════════════════════

def connect(dbname='postgres'):
    import psycopg2
    return psycopg2.connect(**{**PG_BASE, 'dbname': dbname})


def db_exists(dbname: str) -> bool:
    conn = connect('postgres')
    cur  = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
    found = cur.fetchone() is not None
    cur.close(); conn.close()
    return found


def ensure_indexes(dbname: str):
    """Create the workload-specific backbone of secondary indexes (the
    "PK + FK structural baseline" shared by every strategy — see
    PHASE3_PLAN.md §5).  The DDL list is workload-dependent and lives in
    `pg_workloads.WORKLOAD_CONFIGS[workload]['backbone_indexes']`."""
    conn = connect(dbname); conn.autocommit = True; cur = conn.cursor()
    # FIX 2026-06-18: Dexter requires HypoPG; ensure it exists on EVERY db so the
    # advisor is never a silent no-op (tpch_sf1 lacked it -> Dexter recommended
    # nothing -> false 'futile'/parity). Idempotent; needs no superuser (trusted ext).
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
    except Exception as _e:
        print(f"  [setup] WARN: could not ensure hypopg: {_e}")
    if os.environ.get('PROBE_PK_ONLY') == '1':
        # Under-indexed PK-only baseline probe: drop the FK backbone so the
        # only structures are the schema PRIMARY KEYs. Makes the advisor's
        # ix_advisor_bench potentially beneficial (advisor-valuable regime).
        # FIX 2026-06-18: the old name-list drop was a silent NO-OP for PG TPC-H.
        # backbone_indexes DDL names (idx_li_*) had drifted from the actual DB
        # index names (idx_lineitem_*), so DROP IF EXISTS matched nothing -> the
        # PK-only baseline never took effect (no_advisor kept the full backbone,
        # making the advisor look 'futile' -- a measurement artifact). Query the
        # catalog and drop EVERY backbone index (idx*) so it is robust to names.
        cur.execute("SELECT indexname FROM pg_indexes "
                    "WHERE schemaname='public' AND indexname LIKE 'idx%'")
        _drop = [r[0] for r in cur.fetchall()]
        for name in _drop:
            cur.execute(f'DROP INDEX IF EXISTS {name}')
        print(f'  [PROBE] PK-only baseline: dropped {len(_drop)} backbone '
              f'indexes {_drop}, not recreated')
    else:
        for idx in _WC['backbone_indexes']:
            cur.execute(idx)
    cur.execute("ANALYZE")
    cur.close(); conn.close()


def ensure_database(sf: float, dbname: str):
    if db_exists(dbname):
        try:
            conn = connect(dbname); cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {VERIFY_TABLE}")
            n = cur.fetchone()[0]; cur.close(); conn.close()
            if n > 0:
                print(f'  ✓ Database {dbname} exists ({n:,} {VERIFY_TABLE} rows)')
                ensure_indexes(dbname)
                return
        except Exception:
            pass
    raise RuntimeError(
        f"database {dbname} (SF={sf}) not found or has no {VERIFY_TABLE} rows. "
        f"Paper 3D expects the {WORKLOAD} workload pre-loaded on PostgreSQL — "
        f"load it first, then re-run.")


# ══════════════════════════════════════════════════════════════
#  Window feature extraction (lightweight, for HSM gating)
# ══════════════════════════════════════════════════════════════

ALL_TEMPLATES = sorted(QUERIES.keys())  # global template list


def make_window_features(qnames: List[str], exec_times: List[float]) -> dict:
    """Build feature dict for one window (for HSM comparison)."""
    freq = np.array([sum(1 for q in qnames if q == t) for t in ALL_TEMPLATES], dtype=float)
    total = freq.sum()
    if total > 0:
        freq /= total

    tables: Set[str] = set()
    cols:   Set[str] = set()
    for q in set(qnames):
        tables |= QUERY_TABLES.get(q, set())
        cols   |= QUERY_COLS.get(q,   set())

    return {
        'freq': freq, 'n': len(qnames),
        'tables': tables, 'cols': cols,
        'times': np.array(exec_times, dtype=float),
        'qset': set(qnames),
    }


def compute_hsm_breakdown(w_a: dict, w_b: dict) -> dict:
    """Compute full 5-D HSM breakdown {S_R, S_V, S_T, S_A, S_P, HSM}.

    This is the parity entry-point with Mongo's hsm_bridge.compute_window_hsm_breakdown.
    Persisting all 5 components per window is required by:
      - T2 witness-pair extraction (compute_t2_witness_pairs.py needs S_R..S_P)
      - T4 stratified detector quality (report_detector_quality.py reads HSM only,
        but downstream T2 analysis re-uses the same CSV)

    Fixed 2026-04-09 (audit BUG #4): previously only the scalar HSM was returned
    and the other 4 dimensions were silently discarded.
    """
    return hsm_v2(
        w_a['freq'], w_b['freq'],
        w_a['n'],    w_b['n'],
        w_a['tables'], w_b['tables'],
        w_a['cols'],   w_b['cols'],
        w_a['times'],  w_b['times'],
        w_a['qset'],   w_b['qset'],
    )


def compute_hsm(w_a: dict, w_b: dict) -> float:
    """Compute HSM scalar between two window feature dicts (back-compat shim)."""
    return compute_hsm_breakdown(w_a, w_b)['HSM']


# ══════════════════════════════════════════════════════════════
#  Advisor simulation (real CREATE/DROP INDEX)
# ══════════════════════════════════════════════════════════════

import subprocess as _subprocess
import tempfile as _tempfile
import re as _re

DEXTER_BIN = os.environ.get('DEXTER_BIN', 'dexter')


def _dexter_recommend(dbname, query_sqls):
    """Run Dexter (HypoPG-based ISP solver) recommend-only on `query_sqls`.
    Returns (list[(idxname, ddl)], recommend_ms).  Addendum 15.9."""
    qs = list(dict.fromkeys(q.strip().rstrip(';') for q in query_sqls if q.strip()))
    if not qs:
        return [], 0.0
    with _tempfile.NamedTemporaryFile('w', suffix='.sql', delete=False) as fh:
        for q in qs:
            fh.write(q + ';\n')
        path = fh.name
    cmd = [DEXTER_BIN, '-h', PG_BASE['host'], '-p', str(PG_BASE['port']),
           '-U', PG_BASE['user'], '-d', dbname, path, '--min-calls', '1']
    t0 = time.perf_counter()
    try:
        proc = _subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        text = (proc.stdout or '') + (proc.stderr or '')
    except Exception as e:                       # dexter missing / crash
        text = '__DEXTER_ERROR__ ' + str(e)
    finally:
        try: os.unlink(path)
        except OSError: pass
    rec_ms = (time.perf_counter() - t0) * 1000.0
    if '__DEXTER_ERROR__' in text:
        print('    [advisor] Dexter error: ' + text[:200], flush=True)
        return [], rec_ms
    recs = []
    for m in _re.finditer(r'Index found:\s*([^\s(]+)\s*\(([^)]+)\)', text):
        rel  = m.group(1).replace('"', '')
        cols = [c.strip().replace('"', '') for c in m.group(2).split(',')]
        tbl  = rel.split('.')[-1]
        name = _re.sub(r'[^0-9a-zA-Z_]', '_', 'ix_dexter_' + tbl + '_' + '_'.join(cols))[:63]
        ddl  = 'CREATE INDEX IF NOT EXISTS ' + name + ' ON ' + rel + ' (' + ', '.join(cols) + ')'
        recs.append((name, ddl))
    return recs, rec_ms


def invoke_advisor(conn, dbname, query_sqls, created_idx) -> float:
    """Invoke the real advisor (Dexter) on `query_sqls`; apply recommended
    indexes idempotently.  Returns total wall overhead (recommend+create) ms.
    `created_idx` (a set) accumulates index names for end-of-block cleanup.
    All strategies share this recommender; only invocation *timing* varies."""
    recs, rec_ms = _dexter_recommend(dbname, query_sqls)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '600s'")
    t0 = time.perf_counter()
    for name, ddl in recs:
        try:
            cur.execute(ddl); conn.commit(); created_idx.add(name)
        except Exception:
            conn.rollback()
    create_ms = (time.perf_counter() - t0) * 1000.0
    cur.execute("SET statement_timeout = '30s'")
    cur.close()
    return rec_ms + create_ms


def cleanup_advisor_index(conn, created_idx=None):
    """Drop all advisor-created (Dexter) indexes after a block."""
    cur = conn.cursor()
    names = set(created_idx or ())
    try:
        cur.execute("SELECT indexname FROM pg_indexes WHERE indexname LIKE %s",
                    ('ix_dexter_%',))
        names |= {r[0] for r in cur.fetchall()}
    except Exception:
        conn.rollback()
    for name in names:
        try:
            cur.execute('DROP INDEX IF EXISTS ' + name); conn.commit()
        except Exception:
            conn.rollback()
    cur.close()
    if created_idx is not None:
        created_idx.clear()


# ══════════════════════════════════════════════════════════════
#  Strategy decision
# ══════════════════════════════════════════════════════════════

def should_invoke(strategy: str, window_idx: int, hsm_score: float) -> bool:
    """
    Decide whether to invoke advisor for the current window.
    NOTE: window_idx here is 1-based (window 0 is the shared init phase
    handled separately in run_block, so this is never called for win=0).
    """
    # Paper 3C repo: only `dci_gated` is run, and it routes through
    # DCIGate.decide() -- not this scalar gate. The baseline policies of the
    # sibling economics study are intentionally omitted from this repository.
    return False


# ══════════════════════════════════════════════════════════════
#  Run one block for one strategy
# ══════════════════════════════════════════════════════════════

def calibrate_dci_gate(dbname: str, n_cal: int = DCI_N_CAL,
                       use_poisson: bool = True) -> DCIGate:
    """Calibration pass for the DCI gate (Paper 3D RQ5;
    T3.2_DCI_GATE_THEORY.md §11).

    Run `n_cal` STEADY windows -- phase 0 (Reporting), no drift --
    against the live engine, collect the per-window 5-D HSM feature
    vectors, and fit a DCIGate once. mu0 / Sigma0 / the F-thresholds
    are then FROZEN for the official blocks: they are the steady-state
    reference, and a moving reference would mask the drift it must
    detect. The kernel, DCI, routing and detector all stay live per
    window -- only the calibration is frozen. Returns the fitted gate.
    """
    rng = random.Random(20260525)
    np.random.seed(20260525)
    conn = connect(dbname); conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '30s'")
    ph0  = PHASES[0]                                  # the steady phase
    wts0 = np.array(ph0['w'], dtype=float); wts0 /= wts0.sum()

    def _steady_window() -> dict:
        qn = list(np.random.choice(ph0['qs'], size=QUERIES_PW,
                                   p=wts0, replace=True))
        ex = []
        for q in qn:
            if use_poisson:
                time.sleep(rng.expovariate(
                    LAMBDA_MAP.get(ph0['name'], 40.0)))
            t0 = time.perf_counter()
            try:
                cur.execute(QUERIES[q]); cur.fetchall()
            except Exception:
                conn.rollback()
            ex.append((time.perf_counter() - t0) * 1000)
        return make_window_features(qn, ex)

    prev = _steady_window()
    feats = []
    for _ in range(n_cal):
        cur_f = _steady_window()
        b = compute_hsm_breakdown(prev, cur_f)
        feats.append([b['S_R'], b['S_V'], b['S_T'], b['S_A'], b['S_P']])
        prev = cur_f
    cur.close(); conn.close()
    return DCIGate(tau=DCI_TAU, alpha=DCI_ALPHA).fit(
        np.asarray(feats, dtype=float))


def run_block(dbname: str, strategy: str, block_num: int,
              rng_seed: int, use_poisson: bool = True,
              dci_gate: DCIGate = None) -> dict:
    """
    Run 24 windows under one strategy. Returns metrics dict.

    Window 0 is an **initialisation window** shared by all strategies:
      - All strategies invoke the advisor once at window 0
      - This eliminates cold-start bias (HSM_gated has no previous window)
      - Metrics (wall_qps, advisor_calls, etc.) are counted from window 1 onward
      - Window 0 still builds prev_features for HSM computation at window 1
    """
    import psycopg2
    import hashlib
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # Workload fingerprint accumulator. We hash every query name as it is
    # drawn from np.random.choice; with the paired-RCB seed fix, all four
    # strategies in the same block should produce IDENTICAL fingerprints.
    # This is a defensive sanity check for §3 Finding A (RCB pairing).
    workload_fp = hashlib.sha256()

    conn = connect(dbname)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '30s'")

    # ── Window 0: initialisation (shared across all strategies) ──
    ph0 = PHASES[0]
    wts0 = np.array(ph0['w'], dtype=float); wts0 /= wts0.sum()
    qnames_init = list(np.random.choice(ph0['qs'], size=QUERIES_PW, p=wts0, replace=True))
    workload_fp.update(('|'.join(qnames_init) + '\n').encode())
    init_times = []
    for qname in qnames_init:
        if use_poisson:
            time.sleep(rng.expovariate(LAMBDA_MAP.get(ph0['name'], 40.0)))
        t0 = time.perf_counter()
        try:
            cur.execute(QUERIES[qname]); cur.fetchall()
        except Exception:
            conn.rollback()
        init_times.append((time.perf_counter() - t0) * 1000)

    prev_features = make_window_features(qnames_init, init_times)

    # ── Window 0 advisor provisioning (DESIGN FIX 2026-04-09) ──
    # Previously: ALL strategies invoked the advisor once at window 0,
    # which silently gave `no_advisor` the test index `ix_advisor_bench`
    # for the entire measurement window — turning "no_advisor" into
    # "advisor invoked once, then idle" rather than a true index-free
    # baseline. The DDL cost was hidden because invocation happened
    # before `wall_start`.
    #
    # New behaviour: only strategies that name themselves as advisor
    # users (always_on / periodic / hsm_gated) provision the test index
    # at window 0. `no_advisor` runs windows 1..24 WITHOUT the test
    # index — i.e. it is the true "no advisor anywhere" baseline.
    # The pre-existing `ensure_indexes()` (called once per SF) still
    # provides the realistic backbone of indexes; this fix only
    # controls the advisor-recommended `ix_advisor_bench`.
    created_idx: set = set()
    if strategy != 'no_advisor':
        # Window-0: initial advisor pass on the opening (phase-0) workload.
        invoke_advisor(conn, dbname,
                       [QUERIES[q] for q in PHASES[0]['qs']], created_idx)

    # ── Windows 1..N_WINDOWS: strategy-dependent (metrics start here) ──
    total_queries   = 0
    total_ok        = 0
    advisor_calls   = 0
    T_A_total_ms    = 0.0
    hsm_scores      = []
    breakdown_rows: list[dict] = []   # per-window 5-D persistence (BUG #4 fix)

    # Track previous-window phase index so drift_truth can be computed inline
    # without re-walking PHASES at the end of the loop.
    prev_ph_idx = None

    # Each block is an independent workload realisation -- clear the DCI
    # gate's drift trajectory (the fitted mu0/Sigma0/thresholds stay).
    if strategy == 'dci_gated' and dci_gate is not None:
        dci_gate.reset_trajectory()

    wall_start = time.perf_counter()

    for win in range(1, N_WINDOWS + 1):
        # Map win 1..24 to phases (6 windows each, but offset by 1 for init)
        ph_idx = min((win - 1) // WIN_PER_PH, len(PHASES) - 1)
        ph = PHASES[ph_idx]
        wts = np.array(ph['w'], dtype=float)
        wts /= wts.sum()
        qnames_arr = list(np.random.choice(ph['qs'], size=QUERIES_PW, p=wts, replace=True))
        workload_fp.update(('|'.join(qnames_arr) + '\n').encode())

        # Phase-specific Poisson rate
        lam = LAMBDA_MAP.get(ph['name'], 40.0)

        exec_times = []
        win_n_ok = 0
        win_wall_t0 = time.perf_counter()   # window wall clock (incl. delays)
        for qname in qnames_arr:
            # Poisson inter-arrival delay
            if use_poisson:
                delay = rng.expovariate(lam)
                time.sleep(delay)

            t0 = time.perf_counter()
            try:
                cur.execute(QUERIES[qname])
                cur.fetchall()
                ok = True
            except Exception:
                conn.rollback()
                ok = False
            ms = (time.perf_counter() - t0) * 1000
            exec_times.append(ms)
            total_queries += 1
            if ok:
                total_ok += 1
                win_n_ok += 1

        win_wall_ms = (time.perf_counter() - win_wall_t0) * 1000.0

        # Build window features
        cur_features = make_window_features(qnames_arr, exec_times)

        # Compute full 5-D HSM breakdown (BUG #4 fix — was scalar-only before)
        breakdown = compute_hsm_breakdown(prev_features, cur_features)
        hsm_score = breakdown['HSM']
        hsm_scores.append(hsm_score)

        # Strategy decision (win is 1-based here).
        # `dci_gated` routes through the DCIGate -- it needs the full 5-D
        # vector, not the scalar score; every other strategy uses the
        # scalar-score should_invoke().
        invoked = False
        if strategy == 'dci_gated':
            fire = bool(dci_gate.decide(
                [breakdown['S_R'], breakdown['S_V'], breakdown['S_T'],
                 breakdown['S_A'], breakdown['S_P']]))
        else:
            fire = should_invoke(strategy, win, hsm_score)
        if fire:
            t_a = invoke_advisor(
                conn, dbname,
                [QUERIES[qn] for qn in dict.fromkeys(qnames_arr)],
                created_idx)
            advisor_calls += 1
            T_A_total_ms += t_a
            invoked = True

        # Drift truth label: True iff this window's phase differs from previous
        # window's phase. Window 1 has no measured predecessor (the window-0 init
        # always uses PHASES[0]=Reporting), so drift_truth=False for win=1 by
        # the same convention as Mongo 13_mongo_adaptation.py.
        if win == 1:
            drift_truth = False
        else:
            drift_truth = (ph_idx != prev_ph_idx)
        prev_ph_idx = ph_idx

        breakdown_rows.append({
            'block':              block_num,
            'block_seed':         rng_seed,
            'strategy':           strategy,
            'window':             win,
            'phase':              ph['name'],
            'drift_truth':        int(drift_truth),
            'S_R':                round(float(breakdown['S_R']), 6),
            'S_V':                round(float(breakdown['S_V']), 6),
            'S_T':                round(float(breakdown['S_T']), 6),
            'S_A':                round(float(breakdown['S_A']), 6),
            'S_P':                round(float(breakdown['S_P']), 6),
            'HSM':                round(float(breakdown['HSM']), 6),
            'invoked':            int(invoked),
            'n_queries':          len(qnames_arr),
            'n_ok':               win_n_ok,
            'exec_ms_window_sum': round(sum(exec_times), 3),
            'wall_ms_window':     round(win_wall_ms, 3),
            'qps_window':         (round(win_n_ok / (win_wall_ms / 1000.0), 4)
                                   if win_wall_ms > 0 else 0.0),
        })

        # ── Per-window timestamped progress (T3.4c) ──
        _cum_s = time.perf_counter() - wall_start
        _cum_qps = total_queries / _cum_s if _cum_s > 0 else 0.0
        _win_qps_inst = breakdown_rows[-1]['qps_window']
        print(f"    [{ts()}] B{block_num:02d} {strategy:<11s} "
              f"win={win:02d}/{N_WINDOWS} q={total_queries:>3d}/{N_WINDOWS*QUERIES_PW} "
              f"win_qps={_win_qps_inst:5.2f} cum_qps={_cum_qps:5.2f} "
              f"adv={advisor_calls} \u0394T={win_wall_ms/1000.0:5.1f}s",
              flush=True)

        prev_features = cur_features

    wall_time = time.perf_counter() - wall_start
    wall_qps  = total_queries / wall_time if wall_time > 0 else 0.0

    # Cleanup advisor index
    cleanup_advisor_index(conn, created_idx)
    cur.close(); conn.close()

    # Phase-boundary detection: count actual drift points
    # (where consecutive windows belong to different phases)
    n_drifts = 0
    for w in range(1, N_WINDOWS):
        ph_prev = min((w - 1) // WIN_PER_PH, len(PHASES) - 1)
        ph_curr = min(w // WIN_PER_PH, len(PHASES) - 1)
        if ph_prev != ph_curr:
            n_drifts += 1

    # Precision/Recall for HSM_gated (how well it detects real drifts)
    # True positive = invoke at a drift boundary; False positive = invoke at non-drift
    if strategy == 'hsm_gated' and advisor_calls > 0:
        drift_windows = set()
        for w in range(1, N_WINDOWS + 1):
            ph_prev_idx = min((w - 2) // WIN_PER_PH, len(PHASES) - 1) if w > 1 else 0
            ph_curr_idx = min((w - 1) // WIN_PER_PH, len(PHASES) - 1)
            if ph_prev_idx != ph_curr_idx:
                drift_windows.add(w)

        # Reconstruct which windows were invoked
        invoked_windows = set()
        for w_idx, hsm_s in enumerate(hsm_scores):
            if hsm_s < THETA:
                invoked_windows.add(w_idx + 1)  # 1-based

        tp = len(invoked_windows & drift_windows)
        fp = len(invoked_windows - drift_windows)
        fn = len(drift_windows - invoked_windows)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        precision = float('nan')
        recall    = float('nan')

    # Persist the full HSM series so the T4 detector-quality post-processor
    # (`report_detector_quality.py`) can sweep θ off-line and recompute
    # (TPR̂, TNR̂) at any threshold without re-running the workload.
    hsm_scores_csv = ','.join(f'{x:.4f}' for x in hsm_scores)

    # Per-window phase labels (1-based, matching hsm_scores) so the same
    # post-processor can re-stratify pairs into within-/cross-phase strata
    # for the stratified Hoeffding band.
    win_phase_labels = []
    for win in range(1, N_WINDOWS + 1):
        ph_idx = min((win - 1) // WIN_PER_PH, len(PHASES) - 1)
        win_phase_labels.append(PHASES[ph_idx]['name'][:3])

    return {
        'block_metrics': {
            'block':           block_num,
            'block_seed':      rng_seed,
            'strategy':        strategy,
            'wall_qps':        round(wall_qps, 4),
            'wall_time_s':     round(wall_time, 3),
            'queries_total':   total_queries,
            'queries_ok':      total_ok,
            'advisor_calls':   advisor_calls,
            'T_A_total_ms':    round(T_A_total_ms, 2),
            'mean_hsm':        round(float(np.mean(hsm_scores)), 4),
            'hsm_below_theta': sum(1 for h in hsm_scores if h < THETA),
            'n_drift_points':  n_drifts,
            'precision':       round(precision, 4) if not math.isnan(precision) else float('nan'),
            'recall':          round(recall, 4)    if not math.isnan(recall)    else float('nan'),
            'hsm_series':      hsm_scores_csv,         # T4 post-processor input
            'phase_series':    '|'.join(win_phase_labels),
            'p_advisor':       round(advisor_calls / N_WINDOWS, 4),  # T5 p̂_stable input
            'theta':           THETA,                  # T3 G2 traceability
            'k_periodic':      K_PERIODIC,
            'workload_fp':     workload_fp.hexdigest()[:16],  # paired-RCB sanity
            'note':            'window_0=shared_init; metrics from window_1 onward',
        },
        'breakdown_rows':  breakdown_rows,             # 5-D per-window (BUG #4)
    }


# ══════════════════════════════════════════════════════════════
#  Provenance helpers (parity with Mongo 13_mongo_adaptation.py)
# ══════════════════════════════════════════════════════════════

def _git_sha() -> str:
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=here, stderr=subprocess.DEVNULL,
        ).decode().strip() != ""
        return f"{out}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _file_sha(p: str) -> str:
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown"


def write_run_meta(out_path: Path, sf: float, dbname: str,
                   started_at: str, ended_at: str,
                   n_blocks: int, base_seed: int,
                   use_poisson: bool, n_blocks_actual: int) -> None:
    """Write per-SF run_meta.json for full reproducibility (parity with Mongo)."""
    here = os.path.dirname(os.path.abspath(__file__))
    code_root = os.path.dirname(os.path.dirname(here))
    meta = {
        "engine":          "postgres",
        "workload":        WORKLOAD,           # tpch | job (T3.4b)
        "sf":              sf,
        "has_sf_axis":     HAS_SF_AXIS,
        "dbname":          dbname,
        "git_sha":         _git_sha(),
        "started_at":      started_at,
        "ended_at":        ended_at,
        "n_blocks":        n_blocks_actual,
        "n_blocks_target": n_blocks,
        "base_seed":       base_seed,
        "block_seeds":     [base_seed + b * 100 for b in range(1, n_blocks_actual + 1)],
        "strategies":      STRATEGIES,
        "use_poisson":     use_poisson,
        "constants": {
            "N_WINDOWS":   N_WINDOWS,
            "WIN_PER_PH":  WIN_PER_PH,
            "QUERIES_PW":  QUERIES_PW,
            "THETA":       THETA,
            "K_PERIODIC":  K_PERIODIC,
            "W0":          {'R': 0.25, 'V': 0.20, 'T': 0.20, 'A': 0.20, 'P': 0.15},
            "LAMBDA_MAP":  LAMBDA_MAP,
            "DCI_TAU":     DCI_TAU,      # Paper 3D RQ5 DCI-gate config
            "DCI_ALPHA":   DCI_ALPHA,
            "DCI_N_CAL":   DCI_N_CAL,
        },
        "phases":          [{"name": p['name'], "qs": p['qs'], "w": p['w']} for p in PHASES],
        "advisor_index":   ADVISOR_INDEX,
        "advisor_ddl_create": ADVISOR_DDL_CR,
        "file_shas": {
            "pg_adaptation.py":              _file_sha(os.path.abspath(__file__)),
            "pg_workloads.py":               _file_sha(os.path.join(here, "pg_workloads.py")),
            f"{_WC['queries_module']}.py":   _file_sha(os.path.join(here, f"{_WC['queries_module']}.py")),
        },
        "command_line":    " ".join(sys.argv),
        "audit_fixes_applied": [
            "BUG #4 (5-D breakdown persisted; was scalar-only pre 2026-04-09)",
        ],
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', choices=['tpch', 'job'], default='tpch',
                        help='Workload selector (tpch | job) — also pre-parsed '
                             'at module load to configure QUERIES/PHASES/ADVISOR')
    parser.add_argument('--sf', nargs='+', type=float,
                        default=[0.2, 1.0, 3.0] if HAS_SF_AXIS else [1.0],
                        help='Scale factors to run (TPC-H: 0.2/1.0/3.0; '
                             'JOB: 1.0 only — single IMDB instance)')
    parser.add_argument('--no-poisson', action='store_true',
                        help='Disable Poisson inter-arrival delays')
    parser.add_argument('--blocks', type=int, default=N_BLOCKS,
                        help='Number of RCB blocks (default 10)')
    args = parser.parse_args()

    use_poisson = not args.no_poisson
    n_blocks    = args.blocks

    print('=' * 70)
    print('  Adaptation Strategy Comparison — RCB Design (Paper A13)')
    print(f'  Strategies: {STRATEGIES}')
    print(f'  Blocks: {n_blocks}, Windows/block: {N_WINDOWS}, Queries/window: {QUERIES_PW}')
    print(f'  θ = {THETA},  K = {K_PERIODIC},  Poisson = {use_poisson}')
    print('=' * 70)

    all_results = []
    all_breakdowns: list[dict] = []   # 5-D per-window persistence (BUG #4 fix)

    # ── Resume support: load any previously-saved per-SF partial results ──
    # Per-SF CSVs live in results/adaptation_comparison_v2_sfXX.csv. If one
    # exists for an SF in args.sf, we load it into all_results and skip the
    # run for that SF. This makes the script resumable after a crash (e.g.
    # the SF=10 statement_timeout incident of 2026-04-09).
    #
    # Resume requires BOTH the block-metrics CSV and the breakdown CSV to be
    # present and consistent. Breakdown CSV missing → cannot resume (the SF
    # was generated by the pre-BUG-#4 code that did not persist 5-D data).
    resume_loaded = set()
    for sf in args.sf:
        # Filename suffix is `sfX_Y` for TPC-H (per-SF outputs) or just
        # `job` for JOB (single dataset, no SF axis).  Keeps TPC-H output
        # filenames backward-compatible while adding JOB cleanly.
        sf_tag = (f'sf{sf}'.replace('.', '_') if HAS_SF_AXIS else WORKLOAD) + os.environ.get('PROBE_TAG', '')
        resume_path = RESULTS_DIR / f'adaptation_comparison_v2_{sf_tag}.csv'
        breakdown_path = RESULTS_DIR / f'breakdown_per_window_v2_{sf_tag}.csv'
        if resume_path.exists() and breakdown_path.exists():
            try:
                prev_df = pd.read_csv(resume_path)
                bk_df = pd.read_csv(breakdown_path)
                all_results.extend(prev_df.to_dict('records'))
                all_breakdowns.extend(bk_df.to_dict('records'))
                resume_loaded.add(sf)
                print(f'  ✓ RESUME: loaded {len(prev_df)} rows + '
                      f'{len(bk_df)} breakdown rows from SF={sf}')
            except Exception as e:
                print(f'  ! RESUME failed for SF={sf}: {e}')
        elif resume_path.exists() and not breakdown_path.exists():
            print(f'  ! SF={sf}: legacy block-metrics CSV exists but '
                  f'breakdown_per_window CSV is missing — cannot resume '
                  f'(pre-BUG-#4 data). Will re-run.')

    for sf in args.sf:
        if sf in resume_loaded:
            print(f'\n{"─"*70}')
            print(f'  SF = {sf}  SKIPPED (results already on disk)')
            print(f'{"─"*70}')
            continue

        dbname = SF_DB_MAP.get(sf)
        if dbname is None:
            print(f'\n  WARNING: No database mapping for SF={sf}, skipping.')
            continue

        print(f'\n{"─"*70}')
        print(f'  SF = {sf}  (db: {dbname})')
        print(f'{"─"*70}')

        sf_started_at = datetime.now().isoformat(timespec='seconds')
        ensure_database(sf, dbname)

        # Warm up PG caches with one dummy pass
        print(f'  Warming up...')
        try:
            conn = connect(dbname); conn.autocommit = True; cur = conn.cursor()
            cur.execute("SET statement_timeout = '30s'")
            for qname in WARMUP_QIDS:
                try:
                    cur.execute(QUERIES[qname]); cur.fetchall()
                except Exception:
                    conn.rollback()
            cur.close(); conn.close()
        except Exception:
            pass

        base_seed = 7000 + int(sf * 100)

        # ── DCI-gate calibration pass (Paper 3D RQ5) ──
        # Fit mu0/Sigma0 once per (engine, SF) on a steady workload, then
        # freeze the gate for every block (T3.2_DCI_GATE_THEORY.md §11).
        dci_gate = None
        if 'dci_gated' in STRATEGIES:
            print(f'  [{ts()}] Calibrating DCI gate ({DCI_N_CAL} steady windows)...', flush=True)
            dci_gate = calibrate_dci_gate(dbname, use_poisson=use_poisson)
            print(f'    DCI gate fitted: m={dci_gate.config()["m_steady"]} '
                  f'tau={DCI_TAU} alpha={DCI_ALPHA}')

        for block in range(1, n_blocks + 1):
            # Randomise strategy order (Latin Square counterbalancing)
            block_rng = random.Random(base_seed + block)
            order = STRATEGIES.copy()
            block_rng.shuffle(order)

            print(f'\n  [{ts()}] === Block {block:02d}/{n_blocks} START - order = {order} ===', flush=True)

            # ── PAIRED RCB FIX (2026-04-09) ──
            # All four strategies in this block run on the SAME workload
            # realisation (same init queries, same per-window query draws,
            # same Poisson inter-arrival delays). The Latin-square `order`
            # counterbalances cache-state nuisance across blocks; the seed
            # below is what guarantees fair pairing.
            #
            # Previous (BUGGY) version used:
            #   seed = base_seed + block * 100 + STRATEGIES.index(strat)
            # which gave each strategy a DIFFERENT rng_seed -> different
            # query selections inside `run_block`. Even within one block
            # the four strategies were running four different workloads,
            # so wall_qps differences mixed treatment effect with random
            # workload variance. RCB pairing was destroyed.
            block_seed = base_seed + block * 100

            block_fps = {}
            for strat in order:
                print(f'    [{ts()}] -- {strat:<11s} START - {N_WINDOWS} win x {QUERIES_PW} q expected', flush=True)
                result = run_block(
                    dbname, strat, block, block_seed,
                    use_poisson=use_poisson, dci_gate=dci_gate,
                )
                bm = result['block_metrics']
                bm['sf'] = sf
                all_results.append(bm)

                # Stamp every per-window breakdown row with sf so the merged
                # CSV can be filtered downstream by SF.
                for br in result['breakdown_rows']:
                    br['sf'] = sf
                    all_breakdowns.append(br)

                ac = bm['advisor_calls']
                wq = bm['wall_qps']
                ta = bm['T_A_total_ms']
                fp = bm['workload_fp']
                block_fps[strat] = fp
                print(f'    [{ts()}] -- {strat:<11s} end:  wall-QPS={wq:7.3f}  '
                      f'advisor={ac:2d}  T_A={ta:8.1f}ms  fp={fp}', flush=True)

            # ── Paired-RCB sanity assertion ──
            # All four strategies in this block MUST have produced the same
            # workload (same query draws, same order). If this fails, the
            # rng_seed plumbing has regressed and wall_qps numbers cannot
            # be paired. Hard-fail rather than silently producing biased data.
            unique_fps = set(block_fps.values())
            if len(unique_fps) != 1:
                print(f'    ✗ FATAL: workload fingerprints diverged in block {block}:')
                for s, f in block_fps.items():
                    print(f'      {s:15s}  {f}')
                print('    Paired-RCB pairing is broken — see seed handling in main().')
                sys.exit(2)

        # ── Per-SF summary ──
        sf_df = pd.DataFrame([r for r in all_results if r['sf'] == sf])
        print(f'\n  {"="*60}')
        print(f'  SF={sf} Summary (mean ± SD across {n_blocks} blocks):')
        print(f'  {"="*60}')
        print(f'  {"Strategy":15s}  {"wall-QPS":>12s}  {"Advisor":>8s}  {"T_A(ms)":>10s}')
        print(f'  {"-"*50}')
        for strat in STRATEGIES:
            s_df = sf_df[sf_df['strategy'] == strat]
            qps_m = s_df['wall_qps'].mean()
            qps_s = s_df['wall_qps'].std()
            ac_m  = s_df['advisor_calls'].mean()
            ta_m  = s_df['T_A_total_ms'].mean()
            print(f'  {strat:15s}  {qps_m:6.3f}±{qps_s:5.3f}  {ac_m:6.1f}  {ta_m:10.1f}')

        # Speedup: HSM-gated / always-on
        ao_qps = sf_df[sf_df['strategy']=='always_on']['wall_qps'].mean()
        hg_qps = sf_df[sf_df['strategy']=='hsm_gated']['wall_qps'].mean()
        if ao_qps > 0:
            print(f'\n  Speedup (HSM-gated / always-on): {hg_qps/ao_qps:.3f}×')

        ao_ac = sf_df[sf_df['strategy']=='always_on']['advisor_calls'].mean()
        hg_ac = sf_df[sf_df['strategy']=='hsm_gated']['advisor_calls'].mean()
        if ao_ac > 0:
            reduction = (1 - hg_ac / ao_ac) * 100
            print(f'  Advisor call reduction: {reduction:.1f}%')

        # ── Incremental save: one CSV per SF for crash resilience ──
        # Filename suffix is `sfX_Y` for TPC-H (per-SF outputs) or just
        # `job` for JOB (single dataset, no SF axis).  Keeps TPC-H output
        # filenames backward-compatible while adding JOB cleanly.
        sf_tag = (f'sf{sf}'.replace('.', '_') if HAS_SF_AXIS else WORKLOAD) + os.environ.get('PROBE_TAG', '')
        sf_out_path = RESULTS_DIR / f'adaptation_comparison_v2_{sf_tag}.csv'
        sf_df.to_csv(sf_out_path, index=False)
        print(f'  ✓ Per-SF block-metrics saved: {sf_out_path.name}')

        # 5-D per-window breakdown CSV (BUG #4 fix)
        sf_bk_df = pd.DataFrame([b for b in all_breakdowns if b.get('sf') == sf])
        if not sf_bk_df.empty:
            bk_out_path = RESULTS_DIR / f'breakdown_per_window_v2_{sf_tag}.csv'
            sf_bk_df.to_csv(bk_out_path, index=False)
            print(f'  ✓ Per-SF breakdown saved:     {bk_out_path.name}  '
                  f'({len(sf_bk_df)} rows = {n_blocks} blocks × 4 strats × {N_WINDOWS} wins)')

        # Per-SF reproducibility metadata (parity with Mongo run_meta.json)
        sf_ended_at = datetime.now().isoformat(timespec='seconds')
        meta_path = RESULTS_DIR / f'run_meta_v2_{sf_tag}.json'
        write_run_meta(
            meta_path, sf=sf, dbname=dbname,
            started_at=sf_started_at, ended_at=sf_ended_at,
            n_blocks=N_BLOCKS, base_seed=base_seed,
            use_poisson=use_poisson, n_blocks_actual=n_blocks,
        )
        print(f'  ✓ Per-SF run_meta saved:      {meta_path.name}')

    # ── Save merged results across all SFs ──
    out_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / 'adaptation_comparison_v2.csv'
    out_df.to_csv(out_path, index=False)
    print(f'\nAll block-metrics saved: {out_path}')

    # 5-D per-window breakdown — merged across all SFs (BUG #4 fix)
    if all_breakdowns:
        bk_out_df = pd.DataFrame(all_breakdowns)
        bk_out_path = RESULTS_DIR / 'breakdown_per_window_v2.csv'
        bk_out_df.to_csv(bk_out_path, index=False)
        print(f'All breakdowns saved:    {bk_out_path}  ({len(bk_out_df)} rows)')

    # ── Print paper-ready LaTeX table ──
    print('\n' + '='*70)
    print('  LaTeX Table (paper Table throughput):')
    print('='*70)
    for sf in args.sf:
        sf_df = pd.DataFrame([r for r in all_results if r['sf'] == sf])
        if sf_df.empty:
            continue
        vals = {}
        for strat in STRATEGIES:
            s_df = sf_df[sf_df['strategy'] == strat]
            m = s_df['wall_qps'].mean()
            s = s_df['wall_qps'].std()
            vals[strat] = (m, s)

        bl = vals.get('no_advisor', (0,0))
        ao = vals.get('always_on', (0,0))
        pe = vals.get('periodic', (0,0))
        hg = vals.get('hsm_gated', (0,0))
        speedup = hg[0] / ao[0] if ao[0] > 0 else 0

        print(f'  {sf} & ${bl[0]:.2f}\\!\\pm\\!{bl[1]:.2f}$ '
              f'& ${ao[0]:.3f}\\!\\pm\\!{ao[1]:.3f}$ '
              f'& ${pe[0]:.3f}\\!\\pm\\!{pe[1]:.3f}$ '
              f'& ${hg[0]:.3f}\\!\\pm\\!{hg[1]:.3f}$ '
              f'& ${speedup:.3f}\\times$ \\\\')


if __name__ == '__main__':
    main()
