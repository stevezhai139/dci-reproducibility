#!/usr/bin/env python3
"""
mongo_adaptation.py — MongoDB end-to-end adaptation harness (Paper 3D RQ5).

Vendored from Paper 3B `cal/mongo/adaptation/mongo_adaptation_paper3a.py`
(itself a strict mirror of the Postgres `07_adaptation_comparison.py`) and
re-pointed onto the Paper 3D canonical kernel + the engine-agnostic DCI
gate. The four-policy paired-RCB design, the window-0 init phase, the RCB
seed sharing, the gating semantics, and the output schema are all kept at
1:1 parity with the PostgreSQL harness `pg_adaptation.py` so the
cross-engine RQ5 comparison is meaningful.

Paper 3D additions — the identical integration pattern as pg_adaptation.py:
  - a fifth strategy `dci_gated` — the engine-free DCI gate, run
    head-to-head against the HSM-composite theta-gate (`hsm_gated`),
    RCB-paired within every block;
  - calibrate_dci_gate() — the steady-window calibration pass that fits
    the DCIGate once for the engine, then freezes mu0/Sigma0/F-thresholds
    (T3.2_DCI_GATE_THEORY.md §11);
  - per-window throughput (`wall_ms_window`, `qps_window`) in the
    breakdown rows (the block-level `wall_qps` is unchanged).

Parity contract with Postgres step 5
────────────────────────────────────
  STRATEGIES        = ['no_advisor','always_on','periodic','hsm_gated',
                       'dci_gated']
  N_BLOCKS          = 10
  N_WINDOWS         = 24       (4 phases × 6 windows)
  WIN_PER_PH        = 6
  QUERIES_PW        = 20
  THETA             = 0.75     (SIMILARITY threshold; invoke when score < θ)
  K_PERIODIC        = 3
  W0                = {R:0.25, V:0.20, T:0.20, A:0.20, P:0.15}  (from hsm_v2_core)
  detector          = full hsm_v2 (R+V+T+A+P)  via common.hsm_bridge
  window-0 init     = shared init window; all advisor-using strategies invoke
                       advisor exactly once at win=0; no_advisor stays clean
  workload_fp       = sha256 over the same qid-stream that the queries see,
                       so paired strategies in the same block share fp
  RCB seed offset   = block_seed XOR 0xA5A5 → param sampler RNG;
                       all strategies in the same block see identical
                       concrete pipelines.

Engine-specific deltas (intentional)
────────────────────────────────────
  - source: mydb_p3a.combined_clean (built by build_experiment_db.py)
  - advisor: createIndex / dropIndex on the candidate_index of the worst
             qid in the most-recent window; cleanup = drop all advisor
             indexes at end of block (markovian state per block)
  - phase schedule: edge → geo → text → review (24 windows, 6 each)
  - per-strategy seed family: BASE_SEED=9000 (distinct from Postgres 1000-band)

Outputs
───────
  code/end_to_end/mongo/out/<timestamp>/
    block_metrics.csv          # one row per (strategy, block)
    breakdown_per_window.csv   # one row per (strategy, block, window) with
                               #   S_R, S_V, S_T, S_A, S_P, HSM, invoked,
                               #   drift_truth, exec_ms_window_sum,
                               #   wall_ms_window, qps_window
    run_meta.json              # provenance bundle for reproducibility

Parallelism: DO NOT RUN during Postgres experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Local imports ─────────────────────────────────────────────────────
# Paper 3D vendoring: this harness lives at code/end_to_end/mongo/. The
# MongoDB feature path was harmonised onto the canonical kernel in Phase 2
# (task T2.1) and lives at code/cross_engine/. Re-point sys.path at that
# harmonised tree, plus code/end_to_end/ for the DCI gate. The module
# *names* (templates, param_sampler, window_features, hsm_bridge) are
# unchanged — only the directories that hold them moved — so the import
# lines below are kept verbatim. hsm_bridge re-points the kernel
# internally onto code/kernel/ (see cross_engine/PROVENANCE.md).
HERE = os.path.dirname(os.path.abspath(__file__))      # code/end_to_end/mongo/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))           # dci_gate.py (code/end_to_end/)
sys.path.insert(0, os.path.join(HERE, "..", "..", "cross_engine", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cross_engine", "mongo", "workload"))
from templates import (  # noqa: E402
    ALL_TEMPLATES,
    ALL_PHASES,
    ALL_QIDS_SORTED,
)
from param_sampler import materialize_pipeline  # noqa: E402
from window_features import make_window_features  # noqa: E402
from hsm_bridge import (  # noqa: E402
    compute_window_hsm,
    compute_window_hsm_breakdown,
    get_w0,
    is_available as hsm_available,
)
# DCI gate — the engine-agnostic gate under test (Paper 3D RQ5). Built
# once to T3.2_DCI_GATE_THEORY.md §11, unit-tested 5/5, reused by every
# engine harness.
from dci_gate import DCIGate  # noqa: E402  # Paper 3C: DCI-routed gate
sys.path.insert(0, HERE)                       # season_schedule.py / esr_recommender.py
try:
    import season_schedule as _season          # Addendum 21 irregular long-season schedule
    import esr_recommender as _esr              # Addendum 21 real ESR recommender
except Exception as _e:                          # keep legacy import working if absent
    _season = None; _esr = None


# ── Parity constants (mirror Postgres step 5) ─────────────────────────
STRATEGIES = ["dci_gated"]  # Paper 3C repo: DCI-routed gate only
N_BLOCKS = 10           # parity with Postgres step 5
N_WINDOWS = 24
WIN_PER_PH = 6
QUERIES_PW = 20         # parity with Postgres step 5
THETA = 0.75            # SIMILARITY threshold; invoke when hsm < THETA
K_PERIODIC = 3          # invoke every K-th window
BASE_SEED = 9000        # distinct from Postgres seed-band 1000-7000

# ── Paper 3D Addendum 21 — MongoDB seasonality + real ESR recommender ──
# These are set from CLI in main(); defaults preserve the legacy validated run.
ADVISOR_MODE = "worstqid"   # "worstqid" (legacy) | "esr" (real ESR recommender)
SCHEDULE_MODE = "legacy"    # "legacy" | "irregular" | "regular"  (Addendum 16/21)
BASELINE_MODE = "full"     # "full"=text index in backbone (legacy) | "under"=ESR owns text (Addendum 21 churn)

# ── DCI gate (Paper 3D RQ5) ───────────────────────────────────────────
# The engine-free gate, run head-to-head against the HSM-composite
# theta-gate (`hsm_gated`). One harness run yields both, RCB-paired.
# These constants are identical to the PostgreSQL harness pg_adaptation.py.
DCI_TAU   = float(os.environ.get('DCI_TAU', 1.5))  # 3C: env-overridable (RawDCIGate)
DCI_ALPHA = 0.05    # gate false-positive rate (RQ5 sweeps {0.01,0.05,0.10})
DCI_N_CAL = 64      # steady windows in the calibration pass (m ~ 64)

# ── Path P2/P3 override hooks (added 2026-05-05 for Paper 3B-Cal) ─────
# When set via CLI, these REPLACE the defaults above for the hsm_gated
# strategy. W0 from hsm_bridge stays the source of truth for default
# behaviour; these globals only take effect when explicitly set.
W_OVERRIDE: dict | None = None      # e.g. {"R": 0.683, "V": 0.307, ...}
THETA_OVERRIDE: float | None = None  # e.g. 0.741

# Phase schedule for 24 windows: edge × 6, geo × 6, text × 6, review × 6
PHASE_SCHEDULE = (
    ["edge"] * 6 + ["geo"] * 6 + ["text"] * 6 + ["review"] * 6
)
assert len(PHASE_SCHEDULE) == N_WINDOWS

# S5 Part 2 (2026-07-07): per-window op-count schedule for the MIXED-drift
# variant (--schedule mixed). None = legacy behaviour (QUERIES_PW everywhere).
# When set (tuple of len N_WINDOWS), window w (1-based) draws
# VOLUME_SCHEDULE[w-1] ops, and a count change across consecutive windows is
# a drift-truth onset (a pure S_V volume move; see season_schedule.
# make_mixed_schedule).
VOLUME_SCHEDULE: tuple | None = None

SOURCE_DB = "mydb_p3a"
SOURCE_COLL = "combined_clean"

# Paper 3D output dir: code/end_to_end/mongo/out/ (parity with the
# PostgreSQL harness's RESULTS_DIR = code/end_to_end/postgres/out/).
RESULTS_ROOT = os.path.abspath(os.path.join(HERE, "out"))


def log(m: str) -> None:
    """ISO-8601 UTC timestamp prefix (T3.4c, Phase 1 §5)."""
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {m}", flush=True)


# ───────────────────────────────────────────────────────────────────────
# Workload generator — paired-RCB deterministic per (block, seed)
# ───────────────────────────────────────────────────────────────────────

def generate_window(phase_name: str, n: int, rng: random.Random) -> list[str]:
    """Sample n qids from a phase mix using the phase weight vector."""
    phase = ALL_PHASES[phase_name]
    qids = list(phase["mix"].keys())
    weights = [phase["mix"][q] for q in qids]
    return rng.choices(qids, weights=weights, k=n)


def generate_block_workload(block_seed: int) -> list[list[str]]:
    """Generate a deterministic 24-window qid sequence for a single block.

    Includes a window-0 init draw so all 4 strategies share the same initial
    workload AND the same hsm baseline. Window 0 is from the FIRST phase
    (edge) by convention.
    """
    rng = random.Random(block_seed)

    def _count(idx0: int) -> int:
        """Ops for schedule index idx0 (0-based). S5 Part 2 volume axis."""
        return (VOLUME_SCHEDULE[idx0] if VOLUME_SCHEDULE is not None
                else QUERIES_PW)

    windows: list[list[str]] = []
    # window 0 = init (always edge phase; matches Postgres ph0 = PHASES[0])
    windows.append(generate_window(PHASE_SCHEDULE[0], _count(0), rng))
    # windows 1..N_WINDOWS = measured
    for w_idx in range(1, N_WINDOWS + 1):
        ph = PHASE_SCHEDULE[w_idx - 1]
        windows.append(generate_window(ph, _count(w_idx - 1), rng))
    return windows  # length = N_WINDOWS + 1


def fingerprint(windows: list[list[str]]) -> str:
    """sha256 of all qids across all windows (incl. window 0). Cumulative,
    matching the Postgres `workload_fp.update(...)` style."""
    h = hashlib.sha256()
    for w in windows:
        h.update(("|".join(w) + "\n").encode())
    return h.hexdigest()[:16]


# ───────────────────────────────────────────────────────────────────────
# Mongo engine adapter
# ───────────────────────────────────────────────────────────────────────

def connect(uri: str):
    from pymongo import MongoClient
    c = MongoClient(uri, serverSelectionTimeoutMS=5000)
    c.admin.command("ping")
    return c


def ensure_backbone_indexes(coll) -> None:
    """Create the always-on backbone indexes (cheap, no workload signal).

    Backbone = the minimal set every strategy starts with. It does not by
    itself resolve any witness query, so the advisor still has meaningful
    work to do.
    """
    coll.create_index([("type", 1)], name="bb_type")
    coll.create_index([("label", 1)], name="bb_label")
    # T3.7 smoke-debug fix (2026-05-27).  Text index on `abstract` is a
    # STRUCTURAL requirement of the text-phase workload: Q16 uses the
    # $text operator, and MongoDB allows at most ONE text index per
    # collection — so without a baseline text index, every Q16 fails
    # ("text index required for $text query") under every strategy.
    # The advisor cannot recover this case in practice (Q16 fails fast
    # → tiny exec_ms → never in the top-3-slowest the advisor picks),
    # so the workload is structurally unrunnable without this index.
    # Treating it as backbone (alongside bb_type, bb_label) matches
    # PHASE3_PLAN §5 "PK + FK structural baseline" — what a DBA would
    # set up before the advisor ever runs.
    #
    # Addendum 21 (Option B / --baseline under): with the REAL ESR advisor
    # (which recommends a text index from the $text query SHAPE, not from
    # exec_ms), the advisor CAN provision text — so we drop the backbone text
    # index to activate the documented text drop+rebuild CHURN and the
    # under-provisioned ("no_advisor catastrophic on $text") regime. The
    # worst-qid limitation above no longer applies under ADVISOR_MODE="esr".
    if BASELINE_MODE != "under":
        coll.create_index([("abstract", "text")], name="bb_abstract_text")
    else:
        # --baseline under: GUARANTEE no text index exists (a prior full run may
        # have persisted bb_abstract_text on the collection). Drop it so the ESR
        # advisor owns text and no_advisor $text queries genuinely fail. Also
        # drop any leftover text index under a different name (one-text-index).
        for _n, _info in list(coll.index_information().items()):
            if _n == "_id_":
                continue
            _key = _info.get("key", [])
            if _n == "bb_abstract_text" or any(
                    (isinstance(_t, str) and _t == "text") for _, _t in _key):
                try:
                    coll.drop_index(_n)
                    log(f"  [baseline=under] dropped pre-existing text index {_n}")
                except Exception as _e:
                    log(f"  [baseline=under] could not drop {_n}: {_e}")


def drop_advisor_indexes(coll) -> int:
    """Drop every index that is NOT _id_ and NOT a backbone (bb_*).

    Used at the start of every block (markovian per-block state) and at
    end-of-block cleanup. Returns count dropped.
    """
    n = 0
    for name in list(coll.index_information().keys()):
        if name == "_id_" or name.startswith("bb_"):
            continue
        try:
            coll.drop_index(name)
            n += 1
        except Exception as e:
            log(f"  drop_index({name}) failed: {e}")
    return n


def _index_name_for(cand_index: tuple) -> str:
    parts = []
    for k, d in cand_index:
        if d == 1:
            parts.append(f"{k}a")
        elif d == -1:
            parts.append(f"{k}d")
        else:
            parts.append(f"{k}{str(d)[:1]}")
    return "adv_" + "_".join(parts)


def _invoke_advisor_esr(coll, recent_qids: list[str]) -> tuple[int, float]:
    """Real ESR recommender path (Addendum 21). Recommends indexes from the
    WINDOW's query shapes (not the slowest qid), applies them idempotently,
    and honours MongoDB's one-text-index-per-collection limit by dropping a
    prior advisor text index before creating a new one. Returns
    (n_created, wall_ms) where wall_ms includes recommend + real build time."""
    if _esr is None or not recent_qids:
        return (0, 0.0)
    pipelines = [ALL_TEMPLATES[q].pipeline for q in recent_qids if q in ALL_TEMPLATES]
    t0 = time.perf_counter()
    specs = _esr.recommend(pipelines)                 # recommend-time (cheap, but counted)
    info = coll.index_information()
    existing_names = set(info.keys())
    existing_keys = {tuple(sorted((f, k) for f, k in v["key"])) for v in info.values() if "key" in v}
    n_created = 0
    for key in specs:
        name = _esr.spec_name(key)
        model = [(f, (k if k in ("text", "2dsphere") else int(k))) for f, k in key]
        sig = tuple(sorted((f, k) for f, k in model))
        if name in existing_names or sig in existing_keys:
            continue
        is_text = any(k == "text" for _, k in model)
        try:
            if is_text:                                # one text index per collection
                for ex_name, ex_info in list(coll.index_information().items()):
                    if ex_name.startswith("ix_esr_") and any(
                            (isinstance(t, str) and t == "text") for _, t in ex_info.get("key", [])):
                        coll.drop_index(ex_name)
            coll.create_index(model, name=name)        # real build cost timed here
            n_created += 1
        except Exception as e:
            log(f"  esr create_index({name}) failed: {e}")
    return (n_created, (time.perf_counter() - t0) * 1000.0)


def invoke_advisor(coll, recent_exec_ms: list[float], recent_qids: list[str]) -> tuple[int, float]:
    """Pick the candidate index of the slowest qid in the most recent window
    and create it if missing. Returns (n_created, wall_ms_overhead).

    The "advisor" here is intentionally simple: it ranks qids by mean
    exec_ms within the recent window and creates up to 3 missing
    candidate indexes. Real index advisors (e.g., Dexter, AutoIndex) can
    be plugged in by replacing this single function.
    """
    if ADVISOR_MODE == "esr":
        return _invoke_advisor_esr(coll, recent_qids)
    if not recent_exec_ms:
        return (0, 0.0)
    from collections import defaultdict
    agg: dict[str, list[float]] = defaultdict(list)
    for ms, qid in zip(recent_exec_ms, recent_qids):
        agg[qid].append(ms)
    means = sorted(((sum(v) / len(v), q) for q, v in agg.items()), reverse=True)
    # T3.7 smoke-debug fix (2026-05-27).  Build BOTH a name set AND a
    # canonical key-signature set of existing indexes.  The original
    # check (`if name in existing`) only catches name collisions, but
    # MongoDB rejects (errno 85, IndexOptionsConflict) when the
    # candidate's key signature duplicates an existing index under any
    # different name — e.g. advisor's `adv_typea` vs backbone `bb_type`
    # (both index {type: 1}).  Catching this at the key level keeps the
    # log clean and the advisor_calls count meaningful ("calls that
    # would have created a new index" rather than "calls attempted").
    existing_info = coll.index_information()
    existing_names = set(existing_info.keys())
    existing_keys = {tuple(sorted(info["key"])) for info in existing_info.values()
                     if "key" in info}
    n_created = 0
    t0 = time.perf_counter()
    for _, qid in means[:3]:
        cand = ALL_TEMPLATES[qid].candidate_index
        if not cand:
            continue
        name = _index_name_for(cand)
        if name in existing_names:
            continue
        if tuple(sorted(cand)) in existing_keys:
            continue           # key signature already covered (different name OK)
        try:
            coll.create_index(list(cand), name=name)
            n_created += 1
        except Exception as e:
            log(f"  create_index({name}) failed: {e}")
    overhead_ms = (time.perf_counter() - t0) * 1000
    return (n_created, overhead_ms)


def _classify_error(msg: str) -> str:
    """expected = the designed $text-without-index failure of the under-
    provisioned baseline (--baseline under, Addendum 21); everything else is
    unexpected (a real bug) and is surfaced loudly."""
    m = msg.lower()
    if "text index required" in m or ("text index" in m and "$text" in m) or "need a text index" in m:
        return "text_index_required"
    return "other"


def run_window(coll, qids: list[str], rng: random.Random):
    """Execute a window of queries.
    Returns (per-query exec_ms list, n_ok, fails) where fails is a list of
    (qid, error_class, message). Expected text-index errors are tallied
    SILENTLY (they are a designed result under --baseline under); unexpected
    errors are logged loudly so genuine bugs are never hidden."""
    exec_ms: list[float] = []
    n_ok = 0
    fails: list[tuple] = []
    for qid in qids:
        tmpl = ALL_TEMPLATES[qid]
        pipeline = materialize_pipeline(tmpl, rng)
        t0 = time.perf_counter()
        try:
            list(coll.aggregate(pipeline, allowDiskUse=True))
            n_ok += 1
        except Exception as ex:
            cls = _classify_error(str(ex))
            fails.append((qid, cls, str(ex)))
            if cls != "text_index_required":
                log(f"  query {qid} UNEXPECTED failure: {ex}")   # real bug → loud
            # expected text_index_required → silent, tallied into fails
        exec_ms.append((time.perf_counter() - t0) * 1000)
    return exec_ms, n_ok, fails


# ───────────────────────────────────────────────────────────────────────
# Run one (strategy, block) cell
# ───────────────────────────────────────────────────────────────────────

def should_invoke(strategy: str, win_1based: int, hsm_score: float) -> bool:
    """Strategy decision (mirrors Postgres should_invoke).

    win_1based is 1..N_WINDOWS (window 0 is init, never decided here).
    Periodic: invoke at windows 1, 1+K, 1+2K, ...
    HSM-gated: invoke when SIMILARITY < THETA (or THETA_OVERRIDE if set).
    """
    if strategy == "no_advisor":
        return False
    if strategy == "always_on":
        return True
    if strategy == "periodic":
        return ((win_1based - 1) % K_PERIODIC == 0)
    if strategy == "hsm_gated":
        # Path P2/P3: use THETA_OVERRIDE if Steve passed --theta
        theta = THETA_OVERRIDE if THETA_OVERRIDE is not None else THETA
        return hsm_score < theta
    return False


def reweight_hsm(breakdown: dict) -> float:
    """Compute HSM as weighted sum of S_R, S_V, S_T, S_A, S_P with W_OVERRIDE.

    Used for Path P2/P3 verification when Steve passes --w_R/V/T/A/P CLI
    flags. Returns the W0-baked HSM if W_OVERRIDE is None.
    """
    if W_OVERRIDE is None:
        return float(breakdown["HSM"])
    return float(
        W_OVERRIDE["R"] * breakdown["S_R"]
        + W_OVERRIDE["V"] * breakdown["S_V"]
        + W_OVERRIDE["T"] * breakdown["S_T"]
        + W_OVERRIDE["A"] * breakdown["S_A"]
        + W_OVERRIDE["P"] * breakdown["S_P"]
    )


def verify_source(client) -> int:
    """Verify the MongoDB source collection is present and non-empty.

    Paper 3D expects mydb_p3a.combined_clean pre-built on the MongoDBDisk
    external SSD (the MongoDB engine *and* its document volume both live
    on that Thunderbolt SSD). Mirrors the PostgreSQL harness
    `ensure_database()` verify-only check: connect() already fails fast
    if mongod is down; this catches the "engine up but data missing"
    case with one clear message instead of an opaque per-query failure.
    """
    coll = client[SOURCE_DB][SOURCE_COLL]
    n = coll.estimated_document_count()
    if n <= 0:
        raise RuntimeError(
            f"source collection {SOURCE_DB}.{SOURCE_COLL} is empty or "
            f"missing. Paper 3D expects it pre-built on MongoDBDisk — "
            f"mount the SSD, start mongod, build the collection, re-run.")
    log(f"  ✓ source {SOURCE_DB}.{SOURCE_COLL} ({n:,} documents)")
    return n


def calibrate_dci_gate(client, n_cal: int = DCI_N_CAL) -> DCIGate:
    """Calibration pass for the DCI gate (Paper 3D RQ5;
    T3.2_DCI_GATE_THEORY.md §11).

    Run `n_cal` STEADY windows — the first phase (`edge`), no drift —
    against the live collection, collect the per-window 5-D HSM feature
    vectors, and fit a DCIGate once. mu0 / Sigma0 / the F-thresholds are
    then FROZEN for the official blocks: they are the steady-state
    reference, and a moving reference would mask the drift it must
    detect. The kernel, DCI, routing and detector all stay live per
    window — only the calibration is frozen.

    This is the exact engine-agnostic pattern used by the PostgreSQL
    harness `calibrate_dci_gate()`; only the steady-window generator is
    engine-specific (Mongo aggregate pipeline vs Postgres SQL). Returns
    the fitted gate.
    """
    coll = client[SOURCE_DB][SOURCE_COLL]
    drop_advisor_indexes(coll)          # clean steady state, keep backbone
    ensure_backbone_indexes(coll)
    rng = random.Random(20260525)
    steady_phase = PHASE_SCHEDULE[0]    # "edge" — the steady phase
    feats: list[list[float]] = []

    def _steady_window() -> dict:
        qids = generate_window(steady_phase, QUERIES_PW, rng)
        exec_ms, _, _ = run_window(coll, qids, rng)
        return make_window_features(qids, exec_ms)

    prev = _steady_window()
    for _ in range(n_cal):
        cur_f = _steady_window()
        b = compute_window_hsm_breakdown(prev, cur_f)
        feats.append([b["S_R"], b["S_V"], b["S_T"], b["S_A"], b["S_P"]])
        prev = cur_f
    return DCIGate(tau=DCI_TAU, alpha=DCI_ALPHA).fit(
        np.asarray(feats, dtype=float))


def run_block(client, strategy: str, block_idx: int, block_seed: int,
              dci_gate: DCIGate = None) -> dict:
    """Run one (strategy, block) cell of the paired-RCB design."""
    coll = client[SOURCE_DB][SOURCE_COLL]

    # Reset advisor state, keep backbone
    drop_advisor_indexes(coll)
    ensure_backbone_indexes(coll)

    # Build the deterministic qid stream for this block (incl. window 0)
    windows = generate_block_workload(block_seed)
    wp_fp = fingerprint(windows)

    # Param-sampling RNG: same offset for all strategies → identical concrete
    # pipelines across paired strategies in the same block.
    param_rng = random.Random(block_seed ^ 0xA5A5)

    # ── Window 0: shared init (parity with Postgres DESIGN FIX 2026-04-09) ──
    # All strategies execute window-0 queries to seed `prev_features`.
    # Only advisor-using strategies provision the advisor at window 0.
    init_qids = windows[0]
    init_exec_ms, _, _ = run_window(coll, init_qids, param_rng)
    prev_features = make_window_features(init_qids, init_exec_ms)

    if strategy != "no_advisor":
        _n0, _ = invoke_advisor(coll, init_exec_ms, init_qids)

    # ── Windows 1..N_WINDOWS: measured ──
    advisor_calls = 0
    T_A_total_ms = 0.0
    index_builds = (_n0 if strategy != 'no_advisor' else 0)   # distinct NEW indexes built this block (wasteful-build proxy; incl. init)
    queries_total = 0
    queries_ok = 0
    hsm_scores: list[float] = []
    breakdown_rows: list[dict] = []
    block_errors: list[dict] = []
    queries_failed = 0

    # Each block is an independent workload realisation — clear the DCI
    # gate's drift trajectory (the fitted mu0/Sigma0/thresholds stay).
    if strategy == "dci_gated" and dci_gate is not None:
        dci_gate.reset_trajectory()

    wall_start = time.perf_counter()

    for w_1based in range(1, N_WINDOWS + 1):
        qids = windows[w_1based]
        phase_name = PHASE_SCHEDULE[w_1based - 1]

        win_wall_t0 = time.perf_counter()        # window wall clock
        exec_ms, n_ok, win_fails = run_window(coll, qids, param_rng)
        win_wall_ms = (time.perf_counter() - win_wall_t0) * 1000.0
        queries_total += len(qids)
        queries_ok += n_ok
        queries_failed += len(win_fails)
        # tally this window's failures (qid x class) for the separate errors CSV
        _wf = {}
        for _qid, _cls, _msg in win_fails:
            _wf[(_qid, _cls)] = _wf.get((_qid, _cls), 0) + 1
        for (_qid, _cls), _cnt in _wf.items():
            block_errors.append({"block": block_idx, "strategy": strategy,
                                 "window": w_1based, "qid": _qid,
                                 "error_class": _cls, "count": _cnt})

        cur_features = make_window_features(qids, exec_ms)
        breakdown = compute_window_hsm_breakdown(prev_features, cur_features)
        # Path P2/P3: re-weight HSM using W_OVERRIDE if --w_R/V/T/A/P set
        hsm_score = reweight_hsm(breakdown)
        hsm_scores.append(hsm_score)

        # Strategy decision. `dci_gated` routes through the DCIGate on the
        # full per-window 5-D [S_R..S_P] vector; every other strategy uses
        # the scalar-score should_invoke(). This is the identical decision
        # branch as the PostgreSQL harness pg_adaptation.py.
        invoked = False
        gate_dci, gate_regime = "", ""   # S5 Part 2: per-window gate diagnostics
        if strategy == "dci_gated":
            fire = bool(dci_gate.decide(
                [breakdown["S_R"], breakdown["S_V"], breakdown["S_T"],
                 breakdown["S_A"], breakdown["S_P"]]))
            # Persist the gate routing state (dci, 1-D/5-D regime) so the
            # Part 2 analyzer computes detector monitoring cost from the log
            # directly instead of replaying (escalation_replay.py-style).
            _gl = dci_gate.last
            if _gl is not None:
                gate_dci = ("" if math.isnan(_gl["dci"])
                            else round(float(_gl["dci"]), 4))
                gate_regime = _gl["regime"]
        else:
            fire = should_invoke(strategy, w_1based, hsm_score)
        if fire:
            n_new, t_a = invoke_advisor(coll, exec_ms, qids)
            advisor_calls += 1
            T_A_total_ms += t_a
            index_builds += n_new
            invoked = True

        # Drift truth label: True iff phase_name differs from previous window's phase.
        # Window 1 has no measured predecessor, so drift_truth = 0 by definition
        # (the init pass at w_1based=0 is treated as the same phase as w_1based=1).
        if w_1based == 1:
            drift_truth = False
        else:
            drift_truth = (PHASE_SCHEDULE[w_1based - 1] != PHASE_SCHEDULE[w_1based - 2])
            if VOLUME_SCHEDULE is not None:      # S5 Part 2: volume onset
                drift_truth = drift_truth or (
                    VOLUME_SCHEDULE[w_1based - 1] != VOLUME_SCHEDULE[w_1based - 2])

        breakdown_rows.append({
            "block": block_idx,
            "block_seed": block_seed,
            "strategy": strategy,
            "window": w_1based,
            "phase": phase_name,
            "drift_truth": int(drift_truth),
            "S_R": breakdown["S_R"],
            "S_V": breakdown["S_V"],
            "S_T": breakdown["S_T"],
            "S_A": breakdown["S_A"],
            "S_P": breakdown["S_P"],
            "HSM": breakdown["HSM"],
            "invoked": int(invoked),
            "gate_dci": gate_dci,        # S5 Part 2 (blank: non-DCI strategy / warm-up NaN)
            "gate_regime": gate_regime,  # S5 Part 2: "1-D" | "5-D"
            "n_queries": len(qids),
            "n_ok": n_ok,
            "n_failed": len(win_fails),
            "exec_ms_window_sum": round(sum(exec_ms), 3),
            "wall_ms_window": round(win_wall_ms, 3),
            "qps_window": (round(n_ok / (win_wall_ms / 1000.0), 4)
                           if win_wall_ms > 0 else 0.0),
        })

        # ── Per-window timestamped progress (T3.4c) ──
        _cum_s = time.perf_counter() - wall_start
        _cum_qps = queries_total / _cum_s if _cum_s > 0 else 0.0
        _win_qps_inst = breakdown_rows[-1]["qps_window"]
        log(f"  B{block_idx:02d} {strategy:<11s} win={w_1based:02d}/{N_WINDOWS} "
            f"q={queries_total:>3d}/{N_WINDOWS*QUERIES_PW} "
            f"win_qps={_win_qps_inst:5.2f} cum_qps={_cum_qps:5.2f} "
            f"adv={advisor_calls} fail={len(win_fails):>2d} \u0394T={win_wall_ms/1000.0:5.1f}s")

        prev_features = cur_features

    wall_time_s = time.perf_counter() - wall_start
    wall_qps = queries_total / wall_time_s if wall_time_s > 0 else 0.0

    # End-of-block cleanup (markovian state)
    drop_advisor_indexes(coll)

    # Drift-boundary precision/recall (TPR/TNR computed offline by T4 post-processor)
    drift_windows = {b["window"] for b in breakdown_rows if b["drift_truth"] == 1}
    invoked_windows = {b["window"] for b in breakdown_rows if b["invoked"] == 1}
    n_drift = len(drift_windows)
    if strategy == "hsm_gated" and len(invoked_windows) > 0:
        tp = len(invoked_windows & drift_windows)
        fp = len(invoked_windows - drift_windows)
        fn = len(drift_windows - invoked_windows)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        precision = float("nan")
        recall = float("nan")

    return {
        "block_metrics": {
            "block": block_idx,
            "block_seed": block_seed,
            "strategy": strategy,
            "workload_fp": wp_fp,
            "wall_qps": round(wall_qps, 4),
            "wall_time_s": round(wall_time_s, 3),
            "queries_total": queries_total,
            "queries_ok": queries_ok,
            "queries_failed": queries_failed,
            "advisor_calls": advisor_calls,
            "index_builds": index_builds,
            "T_A_total_ms": round(T_A_total_ms, 2),
            "p_advisor": round(advisor_calls / N_WINDOWS, 4),
            "mean_hsm": round(float(np.mean(hsm_scores)), 4),
            "hsm_below_theta": int(sum(1 for h in hsm_scores if h < THETA)),
            "n_drift_points": n_drift,
            "precision": round(precision, 4) if precision == precision else float("nan"),
            "recall": round(recall, 4) if recall == recall else float("nan"),
            "hsm_series": ",".join(f"{x:.4f}" for x in hsm_scores),
            "phase_series": "|".join(PHASE_SCHEDULE),
            "theta": THETA,
            "k_periodic": K_PERIODIC,
        },
        "breakdown_rows": breakdown_rows,
        "block_errors": block_errors,
    }


# ───────────────────────────────────────────────────────────────────────
# Output helpers
# ───────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=HERE, stderr=subprocess.DEVNULL,
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


def write_outputs(outdir: Path, all_metrics: list[dict],
                  all_breakdowns: list[dict], meta: dict,
                  all_errors: list[dict] | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    bm_path = outdir / "block_metrics.csv"
    if all_metrics:
        with open(bm_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            w.writeheader()
            w.writerows(all_metrics)
    log(f"  wrote {bm_path}")

    bw_path = outdir / "breakdown_per_window.csv"
    if all_breakdowns:
        with open(bw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_breakdowns[0].keys()))
            w.writeheader()
            w.writerows(all_breakdowns)
    log(f"  wrote {bw_path}")

    # Separate, structured error tally (Addendum 21 --baseline under): the
    # designed $text-without-index failures of the under-provisioned baseline
    # are expected results, kept OUT of the main log to keep it clean and
    # written here as analysis-ready rows (block, strategy, window, qid,
    # error_class, count). Only written if any failures occurred.
    if all_errors:
        err_path = outdir / "query_errors.csv"
        with open(err_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_errors[0].keys()))
            w.writeheader(); w.writerows(all_errors)
        _by_cls = {}
        for r in all_errors:
            _by_cls[r["error_class"]] = _by_cls.get(r["error_class"], 0) + r["count"]
        log(f"  wrote {err_path}  ({sum(_by_cls.values())} failed queries; by class: {_by_cls})")

    meta_path = outdir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"  wrote {meta_path}")


# ───────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────

def main() -> int:
    global W_OVERRIDE, THETA_OVERRIDE  # Path P2/P3 hooks
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="mongodb://localhost:27017")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--blocks", type=int, default=N_BLOCKS)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    # ── Path P2/P3 overrides (added 2026-05-05 for Paper 3B-Cal) ──────
    ap.add_argument("--w_R", type=float, default=None,
                    help="Path P2/P3: override W0[R]. All 5 of --w_R/V/T/A/P "
                         "must be given together. If unset, W0 from "
                         "hsm_bridge.get_w0() is used (Paper 3A default).")
    ap.add_argument("--w_V", type=float, default=None)
    ap.add_argument("--w_T", type=float, default=None)
    ap.add_argument("--w_A", type=float, default=None)
    ap.add_argument("--w_P", type=float, default=None)
    ap.add_argument("--theta", type=float, default=None,
                    help="Path P2/P3: override THETA constant. If unset, "
                         "0.75 is used (Paper 3A default).")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Path P3 multi-seed: shift block_seed by 100*offset. "
                         "Default 0 = blocks 0..N use seeds 9000, 9100, 9200, "
                         "... (matches Paper 3A 30-Apr run). Use offset=N to "
                         "run a fresh seed range disjoint from prior runs.")
    ap.add_argument("--results-subdir", type=str, default=None,
                    help="Optional subdirectory under results/cross_engine/"
                         "mongo/adaptation/<subdir>/<ts>/. Default: no subdir "
                         "(matches Paper 3A 30-Apr run layout).")
    # ── Paper 3D Addendum 21: seasonality schedule + real ESR recommender ──
    ap.add_argument("--schedule", choices=["legacy", "irregular", "regular",
                                           "mixed"],
                    default="legacy",
                    help="legacy=4x6 fixed (default, validated); irregular=long "
                         "variable seasons (headline); regular=fixed-period "
                         "seasons (sensitivity axis). Addendum 16/21. "
                         "mixed=S5 Part 2 alternating template/volume onsets "
                         "(24 windows; season_schedule.make_mixed_schedule).")
    ap.add_argument("--n-windows", type=int, default=None,
                    help="windows per block for irregular/regular schedules "
                         "(default 120). legacy ignores this (stays 24).")
    ap.add_argument("--advisor", choices=["worstqid", "esr"], default="worstqid",
                    help="worstqid=legacy single-candidate advisor; esr=real "
                         "workload ESR recommender (Addendum 21).")
    ap.add_argument("--min-len", type=int, default=12, help="irregular season min length")
    ap.add_argument("--max-len", type=int, default=36, help="irregular season max length")
    ap.add_argument("--season-seed", type=int, default=None,
                    help="seed for the season schedule (default BASE_SEED). The "
                         "schedule is shared across blocks/strategies for RCB pairing.")
    ap.add_argument("--baseline", choices=["full", "under"], default="full",
                    help="full=backbone includes the abstract text index (legacy); "
                         "under=drop it so the ESR advisor owns text → drop+rebuild "
                         "churn + no_advisor $text errors (Addendum 21 headline, "
                         "the Mongo analogue of relational PK-only). Use with "
                         "--advisor esr.")
    args = ap.parse_args()

    # Apply overrides
    w_args = [args.w_R, args.w_V, args.w_T, args.w_A, args.w_P]
    if any(w is not None for w in w_args):
        if any(w is None for w in w_args):
            log("FATAL: must supply ALL FIVE --w_R/V/T/A/P or none of them")
            return 3
        W_OVERRIDE = {
            "R": float(args.w_R), "V": float(args.w_V), "T": float(args.w_T),
            "A": float(args.w_A), "P": float(args.w_P),
        }
        s = sum(W_OVERRIDE.values())
        if not (0.99 <= s <= 1.01):
            log(f"WARNING: W_OVERRIDE sum = {s:.4f} (expected ~1.0); "
                f"continuing anyway")
    if args.theta is not None:
        THETA_OVERRIDE = float(args.theta)

    # ── Addendum 21 wiring: set advisor + (optionally) rebuild the schedule ──
    global ADVISOR_MODE, SCHEDULE_MODE, PHASE_SCHEDULE, N_WINDOWS, BASELINE_MODE
    global VOLUME_SCHEDULE
    ADVISOR_MODE = args.advisor
    SCHEDULE_MODE = args.schedule
    BASELINE_MODE = args.baseline
    if args.baseline == "under" and args.advisor != "esr":
        log("WARNING: --baseline under expects --advisor esr (worst-qid cannot "
            "provision text from shape); $text will error for ALL strategies.")
    if args.advisor == "esr" and _esr is None:
        log("FATAL: --advisor esr but esr_recommender import failed"); return 5
    if args.schedule == "mixed":
        # S5 Part 2: alternating template/volume onsets (deterministic —
        # no season seed; identical across blocks/strategies/tau configs).
        if _season is None:
            log("FATAL: --schedule needs season_schedule import"); return 5
        nw = args.n_windows or 24
        phase_per_window, count_per_window, boundary_types = \
            _season.make_mixed_schedule(n_windows=nw, base_count=QUERIES_PW)
        PHASE_SCHEDULE = tuple(phase_per_window)
        VOLUME_SCHEDULE = tuple(count_per_window)
        N_WINDOWS = nw
        log(f"  schedule      : mixed (S5 Part 2)  n_windows={nw}  "
            f"onsets={ {k + 1: v for k, v in sorted(boundary_types.items())} } (1-based windows)")
        log(f"  volume sched  : {VOLUME_SCHEDULE}")
    elif args.schedule != "legacy":
        if _season is None:
            log("FATAL: --schedule needs season_schedule import"); return 5
        nw = args.n_windows or 120
        sseed = args.season_seed if args.season_seed is not None else BASE_SEED
        phase_per_window, seasons, drift_set = _season.make_schedule(
            nw, seed=sseed, mode=args.schedule,
            min_len=args.min_len, max_len=args.max_len)
        PHASE_SCHEDULE = tuple(phase_per_window)
        N_WINDOWS = nw
        log(f"  schedule      : {args.schedule}  n_windows={nw} seed={sseed} "
            f"seasons={len(seasons)} drifts={len(drift_set)}")
        log(f"  seasons       : " + ", ".join(f"{x.phase}[{x.start}:{x.end}]" for x in seasons))
    log(f"  advisor mode  : {ADVISOR_MODE}")
    log(f"  baseline      : {BASELINE_MODE}"  + ("  ← ESR owns text (churn + no_advisor $text errors)" if BASELINE_MODE=="under" else ""))

    log(f"mongo_adaptation.py dry_run={args.dry_run} blocks={args.blocks}")
    log(f"  source        : {SOURCE_DB}.{SOURCE_COLL}")
    log(f"  strategies    : {args.strategies}")
    log(f"  N_WINDOWS     : {N_WINDOWS} (init=window0 + measured 1..{N_WINDOWS})")
    log(f"  QUERIES_PW    : {QUERIES_PW}")
    eff_theta = THETA_OVERRIDE if THETA_OVERRIDE is not None else THETA
    log(f"  THETA         : {eff_theta}  (similarity; invoke when score < θ)"
        f"{'  ← OVERRIDE active' if THETA_OVERRIDE is not None else ''}")
    log(f"  K_PERIODIC    : {K_PERIODIC}")
    if W_OVERRIDE is not None:
        log(f"  W (OVERRIDE)  : {W_OVERRIDE}  ← Path P2/P3 calibrated weights")
        log(f"  W0 (default)  : {get_w0()}  (NOT used; OVERRIDE takes precedence)")
    else:
        log(f"  W0            : {get_w0()}")
    seed_preview = [
        BASE_SEED + args.seed_offset * 100 + i * 100
        for i in range(min(3, args.blocks))
    ]
    log(f"  seed_offset   : {args.seed_offset}  (seeds preview: {seed_preview} ...)")
    log(f"  hsm_v2 avail  : {hsm_available()}")
    log(f"  phase schedule: {PHASE_SCHEDULE}")

    if args.dry_run:
        log("DRY-RUN: validating workload generator + features (no mongod contact)")
        for b in range(min(3, args.blocks)):
            seed = BASE_SEED + b * 100
            ws = generate_block_workload(seed)
            fp = fingerprint(ws)
            log(f"  block {b:02d} seed={seed} fp={fp} "
                f"win0={ws[0][:5]}… win1={ws[1][:5]}… "
                f"len(windows)={len(ws)}")
            # Validate feature builder runs end-to-end with synthetic times
            f0 = make_window_features(ws[0], [10.0] * len(ws[0]))
            f1 = make_window_features(ws[1], [10.0] * len(ws[1]))
            sim = compute_window_hsm(f0, f1)
            bd = compute_window_hsm_breakdown(f0, f1)
            log(f"    sim(w0,w1)={sim:.4f}  breakdown={bd}")
        # Verify paired-RCB invariant: same seed → same fingerprint
        s = BASE_SEED + 0
        fp_a = fingerprint(generate_block_workload(s))
        fp_b = fingerprint(generate_block_workload(s))
        assert fp_a == fp_b, f"non-deterministic fingerprint: {fp_a} vs {fp_b}"
        log("  paired-RCB invariant OK (same seed → same fingerprint)")
        # ── DCI gate integration smoke check (Paper 3D RQ5) ──
        # Fit a DCIGate on synthetic steady features and exercise
        # fit()/reset_trajectory()/decide() — validates the DCI import and
        # the gate wiring without contacting mongod.
        _g = DCIGate(tau=DCI_TAU, alpha=DCI_ALPHA)
        _drng = np.random.default_rng(20260525)
        _g.fit(_drng.normal(0.0, 1.0, size=(DCI_N_CAL, 5)))
        _g.reset_trajectory()
        _dec = [bool(_g.decide(list(_drng.normal(0.0, 1.0, 5))))
                for _ in range(5)]
        log(f"  DCI gate smoke OK: m={_g.config()['m_steady']} "
            f"tau={DCI_TAU} alpha={DCI_ALPHA} decisions={_dec}")
        log("DRY-RUN OK")
        return 4

    # ── Real run ──
    if not hsm_available():
        log("FATAL: canonical HSM kernel unavailable; cannot run real adaptation")
        return 2

    client = connect(args.uri)
    verify_source(client)
    started_at = datetime.now().isoformat(timespec="seconds")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.results_subdir:
        outdir = Path(RESULTS_ROOT) / args.results_subdir / ts
    else:
        outdir = Path(RESULTS_ROOT) / ts

    # ── DCI-gate calibration pass (Paper 3D RQ5) ──
    # Fit mu0/Sigma0 once for the engine on a steady workload, then freeze
    # the gate for every block (T3.2_DCI_GATE_THEORY.md §11). MongoDB is a
    # single collection (no scale-factor axis), so calibration runs exactly
    # once — the PostgreSQL harness calibrates once per (engine, SF).
    dci_gate = None
    if "dci_gated" in STRATEGIES:
        log(f"  Calibrating DCI gate ({DCI_N_CAL} steady windows)...")
        dci_gate = calibrate_dci_gate(client)
        log(f"    DCI gate fitted: m={dci_gate.config()['m_steady']} "
            f"tau={DCI_TAU} alpha={DCI_ALPHA}")

    all_metrics: list[dict] = []
    all_breakdowns: list[dict] = []
    all_errors: list[dict] = []

    try:
        for b in range(args.blocks):
            # Path P3 seed_offset shifts the base by 100*offset (e.g. offset=10
            # → first seed = 9000 + 1000 = 10000, disjoint from 30-Apr range)
            block_seed = BASE_SEED + (args.seed_offset + b) * 100
            log(f"=== Block {b:02d}/{args.blocks} START - seed={block_seed} - strategies={args.strategies} ===")
            for strategy in args.strategies:
                log(f"  -- {strategy:<11s} START - {N_WINDOWS} win x {QUERIES_PW} q expected")
                result = run_block(client, strategy, b, block_seed,
                                   dci_gate=dci_gate)
                bm = result["block_metrics"]
                log(f"  -- {strategy:<11s} end:  wall-QPS={bm['wall_qps']:7.3f}  "
                    f"advisor={bm['advisor_calls']:2d}  T_A={bm['T_A_total_ms']:8.1f}ms  "
                    f"fp={bm['workload_fp']}")
                all_metrics.append(bm)
                all_breakdowns.extend(result["breakdown_rows"])
                all_errors.extend(result.get("block_errors", []))
    finally:
        client.close()
        ended_at = datetime.now().isoformat(timespec="seconds")
        meta = {
            "engine": "mongo",
            "git_sha": _git_sha(),
            "n_blocks": args.blocks,
            "strategies": args.strategies,
            "seed_offset": args.seed_offset,
            "constants": {
                "N_WINDOWS": N_WINDOWS,
                "WIN_PER_PH": WIN_PER_PH,
                "QUERIES_PW": QUERIES_PW,
                "THETA": THETA,
                "THETA_effective": THETA_OVERRIDE if THETA_OVERRIDE is not None else THETA,
                "K_PERIODIC": K_PERIODIC,
                "BASE_SEED": BASE_SEED,
                "W0": get_w0(),
                "W_OVERRIDE": W_OVERRIDE,  # null if no override
                "DCI_TAU": DCI_TAU,        # Paper 3D RQ5 DCI-gate config
                "DCI_ALPHA": DCI_ALPHA,
                "DCI_N_CAL": DCI_N_CAL,
            },
            "phase_schedule": PHASE_SCHEDULE,
            "schedule_mode": SCHEDULE_MODE,            # S5 Part 2 provenance
            "volume_schedule": VOLUME_SCHEDULE,        # None unless --schedule mixed
            "advisor_mode": ADVISOR_MODE,
            "baseline_mode": BASELINE_MODE,
            "source": f"{SOURCE_DB}.{SOURCE_COLL}",
            "started_at": started_at,
            "ended_at": ended_at,
            # File SHAs — re-pointed to the Paper 3D tree (the MongoDB
            # feature path now lives at code/cross_engine/, the kernel at
            # code/kernel/, the DCI gate at code/end_to_end/).
            "mongo_adaptation_sha": _file_sha(os.path.abspath(__file__)),
            "templates_sha": _file_sha(os.path.join(HERE, "..", "..", "cross_engine", "mongo", "workload", "templates.py")),
            "kernel_sha": _file_sha(os.path.join(HERE, "..", "..", "kernel", "hsm_v2_kernel.py")),
            "param_sampler_sha": _file_sha(os.path.join(HERE, "..", "..", "cross_engine", "common", "param_sampler.py")),
            "window_features_sha": _file_sha(os.path.join(HERE, "..", "..", "cross_engine", "common", "window_features.py")),
            "hsm_bridge_sha": _file_sha(os.path.join(HERE, "..", "..", "cross_engine", "common", "hsm_bridge.py")),
            "dci_gate_sha": _file_sha(os.path.join(HERE, "..", "dci_gate.py")),
            "command_line": " ".join(sys.argv),
        }
        write_outputs(outdir, all_metrics, all_breakdowns, meta, all_errors)

    log(f"DONE. Results: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
