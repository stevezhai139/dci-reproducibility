# Paper 3D — `code/end_to_end/mongo/` provenance

Phase 3 (T3.3). The MongoDB four-policy adaptation harness, vendored
into the Paper 3D tree and re-pointed onto the canonical kernel + the
harmonised cross-engine feature path.

## Files

**`mongo_adaptation.py`** — vendored from
`Paper 3B/HSM_gated_3B_for_paper3b_v2/code/experiments/cal/mongo/adaptation/mongo_adaptation_paper3a.py`.
That file is itself a strict structural mirror of the PostgreSQL
`07_adaptation_comparison.py` (T3.1 audit confirmed: same `STRATEGIES`,
same paired-RCB block design, same `run_block`, same per-window 5-D
`breakdown`). The four-policy adaptation loop (`no_advisor` /
`always_on` / `periodic` / `hsm_gated`), the `createIndex`/`dropIndex`
advisor, the window-0 shared-init phase, the RCB seed sharing, and the
block-level `wall_qps` throughput are all kept **verbatim**. Only the
following Paper 3D re-point edits were applied:

- `sys.path` — re-pointed to the Paper 3D tree. The MongoDB feature
  path was harmonised onto the canonical kernel in Phase 2 (task T2.1)
  and lives at `code/cross_engine/`; `sys.path` now points at
  `code/cross_engine/common/` (`hsm_bridge`, `param_sampler`,
  `window_features`), `code/cross_engine/mongo/workload/` (`templates`),
  and `code/end_to_end/` (`dci_gate`). The module *names* are
  unchanged, so the `import` lines are kept verbatim;
- the kernel — `hsm_bridge.py` (the harmonised copy in
  `cross_engine/common/`) imports the single canonical kernel
  `code/kernel/hsm_v2_kernel.py`, not the V3 `_v3_hsm/hsm_v2_core.py`.
  `hsm_bridge` calls the kernel only through `hsm_v2(...)`, positionally,
  and reads `W0`; the canonical kernel exports both with a matching
  12-argument signature, so `compute_window_hsm_breakdown()`'s call site
  is unchanged. The only behavioural difference is S_P (the canonical
  kernel's symmetrised FastDTW — see `cross_engine/PROVENANCE.md`);
- `RESULTS_ROOT` → `code/end_to_end/mongo/out/` (parity with the
  PostgreSQL harness's `RESULTS_DIR`);
- `verify_source()` — a new verify-only check (mirrors the PostgreSQL
  harness's `ensure_database()`): fail fast with one clear message if
  `mongod` is up but `mydb_p3a.combined_clean` is empty/missing;
- `run_meta.json` file SHAs — re-pointed to the vendored Paper 3D
  filenames + the harness self-SHA and the `dci_gate.py` SHA added.

The MongoDB engine **and** its document collection
(`mydb_p3a.combined_clean`) both live on **MongoDBDisk** — a 1 TB
Thunderbolt external SSD. Mount the SSD and start `mongod` before any
live MongoDB harness run; the harness connects via a localhost URI.

## Status — T3.3 sandbox build complete

Vendor + re-point + DCI-gate integration: **done**. `mongo_adaptation.py`
compiles, imports clean (kernel = the canonical `hsm_v2_kernel` via the
harmonised `hsm_bridge`), and `--dry-run` passes end-to-end (workload
generator, paired-RCB invariant, 5-D HSM breakdown, DCI-gate
integration smoke check).

DCI-gate integration applied — the **identical** engine-agnostic
pattern as `code/end_to_end/postgres/pg_adaptation.py` (the same
surgical changes transferred verbatim, only the engine-specific
steady-window generator differs):

- `STRATEGIES` → 5 — `…/hsm_gated` (the θ-gate baseline) **plus**
  `dci_gated`; one harness run is the θ-vs-DCI head-to-head, RCB-paired;
- `calibrate_dci_gate()` — the steady-observation calibration pass:
  run `DCI_N_CAL = 64` phase-0 (`edge`) windows, fit `DCIGate` once for
  the engine, freeze μ̂₀/Σ̂₀/F-thresholds (theory §11). MongoDB is a
  single collection (no scale-factor axis), so calibration runs exactly
  once — the PostgreSQL harness calibrates once per (engine, SF);
- `run_block` — gained the `dci_gate` parameter; `dci_gated` routes
  through `DCIGate.decide()` on the per-window 5-D `[S_R..S_P]` vector;
  the gate trajectory is reset at the start of each block;
- per-window throughput — `wall_ms_window`, `qps_window` added to
  `breakdown_rows` (the block-level `wall_qps` is unchanged);
- `run_meta.json` `constants` — `DCI_TAU`/`DCI_ALPHA`/`DCI_N_CAL`
  recorded (the PostgreSQL `write_run_meta` was patched to match, so
  the two harnesses' provenance is symmetric).

## T3.7 smoke-debug fixes (2026-05-27)

The first live-MongoDB smoke (`--blocks 1` against
`mydb_p3a.combined_clean`, 5,000,000 docs on MongoDBDisk) completed
end-to-end with the paired-RCB invariant intact (workload fingerprint
`bc0e8928a13604f9` identical across all 5 strategies), but surfaced
two data-contamination issues that needed harness fixes before the
full T3.7 run.  Both fixes are localised, do not touch the workload
generator or the DCI integration, and were verified by re-smoke.

**Fix 1 — text index added to `ensure_backbone_indexes()`.**  Q16's
MongoDB pipeline uses the `$text` operator, which **requires** a text
index on the target field; MongoDB allows at most one text index per
collection.  The original Paper-3B harness's backbone created only
`bb_type` + `bb_label` — no text index — so every Q16 invocation
failed (`text index required for $text query`, errno 27) under every
strategy.  The advisor cannot recover this case in practice: Q16 fails
fast → tiny `exec_ms` → never in the advisor's top-3-slowest
candidate set, so the compound-text index is never proposed.  The fix
treats the text index as **structural backbone** (matching
`PHASE3_PLAN.md` §5 "PK + FK structural baseline" — what a DBA would
set up before the advisor runs):

```python
coll.create_index([("abstract", "text")], name="bb_abstract_text")
```

**Fix 2 — `invoke_advisor()` now checks key signature, not just name.**
The original check (`if name in existing: continue`) only catches
name collisions.  But MongoDB also rejects (errno 85,
`IndexOptionsConflict`) when the candidate's key signature duplicates
an existing index under a different name — e.g. the advisor's
`adv_typea` (key `{type: 1}`) vs the backbone `bb_type` (same key).
Pre-fix, the harness logged a "failed:" message and continued, but
the noise polluted the log and `advisor_calls` counted attempts that
created nothing.  The fix builds a canonical key-signature set of
existing indexes and skips candidates whose key is already covered
(under any name):

```python
existing_keys = {tuple(sorted(info["key"])) for info in
                 coll.index_information().values() if "key" in info}
...
if tuple(sorted(cand)) in existing_keys: continue
```

After both fixes, `advisor_calls` becomes "the number of times the
advisor would have created a genuinely new index" rather than "the
number of attempts (including no-ops blocked by existing indexes)" —
the more interpretable RQ5 metric.

**Re-smoke** (after fixes, full block × 5 strategies) is the gate
before the full T3.7 run.

## Pre-fix smoke findings (for record)

For comparison with the post-fix run:

| strategy | wall_qps | advisor_calls | T_A (ms) | ok/total |
|---|---|---|---|---|
| dci_gated | 7.81 | 6 | 28,657 | 453 / 480 |
| always_on | 6.71 | 24 | 31,835 | 453 / 480 |
| hsm_gated | 6.47 | 8 | 32,689 | 453 / 480 |
| periodic  | 6.36 | 8 | 32,579 | 453 / 480 |
| no_advisor | 4.51 | 0 |       0 | 453 / 480 |

Even with Q16 systematically failing, the policy ordering preserved
the RQ5 prediction: **on this MongoDB workload the advisor's compound
indexes do materially help queries** (no_advisor is the slowest, not
the fastest as on MySQL TPC-H SF 1 where the backbone covers most of
the workload).  The interesting cross-engine pattern: on PG/MySQL
TPC-H the advisor barely helps; on MongoDB the advisor matters — the
gate's *value* therefore differs per engine, which is part of the RQ5
finding.  The post-fix numbers are the official ones.
