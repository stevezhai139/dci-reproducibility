# Paper 3D — `code/end_to_end/postgres/` provenance

Phase 3 (T3.2). The PostgreSQL four-policy adaptation harness, vendored
into the Paper 3D tree and re-pointed onto the canonical kernel.

## Files

**`pg_adaptation.py`** — vendored from
`Paper 3A/Version 3/code/experiments/v2_10seed/07_adaptation_comparison.py`.
The four-policy adaptation loop (`no_advisor` / `always_on` /
`periodic` / `hsm_gated`), the built-in `createIndex`/`dropIndex`
advisor, the RCB block design, the per-window 5-D HSM scoring, and the
metrics (incl. `wall_qps` throughput) are kept **verbatim**. Only the
following Paper 3D re-point edits were applied:

- `sys.path` — re-pointed to the Paper 3D tree (`code/kernel/`,
  `code/end_to_end/`, the script dir);
- the kernel import — `from hsm_v2_core import …` → `from
  hsm_v2_kernel import …` (the single canonical kernel,
  `code/kernel/`). `hsm_v2` has the identical 12-arg signature, so
  `compute_hsm_breakdown()`'s call site is unchanged; the only
  behavioural difference is S_P (the canonical kernel's symmetrised
  FastDTW — see `cross_engine/PROVENANCE.md`);
- the TPC-H workload import — `import_module('01_run_tpch_10seeds')` →
  `from tpch_queries import …` (the vendored query module);
- `ensure_database()` — the create-and-load path replaced by a
  verify-only check (Paper 3D expects TPC-H pre-loaded on PostgreSQL);
- `RESULTS_DIR` → `code/end_to_end/postgres/out/`;
- `write_run_meta` `file_shas` — updated to the vendored filenames.

**`tpch_queries.py`** — vendored **verbatim** from
`01_run_tpch_10seeds.py` lines 57–202: `QUERIES`, `QUERY_TABLES`,
`QUERY_COLS` (12 TPC-H query templates + table/column metadata).

## Status — T3.2 sandbox build complete

Vendor + re-point + DCI-gate integration: **done**; `pg_adaptation.py`
compiles + imports clean (kernel = the canonical `hsm_v2_kernel`).

DCI-gate integration applied (engine-agnostic pattern — the same
surgical changes transfer to the MongoDB / MySQL harnesses):

- `STRATEGIES` → 5 — `…/hsm_gated` (the θ-gate baseline) **plus**
  `dci_gated`; one harness run is the θ-vs-DCI head-to-head, RCB-paired;
- `calibrate_dci_gate()` — the steady-observation calibration pass:
  run `DCI_N_CAL = 64` phase-0 windows, fit `DCIGate` once per
  (engine, SF), freeze μ̂₀/Σ̂₀/F-thresholds (theory §11);
- `run_block` — `dci_gated` routes through `DCIGate.decide()` on the
  per-window 5-D `[S_R..S_P]` vector; the gate trajectory is reset at
  the start of each block;
- per-window throughput — `wall_ms_window`, `qps_window` added to
  `breakdown_rows` (the block-level `wall_qps` is unchanged);
- `run_meta.json` `constants` — `DCI_TAU`/`DCI_ALPHA`/`DCI_N_CAL`
  recorded (`write_run_meta`, added during T3.3 so this harness's
  provenance stays symmetric with the MongoDB harness's `run_meta`).

**Remaining — a live-PostgreSQL smoke test:**
`python3 pg_adaptation.py --sf 1.0 --blocks 1` (investigator; needs
`psycopg2` + a live PG with TPC-H loaded). The integration is
structurally verified (compiles, imports, signatures); the DCI gate
itself is unit-tested 5/5 (`code/end_to_end/test_dci_gate.py`).
