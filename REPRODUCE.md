# Paper 3D — Reproduction Guide

This is the master run-order guide for reproducing Paper 3D's results
from the vendored code tree. Paper 3D asks whether the DCI-routed gate
is **engine-free**: does the Drift Complexity Index — its value, its
regime map, and its routing threshold τ — transfer across database
data models without per-data-model recalibration?

The guide covers **Phase 1** (canonical kernel + the Paper 3C baseline
reproduction) and **Phase 2** (the cross-data-model signal test,
research questions RQ1–RQ4). Phase 3 (the end-to-end three-engine RQ5)
will be appended when complete.

No database engine is exercised anywhere in Phase 1 or Phase 2 — the
HSM kernel is *pre-execution* (it reads SQL/command text and arrival
timestamps, never runs a query). Everything below is pure Python.

---

## 1. Environment

- **Python** 3.10 or newer (the official runs used Python 3.13.3 on
  the pinned machine; the build/smoke environment used 3.10.12 —
  results are bit-identical across both).
- **Packages:** `numpy`, `scipy`, `pandas`, `fastdtw`, `PyWavelets`,
  and `matplotlib` (the last is needed only for `cost_benefit.py`'s
  optional figure). Install with:

  ```
  pip install -r requirements.txt
  ```

  Two files at the code-tree root pin the environment:
  `requirements.txt` is the dependency list; `environment-lock.txt`
  records the **exact resolved versions** of the official runs
  (Python 3.13.3, numpy 2.4.6, scipy 1.17.1, pandas 3.0.3,
  fastdtw 0.3.4, PyWavelets 1.9.0, matplotlib 3.10.9) — use it to
  recreate the official environment precisely.
- **Official machine:** the numbers that enter the paper were produced
  by the investigator on the pinned Apple M4 MacBook Pro. A different
  machine may reproduce them bit-identically (the pipeline is
  deterministic — see §2) but the *official* artifacts are the M4 runs.

## Configuration

The `end_to_end/` runners read the PostgreSQL role from the `PGUSER`
environment variable (default `postgres`; host/port default to
`localhost:5432`). Copy `config.example.sh` to `config.local.sh`, set your
`PGUSER`, and `source config.local.sh` before running. No personal usernames
or paths are baked into the code.

## 2. Determinism & standing rules

Every experiment is **fully deterministic**: each `(workload, config,
seed)` cell is keyed by a stable SHA-256 seed (`stable_seed()` in
`cost_benefit.py`), so a re-run reproduces prior numbers exactly, not
merely in distribution.

On a fixed platform a re-run is **bit-identical** — the spectral
probe's DCI cross-check against the official RQ1/RQ2 runs is exact
(max |diff| = 0.00e+00). Across CPU architectures, results reproduce
to **machine epsilon** (~1e-15): LAPACK eigensolvers differ by a few
ULP between architectures, which perturbs a handful of recomputed
values in their 15th–16th significant digit. All verdicts, counts,
AUCs (rank-based, hence ULP-immune) and every reported figure are
unaffected — re-running `rq34_analysis.py` against the official probe
on a different machine reproduces every RQ3/RQ4 verdict, with only 4
floating-point values shifting at the 1e-15 level.

The project's standing rules (`HARNESS_CONTRACT.md` §6–§9):

1. **No extrapolation.** Every claimed number comes from running the
   full claimed configuration (50 seeds × 3 configs × the workload
   set) — never inferred from a subset. Only mathematical proofs are
   exempt.
2. **Official numbers are the investigator's.** The 50-seed runs that
   enter the paper are produced on the pinned M4; a sandbox is used
   only to build and smoke-test.
3. **Timestamped outputs.** Every output record carries an ISO-8601
   UTC `analysis_timestamp_utc` and a `run_id`; every run writes to a
   fresh `out/<run_id>/` directory.
4. **Everything vendored with checksums** for the release.

## 3. Integrity check (run first)

Each code/data directory carries a `CHECKSUMS.txt`. Verify there is no
local drift before reproducing anything:

```
cd code
for d in kernel cross_engine geometry_E0 phase2 sdss data/job data/sdss; do
    ( cd "$d" && sha256sum -c CHECKSUMS.txt ) || echo "DRIFT in $d"
done
```

`kernel/CHECKSUMS.txt` is an annotated vendoring record (Paper 3A
v5.0.0 snapshot) rather than a plain `sha256sum -c` list — see the
header inside that file; `kernel/hsm_similarity.py` must hash to the
"EXPECTED" value documented there (one sanctioned relative-import
rewrite vs the v5.0.0 source).

## 4. Run order

All commands are run from `code/`. Each script auto-detects the latest
`out/<run_id>/` of its inputs, so running the steps in order just
works; pass explicit paths only to pin a specific run.

### Phase 1 — canonical kernel + Paper 3C baseline

**1.1 — Reproduce the Paper 3C cost/benefit run (50 seeds).**

```
cd geometry_E0
python3 cost_benefit.py --seeds 50 --job-dir ../data/job/queries
```

`--job-dir` must point at the vendored JOB corpus (`code/data/job/
queries`, 113 queries) so the run is self-contained within Paper 3D.
Writes `geometry_E0/out/<run_id>/` — `cost_benefit_raw.csv` (per-cell
DCI + detection AUCs), `cost_benefit_summary.csv`, `cost_benefit_run.json`.

**1.2 — Verify it reproduces the locked Paper 3C reference.**

```
python3 compare_to_3c.py --run out/<run_id from 1.1> \
                         --ref out/20260522T145920Z
```

`out/20260522T145920Z` is the locked Paper 3C reference run, vendored
into the tree. Expected verdict: **REPRODUCED** (bit-identical). See
`T2_REPRODUCTION.md` for the full investigator protocol.

*(`verify_kernel_equivalence.py` — the T2.1 V3-vs-canonical-kernel
cross-check — needs the external Paper 3A V3 source and is therefore
not part of the standalone Paper 3D reproduction path; its finding
(the canonical kernel's symmetrised FastDTW S_P) is recorded in
`cross_engine/PROVENANCE.md`.)*

### Phase 2 — cross-data-model signal test (RQ1–RQ4)

**2.1 — MongoDB signal-layer DCI (document data model, 50 seeds).**

```
cd ../cross_engine
python3 mongo_signal.py --seeds 50
```

Writes `cross_engine/out/<run_id>/mongo_signal_{raw,summary}.csv` +
`_run.json`. Imports Paper 3C's `build_trajectory` + `analyse`
unchanged, so MongoDB DCI is directly comparable to the relational DCI.

**2.2 — SDSS real-world log adapter.**

```
cd ../sdss
python3 sdss_adapter.py
```

Reads the vendored SkyServer log (`code/data/sdss/SkyLog_Workload.csv`,
99,969 queries) → per-block DCI. Writes `sdss/out/<run_id>/sdss_dci.csv`,
`sdss_windows.csv`, `sdss_run.json`.

**2.3 — Assemble the cross-data-model DCI table.**

```
cd ../phase2
python3 collect_dci.py
```

Assembles 1.1 + 2.1 + 2.2 into one table,
`phase2/out/<run_id>/cross_data_model_dci.csv`. Assembles official
numbers; makes no claim.

**2.4 — RQ1 + RQ2 analysis.**

```
python3 rq12_analysis.py
```

Writes `rq12_results.json`. Expected: **RQ1 PASS** (MongoDB DCI within
the relational spread for every config — no new regime) and **RQ2
PASS** (one threshold τ = 1.5, empty band [1.411, 1.753], separates
1-D-sufficient cells across both data models; pooling the document
engine does not shrink the band).

**2.5a — Spectral probe (RQ3/RQ4 intermediates, 50 seeds).**

```
python3 spectral_probe.py --seeds 50
```

Re-derives the drift-covariance eigenvalue spectrum and the
steady-state covariance Σ — quantities `analyse()` computes internally
and discards. Asserts its re-derived DCI is bit-identical to
`analyse()` and to the official RQ1/RQ2 CSVs. Expected: **600/600
cells, 0 bound violations, DCI cross-check max |diff| = 0.00e+00**.
Writes `spectral_probe_cells.csv`, `spectral_probe_dev.npz`,
`spectral_probe_run.json`. (See `T2.5_SPECTRAL_PROBE.md`.)

**2.5b — RQ3 + RQ4 analysis.**

```
python3 rq34_analysis.py
```

Writes `rq34_results.json`. Expected: the own-Σ AUC anchor reproduces
the official `auc_5d`/`auc_1d` bit-identically; **RQ3 PASS** (both
dominant-mode bounds hold for all 600 spectra + all 100 SDSS real-log
blocks; band position data-model-consistent); **RQ4** — Σ is
structurally per-workload (correlation Frobenius distance ≈ 0.37) but
the detector is operationally robust (cross-data-model Σ costs ≤ 0.002
AUC). Net: the engine-free property is scoped to the router (τ), not
the detector.

## 5. Official run manifest

The directories below are the **official** artifacts; a reproduction
should match them. (Determinism means a faithful re-run reproduces the
per-cell numbers exactly.)

| step | script | official `out/` run | key output |
|---|---|---|---|
| 3C locked reference | *(vendored from Paper 3C)* | `geometry_E0/out/20260522T145920Z` | `cost_benefit_raw.csv` |
| 1.1 — 3C reproduction | `cost_benefit.py` | `geometry_E0/out/20260524T124846Z` | `cost_benefit_*.csv` |
| 2.1 — MongoDB signal | `mongo_signal.py` | `cross_engine/out/20260524T153052Z` | `mongo_signal_*.csv` |
| 2.2 — SDSS adapter | `sdss_adapter.py` | `sdss/out/20260524T153138Z` | `sdss_dci.csv` |
| 2.3 — collect table | `collect_dci.py` | `phase2/out/20260524T153546Z` | `cross_data_model_dci.csv` |
| 2.4 — RQ1/RQ2 | `rq12_analysis.py` | `phase2/out/20260524T154224Z` | `rq12_results.json` |
| 2.5a — spectral probe | `spectral_probe.py` | `phase2/out/20260524T161959Z` | `spectral_probe_*.{csv,npz}` |
| 2.5b — RQ3/RQ4 | `rq34_analysis.py` | `phase2/out/20260524T162121Z` | `rq34_results.json` |

**Non-official runs.** `phase2/out/` also contains sandbox
smoke-test and verification re-run directories (3-seed probe runs,
repeated analysis runs). They are **not official**: the official
artifacts are exactly the eight directories in the table above, and
any other `out/<run_id>/` may be deleted for a clean release. The
`latest_run` auto-detection picks the directory holding the relevant
file; pass an explicit input path (e.g. `--probe-dir`) to pin a
specific run.

## 6. Code tree

| directory | role |
|---|---|
| `kernel/` | the canonical HSM kernel, vendored from Paper 3A v5.0.0 (checksum-verified) |
| `geometry_E0/` | Paper 3C's DCI / cost-benefit code (`cost_benefit.py`, `delong.py`) + Paper 3D's Phase-1 verification scripts. Phase 2 imports only `cost_benefit.py` and `delong.py`; the other vendored 3C scripts are not on the Paper 3D path |
| `cross_engine/` | the MongoDB feature path harmonised onto the canonical kernel + `mongo_signal.py` |
| `sdss/` | the SDSS real-world-log adapter |
| `phase2/` | `collect_dci.py`, `rq12_analysis.py`, `spectral_probe.py`, `rq34_analysis.py` |
| `data/job/` | the vendored JOB query corpus (113 queries) |
| `data/sdss/` | the vendored SkyServer query log |

Findings: `PHASE2_FINDINGS.md`. Project status: `PROGRESS.md`.

---
*Phase 1 + Phase 2 reproduction path complete. Phase 3 (end-to-end
three-engine RQ5) will be appended to this guide when complete.*
