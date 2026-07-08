# Paper 3C — Reproduction Guide

This repository reproduces every number, figure, and table of *"The Drift
Complexity Index"* (Paper 3C): the DCI statistic and its bounds, the regime
map, the routed selector, the kernel-free replications, the SkyServer trace
study, and the live end-to-end deployments on PostgreSQL and MongoDB.

## 1. Environment

- **Offline pipeline** (everything except §6.8-live): Python ≥ 3.10 with
  `pip install -r requirements.txt` (numpy, scipy, pandas, fastdtw,
  PyWavelets, matplotlib). `environment-lock.txt` records the exact resolved
  versions of the official runs (CPython 3.13.3 on the pinned Apple M4
  MacBook Pro, 24 GB, macOS 26).
- **Live end-to-end** (§6.8): additionally `pip install -r
  requirements-e2e.txt` (psycopg2-binary, pymongo), plus PostgreSQL 16.9
  with the `tpch_sf1` database and the HypoPG extension, Dexter 0.6.3 on
  PATH, and MongoDB 8.0.8 serving `mydb_p3a.combined_clean` (5,000,000
  documents). Copy `config.example.sh` → `config.local.sh`, set `PGUSER`
  (and `DEXTER_BIN` if needed), and `source` it.
- **Readiness check (read-only):** `./part2_preflight.sh` verifies all of
  the above and prints the engine/driver versions used in the paper.

## 2. Determinism and provenance rules

Every offline experiment is fully deterministic: each (workload, config,
seed) cell is keyed by a stable SHA-256 seed, so re-runs reproduce prior
numbers exactly. The **official artifacts** are the investigator's runs on
the pinned machine, listed in §3; other `out/<run-id>/` directories are
previews or smoke tests and are not paper inputs. Live runs (§6.8) are
paired by construction — the three detector configurations replay identical
workload draws per block (`workload_fp` asserted by the analyzer) — but
their absolute latencies are machine-specific.

## 3. Paper artifact → script → official run

| Paper artifact | Script(s) | Official output |
|---|---|---|
| Fig. 1, Table 2 (regime map), Fig. 2 (cost–accuracy), per-cell tolerance, E0 detector costs | `geometry_E0/cost_benefit.py` (+ `delong.py`, `seed_analysis.py`) | `geometry_E0/out/20260705T135856Z/` (`cost_benefit_run.json`: 1-D 0.0087 ms, multi-D 2.3486 ms per window) |
| Table 4 (ablation: PR vs alternatives) | `geometry_E0/ablation_complexity.py` | `geometry_E0/out/ablation/` |
| Fig. 3 (τ robustness) | `eval_expansion/plot_tau.py`, `analyze_tau_bootstrap.py` | `eval_expansion/tau_sweep.csv`, `eval_expansion_results.json` |
| §4 $O(D^2)$-vs-$O(D^3)$ scaling | `d_scaling_bench.py` | `d_scaling_bench.csv` (closed form = spectrum to $10^{-13}$; $3.5\times$ at $D{=}5$ → $105\times$ at $D{=}500$) |
| Fig. 4 (principled frontier) | `eval_expansion/blind_spot_probe.py` → `plot_blindspot.py` | `fig_wellfounded.pdf` |
| Table 3 (kernel-free representations) | `feature_agnostic/sweep_agnostic.py`, `sweep_alt.py` | `feature_agnostic/out/` per-cell CSVs (see its README) |
| SkyServer trace (§6.2) | `feature_agnostic/sdss_adapter.py` → `sdss_routing_cost.py` | `feature_agnostic/out/20260705T135649Z/sdss_dci.csv` (100 blocks, 10 below τ, mean 1.850, selector 2.115 ms = 90.0%) |
| §6.8 live, ordinary schedule (Table 5) | `end_to_end/postgres/pg_adaptation.py`, `end_to_end/mongo/mongo_adaptation.py` | `end_to_end/postgres/out_PG_{gated_tau1p5,always1D_tau1e9}/`, `end_to_end/mongo/out/MG_*/` |
| §6.8 escalation fractions (S2) | `escalation_replay.py` | prints PG 0.1458/0.0682, MG 0.2083/0.1364 (deterministic replay of the logged runs) |
| §6.8 firing-level sweep (offline) | `dci_resolution_e2e.py --seeds 50` | `dci_resolution_e2e_results.csv` (mixed recall 0.694/0.931/0.904; gated at 45.3% of multi-D cost) |
| Whitened refinement §(whitened): DCI$_w$ probe | `whitened_dci.py . 50` | `whitened_dci_results_v2.csv`, `whitened_dci_summary_v2.json` (raw = Table 2 exactly; DCI$_w$ ≤ DCI on all 9 cells) |
| Whitened refinement §(whitened): full sufficiency | `s1_whitened_sufficiency.py . 50` | `s1_whitened_sufficiency.csv`, `s1_whitened_sufficiency_summary.json` (whitened matched filter sufficient in all 9 cells under the map's δ rule; ridge-stable on $[10^{-8},10^{-6}]$) |
| §6.8 live mixed drift (Table 6) | `part2_preflight.sh` → `RUN_PART2.sh` → `part2_analyze.py` (schedule pre-check: `part2_validate_offline.py`) | `end_to_end/postgres/out_PART2_PG_mixed_{tau0,tau1p5,tau1e9}/`, `end_to_end/mongo/out/PART2_MG_mixed_*/`, `part2_summary.csv` |

Expected headline numbers for each artifact are embedded as `% Provenance`
comments next to the corresponding table/figure in the paper source and in
the per-directory `PROVENANCE.md`/`README.md` files.

## 4. Run order

**Offline-only (no database; reproduces §4–§6.7 and the offline halves of §6.8):**
1. `python3 geometry_E0/cost_benefit.py --seeds 50` (regime map + costs)
2. `cd eval_expansion && python3 blind_spot_probe.py && python3 plot_blindspot.py`
3. `cd feature_agnostic && python3 sweep_agnostic.py && python3 sdss_adapter.py && cd .. && python3 sdss_routing_cost.py .`
4. `python3 dci_resolution_e2e.py . --seeds 50`
5. `python3 escalation_replay.py .` (replays the shipped live logs)

**Live end-to-end (§6.8; PostgreSQL + MongoDB on localhost):**
1. `./part2_preflight.sh` — must end `READY`
2. ordinary schedule: `end_to_end/RUN_T37_PG.sh` conventions (see
   `end_to_end/postgres/PROVENANCE.md`) — or accept the shipped official logs
3. mixed schedule: `PART2_BLOCKS=1 ./RUN_PART2.sh` (smoke), then
   `PART2_FORCE=1 ./RUN_PART2.sh` (full, ~half a machine-day)
4. `python3 part2_analyze.py .`

## 5. Code tree

| directory | role |
|---|---|
| `kernel/` | the canonical workload-similarity kernel (checksum-pinned) |
| `geometry_E0/` | DCI, regime map, cost–benefit, tolerance machinery |
| `eval_expansion/` | τ-robustness sweep and the principled-frontier probe |
| `feature_agnostic/` | kernel-free representations + the SkyServer adapter |
| `cross_engine/` | the document-store (MongoDB) feature path |
| `end_to_end/` | the DCI gate + live PostgreSQL/MongoDB harnesses (ordinary + mixed schedules) |
| `data/job/`, `data/sdss/` | vendored JOB corpus and SkyServer query log |

## 6. Cleanliness

Official run directories are exactly those named in §3; any other
`out/<run-id>/` may be deleted for a clean checkout. Development history
comments inside harness files reference the research program's internal
task names; they do not affect any result.
