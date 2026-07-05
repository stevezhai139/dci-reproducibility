# Paper 3C — reproducibility (the DCI-routed drift gate)

The **Drift Complexity Index (DCI)** and the DCI-routed selector: a cheap always-on drift detector
that runs the 1-D matched filter when DCI is low and the 5-D Mahalanobis detector when it is high,
gating an index advisor. **This repository is DCI-only** — the sibling economics study's baseline
policies, MySQL harness, and analysis are not included.

## 0. One-time cleanup
The DCI-only tree was assembled by an assistant that could not delete files on the mounted volume.
Run once:
```bash
bash CLEANUP_3D.sh          # removes the leftover sibling (3D) files
```

## 1. The gate
`end_to_end/dci_gate.py` — **standalone**, no external gate dependency. Routes on the **raw**
participation ratio (the DCI of the paper; identical to `geometry_E0/cost_benefit.analyse`):
`DCI = tr(C)^2 / ||C||_F^2`, `C` the raw drift-motion covariance. Below `tau=1.5` → 1-D matched
filter; at/above → 5-D Mahalanobis; each fired at the exact Hotelling-F threshold.
```python
from dci_gate import DCIGate
g = DCIGate(tau=1.5, alpha=0.05).fit(steady_5d_rows)   # unsupervised, frozen
g.reset_trajectory()                                   # per RCB block
verdict = g.decide(window_5d_vector)                   # 1 = invoke advisor, 0 = skip
```

## 2. Setup
Python 3.11+, `numpy scipy pandas fastdtw pywavelets` (+ `psycopg2` for PostgreSQL,
`pymongo` for MongoDB). See `requirements.txt`.

## 3. Reproduce the DETECTION results (no database)
Regime map / DCI bounds / tau-sensitivity / ablation / cross-data-model (MongoDB) are pure Python
(the kernel is pre-execution). The scripts live in `Paper 3C/eval_expansion/`:
- regime map + tau + bootstrap: `analyze_tau_bootstrap.py`
- ablation (why the participation ratio): `ablation_complexity.py`
- cross-data-model MongoDB detection: `cross_engine/mongo_signal.py --seeds 50`
- detector-resolution tau-sweep (offline): `eval_expansion/end_to_end/dci_resolution_offline.py --engine {pgbench,mongo}`

## 4. Reproduce the END-TO-END result (live PostgreSQL + MongoDB, 10 blocks RCB)
The DCI-routed advisor vs the seeded ground-truth drift. `tau=1.5` = DCI-gated; `tau=1e9` =
always-1-D (never escalates) — the contrast shows 1-D's limitation on complex drift.
```bash
cd end_to_end
# PostgreSQL (TPC-H, SF 1.0)
DCI_TAU=1.5  python3 postgres/pg_adaptation.py --workload tpch --blocks 10 --sf 1.0
DCI_TAU=1e9  python3 postgres/pg_adaptation.py --workload tpch --blocks 10 --sf 1.0
# MongoDB
DCI_TAU=1.5  python3 mongo/mongo_adaptation.py --blocks 10
DCI_TAU=1e9  python3 mongo/mongo_adaptation.py --blocks 10
```
Only `dci_gated` runs (hardcoded). Per-block metrics land in `postgres/out/` and `mongo/out/`.

## Layout
```
end_to_end/dci_gate.py           the DCI-routed gate (standalone, raw DCI)
end_to_end/postgres/             PG harness + TPC-H/JOB workloads
end_to_end/mongo/                MongoDB harness + ESR advisor
kernel/                          HSM workload-similarity kernel (prior work; black-box feature source)
geometry_E0/                     DCI + regime-map analysis (cost_benefit.py)
cross_engine/                    MongoDB signal-layer detection + shared feature code
```
