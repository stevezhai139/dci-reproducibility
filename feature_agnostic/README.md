# feature_agnostic/ — Paper 3C feature-representation robustness (kernel-free)

Shows **DCI predicts 1-D-vs-multi-D detector sufficiency in ANY feature
representation** — not just the HSM-tradition kernel used in the main eval.
All results here reproduce from this repo's code + data only (no 3A/3D paths).

## Files
| file | what |
|---|---|
| `features_alt.py` | kernel-free extractors: generic 3-axis similarity, raw template-frequency vector, table-incidence bag. **No `import kernel`.** |
| `sweep_agnostic.py` | canonical sweep: 3 workloads × {template,volume,mixed} × NSEED × 4 representations → DCI + 1-D/multi-D AUC per cell. Includes the HSM-5D baseline (the HSM-tradition instance) for reference. |
| `sweep_alt.py` | fast variant: the 3 kernel-free representations only (skips the HSM kernel; seconds not minutes). |
| `sdss_adapter.py` | real-trace anchor: SkyServer log (`../data/sdss/SkyLog_Workload.csv`) → per-block DCI. |
| `out/` | CSV outputs. |

## Run (from this directory; paths auto-detected)
Full table (HSM baseline + 3 kernel-free reps), 50 seeds:

    NSEED=50 python3 sweep_agnostic.py

Kernel-free reps only (fast):

    NSEED=50 python3 sweep_alt.py

Real-trace DCI (SkyServer):

    python3 sdss_adapter.py --csv ../data/sdss/SkyLog_Workload.csv --block 50

## Reading the result
Per cell: `DCI  1D/mD`. Low DCI (<~1.5) ⇒ 1-D AUC ≈ multi-D AUC (1-D suffices);
high DCI ⇒ multi-D >> 1-D (fusion needed). This holds in every representation —
bounded-similarity reps (HSM-5D, generic-sim3) place ordinary drift at low DCI;
raw reps (raw-freq, table-bag) push it high. Same router, different operating point.

## Kernel version
Numbers depend on the HSM kernel version (`../kernel/`, see `VERSION_PIN.txt`).
Pin it and regenerate all paper numbers from the same pinned kernel.
