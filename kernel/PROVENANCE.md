# HSM kernel — vendored snapshot for paper3b_v2_experiments

**Vendored on:** 2026-05-01
**Vendored from:** `v2/paper3a_kernel/Paper_3A_code/experiments/cal/vendored/`
**Original origin (per upstream):** Paper 3A `Version 3/code/experiments/v2_10seed/hsm_v2_core.py`
+ `Version 3/code/src/hsm/measures.py`

## Files in this snapshot

| File | Purpose | sha256 (see VERSION_PIN.txt) |
|---|---|---|
| `hsm_v2_kernel.py` | Canonical 5-D HSM kernel implementation (S_R, S_V, S_T, S_A, S_P) | `e01a9d0e…` |
| `hsm_similarity.py` | Wrapper exposing similarity API used by validation scripts | `204891dc…` |
| `workload_generator.py` | Synthetic-workload generator used by simulation-based calibration of N_MIN_CHISQ + PURITY thresholds | `7a9e5584…` |
| `__init__.py` | Package init | `dab92aca…` |
| `CHECKSUMS.txt` | Upstream Paper 3A's own integrity record | (carried over) |
| `README.md` | Upstream Paper 3A's kernel documentation | (carried over) |

## Decoupling guarantee

This snapshot is **immutable** for the duration of the v2 experimental
campaign. If Paper 3A's kernel is later revised:

1. The revision will NOT propagate into v2 results automatically.
2. To intentionally upgrade, a new snapshot must be vendored here, the
   `VERSION_PIN.txt` re-generated, and **all v2 experiments re-run**.
3. The PROVENANCE.md must be updated to record the new vendor date and
   upstream origin SHA.

## Verification

To verify the vendored kernel matches the snapshot recorded in
`VERSION_PIN.txt`:

```bash
cd v2/paper3b_v2_experiments/kernel/
sha256sum -c VERSION_PIN.txt
```

All four lines should report `OK`.
