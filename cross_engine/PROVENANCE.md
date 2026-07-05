# cross_engine/ — vendored MongoDB feature path

**What.** The cross-engine feature-extraction path used to turn a
MongoDB workload into the canonical `WindowFeatures` records the HSM
kernel consumes (see `HARNESS_CONTRACT.md` §3–§4):

```
cross_engine/
  common/
    __init__.py            verbatim from Paper 3A V5
    hsm_bridge.py           MODIFIED — re-pointed kernel (see below)
    window_features.py      verbatim from Paper 3A V5
    param_sampler.py        verbatim from Paper 3A V5
  mongo/workload/
    templates.py            verbatim from Paper 3A V5 (24 MongoDB templates)
```

**Vendored on:** 2026-05-24 (Paper 3D Phase 2, task T2.1).
**Vendored from:** `Paper 3A/Version 5/HSM_gated/code/experiments/cross_engine/`.

## The one modification — hsm_bridge.py kernel re-point

In Paper 3A, `hsm_bridge.py` imported the **V3 kernel**
(`_v3_hsm/hsm_v2_core.py`). Paper 3D re-points it at the **single
canonical kernel** (`code/kernel/hsm_v2_kernel.py`) so every engine in
Paper 3D computes DCI with one identical kernel (`HARNESS_CONTRACT.md`
§5; Decision 4 in `ENGINE_FREE_SCOPING.md` §9).

The change is small and safe:

- `hsm_bridge.py` calls the kernel only through `hsm_v2(...)`,
  positionally, and reads `W0`. Both the V3 and the canonical kernel
  export `hsm_v2` with the **same first 12 positional parameters** and
  export `W0`, so the re-point needs **no call-site change** — only the
  import path and module name.
- The V3-only functions (`extract_windows`, `compute_all_pairs`, …) are
  **not used** by the feature path.

## Numerical effect of the re-point (verified, T2.1)

`verify_kernel_equivalence.py` (kept in `code/geometry_E0/`) compared
both kernels' `hsm_v2` over 300 window pairs:

- **S_R, S_V, S_T, S_A** — identical to within V3's cosmetic 4-decimal
  rounding (the canonical kernel returns full precision).
- **S_P** — differs by up to 5×10⁻³ in ~2 % of pairs. Root cause
  (confirmed band-by-band): FastDTW is a direction-dependent
  *approximation* of DTW, so `fastdtw(a,b) ≠ fastdtw(b,a)`. The V3
  kernel called FastDTW one-directionally, making its S_P **not
  symmetric**. The canonical kernel averages both directions ("P3
  symmetrisation") — its S_P is the corrected, symmetric one.

The re-point therefore *upgrades* the MongoDB path's S_P to the
symmetric implementation. The downstream effect on DCI is quantified in
task T2.3.

## Not vendored, on purpose

- `_v3_hsm/` — the old V3 kernel is **not** vendored: the Paper 3D tree
  uses only the canonical kernel.
- `mongo/adaptation/` — the end-to-end four-policy adaptation scripts
  are a Phase 3 (RQ5) concern; they will be vendored then.

## Integrity

`CHECKSUMS.txt` records the SHA-256 of every vendored file. Note that
`hsm_bridge.py`'s checksum is for the **re-pointed** Paper 3D copy, not
the Paper 3A original.
