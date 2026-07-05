# `kernel/` — vendored 5-D HSM kernel for Paper 3C

Frozen snapshot of the Paper 3A HSM kernel, copied verbatim into
`Paper 3C/` so the paper is self-contained and reproducible
independent of Paper 3A's TKDE revision cycle.

**Provenance.** Snapshot `v5.0.0-tkde-submission` (Paper 3A TKDE
submission, 2026-04-20). Copied into `Paper 3C/` on 2026-05-21 from
`Paper 3B/v2/paper3b_v2_experiments/kernel/` — itself a vendored copy
of the Paper 3A source. Full chain in `PROVENANCE.md`.

**Do not edit these files.** They are an immutable snapshot — see §5.

## 1. Files

| File | Purpose |
|---|---|
| `hsm_v2_kernel.py` | Canonical 5-D HSM kernel: component functions `sr_v2 sv_v2 st_v2 sa_v2 sp_v2`, the `hsm_v2()` kernel, the DWT / SAX / FastDTW multi-scale machinery, and `W0` (default weight vector) |
| `hsm_similarity.py` | High-level API: `QueryFeatures`, `WorkloadWindow`, `build_window()`, the per-dimension similarities `s_r s_v s_t s_a s_p`, `hsm_score()`, `should_trigger_advisor()`, `DEFAULT_THETA`, `DEFAULT_WEIGHTS` |
| `workload_generator.py` | TPC-H phase-shifted trace generator: `get_workload_trace()`, `get_window_queries()`, `PHASE_A / PHASE_B / PHASE_C` |
| `__init__.py` | Package marker |
| `CHECKSUMS.txt`, `VERSION_PIN.txt`, `PROVENANCE.md` | Integrity + provenance records |

## 2. Why this kernel matters for Paper 3C

Paper 3C's thesis is that the **5-D kernel is the safety-complete
envelope** and a **1-D inner core** (`s_t` or `s_r` alone) suffices on
the stationary majority of workloads — see
`../calculus_minimal_complexity/VERDICT.md` for the E0 evidence.

The kernel already exposes **every dimension individually** (`s_t(...)`,
`s_r(...)`, `s_v`, `s_a`, `s_p`) *and* the full weighted `hsm_score(...)`.
So the adaptive selector of sketch §C2 — which switches between 1-D and
5-D per window — can be built **on top of this kernel without editing
it**. The DWT/SAX/FastDTW constants in `hsm_v2_kernel.py` are the
multi-scale machinery that E0 found empirically inert on abrupt drift;
3C characterises exactly when it earns its keep.

## 3. Import

With `Paper 3C/` on `sys.path` (the kernel is a package):

```python
from kernel.hsm_similarity import (
    WorkloadWindow, build_window, hsm_score, should_trigger_advisor,
    s_r, s_v, s_t, s_a, s_p, DEFAULT_THETA, DEFAULT_WEIGHTS,
)
from kernel.hsm_v2_kernel import hsm_v2, hsm_score_from_features, W0
from kernel.workload_generator import get_workload_trace, PHASE_A, PHASE_B, PHASE_C
```

`hsm_similarity.py` imports `hsm_v2_kernel` **relatively**
(`from .hsm_v2_kernel import ...`), so `kernel/` must always be
imported as a package — never run a module from inside the folder by
absolute path.

## 4. Integrity check

```bash
cd "Paper 3C/kernel"
sha256sum -c VERSION_PIN.txt      # all 4 lines must report: OK
```

Verified OK at copy time (2026-05-21). Expected hashes:

```
e01a9d0e…  hsm_v2_kernel.py
204891dc…  hsm_similarity.py
7a9e5584…  workload_generator.py
dab92aca…  __init__.py
```

## 5. Re-vendoring

Re-vendor **only** if Paper 3A ships a TKDE revision whose kernel
change Paper 3C semantically depends on. Procedure: copy the new
files, re-apply the 1-line relative-import rewrite in
`hsm_similarity.py`, regenerate `CHECKSUMS.txt` / `VERSION_PIN.txt`,
update `PROVENANCE.md`, and re-run E0. The full sync protocol is in
the original 3B vendoring doc:
`Paper 3B/v2/paper3b_v2_experiments/kernel/README.md` §4.
