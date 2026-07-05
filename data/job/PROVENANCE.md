# JOB / IMDB queries — vendored snapshot

**What.** The 113 Join Order Benchmark (JOB) query files (`<n><letter>.sql`),
the analytic SQL workload over the IMDB schema used by Paper 3C's
`cost_benefit.py` and `kernel_validation.py`.

**Vendored on:** 2026-05-24 (Paper 3D Phase 1, task #22).
**Vendored from:** `Paper 3A/Version 5/HSM_gated/code/data/job/queries/`.
**Why:** Paper 3D's self-containedness policy (`ENGINE_FREE_SCOPING.md`
§7) requires every workload artifact to live inside the Paper 3D code
tree, checksummed, so the GitHub release is reproducible by a reviewer
without any other repository.

**Integrity.** `CHECKSUMS.txt` (one directory up) records the SHA-256 of
every `.sql` file. Verify with:

```bash
cd code/data/job
sha256sum -c CHECKSUMS.txt
```

All 113 lines must report `OK`.

**Usage.** Passed to `cost_benefit.py` via `--job-dir`:

```bash
cd code/geometry_E0
python3 cost_benefit.py --seeds 50 --job-dir ../data/job/queries --outdir out
```
