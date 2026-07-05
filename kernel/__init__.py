"""Vendored Paper 3A kernel for Paper 3B-Cal.

Files here are exact copies of Paper 3A's v5.0.0-tkde-submission
artefacts. DO NOT edit them — if Paper 3A ships a revision that Paper
3B wants to adopt, re-vendor through the sync protocol described in
``README.md`` and update ``CHECKSUMS.txt`` accordingly.

Rationale (see ``README.md`` §1):
- Paper 3A is under TKDE review; its source files may evolve via
  revision branches (``revision-r1`` etc.).
- Paper 3B-Cal's empirical claims must be reproducible against a
  stable kernel, independent of Paper 3A revision cycles.
- Vendoring gives Paper 3B an immutable snapshot of the kernel to
  build on, with rollback safety if Paper 3A changes break Paper 3B.

Precedent: Paper 3A itself vendored ``cross_engine/`` from V3 for its
§V claim (design doc §14 Traceability).
"""
