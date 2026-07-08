#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  RUN_PART2.sh — Paper 3C, S5 Part 2: LIVE mixed-drift 3-τ sweep
#  (PART2_BUILD_SPEC.md, 2026-07-07)
#
#  Matrix: {PG(tpch_mixed), MG(--schedule mixed)} × DCI_TAU ∈ {0, 1.5, 1e9}
#          × PART2_BLOCKS paired-RCB blocks (default 10).
#
#  τ semantics (same DCIGate, same feature stream, three routing settings):
#      DCI_TAU=0    → always-5-D  (safe/expensive detector reference)
#      DCI_TAU=1.5  → DCI-gated   (the paper's method)
#      DCI_TAU=1e9  → always-1-D  (cheap detector once estimable)
#
#  Pre-flight (BOTH already done in-sandbox, re-runnable here):
#      python3 part2_validate_offline.py .        # schedule DCI pre-validation
#  Smoke first (recommended, ~minutes):
#      PART2_BLOCKS=1 ./RUN_PART2.sh
#  Full run (paper numbers; ~half a machine-day):
#      ./RUN_PART2.sh
#  One engine only:
#      PART2_ENGINES=pg ./RUN_PART2.sh     (or =mg)
#
#  Axis discipline (3C vs 3D): this run yields detection fidelity, firings,
#  detector monitoring cost and query-latency fidelity ONLY. No wall_qps,
#  no sign-flip, no regret — those belong to the 3D/journal axis.
#
#  Provenance: paper numbers come ONLY from this script's outputs on the
#  locked env (Apple M4, pinned Python). Outputs:
#      end_to_end/postgres/out_PART2_PG_mixed_<tag>/
#      end_to_end/mongo/out/PART2_MG_mixed_<tag>/<ts>/
#  Then:  python3 part2_analyze.py .
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

BLOCKS="${PART2_BLOCKS:-10}"
ENGINES="${PART2_ENGINES:-pg mg}"
TAUS=("0" "1.5" "1e9")
TAGS=("tau0" "tau1p5" "tau1e9")

# PG connection config (PGUSER etc.) — see config.example.sh
[ -f "$ROOT/config.local.sh" ] && source "$ROOT/config.local.sh"

echo "═══ S5 Part 2 live sweep — blocks=$BLOCKS engines=[$ENGINES] ═══"
date -u +"%Y-%m-%dT%H:%M:%SZ"

cd "$ROOT/end_to_end"

for i in 0 1 2; do
  TAU="${TAUS[$i]}"; TAG="${TAGS[$i]}"

  # ── PostgreSQL: tpch_mixed on tpch_sf1 (Dexter advisor) ──────────────
  if [[ " $ENGINES " == *" pg "* ]]; then
    DEST="postgres/out_PART2_PG_mixed_${TAG}"
    if [ -e "$DEST" ] && [ "${PART2_FORCE:-0}" != "1" ]; then
      echo "── PG $TAG: $DEST exists — skipping (set PART2_FORCE=1 to redo)"
    else
      [ -e "$DEST" ] && rm -rf "$DEST"
      echo "── PG  DCI_TAU=$TAU → $DEST  ($(date -u +%H:%M:%SZ))"
      # NOTE: postgres/out/ is the live working dir; a crashed run can be
      # resumed by re-running (same τ) — the harness resumes per-SF CSVs
      # keyed by PROBE_TAG, so τ configs never cross-contaminate.
      DCI_TAU="$TAU" PROBE_TAG="_p2mx_${TAG}" \
        python3 postgres/pg_adaptation.py --workload tpch_mixed --sf 1.0 \
                --blocks "$BLOCKS"
      mv postgres/out "$DEST"
      echo "── PG $TAG done → $DEST"
    fi
  fi

  # ── MongoDB: --schedule mixed (real ESR recommender, full baseline) ──
  # ESR = the real workload recommender (ARC: "advisor จริง Dexter/ESR").
  # Baseline stays "full" so every query is RUNNABLE under every τ —
  # latency fidelity must not be contaminated by structural $text/geo
  # failures ("under" is the 3D churn story, not this axis).
  if [[ " $ENGINES " == *" mg "* ]]; then
    SUB="PART2_MG_mixed_${TAG}"
    if compgen -G "mongo/out/${SUB}/*" > /dev/null && [ "${PART2_FORCE:-0}" != "1" ]; then
      echo "── MG $TAG: mongo/out/${SUB}/ has runs — skipping (PART2_FORCE=1 to redo)"
    else
      echo "── MG  DCI_TAU=$TAU → mongo/out/${SUB}/  ($(date -u +%H:%M:%SZ))"
      DCI_TAU="$TAU" \
        python3 mongo/mongo_adaptation.py --blocks "$BLOCKS" \
                --schedule mixed --advisor esr \
                --results-subdir "$SUB"
      echo "── MG $TAG done"
    fi
  fi
done

cd "$ROOT"
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "═══ Part 2 sweep complete. Next: python3 part2_analyze.py . ═══"
