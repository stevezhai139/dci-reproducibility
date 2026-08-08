#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  RUN_PART2_V3.sh — Paper 3C: LIVE mixed-drift rerun with DCIGateV3
#  (PART2_V3_RERUN_SPEC.md, 2026-08-08; decision A = measured conditional
#   extraction — det_ms IS the paper's live monitoring cost)
#
#  Matrix: {PG(tpch_mixed), MG(--schedule mixed)} × DCI_RHO ∈ {2.0, 0.35, 0.0}
#          × PART2_BLOCKS paired-RCB blocks (default 10).
#
#  ρ semantics (same DCIGateV3, same feature stream, three routing arms):
#      DCI_RHO=2.0   → always-FULL  (R≤1<ρ; 5-D + S_P every window — reference)
#      DCI_RHO=0.35  → GATED        (the paper's method)
#      DCI_RHO=0.0   → always-CHEAP (union-Bonferroni only; S_P never)
#
#  Smoke first (recommended, ~minutes):
#      PART2_BLOCKS=1 PART2_ENGINES=pg ./RUN_PART2_V3.sh
#  Full run (paper numbers; ~half a machine-day):
#      ./RUN_PART2_V3.sh
#
#  Provenance: paper numbers ONLY from this script's outputs on the locked
#  env. Outputs:
#      end_to_end/postgres/out_PART2V3_PG_mixed_<tag>/
#      end_to_end/mongo/out/PART2V3_MG_mixed_<tag>/<ts>/
#  Then:  python3 part2_analyze.py .   (v3-aware columns)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

BLOCKS="${PART2_BLOCKS:-10}"
ENGINES="${PART2_ENGINES:-pg mg}"
RHOS=("2.0" "0.35" "0.0")
TAGS=("rho2p0" "rho0p35" "rho0")

[ -f "$ROOT/config.local.sh" ] && source "$ROOT/config.local.sh"

echo "═══ Part 2 v3 live sweep — blocks=$BLOCKS engines=[$ENGINES] ═══"
date -u +"%Y-%m-%dT%H:%M:%SZ"

# Guard: MG requires the v3-patched mongo harness (mirror of pg patch).
if [[ " $ENGINES " == *" mg "* ]] && \
   ! grep -q "DCI_GATE_VERSION" "$ROOT/end_to_end/mongo/mongo_adaptation.py"; then
  echo "!! mongo_adaptation.py not v3-patched yet — run PG only for now:"
  echo "     PART2_ENGINES=pg ./RUN_PART2_V3.sh"
  exit 1
fi

cd "$ROOT/end_to_end"

for i in 0 1 2; do
  RHO="${RHOS[$i]}"; TAG="${TAGS[$i]}"

  # ── PostgreSQL: tpch_mixed on tpch_sf1 (Dexter advisor) ──────────────
  if [[ " $ENGINES " == *" pg "* ]]; then
    DEST="postgres/out_PART2V3_PG_mixed_${TAG}"
    if [ -e "$DEST" ] && [ "${PART2_FORCE:-0}" != "1" ]; then
      echo "── PG $TAG: $DEST exists — skipping (set PART2_FORCE=1 to redo)"
    else
      [ -e "$DEST" ] && rm -rf "$DEST"
      echo "── PG  DCI_GATE=v3 DCI_RHO=$RHO → $DEST  ($(date -u +%H:%M:%SZ))"
      DCI_GATE=v3 DCI_RHO="$RHO" PROBE_TAG="_p2v3_${TAG}" \
        python3 postgres/pg_adaptation.py --workload tpch_mixed --sf 1.0 \
                --blocks "$BLOCKS"
      mv postgres/out "$DEST"
      echo "── PG $TAG done → $DEST"
    fi
  fi

  # ── MongoDB: --schedule mixed (real ESR recommender, full baseline) ──
  if [[ " $ENGINES " == *" mg "* ]]; then
    SUB="PART2V3_MG_mixed_${TAG}"
    if compgen -G "mongo/out/${SUB}/*" > /dev/null && [ "${PART2_FORCE:-0}" != "1" ]; then
      echo "── MG $TAG: mongo/out/${SUB}/ has runs — skipping (PART2_FORCE=1 to redo)"
    else
      echo "── MG  DCI_GATE=v3 DCI_RHO=$RHO → mongo/out/${SUB}/  ($(date -u +%H:%M:%SZ))"
      DCI_GATE=v3 DCI_RHO="$RHO" \
        python3 mongo/mongo_adaptation.py --blocks "$BLOCKS" \
                --schedule mixed --advisor esr \
                --results-subdir "$SUB"
      echo "── MG $TAG done"
    fi
  fi
done

cd "$ROOT"
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "═══ Part 2 v3 sweep complete. Next: python3 part2_analyze.py . ═══"
