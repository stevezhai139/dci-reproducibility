#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  RUN_CONTRAST.sh — Paper 3C: live CONTRAST canary (PG only, 3 arms)
#
#  Two-population query-rewrite canary (design X0r; contrast_preflight.py
#  official 2026-08-12). Same DCIGateV3 + feature stream, route pinned:
#      DCI_FORCE=full   → always-FULL   (reference)
#      (unset)          → GATED         (the paper's method)
#      DCI_FORCE=cheap  → always-CHEAP  (union-Bonferroni)
#
#  Smoke:  CONTRAST_BLOCKS=1 ./RUN_CONTRAST.sh          (~4 min)
#  Full:   ./RUN_CONTRAST.sh                            (~2h10 total)
#  Lam:    CONTRAST_WL=tpch_contrast_l05 ./RUN_CONTRAST.sh
#  Then:   python3 contrast_analyze.py .
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
BLOCKS="${CONTRAST_BLOCKS:-10}"
WL="${CONTRAST_WL:-tpch_contrast_l06}"
FORCES=("full" "" "cheap")
TAGS=("afull" "gated" "acheap")
[ -f "$ROOT/config.local.sh" ] && source "$ROOT/config.local.sh"

echo "═══ Contrast live sweep — wl=$WL blocks=$BLOCKS ═══"
date -u +"%Y-%m-%dT%H:%M:%SZ"
cd "$ROOT/end_to_end"
for i in 0 1 2; do
  FORCE="${FORCES[$i]}"; TAG="${TAGS[$i]}"
  DEST="postgres/out_CONTRAST_PG_${WL#tpch_contrast_}_${TAG}"
  if [ -e "$DEST" ] && [ "${CONTRAST_FORCE:-0}" != "1" ]; then
    echo "── PG $TAG: $DEST exists — skipping (CONTRAST_FORCE=1 to redo)"
    continue
  fi
  [ -e "$DEST" ] && rm -rf "$DEST"
  echo "── PG  DCI_GATE=v3 force=[$FORCE] → $DEST  ($(date -u +%H:%M:%SZ))"
  DCI_GATE=v3 DCI_FORCE="$FORCE" PROBE_TAG="_contrast_${TAG}" \
    python3 postgres/pg_adaptation.py --workload "$WL" --sf 1.0 \
            --blocks "$BLOCKS"
  mv postgres/out "$DEST"
  echo "── PG $TAG done → $DEST"
done
echo "═══ done — now: python3 contrast_analyze.py . ═══"
