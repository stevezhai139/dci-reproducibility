#!/usr/bin/env bash
# Paper 3C repo cleanup. The sandbox could not delete these sibling (Paper 3D)
# files from the mounted volume, so run this ONCE on your Mac from repro_3c/.
# After this the repository is DCI-only. Safe: none of these are imported by
# the dci_gated harness (verified).
set -euo pipefail
cd "$(dirname "$0")"

# 3D-only directories (economics analysis, MySQL, SDSS, phase2)
rm -rf end_to_end/analysis end_to_end/mysql end_to_end/sdss phase2 sdss data/sdss

# 3D analysis / benchmark scripts (not imported by pg/mongo_adaptation)
rm -f end_to_end/bench_gate_eval.py \
      end_to_end/bench_hsm_profile.py \
      end_to_end/dci_stratified_analysis.py \
      end_to_end/divergence_analysis.py \
      end_to_end/paired_comparison_analysis.py \
      end_to_end/recompute_invocation_cost.py \
      end_to_end/regime_coverage.py \
      end_to_end/theta_calibration.py \
      end_to_end/test_dci_gate.py

# optional: drop the compatibility shim once nothing references it
# rm -f end_to_end/raw_dci_gate.py

echo "repro_3c is now DCI-only."
