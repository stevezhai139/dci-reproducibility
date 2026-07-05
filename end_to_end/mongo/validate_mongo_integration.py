#!/usr/bin/env python3
"""
Offline validator for the Paper 3D MongoDB seasonality + ESR integration
(Addendum 21 / TOMORROW step 2). NO mongod required — proves the wiring:
   season_schedule  →  per-window phase  →  phase query mix  →  pipelines
                    →  esr_recommender.recommend  →  per-season index set.

It checks the property that makes Mongo the gating-favorable cell:
 (1) consecutive seasons are DIFFERENT modalities (high-magnitude drift), and
 (2) each modality's recommended index set is DIFFERENT (a prior season's
     index is useless next season → forces re-tune → no durable backbone).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
MONGO = os.path.join(HERE)
WORKLOAD = os.path.abspath(os.path.join(HERE, "..", "..", "cross_engine", "mongo", "workload"))
for p in (MONGO, WORKLOAD):
    sys.path.insert(0, p)

import season_schedule as season
import esr_recommender as esr
import templates as T

def phase_qids(phase):
    return list(T.ALL_PHASES[phase]["mix"].keys())

def phase_index_set(phase):
    pls = [T.ALL_TEMPLATES[q].pipeline for q in phase_qids(phase)]
    specs = esr.recommend(pls)
    return [esr.spec_name(k) for k in specs], specs

print("="*84)
print("MongoDB integration — OFFLINE wiring validation (no mongod)")
print("="*84)

# 1) per-modality ESR index set
print("\n[1] ESR recommended index set per modality (from the window query mix):")
idx_by_phase = {}
for ph in season.DEFAULT_PHASES:
    names, specs = phase_index_set(ph)
    idx_by_phase[ph] = set(names)
    print(f"\n  ── {ph} ──  qids={phase_qids(ph)}")
    for k in specs:
        print(f"     {esr.spec_name(k):46} {k}")

# 2) modalities need DIFFERENT indexes (no durable backbone)
print("\n[2] Cross-modality index overlap (Jaccard) — low overlap ⇒ no durable backbone:")
phs = season.DEFAULT_PHASES
for i in range(len(phs)):
    for j in range(i+1, len(phs)):
        a, b = idx_by_phase[phs[i]], idx_by_phase[phs[j]]
        jac = len(a & b)/len(a | b) if (a|b) else 1.0
        print(f"     {phs[i]:6} vs {phs[j]:6}:  shared={sorted(a & b) or '∅'}  Jaccard={jac:.2f}")

# 3) irregular schedule structure (the headline regime)
print("\n[3] Irregular long-season schedule (headline, n=120, seed=9000):")
pw, seasons, drift = season.make_schedule(120, seed=9000, mode="irregular")
print(f"     seasons={len(seasons)}  drift_windows={sorted(drift)}")
boundary_diff = all(seasons[k].phase != seasons[k-1].phase for k in range(1, len(seasons)))
for sgmt in seasons:
    print(f"     win[{sgmt.start:3d}:{sgmt.end:3d}] len={sgmt.length:2d}  {sgmt.phase}")
print(f"     every boundary is a real modality switch (high-magnitude): {boundary_diff}")

# 4) at each drift boundary, prior season's index set does NOT serve the next
print("\n[4] At each season boundary, does the prior index set serve the next season?")
stale_ok = True
for k in range(1, len(seasons)):
    prev_idx = idx_by_phase[seasons[k-1].phase]
    next_idx = idx_by_phase[seasons[k].phase]
    served = next_idx <= prev_idx          # next fully covered by prev?
    if served: stale_ok = False
    print(f"     drift@win{seasons[k].start:3d}: {seasons[k-1].phase}→{seasons[k].phase}  "
          f"prior-serves-next={served}  (need re-tune: {not served})")
print(f"\n  ⇒ every boundary forces a re-tune (gating-favorable regime): {stale_ok}")

# 5) regular mode reproduces the sensitivity-axis schedule
pw_r, seasons_r, drift_r = season.make_schedule(24, seed=9000, mode="regular")
print(f"\n[5] Regular (sensitivity axis), n=24: seasons={len(seasons_r)} "
      f"phases={[s.phase for s in seasons_r]} drift={sorted(drift_r)}")

print("\n" + "="*84)
ok = boundary_diff and stale_ok
print("WIRING VALIDATION:", "PASS ✓" if ok else "CHECK ✗",
      "— schedule + ESR recommender produce a no-durable-backbone, re-tune-every-season regime"
      if ok else "")
