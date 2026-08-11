#!/usr/bin/env python3
"""redset_splice.py -- splice-pair population + geometry (Paper 3C Redset study, spec Sec 3-4).

`pairs` command: enumerate candidate splices over the cluster pool, compute
each candidate's 24-window feature trajectory (adjacent-pair convention,
PAIR-LOCAL exact vocab -- no truncation), run the deployed DCIGateV3 router
over it, and record the geometry the router actually sees:

    R (alignment share) and DCI at/after the onset

Three arms (spec Sec 3):
    cross    A != B                 heterogeneous change (candidate off-axis source)
    within   A == B, distant        natural drift, weaker
    control  A == B, contiguous     no onset -- false-alarm floor on real data

Trajectory layout per candidate (matching the live Part-2 block shape):
    calibration = CAL adjacent-pair features immediately before segment A
    segment A   = SEG windows, then the BOUNDARY pair (last-A, first-B),
    segment B   = SEG windows  ->  2*SEG features, onset at index SEG (0-based)

Methodological guard (spec Sec 4): this stage records geometry for the WHOLE
candidate population; stratified sampling for the bench happens downstream and
the population distribution is always reported alongside.

Usage:
  python3 redset_splice.py pairs --data data --clusters 96,100,31,77,178,33,3,11,56,16,53,14 \
      --out pairs_out [--seg 12 --cal 64 --stride 1000 --seed 20260811]
Writes: pairs_out/candidates.csv, pairs_out/traj_cache.npz
"""
from __future__ import annotations
import argparse, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernel"))
sys.path.insert(0, str(HERE.parent))
from hsm_v2_kernel import sr_v2, st_v2, sp_v2, arrivals_to_qps_series  # noqa: E402
from dci_gate_v3 import DCIGateV3  # noqa: E402

RNG_SEED = 20260811


class Store:
    def __init__(self, path: Path):
        zf = np.load(path)
        self.fp = zf["fp_ids"]; self.arr = zf["arr_rel"]
        self.tbl_flat = zf["tbl_flat"]; self.tbl_off = zf["tbl_off"]
        self.win = int(zf["win"]); self.nw = self.fp.shape[0]

    def tables(self, w):
        a, b = int(self.tbl_off[w * self.win]), int(self.tbl_off[(w + 1) * self.win])
        return set(self.tbl_flat[a:b].tolist())

    def qps(self, w):
        r = self.arr[w].astype(float)
        return arrivals_to_qps_series(r, max(float(r[-1]), 1.0))


def pair_features(sa: Store, wa: int, sb: Store, wb: int, cross: bool):
    """5-axis adjacent-pair similarity, pair-local exact vocab.
    cross=True -> table/fingerprint id spaces differ; ids are namespaced."""
    fa, fb = sa.fp[wa].tolist(), sb.fp[wb].tolist()
    if cross:  # disjoint id spaces: namespace to avoid accidental collisions
        fa = [(0, x) for x in fa]; fb = [(1, x) for x in fb]
        ta = {(0, x) for x in sa.tables(wa)}; tb = {(1, x) for x in sb.tables(wb)}
    else:
        ta, tb = sa.tables(wa), sb.tables(wb)
    ca, cb = Counter(fa), Counter(fb)
    vocab = list(ca.keys() | cb.keys())
    va = np.array([ca.get(f, 0) for f in vocab], float); va /= va.sum()
    vb = np.array([cb.get(f, 0) for f in vocab], float); vb /= vb.sum()
    s_r = float(sr_v2(va, vb)); s_t = float(st_v2(va, vb))
    u = ta | tb
    s_a = float(len(ta & tb) / len(u)) if u else 1.0
    s_p = float(sp_v2(sa.qps(wa), sb.qps(wb), set(fa), set(fb)))
    return (s_r, 1.0, s_t, s_a, s_p)


def a_side(store: Store, a0: int, cal: int, seg: int, memo: dict, key):
    """calibration features + segment-A internal features (cached per (cluster,a0))."""
    if key in memo:
        return memo[key]
    calF = [pair_features(store, w - 1, store, w, False) for w in range(a0 - cal + 1, a0 + 1)]
    segA = [pair_features(store, w - 1, store, w, False) for w in range(a0 + 1, a0 + seg)]
    memo[key] = (np.array(calF), np.array(segA))
    return memo[key]


def build_traj(sa: Store, a0: int, sb: Store, b0: int, cal: int, seg: int,
               cross: bool, memo: dict, akey):
    calF, segA = a_side(sa, a0, cal, seg, memo, akey)
    boundary = pair_features(sa, a0 + seg - 1, sb, b0, cross)
    segB = [pair_features(sb, w - 1, sb, w, False) for w in range(b0 + 1, b0 + seg)]
    fv = np.vstack([segA, np.array([boundary]), np.array(segB)])  # (2*seg-1, 5)
    return calF, fv, boundary


def geometry(calF, fv, onset_idx):
    g = DCIGateV3().fit(calF)
    R_at = dci_at = np.nan; minR_post = np.inf; esc = 0
    for t in range(fv.shape[0]):
        g.decide(fv[t])
        if t == onset_idx:
            R_at, dci_at = g.last["R4s"], g.last["dci4"]
        if onset_idx <= t <= onset_idx + 2:
            minR_post = min(minR_post, g.last["R4s"])
        esc += int(g.last["regime"] == "full")
    return R_at, dci_at, (minR_post if minR_post != np.inf else np.nan), esc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--data", default="data")
    p.add_argument("--clusters", required=True)
    p.add_argument("--out", default="pairs_out")
    p.add_argument("--seg", type=int, default=12)
    p.add_argument("--cal", type=int, default=64)
    p.add_argument("--stride", type=int, default=1000)
    p.add_argument("--gap", type=int, default=5000)
    p.add_argument("--max-cross", type=int, default=1500)
    p.add_argument("--max-within", type=int, default=600)
    p.add_argument("--max-control", type=int, default=400)
    a = ap.parse_args()

    t0 = time.time()
    ids = [c.strip() for c in a.clusters.split(",")]
    rng = np.random.default_rng(RNG_SEED)
    data = Path(a.data); outd = Path(a.out); outd.mkdir(exist_ok=True)
    stores = {c: Store(data / f"store_{c}.npz") for c in ids}
    for c in ids:
        print(f"[pairs] cluster {c}: {stores[c].nw} windows")
    starts = {c: list(range(a.cal + 1, stores[c].nw - 2 * a.seg - 2, a.stride)) for c in ids}

    cands = []
    # cross arm
    combos = [(x, y) for x in ids for y in ids if x != y]
    per = max(1, a.max_cross // len(combos))
    for (x, y) in combos:
        for _ in range(per):
            cands.append(("cross", x, int(rng.choice(starts[x])), y, int(rng.choice(starts[y]))))
    # within-distant arm
    per = max(1, a.max_within // len(ids))
    for x in ids:
        ok = [(s1, s2) for s1 in starts[x] for s2 in starts[x] if s2 >= s1 + a.gap]
        if not ok: continue
        take = rng.choice(len(ok), size=min(per, len(ok)), replace=False)
        for i in take:
            cands.append(("within", x, ok[i][0], x, ok[i][1]))
    # control arm (contiguous, no onset)
    per = max(1, a.max_control // len(ids))
    for x in ids:
        take = rng.choice(len(starts[x]), size=min(per, len(starts[x])), replace=False)
        for i in take:
            s1 = starts[x][i]
            cands.append(("control", x, s1, x, s1 + a.seg))
    print(f"[pairs] candidates: {Counter(c[0] for c in cands)}  [{time.time()-t0:.0f}s]")

    onset_idx = a.seg - 1  # boundary feature position in fv (0-based)
    memo, rows = {}, []
    cache_cal, cache_fv = [], []
    for i, (arm, cx, s1, cy, s2) in enumerate(cands):
        cross = (cx != cy)
        calF, fv, bnd = build_traj(stores[cx], s1, stores[cy], s2, a.cal, a.seg,
                                   cross, memo, (cx, s1))
        R_at, dci_at, minR, esc = geometry(calF, fv, onset_idx)
        rows.append((i, arm, cx, s1, cy, s2, round(R_at, 4), round(dci_at, 4),
                     round(minR, 4), esc,
                     *[round(b, 4) for b in bnd]))
        cache_cal.append(calF.astype(np.float32)); cache_fv.append(fv.astype(np.float32))
        if i % 100 == 0 and i:
            print(f"  ...{i}/{len(cands)} [{time.time()-t0:.0f}s]", flush=True)

    import csv as _csv
    with open(outd / "candidates.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["id", "arm", "cluster_a", "start_a", "cluster_b", "start_b",
                    "R_at_onset", "DCI_at_onset", "minR_onset_p2", "esc_windows",
                    "b_SR", "b_SV", "b_ST", "b_SA", "b_SP"])
        w.writerows(rows)
    np.savez_compressed(outd / "traj_cache.npz",
                        cal=np.array(cache_cal), fv=np.array(cache_fv),
                        onset_idx=onset_idx)
    # population summary — THE number
    import numpy as _np
    arr = [(r[1], r[6]) for r in rows if r[1] != "control" and r[6] == r[6]]
    for arm in ("cross", "within"):
        Rv = _np.array([r for (t, r) in arr if t == arm])
        if len(Rv):
            off = float((_np.array(Rv) < 0.35).mean() * 100)
            print(f"[POPULATION] {arm:7s}: n={len(Rv)}  R quartiles "
                  f"{_np.quantile(Rv,.25):.2f}/{_np.quantile(Rv,.5):.2f}/{_np.quantile(Rv,.75):.2f}"
                  f"  OFF-AXIS (R<0.35): {off:.1f}%")
    print(f"[out] {outd}/candidates.csv + traj_cache.npz  [{time.time()-t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
