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


class View:
    """Raw window material (possibly a mixture) for pair_features_v."""
    def __init__(self, fp_list, tables, arr_rel):
        self.fp_list, self.tables_, self.arr = fp_list, tables, np.asarray(arr_rel, float)

    def qps(self):
        r = np.sort(self.arr)
        return arrivals_to_qps_series(r, max(float(r[-1]), 1.0))


def view_of(st: Store, w: int, ns=None):
    fp = st.fp[w].tolist()
    tb = st.tables(w)
    if ns is not None:
        fp = [(ns, x) for x in fp]; tb = {(ns, x) for x in tb}
    return View(fp, tb, st.arr[w])


def mixed_view(sa: Store, wa: int, sb: Store, wb: int, lam: float, rng, cross: bool):
    """(1-lam)*A + lam*B mixture window: per-query sampling of fingerprints,
    per-query table lists, merged relative arrivals. Deterministic via rng."""
    W = sa.win
    nb = int(round(lam * W)); na = W - nb
    ia = np.sort(rng.choice(W, size=na, replace=False))
    ib = np.sort(rng.choice(W, size=nb, replace=False))
    nsa, nsb = (0, 1) if cross else (None, None)
    fa = sa.fp[wa][ia].tolist(); fb = sb.fp[wb][ib].tolist()
    if cross:
        fa = [(0, x) for x in fa]; fb = [(1, x) for x in fb]
    # per-query table union over sampled rows
    def tabs(st, w, rows, ns):
        out = set()
        base = w * st.win
        for r_ in rows:
            a, b = int(st.tbl_off[base + r_]), int(st.tbl_off[base + r_ + 1])
            for t in st.tbl_flat[a:b].tolist():
                out.add((ns, t) if ns is not None else t)
        return out
    tb = tabs(sa, wa, ia, nsa) | tabs(sb, wb, ib, nsb)
    arr = np.concatenate([sa.arr[wa][ia], sb.arr[wb][ib]])
    return View(fa + fb, tb, arr)


def pair_features_v(va: "View", vb: "View"):
    ca, cb = Counter(va.fp_list), Counter(vb.fp_list)
    vocab = list(ca.keys() | cb.keys())
    x = np.array([ca.get(f, 0) for f in vocab], float); x /= x.sum()
    y = np.array([cb.get(f, 0) for f in vocab], float); y /= y.sum()
    s_r = float(sr_v2(x, y)); s_t = float(st_v2(x, y))
    u = va.tables_ | vb.tables_
    s_a = float(len(va.tables_ & vb.tables_) / len(u)) if u else 1.0
    s_p = float(sp_v2(va.qps(), vb.qps(), set(va.fp_list), set(vb.fp_list)))
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
    b = sub.add_parser("bench")
    b.add_argument("--pairs", default="pairs_out")
    b.add_argument("--repro", default="..")
    L = sub.add_parser("lam")
    L.add_argument("--data", default="data")
    L.add_argument("--pairs", default="pairs_out")
    L.add_argument("--out", default="lam_out")
    L.add_argument("--lambdas", default="0.05,0.1,0.2,0.4,1.0")
    L.add_argument("--aligned-sample", type=int, default=100)
    L.add_argument("--seg", type=int, default=12)
    L.add_argument("--cal", type=int, default=64)
    F = sub.add_parser("lamfit")
    F.add_argument("--pairs", default="lam_out")
    a = ap.parse_args()
    if a.cmd == "bench":
        return bench(a)
    if a.cmd == "lam":
        return lam(a)
    if a.cmd == "lamfit":
        return lamfit(a)

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


def bench(a) -> int:
    """Phase 2: S6 policy suite over the cached splice trajectories, stratified
    by arm x router geometry (aligned R>=0.35 vs off-axis R<0.35 at onset);
    the control arm supplies the real-data false-alarm floor (spec T-C)."""
    import csv as _csv
    import importlib.util, json, time
    t0 = time.time()
    root = Path(a.repro).resolve()

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m
        spec.loader.exec_module(m); return m

    s6 = load(root / "s6_baseline_bench.py", "s6_bench_x")
    gate_mod = load(root / "end_to_end" / "dci_gate.py", "dci_gate_x")
    v2_mod = load(root / "dci_gate_v2.py", "dci_gate_v2_x")
    v3_mod = load(root / "dci_gate_v3.py", "dci_gate_v3_x")
    ov = json.load(open(root / "geometry_E0" / "out" / "20260705T135856Z" / "cost_benefit_run.json"))
    pax = {k: v * 1e3 for k, v in ov["overhead"]["per_dimension_overhead_s"].items()}

    pdir = Path(a.pairs)
    zf = np.load(pdir / "traj_cache.npz")
    CAL, FV, onset = zf["cal"], zf["fv"], int(zf["onset_idx"])
    rows = list(_csv.DictReader(open(pdir / "candidates.csv")))
    assert len(rows) == CAL.shape[0]

    POL = ["always_multiD", "union4_bonf", "dci_v3", "dci_v3_a8",
           "best_axis_oracle", "every_k2", "adwin", "mcusum", "sketch1"]

    def stratum(r):
        if r.get("stratum"):
            return r["stratum"]
        if r["arm"] == "control":
            return "control"
        R = float(r["R_at_onset"]) if r["R_at_onset"] not in ("", "nan") else 1.0
        return f'{r["arm"]}/{"offaxis" if R < 0.35 else "aligned"}'

    agg = {}
    out_rows = []
    for i, r in enumerate(rows):
        cal, fv = CAL[i].astype(float), FV[i].astype(float)
        ctx = s6.Ctx(cal, gate_mod, v2_mod, v3_mod)
        n = fv.shape[0]
        truth = np.zeros(n, int)
        if r["arm"] != "control":
            truth[onset] = 1
        st = stratum(r)
        for pol in POL:
            out = s6.replay(pol, ctx, fv, pax, i)
            if isinstance(out[0], tuple) and out[4] == "ORACLE_STUB":
                # oracle: best per-axis fires on THIS trajectory (recall, then -FA)
                best = None
                for j in range(5):
                    stat = ((fv[:, j] - ctx.mu0[j]) / ctx.sigma0[j]) ** 2
                    f_ = (stat > ctx.thr[1]).astype(int)
                    key = (f_[onset] if truth.any() else 0, -int(f_[truth == 0].sum()))
                    if best is None or key > best[0]:
                        best = (key, f_)
                fires = best[1]
            else:
                fires = out[0]
            hit = int(fires[onset]) if truth.any() else 0
            fa = int(fires[truth == 0].sum())
            k = (st, pol)
            d = agg.setdefault(k, [0, 0, 0, 0])   # n, hits, onsets, fa
            d[0] += 1; d[1] += hit; d[2] += int(truth.any()); d[3] += fa
            out_rows.append((r["id"], st, pol, hit, fa))
        if i % 200 == 0 and i:
            print(f"  ...{i}/{len(rows)} [{time.time()-t0:.0f}s]", flush=True)

    with open(pdir / "bench_results.csv", "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["cand_id", "stratum", "policy", "hit", "fa"])
        w.writerows(out_rows)

    hdr = "%-18s%-17s%6s%8s%9s" % ("stratum", "policy", "n", "recall", "FA/traj")
    print("\n" + hdr)
    summ = {}
    for (st, pol) in sorted(agg):
        n_, h, o, fa = agg[(st, pol)]
        rec = h / o if o else float("nan")
        print(f"{st:18s}{pol:17s}{n_:6d}"
              + (f"{rec:8.3f}" if o else f"{'—':>8s}")
              + f"{fa / n_:9.3f}")
        summ[f"{st}/{pol}"] = {"n": n_, "recall": (round(rec, 4) if o else None),
                               "fa_per_traj": round(fa / n_, 4)}
    json.dump(summ, open(pdir / "bench_summary.json", "w"), indent=1)
    print(f"[out] {pdir}/bench_results.csv + bench_summary.json  [{time.time()-t0:.0f}s]")
    return 0


# Bonferroni cheap-axis and full-test thresholds at m=64, alpha=0.05.
# See REDSET_LAMBDA_PREDICTION.md for the derivation these feed:
#   lam50_union / lam50_full  ~=  sqrt(T1/T5) / sqrt(R~)  =  0.726 / sqrt(R~)
T1_CHEAP, T5_FULL = 6.77, 12.85


def lamfit(a) -> int:
    """Official lambda* extraction: per (parent stratum, policy) recall curves
    from lam_out/bench_results.csv, lam50 by first-crossing linear
    interpolation, observed union/full ratio vs predicted 0.726/sqrt(R~)
    with R~ = median R_at_onset of the stratum's lam=1 (saturated) rows."""
    import csv as _csv, json
    import pandas as pd
    cand = pd.read_csv(Path(a.pairs) / "candidates.csv")
    res = pd.read_csv(Path(a.pairs) / "bench_results.csv")
    parts = res["stratum"].str.rsplit("/l", n=1)
    res["parent"] = parts.str[0]
    res["lam"] = parts.str[1].astype(float)
    cp = cand["stratum"].str.rsplit("/l", n=1)
    cand["parent"] = cp.str[0]
    rt = (cand[cand["lam"] == 1.0].groupby("parent")["R_at_onset"]
          .median().to_dict())

    def lam50(grp):
        g = grp.groupby("lam")["hit"].mean().sort_index()
        ls, rs = g.index.to_numpy(float), g.to_numpy(float)
        if rs[0] >= 0.5:
            return -ls[0], rs  # saturated left: lam50 < grid floor
        for i in range(1, len(rs)):
            if rs[i] >= 0.5 > rs[i - 1]:
                x = ls[i-1] + (ls[i]-ls[i-1]) * (0.5-rs[i-1]) / (rs[i]-rs[i-1])
                return x, rs
        return float("inf"), rs  # never crosses

    out = {}
    for parent, gp in res.groupby("parent"):
        row = {"R_med": round(float(rt.get(parent, float('nan'))), 4)}
        for pol in ("always_multiD", "union4_bonf", "best_axis_oracle",
                    "dci_v3", "dci_v3_a8"):
            x, curve = lam50(gp[gp["policy"] == pol])
            row[pol] = round(x, 4) if np.isfinite(x) else None
            row[pol + "_curve"] = [round(float(v), 3) for v in curve]
        lf, lu = row["always_multiD"], row["union4_bonf"]
        if lf and lu and lf > 0 and lu > 0:
            row["ratio_obs"] = round(lu / lf, 3)
        else:
            row["ratio_obs"] = None
        Rm = row["R_med"]
        row["ratio_pred"] = round(float(np.sqrt(T1_CHEAP / T5_FULL) / np.sqrt(Rm)), 3) if Rm and Rm > 0 else None
        out[parent] = row

    hdr = f"{'parent stratum':<16}{'R~med':>7}{'l50 full':>10}{'l50 union':>10}{'l50 orac':>9}{'l50 dci':>9}{'l50 a8':>8}{'obs':>7}{'pred':>7}"
    print(hdr); print("-" * len(hdr))
    for k, r in out.items():
        f6 = lambda v: ("  <%.2f" % -v if (v is not None and v < 0) else ("  never" if v is None else "%7.3f" % v))
        print(f"{k:<16}{r['R_med']:>7.3f}{f6(r['always_multiD']):>10}{f6(r['union4_bonf']):>10}"
              f"{f6(r['best_axis_oracle']):>9}{f6(r['dci_v3']):>9}{f6(r['dci_v3_a8']):>8}"
              f"{r['ratio_obs'] if r['ratio_obs'] else '--':>7}{r['ratio_pred'] if r['ratio_pred'] else '--':>7}")
    with open(Path(a.pairs) / "lamfit.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[out] {a.pairs}/lamfit.json")
    return 0


def lam(a) -> int:
    """Magnitude-controlled splices: post-onset windows are (1-lam)*A + lam*B
    mixtures of REAL streams (partial-migration semantics). Stratified
    subsample of the phase-1 population; parent stratum carried explicitly so
    the bench aggregates by (parent geometry x lambda). Writes lam_out/ with
    the same filenames bench expects."""
    import csv as _csv, time
    t0 = time.time()
    lams = [float(x) for x in a.lambdas.split(",")]
    rows = list(_csv.DictReader(open(Path(a.pairs) / "candidates.csv")))
    rng0 = np.random.default_rng(RNG_SEED + 1)

    def parent(r):
        R = float(r["R_at_onset"]) if r["R_at_onset"] not in ("", "nan") else 1.0
        return f'{r["arm"]}/{"offaxis" if R < 0.35 else "aligned"}'

    sel = []
    for st_name in ("cross/offaxis", "within/offaxis"):
        sel += [r for r in rows if r["arm"] != "control" and parent(r) == st_name]
    aligned = [r for r in rows if r["arm"] == "cross" and parent(r) == "cross/aligned"]
    idx = rng0.choice(len(aligned), size=min(a.aligned_sample, len(aligned)), replace=False)
    sel += [aligned[i] for i in idx]
    wal = [r for r in rows if r["arm"] == "within" and parent(r) == "within/aligned"]
    idx = rng0.choice(len(wal), size=min(a.aligned_sample // 2, len(wal)), replace=False)
    sel += [wal[i] for i in idx]
    print(f"[lam] {len(sel)} parent candidates x {len(lams)} lambdas")

    ids = sorted({r["cluster_a"] for r in sel} | {r["cluster_b"] for r in sel})
    stores = {c: Store(Path(a.data) / f"store_{c}.npz") for c in ids}
    memo = {}
    out_rows, cache_cal, cache_fv = [], [], []
    onset_idx = a.seg - 1
    k = 0
    for r in sel:
        cx, cy = r["cluster_a"], r["cluster_b"]
        s1, s2 = int(r["start_a"]), int(r["start_b"])
        cross = (cx != cy)
        sa, sb = stores[cx], stores[cy]
        calF, segA = a_side(sa, s1, a.cal, a.seg, memo, (cx, s1))
        lastA = view_of(sa, s1 + a.seg - 1, 0 if cross else None)
        for lm in lams:
            feats = list(segA)
            prev = lastA
            for i in range(a.seg):
                rng = np.random.default_rng([RNG_SEED, int(r["id"]), int(lm * 1000), i])
                cur = mixed_view(sa, s1 + a.seg + i, sb, s2 + i, lm, rng, cross)
                feats.append(pair_features_v(prev, cur))
                prev = cur
            fv = np.array(feats)
            R_at, dci_at, minR, esc = geometry(calF, fv, onset_idx)
            out_rows.append({"id": k, "arm": r["arm"], "cluster_a": cx, "start_a": s1,
                             "cluster_b": cy, "start_b": s2, "lam": lm,
                             "stratum": f"{parent(r)}/l{lm:g}",
                             "R_at_onset": round(R_at, 4), "DCI_at_onset": round(dci_at, 4),
                             "minR_onset_p2": round(minR, 4), "esc_windows": esc})
            cache_cal.append(calF.astype(np.float32)); cache_fv.append(fv.astype(np.float32))
            k += 1
        if len(out_rows) % 100 < len(lams):
            print(f"  ...{len(out_rows)} trajectories [{time.time()-t0:.0f}s]", flush=True)

    outd = Path(a.out); outd.mkdir(exist_ok=True)
    with open(outd / "candidates.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    np.savez_compressed(outd / "traj_cache.npz",
                        cal=np.array(cache_cal), fv=np.array(cache_fv),
                        onset_idx=onset_idx)
    print(f"[out] {outd}/candidates.csv + traj_cache.npz ({k} trajectories) "
          f"[{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
