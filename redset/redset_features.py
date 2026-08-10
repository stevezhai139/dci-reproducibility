#!/usr/bin/env python3
"""redset_features.py -- Redset -> per-window HSM-style features (Paper 3C splice study).

Two commands:

  build-store  <cluster.parquet> --out store_<id>.npz
      Sort by arrival, cut consecutive fixed-count windows (W=1000), intern
      fingerprints/tables, store compact per-window raw material
      (fingerprint ids, relative arrivals, table ids, t0).

  features     <store.npz> --out features_<id>.parquet [--steady 64]
      Compute the five-axis feature vector of every window against the
      reference window W0 (window 0), plus steady-prefix statistics and
      range/stability validation. This is the paper's FOURTH kernel-free
      representation (Sec 6.9 precedent).

DECLARED PROTOCOL DECISIONS (must appear in the paper if the study lands):
  d1  aborted queries dropped; queries with no feature_fingerprint dropped
      (counts reported by build-store).
  d2  feature_fingerprint is a hash proxy; per the Redset README it
      OVERESTIMATES repetition.
  d3  fingerprint vocabulary per cluster = top-VOCAB fingerprints + one
      residual bucket (coverage reported); S_R (Spearman) and S_T (cosine)
      computed on this vector.
  d4  S_V is CONSTANT by construction under fixed-count windows (n_a = n_b);
      volume changes surface through S_P's arrival-density series instead.
  d5  S_A is table-only Jaccard (Redset has no column data): J(read_table_ids).
  d6  S_P = kernel sp_v2 on arrivals_to_qps_series of each window's relative
      arrival times (1-second bins), fingerprint sets as the short-series
      fallback -- the kernel's own S_P pipeline, fed arrival density.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernel"))
from hsm_v2_kernel import sr_v2, st_v2, sp_v2, arrivals_to_qps_series  # noqa: E402

W_DEFAULT = 1000
VOCAB = 2000


# ─────────────────────────── build-store ───────────────────────────

def build_store(parquet: str, out: str, win: int) -> int:
    t0 = time.time()
    cols = ["arrival_timestamp", "feature_fingerprint", "read_table_ids", "was_aborted"]
    df = pd.read_parquet(parquet, columns=cols)
    n_total = len(df)
    df = df[df["was_aborted"] != True]  # noqa: E712
    n_ab = n_total - len(df)
    n_nofp = int(df["feature_fingerprint"].isna().sum())
    df = df.dropna(subset=["feature_fingerprint"])
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    df = df.sort_values("arrival_timestamp", kind="mergesort").reset_index(drop=True)
    kept = len(df)
    print(f"[store] {parquet}: rows {n_total:,} -> kept {kept:,} "
          f"(dropped aborted {n_ab:,}, no-fingerprint {n_nofp:,})  [{time.time()-t0:.0f}s]")

    # intern fingerprints
    fp_cat = df["feature_fingerprint"].astype("category")
    fp_ids_all = fp_cat.cat.codes.to_numpy(np.int32)
    fp_vocab = list(map(str, fp_cat.cat.categories))
    # intern tables (parse comma lists once)
    tbl_map: dict[str, int] = {}
    tbl_concat: list[np.ndarray] = []
    tbl_counts = np.zeros(kept, np.int32)
    raw_tbl = df["read_table_ids"].astype("string").fillna("")
    for i, x in enumerate(raw_tbl):
        if x:
            ids = []
            for t in x.split(","):
                t = t.strip()
                if t:
                    ids.append(tbl_map.setdefault(t, len(tbl_map)))
            tbl_counts[i] = len(ids)
            if ids:
                tbl_concat.append(np.asarray(ids, np.int32))
    tbl_flat = np.concatenate(tbl_concat) if tbl_concat else np.zeros(0, np.int32)
    tbl_off = np.zeros(kept + 1, np.int64); np.cumsum(tbl_counts, out=tbl_off[1:])

    ts_ns = df["arrival_timestamp"].astype("int64").to_numpy()
    nw = kept // win
    print(f"[store] windows: {nw} x {win} (tail {kept - nw*win} dropped); "
          f"fp vocab {len(fp_vocab):,}; table vocab {len(tbl_map):,}")

    fp_ids = fp_ids_all[:nw*win].reshape(nw, win)
    t_start = ts_ns[np.arange(nw) * win]
    arr_rel = ((ts_ns[:nw*win] - np.repeat(t_start, win)) / 1e9).astype(np.float32).reshape(nw, win)
    np.savez_compressed(out, fp_ids=fp_ids, arr_rel=arr_rel, t_start=t_start,
                        tbl_flat=tbl_flat, tbl_off=tbl_off[:nw*win+1], win=win,
                        drops=np.array([n_total, n_ab, n_nofp, kept]))
    Path(out.replace(".npz", "_vocab.json")).write_text(json.dumps(
        {"fingerprints": fp_vocab, "tables": list(tbl_map.keys())}))
    print(f"[store] wrote {out} [{time.time()-t0:.0f}s total]")
    return 0


# ─────────────────────────── features ──────────────────────────────

def window_tables(z, w):
    a, b = int(z["tbl_off"][w * z["win"]]), int(z["tbl_off"][(w + 1) * z["win"]])
    return set(z["tbl_flat"][a:b].tolist())


def freq_on(ids_row, vocab_ids, resid_id):
    c = Counter(ids_row.tolist())
    v = np.array([c.get(f, 0) for f in vocab_ids], float)
    resid = len(ids_row) - v.sum()
    v = np.append(v, resid)
    return v / v.sum()


def features(store: str, out: str, steady: int) -> int:
    t0 = time.time()
    zf = np.load(store)
    z = {k: zf[k] for k in zf.files}; z["win"] = int(z["win"])
    nw = z["fp_ids"].shape[0]
    # cluster vocab: top-VOCAB fingerprints across all windows
    cnt = Counter(z["fp_ids"].ravel().tolist())
    vocab_ids = [f for f, _ in cnt.most_common(VOCAB)]
    cover = sum(cnt[f] for f in vocab_ids) / sum(cnt.values()) * 100
    print(f"[feat] {store}: {nw} windows; vocab top-{VOCAB} covers {cover:.1f}% (residual bucket holds the rest)")

    ref_fp = z["fp_ids"][0]
    ref_freq = freq_on(ref_fp, vocab_ids, None)
    ref_tbl = window_tables(z, 0)
    ref_qps = arrivals_to_qps_series(z["arr_rel"][0].astype(float))
    ref_set = set(ref_fp.tolist())

    rows = []
    for wdx in range(nw):
        fp = z["fp_ids"][wdx]
        fq = freq_on(fp, vocab_ids, None)
        s_r = float(sr_v2(ref_freq, fq))
        s_t = float(st_v2(ref_freq, fq))
        s_v = 1.0  # d4: constant by construction (fixed-count windows)
        tb = window_tables(z, wdx)
        u = ref_tbl | tb
        s_a = float(len(ref_tbl & tb) / len(u)) if u else 1.0
        qps = arrivals_to_qps_series(z["arr_rel"][wdx].astype(float))
        s_p = float(sp_v2(ref_qps, qps, ref_set, set(fp.tolist())))
        rows.append((wdx, int(z["t_start"][wdx]), s_r, s_v, s_t, s_a, s_p))
        if wdx % 500 == 0 and wdx:
            print(f"  ...{wdx}/{nw} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows, columns=["window", "t_start_ns", "S_R", "S_V", "S_T", "S_A", "S_P"])
    df.to_parquet(out)

    # validation block
    F = df[["S_R", "S_T", "S_A", "S_P"]]
    print(f"[feat] ranges: min={F.min().round(3).to_dict()} max={F.max().round(3).to_dict()}")
    sp_ = F.iloc[:steady]
    print(f"[feat] steady prefix (first {steady}): median={sp_.median().round(3).to_dict()}")
    print(f"[feat]                                sigma ={sp_.std().round(4).to_dict()}")
    tail = F.iloc[steady:]
    print(f"[feat] post-prefix medians: {tail.median().round(3).to_dict()}")
    print(f"[feat] wrote {out}  [{time.time()-t0:.0f}s total]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-store"); b.add_argument("parquet")
    b.add_argument("--out", required=True); b.add_argument("--win", type=int, default=W_DEFAULT)
    f = sub.add_parser("features"); f.add_argument("store")
    f.add_argument("--out", required=True); f.add_argument("--steady", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "build-store":
        return build_store(a.parquet, a.out, a.win)
    return features(a.store, a.out, a.steady)


if __name__ == "__main__":
    sys.exit(main())
