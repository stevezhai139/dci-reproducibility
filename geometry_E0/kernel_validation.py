#!/usr/bin/env python3
"""
kernel_validation.py
====================

Paper 3C -- validate DCI and the axis structure against the REAL
vendored 5-D HSM kernel, on three workloads.

WHY
    Everything so far runs on the E0-anchored synthetic harness, whose
    f-vectors are constructed (S_V injected constant, S_P injected
    near-zero). An axis ablation on synthetic data would just
    rediscover what was injected -- circular. This script runs the
    actual Paper 3A kernel (vendored in ../kernel/) on real SQL
    workloads, so axis informativeness is a genuine property of the
    kernel + workload, not an artefact.

WORKLOADS (three, spanning RDBMS and NoSQL)
    tpch   -- TPC-H phase-shifted trace from the vendored
              kernel/workload_generator.py (pattern A,B,A,C).
    job    -- the 113-query Join Order Benchmark (IMDB); its 33
              template families are partitioned into 3 phases and run
              in the same A,B,A,C pattern.
    mongo  -- MongoDB E0: the kernel was already run for Paper 3B;
              its per-window 5-D output is reused as-is (no re-run).

WHAT IT MEASURES, per workload
    f-vector  : f(t) = (S_R,S_V,S_T,S_A,S_P) similarity to a fixed
                reference window (window 0). Real kernel output.
    DCI       : participation ratio of the drift-motion covariance
                spectrum -- the same DCI used everywhere else.
    ABLATION  : a greedy, sequential leave-one-axis-out ablation of the
                drift detector. The detector for an axis-subset A is
                the motion magnitude ||Df_A||; at each step the axis
                whose removal least hurts detection AUC is dropped.
                Output: detection AUC at 5,4,3,2,1 axes + the drop
                order. Greedy/sequential (not independent leave-one-
                out) because the axes are correlated -- independent
                LOO under-counts correlated axes.
    The headline check: does DCI predict how far the ablation can cut?

OUTPUT (timestamped; every CSV row carries a UTC timestamp)
    kernel_validation_summary.csv   per workload: DCI + ablation curve
    kernel_validation_fvectors.csv  every real-kernel f-vector
    kernel_validation_run.json      structured results + verdict
    kernel_validation_fig.png       ablation curves (if matplotlib)

RUN
    pip install pandas numpy scipy --break-system-packages   # mpl optional
    python kernel_validation.py
    # must sit in Paper 3C/geometry_E0/ ; needs ../kernel/ and
    # drift_harness.py alongside.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                 # drift_harness.py
sys.path.insert(0, str(_HERE.parent))          # Paper 3C/  -> import kernel.*

FEATURES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
WINDOW_SIZE = 5
N_WINDOWS = 24
DRIFT_WINDOWS = [6, 12, 18]                    # A,B,A,C phase transitions


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (higher score -> label 1). Ties averaged."""
    labels = np.asarray(labels, float)
    scores = np.asarray(scores, float)
    m = np.isfinite(scores)
    labels, scores = labels[m], scores[m]
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def dci_of(motions: np.ndarray) -> dict:
    """DCI = participation ratio of the motion-covariance spectrum."""
    if motions.shape[0] < 2:
        return {"dci": float("nan"), "pc1_fraction": float("nan"),
                "n_motions": int(motions.shape[0])}
    _, s, _ = np.linalg.svd(motions, full_matrices=False)
    e = s ** 2
    p = e / e.sum() if e.sum() > 0 else e
    inv = float(np.sum(p ** 2))
    return {"dci": float(1.0 / inv) if inv > 0 else float("nan"),
            "pc1_fraction": float(p[0]), "n_motions": int(motions.shape[0])}


def greedy_ablation(fvecs: list, drift_idx: list) -> dict:
    """Greedy sequential leave-one-axis-out ablation of the motion detector.

    fvecs     : list of (n_win, 5) real-kernel f-vector trajectories.
    drift_idx : window indices that are drift events.

    The detector for an axis subset A: score(t) = ||Df_A(t)||, the L2
    norm of the lag-1 motion restricted to A. AUC is taken vs the drift
    labels, pooled over trajectories. Starting from all 5 axes we
    repeatedly drop the axis whose removal leaves the highest AUC.
    """
    # build pooled motion vectors + labels
    motions, labels = [], []
    for f in fvecs:
        for t in range(1, len(f)):
            motions.append(f[t] - f[t - 1])
            labels.append(1 if t in drift_idx else 0)
    M = np.asarray(motions, float)             # (n, 5)
    y = np.asarray(labels, int)

    def subset_auc(keep: list) -> float:
        sc = np.linalg.norm(M[:, keep], axis=1)
        return auc(y, sc)

    keep = list(range(5))
    curve = [{"n_axes": 5, "auc": subset_auc(keep),
              "axes": [FEATURES[i] for i in keep]}]
    drop_order = []
    while len(keep) > 1:
        # drop the axis whose removal best preserves AUC
        best, best_auc = None, -1.0
        for ax in keep:
            cand = [k for k in keep if k != ax]
            a = subset_auc(cand)
            if a > best_auc:
                best, best_auc = ax, a
        keep = [k for k in keep if k != best]
        drop_order.append(FEATURES[best])
        curve.append({"n_axes": len(keep), "auc": best_auc,
                       "axes": [FEATURES[i] for i in keep]})
    return {"curve": curve, "drop_order": drop_order,
            "drift_motion": M[np.asarray(labels) == 1]}


def detector_comparison(trajs: list, drift_idx: list) -> dict:
    """1-D vs 5-D drift detection on real-kernel output.

    trajs : list of (fvec (n,5), composite_score (n,)).
    Three STATIC detectors, all scored vs the drift labels:
      auc_1d       -- top-1 PC of the drift-event deviation d=f-f_steady
      auc_5d_proxy -- ||d||, the full 5-axis deviation magnitude
      auc_5d_real  -- the REAL Paper 3A composite kernel: 1 - hsm_score
    auc_5d_real is the comparison that matters: it is the actual
    weighted + multi-scale HSM kernel, not a homemade 5-axis surrogate.
    """
    alld, allsc, lab = [], [], []
    for fv, sc in trajs:
        n = len(fv)
        nd = np.array([w not in drift_idx for w in range(n)])
        if nd.sum() < 2:
            continue
        f_steady = fv[nd].mean(axis=0)
        for w in range(n):
            alld.append(fv[w] - f_steady)
            allsc.append(sc[w])
            lab.append(1 if w in drift_idx else 0)
    alld = np.asarray(alld, float)
    allsc = np.asarray(allsc, float)
    lab = np.asarray(lab, int)
    if (lab == 1).sum() < 2 or (lab == 0).sum() < 2:
        return {"auc_1d": float("nan"), "auc_5d_proxy": float("nan"),
                "auc_5d_real": float("nan")}
    driftd = alld[lab == 1]
    C = (driftd.T @ driftd) / len(driftd)
    evals, evecs = np.linalg.eigh(C)
    v1 = evecs[:, int(np.argmax(evals))]
    return {
        "auc_1d": float(auc(lab, np.abs(alld @ v1))),
        "auc_5d_proxy": float(auc(lab, np.linalg.norm(alld, axis=1))),
        "auc_5d_real": (float(auc(lab, -allsc))
                        if np.all(np.isfinite(allsc)) else float("nan")),
    }


# --------------------------------------------------------------------------
# Workload builders -> real-kernel f-vector trajectories
# --------------------------------------------------------------------------
def kernel_fvectors(window_sql: list) -> tuple:
    """Run the real 5-D kernel per window vs window 0; return BOTH the
    5-D similarity f-vector and the real composite hsm_score."""
    from kernel.hsm_similarity import build_window, hsm_score
    wins = [build_window(sql) for sql in window_sql]
    ref = wins[0]
    fv, sc = [], []
    for w in wins:
        score, dims = hsm_score(ref, w)
        fv.append([dims[k] for k in FEATURES])
        sc.append(float(score))
    return np.asarray(fv, float), np.asarray(sc, float)


def build_tpch(n_seeds: int) -> list:
    from kernel.workload_generator import get_workload_trace, get_window_queries
    trajs = []
    for seed in range(n_seeds):
        trace = get_workload_trace(queries_per_phase=30, seed=seed)
        wins = [get_window_queries(trace, i * WINDOW_SIZE, WINDOW_SIZE)
                for i in range(N_WINDOWS)]
        trajs.append(kernel_fvectors(wins))
    return trajs


def _load_job_templates(queries_dir: Path) -> dict:
    """Load JOB *.sql grouped by template family (leading number)."""
    fam = {}
    for p in sorted(queries_dir.glob("*.sql")):
        m = re.match(r"^(\d+)([a-z]?)\.sql$", p.name)
        if not m:
            continue
        fam.setdefault(int(m.group(1)), []).append(p.read_text())
    return fam


def build_job(n_seeds: int, queries_dir: Path) -> list:
    fam = _load_job_templates(queries_dir)
    if not fam:
        raise FileNotFoundError(f"no JOB *.sql in {queries_dir}")
    nums = sorted(fam)
    third = len(nums) // 3
    phase_fams = {"A": nums[:third], "B": nums[third:2 * third],
                  "C": nums[2 * third:]}
    phase_sql = {ph: [q for n in fs for q in fam[n]]
                 for ph, fs in phase_fams.items()}
    schedule = ["A", "B", "A", "C"]            # same pattern as TPC-H
    trajs = []
    for seed in range(n_seeds):
        rng = random.Random(1000 + seed)
        trace = []
        for ph in schedule:
            pool = phase_sql[ph]
            seq = []
            while len(seq) < 30:
                batch = pool[:]
                rng.shuffle(batch)
                seq.extend(batch)
            trace.extend(seq[:30])
        wins = [trace[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
                for i in range(N_WINDOWS)]
        trajs.append(kernel_fvectors(wins))
    return trajs


def load_mongo_e0(path: Path) -> tuple:
    """MongoDB E0 -- existing real-kernel output. Returns (trajs, drift_idx).

    drift_idx are derived from the drift_truth column.
    """
    df = pd.read_csv(path)
    if "no_advisor" in set(df.strategy):
        df = df[df.strategy == "no_advisor"]
    has_hsm = "HSM" in df.columns       # the real composite is in the log
    trajs, drift_idx = [], None
    for _, g in df.groupby(["strategy", "block", "block_seed"], sort=False):
        g = g.sort_values("window").reset_index(drop=True)
        fv = g[FEATURES].to_numpy(float)
        sc = (g["HSM"].to_numpy(float) if has_hsm
              else np.full(len(g), np.nan))
        trajs.append((fv, sc))
        if drift_idx is None:
            drift_idx = list(g.index[g.drift_truth == 1])
    return trajs, drift_idx


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="TPC-H / JOB traces to generate")
    ap.add_argument("--e0-input", type=str, default=None,
                    help="MongoDB E0 breakdown_per_window.csv")
    ap.add_argument("--job-dir", type=str, default=None,
                    help="directory of JOB *.sql files")
    args = ap.parse_args()

    out_root = (Path(args.outdir).expanduser().resolve() if args.outdir
                else _HERE / "out")
    run_ts = utc_now_iso()
    run_id = run_ts.replace("-", "").replace(":", "")
    outdir = out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[run ] id={run_id}")

    # default data locations, resolved relative to this script
    import drift_harness as H
    e0_path = (Path(args.e0_input).expanduser().resolve() if args.e0_input
               else H.default_input_path(_HERE))
    job_dir = (Path(args.job_dir).expanduser().resolve() if args.job_dir
               else (_HERE.parent.parent / "Paper 3B" / "HSM_gated_3B"
                     / "code" / "data" / "job" / "queries"))

    datasets = {}        # name -> (trajectories, drift_idx)
    # ---- tpch -----------------------------------------------------------
    try:
        trajs = build_tpch(args.n_seeds)
        datasets["tpch"] = (trajs, DRIFT_WINDOWS)
        print(f"[tpch ] {len(trajs)} traces x {N_WINDOWS} windows  (real kernel)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[tpch ] SKIPPED: {exc}")
    # ---- job ------------------------------------------------------------
    try:
        trajs = build_job(args.n_seeds, job_dir)
        datasets["job"] = (trajs, DRIFT_WINDOWS)
        print(f"[job  ] {len(trajs)} traces x {N_WINDOWS} windows  (real kernel)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[job  ] SKIPPED: {exc}")
    # ---- mongo e0 -------------------------------------------------------
    try:
        trajs, didx = load_mongo_e0(e0_path)
        datasets["mongo_e0"] = (trajs, didx)
        print(f"[mongo] {len(trajs)} trials x {N_WINDOWS} windows  "
              f"(existing kernel output, drift {didx})")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[mongo] SKIPPED: {exc}")

    if not datasets:
        print("[ERR ] no dataset could be loaded", file=sys.stderr)
        return 2

    # ---- analysis -------------------------------------------------------
    summary_rows, fvec_rows, results = [], [], {}
    for name, (trajs, drift_idx) in datasets.items():
        fvec_list = [t[0] for t in trajs]
        abl = greedy_ablation(fvec_list, drift_idx)
        dci = dci_of(abl["drift_motion"])
        cmp = detector_comparison(trajs, drift_idx)
        results[name] = {"dci": dci, "ablation": abl, "compare": cmp}
        # ablation curve rows
        for c in abl["curve"]:
            summary_rows.append({
                "analysis_timestamp_utc": run_ts, "run_id": run_id,
                "workload": name, "n_axes": c["n_axes"],
                "detection_auc": c["auc"], "axes_kept": "+".join(c["axes"]),
                "dci": dci["dci"], "pc1_fraction": dci["pc1_fraction"],
                "n_traces": len(trajs),
            })
        # raw f-vectors + the real composite hsm_score
        for ti, (f, sc) in enumerate(trajs):
            for w in range(len(f)):
                fvec_rows.append({
                    "analysis_timestamp_utc": run_ts, "run_id": run_id,
                    "workload": name, "trace": ti, "window": w,
                    "is_drift": int(w in drift_idx),
                    "hsm_score_real": float(sc[w]),
                    **{k: float(f[w][i]) for i, k in enumerate(FEATURES)},
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "kernel_validation_summary.csv", index=False)
    pd.DataFrame(fvec_rows).to_csv(
        outdir / "kernel_validation_fvectors.csv", index=False)

    # ---- verdict --------------------------------------------------------
    verdict_lines = []
    for name, res in results.items():
        dci = res["dci"]["dci"]
        curve = {c["n_axes"]: c["auc"] for c in res["ablation"]["curve"]}
        auc5 = curve.get(5, float("nan"))
        # smallest #axes whose AUC is within 0.02 of the 5-axis AUC
        min_axes = 5
        for k in (1, 2, 3, 4):
            if np.isfinite(curve.get(k, float("nan"))) and \
               curve[k] >= auc5 - 0.02:
                min_axes = k
                break
        verdict_lines.append(
            f"{name}: DCI={dci:.2f}, ablation says >={min_axes} axes "
            f"needed (AUC {curve.get(min_axes, float('nan')):.3f} vs "
            f"5-axis {auc5:.3f}); drop order "
            f"{' -> '.join(res['ablation']['drop_order'])}. "
            f"DCI~required-axes "
            f"{'CONSISTENT' if abs(dci - min_axes) <= 1.0 else 'DIVERGENT'}.")

    run_json = {
        "run": {"analysis_timestamp_utc": run_ts, "run_id": run_id,
                "script": Path(__file__).name, "n_seeds": args.n_seeds,
                "datasets": list(datasets)},
        "results": {
            name: {
                "dci": res["dci"],
                "ablation_curve": res["ablation"]["curve"],
                "drop_order": res["ablation"]["drop_order"],
                "detector_comparison": res["compare"],
            } for name, res in results.items()},
        "verdict": verdict_lines,
    }
    (outdir / "kernel_validation_run.json").write_text(
        json.dumps(run_json, indent=2, default=str))

    # ---- figure ---------------------------------------------------------
    fig_path = outdir / "kernel_validation_fig.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for name, res in results.items():
            cur = res["ablation"]["curve"]
            xs = [c["n_axes"] for c in cur]
            ys = [c["auc"] for c in cur]
            ax.plot(xs, ys, marker="o",
                    label=f"{name}  (DCI={res['dci']['dci']:.2f})")
        ax.set_xlabel("# axes kept (greedy ablation)")
        ax.set_ylabel("drift detection AUC")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.invert_xaxis()
        ax.axhline(0.5, color="#888", ls=":", lw=1)
        ax.set_title(f"Paper 3C  -  real-kernel axis ablation   run {run_id}",
                     fontsize=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
    except Exception as exc:                                   # noqa: BLE001
        fig_path = None
        print(f"[warn] figure skipped ({exc})")

    # ---- console --------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  REAL-KERNEL VALIDATION   run {run_id}")
    print("=" * 70)
    for name, res in results.items():
        cur = res["ablation"]["curve"]
        cm = res["compare"]
        print(f"\n  [{name}]  DCI = {res['dci']['dci']:.3f}   "
              f"PC1 = {res['dci']['pc1_fraction']:.3f}")
        print(f"    1-D vs 5-D detection AUC:")
        print(f"      1-D  (top-1 PC)         : {cm['auc_1d']:.3f}")
        print(f"      5-D  proxy (||d||)      : {cm['auc_5d_proxy']:.3f}")
        print(f"      5-D  REAL (hsm_score)   : {cm['auc_5d_real']:.3f}")
        print(f"    axis ablation (greedy):")
        for c in cur:
            print(f"      {c['n_axes']} axes  AUC={c['auc']:.3f}   "
                  f"[{'+'.join(c['axes'])}]")
        print(f"    drop order: {' -> '.join(res['ablation']['drop_order'])}")
    print()
    print("  VERDICT (1-D vs the REAL 5-D kernel hsm_score):")
    for name, res in results.items():
        cm = res["compare"]
        gap = (cm["auc_5d_real"] - cm["auc_1d"]
               if np.isfinite(cm["auc_5d_real"]) else float("nan"))
        pxy = (cm["auc_5d_real"] - cm["auc_5d_proxy"]
               if np.isfinite(cm["auc_5d_real"]) else float("nan"))
        print(f"    - {name}: real-5-D - 1-D = {gap:+.3f}   "
              f"(real-5-D - our proxy-5-D = {pxy:+.3f})")
    print()
    print("  VERDICT (axis ablation):")
    for v in verdict_lines:
        print(f"    - {v}")
    print()
    for f in ("kernel_validation_summary.csv", "kernel_validation_fvectors.csv",
              "kernel_validation_run.json"):
        print(f"[out ] {outdir / f}")
    if fig_path:
        print(f"[out ] {fig_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
