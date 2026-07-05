#!/usr/bin/env python3
"""
cost_benefit.py  (v2 -- heterogeneous drift)
============================================

Paper 3C -- the RQ3 headline experiment: when is the cheap 1-D detector
enough, when is the full 5-D kernel needed, and is 5-D worth its cost?

WHY v2.  The first version produced only low-DCI workloads, because
ordinary "the query mix changed" drift moves the kernel's (correlated)
template axes together -- rank-1, DCI ~ 1. To exhibit the regime where
5-D is genuinely needed we drive HETEROGENEOUS drift: across one
trajectory the transitions are of different *types*, each moving a
different kernel axis.

  template transition  -> moves S_R, S_T (and S_A): the query-template
                          mix changes (a feature launch).
  volume   transition  -> moves S_V: the per-window query count changes
                          (a traffic change).
  (An 'arrival' type was tried and dropped: the kernel's S_P does not
   respond to arrival-timing changes when templates are held fixed --
   s_p returns 1.0 -- consistent with E0, where S_P carried no signal.)

  config "template_only" -> all transitions template -> low DCI;
  config "mixed"         -> transitions rotate template / volume, so the
                            pooled drift spans two axes -> high DCI;
  "volume_only"          -> single-axis control.

A 1-D detector watching one axis catches its own transition type but
misses the others -> on "mixed" it fails; the 5-D kernel sees every
axis -> it succeeds. That contrast, with cost attached, is RQ3.

DETECTOR / REFERENCE.  f(t) = the kernel's 5-D similarity of window t
to the preceding window (adjacent-window comparison); in a gated
advisor loop the reference resets each time the advisor re-tunes, so
drift is a transient dip at the transition.
  5-D detector : the Mahalanobis distance of the deviation d=f-f_steady
                 under the steady-window noise covariance -- the
                 optimal use of all five dimensions. (The scalar
                 hsm_score composite is reported too, but a weighted
                 *average* dilutes single-axis drift, so it is a gating
                 score, not a detector.)
  1-D detector : 1 - the single most-informative similarity dimension.

METHOD.  10 seeds per (workload x config) cell, each seed derived
deterministically (hashlib, not the salted built-in hash) so a re-run
reproduces the numbers exactly; report mean, SD and the t-based 95%
CI. Overhead (wall-clock per detection) and throughput are
instrumented directly on the real kernel over 5 timing repeats, so
the cost ratio itself carries an SD. Window size 20 (matching the E0
benchmark) so the kernel statistics are not starved.

OUTPUT (timestamped folder; every CSV row carries a UTC timestamp)
  cost_benefit_raw.csv      one row per (workload, config, seed)
  cost_benefit_summary.csv  per (workload, config): mean / SD / 95% CI
  cost_benefit_epsilon.csv  per (workload, config, axis): the per-axis
                            sufficiency tolerance epsilon_d
  cost_benefit_run.json     overhead, selector aggregate, verdict
  cost_benefit_fig.png      accuracy-vs-DCI + cost/accuracy Pareto

RUN
  pip install pandas numpy scipy --break-system-packages   # mpl optional
  python cost_benefit.py                 # 50 seeds, 3 configs, 3 workloads
  # must sit in Paper 3C/geometry_E0/ ; needs ../kernel/ and delong.py.
  # workloads: TPC-H + JOB (analytic) + pgbench (OLTP).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # Paper 3C/ -> import kernel.*
sys.path.insert(0, str(_HERE))                 # geometry_E0/ -> import delong

from delong import (delong_gap_variances, combine_across_seed,
                    epsilon_from_sigma2)

FEATURES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
WINDOW_SIZE = 20            # queries per window (matches E0; not starved)
WIN_PER_PHASE = 6
N_PHASES = 7                # 6 transitions per trajectory
CONFIGS = ["template_only", "volume_only", "mixed"]
DCI_ROUTE_THRESHOLD = 1.5   # selector: DCI < this -> 1-D, else 5-D
VOLUME_LEVELS = [12, 20, 32]
N_REF = 15                  # minimum reliable seed count (seed_analysis.py);
#                             practical floor delta_d = epsilon_d(N_REF) is
#                             evaluated here, so the sufficiency verdict does
#                             not drift with the run's --seeds.


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary parts.

    Python's built-in hash() is salted per process (PYTHONHASHSEED) for
    str/bytes, so hash(("tpch", "mixed", 0)) differs run to run -- it
    cannot be used to seed a reproducible experiment. hashlib is stable.
    """
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (higher score -> label 1)."""
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


def mean_sd_ci(vals: list) -> dict:
    """Mean, SD, and t-based 95% CI half-width for a sample."""
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    n = len(a)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"),
                "ci95": float("nan"), "n": 0}
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    ci = 0.0
    if n > 1:
        try:
            from scipy.stats import t as t_dist
            tcrit = float(t_dist.ppf(0.975, n - 1))
        except Exception:                                      # noqa: BLE001
            tcrit = 1.96
        ci = tcrit * sd / np.sqrt(n)
    return {"mean": mean, "sd": sd, "ci95": float(ci), "n": n}


# --------------------------------------------------------------------------
# Workload pools  (id -> list of concrete SQL strings)
# --------------------------------------------------------------------------
def tpch_pool() -> dict:
    from kernel.workload_generator import TPCH_TEMPLATES, QUERY_PARAM_POOLS
    pool = {}
    for qid, tmpl in TPCH_TEMPLATES.items():
        variants = []
        for params in QUERY_PARAM_POOLS.get(qid, [{}])[:8]:
            try:
                variants.append(tmpl.format(**params))
            except Exception:                                  # noqa: BLE001
                pass
        pool[qid] = variants or [tmpl]
    return pool


def job_pool(queries_dir: Path) -> dict:
    pool = {}
    for p in sorted(queries_dir.glob("*.sql")):
        m = re.match(r"^(\d+)([a-z]?)\.sql$", p.name)
        if m:
            pool.setdefault(int(m.group(1)), []).append(p.read_text())
    return pool


def pgbench_pool(n_variants: int = 8) -> dict:
    """The pgbench TPC-B-like OLTP workload as a template pool.

    pgbench's built-in `tpcb-like` script is the canonical Postgres
    micro-benchmark: five DML statements over the pgbench_* schema --
    a schema the vendored kernel's extractor already recognises
    (PGBENCH_TABLES / PGBENCH_COLUMNS in hsm_similarity.py). Concrete
    parameter values are substituted to give per-template variants.
    An OLTP workload by design has few statement templates: pgbench is
    a deliberate low-diversity contrast to the analytic TPC-H / JOB
    corpora, and a third SQL workload for the portability check (RQ4).
    """
    rng = random.Random(20260522)
    templates = {
        1: ("UPDATE pgbench_accounts SET abalance = abalance + {delta} "
            "WHERE aid = {aid}"),
        2: "SELECT abalance FROM pgbench_accounts WHERE aid = {aid}",
        3: ("UPDATE pgbench_tellers SET tbalance = tbalance + {delta} "
            "WHERE tid = {tid}"),
        4: ("UPDATE pgbench_branches SET bbalance = bbalance + {delta} "
            "WHERE bid = {bid}"),
        5: ("INSERT INTO pgbench_history (tid, bid, aid, delta, mtime) "
            "VALUES ({tid}, {bid}, {aid}, {delta}, CURRENT_TIMESTAMP)"),
    }
    pool = {}
    for sid, tmpl in templates.items():
        variants = []
        for _ in range(n_variants):
            p = {"delta": rng.randint(-5000, 5000),
                 "aid": rng.randint(1, 100000),
                 "tid": rng.randint(1, 100),
                 "bid": rng.randint(1, 10)}
            try:
                variants.append(tmpl.format(**p))
            except Exception:                                  # noqa: BLE001
                pass
        pool[sid] = variants or [tmpl]
    return pool


# --------------------------------------------------------------------------
# Heterogeneous drift generator -> list of (sql_list, timestamps) windows
# --------------------------------------------------------------------------
def _timestamps(count: int, arrival: str, rng: random.Random) -> list:
    if arrival == "uniform":
        return [float(i) for i in range(count)]
    # bursty: a few tight clusters
    centres = sorted(rng.uniform(0, count) for _ in range(3))
    ts = sorted(rng.gauss(rng.choice(centres), count * 0.03)
                for _ in range(count))
    return [float(x) for x in ts]


def _make_window(pool: dict, templ_ids: list, count: int, arrival: str,
                 rng: random.Random) -> tuple:
    sql = []
    while len(sql) < count:
        for qid in rng.sample(templ_ids, len(templ_ids)):
            sql.append(rng.choice(pool[qid]))
    return sql[:count], _timestamps(count, arrival, rng)


def build_trajectory(pool: dict, config: str, seed: int) -> tuple:
    """Return (windows, drift_idx). windows = list of (sql_list, ts).

    Transitions follow `config`: a fixed type, or 'mixed' = rotate
    template / volume / arrival so the pooled drift spans the axes.
    """
    rng = random.Random(seed)
    ids = list(pool)
    # k active templates per window. Must be < pool size, otherwise a
    # template swap has no spare ids to draw from and is a no-op (this
    # bit a 5-template pgbench pool: every "swap" returned the same
    # set). min(5, len-1) keeps k=5 for the large analytic pools and
    # leaves swap room for the small OLTP pool.
    k = max(2, min(5, len(ids) - 1))
    state = {"templ": rng.sample(ids, k), "count": WINDOW_SIZE,
             "arrival": "uniform"}
    if config == "mixed":
        base = ["template", "volume"]
        types = [base[i % 2] for i in range(N_PHASES - 1)]
    else:
        types = [config.replace("_only", "")] * (N_PHASES - 1)

    windows, drift_idx, widx = [], set(), 0
    for ph in range(N_PHASES):
        if ph > 0:
            tt = types[ph - 1]
            if tt == "template":
                n_swap = max(1, k // 2)
                keep = rng.sample(state["templ"], k - n_swap)
                avail = [i for i in ids if i not in keep]
                state["templ"] = keep + rng.sample(
                    avail, min(n_swap, len(avail)))
            elif tt == "volume":
                state["count"] = rng.choice(
                    [c for c in VOLUME_LEVELS if c != state["count"]])
            elif tt == "arrival":
                state["arrival"] = ("bursty" if state["arrival"] == "uniform"
                                    else "uniform")
            drift_idx.add(widx)
        for _ in range(WIN_PER_PHASE):
            windows.append(_make_window(pool, state["templ"],
                                        state["count"], state["arrival"], rng))
            widx += 1
    return windows, drift_idx


# --------------------------------------------------------------------------
# Real kernel + per-trajectory analysis
# --------------------------------------------------------------------------
def kernel_adjacent(windows: list) -> tuple:
    """f(t) = (5-D dims, composite) of hsm_score(window t-1, window t)."""
    from kernel.hsm_similarity import build_window, hsm_score
    wins = [build_window(sql, ts) for sql, ts in windows]
    fv, sc = [], []
    for t in range(1, len(wins)):
        score, dims = hsm_score(wins[t - 1], wins[t])
        fv.append([dims[k] for k in FEATURES])
        sc.append(float(score))
    return np.asarray(fv, float), np.asarray(sc, float)


def analyse(fv: np.ndarray, sc: np.ndarray, drift_idx: set) -> dict | None:
    """DCI + 1-D (best single dim) + 5-D (Mahalanobis) detection AUC.

    f-index t corresponds to window t+1 (f starts at window 1).

    The 5-D detector is the Mahalanobis distance of the deviation
    d = f - f_steady under the steady-window noise covariance: the
    statistically optimal way to fuse five (correlated) similarity
    dimensions into one drift score. The scalar hsm_score composite
    (auc_5d_composite) and the plain L2 norm of the deviation
    (auc_5d_l2) are reported alongside as references -- the composite
    is a weighted *average*, so it dilutes single-axis drift and
    serves as a gating score, not a detector.

    SUFFICIENCY.  For each axis d, delong_gap_variances returns the AUC
    gap (auc_5d - auc_d) and its DeLong (1988) variance; main() pools
    these across seeds and evaluates the tolerance at a fixed N_REF
    seeds -- delta_ref, the n-stable practical floor. Axis d is
    "sufficient" iff mean_gap <= delta_ref. delta_ref is computed from
    each cell's own variance (no hand-set constant) and does not shrink
    with --seeds, so the regime map is seed-count-stable. See delong.py
    and seed_analysis.py.
    """
    n = len(fv)
    lab = np.array([1 if (i + 1) in drift_idx else 0 for i in range(n)])
    if lab.sum() < 2 or (lab == 0).sum() < 2:
        return None
    nd = lab == 0
    # 1-D: the single most-informative similarity dimension
    per_dim = {FEATURES[j]: auc(lab, 1.0 - fv[:, j])
               for j in range(len(FEATURES))}
    best_dim = max(per_dim, key=lambda d: per_dim[d])
    auc_1d = per_dim[best_dim]
    # deviation from the steady-window mean
    d = fv - fv[nd].mean(axis=0)
    # 5-D detector: Mahalanobis distance under the steady-window covariance
    Sig = np.cov(d[nd].T) + 1e-6 * np.eye(d.shape[1])
    Pinv = np.linalg.pinv(Sig)
    maha = np.einsum("ij,jk,ik->i", d, Pinv, d)
    auc_5d = auc(lab, maha)
    # references: plain L2 norm of the deviation, and the scalar composite
    auc_5d_l2 = auc(lab, np.linalg.norm(d, axis=1))
    auc_5d_composite = auc(lab, 1.0 - sc)
    # per-dimension DeLong gap variances: the 5-D Mahalanobis detector
    # vs each axis (score_d = 1 - S_d, drift lowers similarity). One
    # covariance over all five axes + the 5-D detector, so every
    # cross-correlation is accounted for. gap_d and var_gap_d feed the
    # cell-level across-seed combination in main().
    gv = delong_gap_variances(lab, maha, 1.0 - fv)
    per_dim_gap = {FEATURES[j]: gv["per_dim"][j]["gap"] for j in range(5)}
    per_dim_var = {FEATURES[j]: gv["per_dim"][j]["var_gap"] for j in range(5)}
    # DCI on the drift-window deviation
    driftd = d[lab == 1]
    C = (driftd.T @ driftd) / len(driftd)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    tot = ev.sum()
    dci = float((tot ** 2) / np.sum(ev ** 2)) if tot > 0 else float("nan")
    return {"dci": dci, "auc_1d": auc_1d, "auc_5d": auc_5d,
            "auc_5d_l2": auc_5d_l2, "auc_5d_composite": auc_5d_composite,
            "best_1d_dim": best_dim, "delta_auc": auc_5d - auc_1d,
            "per_dim_gap": per_dim_gap, "per_dim_var": per_dim_var}


def measure_overhead(windows: list, n_pairs: int = 300,
                     n_reps: int = 5) -> dict:
    """Wall-clock per detection: one dimension (S_T) vs full hsm_score.

    The 1-D / 5-D timing loop is repeated `n_reps` times so the cost
    ratio carries an SD -- it is a headline number for RQ3.
    """
    from kernel.hsm_similarity import (build_window, hsm_score,
                                       s_r, s_v, s_t, s_a, s_p)
    wins = [build_window(sql, ts) for sql, ts in windows]
    rng = random.Random(0)
    pairs = [(rng.choice(wins), rng.choice(wins)) for _ in range(n_pairs)]
    fns = {"S_R": s_r, "S_V": s_v, "S_T": s_t, "S_A": s_a, "S_P": s_p}

    def timed(fn):
        for a, b in pairs[:20]:                       # warm-up
            fn(a, b)
        t0 = time.perf_counter()
        for a, b in pairs:
            fn(a, b)
        return (time.perf_counter() - t0) / len(pairs)

    per_dim_reps = {k: [] for k in fns}
    reps_5d = []
    for _ in range(n_reps):
        for k, fn in fns.items():
            per_dim_reps[k].append(timed(fn))
        reps_5d.append(timed(lambda a, b: hsm_score(a, b)))
    per_dim = {k: float(np.mean(v)) for k, v in per_dim_reps.items()}
    reps_1d = per_dim_reps["S_T"]
    t_1d, t_5d = float(np.mean(reps_1d)), float(np.mean(reps_5d))
    ratios = [r5 / r1 for r5, r1 in zip(reps_5d, reps_1d) if r1 > 0]
    rat = mean_sd_ci(ratios)
    return {"overhead_1d_s": t_1d, "overhead_5d_s": t_5d,
            "overhead_1d_sd_s": (float(np.std(reps_1d, ddof=1))
                                 if len(reps_1d) > 1 else 0.0),
            "overhead_5d_sd_s": (float(np.std(reps_5d, ddof=1))
                                 if len(reps_5d) > 1 else 0.0),
            "cost_ratio_5d_over_1d": rat["mean"],
            "cost_ratio_sd": rat["sd"], "cost_ratio_ci95": rat["ci95"],
            "n_overhead_reps": n_reps,
            "throughput_1d_per_s": 1.0 / t_1d if t_1d > 0 else float("nan"),
            "throughput_5d_per_s": 1.0 / t_5d if t_5d > 0 else float("nan"),
            "per_dimension_overhead_s": per_dim}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--n-ref", type=int, default=N_REF,
                    help="seeds at which the practical floor delta_d is "
                         "evaluated (default from seed_analysis.py)")
    ap.add_argument("--job-dir", type=str, default=None)
    args = ap.parse_args()

    out_root = (Path(args.outdir).expanduser().resolve() if args.outdir
                else _HERE / "out")
    run_ts = utc_now_iso()
    run_id = run_ts.replace("-", "").replace(":", "")
    outdir = out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[run ] id={run_id}  seeds={args.seeds}")

    job_dir = (Path(args.job_dir).expanduser().resolve() if args.job_dir
               else (_HERE.parent.parent / "Paper 3B" / "HSM_gated_3B"
                     / "code" / "data" / "job" / "queries"))
    pools = {}
    try:
        pools["tpch"] = tpch_pool()
        print(f"[tpch] {len(pools['tpch'])} templates")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[tpch] SKIPPED: {exc}")
    try:
        jp = job_pool(job_dir)
        if jp:
            pools["job"] = jp
            print(f"[job ] {len(jp)} template families")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[job ] SKIPPED: {exc}")
    try:
        pp = pgbench_pool()
        if pp:
            pools["pgbench"] = pp
            print(f"[pgb ] {len(pp)} statement templates (OLTP)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[pgb ] SKIPPED: {exc}")
    if not pools:
        print("[ERR ] no workload pool available", file=sys.stderr)
        return 2

    # ---- overhead (measured once) ---------------------------------------
    ov = measure_overhead(build_trajectory(next(iter(pools.values())),
                                           "mixed", 0)[0])
    print(f"[cost] 1-D {ov['overhead_1d_s']*1e3:.3f} ms  "
          f"5-D {ov['overhead_5d_s']*1e3:.3f} ms  "
          f"-> {ov['cost_ratio_5d_over_1d']:.1f}x "
          f"+/- {ov['cost_ratio_sd']:.1f}")

    # ---- accuracy: seeds x configs x workloads --------------------------
    raw = []
    for wl, pool in pools.items():
        for cfg in CONFIGS:
            for s in range(args.seeds):
                seed = stable_seed(wl, cfg, s)
                windows, didx = build_trajectory(pool, cfg, seed)
                fv, sc = kernel_adjacent(windows)
                res = analyse(fv, sc, didx)
                if res is None:
                    continue
                row = {
                    "analysis_timestamp_utc": run_ts, "run_id": run_id,
                    "workload": wl, "config": cfg, "seed": s,
                    "dci": res["dci"], "auc_1d": res["auc_1d"],
                    "auc_5d": res["auc_5d"],
                    "auc_5d_l2": res["auc_5d_l2"],
                    "auc_5d_composite": res["auc_5d_composite"],
                    "delta_auc": res["delta_auc"],
                    "best_1d_dim": res["best_1d_dim"],
                }
                for dname in FEATURES:
                    row[f"gap_{dname}"] = res["per_dim_gap"][dname]
                    row[f"vargap_{dname}"] = res["per_dim_var"][dname]
                raw.append(row)
    R = pd.DataFrame(raw)
    if len(R) < 8:
        print(f"[ERR ] only {len(R)} runs produced", file=sys.stderr)
        return 2
    cost_ratio = ov["cost_ratio_5d_over_1d"]
    R["benefit_ratio"] = R["delta_auc"] / cost_ratio
    R.to_csv(outdir / "cost_benefit_raw.csv", index=False)

    # ---- per-cell summary: mean / SD / 95% CI ---------------------------
    summ = []
    for (wl, cfg), g in R.groupby(["workload", "config"]):
        row = {"analysis_timestamp_utc": run_ts, "run_id": run_id,
               "workload": wl, "config": cfg, "n_seeds": len(g)}
        for col in ("dci", "auc_1d", "auc_5d", "auc_5d_composite",
                    "delta_auc", "benefit_ratio"):
            st = mean_sd_ci(list(g[col]))
            row[f"{col}_mean"] = st["mean"]
            row[f"{col}_sd"] = st["sd"]
            row[f"{col}_ci95"] = st["ci95"]
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(outdir / "cost_benefit_summary.csv", index=False)

    # ---- per-dimension epsilon + the n-stable practical floor -----------
    # epsilon_run : the across-seed t-interval of the AUC gap at the run's
    #               own seed count -- it shrinks as 1/sqrt(seeds).
    # delta_ref   : epsilon_d evaluated at a FIXED N_REF seeds -- the
    #               practical floor. It does NOT shrink with --seeds, so
    #               the sufficiency verdict (mean_gap <= delta_ref) is
    #               seed-count-stable. delta_ref is computed from each
    #               cell's own measured variance, so it self-calibrates
    #               per workload -- no hand-set constant.
    n_ref = args.n_ref
    eps_rows = []
    for (wl, cfg), g in R.groupby(["workload", "config"]):
        for dname in FEATURES:
            cc = combine_across_seed(list(g[f"gap_{dname}"]),
                                     list(g[f"vargap_{dname}"]))
            delta_ref = epsilon_from_sigma2(cc["sigma2"], n_ref)
            gap = cc["delta_combined"]
            suff = bool(np.isfinite(gap) and gap <= delta_ref)
            eps_rows.append({
                "analysis_timestamp_utc": run_ts, "run_id": run_id,
                "workload": wl, "config": cfg, "dim": dname,
                "n_seeds": cc["n"], "mean_gap": gap,
                "sd_gap": cc["sd"], "var_floor": cc["var_floor"],
                "epsilon_run": cc["epsilon_cell"], "n_ref": n_ref,
                "delta_ref": delta_ref, "dim_sufficient": suff})
    E = pd.DataFrame(eps_rows)
    E.to_csv(outdir / "cost_benefit_epsilon.csv", index=False)

    # ---- the DCI-adaptive selector vs the two fixed policies ------------
    routed_auc, routed_cost = [], []
    for _, x in R.iterrows():
        use_1d = x["dci"] < DCI_ROUTE_THRESHOLD
        routed_auc.append(x["auc_1d"] if use_1d else x["auc_5d"])
        routed_cost.append(ov["overhead_1d_s"] if use_1d
                            else ov["overhead_5d_s"])
    policies = {
        "always_1D": {"mean_auc": float(R["auc_1d"].mean()),
                      "mean_cost_ms": ov["overhead_1d_s"] * 1e3},
        "always_5D": {"mean_auc": float(R["auc_5d"].mean()),
                      "mean_cost_ms": ov["overhead_5d_s"] * 1e3},
        "DCI_selector": {"mean_auc": float(np.mean(routed_auc)),
                         "mean_cost_ms": float(np.mean(routed_cost) * 1e3),
                         "frac_routed_to_1D": float(
                             (R["dci"] < DCI_ROUTE_THRESHOLD).mean())},
    }
    a1, a5 = policies["always_1D"]["mean_auc"], policies["always_5D"]["mean_auc"]
    sel = policies["DCI_selector"]
    selector_value = {
        "accuracy_of_5D_gain_retained": ((sel["mean_auc"] - a1) / (a5 - a1)
                                         if a5 - a1 > 1e-9 else float("nan")),
        "cost_paid_vs_5D": sel["mean_cost_ms"] /
        policies["always_5D"]["mean_cost_ms"],
    }

    run_json = {
        "run": {"analysis_timestamp_utc": run_ts, "run_id": run_id,
                "script": Path(__file__).name, "seeds": args.seeds,
                "workloads": list(pools), "configs": CONFIGS,
                "window_size": WINDOW_SIZE,
                "dci_route_threshold": DCI_ROUTE_THRESHOLD},
        "overhead": ov,
        "summary": S.to_dict(orient="records"),
        "policies": policies,
        "selector_value": selector_value,
        "benefit_ratio_note": "delta_auc / cost_ratio -- AUC gained from "
                              "5-D per fold of its extra cost; ~0 = 5-D "
                              "not worth paying for.",
        "per_dim_epsilon": E.to_dict(orient="records"),
        "n_ref": n_ref,
        "epsilon_note": "per-axis: epsilon_run = the across-seed "
                        "t-interval of the DeLong AUC gap at the run's "
                        "seed count; delta_ref = epsilon_d evaluated at "
                        "N_REF seeds = the n-stable practical floor. "
                        "Axis d is sufficient iff mean_gap <= delta_ref. "
                        "delta_ref does not shrink with --seeds, so the "
                        "regime map is seed-count-stable, and it is "
                        "computed from each cell's own variance -- no "
                        "hand-set constant. DeLong 1988 for the "
                        "per-trajectory variance; equal-weight pooling "
                        "(DerSimonian-Laird is biased for AUC, see "
                        "verify_epsilon.py).",
    }
    (outdir / "cost_benefit_run.json").write_text(json.dumps(run_json, indent=2))

    # ---- figure ---------------------------------------------------------
    fig_path = outdir / "cost_benefit_fig.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        # colour-blind-safe palette (Okabe-Ito) -- CIDR I&D writing
        # guideline: no green/red, no blue/purple; shape (o/x) also
        # distinguishes 5-D from 1-D so colour is not load-bearing.
        cfg_col = {"template_only": "#E69F00", "volume_only": "#0072B2",
                   "arrival_only": "#CC79A7", "mixed": "#009E73"}
        for cfg in CONFIGS:
            g = R[R.config == cfg]
            ax[0].scatter(g["dci"], g["auc_5d"], s=30, marker="o",
                          c=cfg_col[cfg], label=f"{cfg} 5-D")
            ax[0].scatter(g["dci"], g["auc_1d"], s=30, marker="x",
                          c=cfg_col[cfg])
        ax[0].axvline(DCI_ROUTE_THRESHOLD, color="#888", ls="--")
        ax[0].set_xlabel("DCI"); ax[0].set_ylabel("detection AUC")
        ax[0].set_title("A  accuracy vs DCI   (o = 5-D, x = 1-D)",
                        fontsize=10)
        ax[0].legend(fontsize=7)
        for name, m, col in (("always 1-D", policies["always_1D"], "#1d9e75"),
                             ("always 5-D", policies["always_5D"], "#0a3d62"),
                             ("DCI selector", policies["DCI_selector"],
                              "#d62728")):
            ax[1].scatter([m["mean_cost_ms"]], [m["mean_auc"]], s=150,
                          color=col, edgecolor="black", zorder=3)
            ax[1].annotate(name, (m["mean_cost_ms"], m["mean_auc"]),
                           textcoords="offset points", xytext=(8, 4),
                           fontsize=9)
        ax[1].set_xlabel("mean cost per detection (ms)")
        ax[1].set_ylabel("mean detection AUC")
        ax[1].set_title("B  cost / accuracy -- the selector's value",
                        fontsize=10)
        fig.suptitle(f"Paper 3C  -  cost-benefit (RQ3)   run {run_id}",
                     fontsize=11, weight="bold")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
    except Exception as exc:                                   # noqa: BLE001
        fig_path = None
        print(f"[warn] figure skipped ({exc})")

    # ---- console --------------------------------------------------------
    print()
    print("=" * 74)
    print(f"  COST-BENEFIT (RQ3)   run {run_id}   {len(R)} runs")
    print("=" * 74)
    print(f"  OVERHEAD: 1-D {ov['overhead_1d_s']*1e3:.3f} ms "
          f"({ov['throughput_1d_per_s']:.0f}/s)   "
          f"5-D {ov['overhead_5d_s']*1e3:.3f} ms "
          f"({ov['throughput_5d_per_s']:.0f}/s)   "
          f"ratio {cost_ratio:.1f}x +/- {ov['cost_ratio_sd']:.1f} "
          f"({ov['n_overhead_reps']} reps)")
    print()
    print(f"  {'workload':8} {'config':13} {'DCI':>12} {'auc_1d':>14} "
          f"{'auc_5d':>14}   (mean +/- 95%CI)")
    for _, r in S.sort_values(["workload", "config"]).iterrows():
        print(f"  {r['workload']:8} {r['config']:13} "
              f"{r['dci_mean']:5.2f}+/-{r['dci_ci95']:<5.2f} "
              f"{r['auc_1d_mean']:6.3f}+/-{r['auc_1d_ci95']:<5.3f} "
              f"{r['auc_5d_mean']:6.3f}+/-{r['auc_5d_ci95']:<5.3f}")
    print()
    print(f"  PER-DIMENSION sufficiency  (5-D vs each axis; floor "
          f"delta_ref = epsilon at N_REF={n_ref};")
    print("                              axis sufficient iff "
          "mean_gap <= delta_ref)")
    print(f"  {'workload':8} {'config':13} {'dim':>5} {'mean_gap':>10} "
          f"{'delta_ref':>10} {'sufficient':>11}")
    for _, r in E.sort_values(["workload", "config", "dim"]).iterrows():
        print(f"  {r['workload']:8} {r['config']:13} {r['dim']:>5} "
              f"{r['mean_gap']:10.3f} {r['delta_ref']:10.3f} "
              f"{'yes' if r['dim_sufficient'] else 'NO':>11}")
    print()
    print("  POLICIES:")
    for name, m in policies.items():
        extra = (f"  (routed 1-D {m['frac_routed_to_1D']:.0%})"
                 if "frac_routed_to_1D" in m else "")
        print(f"    {name:13s} mean AUC {m['mean_auc']:.3f}   "
              f"cost {m['mean_cost_ms']:.3f} ms{extra}")
    sv = selector_value
    print(f"  -> selector keeps {sv['accuracy_of_5D_gain_retained']:.0%} of "
          f"the 5-D-over-1-D accuracy gain at "
          f"{sv['cost_paid_vs_5D']:.0%} of the 5-D cost")
    print()
    for f in ("cost_benefit_raw.csv", "cost_benefit_summary.csv",
              "cost_benefit_epsilon.csv", "cost_benefit_run.json"):
        print(f"[out ] {outdir / f}")
    if fig_path:
        print(f"[out ] {fig_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
