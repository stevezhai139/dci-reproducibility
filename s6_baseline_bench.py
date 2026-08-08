#!/usr/bin/env python3
"""s6_baseline_bench.py -- external-baseline sweep for the detector-resolution claim (Paper 3C).

Review item B4. Every policy replays the SAME per-window feature streams (regime-map
trajectories) under a dual cost account:
  feature cost : modeled from locked-env per-axis extraction costs
                 (cost_benefit_run.json per_dimension_overhead_s) x axes actually computed;
  arithmetic   : measured wall-clock of the decision loop (reported separately).

Scoring conventions (both reported):
  strict  : hit = fire exactly at the onset window; FA = any fire at a non-onset window.
  lag-3   : hit = first fire within [onset, onset+3]; FA = fires outside all such zones;
            mean detection delay over lag-credited onsets.  Strict-only scoring is
            structurally unfair to periodic policies (every-k), hence both.
  AUC     : graded, from per-window -log10 p-values on the common finite-sample
            Hotelling-F family (positives = onset windows, negatives = all others).
            Policies with no graded score (every-k off-probe, ADWIN) report blank.

Policies
  dci_v2_P{1,2,4,8}  DCIGateV2 (coordinate-axis cheap tier, probe cadence P)
  dci_v1             v1 gate (whitened matched filter; full features every window)
  dci_v3             DCIGateV3: union-Bonferroni cheap tier always-on; escalate to
                     full kernel while axis-share R4 < rho=0.35 (S_P extracted
                     lazily on escalated windows -- charged via features_used)
  dci_v3_a8          v3 + audit_every=8 (bounds exposure to S_P-only drift)
  dci_v3_r50         v3 with rho=0.50 (router sensitivity)
  always_multiD      5-D Mahalanobis every window
  best_axis_oracle   single coordinate axis chosen POST HOC per run by AUC (upper envelope)
  union4_raw/_bonf   OR over cheap axes S_R,S_V,S_T,S_A at alpha / alpha/4 per axis
  every_k{2,3,4}     full multi-D on every k-th window only, RANDOM per-seed phase offset
                     (fixed-cadence production auditing; unaligned with drift by design)
  adwin / adwin_s    ADWIN (river) on composite HSM score, delta=0.002 / 0.3 (full features)
  mcusum             one-sided CUSUM on the 5-D Mahalanobis stream, h calibrated on steady
  sketch1            matched filter on ONE random whitened direction fixed at fit
                     (sketching-detector pattern; still needs all five features -- the point)

Usage: python3 s6_baseline_bench.py <repro> [--seeds N] [--workloads tpch,job,pgbench]
Writes: s6_baseline_results.csv + s6_baseline_summary.json
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import f as fdist, rankdata

AXES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
CHEAP = [0, 1, 2, 3]          # S_P excluded: 94% of kernel cost
ALPHA = 0.05
LAG = 3                        # lag-tolerance budget (windows)
COMP_W = np.array([.25, .20, .20, .20, .15])   # composite weights (paper Sec 3)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def auc_rank(scores, pos_mask):
    s = np.asarray(scores, float); pos = np.asarray(pos_mask, bool)
    if pos.sum() == 0 or (~pos).sum() == 0 or not np.all(np.isfinite(s)):
        return ""
    r = rankdata(s)
    return round(float((r[pos].mean() - (pos.sum() + 1) / 2) / (~pos).sum()), 4)


class Ctx:
    def __init__(self, cal, gate_mod, v2_mod, v3_mod=None):
        X = np.asarray(cal, float)
        self.m = m = X.shape[0]
        self.mu0 = X.mean(0)
        cov = np.cov(X, rowvar=False, ddof=1) + 1e-6 * np.eye(5)
        self.sigma0 = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        ev, V = np.linalg.eigh(cov)
        self.W = V @ np.diag(np.clip(ev, 1e-12, None) ** -0.5) @ V.T
        self.c = {k: (m * (m - k)) / ((m + 1) * k * (m - 1)) for k in (1, 5)}
        self.thr = {k: float(fdist.ppf(1 - ALPHA, k, m - k)) / self.c[k] for k in (1, 5)}
        self.cal, self.gate_mod, self.v2_mod, self.v3_mod = X, gate_mod, v2_mod, v3_mod
        # mcusum h: calibrated on steady maha path
        Z = (X - self.mu0) @ self.W.T
        d5 = np.einsum("ij,ij->i", Z, Z)
        self.mc_c0 = float(np.median(d5))
        cum, path = 0.0, []
        for v in np.maximum(0.0, d5 - self.mc_c0):
            cum = max(0.0, cum + v); path.append(cum)
        self.mc_h = float(np.quantile(path, 1 - ALPHA)) * 2.0 + 1e-9
        u = np.random.default_rng(20260808).standard_normal(5)
        self.sketch_u = u / np.linalg.norm(u)

    def pval(self, stat, k):
        return float(fdist.sf(stat * self.c[k], k, self.m - k))

    def score(self, stat, k):
        return -np.log10(max(self.pval(stat, k), 1e-300))


def replay(policy, ctx, fv, pax, seed):
    """-> fires[n], scores[n]|None, feat_ms_total, arith_s, note"""
    n = len(fv); fires = np.zeros(n, int); scores = np.full(n, np.nan); feat = 0.0
    note = ""
    t0 = time.perf_counter()
    if policy.startswith("dci_v3"):
        kw = {}
        if policy == "dci_v3_a8":
            kw["audit_every"] = 8
            kw["audit_offset"] = int(np.random.default_rng([seed, 8]).integers(0, 8))
        if policy == "dci_v3_r50":
            kw["rho"] = 0.50
        g = ctx.v3_mod.DCIGateV3(alpha=ALPHA, **kw).fit(ctx.cal)
        for t in range(n):
            fires[t] = g.decide(fv[t])
            feat += sum(pax[a] for a in g.last["features_used"])  # lazy S_P model
            scores[t] = ctx.score(float(g.last["statistic"]), int(g.last["k"]))
    elif policy.startswith("dci_v"):
        if policy.startswith("dci_v2"):
            g = ctx.v2_mod.DCIGateV2(tau=1.5, alpha=ALPHA,
                                     probe_every=int(policy.split("P")[1])).fit(ctx.cal)
        else:
            g = ctx.gate_mod.DCIGate(tau=1.5, alpha=ALPHA).fit(ctx.cal)
            g.reset_trajectory()
        for t in range(n):
            need = g.features_needed() if hasattr(g, "features_needed") else AXES
            feat += sum(pax[a] for a in need)
            fires[t] = g.decide(fv[t])
            k = int(g.last.get("k") or (5 if g.last.get("regime") == "5-D" else 1))
            scores[t] = ctx.score(float(g.last["statistic"]), k)
    elif policy == "always_multiD":
        for t in range(n):
            feat += sum(pax[a] for a in AXES)
            z = ctx.W @ (fv[t] - ctx.mu0); st = float(z @ z)
            fires[t] = int(st > ctx.thr[5]); scores[t] = ctx.score(st, 5)
    elif policy == "best_axis_oracle":
        best = None
        for j in range(5):
            st = ((fv[:, j] - ctx.mu0[j]) / ctx.sigma0[j]) ** 2
            sc = np.array([ctx.score(v, 1) for v in st])
            f_ = (st > ctx.thr[1]).astype(int)
            best = (sc, f_, j) if best is None or sc.max() > best[0].max() else best
        # oracle pick happens in caller (needs truth); stash all-axis data
        return best, None, 0.0, time.perf_counter() - t0, "ORACLE_STUB"
    elif policy.startswith("union4"):
        pcut = ALPHA if policy.endswith("raw") else ALPHA / 4
        for t in range(n):
            feat += sum(pax[AXES[j]] for j in CHEAP)
            d = fv[t] - ctx.mu0
            ps = [ctx.pval((d[j] / ctx.sigma0[j]) ** 2, 1) for j in CHEAP]
            fires[t] = int(min(ps) < pcut)
            scores[t] = -np.log10(max(min(ps), 1e-300))
    elif policy.startswith("every_k"):
        k = int(policy[-1])
        off = int(np.random.default_rng([seed, k]).integers(0, k))
        for t in range(n):
            if (t - off) % k == 0:
                feat += sum(pax[a] for a in AXES)
                z = ctx.W @ (fv[t] - ctx.mu0)
                fires[t] = int(float(z @ z) > ctx.thr[5])
        scores = None; note = f"offset={off}"
    elif policy.startswith("adwin"):
        from river.drift import ADWIN
        det = ADWIN(delta=0.3 if policy.endswith("_s") else 0.002)
        comp = fv @ COMP_W
        for t in range(n):
            feat += sum(pax[a] for a in AXES)
            det.update(float(comp[t]))
            fires[t] = int(det.drift_detected)
        scores = None
    elif policy == "mcusum":
        cum = 0.0
        for t in range(n):
            feat += sum(pax[a] for a in AXES)
            z = ctx.W @ (fv[t] - ctx.mu0)
            cum = max(0.0, cum + float(z @ z) - ctx.mc_c0)
            scores[t] = cum
            if cum > ctx.mc_h:
                fires[t] = 1; cum = 0.0
    elif policy == "sketch1":
        for t in range(n):
            feat += sum(pax[a] for a in AXES)
            z = ctx.W @ (fv[t] - ctx.mu0); st = float(ctx.sketch_u @ z) ** 2
            fires[t] = int(st > ctx.thr[1]); scores[t] = ctx.score(st, 1)
    else:
        raise ValueError(policy)
    return fires, scores, feat, time.perf_counter() - t0, note


def score_run(fires, scores, truth, onsets, otypes):
    n = len(fires)
    hits = int(sum(fires[o] for o in onsets))
    zones = set()
    lag_hits, delays = 0, []
    for o in onsets:
        win = range(o, min(o + LAG + 1, n)); zones.update(win)
        ft = [t for t in win if fires[t]]
        if ft:
            lag_hits += 1; delays.append(ft[0] - o)
    fa = int(sum(fires[t] for t in range(n) if truth[t] == 0))
    fa_lag = int(sum(fires[t] for t in range(n) if t not in zones))
    pos = np.zeros(n, bool); pos[list(onsets)] = True
    return {"hits": hits, "lag_hits": lag_hits,
            "mean_delay": round(float(np.mean(delays)), 2) if delays else "",
            "fa": fa, "fa_lag": fa_lag,
            "auc": auc_rank(scores, pos) if scores is not None else "",
            "T_hits": sum(1 for o in onsets if otypes[o] == "T" and fires[o]),
            "T_n": sum(1 for o in onsets if otypes[o] == "T"),
            "V_hits": sum(1 for o in onsets if otypes[o] == "V" and fires[o]),
            "V_n": sum(1 for o in onsets if otypes[o] == "V")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro"); ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--workloads", default="tpch,job,pgbench")
    ap.add_argument("--configs", default="template_only,volume_only,mixed")
    args = ap.parse_args()
    R = Path(args.repro).resolve()
    sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "geometry_E0"))
    cb = load(R / "geometry_E0" / "cost_benefit.py", "cost_benefit")
    gate_mod = load(R / "end_to_end" / "dci_gate.py", "dci_gate_v1x")
    v2_mod = load(R / "dci_gate_v2.py", "dci_gate_v2x")
    v3_mod = load(R / "dci_gate_v3.py", "dci_gate_v3x")
    ov = json.load(open(R / "geometry_E0" / "out" / "20260705T135856Z" / "cost_benefit_run.json"))
    pax = {k: v * 1e3 for k, v in ov["overhead"]["per_dimension_overhead_s"].items()}
    full_ms = sum(pax.values())
    print("[cost] per-axis ms: " + " ".join(f"{k}={v:.4f}" for k, v in pax.items())
          + f"  full={full_ms:.3f}")
    pools = {}
    for wl in args.workloads.split(","):
        pools[wl] = (cb.tpch_pool() if wl == "tpch" else
                     cb.job_pool(R / "data" / "job" / "queries") if wl == "job" else
                     cb.pgbench_pool())
    POL = ["dci_v3", "dci_v3_a8", "dci_v3_r50",
           "dci_v2_P1", "dci_v2_P2", "dci_v2_P4", "dci_v2_P8", "dci_v1", "always_multiD",
           "best_axis_oracle", "union4_raw", "union4_bonf", "every_k2", "every_k3",
           "every_k4", "adwin", "adwin_s", "mcusum", "sketch1"]
    rows = []
    for wl, pool in pools.items():
        feats, s = [], 0
        while len(feats) < 64 and s < 200:
            seed = cb.stable_seed("cal", "template_only", s)
            w, didx = cb.build_trajectory(pool, "template_only", seed)
            f_, _ = cb.kernel_adjacent(w)
            feats.extend(list(f_[:max(0, min(didx) - 1)])); s += 1
        ctx = Ctx(np.asarray(feats[:64]), gate_mod, v2_mod, v3_mod)
        for cfg in args.configs.split(","):
            for sd in range(args.seeds):
                seed = cb.stable_seed(wl, cfg, sd)
                w, didx = cb.build_trajectory(pool, cfg, seed)
                fv, _ = cb.kernel_adjacent(w)
                fv = np.asarray(fv); n = len(fv)
                truth = np.array([1 if (i + 1) in didx else 0 for i in range(n)])
                onsets = sorted(np.where(truth == 1)[0])
                if cfg == "mixed":
                    otypes = {o: ("T" if i % 2 == 0 else "V") for i, o in enumerate(onsets)}
                else:
                    otypes = {o: ("T" if cfg.startswith("template") else "V") for o in onsets}
                for pol in POL:
                    out = replay(pol, ctx, fv, pax, sd)
                    if out[4] == "ORACLE_STUB":
                        # oracle: pick axis by AUC on THIS run (upper envelope)
                        pos = np.zeros(n, bool); pos[list(onsets)] = True
                        best = None
                        for j in range(5):
                            st = ((fv[:, j] - ctx.mu0[j]) / ctx.sigma0[j]) ** 2
                            sc = np.array([ctx.score(v, 1) for v in st])
                            a = auc_rank(sc, pos); a = -1 if a == "" else a
                            if best is None or a > best[0]:
                                best = (a, (st > ctx.thr[1]).astype(int), sc, j)
                        fires, scores = best[1], best[2]
                        feat, arith, note = n * pax[AXES[best[3]]], out[3], AXES[best[3]]
                    else:
                        fires, scores, feat, arith, note = out
                    sr = score_run(fires, scores, truth, onsets, otypes)
                    no = len(onsets)
                    rows.append({"workload": wl, "config": cfg, "seed": sd, "policy": pol,
                                 "n": n, "onsets": no,
                                 "recall": round(sr["hits"] / no, 4) if no else "",
                                 "recall_lag3": round(sr["lag_hits"] / no, 4) if no else "",
                                 "mean_delay": sr["mean_delay"],
                                 "T_hits": sr["T_hits"], "T_n": sr["T_n"],
                                 "V_hits": sr["V_hits"], "V_n": sr["V_n"],
                                 "fa": sr["fa"], "fa_lag3": sr["fa_lag"], "auc": sr["auc"],
                                 "feat_ms_per_win": round(feat / n, 5),
                                 "feat_pct": round(100 * (feat / n) / full_ms, 2),
                                 "arith_s": round(arith, 5), "note": note})
            print(f"[done] {wl}/{cfg}", flush=True)
    out = R / "s6_baseline_results.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    key = {}
    for r in rows:
        key.setdefault((r["config"], r["policy"]), []).append(r)
    print(f"\n{'config':15s}{'policy':17s}{'rec':>6s}{'lag3':>6s}{'T':>8s}{'V':>8s}"
          f"{'FA':>6s}{'FAl':>5s}{'AUC':>7s}{'dly':>5s}{'feat%':>7s}")
    summ = {}
    for (cfg, pol), rs in sorted(key.items()):
        g = lambda c: [x[c] for x in rs if x[c] != ""]
        rec, lag = np.mean(g("recall")), np.mean(g("recall_lag3"))
        T = f"{sum(x['T_hits'] for x in rs)}/{sum(x['T_n'] for x in rs)}"
        V = f"{sum(x['V_hits'] for x in rs)}/{sum(x['V_n'] for x in rs)}"
        fa, fal = np.mean(g("fa")), np.mean(g("fa_lag3"))
        auc = np.mean(g("auc")) if g("auc") else float("nan")
        dly = np.mean(g("mean_delay")) if g("mean_delay") else float("nan")
        fc = np.mean(g("feat_pct"))
        print(f"{cfg:15s}{pol:17s}{rec:6.2f}{lag:6.2f}{T:>8s}{V:>8s}{fa:6.1f}{fal:5.1f}"
              f"{auc:7.3f}{dly:5.1f}{fc:6.1f}%")
        summ[f"{cfg}/{pol}"] = {"recall": round(float(rec), 4),
                                "recall_lag3": round(float(lag), 4), "T": T, "V": V,
                                "fa": round(float(fa), 2), "fa_lag3": round(float(fal), 2),
                                "auc": (round(float(auc), 4) if np.isfinite(auc) else None),
                                "delay": (round(float(dly), 2) if np.isfinite(dly) else None),
                                "feat_pct": round(float(fc), 2), "n_runs": len(rs)}
    json.dump({"alpha": ALPHA, "lag": LAG, "per_axis_ms": pax, "cells": summ},
              open(R / "s6_baseline_summary.json", "w"), indent=1)
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
