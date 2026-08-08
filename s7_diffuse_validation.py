#!/usr/bin/env python3
"""s7_diffuse_validation.py -- why the escalation path exists (Paper 3C).

The S6 bench shows a Bonferroni union over the four cheap axes ties the full
5-D detector on all nine workload-grounded cells: their drift is axis-aligned.
Theorem 1's counterexample geometry -- drift spread evenly across axes, or
carried by a correlated direction with no dominant coordinate -- does not occur
in those cells, yet it is exactly where any max-statistic tier is provably
underpowered. This script constructs that geometry directly in feature space
and measures every tier on it.

Feature-space synthesis is the honest level for this question: it validates a
GEOMETRIC claim about detector power (union vs Mahalanobis vs routed), not a
claim about workload realism -- that is S6's and the live study's job.

Streams: 41 windows, steady N(0, I_5); transient mean shift delta*u at onset
windows {6,12,18,24,30,36} (1-based; same phase structure as the regime map).
Calibration: 64 fresh steady windows per seed (finite-sample thresholds).

Geometries u:
  axis_sv   e_SV                  control: axis-aligned; union should tie multi-D
  diffuse4  (1,1,1,1,0)/2         Theorem-1 counterexample in the cheap subspace
                                  (per-axis share 0.25; per-axis component delta/2)
  diffuse5  (1,1,1,1,1)/sqrt(5)   diffuse including S_P
  corr_min  (1,-1,1,-1,0)/2       contrast direction under equicorrelated cheap
            with r=0.8            noise (r=0.8): per-axis deviation delta/2 sits in
                                  noise, but the direction has variance 1-r=0.2, so
                                  the whitened norm is delta/sqrt(0.2) = 2.24 delta.
                                  Raw-axis max tests are structurally blind; full
                                  covariance resolution sees it easily. This is the
                                  correlated-drift geometry of the paper's story.
  sp_only   e_SP                  cheap-blind drift: only the audit layer or the
                                  full tier can see it (quantifies the audit floor)

Delta sweep (total shift, sigma units): 2.5, 3.5, 4.5, 6.0.
  m=64 thresholds: union arm needs a per-axis |z| ~ 2.6 (alpha/4); the 5-D arm
  needs total ||z|| ~ 3.6. diffuse4 at delta in [3.5, 5.2] is the predicted
  union-blind / multi-D-visible band.

Policies: always_multiD, union4_bonf, best_axis_oracle, sketch1,
  dci_v3 (rho=0.35), dci_v3_a8 (audit_every=8, random per-seed offset).
Reported: strict recall, FA/run, AUC (rank, -log10 p scores), esc% (fraction of
windows the gate ran the full tier), feat% (modeled per-axis cost, locked
per_dimension_overhead_s).

Usage: python3 s7_diffuse_validation.py <repro> [--seeds N]
Writes: s7_diffuse_results.csv + s7_diffuse_summary.json
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import f as fdist, rankdata

AXES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
CHEAP = [0, 1, 2, 3]
ALPHA = 0.05
N_WIN, M_CAL = 41, 64
ONSETS0 = [5, 11, 17, 23, 29, 35]          # 0-based
GEOMS = {  # name: (drift direction u, equicorrelation r of the cheap noise block)
    "axis_sv":  (np.array([0, 1, 0, 0, 0], float), 0.0),
    "diffuse4": (np.array([1, 1, 1, 1, 0], float) / 2.0, 0.0),
    "diffuse5": (np.ones(5) / np.sqrt(5.0), 0.0),
    "corr_min": (np.array([1, -1, 1, -1, 0], float) / 2.0, 0.8),
    "sp_only":  (np.array([0, 0, 0, 0, 1], float), 0.0),
}
DELTAS = [2.5, 3.5, 4.5, 6.0]
POL = ["always_multiD", "union4_bonf", "best_axis_oracle", "sketch1",
       "dci_v3", "dci_v3_a8"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def auc_rank(scores, pos):
    s = np.asarray(scores, float)
    if pos.sum() == 0 or (~pos).sum() == 0 or not np.all(np.isfinite(s)):
        return ""
    r = rankdata(s)
    return round(float((r[pos].mean() - (pos.sum() + 1) / 2) / (~pos).sum()), 4)


class Cal:
    def __init__(self, X):
        self.m = m = X.shape[0]
        self.mu0 = X.mean(0)
        cov = np.cov(X, rowvar=False, ddof=1) + 1e-6 * np.eye(5)
        self.sigma0 = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        ev, V = np.linalg.eigh(cov)
        self.W = V @ np.diag(np.clip(ev, 1e-12, None) ** -0.5) @ V.T
        self.c = {k: (m * (m - k)) / ((m + 1) * k * (m - 1)) for k in (1, 5)}
        self.thr1b = float(fdist.ppf(1 - ALPHA / 4, 1, m - 1)) / self.c[1]
        self.thr1 = float(fdist.ppf(1 - ALPHA, 1, m - 1)) / self.c[1]
        self.thr5 = float(fdist.ppf(1 - ALPHA, 5, m - 5)) / self.c[5]
        u = np.random.default_rng(20260808).standard_normal(5)
        self.sketch_u = u / np.linalg.norm(u)

    def score(self, stat, k):
        return -np.log10(max(float(fdist.sf(stat * self.c[k], k, self.m - k)), 1e-300))


def replay(pol, cal, F, v3_mod, pax, full_ms, seed, pos):
    n = len(F); fires = np.zeros(n, int); sc = np.full(n, np.nan)
    feat = 0.0; esc = 0; note = ""
    if pol == "always_multiD":
        for t in range(n):
            feat += full_ms
            z = cal.W @ (F[t] - cal.mu0); st = float(z @ z)
            fires[t] = int(st > cal.thr5); sc[t] = cal.score(st, 5)
        esc = n
    elif pol == "union4_bonf":
        cheap_ms = sum(pax[AXES[j]] for j in CHEAP)
        for t in range(n):
            feat += cheap_ms
            d = F[t] - cal.mu0
            st = max((d[j] / cal.sigma0[j]) ** 2 for j in CHEAP)
            fires[t] = int(st > cal.thr1b); sc[t] = cal.score(st, 1)
    elif pol == "best_axis_oracle":
        best = None
        for j in range(5):
            st = ((F[:, j] - cal.mu0[j]) / cal.sigma0[j]) ** 2
            s_ = np.array([cal.score(v, 1) for v in st])
            a = auc_rank(s_, pos); a = -1 if a == "" else a
            if best is None or a > best[0]:
                best = (a, (st > cal.thr1).astype(int), s_, j)
        fires, sc = best[1], best[2]
        feat = n * pax[AXES[best[3]]]; note = AXES[best[3]]
    elif pol == "sketch1":
        for t in range(n):
            feat += full_ms
            z = cal.W @ (F[t] - cal.mu0); st = float(cal.sketch_u @ z) ** 2
            fires[t] = int(st > cal.thr1); sc[t] = cal.score(st, 1)
    elif pol.startswith("dci_v3"):
        kw = {}
        if pol == "dci_v3_a8":
            kw["audit_every"] = 8
            kw["audit_offset"] = int(np.random.default_rng([seed, 8]).integers(0, 8))
        g = v3_mod.DCIGateV3(alpha=ALPHA, **kw)
        g.fit(np.random.default_rng([seed, 777]).standard_normal((M_CAL, 5)) * cal.sigma0 + cal.mu0) \
            if False else g.fit(cal.calX)
        for t in range(n):
            fires[t] = g.decide(F[t])
            feat += sum(pax[a] for a in g.last["features_used"])
            esc += int(g.last["k"] == 5)
            sc[t] = cal.score(float(g.last["statistic"]), int(g.last["k"]))
    else:
        raise ValueError(pol)
    return fires, sc, feat, esc, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro"); ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    R = Path(args.repro).resolve()
    v3_mod = load(R / "dci_gate_v3.py", "dci_gate_v3s7")
    ov = json.load(open(R / "geometry_E0" / "out" / "20260705T135856Z" / "cost_benefit_run.json"))
    pax = {k: v * 1e3 for k, v in ov["overhead"]["per_dimension_overhead_s"].items()}
    full_ms = sum(pax.values())
    rows = []
    for geom, (u, r) in GEOMS.items():
        S = np.eye(5)
        for a in range(4):
            for b in range(4):
                if a != b:
                    S[a, b] = r
        L = np.linalg.cholesky(S)
        for delta in DELTAS:
            for sd in range(args.seeds):
                rng = np.random.default_rng([20260808, abs(hash(geom)) % (2**31), int(delta * 10), sd])
                calX = rng.standard_normal((M_CAL, 5)) @ L.T
                cal = Cal(calX); cal.calX = calX
                F = rng.standard_normal((N_WIN, 5)) @ L.T
                for o in ONSETS0:
                    F[o] += delta * u
                pos = np.zeros(N_WIN, bool); pos[ONSETS0] = True
                for pol in POL:
                    fires, sc, feat, esc, note = replay(pol, cal, F, v3_mod, pax,
                                                        full_ms, sd, pos)
                    hits = int(fires[ONSETS0].sum())
                    fa = int(sum(fires[t] for t in range(N_WIN) if not pos[t]))
                    rows.append({"geom": geom, "delta": delta, "seed": sd, "policy": pol,
                                 "recall": round(hits / len(ONSETS0), 4), "fa": fa,
                                 "auc": auc_rank(sc, pos),
                                 "esc_pct": round(100 * esc / N_WIN, 1),
                                 "feat_pct": round(100 * (feat / N_WIN) / full_ms, 2),
                                 "note": note})
            print(f"[done] {geom}/delta={delta}", flush=True)
    out = R / "s7_diffuse_results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    key = {}
    for r in rows:
        key.setdefault((r["geom"], r["delta"], r["policy"]), []).append(r)
    print(f"\n{'geom':10s}{'delta':>6s} {'policy':17s}{'rec':>6s}{'FA':>6s}{'AUC':>7s}"
          f"{'esc%':>7s}{'feat%':>7s}")
    summ = {}
    for (geom, delta, pol), rs in sorted(key.items()):
        g = lambda c: [x[c] for x in rs if x[c] != ""]
        rec, fa = np.mean(g("recall")), np.mean(g("fa"))
        auc = np.mean(g("auc")) if g("auc") else float("nan")
        ep, fp = np.mean(g("esc_pct")), np.mean(g("feat_pct"))
        print(f"{geom:10s}{delta:6.1f} {pol:17s}{rec:6.2f}{fa:6.1f}{auc:7.3f}{ep:7.1f}{fp:6.1f}%")
        summ[f"{geom}/d{delta}/{pol}"] = {"recall": round(float(rec), 4),
                                          "fa": round(float(fa), 2),
                                          "auc": (round(float(auc), 4) if np.isfinite(auc) else None),
                                          "esc_pct": round(float(ep), 1),
                                          "feat_pct": round(float(fp), 2), "n": len(rs)}
    json.dump({"alpha": ALPHA,
               "geometries": {k: {"u": [round(float(x), 6) for x in u],
                                  "equicorr_r": float(r)}
                              for k, (u, r) in GEOMS.items()},
               "deltas": DELTAS, "cells": summ},
              open(R / "s7_diffuse_summary.json", "w"), indent=1)
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
