#!/usr/bin/env python3
"""
Paper 3C -- HONEST test of the hypothesised DCI blind spot.

Hypothesis: a rank-one drift (DCI~1, routes to 1-D) whose single mode is OFF-AXIS
(a rotated combination, not aligned to any similarity axis) could fool the router:
DCI says "1-D suffices" but the best single-axis detector fails while the 5-D
Mahalanobis detector succeeds.

We TEST it (not assume it). Model, matched to the paper's detectors:
  steady window deviation d ~ N(0, Sigma)          (Sigma = steady-window noise covariance)
  drift  window deviation d = m*v + N(0, Sigma)    (rank-one drift along unit direction v)
  DCI    = tr(C)^2/||C||_F^2 on the drift-window second moment C (as in cost_benefit.py)
  1-D    = best single raw axis  (AUC over j of +/- d_j)
  5-D    = Mahalanobis d^T Sigma^-1 d  (the paper's optimal linear fusion)

Sweep v from axis-aligned (e_1) to maximally spread (uniform), under iid and
correlated noise, at a drift magnitude that keeps DCI in the 1-D regime (<1.5).
If auc_1d collapses while auc_5d holds AT LOW DCI -> blind spot real.
"""
import numpy as np

D = 5
NSTEADY, NDRIFT, NSEED = 400, 200, 40

def auc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def sigma_corr(rho):
    return (1 - rho) * np.eye(D) + rho * np.ones((D, D))

def vdir(t):                      # t=0 -> e_1 (aligned); t=1 -> uniform (spread)
    w = np.array([1.0, t, t, t, t]); return w / np.linalg.norm(w)

def run(Sigma, v, m, seed):
    r = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma)
    steady = r.standard_normal((NSTEADY, D)) @ L.T
    drift = m * v + r.standard_normal((NDRIFT, D)) @ L.T
    mu = steady.mean(axis=0)
    d = np.vstack([steady, drift]) - mu
    lab = np.r_[np.zeros(NSTEADY), np.ones(NDRIFT)]
    dd = d[lab == 1]
    C = dd.T @ dd / len(dd)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None); tot = ev.sum()
    dci = float(tot ** 2 / np.sum(ev ** 2))
    a1 = max(max(auc(lab, d[:, j]), auc(lab, -d[:, j])) for j in range(D))
    Sig = np.cov(d[lab == 0].T) + 1e-6 * np.eye(D)
    Pinv = np.linalg.pinv(Sig)
    maha = np.einsum("ij,jk,ik->i", d, Pinv, d)
    a5 = auc(lab, maha)
    return dci, a1, a5

def sweep(rho, m):
    Sigma = sigma_corr(rho)
    print(f"\n=== noise rho={rho}  drift m={m}  (Sigma off-diag correlation = {rho}) ===")
    print(f"{'t (spread)':>11} {'maxproj':>8} {'DCI':>6} {'AUC_1D':>7} {'AUC_5D':>7} {'gap':>6}")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = vdir(t)
        res = np.array([run(Sigma, v, m, s) for s in range(NSEED)])
        dci, a1, a5 = res.mean(axis=0)
        print(f"{t:>11.2f} {v.max():>8.3f} {dci:>6.2f} {a1:>7.3f} {a5:>7.3f} {a5-a1:>6.3f}")

# choose m so DCI lands in the 1-D regime (<1.5) -- report actual DCI
for rho in [0.0, 0.8]:
    for m in [6.0]:
        sweep(rho, m)

# control: does the 1-D/5-D gap live at HIGH DCI (paper's mixed regime)?  magnitude sweep, spread dir
print("\n=== CONTROL: spread direction (t=1), vary magnitude -> where does DCI / gap live? ===")
print(f"{'m':>5} {'DCI':>6} {'AUC_1D':>7} {'AUC_5D':>7} {'gap':>6}   (iid noise)")
for m in [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]:
    v = vdir(1.0)
    res = np.array([run(np.eye(D), v, m, s) for s in range(NSEED)])
    dci, a1, a5 = res.mean(axis=0)
    print(f"{m:>5.1f} {dci:>6.2f} {a1:>7.3f} {a5:>7.3f} {a5-a1:>6.3f}")
