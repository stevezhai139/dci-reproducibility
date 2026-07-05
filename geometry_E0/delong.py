#!/usr/bin/env python3
"""
delong.py
=========

Paper 3C -- DeLong's nonparametric (co)variance of ROC AUC, the
per-dimension sufficiency tolerance epsilon, and the DerSimonian-Laird
random-effects combination across seeds.

WHY THIS EXISTS
    The sufficiency definition -- "the cheap 1-D detector is enough" --
    is  AUC_1d >= AUC_5d - epsilon.  In Papers 3A/3B epsilon was a flat
    hand-set constant.  Paper 3C derives epsilon from the data:

      * epsilon is PER DIMENSION.  Each kernel axis (S_R,S_V,S_T,S_A,S_P)
        has its own noise floor; collapsing them into one number would
        hide exactly the structure the paper must explain (S_P is inert,
        S_T/S_R are responsive, ...).  So epsilon_d is computed and kept
        separate for every dimension d.

      * epsilon_d is the 95% noise floor of the AUC GAP between the 5-D
        Mahalanobis detector and dimension d:
            epsilon_d = z * sqrt( Var(AUC_5d - AUC_d) ).
        AUC_5d and AUC_d are evaluated on the SAME windows, so the two
        AUC estimates are correlated -- DeLong (1988) estimates the full
        covariance of all six AUCs (five axes + the 5-D detector) at
        once, with no distributional assumption.

      * across the n_s seeds of a cell, the per-seed gaps are pooled.
        combine_across_seed -- the equal-weight t-interval, with the
        cell variance FLOORED at the mean within-seed DeLong variance
        -- is the cell estimator: verify_epsilon.py shows it is
        calibrated (~95% coverage).  The floor stops epsilon_cell from
        collapsing to 0 on a deterministic axis (zero between-seed
        spread) while the within-seed DeLong variance is still > 0.
        Inverse-variance / DerSimonian-Laird pooling under-covers
        badly -- the DeLong variance of an AUC gap depends on the AUC
        level, so the weights correlate with the effect and the pooled
        estimate is biased; combine_random_effects (DerSimonian-Laird
        1986) is retained only for that documented comparison.

    epsilon is a label-dependent EVALUATION quantity only.  It never
    feeds back into the detector -- the detector's per-dimension
    balancing is done, unsupervised, by the Mahalanobis Sigma^-1.

REFERENCES
    E. R. DeLong, D. M. DeLong, D. L. Clarke-Pearson (1988). "Comparing
    the Areas under Two or More Correlated Receiver Operating
    Characteristic Curves: A Nonparametric Approach." Biometrics 44(3):
    837-845.  DOI: 10.2307/2531595.
    R. DerSimonian, N. Laird (1986). "Meta-analysis in clinical
    trials." Controlled Clinical Trials 7(3): 177-188.
    DOI: 10.1016/0197-2456(86)90046-2.
    W. Hoeffding (1948). "A Class of Statistics with Asymptotically
    Normal Distribution." Ann. Math. Statist. 19(3): 293-325.
    DOI: 10.1214/aoms/1177730196.

API
    delong_auc_cov(labels, score_matrix)       -> (aucs, cov)
    delong_epsilon(labels, score_a, score_b)   -> dict   (pairwise)
    delong_gap_variances(labels, ref, others)  -> dict   (5-D vs each axis)
    combine_across_seed(deltas, variances)     -> dict   (cell estimator)
    epsilon_from_sigma2(sigma2, n)             -> float  (epsilon at fixed n;
                                                          the practical floor)
    combine_random_effects(deltas, variances)  -> dict   (DerSimonian-Laird;
                                                          comparison only)
"""
from __future__ import annotations

import numpy as np

Z_975 = 1.959963984540054          # the 0.975 quantile of N(0, 1)


# --------------------------------------------------------------------------
# DeLong covariance of a set of correlated AUCs
# --------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of x (tied values share their average rank), 1-based."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1.0
    return T2


def delong_auc_cov(labels, score_matrix):
    """AUC vector and DeLong covariance for k scores against one label.

    labels       : 1-D array of {0, 1}; 1 = positive (a drift window).
    score_matrix : shape (n_samples, k); each column a score, higher = positive.
    returns (aucs[k], cov[k, k]).  cov is the DeLong covariance of the
            AUC estimates (fast midrank algorithm).  If a class has < 2
            members the result is all-NaN.
    """
    labels = np.asarray(labels).astype(int)
    S = np.asarray(score_matrix, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    pos = labels == 1
    neg = labels == 0
    m, n = int(pos.sum()), int(neg.sum())
    k = S.shape[1]
    if m < 2 or n < 2:
        return np.full(k, np.nan), np.full((k, k), np.nan)
    ordered = np.vstack([S[pos], S[neg]]).T          # (k, m + n)
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(ordered[r, :m])
        ty[r] = _midrank(ordered[r, m:])
        tz[r] = _midrank(ordered[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n                       # (k, m)
    v10 = 1.0 - (tz[:, m:] - ty) / m                 # (k, n)
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


# --------------------------------------------------------------------------
# epsilon: the 95% noise floor of an AUC gap
# --------------------------------------------------------------------------
def delong_epsilon(labels, score_a, score_b, z: float = Z_975) -> dict:
    """Pairwise sufficiency tolerance for the AUC difference (b - a).

    score_a : the cheap detector's score; score_b : the reference's.
    epsilon = z * SE(AUC_b - AUC_a) via DeLong's covariance.
    `a_sufficient` : the b-over-a AUC gain does NOT exceed epsilon.
    """
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    mask = np.isfinite(score_a) & np.isfinite(score_b)
    lab = np.asarray(labels)[mask]
    aucs, cov = delong_auc_cov(
        lab, np.column_stack([score_a[mask], score_b[mask]]))
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se = float(np.sqrt(var_diff)) if np.isfinite(var_diff) and var_diff > 0 \
        else (0.0 if np.isfinite(var_diff) else float("nan"))
    eps = z * se
    diff = auc_b - auc_a
    return {"auc_a": auc_a, "auc_b": auc_b, "auc_diff": diff,
            "se_diff": se, "epsilon": eps,
            "b_significant": bool(np.isfinite(eps) and diff > eps),
            "a_sufficient": bool(np.isfinite(eps) and diff <= eps)}


def delong_gap_variances(labels, score_ref, score_others,
                         z: float = Z_975) -> dict:
    """5-D reference detector vs each of several per-dimension detectors.

    score_ref    : the 5-D detector's score (1-D array).
    score_others : shape (n_samples, k) -- k per-dimension scores.
    Computes one DeLong covariance over all k+1 detectors, then for each
    dimension d returns auc_d, the gap (auc_ref - auc_d), its variance
    var_d, and epsilon_d = z * sqrt(var_d) -- the per-dimension 95%
    noise floor.  One covariance, so correlations between dimensions and
    the 5-D detector are all accounted for.
    """
    ref = np.asarray(score_ref, dtype=float)
    others = np.asarray(score_others, dtype=float)
    if others.ndim == 1:
        others = others[:, None]
    mat = np.column_stack([others, ref])             # others first, ref last
    mask = np.all(np.isfinite(mat), axis=1)
    aucs, cov = delong_auc_cov(np.asarray(labels)[mask], mat[mask])
    k = others.shape[1]
    ri = k                                           # ref index (last)
    per_dim = []
    for j in range(k):
        gap = float(aucs[ri] - aucs[j])
        var = float(cov[ri, ri] + cov[j, j] - 2.0 * cov[j, ri])
        var = var if (np.isfinite(var) and var > 0) else \
            (0.0 if np.isfinite(var) else float("nan"))
        eps = z * np.sqrt(var) if np.isfinite(var) else float("nan")
        per_dim.append({"auc": float(aucs[j]), "gap": gap,
                        "var_gap": var, "epsilon": float(eps),
                        "sufficient": bool(np.isfinite(eps) and gap <= eps)})
    return {"auc_ref": float(aucs[ri]), "per_dim": per_dim}


# --------------------------------------------------------------------------
# Cell-level combination across seeds
# --------------------------------------------------------------------------
def combine_across_seed(deltas, variances=None, conf: float = 0.95) -> dict:
    """Pool n per-seed AUC gaps into a cell tolerance epsilon_cell.

    delta_combined = mean of the per-seed gaps.  The cell variance is
    the equal-weight across-seed sample variance, FLOORED at the mean
    within-seed DeLong variance:

        sigma^2     = max( var(deltas, ddof=1), mean(variances) )
        epsilon_cell = t_{conf, n-1} * sqrt( sigma^2 / n )

    Why the floor.  The across-seed sample variance estimates
    tau^2 + v_bar (between-seed + mean within-seed) and so is >= v_bar
    in expectation -- but in a finite sample it can collapse to 0 when
    the gap is identical on every seed (a deterministic axis, e.g. S_V
    on volume drift).  The irreducible within-seed DeLong variance
    v_bar is still there, so epsilon_cell must not fall below
    t*sqrt(v_bar/n).  With the floor, epsilon_cell is exactly 0 ONLY
    when the gap is genuinely deterministic at the window level
    (between- AND within-seed variance both 0) -- which is the honest
    answer in that case.  `variances` are the per-seed DeLong gap
    variances; pass them whenever available.

    Why equal weights (not inverse-variance).  The DeLong variance of
    an AUC gap depends on the AUC level, so inverse-variance pooling
    (combine_random_effects) correlates the weights with the effect
    and is biased -- verify_epsilon.py shows the under-coverage.
    """
    d = np.asarray(deltas, dtype=float)
    if variances is not None:
        v = np.asarray(variances, dtype=float)
        keep = np.isfinite(d) & np.isfinite(v)
        d, v = d[keep], v[keep]
    else:
        d = d[np.isfinite(d)]
        v = None
    n = int(len(d))
    if n == 0:
        return {"delta_combined": float("nan"), "sd": float("nan"),
                "se": float("nan"), "epsilon_cell": float("nan"),
                "var_floor": float("nan"), "sigma2": float("nan"),
                "n": 0, "df": 0, "sufficient": False}
    mean = float(d.mean())
    if n == 1:
        return {"delta_combined": mean, "sd": float("nan"),
                "se": float("nan"), "epsilon_cell": float("nan"),
                "var_floor": float("nan"), "sigma2": float("nan"),
                "n": 1, "df": 0, "sufficient": False}
    s2_emp = float(d.var(ddof=1))
    vbar = float(np.mean(v)) if (v is not None and len(v)) else 0.0
    sigma2 = max(s2_emp, vbar)
    se = float(np.sqrt(sigma2 / n))
    try:
        from scipy.stats import t as _t
        tcrit = float(_t.ppf(0.5 + conf / 2.0, n - 1))
    except Exception:                                      # noqa: BLE001
        tcrit = Z_975
    eps = float(tcrit * se)
    return {"delta_combined": mean, "sd": float(np.sqrt(s2_emp)),
            "se": se, "epsilon_cell": eps, "var_floor": vbar,
            "sigma2": sigma2, "n": n, "df": n - 1,
            "sufficient": bool(mean <= eps)}


def epsilon_from_sigma2(sigma2: float, n: int, conf: float = 0.95) -> float:
    """The across-seed tolerance epsilon for a floored variance sigma^2
    evaluated at n seeds: t_{conf, n-1} * sqrt(sigma2 / n).

    This is the building block of the n-stable practical floor used by
    Paper 3C: delta_d = epsilon_from_sigma2(sigma2_d, N_REF), with N_REF
    the minimum reliable seed count from seed_analysis.py.  Because
    N_REF is fixed, delta_d does not shrink with the run's seed count,
    so the regime-map verdict (gap_d <= delta_d) is n-stable.
    """
    if n < 2 or not np.isfinite(sigma2) or sigma2 <= 0:
        return 0.0
    try:
        from scipy.stats import t as _t
        tcrit = float(_t.ppf(0.5 + conf / 2.0, n - 1))
    except Exception:                                      # noqa: BLE001
        tcrit = Z_975
    return float(tcrit * np.sqrt(sigma2 / n))


# --------------------------------------------------------------------------
# DerSimonian-Laird random-effects combination -- comparison only
# --------------------------------------------------------------------------
def combine_random_effects(deltas, variances, z: float = Z_975) -> dict:
    """Pool n independent per-seed AUC-gap estimates (DerSimonian-Laird).

    deltas[s]    : seed s's AUC gap (auc_5d - auc_d).
    variances[s] : seed s's DeLong variance of that gap.

    Returns the random-effects pooled gap, its variance, the cell
    epsilon, the between-seed variance tau^2, and Cochran's Q.
    A cell is `sufficient` (1-D enough) iff delta_combined <= epsilon.
    With tau^2 = 0 this reduces to fixed-effect inverse-variance
    pooling; as the DeLong variances dominate it tracks the across-seed
    spread -- it subsumes both limiting cases.
    """
    d = np.asarray(deltas, dtype=float)
    v = np.asarray(variances, dtype=float)
    keep = np.isfinite(d) & np.isfinite(v) & (v > 0)
    d, v = d[keep], v[keep]
    n = int(len(d))
    if n == 0:
        return {"delta_combined": float("nan"), "var_combined": float("nan"),
                "epsilon_cell": float("nan"), "tau2": float("nan"),
                "Q": float("nan"), "n": 0, "df": 0, "sufficient": False}
    if n == 1:
        var_re = float(v[0])
        eps = z * np.sqrt(var_re)
        return {"delta_combined": float(d[0]), "var_combined": var_re,
                "epsilon_cell": float(eps), "tau2": 0.0, "Q": 0.0,
                "n": 1, "df": 0, "sufficient": bool(d[0] <= eps)}
    w = 1.0 / v
    delta_fe = float(np.sum(w * d) / np.sum(w))
    Q = float(np.sum(w * (d - delta_fe) ** 2))       # Cochran's Q
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (n - 1)) / C) if C > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    delta_re = float(np.sum(w_re * d) / np.sum(w_re))
    var_re = float(1.0 / np.sum(w_re))
    eps = float(z * np.sqrt(var_re))
    return {"delta_combined": delta_re, "var_combined": var_re,
            "epsilon_cell": eps, "tau2": float(tau2), "Q": Q,
            "n": n, "df": n - 1, "sufficient": bool(delta_re <= eps)}


# --------------------------------------------------------------------------
if __name__ == "__main__":                           # self-test
    rng = np.random.default_rng(0)

    def _plain_auc(lab, sc):                          # independent reference
        lab = np.asarray(lab, float); sc = np.asarray(sc, float)
        p, n = sc[lab == 1], sc[lab == 0]
        return float(np.mean([(x > y) + 0.5 * (x == y)
                              for x in p for y in n]))

    ok = True
    for _ in range(200):
        lab = rng.integers(0, 2, size=60)
        if lab.sum() < 2 or (lab == 0).sum() < 2:
            continue
        sc = rng.normal(size=60) + lab * rng.uniform(0, 1.5)
        aucs, _ = delong_auc_cov(lab, sc[:, None])
        if abs(aucs[0] - _plain_auc(lab, sc)) > 1e-9:
            ok = False
    print("[1] AUC matches the plain Mann-Whitney reference :", ok)

    # delong_gap_variances on one 'other' must match delong_epsilon
    lab = (rng.random(80) < 0.4).astype(int)
    sa = rng.normal(size=80) + 0.4 * lab
    sb = rng.normal(size=80) + 2.0 * lab
    g = delong_gap_variances(lab, sb, sa[:, None])
    e = delong_epsilon(lab, sa, sb)
    same = abs(g["per_dim"][0]["epsilon"] - e["epsilon"]) < 1e-9
    print("[2] gap_variances == pairwise epsilon            :", same,
          f"(eps={e['epsilon']:.4f})")

    # combine_random_effects: homogeneous seeds -> tau2 = 0
    v = np.full(20, 0.01)
    d = np.full(20, 0.05)                             # identical -> Q ~ 0
    c = combine_random_effects(d, v)
    print(f"[3] homogeneous seeds  -> tau2={c['tau2']:.2e} "
          f"(expect ~0), eps_cell={c['epsilon_cell']:.4f}")

    # heterogeneous seeds -> tau2 > 0 (between-seed SD 0.25 >> sqrt(v)=0.1)
    d2 = rng.normal(0.05, 0.25, size=20)
    c2 = combine_random_effects(d2, v)
    print(f"[4] heterogeneous seeds-> tau2={c2['tau2']:.2e} "
          f"(expect > 0), Q={c2['Q']:.1f} >> df={c2['df']}")

    # combine_across_seed: the within-seed DeLong-variance floor
    d_const = np.full(20, 0.25)            # identical gaps -> s2_emp = 0
    v_const = np.full(20, 0.01)            # but within-seed variance > 0
    no_floor = combine_across_seed(d_const)
    floored = combine_across_seed(d_const, v_const)
    print(f"[5] deterministic gap  -> eps without floor "
          f"{no_floor['epsilon_cell']:.4f} (expect 0), with DeLong floor "
          f"{floored['epsilon_cell']:.4f} (expect > 0)")
