#!/usr/bin/env python3
"""dci_gate_v3.py -- cost-structure-aligned two-tier gate (Paper 3C, post S6 bench).

Why v3. The locked per-axis extraction costs put 94% of the kernel's cost on a
single axis (S_P, the positional kernel); the other four axes together cost ~5%.
The S6 baseline bench showed a Bonferroni union over those four cheap axes ties
the full 5-D detector on binary firing everywhere and strictly dominates the
v2 single-axis gate at every probe cadence. The only economically meaningful
question is therefore WHEN TO PAY FOR S_P + full covariance resolution.

Design.
  CHEAP TIER (every window, ~5% cost): compute the four cheap axes
    (S_R, S_V, S_T, S_A); fire if max_j (d_j/sigma_j)^2 exceeds the
    finite-sample Hotelling-F(1) threshold at alpha/4 (Bonferroni union).
  ROUTER (every window, closed form, no eig): maintain the running 4x4
    outer-product sum M4 of standardised cheap deviations. The union is a
    max-statistic: it is powerful iff SOME axis carries a dominant share of
    DRIFT energy. Noise must not be mistaken for drift: under steady state the
    standardised deviations contribute ~1 per axis per window to diag(M4), and
    perfectly isotropic noise would read as "diffuse" if shares were taken on
    M4 directly. The router therefore separates PRESENCE from ALIGNMENT:
      presence  -- measured in the WHITENED cheap subspace, where any drift
        direction (including low-variance directions of a correlated Sigma0,
        invisible to per-axis tests) contributes its full energy:
            tr_Ew = sum_t ||W4 (f_t - mu0)_cheap||^2 - 4 n
        (W4 = Sigma0[cheap,cheap]^{-1/2}, precomputed in fit(); the running
        quantity is a scalar -- still no eig and no matrix work per window).
        Delta-rule gate: no signal while tr_Ew <= k_sig * sqrt(8 n) -> CHEAP.
      alignment -- fraction of that energy the best RAW axis captures
        (the union is a max test over raw axes; this is the R_axis quantity):
            E_j = max(0, M4_jj - n),   R = max_j E_j / tr_Ew
        R >= rho -> CHEAP (a dominant raw axis carries the drift; the union
                    tests exactly that axis)
        R <  rho -> FULL  (diffuse, correlated, or low-variance-direction
                    drift; a raw-axis max test is underpowered -- the
                    Theorem-1 counterexample geometry, e.g. (1,1,1,1)/2 under
                    isotropic noise, or any contrast direction of a
                    correlated Sigma0)
    We deliberately do NOT escalate on DCI4 >= tau: drift spread over two
    STRONG axes has DCI4 ~ 2 but the union handles it perfectly. DCI4 remains
    reported as the complexity diagnostic; the regime map is unchanged.
    Known limitation (documented): pure covariance-rotation drift with
    unchanged per-axis variances is invisible to the excess diagonal; the
    audit layer bounds exposure.
  FULL TIER (escalated windows, 100% cost): extract S_P lazily, test the 5-D
    whitened Mahalanobis distance against Hotelling-F(5) at alpha.
  AUDIT (optional, audit_every=K): every K-th window is forced full. This
    bounds exposure to drift with ZERO cheap-subspace signature (visible only
    to S_P), which is otherwise invisible to the router. Cost floor ~5% + 95%/K.

Diagnostics in .last: R4, dci4 (cheap-subspace complexity), dci5 (from full
windows seen so far), regime, axis, features_used (for cost accounting).

API mirrors v1/v2: fit(steady) -> decide(f) -> 0/1, reset_trajectory() per block.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import f as _f_dist

RHO_DEFAULT = 0.35
RIDGE = 1e-6
N_AXES = 5
AXES = ["S_R", "S_V", "S_T", "S_A", "S_P"]
CHEAP_IDX = [0, 1, 2, 3]
CHEAP_AXES = [AXES[j] for j in CHEAP_IDX]


def _hotelling_c(m: int, k: int) -> float:
    return (m * (m - k)) / ((m + 1) * k * (m - 1))


def _pr_closed_form(M: np.ndarray, n: int) -> float:
    if n <= 0:
        return float("nan")
    C = M / n
    tr = float(np.trace(C)); f2 = float(np.sum(C * C))
    if tr <= 0.0 or f2 <= 0.0:
        return 1.0
    return (tr * tr) / f2


class DCIGateV3:
    def __init__(self, rho: float = RHO_DEFAULT, alpha: float = 0.05,
                 min_windows: int = 3, audit_every: int | None = None,
                 audit_offset: int = 0, k_sig: float = 3.0,
                 force: str | None = None, ridge: float = RIDGE):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha in (0,1)")
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho in (0,1]")
        if force not in (None, 'full', 'cheap'):
            raise ValueError("force in {None,'full','cheap'}")
        if audit_every is not None and audit_every < 2:
            raise ValueError("audit_every >= 2 or None")
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.min_windows = int(min_windows)
        self.audit_every = audit_every
        self.audit_offset = int(audit_offset)
        self.k_sig = float(k_sig)
        self.force = force   # benchmark arms: pin the route (router still logged)
        self.ridge = float(ridge)
        self._fitted = False
        self.last: dict | None = None

    def fit(self, steady_features) -> "DCIGateV3":
        X = np.asarray(steady_features, dtype=float)
        if X.ndim != 2 or X.shape[1] != N_AXES:
            raise ValueError(f"steady_features must be (m,{N_AXES})")
        m = X.shape[0]
        if m <= N_AXES + 1:
            raise ValueError(f"need m > {N_AXES + 1}")
        self.m = m
        self.mu0 = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1) + self.ridge * np.eye(N_AXES)
        self.Sigma0 = cov
        self.sigma0 = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        evals, evecs = np.linalg.eigh(cov)              # fit()-time only
        evals = np.clip(evals, 1e-12, None)
        self._whiten = evecs @ np.diag(evals ** -0.5) @ evecs.T
        cov4 = cov[np.ix_(CHEAP_IDX, CHEAP_IDX)]
        e4, V4 = np.linalg.eigh(cov4)                   # fit()-time only
        self._W4 = V4 @ np.diag(np.clip(e4, 1e-12, None) ** -0.5) @ V4.T
        # cheap arm: Bonferroni union over 4 axes at alpha/4, F(1, m-1)
        self.thr1_bonf = float(_f_dist.ppf(1.0 - self.alpha / 4.0, 1, m - 1)) \
            / _hotelling_c(m, 1)
        # full arm: 5-D Hotelling at alpha
        self.thr5 = float(_f_dist.ppf(1.0 - self.alpha, N_AXES, m - N_AXES)) \
            / _hotelling_c(m, N_AXES)
        self.reset_trajectory()
        self._fitted = True
        return self

    def reset_trajectory(self) -> None:
        self._M4 = np.zeros((4, 4))     # standardised cheap deviations
        self._sw = 0.0                  # sum ||W4 d_cheap||^2 (whitened energy)
        self._M5 = np.zeros((N_AXES, N_AXES))  # full deviations (full windows only)
        self._n = 0
        self._n5 = 0
        self._t = 0
        self._route = "cheap"           # default cheap: no drift -> do not pay S_P

    def features_needed(self) -> list[str]:
        """Axes to extract BEFORE the next decide(). S_P may additionally be
        extracted lazily on same-window escalation; charge via last['features_used']."""
        if not self._fitted:
            return list(AXES)
        nxt = self._t + 1
        audit = self.audit_every is not None and ((nxt - self.audit_offset) % self.audit_every == 0)
        if self.force is not None:
            return list(AXES) if self.force == 'full' else list(CHEAP_AXES)
        if self._route == "full" or audit:
            return list(AXES)
        return list(CHEAP_AXES)

    def decide(self, feature_vec, fetch_sp=None) -> int:
        """feature_vec may carry NaN at S_P on cheap windows; if the router
        escalates, fetch_sp() is called (lazily, inside the timed path) to
        extract it. Replay harnesses passing full vectors are unchanged."""
        if not self._fitted:
            raise RuntimeError("decide() before fit()")
        f = np.asarray(feature_vec, dtype=float).reshape(-1)
        if f.shape[0] != N_AXES:
            raise ValueError(f"need {N_AXES} entries")
        self._t += 1
        d4 = f[CHEAP_IDX] - self.mu0[CHEAP_IDX]
        dc = d4 / self.sigma0[CHEAP_IDX]                 # standardised cheap devs
        self._M4 += np.outer(dc, dc)
        dw = self._W4 @ d4                               # whitened cheap devs
        self._sw += float(dw @ dw)
        self._n += 1
        E = np.clip(np.diag(self._M4) - self._n, 0.0, None)   # raw-axis excess
        tr_Ew = self._sw - 4.0 * self._n                      # whitened excess (any dir)
        sig_gate = self.k_sig * float(np.sqrt(8.0 * self._n))
        has_signal = tr_Ew > sig_gate
        R4s = float(np.max(E) / tr_Ew) if has_signal else 1.0
        dci4 = _pr_closed_form(self._M4, self._n)
        audit = self.audit_every is not None and ((self._t - self.audit_offset) % self.audit_every == 0)
        full = (has_signal and R4s < self.rho) or audit
        if self.force is not None:
            full = (self.force == 'full')   # arm pin: router stats above still logged
        self._route = "full" if full else "cheap"
        if full:
            if not np.isfinite(f[4]):
                if fetch_sp is None:
                    raise ValueError("full route needs S_P: pass it or provide fetch_sp")
                f = f.copy()
                f[4] = float(fetch_sp())
            d = f - self.mu0
            z = self._whiten @ d
            stat, k, axis = float(z @ z), N_AXES, None
            thr = self.thr5
            self._M5 += np.outer(d, d)
            self._n5 += 1
            used = list(AXES)
        else:
            j = int(np.argmax(dc * dc))
            stat, k, axis = float(np.max(dc * dc)), 1, AXES[j]
            thr = self.thr1_bonf
            used = list(CHEAP_AXES)
        fired = int(stat > thr)
        self.last = {"R4s": R4s, "tr_excess_w": tr_Ew, "has_signal": bool(has_signal),
                     "dci4": dci4,
                     "dci5": (_pr_closed_form(self._M5, self._n5)
                              if self._n5 else float("nan")),
                     "regime": self._route, "k": k, "axis": axis,
                     "statistic": stat, "threshold_F": thr, "fired": fired,
                     "audit": bool(audit), "features_used": used}
        return fired

    def config(self) -> dict:
        c = {"rho": self.rho, "alpha": self.alpha, "k_sig": self.k_sig,
             "audit_offset": self.audit_offset,
             "force": self.force,
             "min_windows": self.min_windows,
             "audit_every": self.audit_every, "ridge": self.ridge, "version": 3}
        if self._fitted:
            c.update({"m_steady": self.m, "thr1_bonf": self.thr1_bonf,
                      "thr5": self.thr5})
        return c
