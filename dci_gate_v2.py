#!/usr/bin/env python3
"""dci_gate_v2.py -- the DCI-routed gate, revision 2 (Paper 3C, post external-review).

Changes vs dci_gate.py (v1), per review items B2/B3:

  1. CHEAP TIER IS A COORDINATE AXIS. v1's 1-D arm was the whitened matched
     filter (v1 @ z)^2 -- a full-kernel-tier detector in disguise (needs all
     five features + a per-window eigh). v2's cheap arm tests ONE standardised
     coordinate (d_j / sigma_j)^2 against the same finite-sample Hotelling-F
     threshold family, where axis j = argmax of the noise-standardised
     trajectory diagonal M_jj / sigma_j^2 (the empirical maximiser of the
     cheap arm's own expected statistic; scale-unbiased across axes). O(D) per window,
     one feature computed, no eigendecomposition.

  2. NO EIG IN THE HOT PATH. Routing DCI uses the closed form
     tr(C)^2 / ||C||_F^2 on a running 5x5 outer-product sum (O(D^2));
     Sigma^{-1/2} for the multi-D arm is precomputed once in fit().
     eigh appears only in fit() -- off the monitoring path.

  3. PROBE-CADENCE ARCHITECTURE (parameter P = probe_every). The paper's Sec 4
     states DCI is computed per block, not per window; v1 nevertheless read
     all five features every window. v2 makes the cost story real: every P-th
     window is a PROBE (full five-axis features -> update trajectory, DCI,
     route, axis choice); between probes the committed tier runs alone --
     the cheap tier computes only its one axis feature. P=1 reproduces
     v1-style every-window routing (with the axis-restricted cheap arm).

Per-window feature requirement is exposed via `features_needed()` so a replay
harness can charge measured per-axis extraction costs faithfully.

API mirrors v1: fit(steady) -> decide(f) -> 0/1, .last diagnostics,
reset_trajectory() per RCB block.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import f as _f_dist

TAU_DEFAULT = 1.5
RIDGE = 1e-6
N_AXES = 5
AXES = ["S_R", "S_V", "S_T", "S_A", "S_P"]


def _hotelling_c(m: int, k: int) -> float:
    return (m * (m - k)) / ((m + 1) * k * (m - 1))


def dci_closed_form(M: np.ndarray, n: int) -> float:
    """Participation ratio of C = M/n via trace and Frobenius norm only."""
    if n <= 0:
        return float("nan")
    C = M / n
    tr = float(np.trace(C))
    f2 = float(np.sum(C * C))
    if tr <= 0.0 or f2 <= 0.0:
        return 1.0
    return (tr * tr) / f2


class DCIGateV2:
    def __init__(self, tau: float = TAU_DEFAULT, alpha: float = 0.05,
                 min_dci_windows: int = 3, probe_every: int = 1,
                 ridge: float = RIDGE):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha in (0,1)")
        if probe_every < 1:
            raise ValueError("probe_every >= 1")
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.min_dci_windows = int(min_dci_windows)
        self.P = int(probe_every)
        self.ridge = float(ridge)
        self._fitted = False
        self.last: dict | None = None

    # ---------------- calibration (one-time; eigh allowed here) ----------
    def fit(self, steady_features) -> "DCIGateV2":
        X = np.asarray(steady_features, dtype=float)
        if X.ndim != 2 or X.shape[1] != N_AXES:
            raise ValueError(f"steady_features must be (m,{N_AXES})")
        m = X.shape[0]
        if m <= N_AXES + 1:
            raise ValueError(f"need m > {N_AXES+1}")
        self.m = m
        self.mu0 = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1) + self.ridge * np.eye(N_AXES)
        self.Sigma0 = cov
        self.sigma0 = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        evals, evecs = np.linalg.eigh(cov)          # fit()-time only
        evals = np.clip(evals, 1e-12, None)
        self._whiten = evecs @ np.diag(evals ** -0.5) @ evecs.T
        self.thr_F = {k: float(_f_dist.ppf(1.0 - self.alpha, k, m - k)) / _hotelling_c(m, k)
                      for k in (1, N_AXES)}
        self.reset_trajectory()
        self._fitted = True
        return self

    def reset_trajectory(self) -> None:
        self._M = np.zeros((N_AXES, N_AXES))   # running sum of d d^T (probes only)
        self._n = 0
        self._t = 0                             # window counter within block
        self._route = "5-D"                     # committed route (5-D until estimable)
        self._axis = None                       # committed cheap axis index

    # ---------------- per-window ----------------------------------------
    def features_needed(self) -> list[str]:
        """Which feature axes the NEXT window must compute (cost accounting)."""
        if not self._fitted:
            return list(AXES)
        nxt = self._t + 1
        if self.P == 1 or (nxt % self.P) == 1 or self._route == "5-D" \
           or self._n < self.min_dci_windows:
            return list(AXES)
        return [AXES[self._axis]]

    def decide(self, feature_vec) -> int:
        if not self._fitted:
            raise RuntimeError("decide() before fit()")
        f = np.asarray(feature_vec, dtype=float).reshape(-1)
        if f.shape[0] != N_AXES:
            raise ValueError(f"need {N_AXES} entries")
        self._t += 1
        probe = (self.P == 1) or (self._t % self.P == 1) \
                or (self._route == "5-D") or (self._n < self.min_dci_windows)
        d = f - self.mu0
        dci = float("nan")
        if probe:
            self._M += np.outer(d, d)
            self._n += 1
            if self._n >= self.min_dci_windows:
                dci = dci_closed_form(self._M, self._n)
                if dci >= self.tau:
                    self._route = "5-D"
                else:
                    self._route = "1-D"
                    self._axis = int(np.argmax(np.diag(self._M) / (self.sigma0 ** 2)))
            else:
                self._route = "5-D"
        if self._route == "5-D":
            z = self._whiten @ d
            stat, k, j = float(z @ z), N_AXES, None
        else:
            j = self._axis
            stat, k = float((d[j] / self.sigma0[j]) ** 2), 1
        thr = self.thr_F[k]
        fired = int(stat > thr)
        self.last = {"dci": dci, "regime": self._route, "k": k,
                     "axis": (None if j is None else AXES[j]),
                     "statistic": stat, "threshold_F": thr, "fired": fired,
                     "probe": bool(probe), "n_probes": self._n,
                     "axis_share": (float(np.max(np.diag(self._M) / (self.sigma0 ** 2)) /
                                          max(np.sum(np.diag(self._M) / (self.sigma0 ** 2)), 1e-300))
                                    if self._n else float("nan"))}
        return fired

    def config(self) -> dict:
        c = {"tau": self.tau, "alpha": self.alpha, "probe_every": self.P,
             "min_dci_windows": self.min_dci_windows, "ridge": self.ridge, "version": 2}
        if self._fitted:
            c.update({"m_steady": self.m, "thr_F": dict(self.thr_F)})
        return c
