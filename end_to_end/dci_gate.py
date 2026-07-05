#!/usr/bin/env python3
"""
dci_gate.py -- the DCI-routed drift gate (Paper 3C).

Self-contained: routes on the *raw* participation ratio (the DCI of
Paper 3C -- identical statistic to the regime map / `cost_benefit.analyse`)
and detects with the 5-D Mahalanobis norm (DCI >= tau) or the 1-D
matched filter onto the dominant mode (DCI < tau), each fired against the
exact finite-sample Hotelling-F threshold (Prop. 6a).

    gate = DCIGate(tau=1.5, alpha=0.05)
    gate.fit(steady_feature_rows)         # unsupervised calibration, frozen
    verdict = gate.decide(window_feature_vec)   # per window -> 1 / 0
    gate.reset_trajectory()               # at each new RCB block

`decide` returns 1 = invoke the advisor, 0 = skip; after each call
`gate.last` holds diagnostics (DCI, regime, statistic, threshold, fired).

Dependencies: numpy, scipy.stats (f, chi2). No database driver, no kernel
import -- the gate consumes 5-D feature vectors produced upstream by the
workload-similarity kernel.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import f as _f_dist, chi2 as _chi2_dist

TAU_DEFAULT = 1.5      # frozen DCI routing threshold (Paper 3C / RQ2)
RIDGE = 1e-6           # steady-covariance ridge (keeps Sigma0 invertible)
N_AXES = 5             # S_R, S_V, S_T, S_A, S_P


def _hotelling_c(m: int, k: int) -> float:
    """Scale constant: D_t * c(m,k) ~ F_{k, m-k} (Prop. 6)."""
    return (m * (m - k)) / ((m + 1) * k * (m - 1))


def participation_ratio(cov: np.ndarray) -> float:
    """DCI = trace(C)^2 / ||C||_F^2 = 1 / sum(p_i^2). Bounded in [1, dim]."""
    ev = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    tot = float(ev.sum())
    fro2 = float(np.sum(ev ** 2))
    if tot <= 0.0 or fro2 <= 0.0:
        return 1.0
    return (tot * tot) / fro2


class DCIGate:
    """DCI-routed gate. Routing DCI is the RAW participation ratio (Paper 3C);
    the 1-D/5-D detectors operate in the whitened space for exact thresholds."""

    def __init__(self, tau: float = TAU_DEFAULT, alpha: float = 0.05,
                 min_dci_windows: int = 3, dci_max_window: int | None = None,
                 ridge: float = RIDGE):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.min_dci_windows = int(min_dci_windows)
        self.dci_max_window = dci_max_window
        self.ridge = float(ridge)
        self._fitted = False
        self.last: dict | None = None

    # -- calibration (unsupervised; frozen for the official blocks) --------
    def fit(self, steady_features) -> "DCIGate":
        X = np.asarray(steady_features, dtype=float)
        if X.ndim != 2 or X.shape[1] != N_AXES:
            raise ValueError(f"steady_features must be (m, {N_AXES}), got {X.shape}")
        m = X.shape[0]
        if m <= N_AXES + 1:
            raise ValueError(f"need m > {N_AXES + 1} steady windows, got m={m}")
        self.m = m
        self.mu0 = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1) + self.ridge * np.eye(N_AXES)
        self.Sigma0 = cov
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-12, None)
        self._whiten = evecs @ np.diag(evals ** -0.5) @ evecs.T
        self.thr_F: dict[int, float] = {}
        for k in (1, N_AXES):
            c = _hotelling_c(m, k)
            self.thr_F[k] = float(_f_dist.ppf(1.0 - self.alpha, k, m - k)) / c
        self._traj: list[np.ndarray] = []       # whitened deviations (detectors)
        self._traj_raw: list[np.ndarray] = []    # raw deviations (routing DCI)
        self._fitted = True
        return self

    # -- per-window decision ------------------------------------------------
    def decide(self, feature_vec) -> int:
        if not self._fitted:
            raise RuntimeError("DCIGate.decide() called before fit()")
        f = np.asarray(feature_vec, dtype=float).reshape(-1)
        if f.shape[0] != N_AXES:
            raise ValueError(f"feature_vec must have {N_AXES} entries")

        d = f - self.mu0                 # raw deviation (routing DCI)
        z = self._whiten @ d             # whitened deviation (detectors)
        self._traj.append(z)
        self._traj_raw.append(d)
        D5 = float(z @ z)                # 5-D Mahalanobis statistic

        traj_w, traj_r = self._traj, self._traj_raw
        if self.dci_max_window is not None:
            traj_w = traj_w[-self.dci_max_window:]
            traj_r = traj_r[-self.dci_max_window:]

        if len(traj_r) >= self.min_dci_windows:
            Dr = np.asarray(traj_r)
            C_raw = (Dr.T @ Dr) / Dr.shape[0]        # RAW drift covariance
            dci = participation_ratio(C_raw)          # <-- the Paper 3C DCI
            Zt = np.asarray(traj_w)
            C_w = (Zt.T @ Zt) / Zt.shape[0]
            evals, evecs = np.linalg.eigh(C_w)
            v1 = evecs[:, int(np.argmax(evals))]      # dominant mode (whitened)
        else:
            dci = float("nan")                        # default 5-D until estimable
            v1 = None

        if np.isnan(dci) or dci >= self.tau:
            regime, k, stat = "5-D", N_AXES, D5
        else:
            regime, k = "1-D", 1
            stat = float((v1 @ z) ** 2)               # matched filter ~ chi2_1

        thr = self.thr_F[k]
        fired = int(stat > thr)
        self.last = {"dci": dci, "regime": regime, "k": k, "statistic": stat,
                     "threshold_F": thr, "fired": fired,
                     "n_trajectory": len(self._traj)}
        return fired

    # -- convenience --------------------------------------------------------
    def reset_trajectory(self) -> None:
        self._traj = []
        self._traj_raw = []

    def config(self) -> dict:
        cfg = {"tau": self.tau, "alpha": self.alpha,
               "min_dci_windows": self.min_dci_windows,
               "dci_max_window": self.dci_max_window, "ridge": self.ridge}
        if self._fitted:
            cfg.update({"m_steady": self.m, "thr_F": dict(self.thr_F)})
        return cfg
