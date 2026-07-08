#!/usr/bin/env python3
"""
features_alt.py -- Paper 3C feature-agnostic extractors (KERNEL-FREE).

Purpose: show DCI's low-/high-complexity regime is NOT an artifact of the
HSM kernel. The SAME drift windows are re-featurised with generic
representations that use NO HSM machinery, and DCI + 1-D/multi-D detection
AUC are recomputed with the identical Paper 3C statistics.

Representations (both from raw SQL, generic parsing only):
  raw-freq : per-window template-frequency (count) vector, union vocab.
  tables   : per-window table-incidence vector (FROM/JOIN tables).

Drift signal = the MOTION of the per-window vector (adjacent difference);
DCI = participation ratio tr(C)^2/||C||_F^2 of the drift-window motion
covariance (Paper 3C Eq. 1). 1-D detector = best single axis by |motion|;
multi-D = Mahalanobis under the steady-motion covariance.

NO import of kernel.* -- generic regex parsing only.
"""
import re
import numpy as np

# ---- generic SQL parsing (no HSM kernel) --------------------------------
_STR = re.compile(r"'[^']*'")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_WS  = re.compile(r"\s+")
_FROMJOIN = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)")

def canon_template(sql: str) -> str:
    """Literal-normalised template: strings->'?', numbers->?, ws collapsed."""
    s = _STR.sub("'?'", sql.lower())
    s = _NUM.sub("?", s)
    return _WS.sub(" ", s).strip()

def extract_tables(sql: str):
    return {m.group(1).lower() for m in _FROMJOIN.finditer(sql.lower())}

def _wsql(win):
    return win[0] if isinstance(win, tuple) else win

def freq_matrix(windows) -> np.ndarray:
    counts = []
    for win in windows:
        c = {}
        for q in _wsql(win):
            t = canon_template(q); c[t] = c.get(t, 0) + 1
        counts.append(c)
    vocab = sorted({t for c in counts for t in c})
    idx = {t: i for i, t in enumerate(vocab)}
    F = np.zeros((len(counts), max(1, len(vocab))))
    for r, c in enumerate(counts):
        for t, n in c.items():
            F[r, idx[t]] = n
    return F

def table_matrix(windows) -> np.ndarray:
    rows = []
    for win in windows:
        agg = {}
        for q in _wsql(win):
            for tb in extract_tables(q):
                agg[tb] = agg.get(tb, 0) + 1
        rows.append(agg)
    vocab = sorted({t for a in rows for t in a})
    idx = {t: i for i, t in enumerate(vocab)}
    F = np.zeros((len(rows), max(1, len(vocab))))
    for r, a in enumerate(rows):
        for t, n in a.items():
            F[r, idx[t]] = n
    return F

# ---- AUC (rank-based, tie-corrected; no scipy dependency) ---------------
def _rankdata(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float); sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks

def _auc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, float)
    npos = int((labels == 1).sum()); nneg = int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _rankdata(scores)
    return (r[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

# ---- motion-based DCI + detectors (Paper 3C statistics) -----------------
def analyse_alt(F: np.ndarray, drift_idx: set):
    """F: (n_windows x D) per-window feature matrix. drift_idx: transition
    window indices. Returns DCI (participation ratio of drift-window motion
    covariance) + 1-D/multi-D detection AUC."""
    M = np.diff(F, axis=0)                       # motion; row t = F[t+1]-F[t]
    lab = np.array([1 if (i + 1) in drift_idx else 0 for i in range(M.shape[0])])
    if lab.sum() < 2 or (lab == 0).sum() < 2:
        return None
    nd = lab == 0
    d = M - M[nd].mean(axis=0)
    # 1-D: best single axis by |motion|
    aucs = [_auc(lab, np.abs(d[:, j])) for j in range(d.shape[1])]
    auc_1d = float(np.nanmax(aucs)); best = int(np.nanargmax(aucs))
    # multi-D: Mahalanobis under steady-motion covariance
    Sig = np.cov(d[nd].T) + 1e-6 * np.eye(d.shape[1])
    Pinv = np.linalg.pinv(Sig)
    maha = np.einsum("ij,jk,ik->i", d, Pinv, d)
    auc_md = _auc(lab, maha)
    # DCI: participation ratio of drift-window motion covariance
    driftd = d[lab == 1]
    C = (driftd.T @ driftd) / len(driftd)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None); tot = ev.sum()
    dci = float(tot ** 2 / np.sum(ev ** 2)) if tot > 0 else float("nan")
    return {"dci": dci, "auc_1d": auc_1d, "auc_md": float(auc_md),
            "best_dim": best, "delta_auc": float(auc_md - auc_1d), "D": int(d.shape[1])}

# ---- generic bounded-similarity summary (kernel-free, NOT HSM) ----------
def _tmpl_counter(win):
    from collections import Counter
    return Counter(canon_template(q) for q in _wsql(win))

def sim_matrix(windows) -> np.ndarray:
    """Per-adjacent-pair 3-axis generic similarity in [0,1]^3:
    (template cosine, volume ratio, table Jaccard). Uses only generic
    parsing -- NOT the HSM kernel's 5 axes / weights / wavelet / DTW."""
    tvec = [_tmpl_counter(w) for w in windows]
    tbl  = [set().union(*[extract_tables(q) for q in _wsql(w)]) if _wsql(w) else set()
            for w in windows]
    cnt  = [sum(c.values()) for c in tvec]
    def cos(a, b):
        keys = set(a) | set(b)
        if not keys: return 1.0
        va = np.array([a.get(k, 0) for k in keys], float)
        vb = np.array([b.get(k, 0) for k in keys], float)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else 0.0
    def jac(a, b):
        return len(a & b) / len(a | b) if (a | b) else 1.0
    rows = []
    for t in range(1, len(windows)):
        s_t = cos(tvec[t-1], tvec[t])
        s_v = (min(cnt[t-1], cnt[t]) / max(cnt[t-1], cnt[t])) if max(cnt[t-1], cnt[t]) > 0 else 1.0
        s_a = jac(tbl[t-1], tbl[t])
        rows.append([s_t, s_v, s_a])
    return np.asarray(rows, float)

def analyse_sim(FV: np.ndarray, drift_idx: set):
    """HSM-style analysis for bounded-similarity features: position
    deviation from steady mean, 1-D score = 1 - similarity."""
    n = FV.shape[0]
    lab = np.array([1 if (i + 1) in drift_idx else 0 for i in range(n)])
    if lab.sum() < 2 or (lab == 0).sum() < 2:
        return None
    nd = lab == 0
    d = FV - FV[nd].mean(axis=0)
    aucs = [_auc(lab, 1.0 - FV[:, j]) for j in range(FV.shape[1])]
    auc_1d = float(np.nanmax(aucs)); best = int(np.nanargmax(aucs))
    Sig = np.cov(d[nd].T) + 1e-6 * np.eye(d.shape[1]); Pinv = np.linalg.pinv(Sig)
    maha = np.einsum("ij,jk,ik->i", d, Pinv, d); auc_md = _auc(lab, maha)
    driftd = d[lab == 1]; C = (driftd.T @ driftd) / len(driftd)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None); tot = ev.sum()
    dci = float(tot ** 2 / np.sum(ev ** 2)) if tot > 0 else float("nan")
    return {"dci": dci, "auc_1d": auc_1d, "auc_md": float(auc_md),
            "best_dim": best, "delta_auc": float(auc_md - auc_1d), "D": int(FV.shape[1])}
