"""
hsm_v2_kernel.py
================
Canonical v2 HSM five-dimension kernel, ported verbatim from

    Paper 3A/Version 3/code/experiments/v2_10seed/hsm_v2_core.py
    Paper 3A/Version 3/code/src/hsm/measures.py

so that all seven HSM_gated validation scripts evaluate the SAME scoring
function that the paper (§III) describes.

Paper §III formulae implemented here:

    S_R = 1 - arccos(ρ_s) / π         # Spearman correlation, angular
    S_V = min(Q_a, Q_b) / max(Q_a, Q_b)
    S_T = 1 - (2/π) · arccos(v̂_A · v̂_B)
    S_A = 0.5 · J(tables) + 0.5 · J(cols)        # dual Jaccard
    S_P = Σ_band λ_b · sc_b   (DWT db4, L=3 → SAX α=4 → FastDTW r=3)
    HSM = 0.25·S_R + 0.20·S_V + 0.20·S_T + 0.20·S_A + 0.15·S_P

All functions are stateless and pure, so every validation script can
import them directly.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Paper-locked parameters ──────────────────────────────────────────────────
W0: Dict[str, float] = {"R": 0.25, "V": 0.20, "T": 0.20, "A": 0.20, "P": 0.15}

WAVELET = "db4"
DWT_LEVEL = 3
SAX_ALPHA = 4
FASTDTW_RADIUS = 3
BAND_WEIGHTS: Tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)

# Series-length thresholds for graceful S_P fallback
_MIN_LEN_L3 = 16
_MIN_LEN_L2 = 8
_MIN_LEN_ABS = 4


# ─── Dimension-level kernels ──────────────────────────────────────────────────

def sr_v2(freq_a: np.ndarray, freq_b: np.ndarray) -> float:
    """S_R: 1 - arccos(Spearman ρ_s) / π (paper Eq. 1).

    Defensive against mismatched sizes: if the two vectors have different
    lengths, the shorter is zero-padded on the right so spearmanr does not
    crash.  Callers that care about correlational semantics should align
    vectors on a shared axis before calling (see hsm_score_from_features
    which does this via freq_map).
    """
    a = np.asarray(freq_a, dtype=float)
    b = np.asarray(freq_b, dtype=float)
    if a.size != b.size:
        m = max(a.size, b.size)
        if a.size < m:
            a = np.concatenate([a, np.zeros(m - a.size)])
        if b.size < m:
            b = np.concatenate([b, np.zeros(m - b.size)])
    if a.size < 2 or b.size < 2 or a.sum() == 0 or b.sum() == 0:
        return 0.5
    rho, _ = spearmanr(a, b)
    if np.isnan(rho):
        rho = 1.0
    rho = float(np.clip(rho, -1.0, 1.0))
    return float(1.0 - math.acos(rho) / math.pi)


def sv_v2(n_a: int, n_b: int) -> float:
    """S_V: min/max query-count ratio (paper Eq. 2)."""
    n_a, n_b = int(n_a), int(n_b)
    m = max(n_a, n_b)
    if m == 0:
        return 1.0
    return float(min(n_a, n_b) / m)


def st_v2(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """S_T: 1 - (2/π)·arccos(v̂_A · v̂_B) (paper Eq. 3; Lemma 1 P4 needs this)."""
    va = np.asarray(vec_a, dtype=float)
    vb = np.asarray(vec_b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 1.0
    cos_val = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return float(1.0 - (2.0 / math.pi) * math.acos(cos_val))


def sa_v2(tables_a: Set[str], tables_b: Set[str],
          cols_a: Set[str], cols_b: Set[str]) -> float:
    """S_A: 0.5·J(tables) + 0.5·J(columns) dual Jaccard (paper Eq. 4)."""
    ta = set(tables_a); tb = set(tables_b)
    ca = set(cols_a);   cb = set(cols_b)
    t_union = len(ta | tb)
    c_union = len(ca | cb)
    j_t = len(ta & tb) / t_union if t_union > 0 else 1.0
    j_c = len(ca & cb) / c_union if c_union > 0 else 1.0
    return float(0.5 * j_t + 0.5 * j_c)


# ─── S_P: DWT + SAX + FastDTW ────────────────────────────────────────────────

def _sax_encode(arr: np.ndarray, alpha: int = SAX_ALPHA) -> np.ndarray:
    """SAX quantisation with Gaussian breakpoints."""
    from scipy.stats import norm
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=int)
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-12:
        return np.zeros(x.size, dtype=int)
    z = (x - mu) / sd
    breakpoints = norm.ppf(np.linspace(0.0, 1.0, alpha + 1)[1:-1])
    return np.digitize(z, breakpoints).astype(int)


def _band_score(ca: np.ndarray, cb: np.ndarray,
                alpha: int = SAX_ALPHA,
                radius: int = FASTDTW_RADIUS) -> float:
    """SAX + FastDTW per-band score, symmetrised for Lemma 1 P3.

    Two corrections to the underlying FastDTW library, both required by
    Lemma 1:

    1. **Self-distance short circuit (P2).** ``fastdtw`` is a recursive
       Sakoe-Chiba band approximation; its multi-resolution coarsening
       can return a strictly positive distance even on byte-identical
       sequences when the input is long enough that the coarsened path
       cannot reach all diagonal cells.  True DTW(s, s) is 0 for any s,
       so when the SAX encodings are equal we return 1.0 directly.

    2. **Symmetrisation (P3).** ``fastdtw(a, b)`` and ``fastdtw(b, a)``
       can differ by O(band-width) numerical noise.  We average both
       directions so the band score is exactly symmetric.
    """
    from fastdtw import fastdtw
    if len(ca) == 0 or len(cb) == 0:
        return 1.0
    sa = _sax_encode(ca, alpha)
    sb = _sax_encode(cb, alpha)
    n_ref = max(len(sa), len(sb))
    denom = n_ref * (alpha - 1)
    if denom == 0:
        return 1.0
    # P2: short-circuit identical sequences (true DTW(s, s) = 0).
    if sa.size == sb.size and np.array_equal(sa, sb):
        return 1.0
    d_ab, _ = fastdtw(sa, sb, radius=radius,
                      dist=lambda x, y: abs(int(x) - int(y)))
    d_ba, _ = fastdtw(sb, sa, radius=radius,
                      dist=lambda x, y: abs(int(x) - int(y)))
    dist = 0.5 * (d_ab + d_ba)
    return float(max(0.0, 1.0 - dist / denom))


def build_qps_series(elapsed_ms: Iterable[float],
                     min_bins: int = 16,
                     bin_seconds: float = 1.0) -> np.ndarray:
    """Construct the q(t) "QPS time series at 1-second resolution"
    defined in paper §III-A line 291.

    Given per-query elapsed times (milliseconds), treat execution as
    sequential and place each query's *arrival* at its cumulative
    finish time.  Bucket those arrivals into 1-second bins giving the
    q(t) series that S_P's DWT + SAX + FastDTW pipeline is defined on.

    Parameters
    ----------
    elapsed_ms : iterable of floats (milliseconds per query)
    min_bins   : pad the output to at least this many points so the
                 level-3 db4 DWT has enough samples (2**3 = 8; we keep
                 16 as a comfortable floor).
    bin_seconds: bucket width; the paper specifies 1 second.

    Returns
    -------
    q_t : 1-D np.ndarray of non-negative counts, length >= min_bins.
    """
    arr = np.asarray(list(elapsed_ms), dtype=float)
    if arr.size == 0:
        return np.zeros(min_bins, dtype=float)
    # Cumulative finish time (seconds) ≈ arrival of next query for a
    # serial executor.  For a trace with true arrival timestamps, pass
    # those directly and skip this helper.
    arrival_s = np.cumsum(np.maximum(arr, 0.0)) / 1000.0
    duration = float(arrival_s[-1])
    if duration <= 0:
        out = np.zeros(min_bins, dtype=float)
        out[0] = float(arr.size)
        return out
    n_bins = max(int(np.ceil(duration / bin_seconds)), min_bins)
    edges = np.linspace(0.0, duration + 1e-9, n_bins + 1)
    q_t, _ = np.histogram(arrival_s, bins=edges)
    return q_t.astype(float)


def arrivals_to_qps_series(arrival_s: Iterable[float],
                           duration_s: float,
                           min_bins: int = 16,
                           bin_seconds: float = 1.0) -> np.ndarray:
    """Same contract as ``build_qps_series`` but for traces that record
    per-query wall-clock arrival timestamps (seconds from window start).
    """
    arr = np.asarray(list(arrival_s), dtype=float)
    if arr.size == 0 or duration_s <= 0:
        return np.zeros(min_bins, dtype=float)
    n_bins = max(int(np.ceil(float(duration_s) / bin_seconds)), min_bins)
    edges = np.linspace(0.0, float(duration_s) + 1e-9, n_bins + 1)
    q_t, _ = np.histogram(arr, bins=edges)
    return q_t.astype(float)


def sp_v2(times_a: Iterable[float], times_b: Iterable[float],
          qset_a: Optional[Set[str]] = None,
          qset_b: Optional[Set[str]] = None) -> float:
    """
    S_P: DWT(db4, L=3) → SAX(α=4) → FastDTW(Sakoe-Chiba r=3),
    band-weighted sum with weights (0.40, 0.20, 0.20, 0.20).

    Graceful fallback: if the series is shorter than the db4 filter can
    support at level 3, we drop to level 2; below len=4 we fall back to
    a Jaccard over query-identifier sets (if provided) to keep the
    composite score finite.
    """
    ta_full = np.asarray(list(times_a), dtype=float)
    tb_full = np.asarray(list(times_b), dtype=float)
    min_len = int(min(ta_full.size, tb_full.size))

    if min_len < _MIN_LEN_ABS:
        if qset_a is None or qset_b is None:
            return 0.5
        union = set(qset_a) | set(qset_b)
        return float(len(set(qset_a) & set(qset_b)) / len(union)) if union else 1.0

    ta = ta_full[:min_len].copy()
    tb = tb_full[:min_len].copy()

    # Robust scale-normalisation to kill raw-magnitude effects.
    scale = max(float(ta.max()), float(tb.max()), 1e-12)
    ta /= scale
    tb /= scale

    level = DWT_LEVEL if min_len >= _MIN_LEN_L3 else (2 if min_len >= _MIN_LEN_L2 else 1)

    try:
        import pywt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs_a = pywt.wavedec(ta, WAVELET, level=level)
            coeffs_b = pywt.wavedec(tb, WAVELET, level=level)
    except Exception:
        if qset_a is None or qset_b is None:
            return 0.5
        union = set(qset_a) | set(qset_b)
        return float(len(set(qset_a) & set(qset_b)) / len(union)) if union else 1.0

    # pywt returns [cA_L, cD_L, cD_{L-1}, ..., cD_1]; pair them band-for-band
    # up to the four paper bands (cA3, cD3, cD2, cD1).  If level<3 we pad
    # with ones so the weighted sum still has four terms.
    scores = [_band_score(a, b) for a, b in zip(coeffs_a, coeffs_b)]
    while len(scores) < 4:
        scores.append(1.0)
    sp = sum(w * s for w, s in zip(BAND_WEIGHTS, scores[:4]))
    return float(np.clip(sp, 0.0, 1.0))


# ─── Aggregation ──────────────────────────────────────────────────────────────

def hsm_v2(freq_a: np.ndarray, freq_b: np.ndarray,
           n_a: int, n_b: int,
           tables_a: Set[str], tables_b: Set[str],
           cols_a: Set[str], cols_b: Set[str],
           times_a: Iterable[float], times_b: Iterable[float],
           qset_a: Optional[Set[str]] = None,
           qset_b: Optional[Set[str]] = None,
           weights: Optional[Dict[str, float]] = None,
           type_vec_a: Optional[np.ndarray] = None,
           type_vec_b: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute the full five-dimension HSM score and its components.

    For the relational extractor (paper §III-B), S_T uses the per-template
    frequency vector aligned on the union vocabulary; this is the default
    path when ``type_vec_a`` / ``type_vec_b`` are not supplied.

    The optional ``type_vec_*`` parameters exist solely for the document
    extractor (paper §III-B MongoDB pipeline categories), where 𝒬 is a
    fixed enumeration {find, aggregate, insert, update, delete} rather
    than a variable template set.  Relational validation scripts do NOT
    pass these — passing a CRUD-tier vector for a relational workload
    would deviate from the paper's "set of SQL query templates"
    definition and is no longer done anywhere in the codebase
    (audit 2026-04-14).
    """
    if weights is None:
        weights = W0

    sr = sr_v2(freq_a, freq_b)
    sv = sv_v2(n_a, n_b)
    if type_vec_a is not None and type_vec_b is not None:
        st = st_v2(type_vec_a, type_vec_b)
    else:
        st = st_v2(freq_a, freq_b)
    sa = sa_v2(tables_a, tables_b, cols_a, cols_b)
    sp = sp_v2(times_a, times_b, qset_a, qset_b)

    hsm = (weights["R"] * sr + weights["V"] * sv + weights["T"] * st
           + weights["A"] * sa + weights["P"] * sp)

    return {
        "S_R": float(sr), "S_V": float(sv), "S_T": float(st),
        "S_A": float(sa), "S_P": float(sp), "HSM": float(hsm),
    }


# ─── Thin adapter used by validation scripts ─────────────────────────────────

def hsm_score_from_features(fa: dict, fb: dict,
                            weights: Optional[Dict[str, float]] = None
                            ) -> Tuple[float, Dict[str, float]]:
    """
    Adapter for validation scripts' feature-dict convention.

    Expected keys in each feature dict (tolerant: missing fields get sane
    defaults so incrementally upgraded scripts don't crash):

        freq_map   : dict[str,int] preferred — {template_id: count}.  When
                     both feature dicts carry this key the adapter aligns
                     both vectors on the union of template IDs so
                     spearmanr sees matched lengths.
        freq       : np.ndarray  fallback per-template frequency vector
                     (used only when freq_map is absent).  Both windows
                     must use the same axis for Spearman to be meaningful.
        n          : int         query count in the window
        tables     : set[str]
        cols       : set[str]
        times      : np.ndarray  per-query timing series (ms)
        qset       : set[str]    query-template identifiers
        type_vec   : np.ndarray  optional, overrides S_T input

    Returns:
        (HSM_score, per_dimension_dict)
    """
    fmap_a = fa.get("freq_map")
    fmap_b = fb.get("freq_map")
    if fmap_a is not None and fmap_b is not None:
        # Align both frequency vectors on the union of template IDs so that
        # np.column_stack inside spearmanr sees matched dimensions.
        axis = sorted(set(fmap_a.keys()) | set(fmap_b.keys()))
        freq_a = np.array([float(fmap_a.get(k, 0.0)) for k in axis], dtype=float)
        freq_b = np.array([float(fmap_b.get(k, 0.0)) for k in axis], dtype=float)
    else:
        freq_a = np.asarray(fa.get("freq", np.array([1.0])), dtype=float)
        freq_b = np.asarray(fb.get("freq", np.array([1.0])), dtype=float)
        if freq_a.size != freq_b.size:
            # Pad the shorter with zeros so spearmanr doesn't crash.  This
            # is a last-resort path; callers should provide freq_map.
            m = max(freq_a.size, freq_b.size)
            if freq_a.size < m:
                freq_a = np.concatenate([freq_a, np.zeros(m - freq_a.size)])
            if freq_b.size < m:
                freq_b = np.concatenate([freq_b, np.zeros(m - freq_b.size)])
    dims = hsm_v2(
        freq_a, freq_b,
        n_a=int(fa.get("n", 0)), n_b=int(fb.get("n", 0)),
        tables_a=set(fa.get("tables", set())), tables_b=set(fb.get("tables", set())),
        cols_a=set(fa.get("cols", set())),     cols_b=set(fb.get("cols", set())),
        times_a=fa.get("times", np.array([])), times_b=fb.get("times", np.array([])),
        qset_a=set(fa.get("qset", set())),     qset_b=set(fb.get("qset", set())),
        weights=weights,
        type_vec_a=fa.get("type_vec"),         type_vec_b=fb.get("type_vec"),
    )
    return dims["HSM"], {
        "S_R": dims["S_R"], "S_V": dims["S_V"], "S_T": dims["S_T"],
        "S_A": dims["S_A"], "S_P": dims["S_P"],
    }


# ─── Raw pair-score dump (for fig01 / fig02 / fig03) ─────────────────────────

def dump_pair_scores_csv(path, within_scores, cross_scores, workload: str) -> None:
    """Persist the reference-seed within/cross pair-score lists to a two-column
    CSV at `path` with header ``workload,group,score``.

    This is the canonical input format consumed by:
        - figures/plot_fig01_score_distribution.py  (groups within/cross)
        - figures/plot_fig03_within_cross_phase.py  (adds workload column)

    Only the reference seed is dumped so the distribution shown in the
    paper's figure is consistent with `within_mean` / `cross_mean` in the
    summary CSV.
    """
    import csv
    import os as _os
    _os.makedirs(_os.path.dirname(path), exist_ok=True) if _os.path.dirname(path) else None
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["workload", "group", "score"])
        for s in within_scores:
            w.writerow([workload, "within", f"{float(s):.6f}"])
        for s in cross_scores:
            w.writerow([workload, "cross", f"{float(s):.6f}"])
