#!/usr/bin/env python3
"""
verify_epsilon.py
=================

Paper 3C -- empirical verification of the sufficiency tolerance epsilon.

epsilon is *derived* in delong.py (DeLong 1988 + DerSimonian-Laird
1986).  Derivation alone is not proof that the number is right in
finite samples -- so this script CHECKS it three ways, on synthetic
data where the truth is known:

  CHECK A -- DeLong SE vs bootstrap SE.
      DeLong gives a closed-form Var(AUC_5d - AUC_d).  An independent
      stratified bootstrap (resample drift and steady windows) gives
      the same variance empirically.  The two must agree:
      ratio = Var_DeLong / Var_bootstrap  ~ 1.

  CHECK B -- per-trajectory coverage.
      Generate two detectors with EQUAL true AUC (true gap = 0).  The
      95% interval [delta_hat +/- epsilon] must cover 0 about 95% of
      the time.  Under-coverage => epsilon too small.

  CHECK C -- DerSimonian-Laird cell coverage + tau^2 recovery.
      Simulate a cell of n_s seeds whose true per-seed gaps are drawn
      around a cell gap Delta with between-seed SD tau_true.  Run the
      full pipeline (per-seed DeLong -> combine_random_effects).  The
      cell interval [Delta_hat +/- epsilon_cell] must cover Delta about
      95% of the time, and the estimated tau^2 must track tau_true^2.

The drift windows per trajectory are FEW (the real harness has 6), so
CHECK B is run at that hard setting and at a larger one to show the
normal approximation converging.

RUN
  python verify_epsilon.py            # ~10-20 s; needs delong.py alongside
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from delong import (delong_auc_cov, delong_epsilon,        # noqa: E402
                    combine_across_seed, combine_random_effects)

SQRT2 = np.sqrt(2.0)


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def auc_to_mu(auc: float) -> float:
    """Separation mu of a Gaussian detector with the given true AUC.

    positive ~ N(mu, 1), negative ~ N(0, 1)  =>  AUC = Phi(mu / sqrt 2).
    """
    return SQRT2 * float(norm.ppf(np.clip(auc, 1e-4, 1 - 1e-4)))


def make_pair(rng, m: int, n: int, mu_a: float, mu_b: float,
              rho: float = 0.6) -> tuple:
    """Two correlated detector scores on m drift + n steady windows.

    The detectors share a fraction rho of their noise (they score the
    SAME windows), so their AUC estimates are correlated -- exactly the
    regime DeLong's covariance is for.
    """
    N = m + n
    lab = np.array([1] * m + [0] * n)
    shared = rng.normal(size=N)
    noise_a = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(size=N)
    noise_b = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(size=N)
    return lab, mu_a * lab + noise_a, mu_b * lab + noise_b


def _auc_pair(lab, sa, sb) -> tuple:
    aucs, _ = delong_auc_cov(lab, np.column_stack([sa, sb]))
    return float(aucs[0]), float(aucs[1])


# --------------------------------------------------------------------------
def check_a_bootstrap(rng, m=6, n=30, trials=120, n_boot=800) -> dict:
    """DeLong variance of the AUC gap vs a stratified bootstrap."""
    ratios = []
    for _ in range(trials):
        mu = auc_to_mu(rng.uniform(0.70, 0.97))
        mu2 = auc_to_mu(rng.uniform(0.70, 0.97))
        lab, sa, sb = make_pair(rng, m, n, mu, mu2)
        var_delong = delong_epsilon(lab, sa, sb)["se_diff"] ** 2
        pos = np.where(lab == 1)[0]
        neg = np.where(lab == 0)[0]
        gaps = np.empty(n_boot)
        for b in range(n_boot):
            pi = rng.choice(pos, size=m, replace=True)
            ni = rng.choice(neg, size=n, replace=True)
            idx = np.concatenate([pi, ni])
            la = np.concatenate([np.ones(m), np.zeros(n)])
            a_a, a_b = _auc_pair(la, sa[idx], sb[idx])
            gaps[b] = a_b - a_a
        var_boot = float(np.var(gaps, ddof=1))
        if var_boot > 1e-9 and np.isfinite(var_delong):
            ratios.append(var_delong / var_boot)
    ratios = np.asarray(ratios)
    return {"m": m, "n": n, "trials": len(ratios),
            "mean_ratio_delong_over_bootstrap": float(ratios.mean()),
            "sd_ratio": float(ratios.std(ddof=1)),
            "median_ratio": float(np.median(ratios))}


def check_b_coverage(rng, m, n, trials=4000, auc_true=0.82) -> dict:
    """True gap = 0: the 95% interval [delta +/- epsilon] must cover 0."""
    mu = auc_to_mu(auc_true)
    covered = 0
    for _ in range(trials):
        lab, sa, sb = make_pair(rng, m, n, mu, mu)
        e = delong_epsilon(lab, sa, sb)
        if np.isfinite(e["epsilon"]) and abs(e["auc_diff"]) <= e["epsilon"]:
            covered += 1
    return {"m": m, "n": n, "trials": trials,
            "empirical_coverage": covered / trials,
            "nominal": 0.95}


def check_c_cell(rng, n_seeds=50, m=6, n=30,
                 delta_cell=0.10, tau_true=0.05, trials=1500) -> dict:
    """Full pipeline: per-seed DeLong -> cell pooling.

    Runs BOTH cell estimators on the same simulated cells:
      * combine_across_seed     -- equal-weight t-interval (recommended)
      * combine_random_effects  -- DerSimonian-Laird inverse-variance
    so their coverage can be compared directly.
    """
    cov_as = cov_dl = 0
    tau2_hats, eps_as, eps_dl = [], [], []
    auc_a = 0.80
    for _ in range(trials):
        deltas, variances = [], []
        for _s in range(n_seeds):
            d_s = delta_cell + rng.normal(0.0, tau_true)
            mu_a = auc_to_mu(auc_a)
            mu_b = auc_to_mu(np.clip(auc_a + d_s, 0.55, 0.995))
            lab, sa, sb = make_pair(rng, m, n, mu_a, mu_b)
            e = delong_epsilon(lab, sa, sb)
            deltas.append(e["auc_diff"])
            variances.append(e["se_diff"] ** 2)
        cas = combine_across_seed(deltas, variances)
        cdl = combine_random_effects(deltas, variances)
        if np.isfinite(cas["epsilon_cell"]) and \
           abs(cas["delta_combined"] - delta_cell) <= cas["epsilon_cell"]:
            cov_as += 1
        if np.isfinite(cdl["epsilon_cell"]) and \
           abs(cdl["delta_combined"] - delta_cell) <= cdl["epsilon_cell"]:
            cov_dl += 1
        tau2_hats.append(cdl["tau2"])
        eps_as.append(cas["epsilon_cell"])
        eps_dl.append(cdl["epsilon_cell"])
    return {"n_seeds": n_seeds, "m": m, "n": n, "trials": trials,
            "delta_cell_true": delta_cell,
            "tau_true": tau_true, "tau2_true": tau_true ** 2,
            "mean_tau2_hat": float(np.mean(tau2_hats)),
            "across_seed": {"coverage": cov_as / trials,
                            "mean_epsilon_cell": float(np.mean(eps_as))},
            "dersimonian_laird": {"coverage": cov_dl / trials,
                                  "mean_epsilon_cell": float(np.mean(eps_dl))},
            "nominal": 0.95}


def check_d_floor() -> dict:
    """The within-seed DeLong-variance floor.

    On a deterministic axis every seed gives the SAME gap, so the
    across-seed sample variance is 0 -- a naive t-CI would report
    epsilon_cell = 0.  The floor at the mean within-seed DeLong
    variance keeps epsilon_cell > 0 whenever the within-trajectory
    uncertainty is real.
    """
    d_const = [0.25] * 50                       # identical gap, all seeds
    v_const = [0.009] * 50                      # real within-seed variance
    no_floor = combine_across_seed(d_const)
    floored = combine_across_seed(d_const, v_const)
    return {"epsilon_no_floor": no_floor["epsilon_cell"],
            "epsilon_with_floor": floored["epsilon_cell"],
            "var_floor": floored["var_floor"]}


# --------------------------------------------------------------------------
def main() -> int:
    rng = np.random.default_rng(20260522)
    run_ts = utc_now_iso()
    print(f"[verify_epsilon]  {run_ts}\n")

    print("CHECK A  --  DeLong SE vs stratified bootstrap")
    a = check_a_bootstrap(rng)
    print(f"  m={a['m']} n={a['n']}  over {a['trials']} datasets x 800 "
          f"bootstrap resamples")
    print(f"  Var_DeLong / Var_bootstrap : mean {a['mean_ratio_delong_over_bootstrap']:.3f}"
          f"  median {a['median_ratio']:.3f}  (target 1.0)\n")

    print("CHECK B  --  per-trajectory coverage of the 95% interval "
          "(true gap = 0)")
    b_small = check_b_coverage(rng, m=6, n=30)
    b_large = check_b_coverage(rng, m=30, n=120)
    print(f"  m=6  n=30  : coverage {b_small['empirical_coverage']:.3f}  "
          f"(nominal 0.95 ; the real harness setting)")
    print(f"  m=30 n=120 : coverage {b_large['empirical_coverage']:.3f}  "
          f"(nominal 0.95 ; normal approx. with more data)\n")

    print("CHECK C  --  cell-level coverage: across-seed t-CI vs "
          "DerSimonian-Laird")
    c = check_c_cell(rng)
    print(f"  cell = {c['n_seeds']} seeds, m={c['m']} n={c['n']}, "
          f"true cell gap {c['delta_cell_true']}, tau_true {c['tau_true']}")
    cas, cdl = c["across_seed"], c["dersimonian_laird"]
    print(f"  across-seed t-CI    : coverage {cas['coverage']:.3f}  "
          f"eps_cell {cas['mean_epsilon_cell']:.4f}   (nominal 0.95)")
    print(f"  DerSimonian-Laird   : coverage {cdl['coverage']:.3f}  "
          f"eps_cell {cdl['mean_epsilon_cell']:.4f}   (inverse-variance "
          f"-> biased for AUC)\n")

    print("CHECK D  --  within-seed DeLong-variance floor "
          "(deterministic axis)")
    dd = check_d_floor()
    print(f"  epsilon_cell : without floor {dd['epsilon_no_floor']:.4f}  "
          f"with floor {dd['epsilon_with_floor']:.4f}  "
          f"(var_floor {dd['var_floor']:.4f})\n")

    verdict = []
    verdict.append(("A bootstrap SE", 0.90 <=
                    a["mean_ratio_delong_over_bootstrap"] <= 1.10))
    verdict.append(("B coverage m=6", b_small["empirical_coverage"] >= 0.90))
    verdict.append(("B coverage m=30", b_large["empirical_coverage"] >= 0.93))
    verdict.append(("C across-seed coverage", cas["coverage"] >= 0.92))
    verdict.append(("D variance floor active",
                    dd["epsilon_no_floor"] == 0.0
                    and dd["epsilon_with_floor"] > 0.0))
    print("VERDICT")
    for name, ok in verdict:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    all_ok = all(ok for _, ok in verdict)
    print(f"\n  epsilon {'VERIFIED' if all_ok else 'NEEDS REVIEW'}")

    out = {"verify_timestamp_utc": run_ts, "check_a": a,
           "check_b_m6": b_small, "check_b_m30": b_large, "check_c": c,
           "check_d": dd,
           "verdict": {n: ok for n, ok in verdict}, "all_passed": all_ok}
    outdir = _HERE / "out"
    outdir.mkdir(exist_ok=True)
    fp = outdir / f"verify_epsilon_{run_ts.replace('-', '').replace(':', '')}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"[out ] {fp}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
