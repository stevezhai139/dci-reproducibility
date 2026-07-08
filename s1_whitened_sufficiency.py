#!/usr/bin/env python3
"""s1_whitened_sufficiency.py -- S1 in full: does the WHITENED map keep the regime story?

Completes the S1 probe (whitened_dci.py). For every (workload, config, seed) cell of the
paper's regime map it computes, on the identical trajectories (same stable seeds, same
kernel path, same steady/drift algebra as cost_benefit.analyse):

  1. raw DCI                        -- fidelity gate: must reproduce Table 2 per digit;
  2. whitened DCI_w = PR(S^-1/2 C S^-1/2)  at the paper's ridge (1e-6);
  3. detector AUCs on the same windows:
        multi-D   : Mahalanobis ||z||^2 in the whitened space (the paper's detector),
        w-1D      : the WHITENED matched filter -- squared projection on the dominant
                    whitened drift mode (the strictly-more-powerful 1-D detector the
                    theorem anticipates; also the detector the live gate deploys),
        raw-1D    : the best single raw axis (the regime map's 1-D detector);
  4. the multi-D-over-w-1D AUC gap with DeLong variance per seed, pooled by the SAME
     delta rule as the regime map (combine_across_seed + delta_ref = epsilon(N_REF=15))
     -> per-cell verdict: is the whitened matched filter sufficient?
  5. ridge sensitivity: DCI_w recomputed at ridge in {1e-8, 1e-6, 1e-4, 1e-2}.

Hypotheses being tested (from the locked whitened_dci.py probe of 2026-07-08):
  H1 cells whose DCI_w collapses toward 1 (volume cells ~1.007; JOB mixed 1.082) should
     have the whitened filter RECOVER the multi-D AUC (sufficient);
  H2 the cell whose DCI_w stays high (pgbench mixed 1.753) should stay insufficient --
     the trichotomy applied in the whitened space;
  H3 TPC-H mixed (DCI_w 1.455, straddling tau) is the informative borderline case;
  H4 DCI_w is stable across reasonable ridges (else report where it is not).

Usage:  python3 s1_whitened_sufficiency.py <path-to-repro_3c> [seeds=50]
Writes: s1_whitened_sufficiency.csv (per seed) + s1_whitened_sufficiency_summary.json
"""
import sys, json, csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(REPRO / "geometry_E0"))

import cost_benefit as cb                              # the paper's own pipeline
from delong import delong_gap_variances, combine_across_seed
from seed_analysis import epsilon_at

SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
RIDGES = [1e-8, 1e-6, 1e-4, 1e-2]
RIDGE_PAPER = 1e-6                                     # cost_benefit convention
N_REF = 15                                             # matches the regime map
OUT_CSV = HERE / "s1_whitened_sufficiency.csv"
OUT_JSON = HERE / "s1_whitened_sufficiency_summary.json"


def pr(M):
    ev = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    tot, f2 = float(ev.sum()), float((ev ** 2).sum())
    return (tot * tot) / f2 if tot > 0 and f2 > 0 else float("nan")


def inv_sqrt(S):
    ev, V = np.linalg.eigh(S)
    ev = np.clip(ev, 1e-12, None)
    return V @ np.diag(ev ** -0.5) @ V.T


def main() -> int:
    makers = {"tpch": cb.tpch_pool,
              "job": lambda: cb.job_pool(REPRO / "data" / "job" / "queries"),
              "pgbench": cb.pgbench_pool}
    rows = []
    for wl, mk in makers.items():
        pool = mk()
        for cfg in cb.CONFIGS:
            for s in range(SEEDS):
                seed = cb.stable_seed(wl, cfg, s)
                windows, didx = cb.build_trajectory(pool, cfg, seed)
                fv, sc = cb.kernel_adjacent(windows)
                n = len(fv)
                lab = np.array([1 if (i + 1) in didx else 0 for i in range(n)])
                if lab.sum() < 2 or (lab == 0).sum() < 2:
                    continue
                nd = lab == 0
                d = fv - fv[nd].mean(axis=0)
                # raw DCI (fidelity; identical algebra to whitened_dci.py)
                driftd = d[lab == 1]
                C_raw = (driftd.T @ driftd) / len(driftd)
                dci_raw = pr(C_raw)
                # steady-noise covariance at the paper's ridge
                Sig = np.cov(d[nd].T) + RIDGE_PAPER * np.eye(d.shape[1])
                W = inv_sqrt(Sig)
                z = d @ W                                   # whitened deviations
                zd = z[lab == 1]
                C_w = (zd.T @ zd) / len(zd)
                dci_w = pr(C_w)
                evw, Vw = np.linalg.eigh(C_w)
                v1 = Vw[:, int(np.argmax(evw))]             # dominant whitened mode
                # detectors on every window
                maha = np.einsum("ij,ij->i", z, z)          # multi-D (whitened norm^2)
                w1d = (z @ v1) ** 2                          # whitened matched filter
                raw_best = max(cb.auc(lab, 1.0 - fv[:, j]) for j in range(fv.shape[1]))
                auc_md = cb.auc(lab, maha)
                auc_w1 = cb.auc(lab, w1d)
                # DeLong gap (multi-D over whitened-1D), one covariance
                gv = delong_gap_variances(lab, maha, w1d[:, None])
                gap = gv["per_dim"][0]["gap"]
                vgap = gv["per_dim"][0]["var_gap"]
                # ridge sweep on DCI_w
                sweep = {}
                for r in RIDGES:
                    Wr = inv_sqrt(np.cov(d[nd].T) + r * np.eye(d.shape[1]))
                    zr = d[lab == 1] @ Wr
                    sweep[r] = pr((zr.T @ zr) / len(zr))
                rows.append({
                    "workload": wl, "config": cfg, "seed": s,
                    "dci_raw": round(dci_raw, 6), "dci_w": round(dci_w, 6),
                    "auc_multiD": round(auc_md, 6), "auc_w1D": round(auc_w1, 6),
                    "auc_raw1D_best": round(raw_best, 6),
                    "gap_md_w1": round(gap, 6), "vargap_md_w1": vgap,
                    **{f"dci_w_r{r:g}": round(sweep[r], 6) for r in RIDGES},
                })
        print(f"[done] {wl}", flush=True)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- per-cell summary with the regime map's delta rule --------------------
    cells = {}
    for r in rows:
        cells.setdefault((r["workload"], r["config"]), []).append(r)
    summ = {}
    hdr = (f"{'cell':26s}{'rawDCI':>8s}{'DCI_w':>8s}{'mD':>7s}{'w1D':>7s}{'raw1D':>7s}"
           f"{'gap':>8s}{'delta':>8s}{'suff?':>7s}   ridge spread (1e-8..1e-2)")
    print("\n" + hdr); print("-" * len(hdr))
    for (wl, cfg), rs in sorted(cells.items()):
        raw = np.mean([x["dci_raw"] for x in rs])
        dw = np.mean([x["dci_w"] for x in rs])
        md = np.mean([x["auc_multiD"] for x in rs])
        w1 = np.mean([x["auc_w1D"] for x in rs])
        r1 = np.mean([x["auc_raw1D_best"] for x in rs])
        cc = combine_across_seed([x["gap_md_w1"] for x in rs],
                                 [x["vargap_md_w1"] for x in rs])
        gap, sig2 = cc["delta_combined"], cc["sigma2"]
        delta = epsilon_at(N_REF, sig2)
        suff = bool(np.isfinite(gap) and gap <= delta)
        spread = [np.mean([x[f"dci_w_r{r:g}"] for x in rs]) for r in RIDGES]
        print(f"{wl:8s}{cfg:18s}{raw:8.3f}{dw:8.3f}{md:7.3f}{w1:7.3f}{r1:7.3f}"
              f"{gap:8.4f}{delta:8.4f}{'YES' if suff else 'no':>7s}   "
              + " ".join(f"{v:.3f}" for v in spread))
        summ[f"{wl}/{cfg}"] = {
            "dci_raw": round(float(raw), 4), "dci_w": round(float(dw), 4),
            "auc_multiD": round(float(md), 4), "auc_w1D": round(float(w1), 4),
            "auc_raw1D_best": round(float(r1), 4),
            "gap_md_over_w1D": round(float(gap), 5),
            "delta_ref_Nref15": round(float(delta), 5), "w1D_sufficient": suff,
            "dci_w_by_ridge": {f"{r:g}": round(float(v), 4)
                               for r, v in zip(RIDGES, spread)},
            "n_seeds": len(rs),
        }
    json.dump(summ, open(OUT_JSON, "w"), indent=1)
    print(f"\n[out] {OUT_CSV.name} + {OUT_JSON.name}")
    print("[note] fidelity gate: the rawDCI column must reproduce Table 2 per digit "
          "before any interpretation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
