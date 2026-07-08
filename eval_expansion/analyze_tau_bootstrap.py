#!/usr/bin/env python3
"""
Paper 3C eval expansion (re-analysis of the verified cost_benefit_raw.csv, 450 runs).
Adds two full-paper pieces WITHOUT any new experiment:
  (1) tau-threshold sensitivity  -- how robust is the routing frontier tau ~ 1.5?
  (2) bootstrap CIs on the selector headline (retained AUC-gain % and monitoring cost %).
Cross-check: at tau=1.5 the routed selector must reproduce the paper's ~63% -> 1D,
  ~97% retained gain, ~37% cost. If it does, the re-analysis matches the harness.

Per-window detector costs are the platform constants already in the paper (Apple M4):
  1-D = 0.008 ms/window, 5-D = 2.23 ms/window.
"""
from __future__ import annotations
import numpy as np, pandas as pd, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
C1D, C5D = 0.008, 2.23           # ms/window (paper constants; M4)
RNG = np.random.default_rng(20260702)

d = pd.read_csv(HERE / "cost_benefit_raw.csv")
cells = list(d.groupby(["workload", "config"]).groups.keys())
always1d = d.auc_1d.mean()
always5d = d.auc_5d.mean()
gain = always5d - always1d

def selector_metrics(df):
    """Per-seed routing: 1-D if dci<tau else 5-D. Returns (routed_auc, frac1d, cost_ms)."""
    # placeholder; tau injected by caller via closure
    raise NotImplementedError

def eval_tau(df, tau):
    use1d = df.dci.values < tau
    routed = np.where(use1d, df.auc_1d.values, df.auc_5d.values)
    frac1d = use1d.mean()
    cost = frac1d * C1D + (1 - frac1d) * C5D
    routed_auc = routed.mean()
    retained = 100.0 * (routed_auc - always1d) / gain if gain > 0 else np.nan
    return dict(tau=round(float(tau), 3), frac_1d=round(float(frac1d), 4),
                routed_auc=round(float(routed_auc), 4),
                cost_ms=round(float(cost), 4),
                cost_pct_of_5d=round(100.0 * cost / C5D, 2),
                retained_gain_pct=round(float(retained), 2))

# ---- (1) tau sweep -------------------------------------------------------
grid = np.round(np.arange(1.00, 2.55, 0.05), 2)
sweep = pd.DataFrame([eval_tau(d, t) for t in grid])
sweep.to_csv(HERE / "tau_sweep.csv", index=False)

at15 = eval_tau(d, 1.5)
# plateau: tau range where retained>=95% AND cost stays low (<50% of 5-D)
plat = sweep[(sweep.retained_gain_pct >= 95.0) & (sweep.cost_pct_of_5d <= 50.0)]
plateau = (float(plat.tau.min()), float(plat.tau.max())) if len(plat) else (None, None)

# per-CELL routing (matches the paper's per-block router) -> empty-band check
cell_dci = d.groupby(["workload", "config"]).dci.mean().sort_values()
low = cell_dci[cell_dci < 1.5]; high = cell_dci[cell_dci >= 1.5]
empty_band = (round(float(low.max()), 3), round(float(high.min()), 3))  # any tau here routes identically

# ---- (2) stratified bootstrap CIs on the selector at tau=1.5 -------------
B = 10000
groups = [g.reset_index(drop=True) for _, g in d.groupby(["workload", "config"])]
ret_bs, cost_bs, frac_bs = [], [], []
for _ in range(B):
    parts = [g.iloc[RNG.integers(0, len(g), len(g))] for g in groups]
    bs = pd.concat(parts, ignore_index=True)
    m = eval_tau(bs, 1.5)
    ret_bs.append(m["retained_gain_pct"]); cost_bs.append(m["cost_pct_of_5d"]); frac_bs.append(m["frac_1d"])
def ci(a): return [round(float(np.percentile(a, 2.5)), 2), round(float(np.percentile(a, 97.5)), 2)]

# bootstrap CI on per-cell mean DCI (corroborate analytic ci95)
dci_ci = {}
for (w, c), g in d.groupby(["workload", "config"]):
    vals = g.dci.values
    bs = [vals[RNG.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
    dci_ci[f"{w}/{c}"] = dict(mean=round(float(vals.mean()), 3),
                              ci95=[round(float(np.percentile(bs, 2.5)), 3),
                                    round(float(np.percentile(bs, 97.5)), 3)])

summary = dict(
    n_runs=int(len(d)), n_cells=len(cells),
    always_1d_auc=round(float(always1d), 4), always_5d_auc=round(float(always5d), 4),
    gain=round(float(gain), 4),
    at_tau_1p5=at15,
    reproduces_paper=dict(frac_1d_paper=0.63, frac_1d_here=at15["frac_1d"],
                          retained_paper=97, retained_here=at15["retained_gain_pct"],
                          cost_pct_paper=37, cost_pct_here=at15["cost_pct_of_5d"],
                          cost_ms_paper=0.83, cost_ms_here=at15["cost_ms"]),
    plateau_tau_range_retained_ge95=plateau,
    per_cell_empty_DCI_band=empty_band,
    bootstrap_tau1p5_B=B,
    retained_gain_pct_CI95=ci(ret_bs),
    cost_pct_of_5d_CI95=ci(cost_bs),
    frac_1d_CI95=ci(frac_bs),
    per_cell_dci_bootstrap=dci_ci,
)
(HERE / "eval_expansion_results.json").write_text(json.dumps(summary, indent=2))

print("== VALIDATION (tau=1.5 vs paper) ==")
print(f"  frac->1D : {at15['frac_1d']*100:.1f}%  (paper 63%)")
print(f"  retained : {at15['retained_gain_pct']:.1f}%  (paper 97%)")
print(f"  cost     : {at15['cost_pct_of_5d']:.1f}% of 5-D = {at15['cost_ms']:.3f} ms  (paper 37% / 0.83 ms)")
print("== tau robustness ==")
print(f"  plateau (retained>=95%, cost<=50%): tau in {plateau}")
print(f"  per-cell empty DCI band (any tau here routes identically): {empty_band}")
print("== bootstrap 95% CIs (B=10000, stratified by cell) ==")
print(f"  retained gain % : {ci(ret_bs)}")
print(f"  cost % of 5-D   : {ci(cost_bs)}")
print(f"  frac -> 1D      : {ci(frac_bs)}")
print("\nsaved: tau_sweep.csv, eval_expansion_results.json")
