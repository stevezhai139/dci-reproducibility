#!/usr/bin/env python3
"""fig3_tau_robustness: retained AUC-gain % and monitoring cost % vs the routing threshold tau."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = pd.read_csv(HERE / "tau_sweep.csv")

fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
l1, = ax1.plot(s.tau, s.retained_gain_pct, color="#0072B2", lw=1.9, ls="-", marker="o", ms=2.6,
               label="retained AUC-gain (%)")
ax1.set_xlabel(r"routing threshold $\tau$")
ax1.set_ylabel("retained AUC-gain (%)", color="#0072B2")
ax1.tick_params(axis="y", labelcolor="#0072B2")
ax1.set_ylim(0, 105)
ax1.axhline(95, color="#0072B2", ls=":", lw=0.8, alpha=0.6)

ax2 = ax1.twinx()
l2, = ax2.plot(s.tau, s.cost_pct_of_5d, color="#D55E00", lw=1.9, ls="--", marker="s", ms=2.6,
               label="monitoring cost (% of multi-D)")
ax2.set_ylabel("monitoring cost (% of multi-D)", color="#D55E00")
ax2.tick_params(axis="y", labelcolor="#D55E00")
ax2.set_ylim(0, 105)

# empty per-cell DCI band [1.411, 1.753] -> any tau here routes identically
ax1.axvspan(1.411, 1.753, color="gray", alpha=0.12, label="empty per-cell DCI band")
ax1.axvline(1.5, color="black", ls="--", lw=1.0)
ax1.annotate(r"$\tau\approx1.5$", xy=(1.5, 8), xytext=(1.62, 16),
             fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.7))

lines = [l1, l2]
ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower center",
           bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig(HERE / "fig3_tau_robustness.pdf", bbox_inches="tight")
print("saved fig3_tau_robustness.pdf")

# key values for the text
for t in (1.0, 1.2, 1.41, 1.5, 1.75, 2.0):
    r = s.iloc[(s.tau - t).abs().idxmin()]
    print(f"  tau={r.tau:.2f}: retained={r.retained_gain_pct:5.1f}%  cost={r.cost_pct_of_5d:5.1f}%  frac1D={r.frac_1d:.2f}")
