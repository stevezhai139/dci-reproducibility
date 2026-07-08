#!/usr/bin/env python3
"""fig2_pareto: cost vs accuracy of the three monitoring policies.
Colorblind-safe (Okabe-Ito) + distinct marker shapes (grayscale-safe); labels placed
inside the frame with no overlap. Data = the three aggregate points from cost_benefit (paper Sec 5.3)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
# (monitoring cost ms/window, mean detection AUC)
pts = {
    "always 1-D":   (0.008, 0.910, "o", "#0072B2"),
    "DCI selector": (0.83,  0.968, "D", "#009E73"),
    "always 5-D":   (2.23,  0.970, "s", "#E69F00"),
}
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for name, (x, y, mk, col) in pts.items():
    ax.scatter([x], [y], s=105, marker=mk, color=col, edgecolor="black",
               linewidth=0.8, zorder=3, label=name)
ax.set_xscale("log")
ax.set_xlim(0.005, 6.5)
ax.set_ylim(0.905, 0.983)
ax.set_xlabel("monitoring cost per window (ms, log scale)")
ax.set_ylabel("mean detection AUC")
ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)

# labels placed to stay inside the frame and clear of the markers/lines
ax.annotate("always 1-D", xy=(0.008, 0.910), xytext=(0.0125, 0.910),
            ha="left", va="center", fontsize=9)
ax.annotate("DCI selector", xy=(0.83, 0.968), xytext=(0.83, 0.9605),
            ha="center", va="top", fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.7))
ax.annotate("always 5-D", xy=(2.23, 0.970), xytext=(2.23, 0.9785),
            ha="center", va="bottom", fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.7))
fig.tight_layout()
fig.savefig(HERE / "fig2_pareto.pdf", bbox_inches="tight")
print("saved fig2_pareto.pdf")
