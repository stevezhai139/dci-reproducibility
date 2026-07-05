#!/usr/bin/env python3
"""
make_figures.py
===============

Build Paper 3C's two CIDR figures from an official cost_benefit run --
no re-run of the experiment, just a re-plot of the recorded CSV/JSON.

  Fig 1  fig1_regime.pdf  -- detection AUC vs DCI (1-D x / 5-D o)
  Fig 2  fig2_pareto.pdf  -- cost / accuracy of the three policies

Colour-blind-safe Okabe-Ito palette; series are distinguished by
marker shape as well as colour (CIDR I&D guideline -- colour is not
load-bearing). Sized for the ACM sigconf single column.

RUN
  python make_figures.py [--raw out/<id>/cost_benefit_raw.csv]
  # default: the newest out/*/cost_benefit_raw.csv beside this script
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

_HERE = Path(__file__).resolve().parent
# Okabe-Ito: no green/red pair, no blue/purple pair.
PALETTE = {"template_only": "#E69F00",     # orange
           "volume_only":   "#0072B2",     # blue
           "mixed":         "#009E73"}     # bluish-green
# Marker SHAPE also encodes the configuration, so no information is
# carried by colour alone (CIDR / ACM I&D accessibility guideline).
SHAPES  = {"template_only": "o", "volume_only": "s", "mixed": "^"}
TAU = 1.5

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()
    if args.raw:
        raw = Path(args.raw).expanduser().resolve()
    else:
        cands = sorted((_HERE / "out").glob("*/cost_benefit_raw.csv"))
        if not cands:
            print("[ERR] no out/*/cost_benefit_raw.csv", file=sys.stderr)
            return 2
        raw = cands[-1]
    run_dir = raw.parent
    R = pd.read_csv(raw)
    runj = json.loads((run_dir / "cost_benefit_run.json").read_text())
    print(f"[in ] {raw}")

    # ---- Fig 1: detection AUC vs DCI ------------------------------------
    fig, ax = plt.subplots(figsize=(3.34, 2.35))
    for cfg, col in PALETTE.items():
        g = R[R.config == cfg]
        m = SHAPES[cfg]
        # 5-D = open marker, 1-D = filled marker; shape = configuration.
        ax.scatter(g["dci"], g["auc_5d"], marker=m, s=16,
                   facecolors="none", edgecolors=col, linewidths=0.7,
                   label=cfg.replace("_", " "))
        ax.scatter(g["dci"], g["auc_1d"], marker=m, s=13, facecolors=col,
                   edgecolors=col, linewidths=0.3, alpha=0.55)
    ax.axvline(TAU, ls="--", color="#444", lw=0.8)
    ax.text(TAU + 0.04, 0.52, r"$\tau\!\approx\!1.5$", fontsize=6.5,
            color="#444")
    ax.set_xlabel("Drift Complexity Index (DCI)")
    ax.set_ylabel("detection AUC")
    ax.set_ylim(0.45, 1.04)
    ax.legend(fontsize=6, frameon=False, loc="lower left",
              handletextpad=0.2, borderpad=0.2)
    fig.tight_layout(pad=0.25)
    fig.savefig(run_dir / "fig1_regime.pdf")
    plt.close(fig)

    # ---- Fig 2: cost / accuracy Pareto ----------------------------------
    pol = runj["policies"]
    fig, ax = plt.subplots(figsize=(3.34, 2.35))
    pts = [("always 1-D", pol["always_1D"], "#0072B2", (7, -2)),
           ("always 5-D", pol["always_5D"], "#E69F00", (-10, 6)),
           ("DCI selector", pol["DCI_selector"], "#009E73", (8, -10))]
    for name, m, col, off in pts:
        ax.scatter([m["mean_cost_ms"]], [m["mean_auc"]], s=80, color=col,
                   edgecolor="black", linewidths=0.6, zorder=3)
        ax.annotate(name, (m["mean_cost_ms"], m["mean_auc"]),
                    textcoords="offset points", xytext=off, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("monitoring cost per window (ms, log scale)")
    ax.set_ylabel("mean detection AUC")
    fig.tight_layout(pad=0.25)
    fig.savefig(run_dir / "fig2_pareto.pdf")
    plt.close(fig)

    print(f"[out] {run_dir / 'fig1_regime.pdf'}")
    print(f"[out] {run_dir / 'fig2_pareto.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
