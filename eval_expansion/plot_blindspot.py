#!/usr/bin/env python3
"""Figure: the 1-D/5-D detectability gap vs DCI -- shows DCI routing is well-founded
(the gap where 5-D is needed lives at HIGH DCI, which DCI routes to 5-D; below tau it closes)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D, NSTEADY, NDRIFT, NSEED = 5, 400, 200, 40
HERE = Path(__file__).resolve().parent

def auc(y, s):
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    n1 = int(y.sum()); n0 = len(y)-n1
    return (ranks[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)

def vspread(): w = np.ones(D); return w/np.linalg.norm(w)   # hardest case for 1-D

def run(m, seed):
    r = np.random.default_rng(seed)
    steady = r.standard_normal((NSTEADY, D)); drift = m*vspread() + r.standard_normal((NDRIFT, D))
    d = np.vstack([steady, drift]) - steady.mean(0); lab = np.r_[np.zeros(NSTEADY), np.ones(NDRIFT)]
    dd = d[lab == 1]; C = dd.T@dd/len(dd); ev = np.clip(np.linalg.eigvalsh(C), 0, None); tot = ev.sum()
    dci = tot**2/np.sum(ev**2)
    a1 = max(max(auc(lab, d[:, j]), auc(lab, -d[:, j])) for j in range(D))
    Pinv = np.linalg.pinv(np.cov(d[lab == 0].T)+1e-6*np.eye(D))
    a5 = auc(lab, np.einsum("ij,jk,ik->i", d, Pinv, d))
    return dci, a1, a5

ms = np.linspace(1.0, 12.0, 24)
rows = np.array([[*np.array([run(m, s) for s in range(NSEED)]).mean(0)] for m in ms])
dci, a1, a5 = rows[:, 0], rows[:, 1], rows[:, 2]
o = np.argsort(dci)                     # sort by DCI ascending

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.axvspan(1.0, 1.5, color="#009E73", alpha=0.08)
ax.axvspan(1.5, dci.max(), color="#D55E00", alpha=0.08)
ax.plot(dci[o], a5[o], color="#D55E00", ls="-", lw=1.9, marker="s", ms=3, label="5-D Mahalanobis")
ax.plot(dci[o], a1[o], color="#0072B2", ls="--", lw=1.9, marker="o", ms=3, label="best 1-D axis")
ax.axvline(1.5, color="black", ls=":", lw=1.0)
ax.annotate(r"routing threshold $\tau\approx1.5$", xy=(1.5, 0.62), xytext=(2.4, 0.60),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
ax.text(1.22, 0.55, "route 1-D", ha="center", fontsize=8, color="#00795c")
ax.text(2.6, 0.55, "route 5-D", ha="center", fontsize=8, color="#a6490b")
ax.set_xlabel("Drift Complexity Index (DCI)  [spread, rank-one drift; iid noise]")
ax.set_ylabel("detection AUC"); ax.set_ylim(0.5, 1.02); ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(HERE/"fig_wellfounded.pdf", bbox_inches="tight")
print("saved fig_wellfounded.pdf")
print(f"{'DCI':>6}{'AUC_1D':>8}{'AUC_5D':>8}{'gap':>7}")
for i in o:
    print(f"{dci[i]:>6.2f}{a1[i]:>8.3f}{a5[i]:>8.3f}{a5[i]-a1[i]:>7.3f}")
