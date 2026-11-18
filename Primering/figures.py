"""Figure fig:prime_ring_shape: rejection share (bars) and mean pad bytes (lines) by churn level."""
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COL = {"E-health": "#0072B2", "OSN-Facebook": "#D55E00", "OSN-Twitter": "#009E73"}
LIGHT = {"E-health": "#B3D4EC", "OSN-Facebook": "#F4C7A6", "OSN-Twitter": "#A8DCC9"}
MK = {"E-health": "o", "OSN-Facebook": "s", "OSN-Twitter": "^"}


def shape_figure(rows, order, path):
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "axes.linewidth": 0.9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    churns = sorted({r["churn"] for r in rows})
    x = np.arange(len(churns)); w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ax2 = ax.twinx()
    pmax, rmax = 1.0, 1.0
    for j, d in enumerate(order):
        rr = {r["churn"]: r for r in rows if r["domain"] == d}
        rej = [rr[c]["reject_pct"] for c in churns]
        pad = [rr[c]["pad_B"] for c in churns]
        off = (j - (len(order) - 1) / 2) * w
        ax.bar(x + off, rej, w, color=LIGHT[d], edgecolor=COL[d], lw=0.9, zorder=1)
        ax2.plot(x + off, pad, color=COL[d], marker=MK[d], lw=1.6, ms=5, mec="white", mew=0.7, zorder=3)
        rmax = max(rmax, max(rej))
        pmax = max(pmax, max(p for p in pad if not math.isnan(p)) if any(not math.isnan(p) for p in pad) else 1)
    ax.set_xticks(x); ax.set_xticklabels([f"{c:.2f}" for c in churns])
    ax.set_xlabel("Churn"); ax.set_ylabel("Rejected (%), bars"); ax2.set_ylabel(r"Mean pad $\bar b$ (B), lines")
    ax.set_ylim(0, min(105, max(10, 1.25 * rmax))); ax2.set_ylim(0, 1.3 * pmax)
    ax.set_zorder(ax2.get_zorder() - 1); ax.patch.set_visible(False)
    ax.grid(True, axis="y", color="0.9", lw=0.6)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    h = [Line2D([0], [0], color=COL[d], marker=MK[d], lw=1.6, ms=5, mec="white", mew=0.7) for d in order]
    fig.legend(h, order, loc="upper center", ncol=len(order), frameon=False, bbox_to_anchor=(0.5, 1.02), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
