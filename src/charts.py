"""Charts for the README, built from output/results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"raw": "#2a78d6", "park": "#eb6834", "coors": "#1baf7a"}
LABELS = {"raw": "Plain Marcel", "park": "Park-adjusted", "coors": "Park + Rockies multipliers"}
COMP_NAMES = {"1B": "Singles", "2B": "Doubles", "3B": "Triples", "HR": "Home runs",
              "BB": "Walks", "HBP": "Hit by pitch", "K": "Strikeouts"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
res = json.load(open(OUT / "results.json"))


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# 1. Coors Field component park factors
pf = res["coors_park_factor_2026"]
comps = list(COMP_NAMES)[::-1]
vals = [pf[c] for c in comps]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh([COMP_NAMES[c] for c in comps], [v - 1 for v in vals], left=1, color=SERIES["raw"], height=0.6)
ax.axvline(1, color=INK2, lw=1)
for i, v in enumerate(vals):
    ax.text(v + (0.015 if v >= 1 else -0.015), i, f"{v:.2f}", va="center",
            ha="left" if v >= 1 else "right", color=INK, fontsize=10)
ax.set_xlim(0.75, 1.85)
ax.set_xlabel("Park factor (1.00 = average park)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
style(ax)
ax.set_title("Coors Field inflates triples, singles and doubles far more than homers",
             loc="left", color=INK, fontsize=12.5, pad=12)
fig.text(0.01, 0.01, "Component park factors, 2023–2025 games, shrunk toward 1. Source: Retrosheet.",
         color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "coors_park_factors.png", dpi=180)

# 2. Projection bias by group
groups = ["all hitters", "Rockies hitters", "joined Rockies", "left Rockies"]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
width = 0.26
for j, v in enumerate(SERIES):
    xs = [i + (j - 1) * (width + 0.02) for i in range(len(groups))]
    ys = [res["results"][v][g]["bias"] * 1000 for g in groups]
    ax.bar(xs, ys, width=width, color=SERIES[v], label=LABELS[v])
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{g[0].upper() + g[1:]}\n(n={res['results']['raw'][g]['n']})" for g in groups])
ax.set_ylabel("Actual − projected wOBA (points)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
style(ax)
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
ax.set_title("Park adjustment removes most of the bias for hitters who change teams",
             loc="left", color=INK, fontsize=12.5, pad=12)
fig.text(0.01, 0.01, "Held-out seasons 2021–2025, hitters with 200+ PA. 1 point = .001 wOBA.",
         color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "projection_bias.png", dpi=180)
print("saved")
