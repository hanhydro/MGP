#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Helvetica"
mpl.rcParams["mathtext.it"] = "Helvetica:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "figures"
FS = 7.0
FIXED, VARIED, SCORE, AGG = "#dfe8f3", "#fbe9d0", "#e4f1e1", "#ece4f2"
Y0, YS = 0.10, 0.90


def ty(y):
    return (y - Y0) / YS


def box(ax, x, y, w, h, text, fc, fs=FS, lw=0.6, weight="normal"):
    y, h = ty(y), h / YS
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc=fc, ec="0.35", lw=lw, transform=ax.transAxes, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            transform=ax.transAxes, weight=weight, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1):
    y0, y1 = ty(y0), ty(y1)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.3", shrinkA=0, shrinkB=0, mutation_scale=7))


fig, axes = plt.subplots(1, 3, figsize=(5.69, 3.15 * YS))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04)
for ax, lab in zip(axes, "abc"):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.98, f"({lab})", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontfamily="Arial", fontsize=8)

ax = axes[0]
box(ax, 0.03, 0.80, 0.94, 0.11, "Seed $s$: training and held-out sets,\nnetwork initialization", FIXED)
arrow(ax, 0.50, 0.80, 0.50, 0.735)
box(ax, 0.01, 0.60, 0.47, 0.13, "Decoder $\\mathbf{D}$\n(per seed and cell)", FIXED, fs=FS - 0.6)
box(ax, 0.52, 0.60, 0.47, 0.13, "POD $\\mathbf{V}_k$, $\\mathbf{u}_c$\n(per seed and cell)", FIXED, fs=FS - 0.6)
arrow(ax, 0.25, 0.60, 0.25, 0.47)
arrow(ax, 0.75, 0.60, 0.75, 0.47)
rows = [("Equation", "Galerkin | LSPG"), ("Residual metric", "raw $\\mathbf{R}$ | $\\mathbf{B}_0\\mathbf{R}$"),
        ("Algorithm", "Newton | LM"), ("Initial guess", "mean | nearest snapshot")]
y = 0.40
for name, opts in rows:
    box(ax, 0.04, y, 0.92, 0.075, f"{name}:  {opts}", VARIED, fs=FS - 0.4)
    y -= 0.095

ax = axes[1]
box(ax, 0.10, 0.80, 0.80, 0.11, "Arm solves each held-out case\nfrom the shared candidate state", SCORE)
arrow(ax, 0.50, 0.80, 0.50, 0.71)
box(ax, 0.10, 0.60, 0.80, 0.10, "Final iterate", SCORE)
arrow(ax, 0.30, 0.60, 0.22, 0.50)
arrow(ax, 0.70, 0.60, 0.78, 0.50)
box(ax, 0.02, 0.37, 0.44, 0.13, "criterion met", SCORE, fs=FS - 0.4)
box(ax, 0.54, 0.37, 0.44, 0.13, "criterion missed\n(iteration cap)", SCORE, fs=FS - 0.4)
arrow(ax, 0.24, 0.37, 0.40, 0.27)
arrow(ax, 0.76, 0.37, 0.60, 0.27)
box(ax, 0.08, 0.13, 0.84, 0.14, "finite return scored by\n$e = ||\\mathbf{u}_{\\mathrm{ref}} - \\mathbf{u}_{\\mathrm{rom}}|| \\, / \\, ||\\mathbf{u}_{\\mathrm{ref}}||$", SCORE)

ax = axes[2]
ladder = [
    ("12 held-out samples", "cell mean $\\mathcal{E}_{sga}$"),
    ("8 cells (1D) or 1 regime (2D)", "seed statistic\n$D_s = \\mathrm{mean}_g \\log(\\mathcal{E}_{sgA}/\\mathcal{E}_{sgB})$"),
    ("20 unused seeds", "two-sided exact sign test on $D_s$"),
    ("contrast family", "Holm adjustment;\ngeometric ratio $\\exp(\\bar{D})$"),
]
y, step = 0.80, 0.225
for level, stat in ladder:
    box(ax, 0.04, y, 0.92, 0.13, f"{level}\n$\\rightarrow$ {stat}", AGG, fs=FS - 0.4)
    if y > 0.3:
        arrow(ax, 0.50, y, 0.50, y - (step - 0.13))
    y -= step

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "figure1_protocol_schematic.pdf")
fig.savefig(OUT / "figure1_protocol_schematic.png", dpi=200)
print("saved", OUT / "figure1_protocol_schematic.pdf")
