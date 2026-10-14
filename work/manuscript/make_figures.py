"""Generate consistent publication figures as vector PDFs and 600-dpi PNGs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ["CALTRANSFER_ROOT"])
FIG = ROOT / "work" / "manuscript" / "figures_revised"
FIG.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "outputs" / "xuetangx_experiment4"
SUP = DATA / "supplement"
PRIMARY = pd.read_csv(DATA / "primary_time_model_summary.csv")
COMMON = pd.read_csv(SUP / "common_pair_summary.csv")
COMMON_CI = pd.read_csv(SUP / "common_pair_bootstrap.csv")

BLUE, ORANGE, GRAY, NAVY = "#2166ac", "#d95f0e", "#718096", "#173453"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.2,
    "axes.labelsize": 9.2, "axes.titlesize": 9.6, "legend.fontsize": 8.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, number):
    fig.savefig(FIG / f"figure{number}.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIG / f"figure{number}.png", dpi=600, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


# Figure 2: primary Brier gain decomposition; fixed model and landmark order.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.2), sharey=True)
for ax, (model, label) in zip(axes, (("logistic", "A  LR"), ("lightgbm", "B  LightGBM"))):
    g = PRIMARY[(PRIMARY.base_model.eq(model)) & PRIMARY.day.isin([7, 14, 28])].sort_values("day")
    for col, name, color, marker in [
        ("G_S_macro", r"Source OOF ($G_S$)", GRAY, "s"),
        ("G_T_macro", r"Historical transfer ($G_T$)", ORANGE, "D"),
        ("G_T_star_macro", r"Target OOF ($G_T^*$)", BLUE, "o")]:
        ax.plot(g.day, g[col], marker=marker, markersize=4.2, linewidth=1.6,
                color=color, label=name)
    ax.axhline(0, color="#b5bec8", linewidth=.8)
    ax.set(xticks=[7, 14, 28], xlabel="Observed landmark (days)")
    ax.text(.02, 1.03, label, transform=ax.transAxes, va="bottom", color=NAVY, weight="bold")
    ax.grid(axis="y", alpha=.15)
axes[0].set_ylabel("Brier gain over raw probabilities")
axes[1].legend(frameon=False, loc="upper right", fontsize=7.2)
fig.tight_layout(w_pad=1.2)
save(fig, 2)


# Figure 3: target intercept-only versus full-Platt calibration.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), sharey=True)
for ax, (model, label) in zip(axes, (("logistic", "A  LR"), ("lightgbm", "B  LightGBM"))):
    g = PRIMARY[(PRIMARY.base_model.eq(model)) & PRIMARY.day.isin([7, 14, 28])].sort_values("day")
    x = np.arange(3); width = .34
    ax.bar(x-width/2, g.G_T_star_intercept_macro, width, color="#94b9df", label="Intercept only")
    ax.bar(x+width/2, g.G_T_star_macro, width, color=BLUE, label="Full Platt")
    ax.set(xticks=x, xticklabels=[str(v) for v in g.day], xlabel="Observed landmark (days)")
    ax.text(.02, 1.03, label, transform=ax.transAxes, va="bottom", color=NAVY, weight="bold")
    ax.grid(axis="y", alpha=.15)
axes[0].set_ylabel("Target OOF Brier gain")
axes[1].legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.2)
save(fig, 3)


# Figure 4: post-primary fixed-pair temporal analysis with cluster intervals.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.2))
for ax, (measure, panel, ylabel) in zip(axes, (
    ("A", "A", "Pair-macro Brier gain"),
    ("target_extra_slope_gain", "B", "Pair-macro Brier gain"))):
    for model, label, color, marker in (("logistic", "LR", BLUE, "o"),
                                        ("lightgbm", "LightGBM", ORANGE, "s")):
        g = COMMON[COMMON.base_model.eq(model)].sort_values("day")
        c = COMMON_CI[(COMMON_CI.base_model.eq(model)) & COMMON_CI.measure.eq(measure)].sort_values("day")
        v = g[measure].to_numpy(); low, high = c.ci_low.to_numpy(), c.ci_high.to_numpy()
        ax.errorbar(g.day, v, yerr=[v-low, high-v], marker=marker, color=color,
                    markersize=4.2, linewidth=1.6, capsize=2.4, label=label)
    ax.set(xticks=[7, 14, 28], xlabel="Observed landmark (days)", ylabel=ylabel)
    ax.text(.02, 1.03, panel, transform=ax.transAxes, va="bottom", color=NAVY, weight="bold")
    ax.grid(axis="y", alpha=.15)
axes[1].legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.1)
save(fig, 4)
print({"empirical_figures": 3, "pdfs": [str(p) for p in sorted(FIG.glob('*.pdf'))]})
