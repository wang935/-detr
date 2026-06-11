import numpy as np

from paper_data import ARM_LABELS, combined_fixed_recall_summary
from paper_plot_style import COLORS, save_fig, plt


summary = combined_fixed_recall_summary()
r090 = summary[summary["target_recall"].eq(0.90)].copy()

models = ["YOLO26n", "RT-DETR-L", "InternImage-FRCNN"]
model_labels = {
    "YOLO26n": "YOLO26n",
    "RT-DETR-L": "RT-DETR-L",
    "InternImage-FRCNN": "InternImage",
}
arms = ["baseline", "baseline_eqstep", "hardneg", "hardneg_sched"]
markers = {"baseline": "o", "baseline_eqstep": "s", "hardneg": "^", "hardneg_sched": "D"}
offsets = {"baseline": 0.24, "baseline_eqstep": 0.08, "hardneg": -0.08, "hardneg_sched": -0.24}

fig, ax = plt.subplots(figsize=(3.55, 2.85))
y_base = np.arange(len(models))[::-1]

for arm in arms:
    xs = []
    xerr = []
    ys = []
    for idx, model in enumerate(models):
        row = r090[(r090["model"].eq(model)) & (r090["arm"].eq(arm))].iloc[0]
        mean = float(row["FPR_mean"])
        std = float(row["FPR_std"])
        xs.append(mean)
        xerr.append([[max(mean - max(mean - std, 1e-4), 1e-4)], [std]])
        ys.append(y_base[idx] + offsets[arm])

    lower = [max(x - e[0][0], 1e-4) for x, e in zip(xs, xerr)]
    upper = [e[1][0] for e in xerr]
    xerr_array = np.array([np.array(xs) - np.array(lower), np.array(upper)])
    ax.errorbar(
        xs,
        ys,
        xerr=xerr_array,
        fmt=markers[arm],
        markersize=4.4,
        linewidth=0,
        elinewidth=1.0,
        capsize=2.2,
        color=COLORS[arm],
        markeredgecolor="white",
        markeredgewidth=0.35,
        label=ARM_LABELS[arm].replace("Scheduled ", "Sched. "),
        zorder=3,
    )

ax.set_xscale("log")
ax.set_xlim(0.001, 0.35)
ax.set_xticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
ax.set_xticklabels(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3"])
ax.set_xlabel("False-positive rate at R0.90 (log scale)")
ax.set_yticks(y_base)
ax.set_yticklabels([model_labels[m] for m in models])
ax.set_ylim(-0.55, len(models) - 0.45)
ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, which="both")
ax.tick_params(axis="y", length=0)

ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    columnspacing=1.0,
    handletextpad=0.35,
)

for y in y_base:
    ax.axhline(y, color="#F1F1F1", linewidth=0.6, zorder=0)

save_fig(fig, "fig2_r090_false_alarm_burden")
