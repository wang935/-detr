import numpy as np

from paper_data import ARM_LABELS, figlib_event_summary
from paper_plot_style import save_fig, plt


df = figlib_event_summary()
models = ["YOLO26n", "RT-DETR-L", "InternImage-FRCNN"]
arms = ["baseline", "baseline_eqstep", "hardneg", "hardneg_sched"]

model_labels = {
    "YOLO26n": "YOLO26n",
    "RT-DETR-L": "RT-DETR-L",
    "InternImage-FRCNN": "InternImage",
}
model_colors = {
    "YOLO26n": "#4C78A8",
    "RT-DETR-L": "#E58932",
    "InternImage-FRCNN": "#59A14F",
}
model_markers = {"YOLO26n": "o", "RT-DETR-L": "s", "InternImage-FRCNN": "^"}
offsets = {"YOLO26n": -0.055, "RT-DETR-L": 0.0, "InternImage-FRCNN": 0.055}

x = np.arange(len(arms))

fig, (ax_far, ax_rec) = plt.subplots(
    2,
    1,
    figsize=(3.55, 3.05),
    sharex=True,
    gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.08},
)

for model in models:
    sub = df[df["model"].eq(model)].set_index("arm").loc[arms].reset_index()
    xpos = x + offsets[model]
    color = model_colors[model]

    ax_far.errorbar(
        xpos,
        sub["clustered_far_mean"],
        yerr=sub["clustered_far_std"],
        marker=model_markers[model],
        markersize=3.8,
        linewidth=1.15,
        capsize=2.0,
        color=color,
        label=model_labels[model],
        zorder=3,
    )
    ax_rec.errorbar(
        xpos,
        sub["event_recall_mean"],
        yerr=sub["event_recall_std"],
        marker=model_markers[model],
        markersize=3.5,
        linewidth=1.0,
        capsize=1.8,
        color=color,
        zorder=3,
    )

ax_far.text(0.01, 0.98, "a", transform=ax_far.transAxes, fontsize=8, fontweight="bold", va="top")
ax_rec.text(0.01, 0.98, "b", transform=ax_rec.transAxes, fontsize=8, fontweight="bold", va="top")

ax_far.set_ylabel("Clustered false alarms\nper proxy-hour")
ax_far.set_ylim(0, 3.25)
ax_far.set_xlim(-0.5, len(arms) - 0.5)
ax_far.grid(axis="y", color="#E6E6E6", linewidth=0.6)
ax_far.legend(frameon=False, ncol=1, loc="upper right", handlelength=1.8)

ax_rec.axhline(0.90, color="#555555", linestyle="--", linewidth=0.75, zorder=1)
ax_rec.text(0.99, 0.9004, "R0.90", transform=ax_rec.get_yaxis_transform(), color="#555555", fontsize=6.8, va="bottom", ha="right")
ax_rec.set_ylabel("Event recall")
ax_rec.set_ylim(0.884, 0.920)
ax_rec.set_yticks([0.890, 0.900, 0.910])
ax_rec.grid(axis="y", color="#E6E6E6", linewidth=0.6)
ax_rec.set_xticks(x)
ax_rec.set_xticklabels([ARM_LABELS[arm].replace("Scheduled ", "Sched. ") for arm in arms], rotation=18, ha="right")
ax_rec.set_xlabel("Training arm")

save_fig(fig, "fig3_figlib_event_alarm_curve")
