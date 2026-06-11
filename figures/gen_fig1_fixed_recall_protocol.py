from matplotlib.patches import FancyArrowPatch, Rectangle

from paper_plot_style import COLORS, save_fig, plt


def panel(ax, x, y, w, h, title, body, accent):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            linewidth=0.8,
            edgecolor="#D8D8D8",
            facecolor="white",
        )
    )
    ax.add_patch(Rectangle((x, y + h - 0.11), w, 0.11, linewidth=0, facecolor=accent))
    ax.text(x + 0.14, y + h - 0.27, title, ha="left", va="top", fontsize=7.4, fontweight="bold", color="#222222")
    ax.text(x + 0.14, y + 0.22, body, ha="left", va="bottom", fontsize=6.4, color="#333333", linespacing=1.18)


def arrow(ax, start, end, color="#6A6A6A"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=0.9,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


fig, ax = plt.subplots(figsize=(7.15, 2.35))
ax.set_axis_off()
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.3)

panels = [
    (0.25, 1.62, 2.15, 1.05, "1  Score", "Detector assigns\nfire/smoke scores", "#6C8EBF"),
    (3.0, 1.62, 2.35, 1.05, "2  Calibrate", "Select threshold\nfor R0.90 on calibration", "#E8B36A"),
    (5.95, 1.62, 2.35, 1.05, "3  Lock", "Apply the same\nthreshold on test data", "#7DB67A"),
    (8.9, 1.62, 2.55, 1.05, "4  Report", "Recall, FPR, FPPI\nand event alarm burden", "#8E6AAD"),
]

for item in panels:
    panel(ax, *item)

for x0, x1 in [(2.42, 2.98), (5.37, 5.93), (8.32, 8.88)]:
    arrow(ax, (x0, 2.15), (x1, 2.15))

ax.plot([3.12, 5.23], [2.86, 2.86], color="#E8B36A", linewidth=1.1)
ax.text(4.18, 3.02, "threshold fixed before test reporting", ha="center", va="bottom", fontsize=6.4, color="#333333")

ax.add_patch(Rectangle((1.65, 0.34), 8.7, 0.56, linewidth=0.7, edgecolor="#D8D8D8", facecolor="#FAFAFA"))
ax.text(
    6.0,
    0.62,
    "Stress axes applied after calibration: S1 visual degradation | S2 semantic distractors | S3 temporal aggregation",
    ha="center",
    va="center",
    fontsize=6.3,
    color="#333333",
)
arrow(ax, (6.0, 0.92), (6.0, 1.58), color="#8A8A8A")

save_fig(fig, "fig1_fixed_recall_protocol")
