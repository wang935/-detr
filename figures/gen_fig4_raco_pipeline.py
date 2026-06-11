from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_plot_style import save_fig, plt


def panel(ax, x, y, w, h, edge="#D9D9D9", fc="#FAFAFA"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.020,rounding_size=0.040",
            linewidth=0.65,
            edgecolor=edge,
            facecolor=fc,
        )
    )


def box(ax, x, y, w, h, title, body, color, fc="#FFFFFF"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.030",
            linewidth=0.75,
            edgecolor=color,
            facecolor=fc,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.16),
            w,
            0.16,
            boxstyle="round,pad=0.018,rounding_size=0.030",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + 0.10,
        y + h - 0.245,
        title,
        ha="left",
        va="top",
        fontsize=7.15,
        fontweight="bold",
        color="#1F1F1F",
    )
    ax.text(
        x + 0.10,
        y + 0.16,
        body,
        ha="left",
        va="bottom",
        fontsize=5.95,
        color="#333333",
        linespacing=1.20,
    )


def pill(ax, x, y, w, text, color, fc="#F7F7F7", h=0.29):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.085",
            linewidth=0.7,
            edgecolor=color,
            facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.65, color="#222222")


def arrow(ax, start, end, color="#6D6D6D"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.80,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


fig, ax = plt.subplots(figsize=(7.2, 2.85))
ax.set_axis_off()
ax.set_xlim(0, 12.7)
ax.set_ylim(0, 3.9)

colors = {
    "score": "#5B86B0",
    "family": "#7AA874",
    "order": "#D49A4A",
    "test": "#9B6AAE",
    "report": "#4F4F4F",
}

fills = {
    "score": "#F6FAFE",
    "family": "#F7FBF6",
    "order": "#FFF8EE",
    "test": "#FCF8FF",
    "report": "#F8F8F8",
}

steps = [
    (0.28, 2.27, 1.82, 0.95, "Frozen scores", "Detector outputs\nfire/smoke scores", colors["score"], fills["score"]),
    (2.52, 2.27, 1.90, 0.95, "Family unit", "Group frames by\nincident family", colors["family"], fills["family"]),
    (4.84, 2.27, 2.02, 0.95, r"$D_{\mathrm{order}}$", "Rank candidate\nalarm rules", colors["order"], fills["order"]),
    (7.24, 2.27, 2.02, 0.95, r"$D_{\mathrm{pval}}$", "Test late/missed\nalarm risk", colors["test"], fills["test"]),
    (9.64, 2.27, 2.30, 0.95, "Held-out report", "Risk and burden\nreported separately", colors["report"], fills["report"]),
]

for item in steps:
    box(ax, *item)

for start_x, end_x in [(2.12, 2.50), (4.44, 4.82), (6.88, 7.22), (9.28, 9.62)]:
    arrow(ax, (start_x, 2.75), (end_x, 2.75))

ax.text(0.28, 3.56, "RACO control layer", ha="left", va="center", fontsize=8.1, fontweight="bold", color="#222222")
ax.plot([0.28, 11.94], [3.38, 3.38], color="#D0D0D0", linewidth=0.7)

ax.plot([7.05, 7.05], [1.82, 2.18], color="#B8B8B8", linewidth=0.7)
ax.text(
    7.05,
    1.72,
    "No reuse of ordering events for p-value evidence",
    ha="center",
    va="top",
    fontsize=5.8,
    color="#444444",
)

box(
    ax,
    0.55,
    0.48,
    2.35,
    0.85,
    "Alarm rule",
    r"$\lambda=(\tau_g,k,n)$" + "\n" + "threshold, count, window",
    "#B8B8B8",
    fc="#FBFBFB",
)
arrow(ax, (3.00, 0.91), (3.40, 0.91), color="#8A8A8A")

panel(ax, 3.45, 0.34, 5.28, 1.05)
ax.text(3.62, 1.25, "Held-out family evidence", ha="left", va="center", fontsize=5.9, fontweight="bold", color="#333333")

pill(ax, 3.72, 0.88, 1.16, "event recall", "#7AA874", fc="#F7FBF6")
pill(ax, 5.08, 0.88, 1.20, "late/missed", "#9B6AAE", fc="#FCF8FF")
pill(ax, 6.48, 0.88, 0.82, "TTD", "#9B6AAE", fc="#FCF8FF")
pill(ax, 3.72, 0.50, 1.66, "clustered alarms", "#D49A4A", fc="#FFF8EE")
pill(ax, 5.58, 0.50, 1.44, "proxy-hour", "#D49A4A", fc="#FFF8EE")
pill(ax, 7.22, 0.50, 1.24, "per-frame FP", "#D49A4A", fc="#FFF8EE")

arrow(ax, (8.78, 0.91), (9.14, 0.91), color="#8A8A8A")
pill(ax, 9.18, 0.72, 1.10, "risk check", "#4F4F4F", fc="#F8F8F8", h=0.38)

save_fig(fig, "fig4_raco_pipeline")
