from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "training_summary_20260607"
OUT_DIR = ROOT / "paper_assets" / "advisor_report_nature_20260607"

ARMS = ["baseline", "baseline_eqstep", "hardneg", "hardneg_sched"]
ARM_LABELS = {
    "baseline": "Positive-only",
    "baseline_eqstep": "Equal-step positive",
    "hardneg": "Hard negatives",
    "hardneg_sched": "Hardneg sched",
}
COLORS = {
    "baseline": "#6F7785",
    "baseline_eqstep": "#7D91BD",
    "hardneg": "#2F9E44",
    "hardneg_sched": "#147A42",
}
TARGETS = [0.80, 0.85, 0.90, 0.95]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(OUT_DIR / f"{stem}.{ext}", **kwargs)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_long = pd.read_csv(SUMMARY_DIR / "fixed_recall_eval_long.csv")
    training = pd.read_csv(SUMMARY_DIR / "training_final_runs.csv")
    for col in [
        "seed",
        "target_recall",
        "actual_recall",
        "FPR",
        "FPPI",
        "map50_B",
        "map50_95_B",
        "precision_B",
        "train_recall_B",
    ]:
        if col in eval_long.columns:
            eval_long[col] = pd.to_numeric(eval_long[col], errors="coerce")
    for col in ["seed", "map50_B", "map50_95_B", "precision_B", "recall_B"]:
        if col in training.columns:
            training[col] = pd.to_numeric(training[col], errors="coerce")
    return eval_long, training


def formal_eval(eval_long: pd.DataFrame) -> pd.DataFrame:
    mask = (
        eval_long["metric_status"].eq("reported")
        & eval_long["arm"].isin(ARMS)
        & eval_long["evidence_level"].isin(["formal_main", "single_arm_probe"])
    )
    return eval_long[mask].copy()


def plot_false_alarm_curves(eval_long: pd.DataFrame) -> None:
    data = formal_eval(eval_long)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)

    for ax, metric, ylabel in zip(axes, ["FPR", "FPPI"], ["Test FPR", "Test FPPI"]):
        for arm in ARMS:
            pivot = data[data["arm"].eq(arm)].pivot(index="seed", columns="target_recall", values=metric)
            mean = pivot[TARGETS].mean(axis=0).to_numpy()
            sd = pivot[TARGETS].std(axis=0).to_numpy()
            ax.plot(
                TARGETS,
                mean,
                marker="o",
                lw=1.9,
                ms=3.8,
                color=COLORS[arm],
                label=ARM_LABELS[arm],
            )
            ax.fill_between(TARGETS, mean - sd, mean + sd, color=COLORS[arm], alpha=0.14, linewidth=0)
        ax.set_xticks(TARGETS)
        ax.set_xlabel("Target recall", fontsize=7.0)
        ax.set_ylabel(ylabel, fontsize=7.0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)

    axes[0].set_title("False-positive rate", fontsize=9.2)
    axes[1].set_title("False positives per image", fontsize=9.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )

    fig.suptitle("Hard negatives shift the whole false-alarm curve downward", fontsize=10.2, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        f"Each point is mean across 3 seeds; shaded area is mean ± standard deviation.",
        ha="center",
        fontsize=6.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    save_all(fig, "advisor_fig1_false_alarm_curves")
    plt.close(fig)


def false_alarm_matrix(seed: int, data: pd.DataFrame) -> np.ndarray:
    seed_df = data[data["seed"].eq(seed)]
    rows = []
    for arm in ARMS:
        arm_df = seed_df[seed_df["arm"].eq(arm)]
        row = []
        for metric in ["FPR", "FPPI"]:
            for target in TARGETS:
                rec = arm_df[arm_df["target_recall"].eq(target)]
                row.append(np.nan if rec.empty else float(rec.iloc[0][metric]))
        rows.append(row)
    return np.array(rows, dtype=float)


def plot_seed_consistency(eval_long: pd.DataFrame) -> None:
    data = formal_eval(eval_long)
    seeds = [11, 22, 33]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), sharey=True)
    vmax = np.nanmax([false_alarm_matrix(seed, data) for seed in seeds])
    cmap = mpl.colormaps["YlGn_r"]

    for ax, seed in zip(axes, seeds):
        mat = false_alarm_matrix(seed, data)
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

        ax.set_title(f"seed {seed}", fontsize=8.7, fontweight="bold")
        ax.set_yticks(np.arange(len(ARMS)))
        ax.set_yticklabels([ARM_LABELS[a] for a in ARMS], fontsize=6.8)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(
            [f"FPR\nR{t:.2f}" for t in TARGETS] + [f"FPPI\nR{t:.2f}" for t in TARGETS],
            fontsize=5.8,
            rotation=20,
            ha="right",
        )
        ax.set_xticks(np.arange(9) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ARMS)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axvline(3.5, color="#222222", linewidth=0.7)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    text = "NA"
                    color = "#777777"
                else:
                    text = f"{val:.4f}"
                    color = "white" if im.norm(val) < 0.5 else "#111111"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color=color,
                )

    fig.suptitle(
        "Per-seed false-alarm sweep remains consistent across all four arms",
        fontsize=10.1,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Cell values are raw FPR/FPPI. Lower is better.",
        ha="center",
        fontsize=6.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    save_all(fig, "advisor_fig2_seed_consistency_heatmaps")
    plt.close(fig)


def plot_tradeoff(eval_long: pd.DataFrame, training: pd.DataFrame) -> None:
    data = formal_eval(eval_long)
    r90 = data[data["target_recall"].eq(0.90)].copy()
    train = training[
        training["arm"].isin(ARMS) & training["evidence_level"].isin(["formal_main", "single_arm_probe"])
    ].copy()
    merged = r90.merge(train[["seed", "arm", "map50_B", "map50_95_B"]], on=["seed", "arm"], suffixes=("", "_train"))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.95))

    for arm in ARMS:
        sub = merged[merged["arm"].eq(arm)]
        axes[0].scatter(
            sub["FPR"],
            sub["map50_B_train"],
            s=42,
            color=COLORS[arm],
            edgecolor="#222222",
            linewidth=0.5,
            label=ARM_LABELS[arm],
        )
    axes[0].set_xlabel("R0.90 FPR")
    axes[0].set_ylabel("mAP50 at epoch 300")
    axes[0].set_title("Low FPR with preserved detection quality", fontsize=9.2)
    axes[0].grid(color="#E5E7EB", linewidth=0.6)
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(1.02, 1.02),
        fontsize=6.5,
    )

    x = np.arange(len(ARMS))
    width = 0.34
    means50 = train.groupby("arm")["map50_B"].mean().reindex(ARMS)
    std50 = train.groupby("arm")["map50_B"].std().reindex(ARMS)
    means95 = train.groupby("arm")["map50_95_B"].mean().reindex(ARMS)
    std95 = train.groupby("arm")["map50_95_B"].std().reindex(ARMS)

    axes[1].bar(
        x - width / 2,
        means50,
        width,
        yerr=std50,
        color=[COLORS[a] for a in ARMS],
        edgecolor="#222222",
        linewidth=0.6,
        capsize=2,
        label="mAP50",
    )
    axes[1].bar(
        x + width / 2,
        means95,
        width,
        yerr=std95,
        color=[COLORS[a] for a in ARMS],
        edgecolor="#222222",
        linewidth=0.6,
        capsize=2,
        alpha=0.62,
        label="mAP50-95",
    )
    for i, arm in enumerate(ARMS):
        vals = train[train["arm"].eq(arm)]
        jitter = np.linspace(-0.05, 0.05, len(vals))
        axes[1].scatter(
            np.full(len(vals), x[i] - width / 2) + jitter,
            vals["map50_B"],
            s=11,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.4,
            zorder=4,
        )
        axes[1].scatter(
            np.full(len(vals), x[i] + width / 2) + jitter,
            vals["map50_95_B"],
            s=11,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.4,
            zorder=4,
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ARM_LABELS[a].replace(" ", "\n") for a in ARMS], fontsize=6.8)
    axes[1].set_ylim(0.38, 0.78)
    axes[1].set_ylabel("mAP at epoch 300")
    axes[1].set_title("Final training metrics", fontsize=9.2)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.01), fontsize=6.5, ncol=1)
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.6)

    fig.suptitle(
        "Hard-negative variants improve false-alarm behavior across the operating set",
        fontsize=10.1,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        f"Each point is one seed; bars show mean ± SD.",
        ha="center",
        fontsize=6.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.08, 0.90, 0.91])
    save_all(fig, "advisor_fig3_map_tradeoff")
    plt.close(fig)


def write_notes() -> None:
    text = """# Advisor Report Figure Notes

- Fig. 1: Multi-seed mean curves for FPR and FPPI across R0.80-R0.95 for four arms.
- Fig. 2: Per-seed heatmaps showing the same false-alarm reduction pattern.
- Fig. 3: Trade-off figure showing mAP performance is preserved under lower false-alarm rates.

Main claim: hard-negative training shifts the whole false-alarm curve downward across operating points, not just at R0.90.
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ADVISOR_FIGURE_NOTES.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    eval_long, training = load_data()
    plot_false_alarm_curves(eval_long)
    plot_seed_consistency(eval_long)
    plot_tradeoff(eval_long, training)
    write_notes()
    print(f"[ok] Wrote advisor report figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
