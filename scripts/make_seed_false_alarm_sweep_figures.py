from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "training_summary_20260607"
OUT_DIR = ROOT / "paper_assets" / "seed_false_alarm_sweep_20260607"

ARM_ORDER = ["baseline", "baseline_eqstep", "hardneg", "hardneg_sched"]
ARM_LABELS = {
    "baseline": "Positive-only",
    "baseline_eqstep": "Equal-step positive",
    "hardneg": "Hard negatives",
    "hardneg_sched": "Hardneg sched",
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
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(OUT_DIR / f"{stem}.{ext}", **kwargs)


def build_matrix(seed: int, data: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    seed_df = data[
        (data["seed"].eq(seed))
        & (data["metric_status"].eq("reported"))
        & (data["evidence_level"].isin(["formal_main", "single_arm_probe"]))
        & (data["arm"].isin(ARM_ORDER))
    ].copy()
    rows = []
    for arm in ARM_ORDER:
        arm_values = []
        arm_df = seed_df[seed_df["arm"].eq(arm)]
        for metric in ["FPR", "FPPI"]:
            for target in TARGETS:
                rec = arm_df[arm_df["target_recall"].eq(target)]
                arm_values.append(np.nan if rec.empty else float(rec.iloc[0][metric]))
        rows.append(arm_values)
    labels = [ARM_LABELS[a] for a in ARM_ORDER]
    columns = [f"FPR\nR{t:.2f}" for t in TARGETS] + [f"FPPI\nR{t:.2f}" for t in TARGETS]
    return np.array(rows, dtype=float), labels, columns


def score_false_alarm(values: np.ndarray) -> np.ndarray:
    max_value = np.nanmax(values)
    if not np.isfinite(max_value) or max_value <= 0:
        return np.zeros_like(values)
    score = 1.0 - (values / max_value)
    score = np.clip(score, 0.0, 1.0)
    score[np.isnan(values)] = np.nan
    return score


def plot_seed(seed: int, data: pd.DataFrame) -> None:
    values, row_labels, col_labels = build_matrix(seed, data)
    score = score_false_alarm(values)
    masked = np.ma.masked_invalid(score)

    fig, ax = plt.subplots(figsize=(7.4, 3.05))
    cmap = mpl.colormaps["YlGn"].copy()
    cmap.set_bad("#F3F4F6")
    ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_xlabel("Fixed-recall false-alarm metric")
    ax.set_ylabel("Arm")
    ax.set_title(f"Seed {seed}: FPR/FPPI sweep from R0.80 to R0.95", fontsize=10, fontweight="bold", pad=10)

    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(3.5, color="#111111", linewidth=0.8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            s = score[i, j]
            if np.isnan(val):
                text, color = "NA", "#555555"
            else:
                text = f"{val:.4f}"
                color = "white" if s > 0.62 else "#111111"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.2, color=color)

    fig.text(
        0.5,
        0.01,
        "单元格为原始数值。颜色越深表示误报负担越低。",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#555555",
    )
    fig.tight_layout(rect=[0.045, 0.08, 1.0, 0.95])
    save_all(fig, f"seed{seed}_fpr_fppi_sweep")
    plt.close(fig)


def main() -> None:
    setup_style()
    data = pd.read_csv(SUMMARY_DIR / "fixed_recall_eval_long.csv")
    for col in ["seed", "target_recall", "FPR", "FPPI"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for seed in [11, 22, 33]:
        plot_seed(seed, data)
    print(f"[ok] Wrote false-alarm sweep figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
