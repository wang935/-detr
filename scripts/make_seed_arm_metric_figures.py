from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "training_summary_20260607"
OUT_DIR = ROOT / "paper_assets" / "seed_arm_metric_20260607"


ARM_ORDER = ["baseline", "baseline_eqstep", "hardneg", "hardneg_sched"]
ARM_LABELS = {
    "baseline": "Positive-only",
    "baseline_eqstep": "Equal-step positive",
    "hardneg": "Hard negatives",
    "hardneg_sched": "Hardneg sched (PV2)",
}
METRICS = [
    ("actual_recall", "R0.90\nrecall", "higher"),
    ("FPR", "R0.90\nFPR", "lower"),
    ("FPPI", "R0.90\nFPPI", "lower"),
    ("map50_B", "mAP50", "higher"),
    ("map50_95_B", "mAP50-95", "higher"),
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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


def format_value(metric: str, value: float) -> str:
    if np.isnan(value):
        return "NA"
    if metric in {"FPR", "FPPI"}:
        return f"{value:.4f}"
    return f"{value:.3f}"


def score_columns(values: np.ndarray) -> np.ndarray:
    scores = np.zeros_like(values, dtype=float)
    for j, (_, _, direction) in enumerate(METRICS):
        col = values[:, j].astype(float)
        valid = ~np.isnan(col)
        if not valid.any():
            scores[:, j] = np.nan
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        if np.isclose(hi, lo):
            norm = np.full_like(col, 0.65, dtype=float)
        else:
            norm = (col - lo) / (hi - lo)
        if direction == "lower":
            norm = 1.0 - norm
        norm[~valid] = np.nan
        scores[:, j] = norm
    return scores


def plot_seed(seed: int, data: pd.DataFrame) -> None:
    r90 = data[
        (data["seed"].eq(seed))
        & (data["target_recall"].eq(0.90))
        & (data["metric_status"].eq("reported"))
        & (data["evidence_level"].isin(["formal_main", "single_arm_probe"]))
    ].copy()
    r90 = r90[r90["arm"].isin(ARM_ORDER)].copy()
    r90["arm"] = pd.Categorical(r90["arm"], categories=ARM_ORDER, ordered=True)
    r90 = r90.sort_values("arm")

    rows = []
    row_labels = []
    evidence_labels = []
    for arm in ARM_ORDER:
        arm_row = r90[r90["arm"].eq(arm)]
        if arm_row.empty:
            rows.append([np.nan] * len(METRICS))
            evidence_labels.append("")
        else:
            rec = arm_row.iloc[0]
            rows.append([float(rec[key]) for key, _, _ in METRICS])
            evidence_labels.append("probe" if rec["evidence_level"] == "single_arm_probe" else "formal")
        row_labels.append(ARM_LABELS[arm])

    values = np.array(rows, dtype=float)
    scores = score_columns(values)
    masked_scores = np.ma.masked_invalid(scores)

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    cmap = mpl.colormaps["YlGn"].copy()
    cmap.set_bad("#F3F4F6")
    ax.imshow(masked_scores, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([label for _, label, _ in METRICS])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Arm")
    ax.set_title(f"Seed {seed}: arm-by-metric summary", fontsize=10, fontweight="bold", pad=10)

    ax.set_xticks(np.arange(len(METRICS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, arm in enumerate(ARM_ORDER):
        for j, (metric, _, _) in enumerate(METRICS):
            val = values[i, j]
            score = scores[i, j]
            color = "white" if not np.isnan(score) and score > 0.62 else "#111111"
            ax.text(j, i, format_value(metric, val), ha="center", va="center", fontsize=7.5, color=color)

    fig.text(
        0.5,
        0.01,
        "Cell text shows raw values. Color is column-wise performance score; FPR/FPPI are inverted so darker green is better. "
        "PV2 is a single-arm probe.",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#555555",
    )
    fig.tight_layout(rect=[0.05, 0.08, 1.0, 0.98])
    save_all(fig, f"seed{seed}_arm_metric_matrix")
    plt.close(fig)


def main() -> None:
    setup_style()
    data = pd.read_csv(SUMMARY_DIR / "fixed_recall_eval_long.csv")
    for col in ["seed", "target_recall", "actual_recall", "FPR", "FPPI", "map50_B", "map50_95_B"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for seed in [11, 22, 33]:
        plot_seed(seed, data)
    print(f"[ok] Wrote seed arm-metric figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
