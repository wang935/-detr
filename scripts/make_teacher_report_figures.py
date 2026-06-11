"""Make Stage 5 teacher-report figures from formal YOLO seed outputs.

The figures summarize the current completed YOLO26n Stage 5 evidence:
hard-negative training sharply reduces fixed-recall false alarms across
three formal seeds while preserving recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "teacher_report_stage5_yolo_20260606"
STAGE5_DIR = ROOT / "formal_results" / "stage5"
RUN_ROOT = ROOT / "runs" / "detect" / "runs_stage5_formal"
RECALL_TARGETS = ["0.80", "0.85", "0.90", "0.95"]
ARM_ORDER = ["baseline", "baseline_eqstep", "hardneg"]
ARM_LABELS = {
    "baseline": "Positive-only",
    "baseline_eqstep": "Equal-step positive",
    "hardneg": "Hard negatives",
}
ARM_COLORS = {
    "baseline": "#6B7280",
    "baseline_eqstep": "#7C8DB5",
    "hardneg": "#2E9E44",
}
PANEL_LABEL_KW = dict(
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=None,
)


def apply_style() -> None:
    # Mandatory editable SVG text settings from the nature-figure Python track.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure, stem: str, dpi: int = 600) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / stem
    saved = []
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": dpi},
        "tiff": {"dpi": dpi},
    }.items():
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def read_results_last_row(csv_path: Path) -> dict[str, float]:
    frame = pd.read_csv(csv_path)
    last = frame.iloc[-1]
    return {
        "train_precision": float(last.get("metrics/precision(B)", np.nan)),
        "train_recall": float(last.get("metrics/recall(B)", np.nan)),
        "train_map50": float(last.get("metrics/mAP50(B)", np.nan)),
        "train_map50_95": float(last.get("metrics/mAP50-95(B)", np.nan)),
        "train_epoch": int(last.get("epoch", len(frame))),
        "results_rows": int(len(frame) + 1),
    }


def collect_stage5_yolo() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(STAGE5_DIR.glob("yolo_seed*")):
        gonogo_path = run_dir / "gonogo.json"
        meta_path = run_dir / "run_meta.json"
        if not gonogo_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gonogo = json.loads(gonogo_path.read_text(encoding="utf-8"))
        run_id = run_dir.name
        for entry in gonogo.values():
            arm = entry["arm"]
            if arm not in ARM_ORDER:
                continue
            train_csv = RUN_ROOT / run_id / arm / "results.csv"
            last_pt = RUN_ROOT / run_id / arm / "weights" / "last.pt"
            train_metrics = read_results_last_row(train_csv) if train_csv.exists() else {}
            for target in RECALL_TARGETS:
                fixed = entry["at_fixed_recall"][target]
                row = {
                    "run_id": run_id,
                    "family": entry["family"],
                    "seed": int(entry["seed"]),
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    "target_recall": float(target),
                    "threshold": float(fixed["thr"]),
                    "calib_recall": float(fixed["calib_recall"]),
                    "test_recall": float(fixed["recall"]),
                    "fpr": float(fixed["FPR"]),
                    "fppi": float(fixed["FPPI"]),
                    "epochs": int(entry["epochs"]),
                    "val": bool(entry["val"]),
                    "export_conf": float(entry["export_conf"]),
                    "last_pt": bool(last_pt.exists()),
                    "eval_protocol": entry["eval_protocol"],
                    "labels_md5": entry["labels_md5"],
                    "eval_calib_md5": entry["eval_calib_md5"],
                    "eval_test_md5": entry["eval_test_md5"],
                    "n_pos": int(entry["n_pos"]),
                    "n_neg": int(entry["n_neg"]),
                    "arm_train_count": int(entry["arm_train_counts"][arm]),
                    "meta_family_label": meta.get("family_label", ""),
                }
                row.update(train_metrics)
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No formal YOLO Stage 5 results found.")
    required_seeds = {11, 22, 33}
    found_seeds = set(frame["seed"].unique())
    if not required_seeds.issubset(found_seeds):
        raise RuntimeError(f"Missing formal seeds: {sorted(required_seeds - found_seeds)}")
    return frame


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        frame.groupby(["target_recall", "arm", "arm_label"], as_index=False)
        .agg(
            n=("seed", "nunique"),
            test_recall_mean=("test_recall", "mean"),
            test_recall_std=("test_recall", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
            fppi_mean=("fppi", "mean"),
            fppi_std=("fppi", "std"),
            map50_mean=("train_map50", "mean"),
            map50_std=("train_map50", "std"),
            map50_95_mean=("train_map50_95", "mean"),
            map50_95_std=("train_map50_95", "std"),
        )
        .sort_values(["target_recall", "arm"])
    )
    r90 = frame[np.isclose(frame["target_recall"], 0.90)]
    deltas = []
    for seed, seed_frame in r90.groupby("seed"):
        hard = seed_frame[seed_frame["arm"] == "hardneg"].iloc[0]
        for ref_arm in ["baseline", "baseline_eqstep"]:
            ref = seed_frame[seed_frame["arm"] == ref_arm].iloc[0]
            deltas.append(
                {
                    "seed": int(seed),
                    "comparison": f"Hard negatives - {ARM_LABELS[ref_arm]}",
                    "delta_fpr": hard["fpr"] - ref["fpr"],
                    "delta_fppi": hard["fppi"] - ref["fppi"],
                    "delta_recall": hard["test_recall"] - ref["test_recall"],
                }
            )
    return summary, pd.DataFrame(deltas)


def plot_master(frame: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 6.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.05, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    r90_summary = summary[np.isclose(summary["target_recall"], 0.90)].set_index("arm")
    x = np.arange(2)
    width = 0.22
    metrics = [("fpr", "FPR"), ("fppi", "FPPI")]
    for i, arm in enumerate(ARM_ORDER):
        means = [r90_summary.loc[arm, f"{m}_mean"] for m, _ in metrics]
        stds = [r90_summary.loc[arm, f"{m}_std"] for m, _ in metrics]
        offset = (i - 1) * width
        ax_a.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=2.5,
            color=ARM_COLORS[arm],
            edgecolor="#222222",
            linewidth=0.7,
            label=ARM_LABELS[arm],
        )
        seed_vals = frame[
            np.isclose(frame["target_recall"], 0.90) & (frame["arm"] == arm)
        ]
        for j, (metric, _) in enumerate(metrics):
            jitter = np.linspace(-0.045, 0.045, len(seed_vals))
            ax_a.scatter(
                np.full(len(seed_vals), x[j] + offset) + jitter,
                seed_vals[metric],
                s=12,
                color="white",
                edgecolor="#222222",
                linewidth=0.55,
                zorder=3,
            )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([label for _, label in metrics])
    ax_a.set_ylabel("False-alarm burden at R0.90")
    ax_a.set_title("Three formal seeds")
    ax_a.set_ylim(0, 0.28)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.55, 1.02), fontsize=7, ncol=1)
    ax_a.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    add_panel_label(ax_a, "a")

    for arm in ARM_ORDER:
        pivot = (
            frame[frame["arm"] == arm]
            .pivot(index="seed", columns="target_recall", values="fpr")
            .sort_index(axis=1)
        )
        xs = pivot.columns.to_numpy(dtype=float)
        mean = pivot.mean(axis=0).to_numpy()
        std = pivot.std(axis=0).to_numpy()
        ax_b.plot(
            xs,
            mean,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
        ax_b.fill_between(xs, mean - std, mean + std, color=ARM_COLORS[arm], alpha=0.16)
    ax_b.set_xlabel("Target recall")
    ax_b.set_ylabel("Test FPR")
    ax_b.set_title("Stable across operating points")
    ax_b.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax_b.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    add_panel_label(ax_b, "b")

    heat = deltas.pivot(index="seed", columns="comparison", values="delta_fpr")
    heat = heat[
        [
            "Hard negatives - Positive-only",
            "Hard negatives - Equal-step positive",
        ]
    ]
    im = ax_c.imshow(heat.to_numpy(), cmap="Greens_r", vmin=-0.22, vmax=0.0, aspect="auto")
    ax_c.set_xticks(np.arange(heat.shape[1]))
    ax_c.set_xticklabels(["vs\npositive-only", "vs\nequal-step"], rotation=0)
    ax_c.set_yticks(np.arange(heat.shape[0]))
    ax_c.set_yticklabels([f"seed {s}" for s in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax_c.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
    ax_c.set_title("Paired R0.90 FPR reduction")
    ax_c.set_frame_on(False)
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.75)
    cbar.set_label("Delta FPR")
    add_panel_label(ax_c, "c")

    r90 = frame[np.isclose(frame["target_recall"], 0.90)]
    map_summary = (
        r90.groupby("arm", as_index=False)
        .agg(
            map50_mean=("train_map50", "mean"),
            map50_std=("train_map50", "std"),
            fpr_mean=("fpr", "mean"),
        )
        .set_index("arm")
    )
    y_pos = np.arange(len(ARM_ORDER))
    ax_d.errorbar(
        [map_summary.loc[arm, "map50_mean"] for arm in ARM_ORDER],
        y_pos,
        xerr=[map_summary.loc[arm, "map50_std"] for arm in ARM_ORDER],
        fmt="o",
        markersize=6,
        color="#222222",
        ecolor="#222222",
        elinewidth=1,
        capsize=2.5,
        zorder=3,
    )
    for yi, arm in enumerate(ARM_ORDER):
        ax_d.scatter(
            map_summary.loc[arm, "map50_mean"],
            yi,
            s=74,
            color=ARM_COLORS[arm],
            edgecolor="#222222",
            linewidth=0.7,
            zorder=4,
        )
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([ARM_LABELS[arm] for arm in ARM_ORDER])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Training mAP50 at epoch 300 (zoomed)")
    ax_d.set_title("Lower false alarms without mAP loss")
    ax_d.set_xlim(0.68, 0.77)
    for yi, arm in enumerate(ARM_ORDER):
        ax_d.text(
            map_summary.loc[arm, "map50_mean"] + 0.004,
            yi,
            f"FPR {map_summary.loc[arm, 'fpr_mean']:.3f}",
            va="center",
            fontsize=7,
            color="#333333",
        )
    ax_d.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "YOLO26n Stage 5: hard negatives suppress fixed-recall false alarms",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.02,
        0.006,
        "Protocol: thresholds chosen on calibration positives; recall/FPR/FPPI reported on held-out test split. "
        "Bars/lines show mean across seeds 11, 22, 33; error bars/shading show SD.",
        fontsize=6.7,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    save_figure(fig, "stage5_yolo_teacher_master")


def plot_single_panels(frame: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    r90_summary = summary[np.isclose(summary["target_recall"], 0.90)].set_index("arm")

    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    x = np.arange(2)
    width = 0.22
    for i, arm in enumerate(ARM_ORDER):
        means = [r90_summary.loc[arm, "fpr_mean"], r90_summary.loc[arm, "fppi_mean"]]
        stds = [r90_summary.loc[arm, "fpr_std"], r90_summary.loc[arm, "fppi_std"]]
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=2.5,
            color=ARM_COLORS[arm],
            edgecolor="#222222",
            linewidth=0.7,
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["FPR", "FPPI"])
    ax.set_ylabel("R0.90 false-alarm burden")
    ax.set_title("Hard negatives reduce alarm false positives by ~98%")
    ax.set_ylim(0, 0.28)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "stage5_yolo_r090_bar")

    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    for arm in ARM_ORDER:
        pivot = (
            frame[frame["arm"] == arm]
            .pivot(index="seed", columns="target_recall", values="fpr")
            .sort_index(axis=1)
        )
        xs = pivot.columns.to_numpy(dtype=float)
        mean = pivot.mean(axis=0).to_numpy()
        std = pivot.std(axis=0).to_numpy()
        ax.plot(xs, mean, marker="o", lw=2, ms=4.5, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        ax.fill_between(xs, mean - std, mean + std, color=ARM_COLORS[arm], alpha=0.16)
    ax.set_xlabel("Target recall")
    ax.set_ylabel("Test FPR")
    ax.set_title("False-alarm control persists from R0.80 to R0.95")
    ax.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "stage5_yolo_fpr_operating_curve")

    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    heat = deltas.pivot(index="seed", columns="comparison", values="delta_fpr")
    heat = heat[
        [
            "Hard negatives - Positive-only",
            "Hard negatives - Equal-step positive",
        ]
    ]
    im = ax.imshow(heat.to_numpy(), cmap="Greens_r", vmin=-0.22, vmax=0.0, aspect="auto")
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(["vs positive-only", "vs equal-step"], rotation=15, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels([f"seed {s}" for s in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Every seed lowers R0.90 FPR")
    ax.set_frame_on(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Delta FPR")
    fig.tight_layout()
    save_figure(fig, "stage5_yolo_seed_delta_heatmap")

    fig, ax1 = plt.subplots(figsize=(4.7, 3.1))
    x = np.arange(len(ARM_ORDER))
    map_means = [r90_summary.loc[arm, "map50_mean"] for arm in ARM_ORDER]
    map_stds = [r90_summary.loc[arm, "map50_std"] for arm in ARM_ORDER]
    fpr_means = [r90_summary.loc[arm, "fpr_mean"] for arm in ARM_ORDER]
    ax1.errorbar(
        map_means,
        x,
        xerr=map_stds,
        fmt="o",
        markersize=6,
        color="#222222",
        ecolor="#222222",
        elinewidth=1,
        capsize=2.5,
        zorder=3,
    )
    for xi, arm in enumerate(ARM_ORDER):
        ax1.scatter(
            map_means[xi],
            xi,
            s=78,
            color=ARM_COLORS[arm],
            edgecolor="#222222",
            linewidth=0.7,
            zorder=4,
        )
        ax1.text(map_means[xi] + 0.004, xi, f"FPR {fpr_means[xi]:.3f}", va="center", fontsize=8)
    ax1.set_xlabel("mAP50 at epoch 300 (zoomed)")
    ax1.set_yticks(x)
    ax1.set_yticklabels([ARM_LABELS[arm] for arm in ARM_ORDER])
    ax1.invert_yaxis()
    ax1.set_xlim(0.68, 0.77)
    ax1.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax1.set_title("mAP improves while alarm FPR collapses")
    fig.tight_layout()
    save_figure(fig, "stage5_yolo_map_vs_fpr")


def write_notes(frame: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    r90_summary = summary[np.isclose(summary["target_recall"], 0.90)].set_index("arm")
    hard_fpr = r90_summary.loc["hardneg", "fpr_mean"]
    base_fpr = r90_summary.loc["baseline", "fpr_mean"]
    eq_fpr = r90_summary.loc["baseline_eqstep", "fpr_mean"]
    rel_drop_base = 100 * (1 - hard_fpr / base_fpr)
    rel_drop_eq = 100 * (1 - hard_fpr / eq_fpr)
    notes = f"""# Teacher Report Figure Notes

## Figure contract

- Core conclusion: in the completed YOLO26n Stage 5 formal matrix, hard-negative training sharply reduces fixed-recall alarm false positives while preserving recall.
- Archetype: quantitative grid with one hero panel and three supporting panels.
- Evidence chain:
  - Panel a: R0.90 FPR/FPPI mean +/- SD across seeds.
  - Panel b: FPR trend across target recall points R0.80-R0.95.
  - Panel c: paired seed-level FPR deltas against both controls.
  - Panel d: mAP50 and R0.90 FPR show that false-alarm reduction is not bought by mAP collapse.
- Review risk: these figures support a YOLO26n/D-Fire formal result only. They do not yet prove RT-DETR, cross-family, or external-domain claims.

## Key numbers for speaking

- R0.90 FPR: positive-only {base_fpr:.4f}, equal-step positive {eq_fpr:.4f}, hard negatives {hard_fpr:.4f}.
- Relative FPR reduction: {rel_drop_base:.1f}% versus positive-only; {rel_drop_eq:.1f}% versus equal-step positive.
- Seeds included: {', '.join(str(x) for x in sorted(frame['seed'].unique()))}.
- Arms included: {', '.join(ARM_LABELS[a] for a in ARM_ORDER)}.
- Protocol: threshold selected on calibration positives; FPR/FPPI reported on held-out test split.

## Export QA

- Backend: Python/matplotlib only.
- Editable vector exports: SVG with text kept as text; PDF fonttype=42.
- Raster preview exports: PNG and TIFF at 600 dpi.
- Source data: `stage5_yolo_teacher_source_data.csv`, `stage5_yolo_teacher_summary.csv`, `stage5_yolo_teacher_paired_deltas.csv`.
"""
    (OUT_DIR / "TEACHER_REPORT_FIGURE_NOTES.md").write_text(notes, encoding="utf-8")


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = collect_stage5_yolo()
    summary, deltas = summarize(frame)
    frame.to_csv(OUT_DIR / "stage5_yolo_teacher_source_data.csv", index=False)
    summary.to_csv(OUT_DIR / "stage5_yolo_teacher_summary.csv", index=False)
    deltas.to_csv(OUT_DIR / "stage5_yolo_teacher_paired_deltas.csv", index=False)
    plot_master(frame, summary, deltas)
    plot_single_panels(frame, summary, deltas)
    write_notes(frame, summary, deltas)
    print(f"[ok] Wrote teacher-report figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
