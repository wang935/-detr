"""Create current Stage 5 report figures.

This script uses Python/matplotlib only. It summarizes:
1. Stage 5 v1 formal YOLO26n all3 results across seeds 11/22/33.
2. Stage5-PV v2 hardneg_sched single-arm probe across seeds 11/22/33.
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
OUT_DIR = ROOT / "paper_assets" / "teacher_report_stage5_current_20260607"
V1_DIR = ROOT / "formal_results" / "stage5"
PV2_DIR = ROOT / "formal_results" / "stage5_pv_v2"
V1_RUN_ROOT = ROOT / "runs" / "detect" / "runs_stage5_formal"
PV2_RUN_ROOT = ROOT / "runs" / "detect" / "runs_stage5_pv_v2"

TARGETS = ["0.80", "0.85", "0.90", "0.95"]
ARMS = ["baseline", "baseline_eqstep", "hardneg"]
LABELS = {
    "baseline": "Positive-only",
    "baseline_eqstep": "Equal-step positive",
    "hardneg": "Hard negatives",
    "hardneg_sched": "Scheduled hard negatives",
}
COLORS = {
    "baseline": "#6B7280",
    "baseline_eqstep": "#7C8DB5",
    "hardneg": "#2E9E44",
    "hardneg_sched": "#167A45",
}


def apply_style() -> None:
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.12,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / stem
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(base.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def final_train_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    last = frame.iloc[-1]
    return {
        "epoch": float(last["epoch"]),
        "train_precision": float(last["metrics/precision(B)"]),
        "train_recall": float(last["metrics/recall(B)"]),
        "map50": float(last["metrics/mAP50(B)"]),
        "map50_95": float(last["metrics/mAP50-95(B)"]),
        "results_lines": float(len(frame) + 1),
    }


def collect_v1() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(V1_DIR.glob("yolo_seed*")):
        gonogo_path = run_dir / "gonogo.json"
        if not gonogo_path.exists():
            continue
        data = json.loads(gonogo_path.read_text(encoding="utf-8"))
        run_id = run_dir.name
        for rep in data.values():
            arm = rep.get("arm")
            if arm not in ARMS:
                continue
            train_path = V1_RUN_ROOT / run_id / arm / "results.csv"
            train = final_train_metrics(train_path)
            for target in TARGETS:
                item = rep["at_fixed_recall"][target]
                rows.append(
                    {
                        "stage": "v1",
                        "run_id": run_id,
                        "seed": int(rep["seed"]),
                        "arm": arm,
                        "label": LABELS[arm],
                        "target": float(target),
                        "threshold": float(item["thr"]),
                        "calib_recall": float(item["calib_recall"]),
                        "recall": float(item["recall"]),
                        "fpr": float(item["FPR"]),
                        "fppi": float(item["FPPI"]),
                        "epochs": int(rep["epochs"]),
                        "val": bool(rep["val"]),
                        "source_kind": "formal all3",
                        **train,
                    }
                )
    return pd.DataFrame(rows)


def collect_pv2() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(PV2_DIR.glob("dfire_yolo26n_seed*")):
        gonogo_path = run_dir / "gonogo.json"
        if not gonogo_path.exists():
            continue
        data = json.loads(gonogo_path.read_text(encoding="utf-8"))
        run_id = run_dir.name
        for rep in data.values():
            if rep.get("arm") != "hardneg_sched":
                continue
            train_path = PV2_RUN_ROOT / run_id / "hardneg_sched" / "results.csv"
            train = final_train_metrics(train_path)
            for target in TARGETS:
                item = rep["at_fixed_recall"][target]
                rows.append(
                    {
                        "stage": "pv2",
                        "run_id": run_id,
                        "seed": int(rep["seed"]),
                        "arm": "hardneg_sched",
                        "label": LABELS["hardneg_sched"],
                        "target": float(target),
                        "threshold": float(item["thr"]),
                        "calib_recall": float(item["calib_recall"]),
                        "recall": float(item["recall"]),
                        "fpr": float(item["FPR"]),
                        "fppi": float(item["FPPI"]),
                        "epochs": int(rep["epochs"]),
                        "val": bool(rep["val"]),
                        "source_kind": "single-arm probe",
                        **train,
                    }
                )
    return pd.DataFrame(rows)


def mean_sd(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(by, as_index=False)
        .agg(
            n=("seed", "nunique"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_sd=("fpr", "std"),
            fppi_mean=("fppi", "mean"),
            fppi_sd=("fppi", "std"),
            map50_mean=("map50", "mean"),
            map50_sd=("map50", "std"),
            map50_95_mean=("map50_95", "mean"),
            map50_95_sd=("map50_95", "std"),
        )
        .fillna(0.0)
    )


def plot_current_master(v1: pd.DataFrame, pv2: pd.DataFrame) -> None:
    v1_summary = mean_sd(v1, ["target", "arm", "label"])
    r90 = v1[v1["target"].eq(0.90)]
    r90s = v1_summary[v1_summary["target"].eq(0.90)].set_index("arm")

    fig = plt.figure(figsize=(7.3, 6.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.05, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(2)
    width = 0.22
    for i, arm in enumerate(ARMS):
        vals = [r90s.loc[arm, "fpr_mean"], r90s.loc[arm, "fppi_mean"]]
        errs = [r90s.loc[arm, "fpr_sd"], r90s.loc[arm, "fppi_sd"]]
        offset = (i - 1) * width
        ax_a.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs,
            capsize=2.5,
            color=COLORS[arm],
            edgecolor="#222222",
            linewidth=0.7,
            label=LABELS[arm],
        )
        for j, metric in enumerate(["fpr", "fppi"]):
            dots = r90[r90["arm"].eq(arm)][metric].to_numpy()
            jitter = np.linspace(-0.04, 0.04, len(dots))
            ax_a.scatter(
                np.full(len(dots), x[j] + offset) + jitter,
                dots,
                s=13,
                facecolor="white",
                edgecolor="#222222",
                linewidth=0.55,
                zorder=3,
            )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["FPR", "FPPI"])
    ax_a.set_ylabel("False-alarm burden at R0.90")
    ax_a.set_title("Formal YOLO26n all3 matrix")
    ax_a.set_ylim(0, 0.28)
    ax_a.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    handles, labels = ax_a.get_legend_handles_labels()
    add_label(ax_a, "a")

    for arm in ARMS:
        pivot = v1[v1["arm"].eq(arm)].pivot(index="seed", columns="target", values="fpr")
        xs = pivot.columns.to_numpy(float)
        mean = pivot.mean(axis=0).to_numpy()
        sd = pivot.std(axis=0).to_numpy()
        ax_b.plot(xs, mean, marker="o", lw=1.8, ms=4, color=COLORS[arm], label=LABELS[arm])
        ax_b.fill_between(xs, mean - sd, mean + sd, color=COLORS[arm], alpha=0.14)
    ax_b.set_xlabel("Target recall")
    ax_b.set_ylabel("Test FPR")
    ax_b.set_title("Stable from R0.80 to R0.95")
    ax_b.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax_b.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    add_label(ax_b, "b")

    deltas = []
    for seed, seed_frame in r90.groupby("seed"):
        hard = seed_frame[seed_frame["arm"].eq("hardneg")].iloc[0]
        for ref_arm in ["baseline", "baseline_eqstep"]:
            ref = seed_frame[seed_frame["arm"].eq(ref_arm)].iloc[0]
            deltas.append(
                {
                    "seed": seed,
                    "comparison": f"vs {LABELS[ref_arm]}",
                    "delta_fpr": hard["fpr"] - ref["fpr"],
                }
            )
    dframe = pd.DataFrame(deltas)
    heat = dframe.pivot(index="seed", columns="comparison", values="delta_fpr")
    heat = heat[["vs Positive-only", "vs Equal-step positive"]]
    im = ax_c.imshow(heat.to_numpy(), cmap="Greens_r", vmin=-0.22, vmax=0.0, aspect="auto")
    ax_c.set_xticks(np.arange(heat.shape[1]))
    ax_c.set_xticklabels(["vs\npositive-only", "vs\nequal-step"])
    ax_c.set_yticks(np.arange(heat.shape[0]))
    ax_c.set_yticklabels([f"seed {s}" for s in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax_c.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax_c.set_title("Every seed lowers R0.90 FPR")
    ax_c.set_frame_on(False)
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.75)
    cbar.set_label("Delta FPR")
    add_label(ax_c, "c")

    # PV2 probe: matched seeds against v1 hardneg.
    pv2_r90 = pv2[pv2["target"].eq(0.90)].copy()
    v1h_r90 = r90[r90["arm"].eq("hardneg")].copy()
    matched = pv2_r90.merge(
        v1h_r90[["seed", "fpr", "recall", "map50"]],
        on="seed",
        suffixes=("_sched", "_v1hard"),
    )
    y = np.arange(len(matched))
    ax_d.hlines(y, matched["fpr_v1hard"], matched["fpr_sched"], color="#A7C8AE", lw=2.2)
    ax_d.scatter(matched["fpr_v1hard"], y, s=46, color=COLORS["hardneg"], edgecolor="#222222", label="v1 hardneg")
    ax_d.scatter(
        matched["fpr_sched"],
        y,
        s=46,
        color=COLORS["hardneg_sched"],
        edgecolor="#222222",
        marker="D",
        label="sched probe",
    )
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([f"seed {int(s)}" for s in matched["seed"]])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("R0.90 FPR")
    ax_d.set_title("PV v2 probe", loc="left")
    ax_d.set_xlim(0, 0.0072)
    ax_d.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax_d.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=7, borderaxespad=0)
    for yi, row in matched.iterrows():
        delta_map = row["map50_sched"] - row["map50_v1hard"]
        ax_d.text(
            0.00695,
            yi,
            f"mAP50 {delta_map:+.3f}",
            va="center",
            ha="right",
            fontsize=7,
            color="#555555",
        )
    add_label(ax_d, "d")

    fig.suptitle(
        "Current Stage 5 evidence: strong v1 result plus a cautious PV v2 probe",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=3,
        fontsize=7,
        handlelength=1.3,
        columnspacing=1.4,
    )
    fig.text(
        0.02,
        0.006,
        "Protocol: thresholds calibrated on positives; recall/FPR/FPPI reported on held-out test split. "
        "PV v2 hardneg_sched is a three-seed single-arm probe, not a strict all4 matrix.",
        fontsize=6.7,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.91])
    save(fig, "stage5_current_master")


def plot_v1_claim(v1: pd.DataFrame) -> None:
    r90 = v1[v1["target"].eq(0.90)]
    summary = mean_sd(r90, ["arm", "label"]).set_index("arm")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(ARMS))
    vals = [summary.loc[arm, "fpr_mean"] for arm in ARMS]
    errs = [summary.loc[arm, "fpr_sd"] for arm in ARMS]
    bars = ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=3,
        color=[COLORS[arm] for arm in ARMS],
        edgecolor="#222222",
        linewidth=0.8,
    )
    for i, arm in enumerate(ARMS):
        dots = r90[r90["arm"].eq(arm)]["fpr"].to_numpy()
        ax.scatter(
            np.full(len(dots), i) + np.linspace(-0.055, 0.055, len(dots)),
            dots,
            s=18,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[arm] for arm in ARMS], rotation=12, ha="right")
    ax.set_ylabel("R0.90 test FPR")
    ax.set_title("Hard negatives reduce YOLO26n alarm FPR by 98.2%")
    ax.set_ylim(0, 0.24)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.annotate(
        "98.2% lower\nthan positive-only",
        xy=(2, vals[2]),
        xytext=(1.35, 0.085),
        arrowprops=dict(arrowstyle="-|>", color="#2E9E44", lw=1),
        fontsize=8,
        color="#167A45",
        ha="left",
    )
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.4f}", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "stage5_v1_r090_fpr_claim")


def plot_pv2_probe(v1: pd.DataFrame, pv2: pd.DataFrame) -> None:
    v1h = v1[(v1["target"].eq(0.90)) & (v1["arm"].eq("hardneg"))]
    sched = pv2[pv2["target"].eq(0.90)]
    matched = sched.merge(
        v1h[["seed", "fpr", "recall", "map50", "map50_95"]],
        on="seed",
        suffixes=("_sched", "_v1hard"),
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.0), width_ratios=[1.0, 1.0])

    x = np.arange(len(matched))
    width = 0.34
    ax1.bar(x - width / 2, matched["fpr_v1hard"], width, color=COLORS["hardneg"], edgecolor="#222222", label="v1 hardneg")
    ax1.bar(
        x + width / 2,
        matched["fpr_sched"],
        width,
        color=COLORS["hardneg_sched"],
        edgecolor="#222222",
        label="sched probe",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"seed {int(s)}" for s in matched["seed"]])
    ax1.set_ylabel("R0.90 test FPR")
    ax1.set_title("Probe lowers FPR slightly")
    ax1.set_ylim(0, 0.005)
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    add_label(ax1, "a")

    ax2.bar(x - width / 2, matched["map50_v1hard"], width, color=COLORS["hardneg"], edgecolor="#222222")
    ax2.bar(x + width / 2, matched["map50_sched"], width, color=COLORS["hardneg_sched"], edgecolor="#222222")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seed {int(s)}" for s in matched["seed"]])
    ax2.set_ylabel("mAP50 at epoch 300")
    ax2.set_title("But mAP50 drops")
    ax2.set_ylim(0.70, 0.77)
    ax2.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    add_label(ax2, "b")

    fig.suptitle("PV v2 hardneg_sched is a useful but cautious three-seed signal", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "stage5_pv2_hardneg_sched_probe")


def write_notes(v1: pd.DataFrame, pv2: pd.DataFrame) -> None:
    r90 = v1[v1["target"].eq(0.90)]
    summary = mean_sd(r90, ["arm", "label"]).set_index("arm")
    base = summary.loc["baseline", "fpr_mean"]
    eq = summary.loc["baseline_eqstep", "fpr_mean"]
    hard = summary.loc["hardneg", "fpr_mean"]
    pv2r = pv2[pv2["target"].eq(0.90)]
    notes = f"""# Current Stage 5 Figure Notes

## Contract

- Core conclusion: YOLO26n Stage 5 v1 is a completed three-seed formal result showing hard-negative training reduces fixed-recall false alarms by roughly two orders of magnitude.
- Secondary conclusion: PV v2 hardneg_sched is a three-seed single-arm probe that further reduces FPR slightly but lowers mAP50; it should remain a cautious ablation signal.
- Archetype: quantitative grid plus two focused supporting charts.
- Backend: Python/matplotlib only.

## Key numbers

- v1 R0.90 FPR: positive-only {base:.4f}, equal-step positive {eq:.4f}, hard negatives {hard:.4f}.
- v1 relative FPR reduction: {100 * (1 - hard / base):.1f}% versus positive-only; {100 * (1 - hard / eq):.1f}% versus equal-step positive.
- PV v2 R0.90 FPR seeds {', '.join(str(int(x)) for x in pv2r['seed'])}: {', '.join(f'{x:.4f}' for x in pv2r['fpr'])}.

## Guardrails

- The v1 YOLO26n result is paper-grade for the D-Fire internal calibration/test protocol.
- Do not describe PV v2 hardneg_sched as a final method claim; it lacks the all4 controls in the same PV v2 matrix.
- Do not claim RT-DETR or external BoWFire generalization from these figures.

## Source data

- stage5_current_v1_source.csv
- stage5_current_pv2_source.csv
- stage5_current_v1_summary.csv
"""
    (OUT_DIR / "CURRENT_STAGE5_FIGURE_NOTES.md").write_text(notes, encoding="utf-8")


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1 = collect_v1()
    pv2 = collect_pv2()
    if v1.empty:
        raise RuntimeError("No Stage 5 v1 YOLO rows found.")
    v1.to_csv(OUT_DIR / "stage5_current_v1_source.csv", index=False)
    pv2.to_csv(OUT_DIR / "stage5_current_pv2_source.csv", index=False)
    mean_sd(v1, ["target", "arm", "label"]).to_csv(OUT_DIR / "stage5_current_v1_summary.csv", index=False)
    plot_current_master(v1, pv2)
    plot_v1_claim(v1)
    if not pv2.empty:
        plot_pv2_probe(v1, pv2)
    write_notes(v1, pv2)
    print(f"[ok] Wrote current Stage 5 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
