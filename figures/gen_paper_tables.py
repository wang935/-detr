from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_data import (
    ARM_LABELS,
    MODEL_LABELS,
    ROOT,
    combined_fixed_recall_summary,
    figlib_event_summary,
    fmt_metric,
    fmt_pm,
    fmt_pm2,
)


FIG_DIR = ROOT / "figures"
MODELS = ["YOLO26n", "RT-DETR-L", "InternImage-FRCNN"]
ARMS = ["baseline", "baseline_eqstep", "hardneg", "hardneg_sched"]


def tex_escape(text: object) -> str:
    return str(text).replace("_", r"\_")


def table_main(summary: pd.DataFrame) -> None:
    r090 = summary[summary["target_recall"].eq(0.90)].copy()
    rows = []
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{D-Fire fixed-recall alarm burden at R0.90. All rows report three 300-epoch seeds (11, 22, and 33).}",
        r"\label{tab:main-r090}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Model & Arm & Seeds & Recall & FPR & FPPI-neg & mAP50 & mAP50-95 \\",
        r"\midrule",
    ]
    for model in MODELS:
        for arm in ARMS:
            row = r090[(r090["model"].eq(model)) & (r090["arm"].eq(arm))].iloc[0]
            rows.append(row.to_dict())
            lines.append(
                f"{tex_escape(MODEL_LABELS[model])} & {tex_escape(ARM_LABELS[arm])} & {row['seeds']} & "
                f"{fmt_metric(row, 'recall')} & {fmt_metric(row, 'FPR')} & {fmt_metric(row, 'FPPI')} & "
                f"{fmt_metric(row, 'mAP50')} & {fmt_metric(row, 'mAP50_95')} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (FIG_DIR / "table_main_r090.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(FIG_DIR / "table_main_r090.csv", index=False)


def table_figlib_event(figlib: pd.DataFrame) -> None:
    rows = []
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{FIGlib event-level alarm burden at R0.90. Clustered FAR is per proxy-hour; TTD is median time-to-detection; all rows use seeds 11, 22, and 33.}",
        r"\label{tab:figlib-event}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Arm & Seeds & Event recall & FPR & FPPI-neg & Clustered FAR / TTD \\",
        r"\midrule",
    ]
    for model in MODELS:
        for arm in ARMS:
            row = figlib[(figlib["model"].eq(model)) & (figlib["arm"].eq(arm))].iloc[0]
            rows.append(row.to_dict())
            lines.append(
                f"{tex_escape(MODEL_LABELS[model])} & {tex_escape(ARM_LABELS[arm])} & {row['seeds']} & "
                f"{fmt_metric(row, 'event_recall')} & {fmt_metric(row, 'FPR')} & {fmt_metric(row, 'FPPI')} & "
                f"{fmt_pm2(row['clustered_far_mean'], row['clustered_far_std'])} / "
                f"{fmt_pm2(row['median_ttd_mean'], row['median_ttd_std'])} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (FIG_DIR / "table_figlib_event.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(FIG_DIR / "table_figlib_event.csv", index=False)


def table_ablation(summary: pd.DataFrame) -> None:
    r090 = summary[summary["target_recall"].eq(0.90)].copy()
    short_model = {
        "YOLO26n": "YOLO26n",
        "RT-DETR-L": "RT-DETR-L",
        "InternImage-FRCNN": "InternImage",
    }
    rows = []
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{D-Fire R0.90 false-alarm reduction relative to each model's baseline. Larger reduction factors indicate lower FPR after hard-negative exposure.}",
        r"\label{tab:ablation-deltas}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Base FPR & HN FPR & Best \\",
        r"\midrule",
    ]
    for model in MODELS:
        base = r090[(r090["model"].eq(model)) & (r090["arm"].eq("baseline"))].iloc[0]
        hard = r090[(r090["model"].eq(model)) & (r090["arm"].eq("hardneg"))].iloc[0]
        sched = r090[(r090["model"].eq(model)) & (r090["arm"].eq("hardneg_sched"))].iloc[0]
        best = hard if hard["FPR_mean"] <= sched["FPR_mean"] else sched
        factor = base["FPR_mean"] / best["FPR_mean"]
        rows.append(
            {
                "model": model,
                "baseline_fpr": base["FPR_mean"],
                "hardneg_fpr": hard["FPR_mean"],
                "hardneg_sched_fpr": sched["FPR_mean"],
                "best_reduction_factor": factor,
            }
        )
        lines.append(
            f"{tex_escape(short_model[model])} & {fmt_pm(base['FPR_mean'], base['FPR_std'])} & "
            f"{fmt_pm(hard['FPR_mean'], hard['FPR_std'])} & {factor:.1f}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (FIG_DIR / "table_ablation_deltas.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(FIG_DIR / "table_ablation_deltas.csv", index=False)


def table_training_budget() -> None:
    rows = [
        {
            "model": model,
            "datasets": "D-Fire + FIGlib",
            "arms": 4,
            "seeds": "11,22,33",
            "epochs": 300,
        }
        for model in MODELS
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Study matrix used in the updated manuscript. Each detector is evaluated under the four training arms on both datasets with three seeds.}",
        r"\label{tab:training-budget}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Arms & Seeds & Epochs \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(MODEL_LABELS[row['model']])} & {row['arms']} & {row['seeds']} & {row['epochs']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (FIG_DIR / "table_training_budget.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(FIG_DIR / "table_training_budget.csv", index=False)


def table_ap_vs_fpr(summary: pd.DataFrame) -> None:
    r090 = summary[summary["target_recall"].eq(0.90)].copy()
    rows = []
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{D-Fire AP/mAP versus fixed-recall false-alarm burden. The false-alarm reductions are much larger than the corresponding AP changes, motivating fixed-recall reporting as a complement to AP/mAP.}",
        r"\label{tab:map-vs-fpr}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Arm & mAP50 & mAP50-95 & R0.90 FPR \\",
        r"\midrule",
    ]
    for model in MODELS:
        for arm in ["baseline", "hardneg", "hardneg_sched"]:
            row = r090[(r090["model"].eq(model)) & (r090["arm"].eq(arm))].iloc[0]
            rows.append(row.to_dict())
            lines.append(
                f"{tex_escape(MODEL_LABELS[model])} & {tex_escape(ARM_LABELS[arm])} & "
                f"{fmt_metric(row, 'mAP50')} & {fmt_metric(row, 'mAP50_95')} & "
                f"{fmt_metric(row, 'FPR')} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (FIG_DIR / "table_map_vs_fpr.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(FIG_DIR / "table_map_vs_fpr.csv", index=False)


def table_completion_gate() -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Completed stress-matrix scope. All rows use three detector families and seeds 11, 22, and 33.}",
        r"\label{tab:completion-gate}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Component & Models and seeds & Manuscript role \\",
        r"\midrule",
        r"D-Fire fixed-recall scoring & 3 models $\times$ 3 seeds & main image-level result \\",
        r"FIGlib event scoring & 3 models $\times$ 3 seeds & external event-level stress result \\",
        r"S1 visual degradation & 3 models $\times$ 3 seeds & robustness pressure \\",
        r"S2 semantic distractors & 3 models $\times$ 3 seeds & external false-alarm pressure \\",
        r"S3 temporal aggregation & 3 models $\times$ 3 seeds & alarm-timing pressure \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    (FIG_DIR / "table_completion_gate.tex").write_text("\n".join(lines), encoding="utf-8")


def latex_includes() -> None:
    text = r"""
% === Fig. 1: Fixed-recall protocol ===
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig1_fixed_recall_protocol.pdf}
    \caption{Recall-anchored stress-evaluation protocol. A calibration split selects the threshold required to reach the target recall; a held-out test split reports recall, false-positive rate (FPR), and false positives per image (FPPI).}
    \label{fig:fixed-recall-protocol}
\end{figure*}

% === Fig. 2: R0.90 false-alarm burden ===
\begin{figure}[t]
    \centering
    \includegraphics[width=0.98\columnwidth]{figures/fig2_r090_false_alarm_burden.pdf}
    \caption{False-alarm burden at the R0.90 operating point on D-Fire. Points show mean \(\pm\) sample standard deviation over seeds 11, 22, and 33 on a logarithmic FPR axis.}
    \label{fig:r090-false-alarm}
\end{figure}

% === Fig. 3: FIGlib event alarm curve ===
\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\columnwidth]{figures/fig3_figlib_event_alarm_curve.pdf}
    \caption{FIGlib event-level stress across training arms. Panel a reports clustered false alarms per proxy-hour; panel b reports event recall relative to the R0.90 target. Points show mean \(\pm\) sample standard deviation over seeds 11, 22, and 33.}
    \label{fig:figlib-event-curve}
\end{figure}
"""
    (FIG_DIR / "latex_includes.tex").write_text(text.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = combined_fixed_recall_summary()
    figlib = figlib_event_summary()
    table_main(summary)
    table_figlib_event(figlib)
    table_ablation(summary)
    table_training_budget()
    table_ap_vs_fpr(summary)
    table_completion_gate()
    latex_includes()
    print(f"Saved LaTeX tables and snippets under {FIG_DIR}")
