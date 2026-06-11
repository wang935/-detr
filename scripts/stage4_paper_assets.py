#!/usr/bin/env python3
"""从现有 Stage 1-3 结果生成中文论文表格和图。"""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets"
TARGETS = ("0.80", "0.85", "0.90", "0.95")


def load_json(rel):
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"[错误] 缺少结果文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def add_rows(rows, method, family, report, source, note=""):
    for target in TARGETS:
        row = report["at_fixed_recall"].get(target)
        if not row:
            continue
        rows.append(
            {
                "method": method,
                "family": family,
                "target_recall": target,
                "threshold": row["thr"],
                "actual_recall": row["recall"],
                "FPR": row["FPR"],
                "FPPI": row["FPPI"],
                "source": source,
                "note": note,
            }
        )


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns:
        output_rows = [{label: row[key] for key, label in columns} for row in rows]
        fieldnames = [label for _, label in columns]
    else:
        output_rows = rows
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(path, rows, title, columns):
    lines = [f"# {title}", ""]
    headers = [label for _, label in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key, _ in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_rows():
    rows = []
    rtdetr = load_json("gonogo_clean5.json")
    daq = load_json("stage2_daq/strong/stage2_daq_strong_summary.json")
    yolo = load_json("preds_stage3_yolo/yolo_gonogo.json")

    add_rows(rows, "RT-DETR 仅正样本", "RT-DETR", rtdetr["model_A"], "gonogo_clean5.json")
    add_rows(rows, "RT-DETR 硬负样本训练", "RT-DETR", rtdetr["model_B"], "gonogo_clean5.json")
    add_rows(
        rows,
        "RT-DETR 硬负样本训练 + DAQ",
        "RT-DETR/DAQ",
        daq["holdout"]["daq_strong"],
        "stage2_daq/strong/stage2_daq_strong_summary.json",
        "仅 D-Fire holdout 有效；外部中性负样本迁移失败",
    )
    add_rows(rows, "YOLO26n 仅正样本", "YOLO", yolo["model_A"], "preds_stage3_yolo/yolo_gonogo.json")
    add_rows(rows, "YOLO26n 硬负样本训练", "YOLO", yolo["model_B"], "preds_stage3_yolo/yolo_gonogo.json")
    return rows


def external_rows():
    ext = load_json("stage3_daq/external_deepquest/stage3_external_summary.json")
    rows = []
    raw = ext["external"]["raw_hardneg_at_dfire_thr"]
    daq = ext["external"]["daq_at_dfire_thr"]
    for target in TARGETS:
        for method, report in (
            ("RT-DETR 硬负样本训练（外部中性负样本）", raw),
            ("RT-DETR 硬负样本训练 + DAQ（外部中性负样本）", daq),
        ):
            row = report.get(target)
            if not row:
                continue
            rows.append(
                {
                    "method": method,
                    "target_recall_fixed_on": "D-Fire 固定召回阈值",
                    "target_recall": target,
                    "threshold": row["thr"],
                    "external_FPR": row["FPR"],
                    "external_FPPI": row["FPPI"],
                    "source": "stage3_daq/external_deepquest/stage3_external_summary.json",
                }
            )
    return rows


def write_r090_claim_summary(path, rows, ext_rows):
    r90 = [r for r in rows if r["target_recall"] == "0.90"]
    ext90 = [r for r in ext_rows if r["target_recall"] == "0.90"]
    lines = [
        "# Stage 4A 论文主张摘要",
        "",
        "## 主实验 R0.90 对比",
        "",
        "| 方法 | FPR | FPPI | 备注 |",
        "|---|---:|---:|---|",
    ]
    for r in r90:
        lines.append(f"| {r['method']} | {r['FPR']:.4f} | {r['FPPI']:.4f} | {r['note']} |")
    lines.extend(
        [
            "",
            "## 外部中性负样本 R0.90",
            "",
            "| 方法 | 外部 FPR | 外部 FPPI |",
            "|---|---:|---:|",
        ]
    )
    for r in ext90:
        lines.append(f"| {r['method']} | {r['external_FPR']:.4f} | {r['external_FPPI']:.4f} |")
    lines.extend(
        [
            "",
            "## 论文安全解读",
            "",
            "- 当前最强、最稳的论文主张是：硬负样本训练能显著降低火焰/烟雾报警误报。",
            "- DAQ 可以作为 DETR query 校准的域内证据和机制分析，但暂时不能当主方法。",
            "- 外部中性负样本失败应作为限制和失败分析报告，而不是隐藏。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}")
        return None
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    r90 = [r for r in rows if r["target_recall"] == "0.90"]
    labels = [r["method"] for r in r90]
    fpr = [r["FPR"] for r in r90]
    fppi = [r["FPPI"] for r in r90]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([i - 0.18 for i in x], fpr, width=0.36, label="FPR@R0.90", color="#3b82f6")
    ax.bar([i + 0.18 for i in x], fppi, width=0.36, label="FPPI@R0.90", color="#ef4444")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, max(max(fpr), max(fppi)) * 1.18)
    ax.set_ylabel("误报指标")
    ax.set_title("固定召回 R0.90 下的误报对比")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = OUT / "fig_r090_fpr_fppi.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = main_rows()
    ext = external_rows()

    main_columns = [
        ("method", "方法"),
        ("family", "模型族"),
        ("target_recall", "目标召回"),
        ("threshold", "阈值"),
        ("actual_recall", "实际召回"),
        ("FPR", "FPR"),
        ("FPPI", "FPPI"),
        ("source", "来源"),
        ("note", "备注"),
    ]
    write_csv(OUT / "table_main_alarm_metrics.csv", rows, main_columns)
    main_md_columns = [
        ("method", "方法"),
        ("family", "模型族"),
        ("target_recall", "目标召回"),
        ("threshold", "阈值"),
        ("actual_recall", "实际召回"),
        ("FPR", "FPR"),
        ("FPPI", "FPPI"),
        ("note", "备注"),
    ]
    write_markdown_table(
        OUT / "TABLE_MAIN_ALARM_METRICS.md",
        rows,
        "主实验报警误报指标",
        main_md_columns,
    )
    external_columns = [
        ("method", "方法"),
        ("target_recall_fixed_on", "召回阈值来源"),
        ("target_recall", "目标召回"),
        ("threshold", "阈值"),
        ("external_FPR", "外部 FPR"),
        ("external_FPPI", "外部 FPPI"),
        ("source", "来源"),
    ]
    write_csv(OUT / "table_external_neutral.csv", ext, external_columns)
    write_markdown_table(
        OUT / "TABLE_EXTERNAL_NEUTRAL.md",
        ext,
        "外部中性负样本误报指标",
        [
            ("method", "方法"),
            ("target_recall_fixed_on", "召回阈值来源"),
            ("target_recall", "目标召回"),
            ("threshold", "阈值"),
            ("external_FPR", "外部 FPR"),
            ("external_FPPI", "外部 FPPI"),
        ],
    )
    write_r090_claim_summary(OUT / "STAGE4A_CLAIM_SUMMARY.md", rows, ext)
    plot = make_plot(rows)

    print(f"[done] wrote {OUT}")
    if plot:
        print(f"[done] plot {plot}")


if __name__ == "__main__":
    main()
