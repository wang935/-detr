#!/usr/bin/env python3
"""审计当前证据包是否足以完成“证据包验证阶段”。"""
import csv
import json
import py_compile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "paper_assets" / "EVIDENCE_PACKAGE_AUDIT.md"
OUT_JSON = ROOT / "paper_assets" / "evidence_package_audit.json"
TOL = 1e-3


REQUIRED_FILES = [
    "gonogo_clean5.json",
    "preds/baseline_clean5.csv",
    "preds/hardneg_clean5.csv",
    "preds_stage3_yolo/baseline.csv",
    "preds_stage3_yolo/hardneg.csv",
    "preds_stage3_yolo/yolo_gonogo.json",
    "stage2_daq/strong/stage2_daq_strong_summary.json",
    "stage3_daq/external_deepquest/stage3_external_summary.json",
    "stage3_daq/external_deepquest/external_hardneg_raw.csv",
    "stage3_daq/external_deepquest/external_daq.csv",
    "data/dfire/eval_labels.csv",
    "paper_assets/STAGE4A_CLAIM_SUMMARY.md",
    "paper_assets/TABLE_MAIN_ALARM_METRICS.md",
    "paper_assets/TABLE_EXTERNAL_NEUTRAL.md",
    "paper_assets/fig_r090_fpr_fppi.png",
    "paper_assets/qualitative/STAGE4B_QUALITATIVE_SUMMARY.md",
    "paper_assets/qualitative/cases.csv",
    "paper_assets/qualitative/contact_all.jpg",
    "refine-logs/FINAL_PROPOSAL.md",
    "refine-logs/RESULT_TO_CLAIM_20260601.md",
    "refine-logs/EXPERIMENT_PLAN.md",
    "refine-logs/EXPERIMENT_TRACKER.md",
]


SCRIPT_FILES = [
    "scripts/stage4_paper_assets.py",
    "scripts/stage4b_qualitative.py",
    "scripts/audit_evidence_package.py",
]


CASE_CATEGORIES = [
    "rtdetr_fixed_by_hardneg",
    "rtdetr_hardneg_residual",
    "yolo_hardneg_residual",
    "daq_external_added_fp",
]


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.notes = []
        self.metrics = {}

    def ok(self, text):
        self.notes.append(text)

    def warn(self, text):
        self.warnings.append(text)

    def error(self, text):
        self.errors.append(text)


def read_json(rel):
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(rel_or_path, encoding="utf-8"):
    path = Path(rel_or_path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def key_name(path_text):
    return Path(path_text).name


def key_full(path_text):
    return str(Path(path_text)).replace("/", "\\").lower()


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


def verify_file_set(audit):
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not path.exists():
            audit.error(f"缺少必要文件：{rel}")
            continue
        if path.is_file() and path.stat().st_size <= 0:
            audit.error(f"文件为空：{rel}")
            continue
        audit.ok(f"文件存在：{rel}")


def verify_scripts_compile(audit):
    for rel in SCRIPT_FILES:
        path = ROOT / rel
        try:
            py_compile.compile(str(path), doraise=True)
            audit.ok(f"脚本可编译：{rel}")
        except Exception as exc:
            audit.error(f"脚本编译失败：{rel}: {exc}")


def load_eval_labels(audit):
    rows = read_csv("data/dfire/eval_labels.csv")
    counts = Counter(row["label"] for row in rows)
    by_name = {}
    duplicate = []
    for row in rows:
        name = key_name(row["image"])
        if name in by_name:
            duplicate.append(name)
        by_name[name] = row["label"]
    total = len(rows)
    pos = sum(1 for row in rows if row["label"] != "none")
    neg = counts["none"]
    audit.metrics["eval_labels"] = {"total": total, "pos": pos, "neg": neg, "counts": dict(counts)}
    if duplicate:
        audit.error(f"eval_labels 存在重复文件名：{duplicate[:5]}")
    if (total, pos, neg) != (7392, 4012, 3380):
        audit.error(f"eval_labels 数量异常：total={total}, pos={pos}, neg={neg}")
    else:
        audit.ok("eval_labels 数量正确：total=7392, pos=4012, neg=3380")
    return by_name


def eval_predictions(pred_rel, labels_by_name, threshold):
    pos_names = [name for name, label in labels_by_name.items() if label != "none"]
    neg_names = [name for name, label in labels_by_name.items() if label == "none"]
    image_hit = defaultdict(bool)
    neg_det_count = 0
    known_rows = 0
    for row in read_csv(pred_rel):
        name = key_name(row["image"])
        label = labels_by_name.get(name)
        if label is None:
            continue
        known_rows += 1
        if float(row["conf"]) >= threshold:
            image_hit[name] = True
            if label == "none":
                neg_det_count += 1
    recall = sum(1 for name in pos_names if image_hit[name]) / len(pos_names)
    fpr = sum(1 for name in neg_names if image_hit[name]) / len(neg_names)
    fppi = neg_det_count / len(neg_names)
    return {"recall": recall, "FPR": fpr, "FPPI": fppi, "known_rows": known_rows}


def verify_model_metrics(audit, labels_by_name):
    rtdetr = read_json("gonogo_clean5.json")
    yolo = read_json("preds_stage3_yolo/yolo_gonogo.json")
    checks = [
        ("RT-DETR 仅正样本", "preds/baseline_clean5.csv", rtdetr["model_A"]["at_fixed_recall"]),
        ("RT-DETR 硬负样本训练", "preds/hardneg_clean5.csv", rtdetr["model_B"]["at_fixed_recall"]),
        ("YOLO26n 仅正样本", "preds_stage3_yolo/baseline.csv", yolo["model_A"]["at_fixed_recall"]),
        ("YOLO26n 硬负样本训练", "preds_stage3_yolo/hardneg.csv", yolo["model_B"]["at_fixed_recall"]),
    ]
    for label, pred_rel, fixed in checks:
        for target, reported in fixed.items():
            actual = eval_predictions(pred_rel, labels_by_name, float(reported["thr"]))
            for metric in ("recall", "FPR", "FPPI"):
                if not close(actual[metric], reported[metric]):
                    audit.error(
                        f"{label} R{target} {metric} 不一致：reported={reported[metric]} actual={actual[metric]:.4f}"
                    )
            if target == "0.90":
                audit.metrics[f"{label} R0.90"] = {
                    "threshold": reported["thr"],
                    "recall": actual["recall"],
                    "FPR": actual["FPR"],
                    "FPPI": actual["FPPI"],
                }
        audit.ok(f"{label} 固定召回指标与预测 CSV 一致")


def verify_stage4_tables(audit):
    rows = read_csv("paper_assets/table_main_alarm_metrics.csv", encoding="utf-8")
    r90 = {row["方法"]: row for row in rows if row["目标召回"] == "0.90"}
    expected = {
        "RT-DETR 仅正样本": ("0.5358", "0.6808"),
        "RT-DETR 硬负样本训练": ("0.1056", "0.1234"),
        "RT-DETR 硬负样本训练 + DAQ": ("0.0811", "0.0811"),
        "YOLO26n 仅正样本": ("0.3704", "0.4920"),
        "YOLO26n 硬负样本训练": ("0.0571", "0.0725"),
    }
    for method, (fpr, fppi) in expected.items():
        row = r90.get(method)
        if not row:
            audit.error(f"主表缺少 R0.90 方法：{method}")
            continue
        if not (close(row["FPR"], fpr) and close(row["FPPI"], fppi)):
            audit.error(f"主表 R0.90 数值不一致：{method}")
    ext_rows = read_csv("paper_assets/table_external_neutral.csv", encoding="utf-8")
    ext90 = {row["方法"]: row for row in ext_rows if row["目标召回"] == "0.90"}
    raw = ext90.get("RT-DETR 硬负样本训练（外部中性负样本）")
    daq = ext90.get("RT-DETR 硬负样本训练 + DAQ（外部中性负样本）")
    if not raw or not daq:
        audit.error("外部中性负样本表缺少 R0.90 raw/DAQ 行")
    else:
        if not (close(raw["外部 FPR"], "0.0730") and close(daq["外部 FPR"], "0.2370")):
            audit.error("外部中性负样本表 R0.90 FPR 不一致")
    audit.ok("Stage 4A 中文表格关键数值一致")


def eval_external(pred_rel, threshold):
    labels = read_csv("stage3_daq/external_deepquest/external_labels.csv")
    names = {key_full(row["image"]) for row in labels}
    hit = defaultdict(bool)
    det_count = 0
    for row in read_csv(pred_rel):
        name = key_full(row["image"])
        if name not in names:
            continue
        if float(row["conf"]) >= threshold:
            hit[name] = True
            det_count += 1
    n = len(names)
    return {"n": n, "FPR": sum(1 for name in names if hit[name]) / n, "FPPI": det_count / n}


def verify_external_metrics(audit):
    summary = read_json("stage3_daq/external_deepquest/stage3_external_summary.json")
    raw = summary["external"]["raw_hardneg_at_dfire_thr"]["0.90"]
    daq = summary["external"]["daq_at_dfire_thr"]["0.90"]
    actual_raw = eval_external("stage3_daq/external_deepquest/external_hardneg_raw.csv", float(raw["thr"]))
    actual_daq = eval_external("stage3_daq/external_deepquest/external_daq.csv", float(daq["thr"]))
    for name, reported, actual in (("raw", raw, actual_raw), ("DAQ", daq, actual_daq)):
        if actual["n"] != 1000:
            audit.error(f"外部中性负样本数量异常：{actual['n']}")
        for metric in ("FPR", "FPPI"):
            if not close(actual[metric], reported[metric]):
                audit.error(f"外部 {name} R0.90 {metric} 不一致：reported={reported[metric]} actual={actual[metric]:.4f}")
    if actual_daq["FPR"] <= actual_raw["FPR"]:
        audit.error("外部 DAQ 未呈现预期失败模式，需重新审查论文叙事")
    else:
        audit.ok("外部中性负样本失败模式成立：DAQ FPR 高于 raw")
    audit.metrics["external_R0.90"] = {"raw": actual_raw, "daq": actual_daq}


def verify_stage4b_cases(audit):
    rows = read_csv("paper_assets/qualitative/cases.csv", encoding="utf-8-sig")
    counts = Counter(row["category"] for row in rows)
    if len(rows) != 32:
        audit.error(f"Stage 4B cases.csv 样例数异常：{len(rows)}")
    for category in CASE_CATEGORIES:
        if counts[category] != 8:
            audit.error(f"Stage 4B 类别样例数异常：{category}={counts[category]}")
    missing = []
    for row in rows:
        rendered = Path(row["rendered_image"])
        if not rendered.exists() or rendered.stat().st_size < 10_000:
            missing.append(str(rendered))
    if missing:
        audit.error(f"Stage 4B 渲染图缺失或过小：{missing[:5]}")
    for category in CASE_CATEGORIES:
        contact = ROOT / "paper_assets" / "qualitative" / f"contact_{category}.jpg"
        if not contact.exists() or contact.stat().st_size < 50_000:
            audit.error(f"Stage 4B 类别拼图异常：{contact}")
    audit.ok("Stage 4B cases.csv、单图和拼图完整")
    audit.metrics["stage4b_cases"] = dict(counts)


def verify_documents(audit):
    claim = (ROOT / "refine-logs" / "RESULT_TO_CLAIM_20260601.md").read_text(encoding="utf-8")
    tracker = (ROOT / "refine-logs" / "EXPERIMENT_TRACKER.md").read_text(encoding="utf-8")
    plan = (ROOT / "refine-logs" / "EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
    if "生成一套统一 eval split" in claim or "挖定性误报图" in claim:
        audit.warn("RESULT_TO_CLAIM 仍含 Stage 4A/4B 之前的缺口表述，需要更新")
    if "Stage 4B | 定性误报图组 | 已完成" not in tracker:
        audit.error("EXPERIMENT_TRACKER 未标记 Stage 4B 已完成")
    if "DAQ 可以作为当前论文的主方法" in claim and "结果不支持的主张" not in claim:
        audit.error("RESULT_TO_CLAIM 含过度 DAQ 主方法表述")
    if "DAQ 优先" in tracker and "不要回到 DAQ 优先叙事" not in tracker:
        audit.warn("追踪文件可能存在 DAQ 优先叙事残留")
    if "当前状态**：已完成，输出在 `paper_assets/qualitative/`" not in plan:
        audit.error("EXPERIMENT_PLAN 未标记 Stage 4B 已完成")
    audit.ok("主线文档未发现阻断性叙事冲突")


def verify_claim_gate(audit):
    metrics = audit.metrics
    rt_base = metrics["RT-DETR 仅正样本 R0.90"]["FPR"]
    rt_hard = metrics["RT-DETR 硬负样本训练 R0.90"]["FPR"]
    y_base = metrics["YOLO26n 仅正样本 R0.90"]["FPR"]
    y_hard = metrics["YOLO26n 硬负样本训练 R0.90"]["FPR"]
    if not (rt_hard < rt_base and y_hard < y_base):
        audit.error("硬负样本训练未在 RT-DETR 和 YOLO 上同时降低 R0.90 FPR")
    else:
        audit.ok("核心主张 C2 成立：硬负样本训练在 RT-DETR 和 YOLO 上均降低 R0.90 FPR")
    daq_ext = metrics["external_R0.90"]["daq"]["FPR"]
    raw_ext = metrics["external_R0.90"]["raw"]["FPR"]
    if daq_ext <= raw_ext:
        audit.error("DAQ 外部失败证据不成立")
    else:
        audit.ok("限制性主张 C3 成立：DAQ 外部中性负样本 FPR 高于 raw")


def write_outputs(audit):
    status = "PASS" if not audit.errors else "FAIL"
    completion = "证据包验证阶段完成" if status == "PASS" else "证据包验证阶段未完成"
    payload = {
        "status": status,
        "completion": completion,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "notes": audit.notes,
        "metrics": audit.metrics,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 证据包自动审计",
        "",
        f"**状态**：{status}",
        f"**结论**：{completion}",
        "",
        "## 错误",
        "",
    ]
    if audit.errors:
        lines.extend(f"- {item}" for item in audit.errors)
    else:
        lines.append("- 无")
    lines.extend(["", "## 警告", ""])
    if audit.warnings:
        lines.extend(f"- {item}" for item in audit.warnings)
    else:
        lines.append("- 无阻断性警告")
    lines.extend(["", "## 关键验证结果", ""])
    for key, value in audit.metrics.items():
        lines.append(f"- `{key}`：`{json.dumps(value, ensure_ascii=False)}`")
    lines.extend(
        [
            "",
            "## 审计说明",
            "",
            "- 本审计验证的是当前证据包能否支撑方向 go/no-go 与论文主线收敛。",
            "- 它不等价于正式投稿前的完整实验矩阵。",
            "- 随机种子重复实验、第二个外部负样本来源和框级可视化属于下一阶段的正式论文增强项。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    audit = Audit()
    verify_file_set(audit)
    verify_scripts_compile(audit)
    labels_by_name = load_eval_labels(audit)
    if not audit.errors:
        verify_model_metrics(audit, labels_by_name)
        verify_stage4_tables(audit)
        verify_external_metrics(audit)
        verify_stage4b_cases(audit)
        verify_documents(audit)
        verify_claim_gate(audit)
    payload = write_outputs(audit)
    print(f"[audit] status={payload['status']} errors={len(audit.errors)} warnings={len(audit.warnings)}")
    print(f"[audit] wrote {OUT_MD}")
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
