#!/usr/bin/env python3
"""生成 Stage 4B 中文定性误报样例图组。"""
import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets" / "qualitative"


CATEGORIES = {
    "rtdetr_fixed_by_hardneg": "RT-DETR 仅正样本误报，硬负样本训练修正",
    "rtdetr_hardneg_residual": "RT-DETR 硬负样本训练剩余误报",
    "yolo_hardneg_residual": "YOLO26n 硬负样本训练剩余误报",
    "daq_external_added_fp": "DAQ 外部中性负样本新增误报",
}


def load_json(rel):
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"[错误] 缺少结果文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_max_scores(path):
    scores = {}
    counts = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image = row["image"]
            conf = float(row["conf"])
            counts[image] = counts.get(image, 0) + 1
            if conf > scores.get(image, -math.inf):
                scores[image] = conf
    return scores, counts


def score_by_name(score_dict):
    return {Path(path).name: score for path, score in score_dict.items()}


def resolve_image(path):
    p = Path(path)
    if p.exists():
        return p
    text = str(path)
    replacements = [
        (r"D:\fire\D-Fire", str(ROOT / "data" / "D-Fire")),
        (str(ROOT / "data" / "D-Fire"), r"D:\fire\D-Fire"),
    ]
    for old, new in replacements:
        if text.startswith(old):
            candidate = Path(text.replace(old, new, 1))
            if candidate.exists():
                return candidate
    return p


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def read_thresholds():
    rtdetr = load_json("gonogo_clean5.json")
    yolo = load_json("preds_stage3_yolo/yolo_gonogo.json")
    external = load_json("stage3_daq/external_deepquest/stage3_external_summary.json")
    return {
        "rtdetr_baseline": rtdetr["model_A"]["at_fixed_recall"]["0.90"]["thr"],
        "rtdetr_hardneg": rtdetr["model_B"]["at_fixed_recall"]["0.90"]["thr"],
        "yolo_hardneg": yolo["model_B"]["at_fixed_recall"]["0.90"]["thr"],
        "external_raw": external["external"]["raw_hardneg_at_dfire_thr"]["0.90"]["thr"],
        "external_daq": external["external"]["daq_at_dfire_thr"]["0.90"]["thr"],
    }


def build_cases(per_category):
    labels = load_labels(ROOT / "data" / "dfire" / "eval_labels.csv")
    negatives = [row["image"] for row in labels if row["label"] == "none"]

    rtdetr_base, _ = load_max_scores(ROOT / "preds" / "baseline_clean5.csv")
    rtdetr_hard, _ = load_max_scores(ROOT / "preds" / "hardneg_clean5.csv")
    yolo_hard, _ = load_max_scores(ROOT / "preds_stage3_yolo" / "hardneg.csv")
    external_raw, _ = load_max_scores(ROOT / "stage3_daq" / "external_deepquest" / "external_hardneg_raw.csv")
    external_daq, _ = load_max_scores(ROOT / "stage3_daq" / "external_deepquest" / "external_daq.csv")

    yolo_hard_name = score_by_name(yolo_hard)
    thresholds = read_thresholds()

    def base_case(category, image):
        return {
            "category": category,
            "category_cn": CATEGORIES[category],
            "image": str(resolve_image(image)),
            "image_name": Path(image).name,
            "rtdetr_baseline_conf": rtdetr_base.get(image),
            "rtdetr_hardneg_conf": rtdetr_hard.get(image),
            "yolo_hardneg_conf": yolo_hard_name.get(Path(image).name),
            "external_raw_conf": None,
            "external_daq_conf": None,
            "reason": "",
        }

    fixed = []
    residual_rtdetr = []
    residual_yolo = []
    for image in negatives:
        b = rtdetr_base.get(image, 0.0)
        h = rtdetr_hard.get(image, 0.0)
        y = yolo_hard_name.get(Path(image).name, 0.0)
        if b >= thresholds["rtdetr_baseline"] and h < thresholds["rtdetr_hardneg"]:
            case = base_case("rtdetr_fixed_by_hardneg", image)
            case["reason"] = "仅正样本模型在 R0.90 阈值下报警；硬负样本训练后不报警。"
            fixed.append((b - h, b, case))
        if h >= thresholds["rtdetr_hardneg"]:
            case = base_case("rtdetr_hardneg_residual", image)
            case["reason"] = "硬负样本训练后仍超过 RT-DETR R0.90 报警阈值。"
            residual_rtdetr.append((h, case))
        if y >= thresholds["yolo_hardneg"]:
            case = base_case("yolo_hardneg_residual", image)
            case["reason"] = "YOLO26n 硬负样本训练后仍超过 R0.90 报警阈值。"
            residual_yolo.append((y, case))

    external_added = []
    for image, daq_conf in external_daq.items():
        raw_conf = external_raw.get(image, 0.0)
        if daq_conf >= thresholds["external_daq"] and raw_conf < thresholds["external_raw"]:
            case = {
                "category": "daq_external_added_fp",
                "category_cn": CATEGORIES["daq_external_added_fp"],
                "image": str(resolve_image(image)),
                "image_name": Path(image).name,
                "rtdetr_baseline_conf": None,
                "rtdetr_hardneg_conf": None,
                "yolo_hardneg_conf": None,
                "external_raw_conf": raw_conf,
                "external_daq_conf": daq_conf,
                "reason": "外部中性负样本上，原始硬负样本输出未报警，但 DAQ 变换后触发报警。",
            }
            external_added.append((daq_conf - raw_conf, daq_conf, case))

    fixed.sort(reverse=True, key=lambda x: (x[0], x[1]))
    residual_rtdetr.sort(reverse=True, key=lambda x: x[0])
    residual_yolo.sort(reverse=True, key=lambda x: x[0])
    external_added.sort(reverse=True, key=lambda x: (x[0], x[1]))

    selected = []
    selected.extend(case for _, _, case in fixed[:per_category])
    selected.extend(case for _, case in residual_rtdetr[:per_category])
    selected.extend(case for _, case in residual_yolo[:per_category])
    selected.extend(case for _, _, case in external_added[:per_category])

    counts = {
        "negatives": len(negatives),
        "rtdetr_baseline_fp": sum(rtdetr_base.get(i, 0.0) >= thresholds["rtdetr_baseline"] for i in negatives),
        "rtdetr_fixed_by_hardneg": len(fixed),
        "rtdetr_hardneg_residual": len(residual_rtdetr),
        "yolo_hardneg_residual": len(residual_yolo),
        "external_raw_fp": sum(v >= thresholds["external_raw"] for v in external_raw.values()),
        "external_daq_fp": sum(v >= thresholds["external_daq"] for v in external_daq.values()),
        "daq_external_added_fp": len(external_added),
    }
    return selected, counts, thresholds


def load_font(size):
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    lines = []
    current = ""
    for ch in text:
        test = current + ch
        width = draw.textbbox((0, 0), test, font=font)[2]
        if width <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def case_score_text(case, thresholds):
    category = case["category"]
    if category == "rtdetr_fixed_by_hardneg":
        return (
            f"RT-DETR 仅正样本 {fmt(case['rtdetr_baseline_conf'])} / 阈值 {thresholds['rtdetr_baseline']:.4f}；"
            f"硬负样本 {fmt(case['rtdetr_hardneg_conf'])} / 阈值 {thresholds['rtdetr_hardneg']:.4f}"
        )
    if category == "rtdetr_hardneg_residual":
        return f"RT-DETR 硬负样本 {fmt(case['rtdetr_hardneg_conf'])} / 阈值 {thresholds['rtdetr_hardneg']:.4f}"
    if category == "yolo_hardneg_residual":
        return f"YOLO26n 硬负样本 {fmt(case['yolo_hardneg_conf'])} / 阈值 {thresholds['yolo_hardneg']:.4f}"
    return (
        f"外部 raw {fmt(case['external_raw_conf'])} / 阈值 {thresholds['external_raw']:.4f}；"
        f"DAQ {fmt(case['external_daq_conf'])} / 阈值 {thresholds['external_daq']:.4f}"
    )


def render_case(case, thresholds, out_path, size=(900, 680)):
    image_path = Path(case["image"])
    canvas_w, canvas_h = size
    header_h = 138
    font_title = load_font(30)
    font_body = load_font(22)
    font_small = load_font(18)

    canvas = Image.new("RGB", size, (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas_w, header_h), fill=(22, 28, 36))

    if image_path.exists():
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.contain(img, (canvas_w, canvas_h - header_h), method=Image.Resampling.LANCZOS)
        x = (canvas_w - img.width) // 2
        y = header_h + (canvas_h - header_h - img.height) // 2
        canvas.paste(img, (x, y))
    else:
        draw.text((24, header_h + 40), f"图像不存在：{image_path}", fill=(180, 30, 30), font=font_body)

    title = case["category_cn"]
    score = case_score_text(case, thresholds)
    reason = case["reason"]

    draw.text((24, 16), title, fill=(255, 255, 255), font=font_title)
    draw.text((24, 54), case["image_name"], fill=(210, 220, 235), font=font_small)

    y = 82
    for line in wrap_text(draw, score, font_small, canvas_w - 48):
        draw.text((24, y), line, fill=(250, 220, 130), font=font_small)
        y += 24
    for line in wrap_text(draw, reason, font_small, canvas_w - 48):
        if y > header_h - 24:
            break
        draw.text((24, y), line, fill=(225, 230, 235), font=font_small)
        y += 24

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return out_path


def write_cases_csv(path, cases):
    columns = [
        "case_id",
        "category",
        "category_cn",
        "image",
        "image_name",
        "rtdetr_baseline_conf",
        "rtdetr_hardneg_conf",
        "yolo_hardneg_conf",
        "external_raw_conf",
        "external_daq_conf",
        "reason",
        "rendered_image",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cases)


def make_contact_sheet(paths, out_path, title, cols=2):
    if not paths:
        return None
    font = load_font(34)
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img = ImageOps.contain(img, (560, 430), method=Image.Resampling.LANCZOS)
        thumb = Image.new("RGB", (560, 430), (245, 245, 242))
        thumb.paste(img, ((560 - img.width) // 2, (430 - img.height) // 2))
        thumbs.append(thumb)
    rows = math.ceil(len(thumbs) / cols)
    header_h = 70
    sheet = Image.new("RGB", (cols * 560, header_h + rows * 430), (250, 250, 248))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header_h), fill=(22, 28, 36))
    draw.text((24, 18), title, fill=(255, 255, 255), font=font)
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 560
        y = header_h + (idx // cols) * 430
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return out_path


def write_summary(path, cases, counts, thresholds):
    by_cat = {key: [c for c in cases if c["category"] == key] for key in CATEGORIES}
    lines = [
        "# Stage 4B 定性误报图组",
        "",
        "**阶段判断**：当前仍处于方向收敛后的初步证据包阶段，还不是正式定稿后的完整实验阶段。Stage 4B 的作用是检查这个方向是否能被图像样例讲清楚。",
        "",
        "## R0.90 阈值",
        "",
        "| 系统 | 阈值 |",
        "|---|---:|",
        f"| RT-DETR 仅正样本 | {thresholds['rtdetr_baseline']:.4f} |",
        f"| RT-DETR 硬负样本训练 | {thresholds['rtdetr_hardneg']:.4f} |",
        f"| YOLO26n 硬负样本训练 | {thresholds['yolo_hardneg']:.4f} |",
        f"| 外部 raw 硬负样本 | {thresholds['external_raw']:.4f} |",
        f"| 外部 DAQ | {thresholds['external_daq']:.4f} |",
        "",
        "## 自动筛选统计",
        "",
        "| 类别 | 数量 |",
        "|---|---:|",
        f"| D-Fire eval 负样本图像 | {counts['negatives']} |",
        f"| RT-DETR 仅正样本 R0.90 误报图像 | {counts['rtdetr_baseline_fp']} |",
        f"| 被硬负样本训练修正的 RT-DETR 误报图像 | {counts['rtdetr_fixed_by_hardneg']} |",
        f"| RT-DETR 硬负样本训练剩余误报图像 | {counts['rtdetr_hardneg_residual']} |",
        f"| YOLO26n 硬负样本训练剩余误报图像 | {counts['yolo_hardneg_residual']} |",
        f"| 外部 raw 硬负样本误报图像 | {counts['external_raw_fp']} |",
        f"| 外部 DAQ 误报图像 | {counts['external_daq_fp']} |",
        f"| DAQ 相比 raw 新增的外部误报图像 | {counts['daq_external_added_fp']} |",
        "",
        "## 输出图组",
        "",
    ]
    for key, title in CATEGORIES.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- 样例数：{len(by_cat[key])}")
        lines.append(f"- 拼图：`contact_{key}.jpg`")
        lines.append("")
    lines.extend(
        [
            "## 论文解读",
            "",
            "- 第一组图用于说明：只用正样本训练会把很多无火/无烟图像判成报警，硬负样本训练能修正一大批误报。",
            "- 第二、三组图用于说明：硬负样本训练不是万能的，RT-DETR 和 YOLO 仍有剩余误报，需要在论文里诚实讨论。",
            "- 第四组图用于说明：DAQ 在外部中性负样本上会新增误报，因此 DAQ 只能作为 query 校准分析和失败分析，不能作为当前主方法。",
            "",
            "## 产物",
            "",
            "- `cases.csv`：所有样例的置信度和筛选原因。",
            "- `contact_all.jpg`：四类样例总拼图。",
            "- `contact_*.jpg`：各类别拼图。",
            "- 各类别子目录：单张带中文说明的样例图。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="生成 Stage 4B 中文定性误报图组")
    parser.add_argument("--per-category", type=int, default=8, help="每类输出多少张样例图")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cases, counts, thresholds = build_cases(args.per_category)

    rendered = []
    indexed_cases = []
    category_paths = {key: [] for key in CATEGORIES}
    category_index = {key: 0 for key in CATEGORIES}
    for case in cases:
        category = case["category"]
        category_index[category] += 1
        case_id = f"{category}_{category_index[category]:02d}"
        out_path = OUT / category / f"{case_id}.jpg"
        render_case(case, thresholds, out_path)
        row = dict(case)
        row["case_id"] = case_id
        row["rendered_image"] = str(out_path)
        indexed_cases.append(row)
        rendered.append(out_path)
        category_paths[category].append(out_path)

    for category, paths in category_paths.items():
        make_contact_sheet(paths, OUT / f"contact_{category}.jpg", CATEGORIES[category])
    make_contact_sheet(rendered, OUT / "contact_all.jpg", "Stage 4B 定性误报图组")

    write_cases_csv(OUT / "cases.csv", indexed_cases)
    write_summary(OUT / "STAGE4B_QUALITATIVE_SUMMARY.md", indexed_cases, counts, thresholds)

    print(f"[完成] 输出目录：{OUT}")
    print(f"[完成] 样例数量：{len(indexed_cases)}")
    print(f"[完成] 汇总：{OUT / 'STAGE4B_QUALITATIVE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
