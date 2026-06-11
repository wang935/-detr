#!/usr/bin/env python3
"""Audit labels before fixed-recall experiments.

Use this before training. It answers whether a dataset has enough positive and
negative units to support recall-constrained reporting and confidence intervals.

Required input columns:
  image,label

Optional columns:
  split,event_id,camera_id,bag_id,scan_id,procedure_id,...
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_csv_list(text):
    return {item.strip().lower() for item in str(text).split(",") if item.strip()}


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "label"}.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain image,label columns")
        return list(reader), list(reader.fieldnames)


def polarity(label, pos_labels, neg_labels):
    value = str(label).strip().lower()
    if value in pos_labels:
        return "pos"
    if value in neg_labels:
        return "neg"
    return "ignored"


def audit(rows, fieldnames, pos_labels, neg_labels, unit_col=None, split_col=None):
    label_counts = Counter(str(row.get("label", "")).strip().lower() for row in rows)
    image_polarity = Counter(polarity(row.get("label", ""), pos_labels, neg_labels) for row in rows)
    unit_col = unit_col if unit_col else "image"
    if unit_col != "image" and unit_col not in fieldnames:
        raise SystemExit(f"[error] missing unit column: {unit_col}")
    if split_col and split_col not in fieldnames:
        raise SystemExit(f"[error] missing split column: {split_col}")

    units = {}
    mixed_units = defaultdict(set)
    split_counts = defaultdict(Counter)
    for row in rows:
        pol = polarity(row.get("label", ""), pos_labels, neg_labels)
        if pol == "ignored":
            continue
        unit = str(row.get(unit_col if unit_col != "image" else "image", "")).strip()
        if not unit:
            continue
        previous = units.get(unit)
        if previous and previous != pol:
            mixed_units[unit].update([previous, pol])
        units[unit] = pol if previous in (None, pol) else "mixed"
        split = str(row.get(split_col, "all")).strip() if split_col else "all"
        split_counts[split][pol] += 1

    unit_counts = Counter(units.values())
    n_pos_units = unit_counts.get("pos", 0)
    n_neg_units = unit_counts.get("neg", 0)
    perfect_recall_lower95 = wilson_interval(n_pos_units, n_pos_units)[0] if n_pos_units else 0.0
    zero_fpr_upper95 = wilson_interval(0, n_neg_units)[1] if n_neg_units else 1.0
    return {
        "n_rows": len(rows),
        "columns": fieldnames,
        "unit_col": unit_col,
        "split_col": split_col or None,
        "label_counts": dict(sorted(label_counts.items())),
        "image_polarity_counts": dict(sorted(image_polarity.items())),
        "unit_counts": dict(sorted(unit_counts.items())),
        "n_mixed_units": len(mixed_units),
        "sample_mixed_units": sorted(mixed_units)[:10],
        "split_counts": {key: dict(value) for key, value in sorted(split_counts.items())},
        "if_all_positive_units_detected": {
            "n_pos_units": n_pos_units,
            "recall_point_estimate": 1.0 if n_pos_units else 0.0,
            "wilson_lower95": perfect_recall_lower95,
        },
        "if_zero_negative_units_alarm": {
            "n_neg_units": n_neg_units,
            "FPR_point_estimate": 0.0,
            "wilson_upper95": zero_fpr_upper95,
        },
    }


def write_markdown(path, result):
    lines = [
        "# Fixed-Recall Label Audit",
        "",
        f"- rows: `{result['n_rows']}`",
        f"- unit column: `{result['unit_col']}`",
        f"- split column: `{result['split_col'] or 'none'}`",
        f"- mixed units: `{result['n_mixed_units']}`",
        "",
        "## Unit Counts",
        "",
        "| polarity | units |",
        "|---|---:|",
    ]
    for key, value in result["unit_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Confidence Bound Sanity",
            "",
            "| hypothetical result | units | 95% Wilson bound |",
            "|---|---:|---:|",
            (
                "| all positive units detected | "
                f"{result['if_all_positive_units_detected']['n_pos_units']} | "
                f"{result['if_all_positive_units_detected']['wilson_lower95']:.4f} lower recall |"
            ),
            (
                "| zero negative units alarm | "
                f"{result['if_zero_negative_units_alarm']['n_neg_units']} | "
                f"{result['if_zero_negative_units_alarm']['wilson_upper95']:.4f} upper FPR |"
            ),
            "",
            "## Label Counts",
            "",
            "| label | rows |",
            "|---|---:|",
        ]
    )
    for key, value in result["label_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Split Counts", "", "| split | pos rows | neg rows |", "|---|---:|---:|"])
    for split, counts in result["split_counts"].items():
        lines.append(f"| {split} | {counts.get('pos', 0)} | {counts.get('neg', 0)} |")
    if result["sample_mixed_units"]:
        lines.extend(["", "## Mixed Unit Samples", ""])
        for unit in result["sample_mixed_units"]:
            lines.append(f"- `{unit}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pos-labels", required=True)
    parser.add_argument("--neg-labels", required=True)
    parser.add_argument("--unit-col", default=None)
    parser.add_argument("--split-col", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    rows, fieldnames = read_rows(args.labels)
    result = audit(
        rows,
        fieldnames,
        parse_csv_list(args.pos_labels),
        parse_csv_list(args.neg_labels),
        unit_col=args.unit_col,
        split_col=args.split_col,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] wrote {args.out}")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.out_md, result)
        print(f"[done] wrote {args.out_md}")


if __name__ == "__main__":
    main()
