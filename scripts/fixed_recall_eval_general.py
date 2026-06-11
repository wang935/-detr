#!/usr/bin/env python3
"""Generic fixed-recall operating-point evaluator.

This is a domain-neutral extension of scripts/fppi_fpr_eval.py. It keeps the
same core protocol but lets the caller define:

- positive/negative label names,
- optional evaluation units such as event_id, bag_id, scan_id, or procedure_id,
- disjoint calibration/test label files.

The operating threshold is chosen on calibration positives only. False-alarm
metrics are then reported on the test split.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


MIN_THRESHOLD = 1e-6


def parse_csv_list(text):
    return {item.strip().lower() for item in str(text).split(",") if item.strip()}


def parse_targets(text):
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0 or value >= 1:
            raise SystemExit(f"[error] target recall must be in (0,1): {value}")
        out.append(value)
    if not out:
        raise SystemExit("[error] no target recalls provided")
    return tuple(out)


def read_labels(path, pos_labels, neg_labels, unit_col=None):
    rows = {}
    units = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain columns: image,label")
        if unit_col and unit_col not in reader.fieldnames:
            raise SystemExit(f"[error] {path} missing unit column: {unit_col}")
        for idx, row in enumerate(reader, start=2):
            image = row["image"].strip()
            label = row["label"].strip().lower()
            if label in pos_labels:
                polarity = "pos"
            elif label in neg_labels:
                polarity = "neg"
            else:
                continue
            unit = row[unit_col].strip() if unit_col else image
            if not unit:
                raise SystemExit(f"[error] {path}:{idx} has empty {unit_col}")
            rows[image] = {"label": label, "polarity": polarity, "unit": unit}
            previous = units.get(unit)
            if previous and previous != polarity:
                raise SystemExit(
                    f"[error] unit {unit!r} mixes positive and negative labels in {path}; "
                    "use a cleaner unit definition"
                )
            units[unit] = polarity
    return rows, units


def read_predictions(path, valid_images=None):
    dets = defaultdict(list)
    valid = set(valid_images) if valid_images is not None else None
    unknown = 0
    bad = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "conf"}.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain columns: image,conf")
        for idx, row in enumerate(reader, start=2):
            image = row["image"].strip()
            if valid is not None and image not in valid:
                unknown += 1
                continue
            try:
                conf = float(row["conf"])
            except Exception:
                bad += 1
                continue
            if not math.isfinite(conf) or conf < 0.0 or conf > 1.0:
                bad += 1
                continue
            dets[image].append(conf)
    for image in dets:
        dets[image].sort(reverse=True)
    return dets, {"unknown_rows": unknown, "bad_rows": bad}


def unit_scores(label_rows, dets):
    scores = defaultdict(list)
    for image, meta in label_rows.items():
        scores[meta["unit"]].extend(dets.get(image, []))
    for unit in scores:
        scores[unit].sort(reverse=True)
    return scores


def max_score(scores, unit):
    vals = scores.get(unit, [])
    return vals[0] if vals else 0.0


def unit_metrics(threshold, units, scores):
    pos_units = [unit for unit, polarity in units.items() if polarity == "pos"]
    neg_units = [unit for unit, polarity in units.items() if polarity == "neg"]
    pos_hits = sum(1 for unit in pos_units if max_score(scores, unit) >= threshold)
    neg_hits = sum(1 for unit in neg_units if max_score(scores, unit) >= threshold)
    neg_dets = sum(sum(1 for conf in scores.get(unit, []) if conf >= threshold) for unit in neg_units)
    recall = pos_hits / len(pos_units) if pos_units else 0.0
    fpr = neg_hits / len(neg_units) if neg_units else 0.0
    fppi = neg_dets / len(neg_units) if neg_units else 0.0
    return {
        "pos_hits": pos_hits,
        "n_pos": len(pos_units),
        "neg_hits": neg_hits,
        "n_neg": len(neg_units),
        "recall": recall,
        "FPR": fpr,
        "FPPI": fppi,
    }


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def choose_thresholds(calib_units, calib_scores, test_units, test_scores, targets):
    observed = {max_score(calib_scores, unit) for unit, polarity in calib_units.items() if polarity == "pos"}
    observed = {value for value in observed if value > MIN_THRESHOLD}
    grid = {idx / 1000 for idx in range(1, 1001)}
    candidates = sorted(observed | grid, reverse=True)
    out = {}
    for target in targets:
        chosen = None
        for threshold in candidates:
            calib = unit_metrics(threshold, calib_units, calib_scores)
            if calib["recall"] >= target:
                test = unit_metrics(threshold, test_units, test_scores)
                chosen = {
                    "threshold": round(float(threshold), 6),
                    "target_recall": target,
                    "calib_recall": round(calib["recall"], 6),
                    "calib_pos_hits": calib["pos_hits"],
                    "calib_n_pos": calib["n_pos"],
                    "test_recall": round(test["recall"], 6),
                    "test_recall_ci95_wilson": [round(v, 6) for v in wilson_interval(test["pos_hits"], test["n_pos"])],
                    "test_pos_hits": test["pos_hits"],
                    "test_n_pos": test["n_pos"],
                    "test_FPR": round(test["FPR"], 6),
                    "test_FPR_ci95_wilson": [round(v, 6) for v in wilson_interval(test["neg_hits"], test["n_neg"])],
                    "test_neg_hits": test["neg_hits"],
                    "test_n_neg": test["n_neg"],
                    "test_FPPI": round(test["FPPI"], 6),
                }
                break
        out[f"{target:.2f}"] = chosen
    return out


def write_csv_summary(path, result):
    rows = []
    for target, item in result["at_fixed_recall"].items():
        row = {"target": target}
        if item:
            row.update(item)
            row["test_recall_ci95_wilson"] = "|".join(map(str, item["test_recall_ci95_wilson"]))
            row["test_FPR_ci95_wilson"] = "|".join(map(str, item["test_FPR_ci95_wilson"]))
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="prediction CSV with image,conf")
    parser.add_argument("--labels", required=True, help="legacy labels CSV or full labels when no split is provided")
    parser.add_argument("--calib-labels", default=None, help="calibration labels CSV")
    parser.add_argument("--test-labels", default=None, help="test labels CSV")
    parser.add_argument("--pos-labels", required=True, help="comma-separated positive labels")
    parser.add_argument("--neg-labels", required=True, help="comma-separated negative labels")
    parser.add_argument("--unit-col", default=None, help="optional labels column for event/bag/scan/procedure unit")
    parser.add_argument("--targets", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--out", default=None, help="JSON output path")
    parser.add_argument("--out-csv", default=None, help="CSV summary output path")
    args = parser.parse_args()

    if bool(args.calib_labels) != bool(args.test_labels):
        raise SystemExit("[error] --calib-labels and --test-labels must be provided together")

    pos_labels = parse_csv_list(args.pos_labels)
    neg_labels = parse_csv_list(args.neg_labels)
    targets = parse_targets(args.targets)

    calib_label_path = args.calib_labels or args.labels
    test_label_path = args.test_labels or args.labels

    calib_rows, calib_units = read_labels(calib_label_path, pos_labels, neg_labels, unit_col=args.unit_col)
    test_rows, test_units = read_labels(test_label_path, pos_labels, neg_labels, unit_col=args.unit_col)
    valid_images = set(calib_rows) | set(test_rows)
    dets, pred_issues = read_predictions(args.pred, valid_images=valid_images)
    calib_scores = unit_scores(calib_rows, dets)
    test_scores = unit_scores(test_rows, dets)

    result = {
        "pred": str(Path(args.pred)),
        "calib_labels": str(Path(calib_label_path)),
        "test_labels": str(Path(test_label_path)),
        "eval_protocol": "legacy_in_sample" if calib_label_path == test_label_path else "calibration_threshold_test_report",
        "unit_col": args.unit_col or "image",
        "pos_labels": sorted(pos_labels),
        "neg_labels": sorted(neg_labels),
        "n_calib_units": len(calib_units),
        "n_test_units": len(test_units),
        "prediction_issues": pred_issues,
        "at_fixed_recall": choose_thresholds(calib_units, calib_scores, test_units, test_scores, targets),
    }

    for target, item in result["at_fixed_recall"].items():
        if not item:
            print(f"{target}: unreachable")
            continue
        print(
            f"{target}: thr={item['threshold']} calib_recall={item['calib_recall']} "
            f"test_recall={item['test_recall']} test_FPR={item['test_FPR']} "
            f"test_FPPI={item['test_FPPI']}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] wrote {args.out}")
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        write_csv_summary(args.out_csv, result)
        print(f"[done] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
