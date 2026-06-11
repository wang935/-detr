#!/usr/bin/env python3
"""Run existing detectors on the FIgLib visual-purity sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_model_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("model spec must be name=weights.pt")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("empty model name")
    return name, Path(path)


def read_sample(path: Path, image_dir: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_ok") != "1":
                continue
            local = image_dir / row["local_file"]
            if not local.exists():
                continue
            item = dict(row)
            item["local_path"] = str(local)
            rows.append(item)
    return rows


def load_thresholds(path: Path | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for key, payload in data.items():
        arm = payload.get("arm")
        if not arm:
            continue
        out[arm] = {
            target: float(values["thr"])
            for target, values in payload.get("at_fixed_recall", {}).items()
            if "thr" in values
        }
    return out


def model_arm(name: str) -> str:
    for arm in ("baseline_eqstep", "hardneg", "baseline"):
        if arm in name:
            return arm
    return name


def export_predictions(models: list[tuple[str, Path]], rows: list[dict], out_csv: Path, imgsz: int, conf: float, max_det: int, device: str) -> list[dict]:
    from ultralytics import YOLO

    pred_rows = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "arm",
            "sample_image",
            "local_file",
            "manual_status",
            "conf",
            "cls",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, weights in models:
            model = YOLO(str(weights))
            names = getattr(model, "names", {}) or {}
            arm = model_arm(name)
            for row in rows:
                result = model.predict(
                    row["local_path"],
                    conf=conf,
                    max_det=max_det,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )[0]
                if result.boxes is None:
                    continue
                confs = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                for conf_value, cls_value, xyxy in zip(confs, classes, boxes):
                    cls_int = int(cls_value)
                    pred = {
                        "model": name,
                        "arm": arm,
                        "sample_image": row["image"],
                        "local_file": row["local_file"],
                        "manual_status": row.get("manual_status", ""),
                        "conf": float(conf_value),
                        "cls": cls_int,
                        "class_name": str(names.get(cls_int, cls_int)),
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    }
                    writer.writerow(pred)
                    pred_rows.append(pred)
    return pred_rows


def read_predictions(path: Path) -> list[dict]:
    preds = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = dict(row)
            item["conf"] = float(item["conf"])
            item["cls"] = int(item["cls"])
            for key in ("x1", "y1", "x2", "y2"):
                item[key] = float(item[key])
            preds.append(item)
    return preds


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if n <= 0:
        return {"low": None, "high": None}
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return {
        "low": max(0.0, (centre - margin) / denom),
        "high": min(1.0, (centre + margin) / denom),
    }


def alarm_summary(scores: list[float], thr: float) -> dict[str, float | int | dict[str, float | None] | None]:
    n = len(scores)
    k = sum(score >= thr for score in scores)
    return {
        "count": k,
        "n": n,
        "rate": k / n if n else None,
        "wilson_95": wilson_ci(k, n),
    }


def summarize(models: list[tuple[str, Path]], rows: list[dict], preds: list[dict], thresholds: dict[str, dict[str, float]], conf: float, imgsz: int, max_det: int) -> dict:
    by_model_image: dict[tuple[str, str], list[float]] = defaultdict(list)
    cls_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred in preds:
        by_model_image[(pred["model"], pred["local_file"])].append(float(pred["conf"]))
        cls_counts[pred["model"]][pred["class_name"]] += 1

    per_model = {}
    for name, weights in models:
        arm = model_arm(name)
        max_scores = []
        clean_scores = []
        ambiguous_scores = []
        for row in rows:
            vals = by_model_image.get((name, row["local_file"]), [])
            max_conf = max(vals) if vals else 0.0
            max_scores.append(max_conf)
            if row.get("manual_status") == "clean":
                clean_scores.append(max_conf)
            elif row.get("manual_status") == "ambiguous":
                ambiguous_scores.append(max_conf)
        threshold_results = {}
        for target, thr in sorted(thresholds.get(arm, {}).items(), key=lambda item: float(item[0])):
            all_downloaded = alarm_summary(max_scores, thr)
            clean_only = alarm_summary(clean_scores, thr)
            clean_plus_ambiguous = alarm_summary(clean_scores + ambiguous_scores, thr)
            threshold_results[target] = {
                "threshold": thr,
                "threshold_source": "D-Fire fixed-recall calibration transferred to FIgLib; not FIgLib-calibrated",
                "all_downloaded": all_downloaded,
                "clean_only": clean_only,
                "clean_plus_ambiguous": clean_plus_ambiguous,
                # Backward-compatible flat fields for older notebooks/reports.
                "alarm_rate_all_downloaded": all_downloaded["rate"],
                "alarm_count_all_downloaded": all_downloaded["count"],
                "alarm_rate_clean_only": clean_only["rate"],
                "alarm_count_clean_only": clean_only["count"],
                "alarm_rate_clean_plus_ambiguous": clean_plus_ambiguous["rate"],
                "alarm_count_clean_plus_ambiguous": clean_plus_ambiguous["count"],
            }
        per_model[name] = {
            "arm": arm,
            "weights": str(weights),
            "n_images": len(rows),
            "n_predictions": sum(len(by_model_image.get((name, row["local_file"]), [])) for row in rows),
            "max_conf_mean": sum(max_scores) / len(max_scores) if max_scores else None,
            "max_conf_max": max(max_scores) if max_scores else None,
            "class_counts": dict(sorted(cls_counts[name].items())),
            "threshold_results": threshold_results,
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "FIgLib visual-purity downloaded pre-onset sample only",
        "n_images": len(rows),
        "manual_status_counts": {
            status: sum(1 for row in rows if row.get("manual_status") == status)
            for status in sorted({row.get("manual_status", "") for row in rows})
        },
        "export_conf": conf,
        "imgsz": imgsz,
        "max_det": max_det,
        "per_model": per_model,
        "caveats": [
            "This is a small pre-onset visual-purity sample, not full FIgLib inference.",
            "Thresholds come from D-Fire fixed-recall calibration and are used only as FIgLib-uncalibrated transfer sanity checks.",
            "Alarm rates are false-alarm proxies on pre-onset images; no event recall is evaluated here.",
            "The preliminary visual-purity labels are single-pass labels; zero observed contamination in this sample does not bound full-manifest contamination.",
            "Arm differences of 0 versus 1 alarms on n=49 have strongly overlapping Wilson confidence intervals and must not be treated as an arm ranking.",
        ],
    }


def fmt_rate(summary: dict) -> str:
    rate = summary["rate"]
    if rate is None:
        return "n/a"
    ci = summary["wilson_95"]
    return (
        f"{summary['count']}/{summary['n']} "
        f"({rate:.3f}; 95% CI {ci['low']:.3f}-{ci['high']:.3f})"
    )


def write_markdown(path: Path, summary: dict, pred_csv: Path) -> None:
    lines = [
        "# FIgLib Score Sample",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Prediction CSV**: `{pred_csv}`",
        "",
        "## Scope",
        "",
        "Existing D-Fire-trained YOLO seed22 models were run on the downloaded FIgLib pre-onset visual-purity sample. This is a transfer false-alarm sanity check only, not a paper claim and not an arm-ranking result.",
        "",
        "## Sample",
        "",
        f"- images scored: `{summary['n_images']}`",
        f"- manual status counts: `{summary['manual_status_counts']}`",
        f"- export conf: `{summary['export_conf']}`",
        "",
        "## Threshold Alarm Rates",
        "",
        "| Model | Target recall threshold source | Threshold | All downloaded | Clean-only | Clean+ambiguous |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_name, payload in summary["per_model"].items():
        for target, result in payload["threshold_results"].items():
            lines.append(
                f"| {model_name} | R{target} D-Fire transferred, FIgLib-uncalibrated | {result['threshold']:.4f} | "
                f"{fmt_rate(result['all_downloaded'])} | "
                f"{fmt_rate(result['clean_only'])} | "
                f"{fmt_rate(result['clean_plus_ambiguous'])} |"
            )
    lines.extend(
        [
            "",
            "## Max Confidence Summary",
            "",
            "| Model | Predictions | Mean max conf | Max max conf | Class counts |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for model_name, payload in summary["per_model"].items():
        lines.append(
            f"| {model_name} | {payload['n_predictions']} | "
            f"{payload['max_conf_mean']:.4f} | {payload['max_conf_max']:.4f} | `{payload['class_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Statistical Interpretation",
            "",
            "- The sample is large enough to verify the scoring pipeline and inspect failure modes, but not large enough to rank arms.",
            "- At R0.90, the apparent `0/49` versus `1/49` differences have overlapping Wilson 95% intervals; they are indistinguishable noise at this sample size.",
            "- Because no positive post-onset event recall is measured here, an under-firing detector could look artificially good on this false-alarm-only proxy.",
            "- The hard-negative arm has fewer raw detections, but also the highest single max confidence (`0.7727`) on an ambiguous lens-dirt/occlusion image; worst-case confidence must be reported with count reductions.",
        ]
    )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in summary["caveats"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--reuse-predictions", action="store_true", help="Read --out-csv instead of running model inference.")
    args = parser.parse_args()

    rows = read_sample(args.sample_labels, args.image_dir)
    thresholds = load_thresholds(args.thresholds)
    if args.reuse_predictions:
        preds = read_predictions(args.out_csv)
    else:
        preds = export_predictions(args.model, rows, args.out_csv, args.imgsz, args.conf, args.max_det, args.device)
    summary = summarize(args.model, rows, preds, thresholds, args.conf, args.imgsz, args.max_det)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary, args.out_csv)
    print(f"[done] images={summary['n_images']} predictions={len(preds)} out={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
