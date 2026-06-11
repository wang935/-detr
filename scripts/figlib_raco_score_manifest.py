#!/usr/bin/env python3
"""Score FIgLib manifest images for the frozen-fold RACO evaluator.

This runner is deliberately conservative:

- it selects only manifest rows used by the fixed-recall evaluator
  (`label in {positive, negative}`);
- it writes one image-level score row per processed image, including true
  `0.0` scores when a detector returns no boxes;
- it can also write raw detection rows for recomputing top scores later;
- dry-run mode creates a scoring plan without downloading images or importing
  detector libraries.

Full scoring can be resumed from an existing image-level CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


IMAGE_FIELDS = [
    "image",
    "model",
    "arm",
    "max_conf",
    "n_detections",
    "top_class",
    "top_box_xyxy",
    "fold",
    "label",
    "eval_unit",
    "event_id",
    "sequence_id",
    "offset_sec",
    "frame_url",
    "local_path",
    "scored_ok",
    "error",
]

DET_FIELDS = [
    "image",
    "model",
    "arm",
    "conf",
    "cls",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
]


def parse_model_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("model spec must be name=weights.pt")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("empty model name")
    return name, Path(path)


def parse_csv_set(text: str | None) -> set[str] | None:
    if not text:
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def model_arm(name: str) -> str:
    for arm in ("baseline_eqstep", "hardneg", "baseline"):
        if arm in name:
            return arm
    return name


def read_folds(path: Path | None) -> dict[str, int]:
    if not path:
        return {}
    mapping = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"fold", "legacy_split_groups"}.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain fold,legacy_split_groups")
        for row in reader:
            fold = int(row["fold"])
            for group in row["legacy_split_groups"].split(";"):
                group = group.strip()
                if not group:
                    continue
                previous = mapping.get(group)
                if previous is not None and previous != fold:
                    raise SystemExit(f"[error] split_group {group!r} appears in folds {previous} and {fold}")
                mapping[group] = fold
    return mapping


def read_manifest(
    path: Path,
    fold_map: dict[str, int],
    labels: set[str],
    folds: set[str] | None,
    max_images: int | None,
    per_fold_label_limit: int | None,
) -> tuple[list[dict], dict]:
    rows = []
    missing_folds = Counter()
    group_counts = Counter()
    seen_images = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "image",
            "label",
            "eval_unit",
            "event_id",
            "sequence_id",
            "offset_sec",
            "frame_url",
            "split_group",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain {sorted(required)}")
        for row in reader:
            label = row["label"].strip().lower()
            if label not in labels:
                continue
            image = row["image"].strip()
            if image in seen_images:
                continue
            seen_images.add(image)
            fold = ""
            if fold_map:
                split_group = row["split_group"].strip()
                if split_group not in fold_map:
                    missing_folds[split_group] += 1
                    continue
                fold = str(fold_map[split_group])
                if folds is not None and fold not in folds:
                    continue
            group_key = (fold, label)
            if per_fold_label_limit is not None and group_counts[group_key] >= per_fold_label_limit:
                continue
            item = dict(row)
            item["label"] = label
            item["fold"] = fold
            rows.append(item)
            group_counts[group_key] += 1
            if max_images is not None and len(rows) >= max_images:
                break
    return rows, {
        "selected_images": len(rows),
        "missing_fold_rows": sum(missing_folds.values()),
        "missing_folds": dict(sorted(missing_folds.items())),
        "per_fold_label_limit": per_fold_label_limit,
    }


def existing_done(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "model", "scored_ok"}.issubset(reader.fieldnames):
            return done
        for row in reader:
            if row.get("scored_ok") == "1":
                done.add((row["model"], row["image"]))
    return done


def cache_path(cache_dir: Path, image: str) -> Path:
    return cache_dir / image.replace("/", "\\")


def download(row: dict, cache_dir: Path, retries: int = 3, retry_sleep: float = 1.0) -> Path:
    target = cache_path(cache_dir, row["image"])
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded_url = urllib.parse.quote(row["frame_url"], safe=":/")
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                encoded_url,
                headers={"User-Agent": "detr-q3-figlib-raco-score/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                target.write_bytes(response.read())
            return target
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"download failed for {row['image']} after {retries} attempts: {last_exc}") from last_exc
    return target


def write_plan(path_json: Path, path_md: Path, rows: list[dict], models: list[tuple[str, Path]], meta: dict) -> None:
    by_fold_label = Counter((row.get("fold", ""), row["label"]) for row in rows)
    by_label = Counter(row["label"] for row in rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_images": len(rows),
        "models": [{"name": name, "arm": model_arm(name), "weights": str(weights)} for name, weights in models],
        "by_label": dict(sorted(by_label.items())),
        "by_fold_label": {f"{fold}|{label}": count for (fold, label), count in sorted(by_fold_label.items())},
        "raw_artifact_contract": {
            "image_level_true_zero_scores": True,
            "detection_level_rows": True,
            "resume_from_scored_ok": True,
        },
        "manifest_meta": meta,
    }
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# RACO Full Scoring Dry Run",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Scope",
        "",
        "Dry-run plan for detector scoring. No images are downloaded and no detector inference is run.",
        "",
        "## Selected Images",
        "",
        f"- total selected images: `{len(rows)}`",
    ]
    for label, count in sorted(by_label.items()):
        lines.append(f"- {label}: `{count}`")
    lines.extend(["", "## Models", "", "| Model | Arm | Weights |", "|---|---|---|"])
    for name, weights in models:
        lines.append(f"| {name} | {model_arm(name)} | `{weights}` |")
    lines.extend(["", "## Fold x Label Counts", "", "| Fold | Label | Images |", "|---:|---|---:|"])
    for (fold, label), count in sorted(by_fold_label.items()):
        lines.append(f"| {fold or 'n/a'} | {label} | {count} |")
    lines.extend(
        [
            "",
            "## Persistence Contract",
            "",
            "- image-level CSV must contain one row per processed image/model, including true `0.0` scores;",
            "- detection-level CSV preserves raw boxes and confidences when available;",
            "- final evaluation must use `figlib_raco_frozen_eval.py --strict-coverage`.",
        ]
    )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_header_if_needed(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def result_detections(result, names: dict) -> list[tuple[float, int, str, list[float]]]:
    detections = []
    if result.boxes is None:
        return detections
    confs = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    for conf_value, cls_value, xyxy in zip(confs, classes, boxes):
        cls_int = int(cls_value)
        detections.append((float(conf_value), cls_int, str(names.get(cls_int, cls_int)), xyxy))
    return detections


def write_score_rows(
    image_writer: csv.DictWriter,
    det_writer: csv.DictWriter,
    row: dict,
    name: str,
    arm: str,
    detections: list[tuple[float, int, str, list[float]]],
    local: str,
    scored_ok: str,
    error: str,
) -> None:
    max_conf = max((item[0] for item in detections), default=0.0)
    top = max(detections, default=None, key=lambda item: item[0])
    image_writer.writerow(
        {
            "image": row["image"],
            "model": name,
            "arm": arm,
            "max_conf": f"{max_conf:.8f}",
            "n_detections": len(detections),
            "top_class": "" if top is None else top[2],
            "top_box_xyxy": "" if top is None else ";".join(f"{v:.2f}" for v in top[3]),
            "fold": row.get("fold", ""),
            "label": row["label"],
            "eval_unit": row["eval_unit"],
            "event_id": row["event_id"],
            "sequence_id": row["sequence_id"],
            "offset_sec": row["offset_sec"],
            "frame_url": row["frame_url"],
            "local_path": local,
            "scored_ok": scored_ok,
            "error": error,
        }
    )
    for conf_value, cls_int, class_name, xyxy in detections:
        det_writer.writerow(
            {
                "image": row["image"],
                "model": name,
                "arm": arm,
                "conf": f"{conf_value:.8f}",
                "cls": cls_int,
                "class_name": class_name,
                "x1": f"{xyxy[0]:.2f}",
                "y1": f"{xyxy[1]:.2f}",
                "x2": f"{xyxy[2]:.2f}",
                "y2": f"{xyxy[3]:.2f}",
            }
        )


def chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def download_batch(batch_rows: list[dict], cache_dir: Path, workers: int) -> tuple[list[tuple[dict, str]], list[tuple[dict, Exception]]]:
    if workers <= 1 or len(batch_rows) <= 1:
        ready = []
        failed = []
        for row in batch_rows:
            try:
                ready.append((row, str(download(row, cache_dir))))
            except Exception as exc:  # noqa: BLE001 - caller decides fail policy.
                failed.append((row, exc))
        return ready, failed

    ready_by_index = {}
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_item = {pool.submit(download, row, cache_dir): (idx, row) for idx, row in enumerate(batch_rows)}
        for future in as_completed(future_to_item):
            idx, row = future_to_item[future]
            try:
                ready_by_index[idx] = (row, str(future.result()))
            except Exception as exc:  # noqa: BLE001 - caller decides fail policy.
                failed.append((row, exc))
    ready = [ready_by_index[idx] for idx in sorted(ready_by_index)]
    return ready, failed


def score_rows(args: argparse.Namespace, rows: list[dict]) -> None:
    from ultralytics import YOLO

    append_header_if_needed(args.out_image_csv, IMAGE_FIELDS)
    append_header_if_needed(args.out_detections_csv, DET_FIELDS)
    done = existing_done(args.out_image_csv)

    models = [(name, YOLO(str(weights))) for name, weights in args.model]
    with args.out_image_csv.open("a", newline="", encoding="utf-8") as image_f, args.out_detections_csv.open(
        "a", newline="", encoding="utf-8"
    ) as det_f:
        image_writer = csv.DictWriter(image_f, fieldnames=IMAGE_FIELDS)
        det_writer = csv.DictWriter(det_f, fieldnames=DET_FIELDS)
        for name, model in models:
            arm = model_arm(name)
            names = getattr(model, "names", {}) or {}
            pending = [row for row in rows if (name, row["image"]) not in done]
            for batch_rows in chunks(pending, max(1, args.batch_size)):
                ready, failed = download_batch(batch_rows, args.cache_dir, args.download_workers)
                for row, exc in failed:
                    if args.fail_on_error:
                        raise exc
                    write_score_rows(image_writer, det_writer, row, name, arm, [], "", "0", repr(exc))
                if not ready:
                    image_f.flush()
                    det_f.flush()
                    continue
                try:
                    results = model.predict(
                        [local for _row, local in ready],
                        conf=args.conf,
                        max_det=args.max_det,
                        imgsz=args.imgsz,
                        device=args.device,
                        batch=max(1, args.batch_size),
                        rect=args.rect,
                        verbose=False,
                    )
                    for (row, local), result in zip(ready, results):
                        write_score_rows(
                            image_writer,
                            det_writer,
                            row,
                            name,
                            arm,
                            result_detections(result, names),
                            local,
                            "1",
                            "",
                        )
                except Exception as exc:  # noqa: BLE001 - persisted for resume/debug.
                    if args.fail_on_error:
                        raise
                    for row, local in ready:
                        write_score_rows(image_writer, det_writer, row, name, arm, [], local, "0", repr(exc))
                image_f.flush()
                det_f.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=None)
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--labels", default="positive,negative")
    parser.add_argument("--fold-filter", default=None, help="Comma-separated fold ids to score")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--per-fold-label-limit", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("idea-stage/figlib_frame_cache"))
    parser.add_argument("--out-image-csv", type=Path, required=True)
    parser.add_argument("--out-detections-csv", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--plan-md", type=Path, default=None)
    parser.add_argument("--selected-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    labels = {item.lower() for item in parse_csv_set(args.labels) or {"positive", "negative"}}
    fold_filter = parse_csv_set(args.fold_filter)
    fold_map = read_folds(args.folds)
    rows, meta = read_manifest(args.manifest, fold_map, labels, fold_filter, args.max_images, args.per_fold_label_limit)
    if args.selected_manifest:
        write_selected_manifest(args.selected_manifest, rows)
    if args.plan_json and args.plan_md:
        write_plan(args.plan_json, args.plan_md, rows, args.model, meta)
    if args.dry_run:
        print(f"[dry-run] selected_images={len(rows)} models={len(args.model)}")
        return 0
    score_rows(args, rows)
    print(f"[done] scored selected_images={len(rows)} models={len(args.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
