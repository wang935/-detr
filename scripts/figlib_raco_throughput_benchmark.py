#!/usr/bin/env python3
"""Benchmark the FIgLib RACO scoring runner on a bounded cached subset.

The goal is not a formal detector-speed benchmark. This creates an auditable
pre-flight gate before full-manifest scoring by measuring the same runner,
weights, manifest selection, and artifact contract used by the frozen RACO
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
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


def parse_ints(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("batch sizes must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return values


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def count_cache_files(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for path in cache_dir.rglob("*") if path.is_file())


def unlink_if_exists(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def run_variant(args: argparse.Namespace, batch_size: int, rect: bool) -> dict:
    tag = f"{'RECT_' if rect else ''}BATCH{batch_size}"
    image_csv = args.out_dir / f"{args.prefix}_{tag}_IMAGE_SCORES.csv"
    det_csv = args.out_dir / f"{args.prefix}_{tag}_DETECTIONS.csv"
    plan_json = args.out_dir / f"{args.prefix}_{tag}_DRYRUN.json"
    plan_md = args.out_dir / f"{args.prefix}_{tag}_DRYRUN.md"
    selected_manifest = args.out_dir / f"{args.prefix}_{tag}_SELECTED_MANIFEST.csv"

    unlink_if_exists([image_csv, det_csv, plan_json, plan_md, selected_manifest])

    cmd = [
        sys.executable,
        str(args.runner),
        "--manifest",
        str(args.manifest),
        "--folds",
        str(args.folds),
        "--model",
        f"{args.model[0]}={args.model[1]}",
        "--cache-dir",
        str(args.cache_dir),
        "--out-image-csv",
        str(image_csv),
        "--out-detections-csv",
        str(det_csv),
        "--plan-json",
        str(plan_json),
        "--plan-md",
        str(plan_md),
        "--selected-manifest",
        str(selected_manifest),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--max-det",
        str(args.max_det),
        "--device",
        args.device,
        "--batch-size",
        str(batch_size),
        "--per-fold-label-limit",
        str(args.per_fold_label_limit),
    ]
    if rect:
        cmd.append("--rect")
    start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=args.workdir, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise SystemExit(
            f"[error] benchmark variant {tag} failed with code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    image_rows = read_csv(image_csv)
    det_rows = read_csv(det_csv)
    scored_ok = Counter(row.get("scored_ok", "") for row in image_rows)
    errors = [row.get("error", "") for row in image_rows if row.get("error", "")]
    model_image_rows = len(image_rows)
    rows_per_second = model_image_rows / elapsed if elapsed > 0 else 0.0
    full_one_arm_seconds = args.full_selected_images / rows_per_second if rows_per_second else None
    full_three_arm_seconds = args.full_selected_images * args.full_arm_count / rows_per_second if rows_per_second else None

    return {
        "tag": tag,
        "batch_size": batch_size,
        "rect": rect,
        "elapsed_seconds": elapsed,
        "selected_images": json.loads(plan_json.read_text(encoding="utf-8"))["selected_images"],
        "image_score_rows": model_image_rows,
        "detection_rows": len(det_rows),
        "scored_ok_counts": dict(scored_ok),
        "error_row_count": len(errors),
        "rows_per_second": rows_per_second,
        "estimated_full_one_arm_hours": None if full_one_arm_seconds is None else full_one_arm_seconds / 3600.0,
        "estimated_full_three_arm_hours": None if full_three_arm_seconds is None else full_three_arm_seconds / 3600.0,
        "image_scores_csv": str(image_csv),
        "detections_csv": str(det_csv),
        "dryrun_json": str(plan_json),
        "selected_manifest": str(selected_manifest),
        "stdout_tail": completed.stdout.strip().splitlines()[-3:],
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(args: argparse.Namespace, variants: list[dict], cache_before: int, cache_after: int) -> None:
    best = max(variants, key=lambda item: item["rows_per_second"])
    cache_expanded = cache_after > cache_before
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded throughput gate for FIgLib RACO full scoring",
        "workdir": str(args.workdir),
        "runner": str(args.runner),
        "manifest": str(args.manifest),
        "folds": str(args.folds),
        "model": {"name": args.model[0], "weights": str(args.model[1])},
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "max_det": args.max_det,
        "per_fold_label_limit": args.per_fold_label_limit,
        "full_selected_images": args.full_selected_images,
        "full_arm_count": args.full_arm_count,
        "cache_files_before": cache_before,
        "cache_files_after": cache_after,
        "cache_expanded_during_benchmark": cache_expanded,
        "variants": variants,
        "best_variant": best["tag"],
        "decision": {
            "batch_runner_available": True,
            "all_rows_scored_ok": all(item["scored_ok_counts"] == {"1": item["image_score_rows"]} for item in variants),
            "recommended_batch_size": best["batch_size"],
            "recommended_rect": best["rect"],
            "estimated_full_three_arm_hours": best["estimated_full_three_arm_hours"],
            "caveat": "Subset timing gate only; full run must still pass strict coverage and may be slower when downloading uncached images.",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RACO Batch Throughput Gate",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Device**: `{args.device}`",
        f"**Model**: `{args.model[0]}`",
        "",
        "## Scope",
        "",
        "Bounded timing gate for the exact FIgLib RACO scoring runner. This is a pre-flight engineering artifact, not a detector-performance claim.",
        "",
        "## Benchmark Setup",
        "",
        f"- selected manifest rule: `per_fold_label_limit={args.per_fold_label_limit}`",
        f"- full scoring reference: `{args.full_selected_images}` images x `{args.full_arm_count}` arms",
        f"- cache files before/after: `{cache_before}` -> `{cache_after}`",
        f"- cache expanded during benchmark: `{cache_expanded}`",
        f"- image size: `{args.imgsz}`; confidence floor: `{args.conf}`; max detections: `{args.max_det}`",
        "",
        "## Results",
        "",
        "| Variant | Rows | Detections | Elapsed s | Rows/s | Est. 1 arm h | Est. 3 arms h | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in variants:
        lines.append(
            f"| {item['tag']} | {item['image_score_rows']} | {item['detection_rows']} | "
            f"{fmt(item['elapsed_seconds'])} | {fmt(item['rows_per_second'])} | "
            f"{fmt(item['estimated_full_one_arm_hours'])} | "
            f"{fmt(item['estimated_full_three_arm_hours'])} | {item['error_row_count']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Recommended variant for the next full-scoring attempt: `{best['tag']}`.",
            f"- Estimated cached-inference time for three arms: `{fmt(best['estimated_full_three_arm_hours'])}` hours.",
            "- Full scoring is allowed to proceed only as an engineering run until the output has one successful image row for every selected image/model and `figlib_raco_frozen_eval.py --strict-coverage` passes.",
            "- Timing variants that run after cache expansion are cached-inference estimates; uncached full-run download latency remains a separate risk.",
            "- Use resume mode and preserve both image-level and detection-level CSVs.",
            "",
            "## Not A Paper Claim",
            "",
            "No arm ranking, FAR value, recall value, or detector-quality conclusion is supported by this benchmark.",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--runner", type=Path, default=Path("scripts/figlib_raco_score_manifest.py"))
    parser.add_argument("--manifest", type=Path, default=Path("idea-stage/figlib_manifest.csv"))
    parser.add_argument("--folds", type=Path, default=Path("idea-stage/FIGLIB_EVENT_FAMILY_FOLDS.csv"))
    parser.add_argument("--model", type=parse_model_spec, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("idea-stage/figlib_frame_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("idea-stage"))
    parser.add_argument("--prefix", default="RACO_SCORING_THROUGHPUT_GPU_BATCH_GATE")
    parser.add_argument("--batch-sizes", type=parse_ints, default=parse_ints("1,8,16,32"))
    parser.add_argument("--rect-batch-sizes", type=parse_ints, default=parse_ints("16"))
    parser.add_argument("--per-fold-label-limit", type=int, default=2)
    parser.add_argument("--full-selected-images", type=int, default=36454)
    parser.add_argument("--full-arm-count", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    args.workdir = args.workdir.resolve()
    args.runner = (args.workdir / args.runner).resolve() if not args.runner.is_absolute() else args.runner
    args.manifest = (args.workdir / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    args.folds = (args.workdir / args.folds).resolve() if not args.folds.is_absolute() else args.folds
    args.cache_dir = (args.workdir / args.cache_dir).resolve() if not args.cache_dir.is_absolute() else args.cache_dir
    args.out_dir = (args.workdir / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    args.out_json = (args.workdir / args.out_json).resolve() if not args.out_json.is_absolute() else args.out_json
    args.out_md = (args.workdir / args.out_md).resolve() if not args.out_md.is_absolute() else args.out_md

    cache_before = count_cache_files(args.cache_dir)
    variants = []
    for batch_size in args.batch_sizes:
        variants.append(run_variant(args, batch_size, rect=False))
    for batch_size in args.rect_batch_sizes:
        variants.append(run_variant(args, batch_size, rect=True))
    cache_after = count_cache_files(args.cache_dir)
    write_report(args, variants, cache_before, cache_after)
    print(f"[done] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
