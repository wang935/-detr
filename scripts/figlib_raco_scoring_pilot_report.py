#!/usr/bin/env python3
"""Summarize a bounded FIgLib RACO scoring pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def wilson_low(k: int, n: int, z: float = 1.959963984540054) -> float | None:
    if n <= 0:
        return None
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * ((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) ** 0.5
    return max(0.0, (centre - margin) / denom)


def pilot_power_floor(selected_manifest: Path | None) -> dict:
    if not selected_manifest:
        return {}
    rows = read_csv(selected_manifest)
    pos_events_by_fold = {}
    for row in rows:
        if row.get("label") != "positive":
            continue
        pos_events_by_fold.setdefault(row.get("fold", ""), set()).add(row.get("event_id", ""))
    calib_counts = {}
    for fold in sorted(pos_events_by_fold):
        calib_events = set()
        for other_fold, events in pos_events_by_fold.items():
            if other_fold != fold:
                calib_events.update(events)
        n = len(calib_events)
        calib_counts[fold] = {
            "calib_positive_event_units": n,
            "max_possible_wilson_lcb_if_all_hit": wilson_low(n, n),
        }
    return {
        "positive_event_units_by_fold": {fold: len(events) for fold, events in sorted(pos_events_by_fold.items())},
        "calib_counts_by_heldout_fold": calib_counts,
        "max_lcb_across_folds": max((item["max_possible_wilson_lcb_if_all_hit"] for item in calib_counts.values()), default=None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun-json", type=Path, required=True)
    parser.add_argument("--image-scores", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, default=None)
    parser.add_argument("--smoke-tests-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    dryrun = json.loads(args.dryrun_json.read_text(encoding="utf-8"))
    eval_payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    image_rows = read_csv(args.image_scores)
    det_rows = read_csv(args.detections)
    model_image_keys = [(row["model"], row["image"]) for row in image_rows]
    power = pilot_power_floor(args.selected_manifest)
    smoke = json.loads(args.smoke_tests_json.read_text(encoding="utf-8")) if args.smoke_tests_json else None

    per_model = {}
    for model in sorted({row["model"] for row in image_rows}):
        rows = [row for row in image_rows if row["model"] == model]
        scores = [float(row["max_conf"]) for row in rows]
        per_model[model] = {
            "image_rows": len(rows),
            "detection_rows": sum(1 for row in det_rows if row["model"] == model),
            "detections_from_image_rows": sum(int(row["n_detections"]) for row in rows),
            "zero_score_rows": sum(score == 0.0 for score in scores),
            "max_conf_max": max(scores) if scores else None,
            "max_conf_mean": sum(scores) / len(scores) if scores else None,
            "label_counts": dict(Counter(row["label"] for row in rows)),
        }

    unreachable = {}
    for model, payload in eval_payload["evaluation"].items():
        unreachable[model] = {
            target: item["unreachable_folds"]
            for target, item in payload["pooled_by_target"].items()
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded real FIgLib scoring pilot, not a method claim",
        "selected_images": dryrun["selected_images"],
        "image_score_rows": len(image_rows),
        "unique_model_image_rows": len(set(model_image_keys)),
        "duplicate_model_image_rows": len(model_image_keys) - len(set(model_image_keys)),
        "detection_rows": len(det_rows),
        "scored_ok_counts": dict(Counter(row["scored_ok"] for row in image_rows)),
        "error_counts": dict(Counter(row["error"] for row in image_rows)),
        "zero_score_rows": sum(float(row["max_conf"]) == 0.0 for row in image_rows),
        "by_label": dict(Counter(row["label"] for row in image_rows)),
        "by_fold_label": {f"{fold}|{label}": count for (fold, label), count in Counter((row["fold"], row["label"]) for row in image_rows).items()},
        "per_model": per_model,
        "pre_registered_eval_unreachable_folds": unreachable,
        "pilot_power_floor": power,
        "smoke_tests": smoke,
        "interpretation": [
            "The real scoring runner successfully downloaded/cached the selected images, ran three YOLO seed22 arms on CPU, and preserved true zero scores.",
            "The pre-registered fixed-recall evaluator ran with strict coverage on the selected manifest.",
            "All target recalls are unreachable in this tiny pilot because calibration folds contain too few events for a 0.90/0.95 Wilson-LCB gate.",
            "This pilot validates the scoring/evaluation plumbing, not detector quality or arm superiority.",
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RACO YOLO Seed22 Real Scoring Pilot",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Scope",
        "",
        "Bounded real scoring pilot for the frozen FIgLib RACO pipeline. This validates the data contract and runner behavior only; it is not a detector-quality or arm-superiority claim.",
        "",
        "## Data Contract Check",
        "",
        f"- selected manifest images: `{summary['selected_images']}`",
        f"- image-level score rows: `{summary['image_score_rows']}`",
        f"- unique model-image rows: `{summary['unique_model_image_rows']}`",
        f"- duplicate model-image rows: `{summary['duplicate_model_image_rows']}`",
        f"- detection rows: `{summary['detection_rows']}`",
        f"- true-zero score rows: `{summary['zero_score_rows']}`",
        f"- scored_ok counts: `{summary['scored_ok_counts']}`",
        f"- error counts: `{summary['error_counts']}`",
        "",
        "## Per-Model Summary",
        "",
        "PLUMBING DIAGNOSTIC ONLY - not a detector-quality or arm-comparison table. The pilot has 20 selected images and is not powered for performance ranking.",
        "",
        "| Model | Image rows | Detection rows | Zero scores | Mean max conf | Max conf |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, payload in per_model.items():
        lines.append(
            f"| {model} | {payload['image_rows']} | {payload['detection_rows']} | "
            f"{payload['zero_score_rows']} | {fmt(payload['max_conf_mean'])} | {fmt(payload['max_conf_max'])} |"
        )
    if power:
        lines.extend(
            [
                "",
                "## Pilot Power Floor",
                "",
                "The pilot intentionally cannot satisfy the registered `0.90` or `0.95` Wilson-LCB fixed-recall gate. It has only one positive event unit per fold, so each held-out fold calibrates on four positive event units.",
                "",
                "| Held-out fold | Calibration positive event units | Max possible Wilson LCB if all are detected |",
                "|---:|---:|---:|",
            ]
        )
        for fold, payload in power["calib_counts_by_heldout_fold"].items():
            lines.append(
                f"| {fold} | {payload['calib_positive_event_units']} | "
                f"{fmt(payload['max_possible_wilson_lcb_if_all_hit'])} |"
            )
    lines.extend(
        [
            "",
            "## Pre-Registered Evaluator Check",
            "",
            "The selected manifest was evaluated with the frozen `calib_lcb`, early-window, clustered-FAR evaluator under `--strict-coverage`.",
            "",
            "| Model | R0.90 unreachable folds | R0.95 unreachable folds |",
            "|---|---:|---:|",
        ]
    )
    for model, payload in unreachable.items():
        lines.append(f"| {model} | {payload.get('0.90', 'n/a')} | {payload.get('0.95', 'n/a')} |")
    if smoke:
        lines.extend(
            [
                "",
                "## Runner Smoke Tests",
                "",
                "| Check | Result | Evidence |",
                "|---|---|---|",
                f"| idempotency/resume | {smoke['idempotency']['result']} | image rows {smoke['idempotency']['image_rows_before']} -> {smoke['idempotency']['image_rows_after']}; detection rows {smoke['idempotency']['detection_rows_before']} -> {smoke['idempotency']['detection_rows_after']} |",
                f"| failure injection | {smoke['failure_injection']['result']} | scored_ok={smoke['failure_injection']['scored_ok']}; error persisted={smoke['failure_injection']['error_persisted']} |",
                f"| plus-sign URL/cache | {smoke['url_encoding_cache']['result']} | {smoke['url_encoding_cache']['cached_plus_path_images']}/{smoke['url_encoding_cache']['plus_path_images']} plus-path images cached |",
                f"| selected-manifest balance | {smoke['selected_manifest_balance']['result']} | 2 negative + 2 positive images per fold |",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This pilot is out-of-prereg as a plumbing artifact; it does not confirm a registered hypothesis.",
            "- FAR and Poisson intervals are not interpreted on this tiny selected sample.",
            "- The per-arm confidence table is not an arm comparison and must not be sorted or quoted as a ranking.",
            "- Full scoring still needs a timed throughput estimate and complete strict coverage on all selected manifest images.",
            "",
            "## Interpretation",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["interpretation"])
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Scale from this bounded pilot to full manifest scoring only after the visual-purity expansion is planned; full scoring must still preserve one image-level row per image/model and pass `--strict-coverage` before reporting any fixed-recall number.",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
