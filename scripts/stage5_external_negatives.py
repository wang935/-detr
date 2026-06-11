#!/usr/bin/env python3
"""Evaluate formal Stage 5 models on an external negative-only image source.

This script does not choose new thresholds on the external source. It reuses the
D-Fire calibration thresholds saved in each Stage 5 ``gonogo.json`` and reports
external negative FPR/FPPI at those frozen operating points.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARM_ORDER = ("baseline", "baseline_eqstep", "hardneg")
ARM_KEYS = ("model_A", "model_C", "model_B")
DEFAULT_TOKENS = (
    "neutral,normal,negative,none,other,no_fire,no-fire,"
    "nonfire,non_fire,non-fire,nofire,nor,not_fire"
)


def parse_tokens(text):
    return [t.strip().lower().replace(" ", "_") for t in str(text).split(",") if t.strip()]


def norm_part(text):
    return str(text).strip().lower().replace(" ", "_")


def find_neutral_images(root, neutral_tokens, limit=0):
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"[error] external root not found: {root}")

    found = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        lowered_parts = [norm_part(part) for part in path.relative_to(root).parts]
        aux_parts = {"gt", "mask", "masks", "label", "labels", "annotation", "annotations", "__macosx"}
        if any(part in aux_parts for part in lowered_parts):
            continue
        if norm_part(path.stem).endswith("_gt"):
            continue
        rel_parts = [norm_part(part) for part in path.relative_to(root).parts[:-1]]
        stem = norm_part(path.stem)
        if any(any(tok in part for tok in neutral_tokens) for part in rel_parts) or any(
            tok in stem for tok in neutral_tokens
        ):
            found.append(str(path.resolve()))

    found = sorted(dict.fromkeys(found))
    if limit and limit > 0:
        found = found[:limit]
    return found


def write_external_labels(path, images):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for image in images:
            writer.writerow([image, "none"])


def load_prediction_csv(path, valid_images):
    valid = set(valid_images)
    dets = defaultdict(list)
    rows = 0
    bad = 0
    unknown = 0
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "conf"}.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain image,conf columns")
        for idx, row in enumerate(reader, start=2):
            image = row["image"].strip()
            try:
                conf = float(row["conf"])
            except Exception:
                bad += 1
                if bad <= 5:
                    print(f"[warn] {path}:{idx} non-numeric conf: {row.get('conf')!r}")
                continue
            if not math.isfinite(conf) or conf < 0.0 or conf > 1.0:
                bad += 1
                if bad <= 5:
                    print(f"[warn] {path}:{idx} invalid conf: {conf}")
                continue
            if image not in valid:
                unknown += 1
                if unknown <= 5:
                    print(f"[warn] {path}:{idx} unknown image ignored: {image}")
                continue
            dets[image].append(conf)
            rows += 1
    for image in dets:
        dets[image].sort(reverse=True)
    return dets, {"rows": rows, "images_with_dets": len(dets), "bad": bad, "unknown": unknown}


def neg_metrics_at(dets, images, threshold):
    if not images:
        return 0.0, 0.0
    fires = 0
    total_dets = 0
    for image in images:
        vals = dets.get(image, [])
        hit_count = sum(1 for conf in vals if conf >= threshold)
        total_dets += hit_count
        if hit_count > 0:
            fires += 1
    return fires / len(images), total_dets / len(images)


def export_predictions(family, weight, images, out_csv, imgsz, conf, max_det):
    try:
        import torch
        from ultralytics import RTDETR, YOLO
    except Exception as exc:
        raise SystemExit(f"[error] torch/ultralytics unavailable: {exc}")

    model_cls = RTDETR if family == "rtdetr" else YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(
        f"[export] family={family} weight={weight} images={len(images)} "
        f"device={device}({gpu}) conf>={conf}"
    )
    model = model_cls(str(weight))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for idx, image in enumerate(images, start=1):
            result = model.predict(
                image,
                conf=conf,
                max_det=max_det,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )[0]
            if result.boxes is not None and len(result.boxes):
                for conf_value in result.boxes.conf.detach().cpu().tolist():
                    writer.writerow([image, float(conf_value)])
                    n_rows += 1
            if idx % 100 == 0 or idx == len(images):
                print(f"  ...{idx}/{len(images)} rows={n_rows}")
    print(f"[export] wrote {out_csv} rows={n_rows}")


def collect_runs(result_root):
    runs = []
    for gonogo_path in sorted(Path(result_root).glob("*_seed*/gonogo.json")):
        data = json.loads(gonogo_path.read_text(encoding="utf-8"))
        for key in ARM_KEYS:
            rep = data.get(key)
            if not rep:
                continue
            if rep.get("arm") not in ARM_ORDER:
                continue
            runs.append({"gonogo": gonogo_path, "run_id": gonogo_path.parent.name, "rep": rep})
    return runs


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["arm"], row["target_recall"])].append(row)

    out = []
    for (family, arm, target), group in sorted(groups.items()):
        fpr_mean, fpr_std = mean_std([row["external_FPR"] for row in group])
        fppi_mean, fppi_std = mean_std([row["external_FPPI"] for row in group])
        out.append(
            {
                "family": family,
                "arm": arm,
                "target_recall": target,
                "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda r: int(r["seed"]))),
                "n": len(group),
                "external_FPR_mean": fpr_mean,
                "external_FPR_std": fpr_std,
                "external_FPPI_mean": fppi_mean,
                "external_FPPI_std": fppi_std,
            }
        )
    return out


def paired_deltas(rows):
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["family"], row["seed"], row["target_recall"])][row["arm"]] = row

    out = []
    for (family, seed, target), arms in sorted(by_key.items()):
        hardneg = arms.get("hardneg")
        if not hardneg:
            continue
        for ref_arm in ("baseline", "baseline_eqstep"):
            ref = arms.get(ref_arm)
            if not ref:
                continue
            out.append(
                {
                    "family": family,
                    "seed": seed,
                    "target_recall": target,
                    "reference_arm": ref_arm,
                    "reference_external_FPR": ref["external_FPR"],
                    "hardneg_external_FPR": hardneg["external_FPR"],
                    "delta_external_FPR": hardneg["external_FPR"] - ref["external_FPR"],
                    "reference_external_FPPI": ref["external_FPPI"],
                    "hardneg_external_FPPI": hardneg["external_FPPI"],
                    "delta_external_FPPI": hardneg["external_FPPI"] - ref["external_FPPI"],
                }
            )
    return out


def fmt_mean_std(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def write_summary(path, args, images, agg, deltas):
    lines = [
        "# Stage 5 External Negative Evaluation",
        "",
        f"- external source: `{args.external_source}`",
        f"- external root: `{Path(args.external_root).resolve()}`",
        f"- external negative images: `{len(images)}`",
        "- threshold protocol: D-Fire calibration thresholds reused from Stage 5 `gonogo.json`",
        "- external images are negative-only and are not used for threshold selection or training",
        "",
        "## Aggregate External FPR/FPPI",
        "",
        "| family | arm | target recall | seeds | external FPR | external FPPI |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['family']} | {row['arm']} | {row['target_recall']} | {row['seeds']} | "
            f"{fmt_mean_std(row['external_FPR_mean'], row['external_FPR_std'])} | "
            f"{fmt_mean_std(row['external_FPPI_mean'], row['external_FPPI_std'])} |"
        )
    lines.extend(["", "## R0.90 Paired Deltas", ""])
    r90 = [row for row in deltas if row["target_recall"] == "0.90"]
    if r90:
        lines.extend(
            [
                "| family | seed | reference | delta external FPR | delta external FPPI |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for row in r90:
            lines.append(
                f"| {row['family']} | {row['seed']} | {row['reference_arm']} | "
                f"{row['delta_external_FPR']:+.4f} | {row['delta_external_FPPI']:+.4f} |"
            )
    else:
        lines.append("No reachable R0.90 paired deltas were available.")

    lines.extend(
        [
            "",
            "## Writing Boundary",
            "",
            "- Use this table as external negative-only false-alarm stress evidence.",
            "- Do not interpret it as external recall, detection mAP, or deployment false-alarm rate.",
            "- If external negatives are too easy, report that explicitly and keep the claim limited.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Stage 5 external negative-only FPR/FPPI evaluation")
    ap.add_argument("--result-root", default="formal_results/stage5")
    ap.add_argument("--external-root", default="external_data/BoWFireDataset/dataset/img")
    ap.add_argument("--external-source", default="BoWFire non-fire/fire-like negatives")
    ap.add_argument("--neutral-tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--out-dir", default="formal_results/stage5_external/bowfire")
    ap.add_argument("--targets", default="0.80,0.85,0.90,0.95")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-external-images", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--force-export", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    targets = {f"{float(item):.2f}" for item in parse_tokens(args.targets)}
    images = find_neutral_images(args.external_root, parse_tokens(args.neutral_tokens), args.limit)
    if len(images) < args.min_external_images:
        raise SystemExit(
            f"[error] only found {len(images)} external negative images under {args.external_root}; "
            "check --external-root/--neutral-tokens or lower --min-external-images."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_external_labels(out_dir / "external_labels.csv", images)

    runs = collect_runs(args.result_root)
    if not runs:
        raise SystemExit(f"[error] no Stage 5 gonogo.json found under {args.result_root}")

    metric_rows = []
    for run in runs:
        rep = run["rep"]
        family = rep.get("family")
        arm = rep.get("arm")
        seed = rep.get("seed")
        weight = Path(rep.get("weight", ""))
        if family not in {"yolo", "rtdetr"}:
            print(f"[skip] unknown family in {run['gonogo']}: {family}")
            continue
        if not weight.exists():
            raise SystemExit(f"[error] missing exported weight for {run['run_id']} {arm}: {weight}")

        pred_path = out_dir / run["run_id"] / f"{arm}.external.csv"
        if not args.skip_export and (args.force_export or not pred_path.exists()):
            export_predictions(family, weight, images, pred_path, args.imgsz, args.conf, args.max_det)
        elif not pred_path.exists():
            raise SystemExit(f"[error] external prediction CSV missing: {pred_path}")
        else:
            print(f"[cache] using {pred_path}")

        dets, stats = load_prediction_csv(pred_path, images)
        for target, item in rep.get("at_fixed_recall", {}).items():
            if target not in targets or not item:
                continue
            fpr, fppi = neg_metrics_at(dets, images, float(item["thr"]))
            metric_rows.append(
                {
                    "external_source": args.external_source,
                    "run_id": run["run_id"],
                    "family": family,
                    "seed": seed,
                    "arm": arm,
                    "target_recall": target,
                    "dfire_threshold": item["thr"],
                    "dfire_calib_recall": item.get("calib_recall", ""),
                    "dfire_test_recall": item.get("recall", ""),
                    "dfire_test_FPR": item.get("FPR", ""),
                    "dfire_test_FPPI": item.get("FPPI", ""),
                    "external_n": len(images),
                    "external_FPR": round(float(fpr), 6),
                    "external_FPPI": round(float(fppi), 6),
                    "prediction_rows": stats["rows"],
                    "prediction_images_with_dets": stats["images_with_dets"],
                }
            )

    if not metric_rows:
        raise SystemExit("[error] no reachable fixed-recall operating points found in Stage 5 runs")

    agg = aggregate(metric_rows)
    deltas = paired_deltas(metric_rows)

    write_csv(out_dir / "stage5_external_metrics.csv", metric_rows, list(metric_rows[0].keys()))
    write_csv(out_dir / "stage5_external_aggregate.csv", agg, list(agg[0].keys()))
    if deltas:
        write_csv(out_dir / "stage5_external_paired_deltas.csv", deltas, list(deltas[0].keys()))
    write_summary(out_dir / "STAGE5_EXTERNAL_NEGATIVE_SUMMARY.md", args, images, agg, deltas)

    print(f"[done] wrote {out_dir / 'STAGE5_EXTERNAL_NEGATIVE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
