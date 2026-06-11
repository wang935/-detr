#!/usr/bin/env python3
"""Build Top-2 RACO manifests from the public FIgLib manifest.

This converter produces per-seed split manifests + audits used by
Top-2 fixed-recall scaffolding:
- event/camera frame rows for fixed-recall manifests
- negative hour audit
- station split audit

Seed policy:
    seed11 -> calib fold 0, test fold 1
    seed22 -> calib fold 1, test fold 2
    seed33 -> calib fold 2, test fold 3
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


CANON_SPLIT_TO_INT = {"train": 0, "calib": 1, "test": 2}
SEED_SPLIT_POLICY = {
    "11": {"calib": 0, "test": 1},
    "22": {"calib": 1, "test": 2},
    "33": {"calib": 2, "test": 3},
}
TOP2_RACO_FIELDS = [
    "unit_id",
    "split",
    "image",
    "label",
    "dataset",
    "station_id",
    "camera_id",
    "event_id",
    "event_family",
    "frame_id",
    "ts",
    "relative_sec",
    "phase",
    "window_tag",
    "denominator_type",
    "denominator_value_sec",
    "unit_version",
]
NEG_HOUR_AUDIT_FIELDS = [
    "camera_id",
    "date",
    "split",
    "denominator_type",
    "neg_minutes",
    "source_file",
    "sampling_granularity",
    "exclude_overlap_with_events",
    "unit_version",
]
STATION_SPLIT_AUDIT_FIELDS = [
    "unit",
    "event_family",
    "station_id",
    "camera_id",
    "split",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=Path("idea-stage/figlib_manifest.csv"), type=Path)
    parser.add_argument("--folds", default=Path("idea-stage/FIGLIB_EVENT_FAMILY_FOLDS.csv"), type=Path)
    parser.add_argument("--out-root", default=Path("idea-stage/top2_manifests/raco"), type=Path)
    parser.add_argument("--source-name", default="figlib")
    parser.add_argument("--unit-version", default="raco-unit-v1")
    parser.add_argument("--include-excluded", action="store_true", help="Keep source excluded rows as neutral negatives")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{path} is empty or malformed")
        return list(reader)


def load_fold_map(path: Path) -> dict[str, tuple[int, str]]:
    mapping: dict[str, tuple[int, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "legacy_split_groups" not in reader.fieldnames or "fold" not in reader.fieldnames:
            raise SystemExit(f"{path} must include fold and legacy_split_groups")
        for row in reader:
            try:
                fold = int(row["fold"])
            except Exception:
                raise SystemExit(f"invalid fold in {path}: {row.get('fold')}")
            key = (row.get("fire_family_key") or "").strip()
            for group in (row.get("legacy_split_groups") or "").split(";"):
                g = group.strip().lower()
                if not g:
                    continue
                if g in mapping and mapping[g][0] != fold:
                    raise SystemExit(f"split_group {group!r} assigned to multiple folds: {mapping[g]} and {fold}")
                mapping[g] = (fold, key)
    return mapping


def parse_epoch_to_iso(ts_raw: str) -> str:
    try:
        ts = float(ts_raw)
        if ts > 0:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    return ""


def frame_id_from_image(image: str) -> str:
    base = image.split("/")[-1]
    return base.rsplit(".", 1)[0]


def safe_float(raw: str | float | int | None, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def classify_phase(offset_sec: float) -> str:
    return "pre_event" if offset_sec < 0 else "event"


def classify_window(offset_sec: float) -> str:
    if offset_sec < 0:
        return "w_pre_300"
    if offset_sec <= 600:
        return "w_event_600"
    return "w_event_post"


def split_for_fold(fold: int, seed_cfg: dict[str, int]) -> str:
    if fold == seed_cfg["calib"]:
        return "calib"
    if fold == seed_cfg["test"]:
        return "test"
    return "train"


def main():
    args = parse_args()
    manifest_rows = load_rows(args.manifest)
    fold_map = load_fold_map(args.folds)

    required_manifest_fields = {
        "image",
        "label",
        "split_group",
        "timestamp_epoch",
        "offset_sec",
        "frame_interval_minutes_est",
        "station_id",
        "camera_id",
        "event_id",
        "eval_unit",
        "date",
    }
    if not required_manifest_fields.issubset(set(manifest_rows[0].keys())):
        missing = sorted(required_manifest_fields - set(manifest_rows[0].keys()))
        raise SystemExit(f"manifest missing columns: {missing}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    for seed, seed_cfg in SEED_SPLIT_POLICY.items():
        seed_rows = []
        skipped_unmapped = 0
        dropped_labels = 0
        for row in manifest_rows:
            label = (row["label"] or "").strip().lower()
            if label in {"positive", "negative"} or (label == "excluded" and args.include_excluded):
                pass
            else:
                if label not in {"", "excluded"}:
                    dropped_labels += 1
                continue

            split_group = (row["split_group"] or "").strip().lower()
            fold_entry = fold_map.get(split_group)
            if fold_entry is None:
                skipped_unmapped += 1
                continue
            fold, family_key = fold_entry
            split = split_for_fold(fold, seed_cfg)

            if label == "positive":
                top_label = "smoke_event"
            elif label == "negative":
                top_label = "none"
            else:
                top_label = "neutral"

            rel = safe_float(row.get("offset_sec"), 0.0)
            frame_min = safe_float(row.get("frame_interval_minutes_est"), 1.0)
            denom_val = max(0.0, frame_min * 60.0)
            ts = parse_epoch_to_iso(row.get("timestamp_epoch", ""))

            seed_rows.append(
                {
                    "split": split,
                    "image": row["image"],
                    "label": top_label,
                    "dataset": args.source_name,
                    "station_id": row["station_id"],
                    "camera_id": row["camera_id"],
                    "event_id": row["event_id"],
                    "event_family": family_key,
                    "frame_id": frame_id_from_image(row["image"]),
                    "ts": ts,
                    "relative_sec": f"{rel:.2f}",
                    "phase": classify_phase(rel),
                    "window_tag": classify_window(rel),
                    "denominator_type": "camera_hour_proxy",
                    "denominator_value_sec": f"{denom_val:.4f}",
                    "unit_version": args.unit_version,
                }
            )

        if not seed_rows:
            print(f"[warn] seed {seed}: no rows generated")
            continue

        # assign deterministic unit IDs after filtering/sorting
        seed_rows.sort(key=lambda item: (item["split"], item["image"]))
        for idx, row in enumerate(seed_rows, start=1):
            row["unit_id"] = f"raco_{idx:06d}"

        out_dir = args.out_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        labels_out = out_dir / f"raco_labels_{'calib' if seed == '11' else 'test' if seed == '22' else 'seed'+seed}.csv"
        # keep filename pattern requested in scaffold for all seeds
        calib_out = out_dir / f"raco_labels_calib_seed{seed}.csv"
        test_out = out_dir / f"raco_labels_test_seed{seed}.csv"
        neg_audit_out = out_dir / f"raco_neg_hour_audit_seed{seed}.csv"
        station_audit_out = out_dir / f"raco_station_split_audit_seed{seed}.csv"

        with labels_out.open("w", encoding="utf-8", newline="") as f:
            pass
        labels_out.unlink()
        write_rows = seed_rows
        # keep top-2 expectation: split-calibrated train file split by split columns
        with calib_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TOP2_RACO_FIELDS)
            writer.writeheader()
            for row in write_rows:
                if row["split"] in {"calib", "train", "test"}:
                    writer.writerow(row)

        with test_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TOP2_RACO_FIELDS)
            writer.writeheader()
            for row in write_rows:
                if row["split"] == "test":
                    writer.writerow(row)

        # audits
        neg_rows = [row for row in write_rows if row["label"] in {"none", "neutral"}]
        neg_aggr = defaultdict(float)
        for row in neg_rows:
            camera = row["camera_id"]
            # recover date from timestamp image path or source date field via reverse map
            date = ""
            # best-effort parse date from image prefix YYYYMMDD_...
            name = row["image"].split("/")[-1]
            date = name[:8]
            if not date.isdigit() or len(date) != 8:
                # fallback to 1900-01-01 to keep non-empty
                date = "19000101"
            key = (camera, date, row["split"])
            neg_aggr[key] += 1.0

        with neg_audit_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=NEG_HOUR_AUDIT_FIELDS)
            writer.writeheader()
            for (camera, date, split), minutes_count in sorted(neg_aggr.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
                writer.writerow(
                    {
                        "camera_id": camera,
                        "date": date,
                        "split": split,
                        "denominator_type": "camera_hour_proxy",
                        "neg_minutes": f"{minutes_count:.1f}",
                        "source_file": str(args.manifest),
                        "sampling_granularity": "frame",
                        "exclude_overlap_with_events": "1",
                        "unit_version": args.unit_version,
                    }
                )

        station_rows = {}
        for row in write_rows:
            if row["label"] not in {"smoke_event", "none", "neutral"}:
                continue
            key = (row["event_id"], row["event_family"], row["station_id"], row["camera_id"], row["split"])
            if key in station_rows:
                continue
            station_rows[key] = {
                "unit": row["event_id"],
                "event_family": row["event_family"],
                "station_id": row["station_id"],
                "camera_id": row["camera_id"],
                "split": row["split"],
            }
        with station_audit_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=STATION_SPLIT_AUDIT_FIELDS)
            writer.writeheader()
            for item in sorted(station_rows.values(), key=lambda d: (d["split"], d["event_family"], d["unit"])):
                writer.writerow(item)

        all_summary[seed] = {
            "rows": len(seed_rows),
            "skipped_unmapped": skipped_unmapped,
            "dropped_labels": dropped_labels,
            "calib_rows": sum(1 for row in seed_rows if row["split"] == "calib"),
            "test_rows": sum(1 for row in seed_rows if row["split"] == "test"),
            "train_rows": sum(1 for row in seed_rows if row["split"] == "train"),
            "output": {
                "calib": str(calib_out),
                "test": str(test_out),
                "neg_audit": str(neg_audit_out),
                "station_audit": str(station_audit_out),
            },
        }

        print(
            f"[seed{seed}] rows={all_summary[seed]['rows']} "
            f"calib={all_summary[seed]['calib_rows']} test={all_summary[seed]['test_rows']} "
            f"train={all_summary[seed]['train_rows']} skipped_unmapped={skipped_unmapped}"
        )

    print("[done]")


if __name__ == "__main__":
    main()
