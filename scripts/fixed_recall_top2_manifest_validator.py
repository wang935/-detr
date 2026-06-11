#!/usr/bin/env python3
"""Top-2 manifest validator for EndoRecall-RC and RACO-Wildfire.

This utility performs light-weight structural checks before fixed-recall evaluation:
 - required column existence
 - required field empties
 - valid split values
 - polarity consistency within units
 - schema-version presence
 - optional denominator checks for RACO

Usage:
  python scripts/fixed_recall_top2_manifest_validator.py \
    --mode endo --manifest idea-stage/top2_manifests/endo/endo_labels_calib_seed11.csv

  python scripts/fixed_recall_top2_manifest_validator.py \
    --mode raco --manifest idea-stage/top2_manifests/raco/raco_labels_calib_seed11.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ENDO_REQUIRED = [
    "unit_id",
    "split",
    "image",
    "label",
    "dataset",
    "procedure_id",
    "polyp_track_id",
    "match_rule_ver",
    "unit_version",
]

RACO_REQUIRED = [
    "unit_id",
    "split",
    "image",
    "label",
    "dataset",
    "station_id",
    "camera_id",
    "event_id",
    "denominator_type",
    "denominator_value_sec",
    "unit_version",
]

SPLITS = {"train", "calib", "test"}

POS_LABELS = {"polyp", "adenoma", "fire_event", "smoke_event", "smoke"}
NEG_LABELS = {"normal", "benign", "background", "none", "neutral"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["endo", "raco"], required=True)
    parser.add_argument("--manifest", required=True, help="Path to csv manifest")
    parser.add_argument("--strict", action="store_true", help="fail on warning")
    parser.add_argument("--out", default=None, help="Optional json report output path")
    parser.add_argument("--unit-col", default=None, help="Override unit column")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--split-col", default="split", help="Split column name")
    parser.add_argument("--id-col", default="unit_id", help="Id column name")
    return parser.parse_args()


def read_rows(path: str):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{path} is empty")
    if reader.fieldnames is None:
        raise RuntimeError(f"{path} missing header row")
    return rows, reader.fieldnames


def validate_common(rows, headers, unit_col, label_col, split_col, id_col, *, mode):
    issues = []
    warnings = []

    required = ENDO_REQUIRED if mode == "endo" else RACO_REQUIRED
    missing = [c for c in required if c not in headers]
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    for idx, row in enumerate(rows, start=2):
        for col in required:
            val = (row.get(col) or "").strip()
            if not val:
                issues.append(f"row {idx}: {col} is empty")
        split = (row.get(split_col) or "").strip()
        if split and split not in SPLITS:
            issues.append(f"row {idx}: split={split} not in {sorted(SPLITS)}")

        if mode == "raco":
            d_type = (row.get("denominator_type") or "").strip()
            d_val = (row.get("denominator_value_sec") or "").strip()
            if d_type not in {"camera_hour_proxy", "true_negative_camera_hour", "na"}:
                issues.append(f"row {idx}: denominator_type={d_type} unsupported")
            if d_type != "na":
                try:
                    v = float(d_val)
                    if v < 0:
                        issues.append(f"row {idx}: negative denominator_value_sec={v}")
                except Exception:
                    issues.append(f"row {idx}: denominator_value_sec invalid ({d_val})")

        if id_col not in row or not (row.get(id_col) or "").strip():
            issues.append(f"row {idx}: {id_col} is empty")
        if unit_col not in row or not (row.get(unit_col) or "").strip():
            issues.append(f"row {idx}: {unit_col} is empty")

    unit_polarity = defaultdict(set)
    for row in rows:
        unit = (row.get(unit_col) or "").strip()
        if not unit:
            continue
        lab = (row.get(label_col) or "").strip().lower()
        if lab in POS_LABELS:
            unit_polarity[unit].add("pos")
        elif lab in NEG_LABELS:
            unit_polarity[unit].add("neg")
        else:
            warnings.append(f"row with unit={unit}: label={lab} ignored in polarity")

    mixed = [u for u, s in unit_polarity.items() if len(s) > 1]
    for u in mixed:
        issues.append(f"unit polarity mixed: {u} has pos+neg labels")

    split_units = defaultdict(set)
    if mode == "endo":
        key = "procedure_id"
        for row in rows:
            split = (row.get(split_col) or "").strip()
            proc = (row.get(key) or "").strip()
            if split and proc:
                split_units[proc].add(split)
        dup = [k for k, v in split_units.items() if len(v) > 1]
        for k in dup:
            warnings.append(f"endo unit split overlap: {key}={k} spans {sorted(split_units[k])}")
    elif mode == "raco":
        key = "event_id"
        for row in rows:
            split = (row.get(split_col) or "").strip()
            event = (row.get(key) or "").strip()
            if split and event:
                split_units[event].add(split)
        dup = [k for k, v in split_units.items() if len(v) > 1]
        for k in dup:
            warnings.append(f"raccoon event overlap: {key}={k} spans {sorted(split_units[k])}")

    return {"issues": issues, "warnings": warnings}


def main():
    args = parse_args()
    rows, headers = read_rows(args.manifest)
    unit_col = args.unit_col or ("polyp_track_id" if args.mode == "endo" else "event_id")
    report = validate_common(
        rows,
        headers,
        unit_col=unit_col,
        label_col=args.label_col,
        split_col=args.split_col,
        id_col=args.id_col,
        mode=args.mode,
    )
    report["n_rows"] = len(rows)
    report["manifest"] = args.manifest
    report["mode"] = args.mode
    report["unit_col"] = unit_col
    report["ok"] = len(report["issues"]) == 0 and (not args.strict or len(report["warnings"]) == 0)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

