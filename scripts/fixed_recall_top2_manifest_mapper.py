#!/usr/bin/env python3
"""Convert source CSV to Top-2 manifest schema using a JSON mapping.

Minimal utility so manual field alignment mistakes are reduced.

Usage:
  python scripts/fixed_recall_top2_manifest_mapper.py \
    --mode endo \
    --source source_labels.csv \
    --mapping FIXED_RECALL_Q1_TOP2_FIELD_MAPPING_EXAMPLES.json \
    --mapping-id endo.sun_seg \
    --out idea-stage/top2_manifests/endo/seed11/endo_labels_mapped.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["endo", "raco"], required=True)
    parser.add_argument("--source", required=True, help="Source CSV path")
    parser.add_argument("--mapping", required=True, help="Mapping json path")
    parser.add_argument("--mapping-id", required=True, help="Mapping entry key id")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--target-default-labels", default="", help="Optional default labels if source lacks label field")
    return parser.parse_args()


def load_mapping(mapping_path: str, mode: str, mapping_id: str):
    payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if mode not in payload:
        raise SystemExit(f"mapping file missing mode: {mode}")
    block = payload[mode]
    if mapping_id not in block["entries"]:
        raise SystemExit(f"mapping_id {mapping_id} not in mode {mode}")
    return block["entries"][mapping_id]["source_fields"], block["entries"][mapping_id].get("defaults", {})


def read_csv(path: str):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header")
        rows = list(reader)
    return rows, reader.fieldnames


def apply(mapping, rows, defaults):
    out_rows = []
    for r in rows:
        out = {}
        for dst, src in mapping.items():
            out[dst] = (r.get(src) or "").strip()
        for k, v in defaults.items():
            out[k] = out.get(k, v)
        out_rows.append(out)
    return out_rows


def main():
    args = parse_args()
    source_rows, _ = read_csv(args.source)
    map_cols, defaults = load_mapping(args.mapping, args.mode, args.mapping_id)
    rows = apply(map_cols, source_rows, defaults=defaults)
    if not rows:
        raise SystemExit("No rows to map")

    # append fallback target label if needed
    if args.target_default_labels and "label" in rows[0]:
        for r in rows:
            if not r.get("label"):
                r["label"] = args.target_default_labels

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()

