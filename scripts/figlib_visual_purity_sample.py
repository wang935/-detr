#!/usr/bin/env python3
"""Create a visual purity sample for FIgLib pre-onset negatives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fold_map(folds_csv: Path) -> dict[str, str]:
    mapping = {}
    for row in read_csv(folds_csv):
        fold = row["fold"]
        for group in row.get("legacy_split_groups", "").split(";"):
            if group:
                mapping[group] = fold
    return mapping


def select_samples(manifest_rows: list[dict], split_to_fold: dict[str, str], per_fold: int) -> list[dict]:
    candidates = []
    seen_sequence = set()
    for row in sorted(
        manifest_rows,
        key=lambda item: (
            split_to_fold.get(item.get("split_group", ""), "unknown"),
            abs(int(item["offset_sec"]) + 300) if item.get("offset_sec") else 10**9,
            item.get("sequence_id", ""),
        ),
    ):
        if row.get("label") != "negative":
            continue
        seq = row.get("sequence_id")
        if seq in seen_sequence:
            continue
        fold = split_to_fold.get(row.get("split_group", ""), "unknown")
        if fold == "unknown":
            continue
        out = dict(row)
        out["fold"] = fold
        candidates.append(out)
        seen_sequence.add(seq)

    by_fold: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_fold[row["fold"]].append(row)
    selected = []
    for fold in sorted(by_fold, key=lambda x: int(x) if x.isdigit() else 999):
        selected.extend(by_fold[fold][:per_fold])
    return selected


def download(url: str, path: Path, timeout: int = 30) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "detr-q3-figlib-visual-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            path.write_bytes(resp.read())
        return True
    except Exception:
        return False


def make_contact_sheet(rows: list[dict], image_dir: Path, out_path: Path, thumb_w: int = 260, thumb_h: int = 180) -> None:
    thumbs = []
    for row in rows:
        local = image_dir / row["local_file"]
        try:
            img = Image.open(local).convert("RGB")
            img.thumbnail((thumb_w, thumb_h - 28))
            canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
            x = (thumb_w - img.width) // 2
            canvas.paste(img, (x, 0))
            draw = ImageDraw.Draw(canvas)
            label = f"f{row['fold']} {row['station_id']} {row['offset_sec']}s"
            draw.text((6, thumb_h - 24), label[:38], fill=(0, 0, 0))
            thumbs.append(canvas)
        except Exception:
            continue
    if not thumbs:
        return
    cols = 5
    rows_n = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows_n * thumb_h), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_outputs(rows: list[dict], out_csv: Path, out_json: Path, out_md: Path, contact_sheet: Path) -> None:
    fieldnames = [
        "fold",
        "image",
        "sequence_id",
        "station_id",
        "fire_name",
        "date",
        "offset_sec",
        "frame_url",
        "local_file",
        "download_ok",
        "manual_status",
        "notes",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "download_ok_count": sum(1 for row in rows if row.get("download_ok") == "1"),
        "fold_counts": dict(sorted({fold: sum(1 for row in rows if row["fold"] == fold) for fold in {r["fold"] for r in rows}}.items())),
        "contact_sheet": str(contact_sheet),
        "purpose": "Manual visual purity check for pre-onset negative frames closest to the -300s exclusion boundary.",
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# FIgLib Visual Purity Sample",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Sample CSV**: `{out_csv}`",
        f"**Contact sheet**: `{contact_sheet}`",
        "",
        "## Purpose",
        "",
        "Sample pre-onset frames closest to the `-300s` exclusion boundary across frozen event-family folds. Manual labels must be filled in `manual_status` before purity claims.",
        "",
        "## Counts",
        "",
        f"- sample rows: `{summary['sample_count']}`",
        f"- downloaded images: `{summary['download_ok_count']}`",
        f"- fold counts: `{summary['fold_counts']}`",
        "",
        "## Manual Status Codes",
        "",
        "- `clean`: no visible smoke/fire cue",
        "- `ambiguous`: possible faint cue / haze / plume-like structure",
        "- `contaminated`: visible smoke/fire cue",
        "- `bad_image`: unreadable or irrelevant frame",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--per-fold", type=int, default=10)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_csv(args.manifest)
    split_to_fold = fold_map(args.folds)
    rows = select_samples(manifest_rows, split_to_fold, args.per_fold)
    for idx, row in enumerate(rows, start=1):
        suffix = Path(row["image"]).name
        row["local_file"] = f"{idx:03d}_fold{row['fold']}_{row['station_id']}_{suffix}"
        row["download_ok"] = "1" if download(row["frame_url"], args.image_dir / row["local_file"]) else "0"
        row["manual_status"] = ""
        row["notes"] = ""
    make_contact_sheet(rows, args.image_dir, args.contact_sheet)
    write_outputs(rows, args.out_csv, args.out_json, args.out_md, args.contact_sheet)
    print(f"[done] sample={len(rows)} contact_sheet={args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
