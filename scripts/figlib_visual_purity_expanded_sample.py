#!/usr/bin/env python3
"""Create an expanded multi-offset visual purity sample for FIgLib negatives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw


FIELDNAMES = [
    "fold",
    "offset_target_sec",
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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_fold_map(path: Path) -> dict[str, str]:
    mapping = {}
    for row in read_csv(path):
        fold = row["fold"]
        for group in row.get("legacy_split_groups", "").split(";"):
            group = group.strip()
            if group:
                mapping[group] = fold
    return mapping


def parse_targets(text: str) -> list[int]:
    targets = []
    for item in text.split(","):
        item = item.strip()
        if item:
            targets.append(int(item))
    if not targets:
        raise SystemExit("[error] no offset targets provided")
    return targets


def select_samples(rows: list[dict], fold_map: dict[str, str], per_fold: int, targets: list[int]) -> list[dict]:
    by_fold_target: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("label") != "negative":
            continue
        fold = fold_map.get(row.get("split_group", ""))
        if fold is None:
            continue
        try:
            offset = int(float(row["offset_sec"]))
        except Exception:
            continue
        for target in targets:
            item = dict(row)
            item["fold"] = fold
            item["offset_target_sec"] = str(target)
            item["_distance"] = abs(offset - target)
            by_fold_target[(fold, target)].append(item)

    selected = []
    used_sequences_by_fold: dict[str, set[str]] = defaultdict(set)
    quota = math.ceil(per_fold / len(targets))
    for fold in sorted({fold for fold, _target in by_fold_target}, key=lambda value: int(value)):
        fold_selected = []
        for target in targets:
            candidates = sorted(
                by_fold_target[(fold, target)],
                key=lambda row: (row["_distance"], row["sequence_id"], row["image"]),
            )
            taken = 0
            for row in candidates:
                if row["sequence_id"] in used_sequences_by_fold[fold]:
                    continue
                out = dict(row)
                out.pop("_distance", None)
                fold_selected.append(out)
                used_sequences_by_fold[fold].add(row["sequence_id"])
                taken += 1
                if taken >= quota or len(fold_selected) >= per_fold:
                    break
            if len(fold_selected) >= per_fold:
                break
        selected.extend(fold_selected[:per_fold])
    return selected


def download(url: str, path: Path, retries: int = 3, timeout: int = 30) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_url = urllib.parse.quote(url, safe=":/")
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(encoded_url, headers={"User-Agent": "detr-q3-figlib-visual-expanded/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                path.write_bytes(response.read())
            return True
        except Exception:
            if attempt < retries:
                time.sleep(attempt)
    return False


def make_contact_sheet(rows: list[dict], image_dir: Path, out_path: Path, thumb_w: int = 260, thumb_h: int = 188) -> None:
    thumbs = []
    for row in rows:
        local = image_dir / row["local_file"]
        try:
            img = Image.open(local).convert("RGB")
            img.thumbnail((thumb_w, thumb_h - 34))
            canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
            x = (thumb_w - img.width) // 2
            canvas.paste(img, (x, 0))
            draw = ImageDraw.Draw(canvas)
            label = f"f{row['fold']} t{row['offset_target_sec']} o{row['offset_sec']} {row['station_id']}"
            draw.text((6, thumb_h - 29), label[:44], fill=(0, 0, 0))
            draw.text((6, thumb_h - 15), Path(row["image"]).name[:44], fill=(0, 0, 0))
            thumbs.append(canvas)
        except Exception:
            continue
    if not thumbs:
        return
    cols = 5
    rows_n = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows_n * thumb_h), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * thumb_w, (idx // cols) * thumb_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_outputs(rows: list[dict], out_csv: Path, out_json: Path, out_md: Path, contact_sheet: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})

    fold_counts = Counter(row["fold"] for row in rows)
    target_counts = Counter(row["offset_target_sec"] for row in rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "download_ok_count": sum(row.get("download_ok") == "1" for row in rows),
        "fold_counts": dict(sorted(fold_counts.items())),
        "offset_target_counts": dict(sorted(target_counts.items(), key=lambda item: int(item[0]))),
        "contact_sheet": str(contact_sheet),
        "purpose": "Expanded multi-offset visual purity sample for FIgLib pre-onset negatives.",
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# FIgLib Expanded Visual Purity Sample",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Sample CSV**: `{out_csv}`",
        f"**Contact sheet**: `{contact_sheet}`",
        "",
        "## Purpose",
        "",
        "Expanded pre-onset visual-purity sample across frozen folds and multiple offset targets. Manual labels must be filled before using this as purity evidence.",
        "",
        "## Counts",
        "",
        f"- sample rows: `{summary['sample_count']}`",
        f"- downloaded images: `{summary['download_ok_count']}`",
        f"- fold counts: `{summary['fold_counts']}`",
        f"- offset target counts: `{summary['offset_target_counts']}`",
        "",
        "## Manual Status Codes",
        "",
        "- `clean`: no visible smoke/fire cue",
        "- `ambiguous`: possible faint cue / haze / plume-like structure",
        "- `contaminated`: visible smoke/fire cue",
        "- `bad_image`: unreadable or irrelevant frame",
        "",
        "## Use",
        "",
        "Use this as the second-stage visual audit sample after the initial 50-row boundary screen. Do not claim full FIgLib negative purity until labels are adjudicated.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--per-fold", type=int, default=20)
    parser.add_argument("--offset-targets", default="-300,-600,-900,-1200")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    folds = read_fold_map(args.folds)
    targets = parse_targets(args.offset_targets)
    rows = select_samples(manifest, folds, args.per_fold, targets)
    for idx, row in enumerate(rows, start=1):
        suffix = Path(row["image"]).name
        row["local_file"] = f"{idx:03d}_fold{row['fold']}_t{row['offset_target_sec']}_{row['station_id']}_{suffix}"
        row["download_ok"] = "1" if download(row["frame_url"], args.image_dir / row["local_file"]) else "0"
        row["manual_status"] = ""
        row["notes"] = ""
    make_contact_sheet(rows, args.image_dir, args.contact_sheet)
    write_outputs(rows, args.out_csv, args.out_json, args.out_md, args.contact_sheet)
    print(f"[done] expanded_sample={len(rows)} contact_sheet={args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
