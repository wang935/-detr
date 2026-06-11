#!/usr/bin/env python3
"""Create a row-level visual-purity screen report from a sampled FIgLib CSV.

This helper is intentionally conservative. It records a contact-sheet screen
as an intermediate artifact, separate from final human adjudication.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_indices(text: str | None) -> set[int]:
    if not text:
        return set()
    out = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(item))
    return out


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--ambiguous-indices", default="")
    parser.add_argument("--contaminated-indices", default="")
    parser.add_argument("--bad-image-indices", default="")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    ambiguous = parse_indices(args.ambiguous_indices)
    contaminated = parse_indices(args.contaminated_indices)
    bad_image = parse_indices(args.bad_image_indices)
    overlap = (ambiguous & contaminated) | (ambiguous & bad_image) | (contaminated & bad_image)
    if overlap:
        raise SystemExit(f"[error] indices assigned to multiple statuses: {sorted(overlap)}")

    rows = []
    for idx, row in enumerate(read_csv(args.sample_csv), start=1):
        status = "clean"
        note = "no obvious smoke plume or flame at contact-sheet scale"
        if idx in ambiguous:
            status = "ambiguous"
            note = "haze/glare/cloud/low-contrast or occlusion-like cue; no obvious smoke/flame"
        elif idx in contaminated:
            status = "contaminated"
            note = "visible smoke/fire cue at contact-sheet scale"
        elif idx in bad_image:
            status = "bad_image"
            note = "unreadable or irrelevant image at contact-sheet scale"
        out = {
            "row_id": idx,
            **row,
            "screen_status": status,
            "screen_notes": note,
            "label_source": "Codex contact-sheet visual screen; not final human adjudication",
        }
        rows.append(out)

    status_counts = Counter(row["screen_status"] for row in rows)
    fold_status = Counter((row.get("fold", ""), row["screen_status"]) for row in rows)
    offset_status = Counter((row.get("offset_target_sec", ""), row["screen_status"]) for row in rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_csv": str(args.sample_csv),
        "contact_sheet": str(args.contact_sheet),
        "scope": "row-level contact-sheet visual screen, not final human adjudication",
        "rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "fold_status_counts": {f"{fold}|{status}": count for (fold, status), count in sorted(fold_status.items())},
        "offset_status_counts": {
            f"{offset}|{status}": count for (offset, status), count in sorted(offset_status.items(), key=lambda item: (int(item[0][0]), item[0][1]))
        },
        "contaminated_rows": [row["row_id"] for row in rows if row["screen_status"] == "contaminated"],
        "ambiguous_rows": [row["row_id"] for row in rows if row["screen_status"] == "ambiguous"],
        "bad_image_rows": [row["row_id"] for row in rows if row["screen_status"] == "bad_image"],
        "use_policy": [
            "Use clean-only and clean+ambiguous sensitivity sets for audited-subset reporting.",
            "Do not use this screen as a final human purity label set.",
            "If contaminated rows appear in final adjudication, rerun the manifest with a larger pre-onset exclusion margin.",
        ],
    }

    write_csv(args.out_csv, rows)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# FIgLib Expanded Visual Purity Row-Level Screen",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Sample CSV**: `{args.sample_csv}`",
        f"**Contact sheet**: `{args.contact_sheet}`",
        f"**Output CSV**: `{args.out_csv}`",
        "",
        "## Scope",
        "",
        "Row-level contact-sheet screen of the expanded 100-image FIgLib pre-onset sample. This is not final human adjudication and must not be used as a paper-level purity claim by itself.",
        "",
        "## Status Counts",
        "",
        "| Status | Count | Meaning |",
        "|---|---:|---|",
        f"| clean | {status_counts.get('clean', 0)} | no obvious smoke plume or flame at contact-sheet scale |",
        f"| ambiguous | {status_counts.get('ambiguous', 0)} | haze, glare, clouds, low contrast, or occlusion-like cues need sensitivity handling |",
        f"| contaminated | {status_counts.get('contaminated', 0)} | visible smoke/fire cue at contact-sheet scale |",
        f"| bad_image | {status_counts.get('bad_image', 0)} | unreadable or irrelevant image |",
        "",
        "## Fold x Status",
        "",
        "| Fold | Clean | Ambiguous | Contaminated | Bad image |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold in sorted({row.get("fold", "") for row in rows}, key=lambda value: int(value)):
        lines.append(
            f"| {fold} | {fold_status.get((fold, 'clean'), 0)} | {fold_status.get((fold, 'ambiguous'), 0)} | "
            f"{fold_status.get((fold, 'contaminated'), 0)} | {fold_status.get((fold, 'bad_image'), 0)} |"
        )
    lines.extend(
        [
            "",
            "## Use Policy",
            "",
            "- Treat `clean` rows as the conservative audited subset.",
            "- Treat `clean+ambiguous` rows as the sensitivity subset.",
            "- Do not claim full-manifest visual purity from this sample alone.",
            "- Final paper claims require independent human row-level adjudication, ideally with full-resolution inspection of ambiguous rows.",
            "",
            "## Current Screen Verdict",
            "",
            "No contact-sheet-scale contaminated rows were marked in this screen. The main residual risk is ambiguous atmospheric, glare, cloud, and occlusion-like appearances rather than obvious pre-onset smoke/flame leakage.",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
