#!/usr/bin/env python3
"""Build a FIgLib frame manifest without downloading image payloads.

The manifest is designed for fixed-recall denominator audits:
- positive frames are collapsed to event/sequence units;
- valid pre-event negatives are kept as frame/minute units;
- ambiguous near-onset frames are marked excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from figlib_metadata_probe import BASE_URL, INDEX_URL, extract_links, fetch_text, parse_sequence_name, sequence_links


FIELDNAMES = [
    "image",
    "label",
    "eval_unit",
    "sequence_id",
    "event_id",
    "station_id",
    "camera_id",
    "fire_name",
    "date",
    "direction",
    "imager",
    "timestamp_epoch",
    "offset_sec",
    "frame_url",
    "is_valid_negative_frame",
    "is_positive_event_frame",
    "split_group",
    "frame_interval_minutes_est",
]


def parse_frame_name(name: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)_([+-]\d+)\.jpg$", name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def median_interval_minutes(offsets: list[int]) -> float:
    diffs = sorted({b - a for a, b in zip(sorted(offsets), sorted(offsets)[1:]) if b > a})
    if not diffs:
        return 1.0
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return diffs[mid] / 60.0
    return ((diffs[mid - 1] + diffs[mid]) / 2.0) / 60.0


def build_sequence_rows(sequence: str, negative_max_offset: int, positive_min_offset: int) -> tuple[str, list[dict], str | None]:
    try:
        html = fetch_text(f"{BASE_URL}{sequence}/index.html")
        links = extract_links(html)
        jpgs = sorted(
            urllib.parse.unquote(link)
            for link in links
            if urllib.parse.unquote(link).lower().endswith(".jpg")
        )
        meta = parse_sequence_name(sequence)
        station = meta.get("station") or ""
        direction = meta.get("direction") or ""
        imager = meta.get("imager") or ""
        fire_name = meta.get("fire_name") or ""
        date = meta.get("date") or ""
        camera_id = "-".join(part for part in [station, direction, imager] if part)
        split_group = "|".join(part for part in [date[:8], fire_name] if part)
        parsed_frames = []
        for jpg in jpgs:
            timestamp_epoch, offset_sec = parse_frame_name(jpg)
            parsed_frames.append((jpg, timestamp_epoch, offset_sec))
        interval_minutes = median_interval_minutes([offset for _, _, offset in parsed_frames if offset is not None])
        rows = []
        for jpg, timestamp_epoch, offset_sec in parsed_frames:
            if offset_sec is None:
                label = "excluded"
                eval_unit = f"excluded:{sequence}:{jpg}"
                is_neg = "0"
                is_pos = "0"
            elif offset_sec <= negative_max_offset:
                label = "negative"
                eval_unit = f"pre:{sequence}:{offset_sec:+06d}"
                is_neg = "1"
                is_pos = "0"
            elif offset_sec >= positive_min_offset:
                label = "positive"
                eval_unit = f"event:{sequence}"
                is_neg = "0"
                is_pos = "1"
            else:
                label = "excluded"
                eval_unit = f"excluded:{sequence}:{offset_sec:+06d}"
                is_neg = "0"
                is_pos = "0"
            frame_url = f"{BASE_URL}{sequence}/{jpg}"
            rows.append(
                {
                    "image": f"{sequence}/{jpg}",
                    "label": label,
                    "eval_unit": eval_unit,
                    "sequence_id": sequence,
                    "event_id": sequence,
                    "station_id": station,
                    "camera_id": camera_id,
                    "fire_name": fire_name,
                    "date": date,
                    "direction": direction,
                    "imager": imager,
                    "timestamp_epoch": "" if timestamp_epoch is None else str(timestamp_epoch),
                    "offset_sec": "" if offset_sec is None else str(offset_sec),
                    "frame_url": frame_url,
                    "is_valid_negative_frame": is_neg,
                    "is_positive_event_frame": is_pos,
                    "split_group": split_group,
                    "frame_interval_minutes_est": f"{interval_minutes:.3f}",
                }
            )
        return sequence, rows, None
    except Exception as exc:  # noqa: BLE001 - audit should report failed pages.
        return sequence, [], repr(exc)


def summarize(
    rows: list[dict],
    errors: dict[str, str],
    empty_sequences: list[str],
    listed_sequence_count: int,
    negative_max_offset: int,
    positive_min_offset: int,
) -> dict:
    label_counts = Counter(row["label"] for row in rows)
    station_counts = Counter(row["station_id"] for row in rows if row["station_id"])
    sequence_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sequence_rows[row["sequence_id"]].append(row)

    sequence_stats = {}
    negative_span_minutes = 0.0
    negative_frame_minutes = 0.0
    positive_sequences = set()
    negative_sequences = set()
    frame_count_distribution = Counter()
    for sequence, seq_rows in sequence_rows.items():
        offsets = [int(row["offset_sec"]) for row in seq_rows if row["offset_sec"]]
        valid_negs = [v for v in offsets if v <= negative_max_offset]
        positives = [v for v in offsets if v >= positive_min_offset]
        if positives:
            positive_sequences.add(sequence)
        if valid_negs:
            negative_sequences.add(sequence)
        span_minutes = ((max(valid_negs) - min(valid_negs)) / 60.0) if len(valid_negs) >= 2 else 0.0
        negative_span_minutes += span_minutes
        negative_frame_minutes += sum(
            float(row["frame_interval_minutes_est"]) for row in seq_rows if row["label"] == "negative"
        )
        frame_count_distribution[len(seq_rows)] += 1
        sequence_stats[sequence] = {
            "frame_count": len(seq_rows),
            "offset_min": min(offsets) if offsets else None,
            "offset_max": max(offsets) if offsets else None,
            "valid_negative_frames": len(valid_negs),
            "positive_frames": len(positives),
            "negative_span_minutes": span_minutes,
        }

    positive_units = {row["eval_unit"] for row in rows if row["label"] == "positive"}
    negative_units = {row["eval_unit"] for row in rows if row["label"] == "negative"}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_index": INDEX_URL,
        "negative_window": f"offset_sec <= {negative_max_offset}",
        "positive_window": f"offset_sec >= {positive_min_offset}",
        "listed_sequence_count": listed_sequence_count,
        "sequence_count": len(sequence_rows),
        "empty_sequence_count": len(empty_sequences),
        "empty_sequences": sorted(empty_sequences),
        "failed_sequence_count": len(errors),
        "failed_sequences": errors,
        "frame_count": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "positive_event_units": len(positive_units),
        "negative_frame_units": len(negative_units),
        "positive_sequences": len(positive_sequences),
        "negative_sequences": len(negative_sequences),
        "negative_frame_hours_discrete": negative_frame_minutes / 60.0,
        "negative_span_hours_conservative": negative_span_minutes / 60.0,
        "unique_station_count": len(station_counts),
        "top_station_counts_by_frame": station_counts.most_common(20),
        "frame_count_distribution": dict(sorted(frame_count_distribution.items())),
        "gate_checks": {
            "positive_sequences_ge_100": len(positive_sequences) >= 100,
            "negative_span_hours_ge_100": (negative_span_minutes / 60.0) >= 100,
            "negative_frame_hours_ge_100": (negative_frame_minutes / 60.0) >= 100,
            "failed_sequences_zero": len(errors) == 0,
            "empty_sequences_zero": len(empty_sequences) == 0,
        },
        "notes": [
            "Rows are frame metadata only; no image payloads are downloaded.",
            "Positive eval_unit collapses all post-onset frames from a sequence to one event unit.",
            "Negative eval_unit keeps each valid pre-event frame as a frame/minute unit.",
            "Camera-hour values are proxies from sampled still frames, not continuous video duration.",
            "A visual audit is still required for pre-onset negative purity.",
        ],
        "sample_sequence_stats": dict(list(sequence_stats.items())[:5]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict, csv_path: Path, json_path: Path) -> None:
    lines = [
        "# FIgLib Manifest Audit",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Manifest CSV**: `{csv_path}`",
        f"**Summary JSON**: `{json_path}`",
        "",
        "## Verdict",
        "",
    ]
    gates = summary["gate_checks"]
    if gates["positive_sequences_ge_100"] and gates["negative_span_hours_ge_100"] and gates["failed_sequences_zero"]:
        lines.append("FIgLib passes the first denominator-size gate for a RACO-Wildfire fixed-recall pilot, subject to leakage split and visual purity audits.")
    else:
        lines.append("FIgLib does not yet pass all denominator-size gates; inspect the failed gate table before promoting to training.")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Sequences parsed | {summary['sequence_count']} |",
            f"| Listed sequences | {summary['listed_sequence_count']} |",
            f"| Empty sequence listings | {summary['empty_sequence_count']} |",
            f"| Failed sequences | {summary['failed_sequence_count']} |",
            f"| Frames listed | {summary['frame_count']} |",
            f"| Positive event units | {summary['positive_event_units']} |",
            f"| Negative frame/minute units | {summary['negative_frame_units']} |",
            f"| Conservative negative span hours | {summary['negative_span_hours_conservative']:.2f} |",
            f"| Discrete negative frame hours | {summary['negative_frame_hours_discrete']:.2f} |",
            f"| Unique station estimate | {summary['unique_station_count']} |",
            "",
            "## Gate Checks",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Label Counts",
            "",
            "| Label | Rows |",
            "|---|---:|",
        ]
    )
    for key, value in summary["label_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Negative camera-hours are still-frame proxy hours, not continuous video hours.",
            "- Positive event recall over `[0, +2400]` must be paired with time-to-detection or an early-window metric.",
            "- Pre-onset negatives need visual audit near the exclusion boundary.",
            "- Station/fire-family split is not yet assigned; this manifest only provides split keys.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--negative-max-offset", type=int, default=-300)
    parser.add_argument("--positive-min-offset", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="debug limit on number of sequences")
    args = parser.parse_args()

    seqs = sequence_links(extract_links(fetch_text(INDEX_URL)))
    if args.limit:
        seqs = seqs[: args.limit]

    rows: list[dict] = []
    errors: dict[str, str] = {}
    empty_sequences: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(build_sequence_rows, seq, args.negative_max_offset, args.positive_min_offset): seq
            for seq in seqs
        }
        for future in as_completed(futures):
            seq, seq_rows, error = future.result()
            if error:
                errors[seq] = error
            elif not seq_rows:
                empty_sequences.append(seq)
            rows.extend(seq_rows)

    rows.sort(key=lambda row: (row["sequence_id"], int(row["offset_sec"]) if row["offset_sec"] else 10**9))
    summary = summarize(rows, errors, empty_sequences, len(seqs), args.negative_max_offset, args.positive_min_offset)

    write_csv(args.out_csv, rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary, args.out_csv, args.out_json)
    print(
        f"[done] sequences={summary['sequence_count']} frames={summary['frame_count']} "
        f"neg_span_hours={summary['negative_span_hours_conservative']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
