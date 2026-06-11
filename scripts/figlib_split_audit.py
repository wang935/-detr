#!/usr/bin/env python3
"""Audit FIgLib split/leakage structure from a manifest CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def unit_stats(rows: list[dict], group_col: str) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "positive_event_units": 0,
            "negative_frame_units": 0,
            "negative_hours_proxy": 0.0,
            "positive_rows": 0,
            "negative_rows": 0,
        }
    )
    seen_pos = set()
    seen_neg = set()
    for row in rows:
        group = row.get(group_col) or "UNKNOWN"
        label = row.get("label")
        unit = row.get("eval_unit")
        if label == "positive":
            stats[group]["positive_rows"] += 1
            if unit not in seen_pos:
                stats[group]["positive_event_units"] += 1
                seen_pos.add(unit)
        elif label == "negative":
            stats[group]["negative_rows"] += 1
            if unit not in seen_neg:
                stats[group]["negative_frame_units"] += 1
                try:
                    stats[group]["negative_hours_proxy"] += float(row.get("frame_interval_minutes_est") or 1.0) / 60.0
                except ValueError:
                    stats[group]["negative_hours_proxy"] += 1.0 / 60.0
                seen_neg.add(unit)
    return {key: dict(value) for key, value in stats.items()}


GENERIC_FIRE_NAMES = {
    "",
    "fire",
    "prescribed",
    "prescribedfire",
    "vegmgmt",
    "wildlanddrills",
    "wildlanddrills",
    "structfire",
    "structurefire",
    "house",
}


def normalized_fire_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def primary_family_key(row: dict) -> str:
    date = (row.get("date") or "")[:8]
    year = date[:4] if len(date) >= 4 else "unknownyear"
    name = normalized_fire_name(row.get("fire_name") or "")
    if name in GENERIC_FIRE_NAMES:
        return f"{date or 'unknowndate'}|{name or 'unknown'}"
    return f"{year}|{name}"


def unit_stats_by_key(rows: list[dict], key_func) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "positive_event_units": 0,
            "negative_frame_units": 0,
            "negative_hours_proxy": 0.0,
            "positive_rows": 0,
            "negative_rows": 0,
        }
    )
    seen_pos = set()
    seen_neg = set()
    for row in rows:
        group = key_func(row)
        label = row.get("label")
        unit = row.get("eval_unit")
        if label == "positive":
            stats[group]["positive_rows"] += 1
            if unit not in seen_pos:
                stats[group]["positive_event_units"] += 1
                seen_pos.add(unit)
        elif label == "negative":
            stats[group]["negative_rows"] += 1
            if unit not in seen_neg:
                stats[group]["negative_frame_units"] += 1
                try:
                    stats[group]["negative_hours_proxy"] += float(row.get("frame_interval_minutes_est") or 1.0) / 60.0
                except ValueError:
                    stats[group]["negative_hours_proxy"] += 1.0 / 60.0
                seen_neg.add(unit)
    return {key: dict(value) for key, value in stats.items()}


def greedy_folds(group_stats: dict[str, dict[str, float]], k: int) -> list[dict]:
    folds = [
        {"fold": idx, "groups": [], "positive_event_units": 0, "negative_hours_proxy": 0.0}
        for idx in range(k)
    ]
    ordered = sorted(
        group_stats.items(),
        key=lambda item: (item[1].get("positive_event_units", 0), item[1].get("negative_hours_proxy", 0.0)),
        reverse=True,
    )
    for group, stats in ordered:
        target = min(folds, key=lambda fold: (fold["positive_event_units"], fold["negative_hours_proxy"]))
        target["groups"].append(group)
        target["positive_event_units"] += int(stats.get("positive_event_units", 0))
        target["negative_hours_proxy"] += float(stats.get("negative_hours_proxy", 0.0))
    return folds


def summarize(rows: list[dict], fold_count: int) -> dict:
    station_stats = unit_stats(rows, "station_id")
    legacy_family_stats = unit_stats(rows, "split_group")
    family_stats = unit_stats_by_key(rows, primary_family_key)
    family_to_stations: dict[str, set[str]] = defaultdict(set)
    legacy_family_to_stations: dict[str, set[str]] = defaultdict(set)
    family_to_legacy_groups: dict[str, set[str]] = defaultdict(set)
    station_to_families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("label") not in {"positive", "negative"}:
            continue
        legacy_family = row.get("split_group") or "UNKNOWN"
        family = primary_family_key(row)
        station = row.get("station_id") or "UNKNOWN"
        family_to_stations[family].add(station)
        legacy_family_to_stations[legacy_family].add(station)
        family_to_legacy_groups[family].add(legacy_family)
        station_to_families[station].add(legacy_family)

    multi_station_families = {
        family: sorted(stations)
        for family, stations in family_to_stations.items()
        if len(stations) > 1
    }
    legacy_multi_station_families = {
        family: sorted(stations)
        for family, stations in legacy_family_to_stations.items()
        if len(stations) > 1
    }
    folds = greedy_folds(family_stats, fold_count)
    fold_assignments = []
    for fold in folds:
        for group in fold["groups"]:
            fold_assignments.append(
                {
                    "fire_family_key": group,
                    "fold": fold["fold"],
                    "legacy_split_group_count": len(family_to_legacy_groups[group]),
                    "legacy_split_groups": sorted(family_to_legacy_groups[group]),
                    "station_count": len(family_to_stations[group]),
                    "stations": sorted(family_to_stations[group]),
                    **family_stats[group],
                }
            )
    station_rank = sorted(
        station_stats.items(),
        key=lambda item: (item[1]["positive_event_units"], item[1]["negative_hours_proxy"]),
        reverse=True,
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "station_count": len(station_stats),
        "legacy_split_group_count": len(legacy_family_stats),
        "fire_family_key_count": len(family_stats),
        "multi_station_split_group_count": len(multi_station_families),
        "legacy_multi_station_split_group_count": len(legacy_multi_station_families),
        "multi_station_split_group_examples": dict(list(multi_station_families.items())[:20]),
        "legacy_multi_station_split_group_examples": dict(list(legacy_multi_station_families.items())[:20]),
        "top_station_stats": [
            {"station_id": key, **value}
            for key, value in station_rank[:20]
        ],
        "event_family_fold_plan": folds,
        "fold_assignments": fold_assignments,
        "fold_count": fold_count,
        "recommendation": (
            "Use event-family folds as the primary leakage-safe split, then report station-held-out "
            "stress separately or exclude overlapping fire families when doing leave-one-station."
        ),
        "caveats": [
            "primary fire_family_key is year plus normalized fire_name for named fires, and date plus generic name for generic FIRE/prescribed labels.",
            "fire_family_key is still filename-derived and may merge distinct same-year same-name incidents or split poorly named incidents.",
            "A true leave-one-station split can leak the same fire family through other stations unless overlapping split_groups are excluded.",
            "This audit does not inspect images and cannot certify pre-onset smoke purity.",
        ],
    }


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# FIgLib Split Audit",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Verdict",
        "",
        "The manifest supports an event-family split as the safer primary anti-leakage split. Pure leave-one-station evaluation is possible only as a stress test unless overlapping fire families are excluded from calibration/training.",
        "",
        "## Counts",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Rows | {summary['row_count']} |",
        f"| Stations | {summary['station_count']} |",
        f"| Legacy split groups (`date|fire_name`) | {summary['legacy_split_group_count']} |",
        f"| Primary fire-family keys | {summary['fire_family_key_count']} |",
        f"| Split groups spanning multiple stations | {summary['multi_station_split_group_count']} |",
        "",
        "## Event-Family Fold Plan",
        "",
        "| Fold | Split groups | Positive event units | Negative hours proxy |",
        "|---:|---:|---:|---:|",
    ]
    for fold in summary["event_family_fold_plan"]:
        lines.append(
            f"| {fold['fold']} | {len(fold['groups'])} | "
            f"{fold['positive_event_units']} | {fold['negative_hours_proxy']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Top Stations",
            "",
            "| Station | Positive event units | Negative frame units | Negative hours proxy |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in summary["top_station_stats"][:20]:
        lines.append(
            f"| {item['station_id']} | {item['positive_event_units']} | "
            f"{item['negative_frame_units']} | {item['negative_hours_proxy']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["caveats"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-folds-csv", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    summary = summarize(rows, args.folds)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    if args.out_folds_csv:
        args.out_folds_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_folds_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "fire_family_key",
                "fold",
                "positive_event_units",
                "negative_frame_units",
                "negative_hours_proxy",
                "positive_rows",
                "negative_rows",
                "legacy_split_group_count",
                "legacy_split_groups",
                "station_count",
                "stations",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary["fold_assignments"]:
                out = dict(row)
                out["legacy_split_groups"] = ";".join(out["legacy_split_groups"])
                out["stations"] = ";".join(out["stations"])
                writer.writerow({key: out.get(key, "") for key in fieldnames})
    print(
        f"[done] stations={summary['station_count']} fire_family_keys={summary['fire_family_key_count']} "
        f"multi_station_groups={summary['multi_station_split_group_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
