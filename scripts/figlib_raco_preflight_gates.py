#!/usr/bin/env python3
"""Pre-flight gates before full FIgLib RACO detector scoring."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_targets(text: str) -> list[float]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise SystemExit("[error] no targets")
    return out


def wilson_low(k: int, n: int, z: float = 1.959963984540054) -> float | None:
    if n <= 0:
        return None
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * ((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) ** 0.5
    return max(0.0, (centre - margin) / denom)


def read_folds(path: Path) -> dict[str, str]:
    mapping = {}
    for row in read_csv(path):
        fold = row["fold"]
        for group in row.get("legacy_split_groups", "").split(";"):
            group = group.strip()
            if group:
                previous = mapping.get(group)
                if previous is not None and previous != fold:
                    raise SystemExit(f"[error] split_group {group!r} maps to folds {previous} and {fold}")
                mapping[group] = fold
    return mapping


def attach_folds(manifest: list[dict], fold_map: dict[str, str]) -> list[dict]:
    rows = []
    missing = Counter()
    for row in manifest:
        if row.get("label") not in {"positive", "negative"}:
            continue
        split_group = row.get("split_group", "")
        fold = fold_map.get(split_group)
        if fold is None:
            missing[split_group] += 1
            continue
        item = dict(row)
        item["fold"] = fold
        rows.append(item)
    if missing:
        raise SystemExit(f"[error] missing fold mappings: {dict(missing.most_common(5))}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--targets", default="0.90,0.95")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    rows = attach_folds(read_csv(args.manifest), read_folds(args.folds))
    folds = sorted({row["fold"] for row in rows}, key=int)
    positive_events_by_fold: dict[str, set[str]] = defaultdict(set)
    positive_rows_by_fold = Counter()
    negative_units_by_fold: dict[str, set[str]] = defaultdict(set)
    negative_rows_by_fold = Counter()
    by_label = Counter(row["label"] for row in rows)

    for row in rows:
        fold = row["fold"]
        if row["label"] == "positive":
            positive_events_by_fold[fold].add(row["event_id"])
            positive_rows_by_fold[fold] += 1
        elif row["label"] == "negative":
            negative_units_by_fold[fold].add(row["eval_unit"])
            negative_rows_by_fold[fold] += 1

    per_fold = {}
    target_pass = {f"{target:.2f}": True for target in targets}
    for heldout in folds:
        calib_events = set()
        for fold, events in positive_events_by_fold.items():
            if fold != heldout:
                calib_events.update(events)
        n = len(calib_events)
        max_lcb = wilson_low(n, n)
        target_payload = {}
        for target in targets:
            ok = max_lcb is not None and max_lcb >= target
            target_payload[f"{target:.2f}"] = {
                "max_possible_wilson_lcb_if_all_hit": max_lcb,
                "statistically_attainable_if_all_events_hit": ok,
            }
            target_pass[f"{target:.2f}"] = target_pass[f"{target:.2f}"] and ok
        per_fold[heldout] = {
            "heldout_positive_event_units": len(positive_events_by_fold[heldout]),
            "heldout_positive_rows": positive_rows_by_fold[heldout],
            "heldout_negative_units": len(negative_units_by_fold[heldout]),
            "heldout_negative_rows": negative_rows_by_fold[heldout],
            "calibration_positive_event_units": n,
            "targets": target_payload,
        }

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pre-full-scoring statistical attainability gate for calib_lcb fixed recall",
        "manifest": str(args.manifest),
        "folds": str(args.folds),
        "rows": len(rows),
        "by_label": dict(sorted(by_label.items())),
        "total_positive_event_units": len(set().union(*positive_events_by_fold.values())),
        "folds_seen": folds,
        "targets": [f"{target:.2f}" for target in targets],
        "target_attainability_if_all_events_hit": target_pass,
        "per_heldout_fold": per_fold,
        "verdict": "pass" if all(target_pass.values()) else "fail",
        "caveat": "This checks statistical event-count sufficiency only. It does not guarantee any detector will hit enough early events.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RACO Full-Scoring Preflight Gates",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Scope",
        "",
        "Statistical preflight for the frozen `calib_lcb` fixed-recall rule before full detector scoring. This checks whether each calibration pool has enough positive event units for a Wilson lower confidence bound to reach the target if all calibration events are detected.",
        "",
        "## Verdict",
        "",
        f"- event-count attainability gate: `{payload['verdict']}`",
        f"- total positive event units: `{payload['total_positive_event_units']}`",
        f"- targets: `{payload['targets']}`",
        "- caveat: this does not guarantee detector success; it only prevents spending scoring time on a statistically impossible LCB target.",
        "",
        "## Held-Out Fold Checks",
        "",
        "| Held-out fold | Held-out events | Calibration events | Max LCB if all hit | R0.90 attainable | R0.95 attainable | Negative units |",
        "|---:|---:|---:|---:|---|---|---:|",
    ]
    for fold, item in per_fold.items():
        first_target = item["targets"][f"{targets[0]:.2f}"]
        lines.append(
            f"| {fold} | {item['heldout_positive_event_units']} | {item['calibration_positive_event_units']} | "
            f"{first_target['max_possible_wilson_lcb_if_all_hit']:.4f} | "
            f"{item['targets'].get('0.90', {}).get('statistically_attainable_if_all_events_hit', 'n/a')} | "
            f"{item['targets'].get('0.95', {}).get('statistically_attainable_if_all_events_hit', 'n/a')} | "
            f"{item['heldout_negative_units']} |"
        )
    lines.extend(
        [
            "",
            "## Use Policy",
            "",
            "- If this gate fails, do not start full scoring until the recall target or unit pooling rule is re-registered.",
            "- If this gate passes but detector scoring later makes a target unreachable, report it as an evaluator result, not as a runtime/preflight failure.",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
