#!/usr/bin/env python3
"""Frozen-fold RACO evaluator for HPWREN FIgLib.

The protocol is intentionally event-centric:

- calibration folds select operating thresholds using positive post-onset
  event recall;
- held-out folds report event recall, time-to-detection, and false alarms per
  still-frame pre-event camera-hour proxy;
- negative units use the manifest's pre-event frame/minute `eval_unit` field.

This script can also run without predictions to emit the frozen protocol
denominator summary used before detector scoring starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


MIN_THRESHOLD = 1e-6


def parse_targets(text: str) -> list[float]:
    targets = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0 or value >= 1.0:
            raise SystemExit(f"[error] target recall must be in (0,1): {value}")
        targets.append(value)
    if not targets:
        raise SystemExit("[error] no target recalls provided")
    return targets


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float | None]:
    if n <= 0:
        return [None, None]
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return [max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)]


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
        if term == 0.0:
            break
    return min(1.0, max(0.0, total))


def solve_poisson_cdf(k: int, target: float) -> float:
    lo = 0.0
    hi = max(1.0, k + 10.0 * math.sqrt(max(k, 1)))
    while poisson_cdf(k, hi) > target:
        hi *= 2.0
        if hi > 1e7:
            break
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if poisson_cdf(k, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def poisson_count_ci_exact(k: int, alpha: float = 0.05) -> list[float]:
    if k < 0:
        raise ValueError("Poisson count must be non-negative")
    if k > 1000:
        # Large-count fallback: exact inversion is unnecessary for the rare-event
        # FAR regime and can underflow; the normal approximation is tight here.
        z = 1.959963984540054
        margin = z * math.sqrt(k)
        return [max(0.0, k - margin), k + margin]
    lower = 0.0 if k == 0 else solve_poisson_cdf(k - 1, 1.0 - alpha / 2.0)
    upper = solve_poisson_cdf(k, alpha / 2.0)
    return [lower, upper]


def poisson_rate_ci(k: int, exposure: float) -> list[float | None]:
    if exposure <= 0:
        return [None, None]
    count_ci = poisson_count_ci_exact(k)
    return [count_ci[0] / exposure, count_ci[1] / exposure]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def read_folds(path: Path) -> dict[str, int]:
    split_to_fold: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"fold", "legacy_split_groups"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain {sorted(required)}")
        for row in reader:
            fold = int(row["fold"])
            groups = [item.strip() for item in row["legacy_split_groups"].split(";") if item.strip()]
            for group in groups:
                previous = split_to_fold.get(group)
                if previous is not None and previous != fold:
                    raise SystemExit(f"[error] split_group {group!r} maps to both fold {previous} and {fold}")
                split_to_fold[group] = fold
    return split_to_fold


def read_manifest(path: Path, split_to_fold: dict[str, int], allow_missing_splits: bool = False) -> tuple[list[dict], dict]:
    rows = []
    missing_split = defaultdict(int)
    event_folds = defaultdict(set)
    image_labels = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "image",
            "label",
            "eval_unit",
            "event_id",
            "sequence_id",
            "split_group",
            "offset_sec",
            "frame_interval_minutes_est",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain {sorted(required)}")
        for row in reader:
            label = row["label"].strip().lower()
            if label not in {"positive", "negative"}:
                continue
            split_group = row["split_group"].strip()
            fold = split_to_fold.get(split_group)
            if fold is None:
                missing_split[split_group] += 1
                continue
            item = dict(row)
            item["label"] = label
            item["fold"] = fold
            item["offset_sec"] = float(row["offset_sec"])
            item["frame_interval_minutes_est"] = float(row.get("frame_interval_minutes_est") or 1.0)
            rows.append(item)
            event_folds[item["event_id"]].add(fold)
            image_labels[item["image"]].add(label)
    if missing_split and not allow_missing_splits:
        sample = ", ".join(sorted(missing_split)[:5])
        raise SystemExit(
            f"[error] {sum(missing_split.values())} manifest rows have split_group values not present in folds: {sample}"
        )
    multi_fold_events = {event: sorted(folds) for event, folds in event_folds.items() if len(folds) > 1}
    if multi_fold_events:
        sample = dict(list(sorted(multi_fold_events.items()))[:5])
        raise SystemExit(f"[error] event_id spans multiple folds, possible leakage: {sample}")
    mixed_label_images = {image: sorted(labels) for image, labels in image_labels.items() if {"positive", "negative"}.issubset(labels)}
    if mixed_label_images:
        sample = dict(list(sorted(mixed_label_images.items()))[:5])
        raise SystemExit(f"[error] image has both positive and negative labels: {sample}")
    issues = {
        "missing_split_group_rows": sum(missing_split.values()),
        "missing_split_groups": dict(sorted(missing_split.items())),
        "validated_event_fold_disjoint": True,
        "validated_positive_negative_image_disjoint": True,
    }
    return rows, issues


def protocol_summary(rows: list[dict], issues: dict) -> dict:
    per_fold = {}
    total_pos_events = set()
    total_neg_units = set()
    total_neg_span_hours = 0.0
    total_neg_discrete_unique_hours = 0.0
    seen_neg_units = set()

    for fold in sorted({row["fold"] for row in rows}):
        fold_rows = [row for row in rows if row["fold"] == fold]
        pos_events = {row["event_id"] for row in fold_rows if row["label"] == "positive"}
        neg_units, neg_span_hours, neg_discrete_unique_hours = negative_denominators(fold_rows)
        per_fold[str(fold)] = {
            "positive_event_units": len(pos_events),
            "positive_rows": sum(1 for row in fold_rows if row["label"] == "positive"),
            "negative_units": len(neg_units),
            "negative_rows": sum(1 for row in fold_rows if row["label"] == "negative"),
            "negative_span_hours_conservative": neg_span_hours,
            "negative_frame_hours_discrete_unique": neg_discrete_unique_hours,
        }
        total_pos_events.update(pos_events)
        for unit, row in neg_units.items():
            if unit not in seen_neg_units:
                seen_neg_units.add(unit)
                total_neg_units.add(unit)
                total_neg_discrete_unique_hours += row["frame_interval_minutes_est"] / 60.0
        total_neg_span_hours += neg_span_hours

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "FIgLib frozen event-family 5-fold RACO protocol",
        "unit_of_analysis": {
            "positive": "event_id over post-onset positive frames",
            "negative": "unique pre-event eval_unit still-frame/minute units",
            "far_exposure": "canonical conservative sum over per-sequence valid pre-event offset spans",
        },
        "fold_count": len(per_fold),
        "total_positive_event_units": len(total_pos_events),
        "total_negative_units": len(total_neg_units),
        "total_negative_span_hours_conservative": total_neg_span_hours,
        "total_negative_frame_hours_discrete_unique": total_neg_discrete_unique_hours,
        # Backward-compatible alias: use the canonical conservative denominator.
        "total_negative_hours_proxy": total_neg_span_hours,
        "per_fold": per_fold,
        "manifest_issues": issues,
    }


def negative_denominators(rows: list[dict]) -> tuple[dict[str, dict], float, float]:
    neg_units = {}
    offsets_by_sequence = defaultdict(set)
    for row in rows:
        if row["label"] != "negative":
            continue
        neg_units.setdefault(row["eval_unit"], row)
        offsets_by_sequence[row["sequence_id"]].add(row["offset_sec"])
    span_hours = 0.0
    for offsets in offsets_by_sequence.values():
        if len(offsets) >= 2:
            span_hours += (max(offsets) - min(offsets)) / 3600.0
    discrete_unique_hours = sum(row["frame_interval_minutes_est"] / 60.0 for row in neg_units.values())
    return neg_units, span_hours, discrete_unique_hours


def detect_column(fieldnames: list[str], requested: str | None, candidates: list[str], kind: str) -> str | None:
    if requested:
        if requested not in fieldnames:
            raise SystemExit(f"[error] requested {kind} column {requested!r} not found")
        return requested
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def read_scores(path: Path, image_col: str | None, score_col: str | None, model_col: str | None) -> tuple[dict, dict]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    raw_rows = 0
    bad_rows = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"[error] empty prediction CSV: {path}")
        detected_image_col = detect_column(reader.fieldnames, image_col, ["image", "sample_image", "local_file"], "image")
        detected_score_col = detect_column(reader.fieldnames, score_col, ["max_conf", "score", "conf"], "score")
        detected_model_col = detect_column(reader.fieldnames, model_col, ["arm", "model"], "model")
        if detected_image_col is None or detected_score_col is None:
            raise SystemExit(f"[error] {path} needs image and score/conf columns")
        for row in reader:
            raw_rows += 1
            image = row.get(detected_image_col, "").strip()
            model = row.get(detected_model_col, "default").strip() if detected_model_col else "default"
            if not model:
                model = "default"
            try:
                score = float(row[detected_score_col])
            except Exception:
                bad_rows += 1
                continue
            if not image or not math.isfinite(score):
                bad_rows += 1
                continue
            score = max(0.0, min(1.0, score))
            previous = scores[model].get(image, 0.0)
            if image not in scores[model] or score > previous:
                scores[model][image] = score
    meta = {
        "raw_prediction_rows": raw_rows,
        "bad_prediction_rows": bad_rows,
        "image_col": detected_image_col,
        "score_col": detected_score_col,
        "model_col": detected_model_col or "default",
        "models": sorted(scores),
    }
    return scores, meta


def build_units(rows: list[dict], folds: set[int], model_scores: dict[str, float]) -> dict:
    events = defaultdict(list)
    neg_units = {}
    neg_records = {}
    all_images = set()
    scored_images = set()
    fold_rows = []

    for row in rows:
        if row["fold"] not in folds:
            continue
        fold_rows.append(row)
        all_images.add(row["image"])
        score = model_scores.get(row["image"], 0.0)
        if row["image"] in model_scores:
            scored_images.add(row["image"])
        if row["label"] == "positive":
            events[row["event_id"]].append((row["offset_sec"], score, row["image"]))
        elif row["label"] == "negative":
            unit = row["eval_unit"]
            previous = neg_units.get(unit, 0.0)
            if score > previous:
                neg_units[unit] = score
                neg_records[unit] = {
                    "sequence_id": row["sequence_id"],
                    "offset_sec": row["offset_sec"],
                    "score": score,
                    "image": row["image"],
                }

    event_scores = {}
    event_offsets = {}
    for event_id, frames in events.items():
        frames.sort(key=lambda item: item[0])
        event_scores[event_id] = max((score for _, score, _ in frames), default=0.0)
        event_offsets[event_id] = frames

    _denom_units, neg_span_hours, neg_discrete_unique_hours = negative_denominators(fold_rows)
    return {
        "event_scores": event_scores,
        "event_offsets": event_offsets,
        "neg_scores": neg_units,
        "neg_records": neg_records,
        "negative_span_hours_conservative": neg_span_hours,
        "negative_frame_hours_discrete_unique": neg_discrete_unique_hours,
        "n_images": len(all_images),
        "n_scored_images": len(scored_images),
    }


def choose_threshold(
    units: dict,
    target: float,
    threshold_rule: str,
    recall_mode: str,
    early_window_sec: float,
    debounce_sec: float,
) -> tuple[float, dict] | tuple[None, None]:
    observed = {score for score in units["event_scores"].values() if score > MIN_THRESHOLD}
    grid = {idx / 1000.0 for idx in range(1, 1001)}
    candidates = sorted(observed | grid, reverse=True)
    n_events = len(units["event_scores"])
    if n_events == 0:
        return None, None
    for threshold in candidates:
        metrics = metrics_at_threshold(units, threshold, early_window_sec, debounce_sec)
        if recall_mode == "early":
            recall = metrics["early_detection_rate"]
            lcb = metrics["early_detection_rate_ci95_wilson"][0]
        elif recall_mode == "full":
            recall = metrics["event_recall"]
            lcb = metrics["event_recall_ci95_wilson"][0]
        else:
            raise ValueError(f"unknown recall_mode: {recall_mode}")
        criterion = lcb if threshold_rule == "calib_lcb" else recall
        if criterion is not None and criterion >= target:
            return float(threshold), metrics
    return None, None


def clustered_alarm_events(neg_records: dict[str, dict], threshold: float, debounce_sec: float) -> int:
    by_sequence = defaultdict(list)
    for record in neg_records.values():
        if record["score"] >= threshold:
            by_sequence[record["sequence_id"]].append(record["offset_sec"])
    clusters = 0
    for offsets in by_sequence.values():
        previous = None
        for offset in sorted(offsets):
            if previous is None or (offset - previous) > debounce_sec:
                clusters += 1
            previous = offset
    return clusters


def km_median_detection_time(observations: list[tuple[float, bool]]) -> float | None:
    if not observations:
        return None
    n_at_risk = len(observations)
    survival = 1.0
    by_time = defaultdict(lambda: {"events": 0, "censored": 0})
    for time_sec, observed in observations:
        key = float(time_sec)
        if observed:
            by_time[key]["events"] += 1
        else:
            by_time[key]["censored"] += 1
    for time_sec in sorted(by_time):
        events = by_time[time_sec]["events"]
        censored = by_time[time_sec]["censored"]
        if n_at_risk <= 0:
            break
        if events:
            survival *= 1.0 - (events / n_at_risk)
            if survival <= 0.5:
                return time_sec
        n_at_risk -= events + censored
    return None


def metrics_at_threshold(units: dict, threshold: float, early_window_sec: float, debounce_sec: float) -> dict:
    event_scores = units["event_scores"]
    event_offsets = units["event_offsets"]
    neg_scores = units["neg_scores"]
    neg_records = units["neg_records"]

    hit_events = [event for event, score in event_scores.items() if score >= threshold]
    ttd_values = []
    early_ttd_values = []
    km_observations = []
    for event in hit_events:
        for offset_sec, score, _image in event_offsets[event]:
            if score >= threshold:
                ttd_values.append(offset_sec)
                if 0.0 <= offset_sec <= early_window_sec:
                    early_ttd_values.append(offset_sec)
                break
    for event, frames in event_offsets.items():
        detected_in_window = None
        for offset_sec, score, _image in frames:
            if 0.0 <= offset_sec <= early_window_sec and score >= threshold:
                detected_in_window = offset_sec
                break
        if detected_in_window is None:
            km_observations.append((early_window_sec, False))
        else:
            km_observations.append((detected_in_window, True))

    neg_hits = sum(score >= threshold for score in neg_scores.values())
    alarm_events = clustered_alarm_events(neg_records, threshold, debounce_sec)
    neg_hours_total = units["negative_span_hours_conservative"]
    n_events = len(event_scores)
    n_neg = len(neg_scores)
    recall = len(hit_events) / n_events if n_events else 0.0
    early_recall = len(early_ttd_values) / n_events if n_events else 0.0
    fpr = neg_hits / n_neg if n_neg else 0.0
    far = alarm_events / neg_hours_total if neg_hours_total > 0 else 0.0
    return {
        "threshold": threshold,
        "event_hits": len(hit_events),
        "n_events": n_events,
        "event_recall": recall,
        "event_recall_ci95_wilson": wilson_ci(len(hit_events), n_events),
        "early_window_sec": early_window_sec,
        "early_detection_hits": len(early_ttd_values),
        "early_detection_rate": early_recall,
        "early_detection_rate_ci95_wilson": wilson_ci(len(early_ttd_values), n_events),
        "negative_unit_hits": neg_hits,
        "n_negative_units": n_neg,
        "clustered_alarm_events": alarm_events,
        "debounce_sec": debounce_sec,
        "negative_hours_proxy": neg_hours_total,
        "negative_span_hours_conservative": neg_hours_total,
        "negative_frame_hours_discrete_unique": units["negative_frame_hours_discrete_unique"],
        "negative_unit_fpr": fpr,
        "negative_unit_fpr_ci95_wilson": wilson_ci(neg_hits, n_neg),
        "far_per_pre_event_camera_hour_proxy": far,
        "far_ci95_poisson_exact": poisson_rate_ci(alarm_events, neg_hours_total),
        "ttd_hit_count": len(ttd_values),
        "ttd_mean_sec": mean(ttd_values) if ttd_values else None,
        "ttd_median_sec": median(ttd_values) if ttd_values else None,
        "ttd_p90_sec": quantile(ttd_values, 0.90),
        "early_ttd_hit_count": len(early_ttd_values),
        "early_ttd_median_detected_sec": median(early_ttd_values) if early_ttd_values else None,
        "early_ttd_km_median_sec": km_median_detection_time(km_observations),
    }


def evaluate(
    rows: list[dict],
    scores_by_model: dict[str, dict[str, float]],
    targets: list[float],
    strict_coverage: bool,
    threshold_rule: str,
    recall_mode: str,
    early_window_sec: float,
    debounce_sec: float,
) -> dict:
    folds = sorted({row["fold"] for row in rows})
    output = {}
    for model, model_scores in scores_by_model.items():
        fold_results = {}
        pooled = {f"{target:.2f}": defaultdict(float) for target in targets}
        for bucket in pooled.values():
            bucket["early_ttd_detected_values"] = []
        coverage_errors = []
        for test_fold in folds:
            calib_folds = set(folds) - {test_fold}
            test_folds = {test_fold}
            calib_units = build_units(rows, calib_folds, model_scores)
            test_units = build_units(rows, test_folds, model_scores)
            fold_payload = {
                "calib_folds": sorted(calib_folds),
                "test_fold": test_fold,
                "calib_image_coverage": safe_div(calib_units["n_scored_images"], calib_units["n_images"]),
                "test_image_coverage": safe_div(test_units["n_scored_images"], test_units["n_images"]),
                "targets": {},
            }
            if strict_coverage and (
                calib_units["n_scored_images"] < calib_units["n_images"]
                or test_units["n_scored_images"] < test_units["n_images"]
            ):
                coverage_errors.append(test_fold)
            for target in targets:
                threshold, calib_metrics = choose_threshold(
                    calib_units,
                    target,
                    threshold_rule,
                    recall_mode,
                    early_window_sec,
                    debounce_sec,
                )
                key = f"{target:.2f}"
                if threshold is None:
                    fold_payload["targets"][key] = None
                    pooled[key]["unreachable_folds"] += 1
                    continue
                test_metrics = metrics_at_threshold(test_units, threshold, early_window_sec, debounce_sec)
                fold_payload["targets"][key] = {
                    "target_recall": target,
                    "threshold": threshold,
                    "threshold_rule": threshold_rule,
                    "recall_mode": recall_mode,
                    "calib_event_recall": calib_metrics["event_recall"],
                    "calib_event_recall_lcb95": calib_metrics["event_recall_ci95_wilson"][0],
                    "calib_early_detection_rate": calib_metrics["early_detection_rate"],
                    "calib_early_detection_lcb95": calib_metrics["early_detection_rate_ci95_wilson"][0],
                    "calib_event_hits": calib_metrics["event_hits"],
                    "calib_early_detection_hits": calib_metrics["early_detection_hits"],
                    "calib_n_events": calib_metrics["n_events"],
                    "test": test_metrics,
                }
                bucket = pooled[key]
                bucket["event_hits"] += test_metrics["event_hits"]
                bucket["early_detection_hits"] += test_metrics["early_detection_hits"]
                bucket["n_events"] += test_metrics["n_events"]
                bucket["negative_unit_hits"] += test_metrics["negative_unit_hits"]
                bucket["clustered_alarm_events"] += test_metrics["clustered_alarm_events"]
                bucket["n_negative_units"] += test_metrics["n_negative_units"]
                bucket["negative_hours_proxy"] += test_metrics["negative_hours_proxy"]
                if test_metrics["ttd_mean_sec"] is not None:
                    bucket["ttd_weighted_sum"] += test_metrics["ttd_mean_sec"] * test_metrics["ttd_hit_count"]
                    bucket["ttd_hit_count"] += test_metrics["ttd_hit_count"]
                if test_metrics["early_ttd_median_detected_sec"] is not None:
                    bucket["early_ttd_detected_values"].append(test_metrics["early_ttd_median_detected_sec"])
            fold_results[str(test_fold)] = fold_payload
        if coverage_errors:
            raise SystemExit(f"[error] model {model} has incomplete scoring coverage for folds: {coverage_errors}")
        pooled_results = {}
        for key, bucket in pooled.items():
            event_hits = int(bucket["event_hits"])
            early_hits = int(bucket["early_detection_hits"])
            n_events = int(bucket["n_events"])
            neg_hits = int(bucket["negative_unit_hits"])
            alarm_events = int(bucket["clustered_alarm_events"])
            n_neg = int(bucket["n_negative_units"])
            hours = float(bucket["negative_hours_proxy"])
            ttd_hits = int(bucket["ttd_hit_count"])
            pooled_results[key] = {
                "n_folds": len(folds),
                "unreachable_folds": int(bucket["unreachable_folds"]),
                "threshold_rule": threshold_rule,
                "recall_mode": recall_mode,
                "early_window_sec": early_window_sec,
                "debounce_sec": debounce_sec,
                "event_hits": event_hits,
                "n_events": n_events,
                "event_recall": safe_div(event_hits, n_events),
                "event_recall_ci95_wilson": wilson_ci(event_hits, n_events),
                "early_detection_hits": early_hits,
                "early_detection_rate": safe_div(early_hits, n_events),
                "early_detection_rate_ci95_wilson": wilson_ci(early_hits, n_events),
                "negative_unit_hits": neg_hits,
                "n_negative_units": n_neg,
                "negative_unit_fpr": safe_div(neg_hits, n_neg),
                "negative_unit_fpr_ci95_wilson": wilson_ci(neg_hits, n_neg),
                "clustered_alarm_events": alarm_events,
                "negative_hours_proxy": hours,
                "far_per_pre_event_camera_hour_proxy": safe_div(alarm_events, hours),
                "far_ci95_poisson_exact": poisson_rate_ci(alarm_events, hours),
                "ttd_hit_count": ttd_hits,
                "ttd_mean_sec": safe_div(bucket["ttd_weighted_sum"], ttd_hits) if ttd_hits else None,
                "early_ttd_median_detected_sec_macro": median(bucket["early_ttd_detected_values"])
                if bucket["early_ttd_detected_values"]
                else None,
            }
        output[model] = {
            "fold_results": fold_results,
            "pooled_by_target": pooled_results,
        }
    return output


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_ci(values: list[float | None], digits: int = 3) -> str:
    if not values or values[0] is None or values[1] is None:
        return "n/a"
    return f"{values[0]:.{digits}f}-{values[1]:.{digits}f}"


def write_protocol_md(path: Path, summary: dict) -> None:
    lines = [
        "# RACO Frozen-Fold Protocol Summary",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Scope",
        "",
        "This artifact locks the denominator and split protocol before full detector scoring. It contains no method-performance claim.",
        "",
        "## Units",
        "",
        f"- positive unit: `{summary['unit_of_analysis']['positive']}`",
        f"- negative unit: `{summary['unit_of_analysis']['negative']}`",
        f"- FAR exposure: `{summary['unit_of_analysis']['far_exposure']}`",
        "",
        "## Denominators",
        "",
        f"- folds: `{summary['fold_count']}`",
        f"- positive event units: `{summary['total_positive_event_units']}`",
        f"- negative units: `{summary['total_negative_units']}`",
        f"- canonical conservative negative span hours: `{summary['total_negative_span_hours_conservative']:.2f}`",
        f"- supplementary discrete frame-hour proxy: `{summary['total_negative_frame_hours_discrete_unique']:.2f}`",
        "",
        "| Fold | Positive event units | Negative units | Conservative span hours | Discrete unique frame-hours |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold, payload in summary["per_fold"].items():
        lines.append(
            f"| {fold} | {payload['positive_event_units']} | "
            f"{payload['negative_units']} | {payload['negative_span_hours_conservative']:.2f} | "
            f"{payload['negative_frame_hours_discrete_unique']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-Recall Rule",
            "",
            "For each held-out fold, thresholds/controllers are selected on the other four folds using a fixed-recall validity rule. The default gate requires the calibration Wilson lower confidence bound for early event detection to meet the target recall. The held-out fold then reports early detection rate, full event recall, censored/median time-to-detection, negative-unit FPR, and clustered false alarms per conservative pre-event camera-hour proxy.",
            "",
            "## Required Claim Gate",
            "",
            "A detector/controller arm may be called better only if it preserves held-out fixed-recall validity by confidence-bound criteria and improves clustered FAR and/or early-window TTD with confidence intervals against global threshold, per-camera threshold, conditional/Mondrian CRC, and learned-threshold baselines.",
        ]
    )
    if summary["manifest_issues"]["missing_split_group_rows"]:
        lines.extend(["", "## Manifest Issues", ""])
        lines.append(f"- missing split-group rows: `{summary['manifest_issues']['missing_split_group_rows']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_md(path: Path, result: dict) -> None:
    lines = [
        "# RACO Frozen-Fold Evaluation",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Scope",
        "",
        "Frozen-fold evaluation of fixed-recall event recall, time-to-detection, and pre-event FAR. This is only claim-worthy when prediction coverage is complete and the input scores come from a pre-registered detector run. Synthetic or oracle-style scores are software smoke tests only.",
        "",
        "## Prediction Metadata",
        "",
        f"- prediction CSV: `{result['prediction_csv']}`",
        f"- image column: `{result['prediction_meta']['image_col']}`",
        f"- score column: `{result['prediction_meta']['score_col']}`",
        f"- model column: `{result['prediction_meta']['model_col']}`",
        f"- raw rows: `{result['prediction_meta']['raw_prediction_rows']}`",
        f"- bad rows: `{result['prediction_meta']['bad_prediction_rows']}`",
        "",
    ]
    for model, payload in result["evaluation"].items():
        lines.extend(
            [
                f"## Model `{model}`",
                "",
                "| Target recall | Early detection rate | Full event recall | Clustered FAR/hour | Alarm events / hours | Median early TTD |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for target, pooled in payload["pooled_by_target"].items():
            if pooled["n_events"] == 0 and pooled["unreachable_folds"]:
                lines.append(
                    f"| {target} | unreachable by `{pooled['threshold_rule']}` "
                    f"({pooled['unreachable_folds']}/{pooled['n_folds']} folds) | "
                    "unreachable | unreachable | unreachable | unreachable |"
                )
            else:
                lines.append(
                    f"| {target} | "
                    f"{pooled['early_detection_hits']}/{pooled['n_events']} ({fmt_float(pooled['early_detection_rate'])}; CI {fmt_ci(pooled['early_detection_rate_ci95_wilson'])}) | "
                    f"{pooled['event_hits']}/{pooled['n_events']} ({fmt_float(pooled['event_recall'])}; CI {fmt_ci(pooled['event_recall_ci95_wilson'])}) | "
                    f"{fmt_float(pooled['far_per_pre_event_camera_hour_proxy'])} (CI {fmt_ci(pooled['far_ci95_poisson_exact'])}) | "
                    f"{pooled['clustered_alarm_events']}/{fmt_float(pooled['negative_hours_proxy'], 2)} | "
                    f"{fmt_float(pooled['early_ttd_median_detected_sec_macro'], 1)} |"
                )
        lines.append("")
        lines.extend(
            [
                f"- threshold rule: `{next(iter(payload['pooled_by_target'].values()))['threshold_rule']}`",
                f"- recall mode: `{next(iter(payload['pooled_by_target'].values()))['recall_mode']}`",
                f"- early window: `{next(iter(payload['pooled_by_target'].values()))['early_window_sec']}` sec",
                f"- false-alarm debounce: `{next(iter(payload['pooled_by_target'].values()))['debounce_sec']}` sec",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--pred-csv", type=Path, default=None)
    parser.add_argument("--targets", default="0.90,0.95")
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--model-col", default=None)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--allow-missing-splits", action="store_true")
    parser.add_argument("--threshold-rule", choices=["calib_lcb", "point"], default="calib_lcb")
    parser.add_argument("--recall-mode", choices=["early", "full"], default="early")
    parser.add_argument("--early-window-sec", type=float, default=600.0)
    parser.add_argument("--debounce-sec", type=float, default=300.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    split_to_fold = read_folds(args.folds)
    rows, issues = read_manifest(args.manifest, split_to_fold, allow_missing_splits=args.allow_missing_splits)
    summary = protocol_summary(rows, issues)

    if not args.pred_csv:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_protocol_md(args.out_md, summary)
        print(
            "[done] protocol "
            f"pos_events={summary['total_positive_event_units']} "
            f"neg_units={summary['total_negative_units']} "
            f"neg_hours={summary['total_negative_hours_proxy']:.2f}"
        )
        return 0

    scores_by_model, pred_meta = read_scores(args.pred_csv, args.image_col, args.score_col, args.model_col)
    eval_result = evaluate(
        rows,
        scores_by_model,
        targets,
        args.strict_coverage,
        args.threshold_rule,
        args.recall_mode,
        args.early_window_sec,
        args.debounce_sec,
    )
    result = {
        **summary,
        "prediction_csv": str(args.pred_csv),
        "targets": targets,
        "threshold_rule": args.threshold_rule,
        "recall_mode": args.recall_mode,
        "early_window_sec": args.early_window_sec,
        "debounce_sec": args.debounce_sec,
        "prediction_meta": pred_meta,
        "evaluation": eval_result,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_eval_md(args.out_md, result)
    print(f"[done] evaluated models={','.join(pred_meta['models'])} out={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
