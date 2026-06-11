#!/usr/bin/env python3
"""Reachability preflight for selected FIgLib frame URLs."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def request_url(url: str, method: str, timeout: float) -> tuple[int | None, str, int | None]:
    encoded = urllib.parse.quote(url, safe=":/")
    headers = {"User-Agent": "detr-q3-figlib-url-preflight/1.0"}
    if method == "GET":
        headers["Range"] = "bytes=0-0"
    request = urllib.request.Request(encoded, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if method == "GET":
            response.read(1)
        length = response.headers.get("Content-Length")
        return response.getcode(), method, int(length) if length and length.isdigit() else None


def check_one(row: dict, timeout: float, retries: int) -> dict:
    url = row["frame_url"]
    last_error = ""
    for attempt in range(1, retries + 1):
        for method in ("HEAD", "GET"):
            try:
                status, used_method, content_length = request_url(url, method, timeout)
                ok = status in {200, 206}
                return {
                    "image": row.get("image", ""),
                    "label": row.get("label", ""),
                    "fold": row.get("fold", ""),
                    "eval_unit": row.get("eval_unit", ""),
                    "event_id": row.get("event_id", ""),
                    "sequence_id": row.get("sequence_id", ""),
                    "frame_url": url,
                    "ok": "1" if ok else "0",
                    "status": "" if status is None else str(status),
                    "method": used_method,
                    "content_length": "" if content_length is None else str(content_length),
                    "attempts": attempt,
                    "error": "" if ok else f"unexpected status {status}",
                }
            except urllib.error.HTTPError as exc:
                last_error = f"HTTPError {exc.code}: {exc.reason}"
                if exc.code not in {405, 403}:
                    break
            except Exception as exc:  # noqa: BLE001 - persisted in preflight report.
                last_error = repr(exc)
        if attempt < retries:
            time.sleep(0.2 * attempt)
    return {
        "image": row.get("image", ""),
        "label": row.get("label", ""),
        "fold": row.get("fold", ""),
        "eval_unit": row.get("eval_unit", ""),
        "event_id": row.get("event_id", ""),
        "sequence_id": row.get("sequence_id", ""),
        "frame_url": url,
        "ok": "0",
        "status": "",
        "method": "",
        "content_length": "",
        "attempts": retries,
        "error": last_error,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image",
        "label",
        "fold",
        "eval_unit",
        "event_id",
        "sequence_id",
        "frame_url",
        "ok",
        "status",
        "method",
        "content_length",
        "attempts",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--max-urls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.selected_manifest)
    if args.max_urls is not None:
        rows = rows[: args.max_urls]
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(check_one, row, args.timeout, args.retries) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - start
    results.sort(key=lambda row: row["image"])
    failures = [row for row in results if row["ok"] != "1"]
    ok_count = len(results) - len(failures)
    status_counts = Counter(row["status"] or "error" for row in results)
    method_counts = Counter(row["method"] or "none" for row in results)
    label_failures = Counter(row["label"] for row in failures)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "selected FIgLib frame URL reachability preflight",
        "selected_manifest": str(args.selected_manifest),
        "checked_urls": len(results),
        "ok_urls": ok_count,
        "failed_urls": len(failures),
        "workers": args.workers,
        "timeout_sec": args.timeout,
        "retries": args.retries,
        "elapsed_seconds": elapsed,
        "urls_per_second": len(results) / elapsed if elapsed > 0 else None,
        "status_counts": dict(sorted(status_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "failure_label_counts": dict(sorted(label_failures.items())),
        "verdict": "pass" if not failures else "fail",
        "failure_csv": str(args.out_csv),
        "policy": "If any URL fails, do not start full scoring on the original selected manifest; freeze an exclusion addendum first.",
    }
    write_csv(args.out_csv, results)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# FIgLib URL Reachability Preflight",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Scope",
        "",
        "Reachability preflight over the selected FIgLib frame URLs before full detector scoring. This does not run detector inference and does not establish visual purity.",
        "",
        "## Verdict",
        "",
        f"- verdict: `{payload['verdict']}`",
        f"- checked URLs: `{len(results)}`",
        f"- ok URLs: `{ok_count}`",
        f"- failed URLs: `{len(failures)}`",
        f"- elapsed seconds: `{elapsed:.2f}`",
        f"- URLs/sec: `{payload['urls_per_second']:.2f}`",
        "",
        "## Counts",
        "",
        f"- status counts: `{payload['status_counts']}`",
        f"- method counts: `{payload['method_counts']}`",
        f"- failure label counts: `{payload['failure_label_counts']}`",
        "",
        "## Use Policy",
        "",
        "- If this preflight passes, the first full scoring attempt may use `--fail-on-error`.",
        "- If this preflight fails, do not start full scoring on the original selected manifest.",
        "- Any exclusion requires a dated pre-registered addendum and denominator/protocol regeneration before reportable evaluation.",
    ]
    if failures:
        lines.extend(["", "## Failure Sample", "", "| Image | Label | Error |", "|---|---|---|"])
        for row in failures[:20]:
            lines.append(f"| `{row['image']}` | {row['label']} | `{row['error']}` |")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] checked={len(results)} failures={len(failures)} elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
