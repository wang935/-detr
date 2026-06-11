#!/usr/bin/env python3
"""Lightweight FIgLib metadata probe.

Fetches public HPWREN FIgLib directory pages and summarizes sequence names,
sample offsets, and downloadable tar headers without downloading large images.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import subprocess
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


INDEX_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html"
TAR_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html"
BASE_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/"


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr = dict(attrs)
        href = attr.get("href")
        if href:
            self.links.append(href)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36 detr-q3-figlib-probe/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_text(url: str, timeout: int = 30) -> str:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        completed = subprocess.run(
            ["curl.exe", "-L", "--max-time", str(timeout), "-A", HEADERS["User-Agent"], url],
            check=True,
            capture_output=True,
        )
        return completed.stdout.decode("utf-8", errors="replace")


def head(url: str, timeout: int = 30) -> dict[str, str | int | None]:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, method="HEAD", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return {
                "status": resp.status,
                "content_type": resp.headers.get("Content-Type"),
                "content_length": int(resp.headers["Content-Length"]) if resp.headers.get("Content-Length") else None,
                "last_modified": resp.headers.get("Last-Modified"),
                "etag": resp.headers.get("ETag"),
            }
    except Exception:
        completed = subprocess.run(
            ["curl.exe", "-I", "--max-time", str(timeout), "-A", HEADERS["User-Agent"], url],
            check=True,
            capture_output=True,
            text=True,
        )
        headers: dict[str, str] = {}
        status = None
        for line in completed.stdout.splitlines():
            if line.startswith("HTTP/"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    status = int(parts[1])
            elif ":" in line:
                key, value = line.split(":", 1)
                headers[key.lower()] = value.strip()
        length = headers.get("content-length")
        return {
            "status": status,
            "content_type": headers.get("content-type"),
            "content_length": int(length) if length and length.isdigit() else None,
            "last_modified": headers.get("last-modified"),
            "etag": headers.get("etag"),
        }


def extract_links(html: str) -> list[str]:
    parser = AnchorParser()
    parser.feed(html)
    return parser.links


def sequence_links(links: Iterable[str]) -> list[str]:
    seqs = []
    for link in links:
        name = link.rstrip("/")
        if name.endswith("/index.html"):
            name = name[: -len("/index.html")]
        if re.match(r"^\d{8}(?:[._-]\d{6})?[-_]", name):
            seqs.append(name)
    return sorted(set(seqs))


def parse_sequence_name(name: str) -> dict[str, str | None]:
    parts = re.split(r"[_-]", name)
    date = parts[0] if parts else None
    fire_name = None
    station = None
    direction = None
    imager = None
    if len(parts) >= 4:
        if parts[-1] in {"c", "m"} and len(parts) >= 5:
            station = parts[-4]
            direction = parts[-3]
            imager = "-".join(parts[-2:])
            fire_tokens = parts[1:-4]
        else:
            station = parts[-3]
            direction = parts[-2]
            imager = parts[-1]
            fire_tokens = parts[1:-3]
        fire_name = "-".join(fire_tokens) if fire_tokens else None
    return {
        "date": date,
        "fire_name": fire_name,
        "station": station,
        "direction": direction,
        "imager": imager,
    }


def sample_sequence(sequence: str) -> dict[str, object]:
    url = f"{BASE_URL}{sequence}/index.html"
    html = fetch_text(url)
    links = extract_links(html)
    jpgs = [urllib.parse.unquote(link) for link in links if urllib.parse.unquote(link).lower().endswith(".jpg")]
    offsets = []
    for jpg in jpgs:
        match = re.search(r"_([+-]\d+)\.jpg$", jpg)
        if match:
            offsets.append(int(match.group(1)))
    negative_offsets = [v for v in offsets if v < 0]
    positive_offsets = [v for v in offsets if v >= 0]
    valid_neg_5min = [v for v in offsets if v <= -300]
    return {
        "sequence": sequence,
        "url": url,
        "jpg_count": len(jpgs),
        "offset_min": min(offsets) if offsets else None,
        "offset_max": max(offsets) if offsets else None,
        "negative_frame_count": len(negative_offsets),
        "nonnegative_frame_count": len(positive_offsets),
        "valid_negative_frames_with_5min_exclusion": len(valid_neg_5min),
        "realized_negative_minutes_with_5min_exclusion": (
            (max(valid_neg_5min) - min(valid_neg_5min)) / 60.0 if len(valid_neg_5min) >= 2 else 0.0
        ),
        "first_jpg_head": head(f"{BASE_URL}{sequence}/{jpgs[0]}") if jpgs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sample", action="append", default=[])
    args = parser.parse_args()

    index_html = fetch_text(INDEX_URL)
    tar_html = fetch_text(TAR_URL)
    seqs = sequence_links(extract_links(index_html))
    tar_links = [link for link in extract_links(tar_html) if link.endswith(".tgz")]
    parsed = [parse_sequence_name(seq) for seq in seqs]
    station_counts = Counter(p["station"] for p in parsed if p["station"])
    year_counts = Counter((p["date"] or "")[:4] for p in parsed if p["date"])
    sample_names = args.sample or [seqs[0], "20170708_Whittier_syp-n-mobo-c", "20220210-EmeraldFire-marconi-w-mobo-c"]
    sample_names = [s for s in dict.fromkeys(sample_names) if s in seqs]
    samples = [sample_sequence(s) for s in sample_names]
    tar_sample = None
    if "20170708_Whittier_syp-n-mobo-c.tgz" in tar_links:
        tar_sample = head(f"{BASE_URL}Tar/20170708_Whittier_syp-n-mobo-c.tgz")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": {
            "index": INDEX_URL,
            "tar_index": TAR_URL,
        },
        "sequence_count": len(seqs),
        "tar_sequence_count": len(set(link[:-4] for link in tar_links)),
        "sequence_count_vs_tar_count_note": (
            "The sequence directory index and tar index are separate public listings; "
            "any 503/504 mismatch must be reconciled by the full manifest builder."
        ),
        "station_parse_method": "right-to-left station-direction-imager parser; still heuristic until full manifest audit",
        "unique_station_count_estimate": len(station_counts),
        "top_station_counts": station_counts.most_common(20),
        "year_counts": sorted(year_counts.items()),
        "sampled_sequences_n": len(samples),
        "sample_sequences": samples,
        "tar_sample_head": tar_sample,
        "notes": [
            "Directory metadata is public and no large image archive is downloaded.",
            "Station parsing is heuristic because sequence names are semi-structured.",
            "Realized negative camera-hours must be recomputed after event grouping and pre-onset exclusion are finalized.",
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
