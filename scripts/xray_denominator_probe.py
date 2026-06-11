#!/usr/bin/env python3
"""Lightweight X-ray denominator metadata probe.

This does not download datasets. It fetches the official SIXray data_list and
checks whether small metadata links are reachable, then writes the evidence
state used by the XRayThreat-RC denominator audit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SIXRAY_DATA_LIST = "https://raw.githubusercontent.com/MeioJane/SIXray/master/data_list.txt"


def fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "detr-q3-xray-probe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        completed = subprocess.run(
            ["curl.exe", "-L", "--max-time", str(timeout), "-A", "detr-q3-xray-probe/1.0", url],
            check=True,
            capture_output=True,
        )
        return completed.stdout.decode("utf-8", errors="replace")


def head_status(url: str, timeout: int = 20) -> dict[str, object]:
    completed = subprocess.run(
        ["curl.exe", "-sS", "-I", "--max-time", str(timeout), url],
        capture_output=True,
        text=True,
    )
    status = None
    headers: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                status = int(parts[1])
        elif ":" in line:
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip()
    return {
        "url": url,
        "returncode": completed.returncode,
        "status": status,
        "content_type": headers.get("content-type"),
        "content_length": headers.get("content-length"),
        "stderr": completed.stderr.strip()[:300],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data_list_text = fetch_text(SIXRAY_DATA_LIST)
    links = [line.strip() for line in data_list_text.splitlines() if line.strip()]
    metadata_links = [link for link in links if re.search(r"(Annotation|ImageSet).*\.zip$", link)]
    head_results = {}
    for link in metadata_links:
        name = link.rsplit("/", 1)[-1]
        variants = [link]
        if link.startswith("http://"):
            variants.append("https://" + link[len("http://") :])
        head_results[name] = [head_status(url) for url in variants]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": {
            "sixray_data_list": SIXRAY_DATA_LIST,
            "pidray_repo": "https://github.com/lutao2021/PIDray",
            "sixray_repo": "https://github.com/MeioJane/SIXray",
            "sixrayd_page": "https://rose1.ntu.edu.sg/dataset/SIXray-D/",
            "opixray_repo": "https://github.com/OPIXray-author/OPIXray",
        },
        "sixray_data_list": {
            "line_count": len(links),
            "links": links,
            "metadata_links": metadata_links,
            "head_results": head_results,
            "interpretation": (
                "The official GitHub data_list is reachable, but the tested "
                "Kingsoft metadata artifacts return 404 in this environment. "
                "This does not test the Baidu download path."
            ),
        },
        "public_metadata_manual_from_source_pages": {
            "PIDray": {
                "total_images": 124486,
                "categories": 12,
                "splits": {"train": 76913, "easy": 24758, "hard": 9746, "hidden": 13069},
            },
            "SIXray": {
                "total_images": 1059231,
                "positive_images_original_repo": 8929,
                "classes": ["gun", "knife", "wrench", "pliers", "scissors", "hammer"],
            },
            "SIXray-D": {
                "positive_images": 11401,
                "annotation_boxes": 23470,
                "annotated_classes": ["gun", "knife", "wrench", "pliers", "scissors"],
                "negative_denominator_count": "unknown until access/original SIXray folder audit",
            },
            "OPIXray": {
                "v1_images": 8885,
                "v2_background_images": 10000,
            },
        },
        "current_verdict": "CONDITIONAL_GO_ACCESS_REQUIRED",
        "go_requires": [
            "SIXray-D annotations or equivalent detection boxes for positives.",
            "Original SIXray/SIXray-D image folders with confirmed same-domain negative images.",
            "A counted negative denominator, target >=20000 or explicit smaller-pilot CI.",
            "Duplicate/group split audit before training.",
        ],
        "blocked_claims": [
            "bag-level FAR",
            "PIDray-only FAR",
            "mixed-dataset recall/FAR headline",
            "training-ready Q1 primary before access and counted negatives",
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
