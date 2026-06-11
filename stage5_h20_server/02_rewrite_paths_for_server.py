#!/usr/bin/env python3
"""Rewrite Windows D:\\detr_Q3 paths in data manifests after copying to Linux."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_WINDOWS_ROOTS = (
    r"D:\detr_Q3",
    "D:/detr_Q3",
    r"D:\\detr_Q3",
)
TEXT_SUFFIXES = {".csv", ".txt", ".yaml", ".yml"}
DATA_DIRS = (
    "data/dfire_local",
    "data/dfire",
    "data/stage5_pv_v2/dfire",
    "data/stage5_pv_v2/dfs",
)


def rewrite_text(text: str, root: str, windows_roots: tuple[str, ...]) -> str:
    updated = text
    for win_root in windows_roots:
        updated = updated.replace(win_root, root)
    if updated != text:
        updated = updated.replace("\\", "/")
    return updated


def iter_files(root: Path):
    for rel_dir in DATA_DIRS:
        base = root / rel_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--windows-root", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-windows", action="store_true")
    args = ap.parse_args()

    if os.name == "nt" and not args.force_windows:
        raise SystemExit("[error] refusing to rewrite paths on Windows; run this on the Linux H20 server")

    root = Path(args.root).resolve()
    root_text = root.as_posix()
    windows_roots = tuple(args.windows_root) + DEFAULT_WINDOWS_ROOTS
    changed = []

    for path in iter_files(root):
        text = path.read_text(encoding="utf-8")
        updated = rewrite_text(text, root_text, windows_roots)
        if updated == text:
            continue
        changed.append(path)
        if not args.dry_run:
            path.write_text(updated, encoding="utf-8")

    action = "would rewrite" if args.dry_run else "rewrote"
    for path in changed:
        print(f"[pathfix] {action}: {path.relative_to(root)}")
    print(f"[ok] path rewrite checked, changed={len(changed)}, root={root_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
