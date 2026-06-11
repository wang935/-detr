#!/usr/bin/env python3
"""Run and summarize Stage 3A multi-split DAQ stability tests."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


TARGETS = ("0.80", "0.85", "0.90", "0.95")


def safe_name(text):
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out) or "default"


def metric_row(summary, salt):
    raw = summary["holdout"]["raw_hardneg"]["at_fixed_recall"]
    daq = summary["holdout"]["daq_strong"]["at_fixed_recall"]
    selected = summary["selected_candidate"]
    row = {
        "salt": salt,
        "verdict": summary["verdict"],
        "selected_lam": selected["lam"],
        "selected_topk": selected["topk"],
        "selected_rank_gamma": selected["rank_gamma"],
        "selected_floor": selected["floor"],
    }
    for target in TARGETS:
        a = raw[target]
        b = daq[target]
        row[f"raw_FPR@{target}"] = a["FPR"]
        row[f"daq_FPR@{target}"] = b["FPR"]
        row[f"delta_FPR@{target}"] = round(b["FPR"] - a["FPR"], 4)
        row[f"raw_FPPI@{target}"] = a["FPPI"]
        row[f"daq_FPPI@{target}"] = b["FPPI"]
        row[f"delta_FPPI@{target}"] = round(b["FPPI"] - a["FPPI"], 4)
    return row


def write_markdown(path, rows):
    lines = [
        "# Stage 3A Multi-Split Summary",
        "",
        "| split | selected | R0.90 FPR raw | R0.90 FPR DAQ | delta | R0.90 FPPI raw | R0.90 FPPI DAQ | delta | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        selected = f"lam={row['selected_lam']},topk={row['selected_topk']},gamma={row['selected_rank_gamma']}"
        lines.append(
            f"| {row['salt']} | {selected} | "
            f"{row['raw_FPR@0.90']:.4f} | {row['daq_FPR@0.90']:.4f} | {row['delta_FPR@0.90']:+.4f} | "
            f"{row['raw_FPPI@0.90']:.4f} | {row['daq_FPPI@0.90']:.4f} | {row['delta_FPPI@0.90']:+.4f} | "
            f"{row['verdict']} |"
        )
    if rows:
        mean_delta_fpr = sum(r["delta_FPR@0.90"] for r in rows) / len(rows)
        mean_delta_fppi = sum(r["delta_FPPI@0.90"] for r in rows) / len(rows)
        ok_splits = sum(1 for r in rows if r["delta_FPR@0.90"] < 0 and r["delta_FPPI@0.90"] <= 0)
        lines += [
            "",
            "## Aggregate",
            "",
            f"- splits: `{len(rows)}`",
            f"- R0.90 mean delta FPR: `{mean_delta_fpr:+.4f}`",
            f"- R0.90 mean delta FPPI: `{mean_delta_fppi:+.4f}`",
            f"- splits with both FPR and FPPI improved: `{ok_splits}/{len(rows)}`",
        ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salts", default="oneclick,s1,s2,s3,s4,s5")
    ap.add_argument("--out-root", default="stage3_daq/multisplit")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extra-args", default="", help="extra args passed to stage2_daq_strong.py")
    args = ap.parse_args()

    repo = Path.cwd()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    salts = [s.strip() for s in args.salts.split(",") if s.strip()]
    rows = []
    summaries = {}

    for salt in salts:
        out_dir = out_root / f"split_{safe_name(salt)}"
        summary_path = out_dir / "stage2_daq_strong_summary.json"
        if args.force or not summary_path.exists():
            cmd = [
                sys.executable,
                str(repo / "scripts" / "stage2_daq_strong.py"),
                "--out-dir",
                str(out_dir),
                "--split-salt",
                salt,
            ]
            if args.extra_args:
                cmd += args.extra_args.split()
            print("[run]", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=repo, check=True)
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        summaries[salt] = summary
        rows.append(metric_row(summary, salt))

    csv_path = out_root / "stage3_multisplit_summary.csv"
    json_path = out_root / "stage3_multisplit_summary.json"
    md_path = out_root / "STAGE3_MULTISPLIT_SUMMARY.md"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summaries": summaries}, f, indent=2)
    write_markdown(md_path, rows)

    print(f"[done] wrote {md_path}")
    if rows:
        mean_delta = sum(r["delta_FPR@0.90"] for r in rows) / len(rows)
        print(f"[summary] R0.90 mean delta FPR={mean_delta:+.4f}")


if __name__ == "__main__":
    main()

