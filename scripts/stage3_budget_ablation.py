#!/usr/bin/env python3
"""Stage 3B query-budget ablation for DAQ-Alarm DETR."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import stage2_daq_oneclick as base
import stage2_daq_strong as strong


TARGETS = ("0.80", "0.85", "0.90", "0.95")


def eval_variant(name, labels_pos, scores, gate, max_det, lam, topk, rank_gamma, floor):
    if name == "hardneg_raw":
        rep = strong.eval_score_matrix(labels_pos, scores)
    else:
        use_gate = np.ones_like(gate) if name.startswith("topk") else gate
        rep = strong.eval_score_matrix(
            labels_pos,
            strong.transform_scores(
                scores,
                use_gate,
                lam=lam,
                topk=topk,
                rank_gamma=rank_gamma,
                floor=floor,
            ),
        )
    return rep


def flatten(name, rep, raw):
    row = {"variant": name}
    for target in TARGETS:
        r = rep["at_fixed_recall"][target]
        b = raw["at_fixed_recall"][target]
        row[f"FPR@{target}"] = r["FPR"]
        row[f"FPPI@{target}"] = r["FPPI"]
        row[f"delta_FPR@{target}"] = round(r["FPR"] - b["FPR"], 4)
        row[f"delta_FPPI@{target}"] = round(r["FPPI"] - b["FPPI"], 4)
    return row


def write_report(path, rows, split_salt):
    lines = [
        "# Stage 3B Query-Budget Ablation",
        "",
        f"- split_salt: `{split_salt}`",
        "",
        "| variant | R0.90 FPR | delta | R0.90 FPPI | delta | R0.95 FPR | delta | R0.95 FPPI | delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | "
            f"{row['FPR@0.90']:.4f} | {row['delta_FPR@0.90']:+.4f} | "
            f"{row['FPPI@0.90']:.4f} | {row['delta_FPPI@0.90']:+.4f} | "
            f"{row['FPR@0.95']:.4f} | {row['delta_FPR@0.95']:+.4f} | "
            f"{row['FPPI@0.95']:.4f} | {row['delta_FPPI@0.95']:+.4f} |"
        )
    lines += [
        "",
        "## Reading Guide",
        "",
        "- `hardneg_raw` is the hard-negative RT-DETR reference.",
        "- `topk1_only` tests whether simply keeping one detection explains the gain.",
        "- `gate_only` tests image-level query calibration without a query budget.",
        "- `gate_topk*` tests the combined DAQ alarm decision policy.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/dfire_local/eval_labels.csv")
    ap.add_argument("--hardneg-pred", default="preds/hardneg_clean5.csv")
    ap.add_argument("--out-dir", default="stage3_daq/ablation")
    ap.add_argument("--split-salt", default="oneclick")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=4.0)
    args = ap.parse_args()

    audit = base.Audit()
    labels = base.read_labels(args.labels, audit)
    dets = base.read_predictions(args.hardneg_pred, labels, audit, "hardneg")
    if not audit.ok():
        audit.print()
        raise SystemExit(2)

    calib_imgs, holdout_imgs = strong.salted_split(labels, args.calib_frac, args.split_salt)
    x_calib, y_calib = base.build_matrix(labels, dets, calib_imgs)
    x_holdout, _ = base.build_matrix(labels, dets, holdout_imgs)
    model = base.fit_logreg(x_calib, y_calib, args.steps, args.lr, args.l2)
    gate_holdout = base.predict_gate(model, x_holdout)
    holdout_pos, holdout_scores = strong.conf_matrix(labels, dets, holdout_imgs, args.max_det)

    raw = strong.eval_score_matrix(holdout_pos, holdout_scores)
    variants = [
        ("hardneg_raw", args.lam, args.max_det, 0.0, 0.0),
        ("topk1_only", 0.0, 1, 0.0, 0.0),
        ("topk2_only", 0.0, 2, 0.0, 0.0),
        ("gate_only", args.lam, args.max_det, 0.0, 0.0),
        ("gate_rank15", args.lam, args.max_det, 1.5, 0.0),
        ("gate_topk1", args.lam, 1, 0.0, 0.0),
        ("gate_topk2", args.lam, 2, 0.0, 0.0),
        ("gate_topk3", args.lam, 3, 0.0, 0.0),
        ("gate_topk2_rank15", args.lam, 2, 1.5, 0.0),
        ("gate_topk3_rank15", args.lam, 3, 1.5, 0.0),
    ]

    reports = {}
    rows = []
    for name, lam, topk, gamma, floor in variants:
        rep = eval_variant(name, holdout_pos, holdout_scores, gate_holdout, args.max_det, lam, topk, gamma, floor)
        reports[name] = rep
        rows.append(flatten(name, rep, raw))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stage3_budget_ablation.csv"
    json_path = out_dir / "stage3_budget_ablation.json"
    md_path = out_dir / "STAGE3_BUDGET_ABLATION.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"split_salt": args.split_salt, "rows": rows, "reports": reports}, f, indent=2)
    write_report(md_path, rows, args.split_salt)

    print(f"[done] wrote {md_path}")
    for row in rows:
        print(
            f"{row['variant']:>18}  "
            f"R90 FPR={row['FPR@0.90']:.4f} ({row['delta_FPR@0.90']:+.4f})  "
            f"FPPI={row['FPPI@0.90']:.4f} ({row['delta_FPPI@0.90']:+.4f})"
        )


if __name__ == "__main__":
    main()
