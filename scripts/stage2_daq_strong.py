#!/usr/bin/env python3
"""
Stage 2 DAQ strong pilot.

This is still post-hoc and still uses only exported RT-DETR predictions. The
stronger test combines:
  1) an image-level query-distribution gate learned on calibration images;
  2) a query budget (top-k) to prevent FPPI inflation;
  3) optional rank decay for lower-ranked query outputs.

Candidate selection is calibration-only. The held-out split is used once for the
final report.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import stage2_daq_oneclick as base


def parse_float_list(text):
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    return values


def parse_int_list(text):
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if raw:
            values.append(int(raw))
    return values


def salted_split(labels, calib_frac, salt):
    if str(salt).strip() in ("", "oneclick"):
        return base.stratified_split(labels, calib_frac)

    groups = {0: [], 1: []}
    for img, lab in labels.items():
        groups[1 if base.is_pos(lab) else 0].append(img)

    calib = []
    holdout = []
    for imgs in groups.values():
        ordered = sorted(
            imgs,
            key=lambda p: hashlib.md5((str(salt) + "|" + Path(p).name).encode("utf-8")).hexdigest(),
        )
        n_calib = int(round(len(ordered) * calib_frac))
        calib.extend(ordered[:n_calib])
        holdout.extend(ordered[n_calib:])
    return sorted(calib), sorted(holdout)


def conf_matrix(labels, dets, imgs, max_det):
    scores = np.zeros((len(imgs), max_det), dtype=np.float64)
    labels_pos = np.asarray([base.is_pos(labels[img]) for img in imgs], dtype=bool)
    for row, img in enumerate(imgs):
        vals = np.asarray(dets.get(img, []), dtype=np.float64)
        if vals.size:
            vals = np.sort(vals)[::-1]
            n = min(max_det, vals.size)
            scores[row, :n] = vals[:n]
    return labels_pos, scores


def eval_score_matrix(labels_pos, scores, targets=base.TARGETS):
    max_scores = scores.max(axis=1) if scores.size else np.zeros(labels_pos.shape[0], dtype=np.float64)
    pos_max = max_scores[labels_pos]
    neg_max = max_scores[~labels_pos]
    neg_scores = scores[~labels_pos]

    if pos_max.size == 0 or neg_max.size == 0:
        return {"n_pos": int(pos_max.size), "n_neg": int(neg_max.size), "at_fixed_recall": {}}

    candidates = np.unique(np.concatenate([pos_max, np.linspace(0.0, 1.0, 201)]))[::-1]
    out = {"n_pos": int(pos_max.size), "n_neg": int(neg_max.size), "at_fixed_recall": {}}
    for target in targets:
        chosen = None
        for thr in candidates:
            recall = float(np.mean(pos_max >= thr))
            if recall + 1e-12 >= target:
                fpr = float(np.mean(neg_max >= thr))
                fppi = float(np.sum(neg_scores >= thr) / max(int(neg_max.size), 1))
                chosen = {
                    "thr": round(float(thr), 4),
                    "recall": round(float(recall), 4),
                    "FPR": round(float(fpr), 4),
                    "FPPI": round(float(fppi), 4),
                }
                break
        out["at_fixed_recall"][f"{target:.2f}"] = chosen
    return out


def transform_scores(scores, gate, lam, topk, rank_gamma, floor):
    out = scores.copy()
    if topk < out.shape[1]:
        out[:, topk:] = -1.0

    if rank_gamma > 0.0:
        ranks = np.arange(1, out.shape[1] + 1, dtype=np.float64)
        out = out / np.power(ranks[None, :], rank_gamma)

    scale = np.maximum(float(floor), np.power(np.clip(gate, 0.0, 1.0), float(lam)))
    out = out * scale[:, None]
    out[out < 0.0] = -1.0
    return out


def variant_name(candidate):
    return (
        f"lambda={candidate['lam']},topk={candidate['topk']},"
        f"rank_gamma={candidate['rank_gamma']},floor={candidate['floor']}"
    )


def scan_candidates(scores, labels_pos, gate, raw_rep, args):
    lambdas = parse_float_list(args.lambdas)
    topks = parse_int_list(args.topks)
    rank_gammas = parse_float_list(args.rank_gammas)
    floors = parse_float_list(args.floors)

    if 0.0 not in lambdas:
        lambdas = [0.0] + lambdas
    if scores.shape[1] not in topks:
        topks.append(scores.shape[1])

    raw_row = raw_rep["at_fixed_recall"].get(args.main_target)
    if not raw_row:
        return [], []

    rows = []
    for floor in floors:
        for lam in lambdas:
            for topk in topks:
                for rank_gamma in rank_gammas:
                    if topk == 1 and rank_gamma != 0.0:
                        continue
                    candidate = {
                        "lam": float(lam),
                        "topk": int(topk),
                        "rank_gamma": float(rank_gamma),
                        "floor": float(floor),
                    }
                    transformed = transform_scores(scores, gate, **candidate)
                    rep = eval_score_matrix(labels_pos, transformed)
                    main = rep["at_fixed_recall"].get(args.main_target)
                    if not main:
                        continue
                    fpr_gain = raw_row["FPR"] - main["FPR"]
                    fppi_delta = main["FPPI"] - raw_row["FPPI"]
                    fppi_bad = main["FPPI"] > raw_row["FPPI"] * (1.0 + args.fppi_tol)
                    rows.append(
                        {
                            "candidate": candidate,
                            "report": rep,
                            "main": main,
                            "fpr_gain": round(float(fpr_gain), 4),
                            "fppi_delta": round(float(fppi_delta), 4),
                            "fppi_bad": bool(fppi_bad),
                        }
                    )

    def key(row):
        main = row["main"]
        c = row["candidate"]
        return (
            1 if row["fppi_bad"] else 0,
            main["FPR"],
            main["FPPI"],
            c["topk"],
            c["rank_gamma"],
            c["lam"],
            c["floor"],
        )

    rows.sort(key=key)
    return rows[: args.keep_leaderboard], rows


def write_preds(path, imgs, scores):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for img, vals in zip(imgs, scores):
            keep = vals[vals >= 0.0]
            keep = np.sort(keep)[::-1]
            for conf in keep:
                writer.writerow([img, float(conf)])


def write_leaderboard(path, rows, main_target):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank", "lambda", "topk", "rank_gamma", "floor",
                f"FPR@{main_target}", f"FPPI@{main_target}",
                "fpr_gain", "fppi_delta", "fppi_bad",
            ]
        )
        for idx, row in enumerate(rows, start=1):
            c = row["candidate"]
            main = row["main"]
            writer.writerow(
                [
                    idx, c["lam"], c["topk"], c["rank_gamma"], c["floor"],
                    main["FPR"], main["FPPI"], row["fpr_gain"], row["fppi_delta"], row["fppi_bad"],
                ]
            )


def write_gate_scores(path, labels, imgs, split_name, gate, raw_scores):
    exists = Path(path).exists()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["image", "label", "split", "raw_top1", "gate"])
        for img, g, row in zip(imgs, gate, raw_scores):
            writer.writerow([img, labels[img], split_name, float(row.max()), float(g)])


def compare_rows(raw, strong, main_target, min_fpr_gain, fppi_tol):
    raw_row = raw["at_fixed_recall"].get(main_target)
    strong_row = strong["at_fixed_recall"].get(main_target)
    if not raw_row or not strong_row:
        return "FAIL: main target recall is unreachable"
    fpr_gain = raw_row["FPR"] - strong_row["FPR"]
    fppi_ok = strong_row["FPPI"] <= raw_row["FPPI"] * (1.0 + fppi_tol)
    if fpr_gain >= min_fpr_gain and fppi_ok:
        return "GO: strong DAQ beats hardneg on holdout under FPPI guard"
    if fpr_gain > 0 and fppi_ok:
        return "BORDERLINE-GO: strong DAQ improves FPR/FPPI but margin is modest"
    if fpr_gain > 0:
        return "NO-GO: FPR improves but FPPI inflation breaks the alarm metric"
    return "NO-GO: strong DAQ does not beat hardneg on holdout"


def report_table(raw, strong):
    lines = []
    lines.append("| target recall | hardneg FPR | strong DAQ FPR | delta FPR | hardneg FPPI | strong DAQ FPPI | delta FPPI |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for target in ("0.80", "0.85", "0.90", "0.95"):
        a = raw["at_fixed_recall"].get(target)
        b = strong["at_fixed_recall"].get(target)
        if not a or not b:
            continue
        lines.append(
            f"| {target} | {a['FPR']:.4f} | {b['FPR']:.4f} | {b['FPR'] - a['FPR']:+.4f} | "
            f"{a['FPPI']:.4f} | {b['FPPI']:.4f} | {b['FPPI'] - a['FPPI']:+.4f} |"
        )
    return lines


def write_report(path, summary):
    c = summary["selected_candidate"]
    lines = [
        "# Stage 2 DAQ Strong Report",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- main_target: `{summary['main_target']}`",
        f"- selected: `{variant_name(c)}`",
        f"- calibration images: `{summary['n_calib']}`",
        f"- holdout images: `{summary['n_holdout']}`",
        f"- gate holdout pos_mean: `{summary['gate_stats']['holdout_pos_mean']:.4f}`",
        f"- gate holdout neg_mean: `{summary['gate_stats']['holdout_neg_mean']:.4f}`",
        "",
    ]
    lines.extend(report_table(summary["holdout"]["raw_hardneg"], summary["holdout"]["daq_strong"]))
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "- This is an alarm-level post-hoc pilot, not yet a detector-architecture contribution.",
            "- If `topk=1` is selected, the result supports scene-level false-alarm suppression; box-level recall still needs a later detector-side validation.",
            "- Candidate selection used calibration only; holdout numbers are the numbers to cite.",
        ]
    )
    if summary["audit"]["warnings"]:
        lines.extend(["", "## Warnings"])
        for msg in summary["audit"]["warnings"]:
            lines.append(f"- {msg}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test(audit):
    base.run_self_test(audit)
    labels_pos = np.asarray([True, True, False, False])
    scores = np.asarray(
        [
            [0.9, 0.4, 0.2],
            [0.8, 0.3, 0.1],
            [0.7, 0.6, 0.5],
            [0.2, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    gate = np.asarray([1.0, 0.8, 0.2, 0.2], dtype=np.float64)
    transformed = transform_scores(scores, gate, lam=2.0, topk=1, rank_gamma=0.0, floor=0.0)
    if transformed.shape != scores.shape:
        audit.error("strong self-test failed: transformed score matrix shape changed")
    if np.any(transformed[:, 1:] >= 0.0):
        audit.error("strong self-test failed: topk=1 did not drop lower queries")
    if transformed[2, 0] >= scores[2, 0]:
        audit.error("strong self-test failed: low gate did not suppress a negative top score")
    rep = eval_score_matrix(labels_pos, transformed)
    if "0.80" not in rep["at_fixed_recall"]:
        audit.error("strong self-test failed: evaluator did not produce target metrics")
    audit.note("strong transform self-test passed")


def main():
    ap = argparse.ArgumentParser(description="One-click Stage 2 DAQ strong post-hoc pilot")
    ap.add_argument("--labels", default="data/dfire_local/eval_labels.csv")
    ap.add_argument("--hardneg-pred", default="preds/hardneg_clean5.csv")
    ap.add_argument("--baseline-pred", default="preds/baseline_clean5.csv")
    ap.add_argument("--baseline-results", default="runs/detect/runs_tierb/baseline/results.csv")
    ap.add_argument("--hardneg-results", default="runs/detect/runs_tierb/hardneg/results.csv")
    ap.add_argument("--baseline-weight", default="runs/detect/runs_tierb/baseline/weights/best.pt")
    ap.add_argument("--hardneg-weight", default="runs/detect/runs_tierb/hardneg/weights/best.pt")
    ap.add_argument("--out-dir", default="stage2_daq/strong")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--split-salt", default="", help="empty/oneclick reuses the original oneclick split")
    ap.add_argument("--main-target", default="0.90", choices=["0.80", "0.85", "0.90", "0.95"])
    ap.add_argument("--lambdas", default="0,0.25,0.50,0.75,1.0,1.5,2.0,3.0,4.0,5.0")
    ap.add_argument("--topks", default="1,2,3,5,10,20,100")
    ap.add_argument("--rank-gammas", default="0,0.5,1.0,1.5")
    ap.add_argument("--floors", default="0")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.02)
    ap.add_argument("--fppi-tol", type=float, default=0.05)
    ap.add_argument("--min-fpr-gain", type=float, default=0.015)
    ap.add_argument("--keep-leaderboard", type=int, default=40)
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit non-zero when warnings are present")
    args = ap.parse_args()

    audit = base.Audit()
    if not args.skip_self_test:
        run_self_test(audit)

    if args.max_det <= 0:
        audit.error("--max-det must be positive")
    if args.calib_frac <= 0.05 or args.calib_frac >= 0.95:
        audit.error("--calib-frac must leave meaningful calibration and holdout splits")

    for path in (args.baseline_weight, args.hardneg_weight):
        if not Path(path).exists():
            audit.warn(f"weight missing: {path}")

    labels = base.read_labels(args.labels, audit)
    baseline_dets = base.read_predictions(args.baseline_pred, labels, audit, "baseline")
    hardneg_dets = base.read_predictions(args.hardneg_pred, labels, audit, "hardneg")
    base.parse_results_csv(args.baseline_results, audit, "baseline")
    base.parse_results_csv(args.hardneg_results, audit, "hardneg")
    base.check_timestamps(args.baseline_pred, args.baseline_weight, audit, "baseline")
    base.check_timestamps(args.hardneg_pred, args.hardneg_weight, audit, "hardneg")

    if not audit.ok():
        audit.print()
        sys.exit(2)

    calib_imgs, holdout_imgs = salted_split(labels, args.calib_frac, args.split_salt)
    if set(calib_imgs) & set(holdout_imgs):
        audit.error("split leakage: calibration and holdout overlap")
    if not calib_imgs or not holdout_imgs:
        audit.error("empty calibration or holdout split")
    if not audit.ok():
        audit.print()
        sys.exit(2)

    x_calib, y_calib = base.build_matrix(labels, hardneg_dets, calib_imgs)
    x_holdout, y_holdout = base.build_matrix(labels, hardneg_dets, holdout_imgs)
    model = base.fit_logreg(x_calib, y_calib, args.steps, args.lr, args.l2)
    gate_calib = base.predict_gate(model, x_calib)
    gate_holdout = base.predict_gate(model, x_holdout)

    calib_pos, calib_scores = conf_matrix(labels, hardneg_dets, calib_imgs, args.max_det)
    holdout_pos, holdout_scores = conf_matrix(labels, hardneg_dets, holdout_imgs, args.max_det)
    baseline_holdout_pos, baseline_holdout_scores = conf_matrix(labels, baseline_dets, holdout_imgs, args.max_det)

    raw_calib = eval_score_matrix(calib_pos, calib_scores)
    raw_holdout = eval_score_matrix(holdout_pos, holdout_scores)
    baseline_holdout = eval_score_matrix(baseline_holdout_pos, baseline_holdout_scores)

    top_rows, all_rows = scan_candidates(calib_scores, calib_pos, gate_calib, raw_calib, args)
    if not top_rows:
        audit.error("no candidate variants were evaluable")
        audit.print()
        sys.exit(2)

    selected = top_rows[0]["candidate"]
    strong_holdout_scores = transform_scores(holdout_scores, gate_holdout, **selected)
    strong_holdout = eval_score_matrix(holdout_pos, strong_holdout_scores)
    verdict = compare_rows(raw_holdout, strong_holdout, args.main_target, args.min_fpr_gain, args.fppi_tol)

    if selected["topk"] == 1:
        audit.warn("selected topk=1: treat this as scene-level alarm suppression, not full box-level detector output")
    gate_stats = {
        "calib_pos_mean": float(gate_calib[y_calib > 0.5].mean()),
        "calib_neg_mean": float(gate_calib[y_calib < 0.5].mean()),
        "holdout_pos_mean": float(gate_holdout[y_holdout > 0.5].mean()),
        "holdout_neg_mean": float(gate_holdout[y_holdout < 0.5].mean()),
    }
    if gate_stats["holdout_pos_mean"] <= gate_stats["holdout_neg_mean"]:
        audit.warn("gate does not separate positives above negatives on holdout")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_preds(out_dir / "hardneg_holdout.csv", holdout_imgs, holdout_scores)
    write_preds(out_dir / "daq_strong_holdout.csv", holdout_imgs, strong_holdout_scores)
    base.write_labels(out_dir / "holdout_labels.csv", labels, holdout_imgs)
    gate_scores_path = out_dir / "gate_scores.csv"
    if gate_scores_path.exists():
        gate_scores_path.unlink()
    write_gate_scores(gate_scores_path, labels, calib_imgs, "calib", gate_calib, calib_scores)
    write_gate_scores(gate_scores_path, labels, holdout_imgs, "holdout", gate_holdout, holdout_scores)
    write_leaderboard(out_dir / "candidate_leaderboard.csv", top_rows, args.main_target)

    summary = {
        "verdict": verdict,
        "main_target": args.main_target,
        "selected_candidate": selected,
        "n_calib": len(calib_imgs),
        "n_holdout": len(holdout_imgs),
        "split_salt": args.split_salt,
        "gate_stats": gate_stats,
        "audit": {"errors": audit.errors, "warnings": audit.warnings, "notes": audit.notes},
        "baseline_holdout": baseline_holdout,
        "holdout": {"raw_hardneg": raw_holdout, "daq_strong": strong_holdout},
        "calibration": {
            "raw_hardneg": raw_calib,
            "selected": top_rows[0],
            "leaderboard": top_rows,
            "candidate_count": len(all_rows),
        },
        "artifacts": {
            "summary_json": str(out_dir / "stage2_daq_strong_summary.json"),
            "report_md": str(out_dir / "STAGE2_DAQ_STRONG_REPORT.md"),
            "hardneg_holdout": str(out_dir / "hardneg_holdout.csv"),
            "daq_strong_holdout": str(out_dir / "daq_strong_holdout.csv"),
            "holdout_labels": str(out_dir / "holdout_labels.csv"),
            "gate_scores": str(gate_scores_path),
            "candidate_leaderboard": str(out_dir / "candidate_leaderboard.csv"),
        },
    }

    with (out_dir / "stage2_daq_strong_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(out_dir / "STAGE2_DAQ_STRONG_REPORT.md", summary)

    audit.print()
    print("\n=== Stage 2 DAQ strong holdout result ===")
    print(f"selected={variant_name(selected)} main_target={args.main_target}")
    print(f"gate_holdout_pos_mean={gate_stats['holdout_pos_mean']:.4f} neg_mean={gate_stats['holdout_neg_mean']:.4f}")
    print("\n".join(report_table(raw_holdout, strong_holdout)))
    print(f"\n[verdict] {verdict}")
    print(f"[done] wrote {out_dir / 'STAGE2_DAQ_STRONG_REPORT.md'}")
    if args.strict and audit.warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()
