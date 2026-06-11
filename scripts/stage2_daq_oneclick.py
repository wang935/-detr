#!/usr/bin/env python3
"""
One-click Stage 2 DAQ pilot.

This script is intentionally defensive. It checks the artifacts from Tier B,
validates prediction/label CSVs, runs a synthetic self-test for the metric code,
then performs a held-out DAQ post-hoc calibration test against the hard-negative
baseline.

It is a kill-gate, not the final paper method:
  - If this cannot beat hardneg on holdout FPR without inflating FPPI, a learned
    query-calibration module is unlikely to be worth long training yet.
  - If it beats hardneg cleanly, the next step is to move the gate into the model
    and rerun with clean repeated experiments.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VALID_LABELS = {"fire", "smoke", "none", "distractor"}
POS_LABELS = {"fire", "smoke"}
TARGETS = (0.80, 0.85, 0.90, 0.95)
FEATURE_NAMES = [
    "top1", "top2", "top3", "top5", "top10", "top20",
    "mean", "std", "median", "q75", "q90",
    "frac_ge_005", "frac_ge_010", "frac_ge_020", "frac_ge_030",
    "frac_ge_050", "frac_lt_010", "hist_entropy",
    "top1_minus_top5", "top5_minus_top20", "top1_div_top5",
]
FEATURE_NAMES += [f"hist_{i}" for i in range(10)]


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.notes = []

    def error(self, msg):
        self.errors.append(str(msg))

    def warn(self, msg):
        self.warnings.append(str(msg))

    def note(self, msg):
        self.notes.append(str(msg))

    def ok(self):
        return not self.errors

    def print(self):
        print("\n=== self-check / artifact audit ===")
        for msg in self.errors:
            print(f"[ERROR] {msg}")
        for msg in self.warnings:
            print(f"[WARN]  {msg}")
        for msg in self.notes:
            print(f"[OK]    {msg}")
        print(f"[audit] errors={len(self.errors)} warnings={len(self.warnings)}")


def is_pos(label):
    return label in POS_LABELS


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def md5_key(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read_labels(path, audit):
    labels = {}
    path = Path(path)
    if not path.exists():
        audit.error(f"labels file missing: {path}")
        return labels
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "label"}.issubset(reader.fieldnames):
            audit.error(f"{path} must contain columns image,label")
            return labels
        for row_idx, row in enumerate(reader, start=2):
            img = row["image"].strip()
            lab = row["label"].strip().lower()
            if not img:
                audit.error(f"{path}:{row_idx} empty image path")
                continue
            if lab not in VALID_LABELS:
                audit.error(f"{path}:{row_idx} invalid label {lab!r}")
                continue
            if img in labels:
                audit.error(f"{path}:{row_idx} duplicate label image: {img}")
                continue
            labels[img] = lab
    counts = Counter(labels.values())
    n_pos = sum(counts[k] for k in POS_LABELS)
    n_neg = len(labels) - n_pos
    if n_pos < 100 or n_neg < 100:
        audit.error(f"too few eval positives/negatives: pos={n_pos}, neg={n_neg}")
    else:
        audit.note(f"labels loaded: total={len(labels)}, pos={n_pos}, neg={n_neg}, counts={dict(counts)}")
    return labels


def read_predictions(path, labels, audit, name):
    dets = defaultdict(list)
    row_count = 0
    unknown = 0
    bad_conf = 0
    path = Path(path)
    if not path.exists():
        audit.error(f"{name} predictions missing: {path}")
        return dets
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "conf"}.issubset(reader.fieldnames):
            audit.error(f"{path} must contain columns image,conf")
            return dets
        for row_idx, row in enumerate(reader, start=2):
            img = row["image"].strip()
            try:
                conf = float(row["conf"])
            except Exception:
                bad_conf += 1
                if bad_conf <= 5:
                    audit.error(f"{path}:{row_idx} non-numeric conf: {row.get('conf')!r}")
                continue
            if not np.isfinite(conf) or conf < 0.0 or conf > 1.0:
                bad_conf += 1
                if bad_conf <= 5:
                    audit.error(f"{path}:{row_idx} invalid conf out of [0,1]: {conf}")
                continue
            if img not in labels:
                unknown += 1
                if unknown <= 5:
                    audit.error(f"{path}:{row_idx} prediction image not in labels: {img}")
                continue
            dets[img].append(conf)
            row_count += 1
    for img in list(dets):
        dets[img].sort(reverse=True)

    coverage = len(dets) / max(len(labels), 1)
    per_img = np.asarray([len(dets.get(img, [])) for img in labels], dtype=np.float64)
    if row_count == 0:
        audit.error(f"{name} predictions are empty")
    if coverage < 0.95:
        audit.warn(f"{name} covers only {coverage:.1%} of label images; no-detection images are allowed but verify export conf")
    if per_img.size and np.median(per_img) < 10:
        audit.warn(f"{name} median detections/image is {np.median(per_img):.1f}; DAQ query-distribution signal may be weak")
    audit.note(
        f"{name} predictions loaded: rows={row_count}, images={len(dets)}, "
        f"median_det={np.median(per_img):.1f}, max_det={int(per_img.max()) if per_img.size else 0}"
    )
    return dets


def parse_results_csv(path, audit, name):
    path = Path(path)
    if not path.exists():
        audit.warn(f"{name} results.csv missing: {path}")
        return None
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        audit.warn(f"{name} results.csv is empty")
        return None
    epochs = [row.get("epoch", "") for row in rows]
    dup_epochs = [ep for ep, count in Counter(epochs).items() if count > 1]
    if dup_epochs:
        audit.warn(f"{name} results.csv has duplicate epoch rows: {dup_epochs}")
    numeric_keys = [
        "train/giou_loss", "train/cls_loss", "train/l1_loss",
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    ]
    missing_keys = [key for key in numeric_keys if key not in (rows[0].keys() if rows else [])]
    if missing_keys:
        audit.warn(f"{name} results.csv missing optional metric columns: {missing_keys}")
    for row_idx, row in enumerate(rows, start=2):
        for key in numeric_keys:
            if key not in row:
                continue
            try:
                value = float(row.get(key, "nan"))
            except Exception:
                audit.warn(f"{name} results.csv:{row_idx} non-numeric {key}")
                continue
            if not np.isfinite(value):
                audit.warn(f"{name} results.csv:{row_idx} non-finite optional metric {key}={row.get(key)}")
    for row_idx, row in enumerate(rows, start=2):
        for key in ("val/giou_loss", "val/cls_loss", "val/l1_loss"):
            raw = row.get(key, "")
            if raw and raw.lower() in ("nan", "inf", "-inf"):
                audit.warn(f"{name} results.csv:{row_idx} non-finite validation loss {key}={raw}")
    final = rows[-1]
    audit.note(
        f"{name} final row: epoch={final.get('epoch')} "
        f"mAP50={final.get('metrics/mAP50(B)')} recall={final.get('metrics/recall(B)')}"
    )
    return rows


def check_timestamps(pred_path, weight_path, audit, name):
    pred = Path(pred_path)
    weight = Path(weight_path)
    if not pred.exists() or not weight.exists():
        return
    if pred.stat().st_mtime + 1 < weight.stat().st_mtime:
        audit.warn(f"{name} predictions are older than weights; re-export before trusting results")
    else:
        audit.note(f"{name} prediction timestamp is compatible with weight timestamp")


def entropy_from_hist(hist):
    hist = hist[hist > 0]
    return float(-(hist * np.log(hist)).sum())


def features_from_conf(conf):
    arr = np.asarray(conf, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1, dtype=np.float64)
    arr = np.sort(arr)[::-1]

    def top_mean(k):
        return float(arr[: min(k, arr.size)].mean())

    hist, _ = np.histogram(arr, bins=np.linspace(0.0, 1.0, 11))
    hist = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    top1 = float(arr[0])
    top5 = top_mean(5)
    top20 = top_mean(20)
    feats = [
        top1, top_mean(2), top_mean(3), top5, top_mean(10), top20,
        float(arr.mean()), float(arr.std()), float(np.median(arr)),
        float(np.quantile(arr, 0.75)), float(np.quantile(arr, 0.90)),
        float((arr >= 0.05).mean()), float((arr >= 0.10).mean()),
        float((arr >= 0.20).mean()), float((arr >= 0.30).mean()),
        float((arr >= 0.50).mean()), float((arr < 0.10).mean()),
        entropy_from_hist(hist), top1 - top5, top5 - top20,
        top1 / max(top5, 1e-6),
    ]
    return np.asarray(feats + hist.tolist(), dtype=np.float64)


def stratified_split(labels, calib_frac):
    groups = {0: [], 1: []}
    for img, lab in labels.items():
        groups[1 if is_pos(lab) else 0].append(img)
    calib = []
    holdout = []
    for key, imgs in groups.items():
        ordered = sorted(imgs, key=lambda p: md5_key(os.path.basename(p)))
        n_calib = int(round(len(ordered) * calib_frac))
        calib.extend(ordered[:n_calib])
        holdout.extend(ordered[n_calib:])
    calib = sorted(calib)
    holdout = sorted(holdout)
    return calib, holdout


def build_matrix(labels, dets, imgs):
    x = np.vstack([features_from_conf(dets.get(img, [])) for img in imgs])
    y = np.asarray([1.0 if is_pos(labels[img]) else 0.0 for img in imgs], dtype=np.float64)
    return x, y


def fit_logreg(x, y, steps, lr, l2):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    w = np.zeros(xb.shape[1], dtype=np.float64)

    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    weights = np.where(y > 0.5, 0.5 / pos, 0.5 / neg)
    weights = weights / weights.mean()

    for _ in range(steps):
        p = sigmoid(xb @ w)
        err = (p - y) * weights
        grad = (xb.T @ err) / xb.shape[0]
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return {"mean": mean, "std": std, "w": w}


def predict_gate(model, x):
    xs = (x - model["mean"]) / model["std"]
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return sigmoid(xb @ model["w"])


def make_arrays(labels, dets, imgs, gate=None, lam=0.0):
    max_scores = np.zeros(len(imgs), dtype=np.float64)
    labels_pos = np.asarray([is_pos(labels[img]) for img in imgs], dtype=bool)
    sorted_scores = []
    for idx, img in enumerate(imgs):
        vals = np.asarray(dets.get(img, []), dtype=np.float64)
        if gate is not None:
            vals = vals * float(np.clip(gate[idx], 0.0, 1.0) ** lam)
        vals = np.sort(vals)
        sorted_scores.append(vals)
        max_scores[idx] = float(vals[-1]) if vals.size else 0.0
    return labels_pos, max_scores, sorted_scores


def count_ge(sorted_scores, thr, indices):
    total = 0
    for idx in indices:
        vals = sorted_scores[idx]
        if vals.size:
            total += vals.size - int(np.searchsorted(vals, thr, side="left"))
    return total


def eval_arrays(labels_pos, max_scores, sorted_scores, targets=TARGETS):
    pos_idx = np.where(labels_pos)[0]
    neg_idx = np.where(~labels_pos)[0]
    pos_max = max_scores[pos_idx]
    neg_max = max_scores[neg_idx]
    thrs = np.asarray(sorted(set(pos_max.tolist()) | set(np.linspace(0, 1, 201)), reverse=True), dtype=np.float64)
    out = {"n_pos": int(pos_idx.size), "n_neg": int(neg_idx.size), "at_fixed_recall": {}}
    for target in targets:
        chosen = None
        for thr in thrs:
            recall = float(np.mean(pos_max >= thr)) if pos_idx.size else 0.0
            if recall >= target:
                fpr = float(np.mean(neg_max >= thr)) if neg_idx.size else 0.0
                fppi = float(count_ge(sorted_scores, float(thr), neg_idx) / max(int(neg_idx.size), 1))
                chosen = {
                    "thr": round(float(thr), 4),
                    "recall": round(float(recall), 4),
                    "FPR": round(float(fpr), 4),
                    "FPPI": round(float(fppi), 4),
                }
                break
        out["at_fixed_recall"][f"{target:.2f}"] = chosen
    return out


def eval_dets(labels, dets, imgs, gate=None, lam=0.0):
    labels_pos, max_scores, sorted_scores = make_arrays(labels, dets, imgs, gate, lam)
    return eval_arrays(labels_pos, max_scores, sorted_scores)


def write_preds(path, labels, dets, imgs, gate=None, lam=0.0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for idx, img in enumerate(imgs):
            scale = 1.0 if gate is None else float(np.clip(gate[idx], 0.0, 1.0) ** lam)
            for conf in dets.get(img, []):
                writer.writerow([img, float(conf * scale)])


def write_labels(path, labels, imgs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for img in imgs:
            writer.writerow([img, labels[img]])


def write_gate_scores(path, labels, imgs, split_name, gate, dets):
    exists = Path(path).exists()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["image", "label", "split", "raw_top1", "gate"])
        for img, g in zip(imgs, gate):
            vals = dets.get(img, [])
            writer.writerow([img, labels[img], split_name, max(vals) if vals else 0.0, float(g)])


def select_lambda(raw_rep, variant_reps, main_target, fppi_tol):
    raw_row = raw_rep["at_fixed_recall"][main_target]
    if not raw_row:
        return 0.0

    def rank(item):
        lam, rep = item
        row = rep["at_fixed_recall"][main_target]
        if not row:
            return (1, math.inf, math.inf)
        fppi_limit = raw_row["FPPI"] * (1.0 + fppi_tol)
        fppi_bad = row["FPPI"] > fppi_limit
        return (1 if fppi_bad else 0, row["FPR"], row["FPPI"])

    return float(min(variant_reps.items(), key=rank)[0])


def run_self_test(audit):
    labels = {
        "p1": "fire", "p2": "smoke", "p3": "fire", "p4": "smoke",
        "n1": "none", "n2": "none", "n3": "none", "n4": "distractor",
    }
    dets = {
        "p1": [0.9, 0.1], "p2": [0.8], "p3": [0.7], "p4": [0.6],
        "n1": [0.55], "n2": [0.2], "n3": [], "n4": [0.1],
    }
    imgs = sorted(labels)
    rep = eval_dets(labels, dets, imgs)
    row = rep["at_fixed_recall"]["0.80"]
    if not row or row["recall"] < 0.8:
        audit.error("self-test failed: fixed-recall evaluator did not reach target")
    raw_top = np.asarray([max(dets.get(img, [0.0]) or [0.0]) for img in imgs])
    fake_gate = np.full(len(imgs), 0.5)
    labels_pos, adjusted_top, _ = make_arrays(labels, dets, imgs, fake_gate, 1.0)
    if np.any(adjusted_top > raw_top + 1e-12):
        audit.error("self-test failed: down-gate increased confidence")
    if labels_pos.sum() != 4:
        audit.error("self-test failed: positive label mask wrong")
    audit.note("synthetic self-test passed")


def write_report(path, summary, audit):
    raw = summary["holdout"]["raw_hardneg"]["at_fixed_recall"]
    daq = summary["holdout"]["daq"]["at_fixed_recall"]
    lines = []
    lines.append("# Stage 2 DAQ One-Click Report")
    lines.append("")
    lines.append(f"- selected_lambda: `{summary['selected_lambda']}`")
    lines.append(f"- main_target: `{summary['main_target']}`")
    lines.append(f"- calibration images: `{summary['n_calib']}`")
    lines.append(f"- holdout images: `{summary['n_holdout']}`")
    lines.append(f"- gate holdout pos_mean: `{summary['gate_stats']['holdout_pos_mean']:.4f}`")
    lines.append(f"- gate holdout neg_mean: `{summary['gate_stats']['holdout_neg_mean']:.4f}`")
    lines.append("")
    lines.append("| target recall | hardneg FPR | DAQ FPR | delta FPR | hardneg FPPI | DAQ FPPI | delta FPPI |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for target in ("0.80", "0.85", "0.90", "0.95"):
        a = raw.get(target)
        b = daq.get(target)
        if not a or not b:
            continue
        lines.append(
            f"| {target} | {a['FPR']:.4f} | {b['FPR']:.4f} | {b['FPR'] - a['FPR']:+.4f} | "
            f"{a['FPPI']:.4f} | {b['FPPI']:.4f} | {b['FPPI'] - a['FPPI']:+.4f} |"
        )
    lines.append("")
    verdict = summary["verdict"]
    lines.append(f"**Verdict:** {verdict}")
    if audit.warnings:
        lines.append("")
        lines.append("## Warnings")
        for msg in audit.warnings:
            lines.append(f"- {msg}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="One-click Stage 2 DAQ post-hoc kill-gate")
    ap.add_argument("--labels", default="data/dfire_local/eval_labels.csv")
    ap.add_argument("--hardneg-pred", default="preds/hardneg_clean5.csv")
    ap.add_argument("--baseline-pred", default="preds/baseline_clean5.csv")
    ap.add_argument("--baseline-results", default="runs/detect/runs_tierb/baseline/results.csv")
    ap.add_argument("--hardneg-results", default="runs/detect/runs_tierb/hardneg/results.csv")
    ap.add_argument("--baseline-weight", default="runs/detect/runs_tierb/baseline/weights/best.pt")
    ap.add_argument("--hardneg-weight", default="runs/detect/runs_tierb/hardneg/weights/best.pt")
    ap.add_argument("--out-dir", default="stage2_daq/oneclick")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--main-target", default="0.90", choices=["0.80", "0.85", "0.90", "0.95"])
    ap.add_argument("--lambdas", default="0,0.10,0.25,0.50,0.75,1.0,1.5,2.0,3.0,4.0,5.0")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.02)
    ap.add_argument("--fppi-tol", type=float, default=0.05)
    ap.add_argument("--min-fpr-gain", type=float, default=0.02)
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit non-zero when warnings are present")
    args = ap.parse_args()

    audit = Audit()
    if not args.skip_self_test:
        run_self_test(audit)

    for path in (args.baseline_weight, args.hardneg_weight):
        if not Path(path).exists():
            audit.warn(f"weight missing: {path}")

    labels = read_labels(args.labels, audit)
    baseline_dets = read_predictions(args.baseline_pred, labels, audit, "baseline")
    hardneg_dets = read_predictions(args.hardneg_pred, labels, audit, "hardneg")
    parse_results_csv(args.baseline_results, audit, "baseline")
    parse_results_csv(args.hardneg_results, audit, "hardneg")
    check_timestamps(args.baseline_pred, args.baseline_weight, audit, "baseline")
    check_timestamps(args.hardneg_pred, args.hardneg_weight, audit, "hardneg")

    if args.calib_frac <= 0.05 or args.calib_frac >= 0.95:
        audit.error("--calib-frac must leave meaningful calibration and holdout splits")
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    if not lambdas:
        audit.error("no lambdas provided")
    if 0.0 not in lambdas:
        lambdas = [0.0] + lambdas
        audit.warn("lambda 0 was added so raw hardneg is always a selectable fallback")

    if not audit.ok():
        audit.print()
        sys.exit(2)

    calib_imgs, holdout_imgs = stratified_split(labels, args.calib_frac)
    if set(calib_imgs) & set(holdout_imgs):
        audit.error("split leakage: calibration and holdout overlap")
    if not calib_imgs or not holdout_imgs:
        audit.error("empty calibration or holdout split")
    if not audit.ok():
        audit.print()
        sys.exit(2)

    x_calib, y_calib = build_matrix(labels, hardneg_dets, calib_imgs)
    x_holdout, y_holdout = build_matrix(labels, hardneg_dets, holdout_imgs)
    model = fit_logreg(x_calib, y_calib, args.steps, args.lr, args.l2)
    gate_calib = predict_gate(model, x_calib)
    gate_holdout = predict_gate(model, x_holdout)

    raw_calib = eval_dets(labels, hardneg_dets, calib_imgs)
    raw_holdout = eval_dets(labels, hardneg_dets, holdout_imgs)
    baseline_holdout = eval_dets(labels, baseline_dets, holdout_imgs)

    calib_variants = {}
    for lam in lambdas:
        calib_variants[str(lam)] = eval_dets(labels, hardneg_dets, calib_imgs, gate_calib, lam)
    selected = select_lambda(raw_calib, calib_variants, args.main_target, args.fppi_tol)
    daq_holdout = eval_dets(labels, hardneg_dets, holdout_imgs, gate_holdout, selected)

    raw_main = raw_holdout["at_fixed_recall"][args.main_target]
    daq_main = daq_holdout["at_fixed_recall"][args.main_target]
    if not raw_main or not daq_main:
        verdict = "FAIL: main target recall is unreachable"
    else:
        fpr_gain = raw_main["FPR"] - daq_main["FPR"]
        fppi_delta = daq_main["FPPI"] - raw_main["FPPI"]
        if fpr_gain >= args.min_fpr_gain and daq_main["FPPI"] <= raw_main["FPPI"] * (1.0 + args.fppi_tol):
            verdict = "GO: post-hoc DAQ beats hardneg on holdout under FPPI guard"
        elif fpr_gain > 0 and daq_main["FPPI"] <= raw_main["FPPI"] * (1.0 + args.fppi_tol):
            verdict = "WEAK-GO: DAQ improves FPR but below requested margin"
        elif fpr_gain > 0:
            verdict = "NO-GO: FPR improves but FPPI inflation breaks the alarm metric"
        else:
            verdict = "NO-GO: DAQ does not beat hardneg on holdout"

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
    write_preds(out_dir / "hardneg_holdout.csv", labels, hardneg_dets, holdout_imgs)
    write_preds(out_dir / "daq_holdout.csv", labels, hardneg_dets, holdout_imgs, gate_holdout, selected)
    write_labels(out_dir / "holdout_labels.csv", labels, holdout_imgs)
    gate_scores_path = out_dir / "gate_scores.csv"
    if gate_scores_path.exists():
        gate_scores_path.unlink()
    write_gate_scores(gate_scores_path, labels, calib_imgs, "calib", gate_calib, hardneg_dets)
    write_gate_scores(gate_scores_path, labels, holdout_imgs, "holdout", gate_holdout, hardneg_dets)

    summary = {
        "verdict": verdict,
        "main_target": args.main_target,
        "selected_lambda": selected,
        "n_calib": len(calib_imgs),
        "n_holdout": len(holdout_imgs),
        "gate_stats": gate_stats,
        "audit": {"errors": audit.errors, "warnings": audit.warnings, "notes": audit.notes},
        "baseline_holdout": baseline_holdout,
        "holdout": {"raw_hardneg": raw_holdout, "daq": daq_holdout},
        "calibration": {"raw_hardneg": raw_calib, "variants": calib_variants},
        "artifacts": {
            "summary_json": str(out_dir / "stage2_daq_summary.json"),
            "report_md": str(out_dir / "STAGE2_DAQ_REPORT.md"),
            "hardneg_holdout": str(out_dir / "hardneg_holdout.csv"),
            "daq_holdout": str(out_dir / "daq_holdout.csv"),
            "holdout_labels": str(out_dir / "holdout_labels.csv"),
            "gate_scores": str(gate_scores_path),
        },
    }
    with (out_dir / "stage2_daq_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(out_dir / "STAGE2_DAQ_REPORT.md", summary, audit)

    audit.print()
    print("\n=== Stage 2 DAQ holdout result ===")
    print(f"selected_lambda={selected} main_target={args.main_target}")
    print(f"gate_holdout_pos_mean={gate_stats['holdout_pos_mean']:.4f} neg_mean={gate_stats['holdout_neg_mean']:.4f}")
    print("| target | hardneg FPR | DAQ FPR | delta FPR | hardneg FPPI | DAQ FPPI | delta FPPI |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for target in ("0.80", "0.85", "0.90", "0.95"):
        a = raw_holdout["at_fixed_recall"].get(target)
        b = daq_holdout["at_fixed_recall"].get(target)
        if a and b:
            print(
                f"| {target} | {a['FPR']:.4f} | {b['FPR']:.4f} | {b['FPR'] - a['FPR']:+.4f} | "
                f"{a['FPPI']:.4f} | {b['FPPI']:.4f} | {b['FPPI'] - a['FPPI']:+.4f} |"
            )
    print(f"\n[verdict] {verdict}")
    print(f"[done] wrote {out_dir / 'STAGE2_DAQ_REPORT.md'}")
    if args.strict and audit.warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()
