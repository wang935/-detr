#!/usr/bin/env python3
"""
Post-hoc DAQ pilot from exported RT-DETR query/confidence distributions.

This is a fast Stage-2 kill-gate before changing the detector architecture:
read hard-negative predictions exported with low conf and high max_det, learn an
image-level query-distribution gate on a calibration split, then test whether the
gate lowers FPR/FPPI on a held-out split at fixed recall.

The holdout result is the only number that matters. The calibration split is
used only to fit the gate and choose the gate strength.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np


TARGETS = (0.80, 0.85, 0.90, 0.95)


def _sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def load_labels(path):
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["image"].strip()] = row["label"].strip().lower()
    return labels


def load_dets(path):
    dets = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dets[row["image"].strip()].append(float(row["conf"]))
    for img in list(dets):
        dets[img] = sorted(dets[img], reverse=True)
    return dets


def is_pos(label):
    return label in ("fire", "smoke")


def stratified_split(labels, calib_frac):
    groups = {0: [], 1: []}
    for img, lab in labels.items():
        groups[1 if is_pos(lab) else 0].append(img)

    calib, holdout = set(), set()
    for key, imgs in groups.items():
        ordered = sorted(
            imgs,
            key=lambda s: hashlib.md5(os.path.basename(s).encode("utf-8")).hexdigest(),
        )
        n_calib = int(round(len(ordered) * calib_frac))
        calib.update(ordered[:n_calib])
        holdout.update(ordered[n_calib:])
    return sorted(calib), sorted(holdout)


def qfeatures(conf):
    conf = np.asarray(sorted(conf, reverse=True), dtype=np.float64)
    if conf.size == 0:
        conf = np.zeros(1, dtype=np.float64)

    def top_mean(k):
        return float(conf[: min(k, conf.size)].mean())

    hist, _ = np.histogram(conf, bins=np.linspace(0.0, 1.0, 11))
    hist = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    top1 = float(conf[0])
    top5 = top_mean(5)
    top20 = top_mean(20)
    feats = [
        top1,
        top_mean(2),
        top_mean(3),
        top5,
        top_mean(10),
        top20,
        float(conf.mean()),
        float(conf.std()),
        float(np.median(conf)),
        float(np.quantile(conf, 0.75)),
        float(np.quantile(conf, 0.90)),
        float((conf >= 0.05).mean()),
        float((conf >= 0.10).mean()),
        float((conf >= 0.20).mean()),
        float((conf >= 0.30).mean()),
        float((conf >= 0.50).mean()),
        float((conf < 0.10).mean()),
        _entropy(hist),
        top1 - top5,
        top5 - top20,
        top1 / max(top5, 1e-6),
    ]
    return np.asarray(feats + hist.tolist(), dtype=np.float64)


def build_xy(labels, dets, imgs):
    x = np.vstack([qfeatures(dets.get(img, [])) for img in imgs])
    y = np.asarray([1.0 if is_pos(labels[img]) else 0.0 for img in imgs], dtype=np.float64)
    return x, y


def fit_logreg(x, y, steps=2500, lr=0.05, l2=0.02):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    w = np.zeros(xb.shape[1], dtype=np.float64)

    # mild class balancing: the split is close to balanced, but keep it explicit.
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    weights = np.where(y > 0.5, 0.5 / pos, 0.5 / neg)
    weights = weights / weights.mean()

    for _ in range(steps):
        p = _sigmoid(xb @ w)
        err = (p - y) * weights
        grad = (xb.T @ err) / xb.shape[0]
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return {"mean": mean, "std": std, "w": w}


def predict_gate(model, x):
    xs = (x - model["mean"]) / model["std"]
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return _sigmoid(xb @ model["w"])


def adjust_dets(dets, imgs, gates, lam, mode):
    out = {}
    for img, gate in zip(imgs, gates):
        vals = dets.get(img, [])
        if not vals:
            out[img] = []
            continue
        if mode == "down":
            # Conservative gate: never increase any detection confidence.
            # This is safer for FPPI than logit blending, and matches the
            # false-alarm suppression objective of the first DAQ pilot.
            scale = float(np.clip(gate, 0.0, 1.0) ** lam)
            out[img] = [float(c * scale) for c in vals]
        elif mode == "logit":
            z_gate = lam * float(_logit(gate))
            out[img] = [float(_sigmoid(float(_logit(c)) + z_gate)) for c in vals]
        else:
            raise ValueError(f"unknown mode: {mode}")
    return out


def maxc(dets, img):
    vals = dets.get(img, [])
    return max(vals) if vals else 0.0


def eval_at_fixed_recall(labels, dets, imgs, targets=TARGETS):
    pos = [img for img in imgs if is_pos(labels[img])]
    neg = [img for img in imgs if not is_pos(labels[img])]
    thrs = sorted({maxc(dets, img) for img in pos} | set(np.linspace(0, 1, 201)), reverse=True)
    out = {"n_pos": len(pos), "n_neg": len(neg), "at_fixed_recall": {}}
    for target in targets:
        chosen = None
        for thr in thrs:
            recall = np.mean([1.0 if maxc(dets, img) >= thr else 0.0 for img in pos])
            if recall >= target:
                fpr = np.mean([1.0 if maxc(dets, img) >= thr else 0.0 for img in neg])
                fppi = np.mean([sum(1 for c in dets.get(img, []) if c >= thr) for img in neg])
                chosen = {
                    "thr": round(float(thr), 4),
                    "recall": round(float(recall), 4),
                    "FPR": round(float(fpr), 4),
                    "FPPI": round(float(fppi), 4),
                }
                break
        out["at_fixed_recall"][f"{target:.2f}"] = chosen
    return out


def write_preds(path, dets, imgs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "conf"])
        for img in imgs:
            for conf in dets.get(img, []):
                w.writerow([img, conf])


def write_labels(path, labels, imgs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "label"])
        for img in imgs:
            w.writerow([img, labels[img]])


def print_report(name, rep):
    print(f"\n=== {name} (pos={rep['n_pos']}, neg={rep['n_neg']}) ===")
    print(f"{'recall':>8} | {'thr':>7} | {'FPR':>7} | {'FPPI':>7}")
    for target, row in rep["at_fixed_recall"].items():
        if row:
            print(f"{target:>8} | {row['thr']:>7} | {row['FPR']:>7} | {row['FPPI']:>7}")
        else:
            print(f"{target:>8} | unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="hard-negative predictions CSV")
    ap.add_argument("--labels", required=True, help="eval labels CSV")
    ap.add_argument("--out-dir", default="stage2_daq")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--target", default="0.90", choices=["0.80", "0.85", "0.90", "0.95"])
    ap.add_argument("--lambdas", default="0.25,0.5,1.0,1.5,2.0,3.0")
    ap.add_argument("--mode", default="down", choices=["down", "logit"])
    args = ap.parse_args()

    labels = load_labels(args.labels)
    dets = load_dets(args.pred)
    calib_imgs, holdout_imgs = stratified_split(labels, args.calib_frac)
    x_calib, y_calib = build_xy(labels, dets, calib_imgs)
    x_holdout, y_holdout = build_xy(labels, dets, holdout_imgs)

    model = fit_logreg(x_calib, y_calib)
    gate_calib = predict_gate(model, x_calib)
    gate_holdout = predict_gate(model, x_holdout)

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    raw_calib = {img: dets.get(img, []) for img in calib_imgs}
    raw_holdout = {img: dets.get(img, []) for img in holdout_imgs}
    raw_calib_rep = eval_at_fixed_recall(labels, raw_calib, calib_imgs)
    raw_holdout_rep = eval_at_fixed_recall(labels, raw_holdout, holdout_imgs)

    calib_variants = {}
    for lam in lambdas:
        adj = adjust_dets(dets, calib_imgs, gate_calib, lam, args.mode)
        calib_variants[str(lam)] = eval_at_fixed_recall(labels, adj, calib_imgs)

    raw_target_row = raw_calib_rep["at_fixed_recall"][args.target]

    def key_for(rep):
        row = rep["at_fixed_recall"][args.target]
        if not row:
            return (math.inf, math.inf)
        # Prefer FPR gains that do not inflate FPPI. If no lambda satisfies
        # this guard, the FPPI term will still push the selection away from
        # pathological "many boxes survive" behavior.
        fppi_guard = row["FPPI"] > raw_target_row["FPPI"] * 1.05
        return (1 if fppi_guard else 0, row["FPR"], row["FPPI"])

    best_lam = min(lambdas, key=lambda lam: key_for(calib_variants[str(lam)]))
    holdout_adj = adjust_dets(dets, holdout_imgs, gate_holdout, best_lam, args.mode)
    holdout_rep = eval_at_fixed_recall(labels, holdout_adj, holdout_imgs)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    hardneg_holdout_csv = os.path.join(out_dir, "hardneg_holdout.csv")
    daq_holdout_csv = os.path.join(out_dir, "daq_posthoc_holdout.csv")
    holdout_labels_csv = os.path.join(out_dir, "holdout_labels.csv")
    write_preds(hardneg_holdout_csv, raw_holdout, holdout_imgs)
    write_preds(daq_holdout_csv, holdout_adj, holdout_imgs)
    write_labels(holdout_labels_csv, labels, holdout_imgs)

    summary = {
        "pred": args.pred,
        "labels": args.labels,
        "calib_frac": args.calib_frac,
        "target_for_lambda_selection": args.target,
        "mode": args.mode,
        "n_calib": int(len(calib_imgs)),
        "n_holdout": int(len(holdout_imgs)),
        "gate_stats": {
            "calib_pos_mean": float(gate_calib[y_calib > 0.5].mean()),
            "calib_neg_mean": float(gate_calib[y_calib < 0.5].mean()),
            "holdout_pos_mean": float(gate_holdout[y_holdout > 0.5].mean()),
            "holdout_neg_mean": float(gate_holdout[y_holdout < 0.5].mean()),
        },
        "selected_lambda": best_lam,
        "calib": {"raw": raw_calib_rep, "variants": calib_variants},
        "holdout": {"raw_hardneg": raw_holdout_rep, "daq_posthoc": holdout_rep},
        "artifacts": {
            "hardneg_holdout": hardneg_holdout_csv,
            "daq_posthoc_holdout": daq_holdout_csv,
            "holdout_labels": holdout_labels_csv,
        },
    }
    out_json = os.path.join(out_dir, "daq_posthoc_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[split] calib={len(calib_imgs)} holdout={len(holdout_imgs)}")
    print(
        "[gate] holdout pos_mean="
        f"{summary['gate_stats']['holdout_pos_mean']:.4f} "
        f"neg_mean={summary['gate_stats']['holdout_neg_mean']:.4f}"
    )
    print(f"[select] lambda={best_lam} selected on calibration target recall {args.target}")
    print_report("holdout raw hardneg", raw_holdout_rep)
    print_report("holdout DAQ posthoc", holdout_rep)

    a = raw_holdout_rep["at_fixed_recall"][args.target]
    b = holdout_rep["at_fixed_recall"][args.target]
    if a and b:
        print(
            f"\n[delta @{args.target}] FPR {a['FPR']} -> {b['FPR']} "
            f"({b['FPR'] - a['FPR']:+.4f}), FPPI {a['FPPI']} -> {b['FPPI']} "
            f"({b['FPPI'] - a['FPPI']:+.4f})"
        )
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
