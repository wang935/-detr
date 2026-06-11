import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGETS = (0.80, 0.85, 0.90, 0.95)
MIN_THRESHOLD = 1e-3
ARM_RESULT_KEYS = {
    "baseline": "model_A",
    "hardneg": "model_B",
    "baseline_eqstep": "model_C",
    "hardneg_sched": "model_D",
}


def read_labels(path):
    labels = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["image"].strip()] = row["label"].strip().lower()
    return labels


def read_preds(path):
    dets = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dets[row["image"].strip()].append(float(row["conf"]))
    return dets


def max_conf(dets, image):
    vals = dets.get(image, [])
    return max(vals) if vals else 0.0


def evaluate(calib_labels, test_labels, dets):
    pos = {"fire", "smoke"}
    neg = {"none", "distractor", "other"}
    calib_pos = [image for image, label in calib_labels.items() if label in pos]
    test_pos = [image for image, label in test_labels.items() if label in pos]
    test_neg = [image for image, label in test_labels.items() if label in neg]
    observed = {max_conf(dets, image) for image in calib_pos if max_conf(dets, image) > MIN_THRESHOLD}
    grid = {MIN_THRESHOLD}
    grid.update(float(x) for x in np.linspace(0, 1, 201) if x >= MIN_THRESHOLD)
    thresholds = sorted(observed | grid, reverse=True)
    out = {"n_pos": len(test_pos), "n_neg": len(test_neg), "at_fixed_recall": {}}
    for target in TARGETS:
        chosen = None
        for thr in thresholds:
            calib_recall = float(np.mean([max_conf(dets, image) >= thr for image in calib_pos])) if calib_pos else 0.0
            if calib_recall >= target:
                recall = float(np.mean([max_conf(dets, image) >= thr for image in test_pos])) if test_pos else 0.0
                fpr = float(np.mean([max_conf(dets, image) >= thr for image in test_neg])) if test_neg else 0.0
                fppi = float(np.mean([sum(1 for c in dets.get(image, []) if c >= thr) for image in test_neg])) if test_neg else 0.0
                chosen = {
                    "thr": round(float(thr), 4),
                    "calib_recall": round(calib_recall, 4),
                    "recall": round(recall, 4),
                    "FPR": round(fpr, 4),
                    "FPPI": round(fppi, 4),
                }
                break
        out["at_fixed_recall"][f"{target:.2f}"] = chosen
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="baseline,baseline_eqstep,hardneg,hardneg_sched")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--family", default="internimage_t_maskrcnn")
    ap.add_argument("--dataset", default="dfire")
    ap.add_argument("--calib-labels", default="data/dfire_local/eval_calib_labels.csv")
    ap.add_argument("--test-labels", default="data/dfire_local/eval_test_labels.csv")
    args = ap.parse_args()

    calib = read_labels(args.calib_labels)
    test = read_labels(args.test_labels)
    out = {}
    for arm in [x.strip() for x in args.arms.split(",") if x.strip()]:
        pred = Path(args.out_dir) / f"{arm}.csv"
        if not pred.exists():
            continue
        rep = evaluate(calib, test, read_preds(pred))
        out[ARM_RESULT_KEYS[arm]] = {
            "dataset": args.dataset,
            "family": args.family,
            "arm": arm,
            "seed": args.seed,
            **rep,
        }
        r90 = rep["at_fixed_recall"].get("0.90")
        if r90:
            print(
                f"[eval] {arm} R0.90 thr={r90['thr']} "
                f"test_recall={r90['recall']} FPR={r90['FPR']} FPPI={r90['FPPI']}"
            )
    path = Path(args.out_dir) / "gonogo.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eval] wrote {path}")


if __name__ == "__main__":
    main()
