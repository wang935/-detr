#!/usr/bin/env python3
"""YOLO baseline train/export/eval helper for Stage 3D."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def need_ultralytics():
    try:
        from ultralytics import YOLO  # noqa
    except Exception as exc:
        raise SystemExit(f"[error] ultralytics unavailable: {exc}. Run stage3_3060\\00_create_or_check_env.bat")


def patch_ultralytics_polars():
    """Avoid a broken polars import path during checkpoint saving on this host."""
    try:
        from ultralytics.engine.trainer import BaseTrainer
    except Exception as exc:
        raise SystemExit(f"[error] cannot patch Ultralytics trainer: {exc}")

    def read_results_csv_no_polars(self):
        path = Path(self.csv)
        if not path.exists():
            return {}
        try:
            rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        except Exception:
            return {}
        keys = rows[0].keys() if rows else []
        return {key: [row.get(key, "") for row in rows] for key in keys}

    BaseTrainer.read_results_csv = read_results_csv_no_polars


def train_arm(args, arm, data):
    need_ultralytics()
    patch_ultralytics_polars()
    import torch
    from ultralytics import YOLO

    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] arm={arm} weights={args.weights} data={data} device={dev} epochs={args.epochs}")
    model = YOLO(args.weights)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        project=args.project,
        name=arm,
        exist_ok=True,
        workers=args.workers,
        plots=False,
        val=False,
    )
    save_dir = Path(getattr(model.trainer, "save_dir", Path(args.project) / arm))
    print(f"[train] save_dir={save_dir}")
    weights_dir = save_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if best.exists():
        return best
    if last.exists():
        print(f"[train][warn] {best} missing; using {last}")
        return last
    raise SystemExit(f"[error] training finished but no weights found under {weights_dir}")


def export_preds(weights, labels_csv, out_csv, imgsz, conf, max_det, limit):
    need_ultralytics()
    import torch
    from ultralytics import YOLO

    imgs = [r["image"] for r in csv.DictReader(open(labels_csv, encoding="utf-8"))]
    if limit:
        imgs = imgs[:limit]
    if not Path(weights).exists():
        raise SystemExit(f"[error] weights missing for export: {weights}")
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"[export] weights={weights} images={len(imgs)} device={dev} out={out_csv}")
    model = YOLO(str(weights))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for idx, img in enumerate(imgs, start=1):
            result = model.predict(img, conf=conf, max_det=max_det, imgsz=imgsz, device=dev, verbose=False)[0]
            if result.boxes is not None and len(result.boxes):
                for c in result.boxes.conf.detach().cpu().tolist():
                    writer.writerow([img, float(c)])
                    rows += 1
            if idx % 200 == 0:
                print(f"  ...{idx}/{len(imgs)}")
    print(f"[export] wrote {out_csv} rows={rows}")


def load(pred_csv, labels_csv):
    labels = {r["image"].strip(): r["label"].strip().lower() for r in csv.DictReader(open(labels_csv, encoding="utf-8"))}
    dets = defaultdict(list)
    for r in csv.DictReader(open(pred_csv, encoding="utf-8")):
        dets[r["image"].strip()].append(float(r["conf"]))
    return labels, dets


def max_conf(dets, img):
    vals = dets.get(img, [])
    return max(vals) if vals else 0.0


def evaluate(labels, dets, targets=(0.80, 0.85, 0.90, 0.95)):
    pos = [i for i, lab in labels.items() if lab in ("fire", "smoke")]
    neg = [i for i, lab in labels.items() if lab in ("none", "distractor")]
    thresholds = sorted({max_conf(dets, i) for i in pos} | set(np.linspace(0, 1, 201)), reverse=True)
    out = {"n_pos": len(pos), "n_neg": len(neg), "at_fixed_recall": {}}
    for target in targets:
        chosen = None
        for thr in thresholds:
            recall = float(np.mean([max_conf(dets, i) >= thr for i in pos])) if pos else 0.0
            if recall >= target:
                fpr = float(np.mean([max_conf(dets, i) >= thr for i in neg])) if neg else 0.0
                fppi = float(np.mean([sum(1 for c in dets.get(i, []) if c >= thr) for i in neg])) if neg else 0.0
                chosen = {
                    "thr": round(float(thr), 4),
                    "recall": round(recall, 4),
                    "FPR": round(fpr, 4),
                    "FPPI": round(fppi, 4),
                }
                break
        out["at_fixed_recall"][f"{target:.2f}"] = chosen
    return out


def print_report(name, rep):
    print(f"\n=== {name} pos={rep['n_pos']} neg={rep['n_neg']} ===")
    for target, row in rep["at_fixed_recall"].items():
        print(f"{target}: {row}")


def eval_preds(labels_csv, pred_a, pred_b, out_json):
    labels, dets_a = load(pred_a, labels_csv)
    rep_a = evaluate(labels, dets_a)
    result = {"model_A": {"pred": pred_a, **rep_a}}
    print_report(pred_a, rep_a)
    if pred_b:
        _, dets_b = load(pred_b, labels_csv)
        rep_b = evaluate(labels, dets_b)
        result["model_B"] = {"pred": pred_b, **rep_b}
        print_report(pred_b, rep_b)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["train", "export", "eval", "all"])
    ap.add_argument("--arm", default="both", choices=["baseline", "hardneg", "both"])
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--baseline-data", default="data/dfire_local/dfire_posonly.yaml")
    ap.add_argument("--hardneg-data", default="data/dfire_local/dfire_full.yaml")
    ap.add_argument("--labels", default="data/dfire_local/eval_labels.csv")
    ap.add_argument("--project", default="runs/detect/runs_stage3_yolo")
    ap.add_argument("--pred-dir", default="preds_stage3_yolo")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.project = str(Path(args.project).resolve())

    arms = ["baseline", "hardneg"] if args.arm == "both" else [args.arm]
    weight_paths = {
        "baseline": Path(args.project) / "baseline" / "weights" / "best.pt",
        "hardneg": Path(args.project) / "hardneg" / "weights" / "best.pt",
    }
    if args.mode in ("train", "all"):
        for arm in arms:
            data = args.baseline_data if arm == "baseline" else args.hardneg_data
            weight_paths[arm] = train_arm(args, arm, data)

    if args.mode in ("export", "all"):
        for arm in arms:
            export_preds(
                weight_paths[arm],
                args.labels,
                Path(args.pred_dir) / f"{arm}.csv",
                args.imgsz,
                args.conf,
                args.max_det,
                args.limit,
            )

    if args.mode in ("eval", "all"):
        pred_a = str(Path(args.pred_dir) / "baseline.csv")
        pred_b = str(Path(args.pred_dir) / "hardneg.csv") if (Path(args.pred_dir) / "hardneg.csv").exists() else None
        eval_preds(args.labels, pred_a, pred_b, Path(args.pred_dir) / "yolo_gonogo.json")


if __name__ == "__main__":
    main()
