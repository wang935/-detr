#!/usr/bin/env python3
"""Compute conventional detection metrics for Stage5-PV v2 last.pt weights."""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_class(family):
    if family == "yolo26n":
        from ultralytics import YOLO

        return YOLO
    from ultralytics import RTDETR

    return RTDETR


def default_weight_from_meta(meta, arm):
    exported = meta.get("exported_weights") or {}
    weight = Path(exported.get(arm, ""))
    if weight.exists():
        return weight
    run_dir = Path(meta.get("run_dir", ""))
    if run_dir:
        weight = run_dir / arm / "weights" / "last.pt"
    return weight


def get_box_metric(metrics, name):
    box = getattr(metrics, "box", None)
    value = getattr(box, name, None) if box is not None else None
    try:
        return float(value)
    except Exception:
        return None


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Conventional mAP metrics for Stage5-PV v2")
    ap.add_argument("--result-root", default="formal_results/stage5_pv_v2")
    ap.add_argument("--out", default="formal_results/stage5_pv_v2/stage5_pv_v2_map_metrics.csv")
    ap.add_argument("--device", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit-runs", type=int, default=0)
    args = ap.parse_args()

    rows = []
    runs = sorted((ROOT / args.result_root).glob("*_seed*/run_meta.json"))
    if args.limit_runs:
        runs = runs[: args.limit_runs]
    for meta_path in runs:
        meta = read_json(meta_path)
        family = meta["family"]
        dataset = meta["dataset"]
        model_cls = load_model_class(family)
        for arm in meta.get("arms", []):
            weight = default_weight_from_meta(meta, arm)
            if not weight.exists():
                raise SystemExit(f"[error] missing last.pt for {meta_path} arm={arm}: {weight}")
            data_key = {
                "baseline": "baseline_data",
                "baseline_eqstep": "eqstep_data",
                "hardneg": "hardneg_data",
                "hardneg_sched": "sched_data",
            }[arm]
            data_yaml = meta[data_key]
            print(f"[val] dataset={dataset} family={family} seed={meta['seed']} arm={arm} weight={weight}")
            model = model_cls(str(weight))
            metrics = model.val(
                data=data_yaml,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device or None,
                plots=False,
                save_json=False,
                verbose=False,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "seed": meta["seed"],
                    "arm": arm,
                    "weight": str(weight),
                    "data": data_yaml,
                    "precision": get_box_metric(metrics, "mp"),
                    "recall": get_box_metric(metrics, "mr"),
                    "mAP50": get_box_metric(metrics, "map50"),
                    "mAP50_95": get_box_metric(metrics, "map"),
                }
            )
            write_csv(ROOT / args.out, rows)
    write_csv(ROOT / args.out, rows)
    print(f"[done] wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
