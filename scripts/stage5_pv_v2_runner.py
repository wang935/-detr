#!/usr/bin/env python3
"""Stage5-PV v2 runner isolated from the original Stage 5 scripts."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (0.80, 0.85, 0.90, 0.95)
MIN_THRESHOLD = 1e-3
DEFAULT_SCHED_NEG_FRAC = 0.60
ARM_ORDER = ("baseline", "baseline_eqstep", "hardneg", "hardneg_sched")
ARM_RESULT_KEYS = {
    "baseline": "model_A",
    "hardneg": "model_B",
    "baseline_eqstep": "model_C",
    "hardneg_sched": "model_D",
}
ARM_GROUPS = {
    "baseline": ["baseline"],
    "baseline_eqstep": ["baseline_eqstep"],
    "hardneg": ["hardneg"],
    "hardneg_sched": ["hardneg_sched"],
    "all4": list(ARM_ORDER),
}
DATASET_DEFAULTS = {
    "dfire": {
        "baseline_data": "data/dfire_local/dfire_posonly.yaml",
        "eqstep_data": "data/stage5_pv_v2/dfire/dfire_posonly_equalstep.yaml",
        "hardneg_data": "data/dfire_local/dfire_full.yaml",
        "sched_data": "data/stage5_pv_v2/dfire/dfire_hardneg_sched.yaml",
        "labels": "data/dfire_local/eval_labels.csv",
        "eval_calib_labels": "data/dfire_local/eval_calib_labels.csv",
        "eval_test_labels": "data/dfire_local/eval_test_labels.csv",
        "pos_labels": "fire,smoke",
        "neg_labels": "none,distractor,other",
    },
    "dfs": {
        "baseline_data": "data/stage5_pv_v2/dfs/dfs_posonly.yaml",
        "eqstep_data": "data/stage5_pv_v2/dfs/dfs_posonly_equalstep.yaml",
        "hardneg_data": "data/stage5_pv_v2/dfs/dfs_full.yaml",
        "sched_data": "data/stage5_pv_v2/dfs/dfs_hardneg_sched.yaml",
        "labels": "data/stage5_pv_v2/dfs/eval_labels.csv",
        "eval_calib_labels": "data/stage5_pv_v2/dfs/eval_calib_labels.csv",
        "eval_test_labels": "data/stage5_pv_v2/dfs/eval_test_labels.csv",
        "pos_labels": "fire,smoke",
        "neg_labels": "other,none,distractor",
    },
}


def resolve(path):
    return Path(path).resolve()


def is_downloadable_weight(value):
    path = Path(str(value))
    return not path.is_absolute() and len(path.parts) == 1 and path.suffix.lower() == ".pt"


def model_weight_arg(value):
    if is_downloadable_weight(value) and not Path(value).exists():
        return str(value)
    return str(resolve(value))


def need_ultralytics():
    try:
        import torch  # noqa
        import ultralytics  # noqa
    except Exception as exc:
        raise SystemExit(f"[error] missing torch/ultralytics: {exc}")


def patch_ultralytics_polars():
    try:
        from ultralytics.engine.trainer import BaseTrainer
    except Exception as exc:
        raise SystemExit(f"[error] cannot patch Ultralytics trainer: {exc}")

    def read_results_csv_no_polars(self):
        path = Path(self.csv)
        if not path.exists():
            return {}
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        keys = rows[0].keys() if rows else []
        return {key: [row.get(key, "") for row in rows] for key in keys}

    BaseTrainer.read_results_csv = read_results_csv_no_polars


def family_defaults(family):
    if family == "yolo26n":
        return {"weights": "yolo26n.pt", "batch": 16, "epochs": 300, "label": "YOLO26n", "backend": "yolo"}
    if family == "rtdetr":
        return {"weights": "rtdetr-l.pt", "batch": 4, "epochs": 300, "label": "RT-DETR-L", "backend": "rtdetr"}
    raise ValueError(family)


def yaml_load(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def yaml_value(path, key):
    data = yaml_load(path)
    if isinstance(data, dict) and key in data:
        return data[key]
    raise SystemExit(f"[error] {path} missing {key}:")


def yaml_names(path):
    names = yaml_value(path, "names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(v) for idx, v in enumerate(names)}
    raise SystemExit(f"[error] {path} has unsupported names format")


def write_yaml_like(src_yaml, dst_yaml, train_txt):
    data = yaml_load(src_yaml)
    data["train"] = str(resolve(train_txt))
    import yaml

    atomic_write_text(dst_yaml, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def list_from_yaml(data_yaml, key):
    value = yaml_value(data_yaml, key)
    path = Path(value)
    if path.is_dir():
        return sorted(str(p.resolve()) for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not path.exists():
        raise SystemExit(f"[error] list path not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_images(data_yaml):
    return list_from_yaml(data_yaml, "train")


def val_images(data_yaml):
    return list_from_yaml(data_yaml, "val")


def image_label_path(image):
    p = Path(image)
    parts = list(p.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    if "images" in p.as_posix():
        return Path(p.as_posix().replace("/images/", "/labels/")).with_suffix(".txt")
    return p.with_suffix(".txt")


def is_positive_image(image):
    label = image_label_path(image)
    if not label.exists():
        return False
    text = label.read_text(encoding="utf-8", errors="ignore").strip()
    return bool(text)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_list_file(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_list(path, items):
    atomic_write_text(path, "\n".join(items) + ("\n" if items else ""))


def ensure_equalstep_dataset(args):
    baseline_images = train_images(args.baseline_data)
    hardneg_images = train_images(args.hardneg_data)
    if not baseline_images or not hardneg_images:
        raise SystemExit("[error] baseline/hardneg train lists must be non-empty")
    target = len(hardneg_images)
    train_txt = Path(args.eqstep_data).with_name("train_posonly_equalstep.txt")
    if Path(args.eqstep_data).exists() and train_txt.exists():
        existing = read_list_file(train_txt)
        if len(existing) == target:
            print(f"[eqstep] ready {args.eqstep_data} count={len(existing)}")
            return
    rng = np.random.default_rng(20260604)
    shuffled = list(baseline_images)
    rng.shuffle(shuffled)
    repeated = []
    while len(repeated) < target:
        repeated.extend(shuffled)
    repeated = repeated[:target]
    write_list(train_txt, repeated)
    write_yaml_like(args.baseline_data, args.eqstep_data, train_txt)
    print(f"[eqstep] wrote {args.eqstep_data} count={len(repeated)}")


def ensure_sched_dataset(args):
    sched_yaml = Path(args.sched_data)
    audit_path = sched_yaml.with_name("hardneg_sched_audit.json")
    if sched_yaml.exists() and audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception:
            audit = {}
        if abs(float(audit.get("requested_sched_neg_frac", -1.0)) - float(args.sched_neg_frac)) < 1e-9:
            return
    baseline_images = train_images(args.baseline_data)
    hardneg_images = train_images(args.hardneg_data)
    negatives = [image for image in hardneg_images if not is_positive_image(image)]
    positives = list(baseline_images)
    target = len(hardneg_images)
    if not positives or not negatives:
        sched = list(hardneg_images)
        target_neg = len(negatives)
        target_pos = target - target_neg
    else:
        base_neg_frac = len(negatives) / max(target, 1)
        neg_frac = min(0.85, max(args.sched_neg_frac, base_neg_frac))
        target_neg = min(target - 1, max(len(negatives), int(round(target * neg_frac))))
        target_pos = target - target_neg
        rng = np.random.default_rng(20260604)
        pos_pool = list(positives)
        neg_pool = list(negatives)
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)
        sched = []
        while len(sched) < target_pos:
            sched.extend(pos_pool)
        sched = sched[:target_pos]
        neg_sched = []
        while len(neg_sched) < target_neg:
            neg_sched.extend(neg_pool)
        sched.extend(neg_sched[:target_neg])
        rng.shuffle(sched)
    train_txt = sched_yaml.with_name("train_hardneg_sched.txt")
    write_list(train_txt, sched[:target])
    write_yaml_like(args.hardneg_data, sched_yaml, train_txt)
    audit = {
        "dataset": args.dataset,
        "definition": "static hard-negative reweighted list with total image count matched to hardneg",
        "target_train_count": target,
        "baseline_positive_count": len(positives),
        "hardneg_negative_count": len(negatives),
        "sched_positive_count": target_pos,
        "sched_negative_count": target_neg,
        "unique_positive_coverage": round(target_pos / max(len(positives), 1), 4),
        "positive_coverage_caveat": "This arm increases negative exposure at fixed total steps by reducing unique positive coverage.",
        "requested_sched_neg_frac": args.sched_neg_frac,
        "sched_neg_frac": round(target_neg / max(target, 1), 4),
        "claim_boundary": "PV-motivated hard negatives only; do not claim real PV deployment validation without PV images.",
    }
    atomic_write_text(audit_path, json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[sched] wrote {sched_yaml} count={target} negatives={target_neg}")


def selected_arms(args):
    return ARM_GROUPS[args.arm]


def arm_data(args, arm):
    if arm == "baseline":
        return args.baseline_data
    if arm == "baseline_eqstep":
        return args.eqstep_data
    if arm == "hardneg":
        return args.hardneg_data
    if arm == "hardneg_sched":
        return args.sched_data
    raise ValueError(arm)


def run_id(args):
    return f"{args.dataset}_{args.family}_seed{args.seed}"


def run_dir(args):
    return resolve(args.run_root) / run_id(args)


def out_dir(args):
    return resolve(args.out_root) / run_id(args)


def pred_path(args, arm):
    return out_dir(args) / f"{arm}.csv"


def pred_meta_path(args, arm):
    return out_dir(args) / f"{arm}.pred_meta.json"


def result_path(args):
    return out_dir(args) / "gonogo.json"


def metadata_path(args):
    return out_dir(args) / "run_meta.json"


def arm_weight(args, arm):
    weights_dir = run_dir(args) / arm / "weights"
    last = weights_dir / "last.pt"
    best = weights_dir / "best.pt"
    if not args.val and last.exists():
        return last
    if best.exists():
        return best
    return last


def read_labels(labels_csv):
    labels = {}
    with Path(labels_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["image"].strip()] = row["label"].strip().lower()
    if not labels:
        raise SystemExit(f"[error] labels empty: {labels_csv}")
    return labels


def norm_path(path):
    return str(Path(path).resolve()).lower()


def check_image_paths(images, label, full=False):
    missing = []
    for idx, image in enumerate(images, start=1):
        if not Path(image).exists():
            missing.append(image)
            if len(missing) >= 5:
                break
        if not full and idx >= 10:
            break
    if missing:
        raise SystemExit(f"[error] {label} images missing:\n" + "\n".join(missing))
    print(f"[smoke] {label} image paths OK")


def file_md5(path):
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_eval_split(full, calib, test, pos_labels, neg_labels):
    if set(calib) & set(test):
        raise SystemExit("[error] calibration/test overlap")
    if set(full) != (set(calib) | set(test)):
        raise SystemExit("[error] calibration/test do not cover full eval labels")
    pos = set(pos_labels)
    neg = set(neg_labels)
    for name, split in (("calib", calib), ("test", test)):
        labels = set(split.values())
        if not (labels & pos) or not (labels & neg):
            raise SystemExit(f"[error] {name} split must include positive and negative labels")


def ensure_eval_protocol(args):
    full = read_labels(args.labels)
    calib = read_labels(args.eval_calib_labels)
    test = read_labels(args.eval_test_labels)
    validate_eval_split(full, calib, test, args.pos_labels, args.neg_labels)


def check_train_eval_overlap(args):
    eval_set = {norm_path(image) for image in read_labels(args.labels)}
    for arm in selected_arms(args):
        overlap = {norm_path(image) for image in train_images(arm_data(args, arm))} & eval_set
        if overlap:
            raise SystemExit(f"[error] {args.dataset}/{arm} train overlaps eval pool: {sorted(overlap)[:5]}")


def check_inputs(args):
    if "baseline_eqstep" in selected_arms(args):
        ensure_equalstep_dataset(args)
    if "hardneg_sched" in selected_arms(args):
        ensure_sched_dataset(args)
    required = [args.labels, args.eval_calib_labels, args.eval_test_labels]
    if not is_downloadable_weight(args.weights):
        required.append(args.weights)
    required.extend(arm_data(args, arm) for arm in selected_arms(args))
    missing = [str(resolve(p)) for p in required if not resolve(p).exists()]
    if missing:
        raise SystemExit("[error] missing inputs:\n" + "\n".join(missing))
    if args.conf > MIN_THRESHOLD:
        raise SystemExit(f"[error] --conf={args.conf} exceeds formal export floor {MIN_THRESHOLD}")
    ensure_eval_protocol(args)
    check_train_eval_overlap(args)


def max_conf(dets, image):
    vals = dets.get(image, [])
    return max(vals) if vals else 0.0


def read_preds(path):
    dets = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dets[row["image"].strip()].append(float(row["conf"]))
    return dets


def evaluate(args, calib_labels, test_labels, dets):
    pos = set(args.pos_labels)
    neg = set(args.neg_labels)
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


def load_model_class(args):
    need_ultralytics()
    patch_ultralytics_polars()
    backend = family_defaults(args.family)["backend"]
    if backend == "yolo":
        from ultralytics import YOLO

        return YOLO
    from ultralytics import RTDETR

    return RTDETR


def device_arg(args):
    if args.device:
        return args.device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train_arm(args, arm):
    save_dir = run_dir(args) / arm
    if save_dir.exists() and not args.overwrite:
        raise SystemExit(f"[error] train dir exists: {save_dir}")
    model = load_model_class(args)(model_weight_arg(args.weights))
    model.train(
        data=str(resolve(arm_data(args, arm))),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_arg(args),
        project=str(run_dir(args)),
        name=arm,
        exist_ok=args.overwrite,
        workers=args.workers,
        plots=False,
        val=args.val,
        seed=args.seed,
        deterministic=args.deterministic,
        amp=args.amp,
    )


def export_arm(args, arm):
    weight = arm_weight(args, arm)
    if not weight.exists():
        raise SystemExit(f"[error] missing weight: {weight}")
    images = list(read_labels(args.labels))
    if args.limit:
        images = images[: args.limit]
    model = load_model_class(args)(str(weight))
    out = pred_path(args, arm)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for idx, image in enumerate(images, start=1):
            result = model.predict(image, conf=args.conf, max_det=args.max_det, imgsz=args.imgsz, device=device_arg(args), verbose=False)[0]
            if result.boxes is not None and len(result.boxes):
                for conf in result.boxes.conf.detach().cpu().tolist():
                    writer.writerow([image, float(conf)])
                    rows += 1
            if idx % 200 == 0 or idx == len(images):
                print(f"[export] {arm} {idx}/{len(images)} rows={rows}")
    meta = {
        "dataset": args.dataset,
        "family": args.family,
        "arm": arm,
        "seed": args.seed,
        "pred": str(out),
        "weight": str(weight),
        "export_conf": args.conf,
        "max_det": args.max_det,
        "imgsz": args.imgsz,
        "n_images": len(images),
        "n_rows": rows,
    }
    pred_meta_path(args, arm).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def arm_train_counts(args):
    return {arm: len(train_images(arm_data(args, arm))) for arm in selected_arms(args)}


def exported_weights(args):
    weights = {}
    for arm in selected_arms(args):
        weight = arm_weight(args, arm)
        weights[arm] = str(weight) if weight.exists() else ""
    return weights


def eval_split_metadata(args):
    return {
        "eval_protocol": "calibration_threshold_test_report",
        "labels_md5": file_md5(args.labels),
        "eval_calib_md5": file_md5(args.eval_calib_labels),
        "eval_test_md5": file_md5(args.eval_test_labels),
        "eval_split_salt": args.eval_split_salt,
        "calib_frac": args.calib_frac,
    }


def eval_outputs(args):
    calib = read_labels(args.eval_calib_labels)
    test = read_labels(args.eval_test_labels)
    split_meta = eval_split_metadata(args)
    out = {}
    for arm in selected_arms(args):
        pred = pred_path(args, arm)
        if not pred.exists():
            continue
        rep = evaluate(args, calib, test, read_preds(pred))
        out[ARM_RESULT_KEYS[arm]] = {
            "dataset": args.dataset,
            "family": args.family,
            "arm": arm,
            "seed": args.seed,
            "pred": str(pred),
            "weight": str(arm_weight(args, arm)),
            "runner_mode": args.mode,
            "epochs": args.epochs if args.mode != "eval" else None,
            "val": args.val,
            "export_conf": args.conf,
            "pred_meta": str(pred_meta_path(args, arm)),
            "arm_train_counts": arm_train_counts(args),
            "standard_metrics": (
                json.loads(val_path(args, arm).read_text(encoding="utf-8"))
                if val_path(args, arm).exists()
                else None
            ),
            **split_meta,
            **rep,
        }
    if not out:
        raise SystemExit(f"[error] no predictions to evaluate under {out_dir(args)}")
    out_dir(args).mkdir(parents=True, exist_ok=True)
    result_path(args).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {result_path(args)}")


def write_metadata(args):
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.executable,
        "dataset": args.dataset,
        "family": args.family,
        "family_label": family_defaults(args.family)["label"],
        "seed": args.seed,
        "mode": args.mode,
        "arms": selected_arms(args),
        "weights": model_weight_arg(args.weights),
        "exported_weights": exported_weights(args),
        "baseline_data": str(resolve(args.baseline_data)),
        "eqstep_data": str(resolve(args.eqstep_data)),
        "hardneg_data": str(resolve(args.hardneg_data)),
        "sched_data": str(resolve(args.sched_data)),
        "arm_train_counts": arm_train_counts(args),
        "labels": str(resolve(args.labels)),
        "eval_calib_labels": str(resolve(args.eval_calib_labels)),
        "eval_test_labels": str(resolve(args.eval_test_labels)),
        **eval_split_metadata(args),
        "run_dir": str(run_dir(args)),
        "out_dir": str(out_dir(args)),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "conf": args.conf,
        "export_conf": args.conf,
        "max_det": args.max_det,
        "val": args.val,
        "deterministic": args.deterministic,
        "amp": args.amp,
    }
    out_dir(args).mkdir(parents=True, exist_ok=True)
    metadata_path(args).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[meta] wrote {metadata_path(args)}")


def smoke(args):
    labels = read_labels(args.labels)
    counts = defaultdict(int)
    for label in labels.values():
        counts[label] += 1
    print(f"[smoke] dataset={args.dataset} family={args.family} labels={len(labels)} counts={dict(counts)}")
    check_image_paths(labels, "eval", full=True)
    for arm in selected_arms(args):
        images = train_images(arm_data(args, arm))
        check_image_paths(images, f"{arm} train", full=False)
        print(f"[smoke] {arm} train_count={len(images)}")
    need_ultralytics()


def parse_args():
    ap = argparse.ArgumentParser(description="Stage5-PV v2 runner")
    ap.add_argument("--mode", choices=["smoke", "train", "export", "metrics", "eval", "all"], default="all")
    ap.add_argument(
        "--metrics",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="训练后在 test 集上计算 mAP/precision/recall 等标准指标；设为 false 可跳过（如快速 mincheck）",
    )
    ap.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="dfire")
    ap.add_argument("--family", choices=["yolo26n", "rtdetr"], required=True)
    ap.add_argument("--arm", choices=list(ARM_GROUPS), default="all4")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--baseline-data", default="")
    ap.add_argument("--eqstep-data", default="")
    ap.add_argument("--hardneg-data", default="")
    ap.add_argument("--sched-data", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--eval-calib-labels", default="")
    ap.add_argument("--eval-test-labels", default="")
    ap.add_argument("--pos-labels", default="")
    ap.add_argument("--neg-labels", default="")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--eval-split-salt", default="stage5_pv_v2_eval")
    ap.add_argument("--run-root", default="runs/detect/runs_stage5_pv_v2")
    ap.add_argument("--out-root", default="formal_results/stage5_pv_v2")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--conf", type=float, default=MIN_THRESHOLD)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sched-neg-frac", type=float, default=DEFAULT_SCHED_NEG_FRAC)
    ap.add_argument("--val", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=False)
    ap.add_argument("--deterministic", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True)
    ap.add_argument("--amp", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, value in defaults.items():
        if getattr(args, key, "") == "":
            setattr(args, key, value)
    fam = family_defaults(args.family)
    if not args.weights:
        args.weights = fam["weights"]
    if args.epochs <= 0:
        args.epochs = fam["epochs"]
    if args.batch <= 0:
        args.batch = fam["batch"]
    args.pos_labels = [x.strip().lower() for x in args.pos_labels.split(",") if x.strip()]
    args.neg_labels = [x.strip().lower() for x in args.neg_labels.split(",") if x.strip()]
    if not 0.0 < args.sched_neg_frac < 1.0:
        raise SystemExit("[error] --sched-neg-frac must be between 0 and 1")
    return args


def val_path(args, arm):
    return out_dir(args) / f"{arm}.val.json"


def val_metrics(args, arm):
    """训练后在 test 集上计算标准检测指标：mAP50/mAP50-95/mAP75、precision、recall、F1、各类 AP。

    这一步独立于固定召回 FPR/FPPI 协议：在 data yaml 的 val 集上跑一次标准
    Ultralytics 验证，不改变训练过程，也不参与阈值校准。
    """
    data_yaml = str(resolve(arm_data(args, arm)))
    weight = arm_weight(args, arm)
    if not weight.exists():
        raise SystemExit(f"[error] missing weight, cannot compute standard metrics: {weight}")
    model_cls = load_model_class(args)
    dev = device_arg(args)
    res = model_cls(str(weight)).val(
        data=data_yaml,
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        workers=args.workers,
        conf=0.001,
        iou=0.7,
        plots=True,
        save_json=False,
        verbose=False,
        project=str(run_dir(args)),
        name=f"{arm}_val",
        exist_ok=True,
    )
    box = res.box
    names = getattr(res, "names", {}) or {}

    def _round(value):
        try:
            return round(float(value), 4)
        except Exception:
            return None

    per_class = {}
    try:
        for i, cls in enumerate(list(box.ap_class_index)):
            p = float(box.p[i])
            r = float(box.r[i])
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            per_class[str(names.get(int(cls), int(cls)))] = {
                "AP50": _round(box.ap50[i]),
                "AP50_95": _round(box.ap[i]),
                "precision": _round(p),
                "recall": _round(r),
                "f1": _round(f1),
            }
    except Exception as exc:
        per_class = {"_warn": f"per-class metrics unavailable: {exc}"}

    mp = float(box.mp)
    mr = float(box.mr)
    mean_f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
    speed = getattr(res, "speed", {}) or {}
    return {
        "dataset": args.dataset,
        "val_set": data_yaml,
        "weight": str(weight),
        "split": "test",
        "mAP50": _round(box.map50),
        "mAP50_95": _round(box.map),
        "mAP75": _round(box.map75),
        "precision": _round(mp),
        "recall": _round(mr),
        "f1": _round(mean_f1),
        "fitness": _round(getattr(res, "fitness", None)),
        "per_class": per_class,
        "speed_ms_per_image": {key: _round(val) for key, val in speed.items()},
    }


def run_val(args):
    out_dir(args).mkdir(parents=True, exist_ok=True)
    for arm in selected_arms(args):
        try:
            metrics = val_metrics(args, arm)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[warn] standard metrics failed arm={arm}: {exc}")
            continue
        val_path(args, arm).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[metrics] {arm} mAP50={metrics['mAP50']} mAP50-95={metrics['mAP50_95']} "
            f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}"
        )


def main():
    args = parse_args()
    check_inputs(args)
    if args.mode == "smoke":
        smoke(args)
        return
    arms = selected_arms(args)
    if args.mode in ("train", "all"):
        for arm in arms:
            train_arm(args, arm)
    if args.mode in ("export", "all"):
        for arm in arms:
            export_arm(args, arm)
    if args.mode in ("metrics", "all") and args.metrics:
        run_val(args)
    if args.mode in ("eval", "all"):
        eval_outputs(args)
    write_metadata(args)


if __name__ == "__main__":
    main()
