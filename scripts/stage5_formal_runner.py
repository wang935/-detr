#!/usr/bin/env python3
"""Stage 5 正式重复实验 runner。

目标：在 D-Fire eval pool 上导出预测，并用 calibration/test 分离的
固定召回 FPR/FPPI 复查“硬负样本训练降低误报”是否稳定。
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (0.80, 0.85, 0.90, 0.95)
MIN_THRESHOLD = 1e-3
ARM_ORDER = ("baseline", "baseline_eqstep", "hardneg")
ARM_RESULT_KEYS = {
    "baseline": "model_A",
    "hardneg": "model_B",
    "baseline_eqstep": "model_C",
}
ARM_GROUPS = {
    "baseline": ["baseline"],
    "baseline_eqstep": ["baseline_eqstep"],
    "hardneg": ["hardneg"],
    "both": ["baseline", "hardneg"],
    "main": ["baseline", "hardneg"],
    "all3": list(ARM_ORDER),
}


def need_ultralytics():
    try:
        import torch  # noqa
        import ultralytics  # noqa
    except Exception as exc:
        raise SystemExit(f"[错误] 缺少 torch/ultralytics：{exc}")


def patch_ultralytics_polars():
    """规避当前主机上 checkpoint 保存时可能触发的 polars 读取问题。"""
    try:
        from ultralytics.engine.trainer import BaseTrainer
    except Exception as exc:
        raise SystemExit(f"[错误] 无法 patch Ultralytics trainer：{exc}")

    def read_results_csv_no_polars(self):
        path = Path(self.csv)
        if not path.exists():
            return {}
        try:
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        except Exception:
            return {}
        keys = rows[0].keys() if rows else []
        return {key: [row.get(key, "") for row in rows] for key in keys}

    BaseTrainer.read_results_csv = read_results_csv_no_polars


def resolve(path):
    return Path(path).resolve()


def family_defaults(family):
    if family == "yolo":
        return {"weights": "yolo26n.pt", "batch": 16, "epochs": 300, "label": "YOLO26n"}
    if family == "rtdetr":
        return {"weights": "rtdetr-l.pt", "batch": 4, "epochs": 300, "label": "RT-DETR-L"}
    raise ValueError(family)


def arm_data(args, arm):
    if arm == "baseline":
        return args.baseline_data
    if arm == "baseline_eqstep":
        return args.eqstep_data
    if arm == "hardneg":
        return args.hardneg_data
    raise ValueError(arm)


def arm_weight(args, arm):
    weights_dir = run_dir(args) / arm / "weights"
    last = weights_dir / "last.pt"
    if not args.val and last.exists():
        return last
    best = weights_dir / "best.pt"
    if best.exists():
        return best
    if last.exists():
        print(f"[warn] best.pt 不存在，使用 last.pt：{last}")
        return last
    return best


def run_id(args):
    return f"{args.family}_seed{args.seed}"


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


def check_inputs(args):
    if "baseline_eqstep" in selected_arms(args):
        ensure_equalstep_dataset(args)
    required = [
        args.labels,
        args.baseline_data,
        args.hardneg_data,
        args.weights,
    ]
    if "baseline_eqstep" in selected_arms(args):
        required.append(args.eqstep_data)
    missing = [str(resolve(p)) for p in required if not resolve(p).exists()]
    if missing:
        raise SystemExit("[错误] 缺少输入文件：\n" + "\n".join(missing))
    if not args.legacy_in_sample_eval and args.conf > MIN_THRESHOLD:
        raise SystemExit(
            f"[error] --conf={args.conf} is higher than formal min threshold {MIN_THRESHOLD}; "
            f"use --conf {MIN_THRESHOLD} or lower for Stage 5."
        )
    check_val_leakage(args)
    ensure_eval_protocol(args)
    check_train_eval_overlap(args)


def yaml_value(path, key):
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict) and key in data:
            return str(data[key])
    except Exception:
        pass

    prefix = f"{key}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip().strip("\"'")
    raise SystemExit(f"[错误] {path} 缺少 {key}: 字段")


def check_image_paths(images, label, full=False):
    missing = []
    total = 0
    for image in images:
        total += 1
        if not Path(image).exists():
            missing.append(image)
            if len(missing) >= 5:
                break
        if not full and total >= 10:
            break
    if missing:
        raise SystemExit(f"[错误] {label} 图片不可访问，示例：\n" + "\n".join(missing))
    scope = "全部" if full else "抽样"
    print(f"[smoke] {label} {scope}图片路径可访问")


def check_train_list(data_yaml, label):
    train_txt = Path(yaml_value(data_yaml, "train"))
    if not train_txt.exists():
        raise SystemExit(f"[错误] {label} train 列表不存在：{train_txt}")
    images = [line.strip() for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not images:
        raise SystemExit(f"[错误] {label} train 列表为空：{train_txt}")
    check_image_paths(images, f"{label} train", full=False)
    print(f"[smoke] {label} train 图片数={len(images)}")


def train_images(data_yaml):
    train_txt = Path(yaml_value(data_yaml, "train"))
    if not train_txt.exists():
        raise SystemExit(f"[错误] train 列表不存在：{train_txt}")
    return [line.strip() for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_image_path(path):
    return str(Path(path).resolve()).lower()


def val_images(data_yaml):
    val_value = yaml_value(data_yaml, "val")
    val_path = Path(val_value)
    if not val_path.exists():
        return []
    return [line.strip() for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_val_leakage(args):
    if not args.val:
        return
    eval_images = set(read_labels(resolve(args.labels)))
    for arm in selected_arms(args):
        vals = set(val_images(resolve(arm_data(args, arm))))
        overlap = eval_images & vals
        if overlap:
            sample = "\n".join(sorted(overlap)[:5])
            raise SystemExit(
                f"[错误] --val true 会让 {arm} 的 val 列表与 eval pool 重叠，示例：\n{sample}\n"
                "正式实验必须使用 --val false，或改成训练内部 validation split。"
            )


def check_train_eval_overlap(args):
    eval_images = {norm_image_path(image) for image in read_labels(resolve(args.labels))}
    for arm in selected_arms(args):
        train_set = {norm_image_path(image) for image in train_images(resolve(arm_data(args, arm)))}
        overlap = train_set & eval_images
        if overlap:
            sample = "\n".join(sorted(overlap)[:5])
            raise SystemExit(
                f"[error] {arm} train list overlaps eval pool; formal Stage 5 would leak. Examples:\n{sample}"
            )


def ensure_equalstep_dataset(args):
    """Create a deterministic positive-only list with the same length as hardneg.

    This controls for optimizer-step count: baseline_eqstep still contains only
    positive training images, but each epoch has the same number of batches as
    the hard-negative arm at the same batch size.
    """
    baseline_yaml = resolve(args.baseline_data)
    hardneg_yaml = resolve(args.hardneg_data)
    eqstep_yaml = resolve(args.eqstep_data)
    eqstep_train = eqstep_yaml.with_name("train_posonly_equalstep.txt")

    baseline_images = train_images(baseline_yaml)
    hardneg_images = train_images(hardneg_yaml)
    if not baseline_images or not hardneg_images:
        raise SystemExit("[错误] baseline/hardneg train 列表不能为空")
    target_count = len(hardneg_images)
    rng = np.random.default_rng(20260602)
    shuffled = list(baseline_images)
    rng.shuffle(shuffled)
    repeated = []
    while len(repeated) < target_count:
        repeated.extend(shuffled)
    repeated = repeated[:target_count]

    train_text = "\n".join(repeated) + "\n"
    if not eqstep_train.exists() or eqstep_train.read_text(encoding="utf-8") != train_text:
        eqstep_train.write_text(train_text, encoding="utf-8")

    base_path = yaml_value(baseline_yaml, "path")
    val_path = yaml_value(baseline_yaml, "val")
    yaml_text = (
        f"path: {base_path}\n"
        f"train: {eqstep_train}\n"
        f"val: {val_path}\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )
    if not eqstep_yaml.exists() or eqstep_yaml.read_text(encoding="utf-8") != yaml_text:
        eqstep_yaml.write_text(yaml_text, encoding="utf-8")
    print(
        f"[eqstep] baseline_unique={len(baseline_images)} "
        f"hardneg_steps_source={target_count} eqstep_train={eqstep_train}"
    )


def read_labels(labels_csv):
    labels = {}
    with Path(labels_csv).open("r", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=2):
            image = row["image"].strip()
            if image in labels:
                raise SystemExit(f"[错误] {labels_csv}:{idx} 重复 image：{image}")
            labels[image] = row["label"].strip().lower()
    if not labels:
        raise SystemExit(f"[错误] labels 为空：{labels_csv}")
    return labels


def file_md5(path):
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_labels(path, labels, images):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for image in images:
            writer.writerow([image, labels[image]])


def stratified_calib_test_split(labels, frac, salt):
    by_label = defaultdict(list)
    for image, label in labels.items():
        by_label[label].append(image)
    calib = []
    test = []
    for label, images in sorted(by_label.items()):
        ordered = sorted(
            images,
            key=lambda image: hashlib.md5((str(salt) + "|" + label + "|" + image).encode("utf-8")).hexdigest(),
        )
        if len(ordered) == 1:
            test.extend(ordered)
            continue
        n_calib = int(round(len(ordered) * frac))
        n_calib = min(max(n_calib, 1), len(ordered) - 1)
        calib.extend(ordered[:n_calib])
        test.extend(ordered[n_calib:])
    return sorted(calib), sorted(test)


def validate_eval_split(full, calib, test, calib_path, test_path):
    if not calib or not test:
        raise SystemExit("[错误] calibration/test labels 不能为空")
    overlap = set(calib) & set(test)
    if overlap:
        sample = "\n".join(sorted(overlap)[:5])
        raise SystemExit(f"[错误] calibration/test split 重叠，示例：\n{sample}")
    missing_from_splits = set(full) - (set(calib) | set(test))
    extra_in_splits = (set(calib) | set(test)) - set(full)
    if missing_from_splits or extra_in_splits:
        raise SystemExit(
            "[错误] calibration/test split 必须完整覆盖 full eval pool："
            f"missing={len(missing_from_splits)} extra={len(extra_in_splits)}"
        )
    for split_name, split in (("calibration", calib), ("test", test)):
        missing = [image for image in split if image not in full]
        if missing:
            raise SystemExit(f"[错误] {split_name} labels 含 full labels 外图片：{missing[:3]}")
        mismatch = [image for image, label in split.items() if full[image] != label]
        if mismatch:
            raise SystemExit(f"[错误] {split_name} labels 与 full labels 标签不一致：{mismatch[:3]}")
    for label in ("fire", "smoke"):
        if label not in set(calib.values()) or label not in set(test.values()):
            raise SystemExit(
                f"[错误] calibration/test split 缺少 {label} 类；"
                f"请检查 {calib_path} 和 {test_path}"
            )
    neg_labels = {"none", "distractor"}
    if not (neg_labels & set(calib.values())) or not (neg_labels & set(test.values())):
        raise SystemExit(
            "[错误] calibration/test split 必须在两侧都包含 none 或 distractor 负样本；"
            f"请检查 {calib_path} 和 {test_path}"
        )


def ensure_eval_protocol(args):
    if args.legacy_in_sample_eval:
        return
    full = read_labels(resolve(args.labels))
    calib_path = resolve(args.eval_calib_labels)
    test_path = resolve(args.eval_test_labels)
    calib_exists = calib_path.exists()
    test_exists = test_path.exists()
    if calib_exists != test_exists:
        raise SystemExit(f"[错误] calibration/test labels 必须成对存在：{calib_path}, {test_path}")
    if not calib_exists and not test_exists:
        if args.calib_frac <= 0.05 or args.calib_frac >= 0.95:
            raise SystemExit("[错误] --calib-frac 必须在 0.05 到 0.95 之间")
        calib_imgs, test_imgs = stratified_calib_test_split(full, args.calib_frac, args.eval_split_salt)
        write_labels(calib_path, full, calib_imgs)
        write_labels(test_path, full, test_imgs)
        print(
            f"[eval-split] wrote calibration={calib_path} ({len(calib_imgs)}) "
            f"test={test_path} ({len(test_imgs)})"
        )
    calib = read_labels(calib_path)
    test = read_labels(test_path)
    validate_eval_split(full, calib, test, calib_path, test_path)


def read_preds(pred_csv):
    dets = defaultdict(list)
    path = Path(pred_csv)
    if not path.exists():
        raise SystemExit(f"[错误] 预测文件不存在：{path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "conf"}.issubset(reader.fieldnames):
            raise SystemExit(f"[错误] {path} 必须包含 image,conf 列")
        for idx, row in enumerate(reader, start=2):
            image = row["image"].strip()
            try:
                conf = float(row["conf"])
            except Exception as exc:
                raise SystemExit(f"[错误] {path}:{idx} conf 非数字：{row.get('conf')!r}") from exc
            if not np.isfinite(conf) or conf < 0.0 or conf > 1.0:
                raise SystemExit(f"[错误] {path}:{idx} conf 越界：{conf}")
            dets[image].append(conf)
    return dets


def max_conf(dets, image):
    vals = dets.get(image, [])
    return max(vals) if vals else 0.0


def eval_split_metadata(args):
    if args.legacy_in_sample_eval:
        labels_path = resolve(args.labels)
        return {
            "eval_protocol": "legacy_in_sample",
            "labels_md5": file_md5(labels_path),
            "eval_calib_md5": "",
            "eval_test_md5": "",
            "eval_split_salt": "",
            "calib_frac": None,
        }
    labels_path = resolve(args.labels)
    calib_path = resolve(args.eval_calib_labels)
    test_path = resolve(args.eval_test_labels)
    return {
        "eval_protocol": "calibration_threshold_test_report",
        "labels_md5": file_md5(labels_path),
        "eval_calib_md5": file_md5(calib_path),
        "eval_test_md5": file_md5(test_path),
        "eval_split_salt": args.eval_split_salt,
        "calib_frac": args.calib_frac,
    }


def evaluate(calib_labels, test_labels, dets):
    calib_pos = [image for image, label in calib_labels.items() if label in ("fire", "smoke")]
    calib_neg = [image for image, label in calib_labels.items() if label in ("none", "distractor")]
    test_pos = [image for image, label in test_labels.items() if label in ("fire", "smoke")]
    test_neg = [image for image, label in test_labels.items() if label in ("none", "distractor")]
    observed = {max_conf(dets, image) for image in calib_pos if max_conf(dets, image) > MIN_THRESHOLD}
    grid = {float(x) for x in np.linspace(0, 1, 201) if x > MIN_THRESHOLD}
    thresholds = sorted(observed | grid, reverse=True)
    out = {
        "n_pos": len(test_pos),
        "n_neg": len(test_neg),
        "n_calib_pos": len(calib_pos),
        "n_calib_neg": len(calib_neg),
        "eval_protocol": "calibration_threshold_test_report",
        "at_fixed_recall": {},
    }
    if not calib_pos or not calib_neg or not test_pos or not test_neg:
        raise SystemExit(
            "[错误] calibration/test split 异常："
            f"calib_pos={len(calib_pos)} calib_neg={len(calib_neg)} "
            f"test_pos={len(test_pos)} test_neg={len(test_neg)}"
        )
    for target in TARGETS:
        chosen = None
        for thr in thresholds:
            calib_recall = float(np.mean([max_conf(dets, image) >= thr for image in calib_pos]))
            if calib_recall >= target:
                test_recall = float(np.mean([max_conf(dets, image) >= thr for image in test_pos]))
                fpr = float(np.mean([max_conf(dets, image) >= thr for image in test_neg]))
                fppi = float(np.mean([sum(1 for c in dets.get(image, []) if c >= thr) for image in test_neg]))
                chosen = {
                    "thr": round(float(thr), 4),
                    "calib_recall": round(calib_recall, 4),
                    "recall": round(test_recall, 4),
                    "FPR": round(fpr, 4),
                    "FPPI": round(fppi, 4),
                }
                break
        out["at_fixed_recall"][f"{target:.2f}"] = chosen
    return out


def load_model_class(family):
    need_ultralytics()
    patch_ultralytics_polars()
    if family == "yolo":
        from ultralytics import YOLO

        return YOLO
    if family == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR
    raise SystemExit(f"[错误] unknown family: {family}")


def device_arg(args):
    if args.device != "":
        return args.device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train_arm(args, arm):
    import torch

    save_dir = run_dir(args) / arm
    if save_dir.exists() and not args.overwrite:
        raise SystemExit(f"[错误] 训练目录已存在，避免覆盖：{save_dir}\n如确认重跑，添加 --overwrite。")

    model_cls = load_model_class(args.family)
    dev = device_arg(args)
    gpu = torch.cuda.get_device_name(0) if dev != "cpu" and torch.cuda.is_available() else "CPU"
    print(
        f"[train] family={args.family} arm={arm} seed={args.seed} "
        f"device={dev}({gpu}) epochs={args.epochs} batch={args.batch}"
    )
    model = model_cls(str(resolve(args.weights)))
    model.train(
        data=str(resolve(arm_data(args, arm))),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
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
    last = save_dir / "weights" / "last.pt"
    best = save_dir / "weights" / "best.pt"
    if not args.val and last.exists():
        return last
    if best.exists():
        return best
    if last.exists():
        print(f"[warn] best.pt 不存在，使用 last.pt：{last}")
        return last
    raise SystemExit(f"[错误] 训练完成但没有权重：{save_dir / 'weights'}")


def export_arm(args, arm):
    import torch

    weights = arm_weight(args, arm)
    if not weights.exists():
        raise SystemExit(f"[错误] 缺少权重，无法导出：{weights}")
    out = pred_path(args, arm)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"[错误] 预测文件已存在，避免覆盖：{out}\n如确认重跑，添加 --overwrite。")
    labels = read_labels(resolve(args.labels))
    images = list(labels.keys())
    if args.limit:
        images = images[: args.limit]
    model_cls = load_model_class(args.family)
    dev = device_arg(args)
    gpu = torch.cuda.get_device_name(0) if dev != "cpu" and torch.cuda.is_available() else "CPU"
    print(f"[export] arm={arm} weights={weights} images={len(images)} device={dev}({gpu}) out={out}")
    model = model_cls(str(weights))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for idx, image in enumerate(images, start=1):
            result = model.predict(
                image,
                conf=args.conf,
                max_det=args.max_det,
                imgsz=args.imgsz,
                device=dev,
                verbose=False,
            )[0]
            if result.boxes is not None and len(result.boxes):
                for conf in result.boxes.conf.detach().cpu().tolist():
                    writer.writerow([image, float(conf)])
                    rows += 1
            if idx % 200 == 0 or idx == len(images):
                print(f"  ...{idx}/{len(images)} rows={rows}")
    print(f"[export] wrote {out} rows={rows}")


    pred_meta = {
        "family": args.family,
        "arm": arm,
        "seed": args.seed,
        "pred": str(out),
        "weight": str(weights),
        "export_conf": args.conf,
        "max_det": args.max_det,
        "imgsz": args.imgsz,
        "labels": str(resolve(args.labels)),
        "n_images": len(images),
        "n_rows": rows,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    pred_meta_path(args, arm).write_text(json.dumps(pred_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] wrote {pred_meta_path(args, arm)}")


def load_pred_meta(args, arm):
    path = pred_meta_path(args, arm)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"[error] cannot read prediction metadata: {path} ({exc})")


def arm_train_counts(args):
    counts = {}
    for arm in selected_arms(args):
        try:
            counts[arm] = len(train_images(resolve(arm_data(args, arm))))
        except Exception:
            counts[arm] = None
    return counts


def eval_outputs(args):
    if args.legacy_in_sample_eval:
        labels = read_labels(resolve(args.labels))
        calib_labels = labels
        test_labels = labels
    else:
        calib_labels = read_labels(resolve(args.eval_calib_labels))
        test_labels = read_labels(resolve(args.eval_test_labels))
    split_meta = eval_split_metadata(args)
    out = {}
    for arm in selected_arms(args):
        path = pred_path(args, arm)
        if not path.exists():
            continue
        pred_meta = load_pred_meta(args, arm)
        rep = evaluate(calib_labels, test_labels, read_preds(path))
        if args.legacy_in_sample_eval:
            rep["eval_protocol"] = "legacy_in_sample"
        out[ARM_RESULT_KEYS[arm]] = {
            "family": args.family,
            "arm": arm,
            "seed": args.seed,
            "pred": str(path),
            "weight": str(arm_weight(args, arm)),
            "runner_mode": args.mode,
            "epochs": args.epochs if args.mode != "eval" else None,
            "val": args.val,
            "export_conf": pred_meta.get("export_conf"),
            "pred_meta": str(pred_meta_path(args, arm)) if pred_meta else "",
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
        raise SystemExit(f"[错误] 没有可评估的预测文件：{out_dir(args)}")
    out_dir(args).mkdir(parents=True, exist_ok=True)
    result_path(args).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {result_path(args)}")
    for key, rep in out.items():
        row = rep["at_fixed_recall"]["0.90"]
        if not row:
            print(f"[eval] {key} R0.90 unreachable on calibration split")
            continue
        print(
            f"[eval] {key} R0.90 thr={row['thr']} "
            f"calib_recall={row.get('calib_recall', row['recall'])} "
            f"test_recall={row['recall']} FPR={row['FPR']} FPPI={row['FPPI']}"
        )
    return out


def write_metadata(args):
    meta_path = metadata_path(args)
    exported_weights = {}
    for arm in selected_arms(args):
        try:
            weight = arm_weight(args, arm)
            exported_weights[arm] = str(weight) if weight.exists() else ""
        except Exception:
            exported_weights[arm] = ""
    if args.mode == "eval" and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta.update(
            {
                "last_eval_at": datetime.now().isoformat(timespec="seconds"),
                "labels": str(resolve(args.labels)),
                "eval_calib_labels": str(resolve(args.eval_calib_labels)) if not args.legacy_in_sample_eval else "",
                "eval_test_labels": str(resolve(args.eval_test_labels)) if not args.legacy_in_sample_eval else "",
                "exported_weights": exported_weights,
                "conf": args.conf,
                "export_conf": args.conf,
                **eval_split_metadata(args),
            }
        )
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[meta] updated eval fields in {meta_path}")
        return

    if args.mode == "eval":
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "python": sys.executable,
            "family": args.family,
            "family_label": family_defaults(args.family)["label"],
            "seed": args.seed,
            "mode": args.mode,
            "arms": selected_arms(args),
            "training_metadata": "unknown_eval_only",
            "weights": str(resolve(args.weights)),
            "exported_weights": exported_weights,
            "labels": str(resolve(args.labels)),
            "eval_calib_labels": str(resolve(args.eval_calib_labels)) if not args.legacy_in_sample_eval else "",
            "eval_test_labels": str(resolve(args.eval_test_labels)) if not args.legacy_in_sample_eval else "",
            **eval_split_metadata(args),
            "run_dir": str(run_dir(args)),
            "out_dir": str(out_dir(args)),
            "epochs": None,
            "batch": None,
            "imgsz": args.imgsz,
            "workers": None,
            "conf": args.conf,
            "export_conf": args.conf,
            "max_det": args.max_det,
            "val": args.val,
            "deterministic": args.deterministic,
            "amp": args.amp,
        }
        out_dir(args).mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[meta] wrote eval-only metadata {meta_path}")
        return

    train_counts = arm_train_counts(args)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.executable,
        "family": args.family,
        "family_label": family_defaults(args.family)["label"],
        "seed": args.seed,
        "mode": args.mode,
        "arms": selected_arms(args),
        "weights": str(resolve(args.weights)),
        "exported_weights": exported_weights,
        "baseline_data": str(resolve(args.baseline_data)),
        "eqstep_data": str(resolve(args.eqstep_data)),
        "hardneg_data": str(resolve(args.hardneg_data)),
        "arm_train_counts": train_counts,
        "labels": str(resolve(args.labels)),
        "eval_calib_labels": str(resolve(args.eval_calib_labels)) if not args.legacy_in_sample_eval else "",
        "eval_test_labels": str(resolve(args.eval_test_labels)) if not args.legacy_in_sample_eval else "",
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
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[meta] wrote {meta_path}")


def selected_arms(args):
    return ARM_GROUPS[args.arm]


def smoke(args):
    labels = read_labels(resolve(args.labels))
    counts = defaultdict(int)
    for label in labels.values():
        counts[label] += 1
    print(f"[smoke] labels={len(labels)} counts={dict(counts)}")
    check_image_paths(labels.keys(), "eval", full=True)
    if not args.legacy_in_sample_eval:
        calib = read_labels(resolve(args.eval_calib_labels))
        test = read_labels(resolve(args.eval_test_labels))
        print(f"[smoke] eval calibration={len(calib)} test={len(test)}")
    check_train_list(resolve(args.baseline_data), "baseline")
    if "baseline_eqstep" in selected_arms(args):
        check_train_list(resolve(args.eqstep_data), "baseline_eqstep")
    check_train_list(resolve(args.hardneg_data), "hardneg")
    print(f"[smoke] family={args.family} seed={args.seed} run_dir={run_dir(args)} out_dir={out_dir(args)}")
    need_ultralytics()
    print("[smoke] torch/ultralytics OK")


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 5 正式重复实验 runner")
    ap.add_argument("--mode", choices=["smoke", "train", "export", "metrics", "eval", "all"], default="all")
    ap.add_argument(
        "--metrics",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="训练后在 test 集上计算 mAP/precision/recall 等标准指标；设为 false 可跳过（如快速 mincheck）",
    )
    ap.add_argument("--family", choices=["yolo", "rtdetr"], required=True)
    ap.add_argument("--arm", choices=list(ARM_GROUPS), default="all3")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--baseline-data", default="data/dfire_local/dfire_posonly.yaml")
    ap.add_argument("--eqstep-data", default="data/dfire_local/dfire_posonly_equalstep.yaml")
    ap.add_argument("--hardneg-data", default="data/dfire_local/dfire_full.yaml")
    ap.add_argument("--labels", default="data/dfire_local/eval_labels.csv")
    ap.add_argument("--eval-calib-labels", default="data/dfire_local/eval_calib_labels.csv")
    ap.add_argument("--eval-test-labels", default="data/dfire_local/eval_test_labels.csv")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--eval-split-salt", default="stage5_eval_v1")
    ap.add_argument("--run-root", default="runs/detect/runs_stage5_formal")
    ap.add_argument("--out-root", default="formal_results/stage5")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="", help="默认自动选择 0/cpu；也可传 0 或 cpu")
    ap.add_argument("--conf", type=float, default=MIN_THRESHOLD)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="只导出前 N 张，用于 smoke export")
    ap.add_argument("--val", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=False)
    ap.add_argument("--deterministic", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True)
    ap.add_argument("--amp", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True)
    ap.add_argument(
        "--legacy-in-sample-eval",
        action="store_true",
        help="兼容旧协议：同一 labels 上选固定召回阈值并报告 FPR/FPPI；正式实验不要使用。",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fam = family_defaults(args.family)
    if not args.weights:
        args.weights = fam["weights"]
    if args.epochs <= 0:
        args.epochs = fam["epochs"]
    if args.batch <= 0:
        args.batch = fam["batch"]
    return args


def val_path(args, arm):
    return out_dir(args) / f"{arm}.val.json"


def val_metrics(args, arm):
    """训练后在 test 集上计算标准检测指标：mAP50/mAP50-95/mAP75、precision、recall、F1、各类 AP。

    这一步独立于固定召回 FPR/FPPI 协议：它在 data yaml 的 val 集（D-Fire test）上
    跑一次标准 Ultralytics 验证，不改变训练过程，也不参与阈值校准。
    """
    data_yaml = str(resolve(arm_data(args, arm)))
    weight = arm_weight(args, arm)
    if not weight.exists():
        raise SystemExit(f"[错误] 缺少权重，无法计算标准指标：{weight}")
    model_cls = load_model_class(args.family)
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
            print(f"[warn] 标准指标计算失败 arm={arm}: {exc}")
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
