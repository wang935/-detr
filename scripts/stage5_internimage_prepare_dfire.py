import argparse
import csv
import json
import math
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [
    {"id": 1, "name": "smoke", "supercategory": "fire_smoke"},
    {"id": 2, "name": "fire", "supercategory": "fire_smoke"},
]

ARM_LISTS = {
    "baseline": "data/dfire_local/train_posonly.txt",
    "baseline_eqstep": "data/stage5_pv_v2/dfire/train_posonly_equalstep.txt",
    "hardneg": "data/dfire_local/train_full.txt",
    "hardneg_sched": "data/stage5_pv_v2/dfire/train_hardneg_sched.txt",
}


def resolve_path(path):
    text = str(path).strip().strip('"')
    text = text.replace("D:\\fire\\D-Fire", str(ROOT / "data" / "D-Fire"))
    p = Path(text)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def read_list(path):
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return [resolve_path(line) for line in f if line.strip()]


def read_label_csv(path):
    out = []
    with resolve_path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(resolve_path(row["image"]))
    return out


def label_path(image):
    text = str(image)
    if "\\images\\" in text:
        text = text.replace("\\images\\", "\\labels\\")
    elif "/images/" in text:
        text = text.replace("/images/", "/labels/")
    else:
        return image.with_suffix(".txt")
    return Path(text).with_suffix(".txt")


def image_size(image):
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image}")
    height, width = img.shape[:2]
    return width, height


def yolo_boxes(image, width, height):
    labels = label_path(image)
    if not labels.exists():
        return []
    boxes = []
    with labels.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = parts[:5]
            cls_id = int(float(cls)) + 1
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            w = max(1.0, bw * width)
            h = max(1.0, bh * height)
            x = min(max(0.0, xc * width - w / 2), max(0.0, width - 1.0))
            y = min(max(0.0, yc * height - h / 2), max(0.0, height - 1.0))
            w = min(w, width - x)
            h = min(h, height - y)
            if w <= 0 or h <= 0 or cls_id not in (1, 2):
                continue
            boxes.append((cls_id, x, y, w, h))
    return boxes


def build_coco(images, out_path, dataset_name, force=False):
    out_path = resolve_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists")
        return
    coco = {
        "info": {"description": dataset_name, "version": "stage5_internimage"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    missing = []
    for image_id, image in enumerate(images, start=1):
        if not image.exists():
            missing.append(str(image))
            continue
        width, height = image_size(image)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(image),
                "width": width,
                "height": height,
            }
        )
        for cls_id, x, y, w, h in yolo_boxes(image, width, height):
            seg = [x, y, x + w, y, x + w, y + h, x, y + h]
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "segmentation": [seg],
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    if missing:
        raise SystemExit("[error] missing images:\n" + "\n".join(missing[:20]))
    out_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    positives = len({ann["image_id"] for ann in coco["annotations"]})
    print(
        f"[coco] {out_path} images={len(coco['images'])} "
        f"positive_images={positives} annotations={len(coco['annotations'])}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/stage5_internimage/dfire/annotations")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = resolve_path(args.out_dir)
    for arm, list_path in ARM_LISTS.items():
        build_coco(read_list(list_path), out_dir / f"train_{arm}.json", f"dfire_{arm}", args.force)
    build_coco(read_label_csv("data/dfire_local/eval_labels.csv"), out_dir / "eval_all.json", "dfire_eval_all", args.force)
    build_coco(
        read_label_csv("data/dfire_local/eval_calib_labels.csv"),
        out_dir / "eval_calib.json",
        "dfire_eval_calib",
        args.force,
    )
    build_coco(
        read_label_csv("data/dfire_local/eval_test_labels.csv"),
        out_dir / "eval_test.json",
        "dfire_eval_test",
        args.force,
    )


if __name__ == "__main__":
    main()
