#!/usr/bin/env python3
"""Prepare DFS-FIRE-SMOKE style VOC data for Stage5-PV v2.

The script is intentionally conservative: it refuses to fabricate a detection
dataset from the local DeepQuest image-classification folders. It expects VOC
XML annotations and converts only fire/smoke boxes to YOLO labels. Images with
only ``other`` objects (or no positive boxes) become negative images.
"""
import argparse
import csv
import hashlib
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POSITIVE_CLASSES = {"fire": 1, "smoke": 0}
NEGATIVE_CLASSES = {"other", "none", "negative", "background"}
DEFAULT_SCHED_NEG_FRAC = 0.60
PV_HARDNEG_TAXONOMY = [
    "solar_glare",
    "cloud_shadow",
    "metal_rack_reflection",
    "dust_soiling",
    "bird_droppings",
    "snow_or_strong_light",
    "cleaning_robot",
]
CLASS_ALIASES = {
    "flame": "fire",
    "fire": "fire",
    "smoke": "smoke",
    "other": "other",
}


def resolve(path):
    return Path(path).resolve()


def norm_class(name):
    key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(key, key)


def find_xmls(source):
    return sorted(p for p in source.rglob("*.xml") if p.is_file())


def find_image(source, xml_path, filename):
    candidates = []
    if filename:
        candidates.append(xml_path.parent / filename)
        candidates.extend(source.rglob(filename))
    stem = xml_path.stem
    for ext in IMG_EXTS:
        candidates.append(xml_path.with_suffix(ext))
        candidates.extend(source.rglob(stem + ext))
    for path in candidates:
        if path.exists() and path.suffix.lower() in IMG_EXTS:
            return path
    return None


def split_name(path):
    parts = {p.lower() for p in path.parts}
    if "test" in parts:
        return "test"
    if "val" in parts or "valid" in parts:
        return "test"
    if "train" in parts:
        return "train"
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()
    return "test" if int(h[:8], 16) % 5 == 0 else "train"


def parse_voc(xml_path, source):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", default="0"))) if size is not None else 0
    height = int(float(size.findtext("height", default="0"))) if size is not None else 0
    filename = root.findtext("filename", default="")
    image_path = find_image(source, xml_path, filename)
    if image_path is None:
        raise SystemExit(f"[error] image not found for VOC file: {xml_path}")
    if width <= 0 or height <= 0:
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as exc:
            raise SystemExit(f"[error] invalid size in {xml_path} and cannot read image: {exc}")

    boxes = []
    raw_classes = []
    for obj in root.findall("object"):
        cls = norm_class(obj.findtext("name", default=""))
        raw_classes.append(cls)
        b = obj.find("bndbox")
        if cls not in POSITIVE_CLASSES or b is None:
            continue
        xmin = float(b.findtext("xmin", default="0"))
        ymin = float(b.findtext("ymin", default="0"))
        xmax = float(b.findtext("xmax", default="0"))
        ymax = float(b.findtext("ymax", default="0"))
        xmin, xmax = sorted((max(0.0, xmin), min(float(width), xmax)))
        ymin, ymax = sorted((max(0.0, ymin), min(float(height), ymax)))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        boxes.append((POSITIVE_CLASSES[cls], cx, cy, bw, bh))
    return image_path, boxes, raw_classes


def write_list(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in items) + ("\n" if items else ""), encoding="utf-8")


def write_yaml(path, train_txt, val_txt):
    text = (
        f"path: {path.parent}\n"
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )
    path.write_text(text, encoding="utf-8")


def stratified_eval_split(labels, frac, salt):
    by_label = defaultdict(list)
    for image, label in labels.items():
        by_label[label].append(image)
    calib = []
    test = []
    for label, images in sorted(by_label.items()):
        ordered = sorted(
            images,
            key=lambda image: hashlib.md5(f"{salt}|{label}|{image}".encode("utf-8")).hexdigest(),
        )
        if len(ordered) == 1:
            test.extend(ordered)
            continue
        n_calib = min(max(int(round(len(ordered) * frac)), 1), len(ordered) - 1)
        calib.extend(ordered[:n_calib])
        test.extend(ordered[n_calib:])
    return sorted(calib), sorted(test)


def write_label_csv(path, labels, images):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for image in images:
            writer.writerow([image, labels[image]])


def copy_or_link(src, dst, copy_images):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)


def unique_image_name(image):
    digest = hashlib.md5(str(image.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{image.stem}_{digest}{image.suffix.lower()}"


def reweighted_sched(positives, negatives, target_count, neg_frac, seed=20260604):
    if not positives or not negatives:
        return list(positives) + list(negatives)
    import numpy as np

    base_neg_frac = len(negatives) / max(target_count, 1)
    target_neg_frac = min(0.85, max(neg_frac, base_neg_frac))
    target_neg = min(target_count - 1, max(len(negatives), int(round(target_count * target_neg_frac))))
    target_pos = target_count - target_neg
    rng = np.random.default_rng(seed)
    pos_pool = list(positives)
    neg_pool = list(negatives)
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    sched = []
    while len(sched) < target_pos:
        sched.extend(pos_pool)
    neg_sched = []
    while len(neg_sched) < target_neg:
        neg_sched.extend(neg_pool)
    sched = sched[:target_pos] + neg_sched[:target_neg]
    rng.shuffle(sched)
    return sched


def main():
    ap = argparse.ArgumentParser(description="Prepare DFS VOC data for Stage5-PV v2")
    ap.add_argument("--source", default="external_data/DFS-FIRE-SMOKE-Dataset")
    ap.add_argument("--out-root", default="data/stage5_pv_v2/dfs")
    ap.add_argument("--copy-images", action="store_true", help="copy images instead of symlink/fallback-copy")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--eval-split-salt", default="stage5_pv_v2_dfs_eval")
    ap.add_argument("--audit-negative-samples", type=int, default=50)
    ap.add_argument("--sched-neg-frac", type=float, default=DEFAULT_SCHED_NEG_FRAC)
    args = ap.parse_args()

    source = resolve(args.source)
    out = resolve(args.out_root)
    if not 0.0 < args.sched_neg_frac < 1.0:
        raise SystemExit("[error] --sched-neg-frac must be between 0 and 1")
    if not source.exists():
        raise SystemExit(
            f"[error] DFS source not found: {source}\n"
            "Download/clone DFS-FIRE-SMOKE-Dataset first; do not use the local "
            "DeepQuest Fire-Smoke-Dataset classification folders as detection data."
        )
    xmls = find_xmls(source)
    if not xmls:
        raise SystemExit(f"[error] no VOC XML files found under {source}")

    images_out = out / "images"
    labels_out = out / "labels"
    train_pos = []
    train_full = []
    eval_images = []
    eval_labels = {}
    hard_negatives = []
    class_counts = Counter()
    split_counts = Counter()
    negative_audit = []

    for xml in xmls:
        image, boxes, raw_classes = parse_voc(xml, source)
        split = split_name(xml)
        rel = f"{split}/{unique_image_name(image)}"
        dst_image = images_out / rel
        dst_label = labels_out / split / (Path(rel).stem + ".txt")
        copy_or_link(image, dst_image, args.copy_images)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text(
            "\n".join(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in boxes)
            + ("\n" if boxes else ""),
            encoding="utf-8",
        )
        for cls in raw_classes:
            class_counts[cls] += 1
        split_counts[split] += 1
        image_abs = str(dst_image.resolve())
        is_positive = bool(boxes)
        if split == "train":
            train_full.append(image_abs)
            if is_positive:
                train_pos.append(image_abs)
            else:
                hard_negatives.append(image_abs)
                if len(negative_audit) < args.audit_negative_samples:
                    negative_audit.append(image_abs)
        else:
            eval_images.append(image_abs)
            labels = {cls for cls, *_ in boxes}
            if 1 in labels:
                eval_labels[image_abs] = "fire"
            elif 0 in labels:
                eval_labels[image_abs] = "smoke"
            else:
                eval_labels[image_abs] = "other"

    if not train_pos or not train_full or not eval_images:
        raise SystemExit(
            f"[error] prepared split is unusable: train_pos={len(train_pos)} "
            f"train_full={len(train_full)} eval={len(eval_images)}"
        )

    target_count = len(train_full)
    sched = reweighted_sched(train_pos, hard_negatives, target_count, args.sched_neg_frac)
    if len(sched) < target_count:
        sched = list(train_full)

    train_pos_txt = out / "train_posonly.txt"
    train_full_txt = out / "train_full.txt"
    train_sched_txt = out / "train_hardneg_sched.txt"
    eval_txt = out / "eval.txt"
    write_list(train_pos_txt, train_pos)
    write_list(train_full_txt, train_full)
    write_list(train_sched_txt, sched[:target_count])
    write_list(eval_txt, eval_images)
    write_yaml(out / "dfs_posonly.yaml", train_pos_txt, eval_txt)
    write_yaml(out / "dfs_full.yaml", train_full_txt, eval_txt)
    write_yaml(out / "dfs_hardneg_sched.yaml", train_sched_txt, eval_txt)

    eval_labels_csv = out / "eval_labels.csv"
    write_label_csv(eval_labels_csv, eval_labels, sorted(eval_labels))
    calib, test = stratified_eval_split(eval_labels, args.calib_frac, args.eval_split_salt)
    write_label_csv(out / "eval_calib_labels.csv", eval_labels, calib)
    write_label_csv(out / "eval_test_labels.csv", eval_labels, test)

    train_pos_set = set(train_pos)
    hard_negative_set = set(hard_negatives)
    audit = {
        "source": str(source),
        "out_root": str(out),
        "n_xml": len(xmls),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "train_posonly": len(train_pos),
        "train_full": len(train_full),
        "train_hardneg_sched": target_count,
        "train_hardneg_sched_positive": sum(1 for image in sched[:target_count] if image in train_pos_set),
        "train_hardneg_sched_negative": sum(1 for image in sched[:target_count] if image in hard_negative_set),
        "unique_positive_coverage": round(
            sum(1 for image in sched[:target_count] if image in train_pos_set) / max(len(train_pos), 1),
            4,
        ),
        "positive_coverage_caveat": "This arm increases negative exposure at fixed total steps by reducing unique positive coverage.",
        "requested_sched_neg_frac": args.sched_neg_frac,
        "sched_neg_frac": args.sched_neg_frac,
        "train_negative_images": len(hard_negatives),
        "eval_images": len(eval_images),
        "eval_label_counts": dict(Counter(eval_labels.values())),
        "negative_audit_samples": negative_audit,
        "pv_hard_negative_taxonomy": PV_HARDNEG_TAXONOMY,
        "claim_boundary": "DFS other/negative images are PV-motivated hard negatives, not real PV station deployment evidence.",
        "note": "Inspect negative_audit_samples manually before claiming DFS other/negative images are clean hard negatives.",
    }
    (out / "prepare_audit.json").write_text(
        __import__("json").dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] prepared DFS v2 data at {out}")
    print(f"[audit] {out / 'prepare_audit.json'}")


if __name__ == "__main__":
    main()
