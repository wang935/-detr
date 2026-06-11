from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO


ROOT = Path(r"D:\detr_Q3")
IMG_DIR = ROOT / "pilot_images"
OUT_DIR = ROOT / "pilot_outputs"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


QUERIES = [
    ("fire", "flames fire burning outdoor", 6),
    ("smoke", "wildfire smoke plume", 4),
    ("distractor", "orange sunset sky", 3),
    ("distractor", "traffic light night orange", 3),
    ("distractor", "orange flower close up", 3),
    ("distractor", "campfire person", 2),
]


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Codex local pilot"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        if len(data) < 1024:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def commons_images(query: str, limit: int) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": str(limit * 3),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "960",
        "origin": "*",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Codex local pilot"})
        with urllib.request.urlopen(req, timeout=25) as r:
            payload = json.loads(r.read().decode("utf-8"))
        pages = payload.get("query", {}).get("pages", {})
        urls = []
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            u = info.get("thumburl") or info.get("url")
            if u and any(u.lower().split("?")[0].endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                urls.append(u)
        return urls[:limit]
    except Exception:
        return []


def prepare_images() -> list[tuple[Path, str]]:
    manifest: list[tuple[Path, str]] = []

    dfire_url = "https://raw.githubusercontent.com/gaia-solutions-on-demand/DFireDataset/master/figures/dfire_examples.png"
    dfire_path = IMG_DIR / "dfire_examples.png"
    if download(dfire_url, dfire_path):
        manifest.append((dfire_path, "mixed_dfire_examples"))

    idx = 0
    for label, query, limit in QUERIES:
        for u in commons_images(query, limit):
            idx += 1
            suffix = Path(urllib.parse.urlparse(u).path).suffix.lower()
            if suffix not in [".jpg", ".jpeg", ".png"]:
                suffix = ".jpg"
            path = IMG_DIR / f"{label}_{idx:02d}{suffix}"
            if download(u, path):
                manifest.append((path, label))
    return manifest


def flame_mask_stats(img: np.ndarray) -> dict[str, float | int]:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    red_or_orange = ((hue <= 35) | (hue >= 170)) & (sat >= 60) & (val >= 80)
    rgb_order = (r > g) & (g >= b * 0.75) & (r >= 120)
    bright_warm = red_or_orange & rgb_order
    mask = bright_warm.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    areas = [a for a in areas if a >= max(20, 0.0002 * h * w)]
    return {
        "warm_pixel_ratio": float(mask.mean() / 255.0),
        "warm_components": int(len(areas)),
        "largest_warm_component_ratio": float(max(areas) / (h * w)) if areas else 0.0,
    }


def run_model(model, image_path: Path, imgsz: int = 640) -> tuple[dict, np.ndarray]:
    t0 = time.perf_counter()
    results = model.predict(str(image_path), imgsz=imgsz, conf=0.15, verbose=False, device=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    r = results[0]
    boxes = r.boxes
    names = r.names
    dets = []
    if boxes is not None:
        for cls, conf, xyxy in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()):
            dets.append({
                "class": names[int(cls)],
                "conf": float(conf),
                "xyxy": [float(x) for x in xyxy],
            })
    top = sorted(dets, key=lambda d: d["conf"], reverse=True)[:5]
    counts: dict[str, int] = {}
    for d in dets:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    annotated = r.plot()
    return {
        "inference_ms": ms,
        "num_detections": len(dets),
        "class_counts": counts,
        "top_detections": top,
    }, annotated


def main() -> None:
    image_manifest = prepare_images()
    if not image_manifest:
        raise RuntimeError("No pilot images downloaded.")

    models = [
        ("rtdetr_l_coco", RTDETR(str(ROOT / "rtdetr-l.pt"))),
        ("yolo26n_coco", YOLO(str(ROOT / "yolo26n.pt"))),
    ]

    rows = []
    details = []
    for image_path, label in image_manifest:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        mask_stats = flame_mask_stats(img)
        for model_name, model in models:
            pred, annotated = run_model(model, image_path)
            out_img = OUT_DIR / f"{image_path.stem}_{model_name}.jpg"
            cv2.imwrite(str(out_img), annotated)
            row = {
                "image": image_path.name,
                "pilot_label": label,
                "model": model_name,
                **mask_stats,
                "inference_ms": pred["inference_ms"],
                "num_detections": pred["num_detections"],
                "classes": ";".join(f"{k}:{v}" for k, v in sorted(pred["class_counts"].items())),
                "top_detection": pred["top_detections"][0]["class"] if pred["top_detections"] else "",
                "top_conf": pred["top_detections"][0]["conf"] if pred["top_detections"] else 0.0,
                "annotated": str(out_img),
            }
            rows.append(row)
            details.append({**row, "top_detections_json": pred["top_detections"]})

    csv_path = OUT_DIR / "pilot_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT_DIR / "pilot_details.json"
    json_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {}
    for label in sorted({r["pilot_label"] for r in rows}):
        subset = [r for r in rows if r["pilot_label"] == label]
        summary[label] = {
            "rows": len(subset),
            "images": len({r["image"] for r in subset}),
            "mean_warm_pixel_ratio": float(np.mean([r["warm_pixel_ratio"] for r in subset])),
            "mean_largest_warm_component_ratio": float(np.mean([r["largest_warm_component_ratio"] for r in subset])),
            "mean_inference_ms": float(np.mean([r["inference_ms"] for r in subset])),
            "mean_detections": float(np.mean([r["num_detections"] for r in subset])),
        }
    (OUT_DIR / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "images": len({p.name for p, _ in image_manifest}),
        "rows": len(rows),
        "csv": str(csv_path),
        "summary": summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
