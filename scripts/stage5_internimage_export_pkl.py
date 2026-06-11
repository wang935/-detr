import argparse
import csv
import json
import pickle
from pathlib import Path


def bbox_results(result):
    if isinstance(result, tuple):
        result = result[0]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    ann = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    images = ann["images"]
    with Path(args.pkl).open("rb") as f:
        results = pickle.load(f)
    if len(results) != len(images):
        raise SystemExit(f"[error] results/images mismatch: {len(results)} vs {len(images)}")

    rows = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for image, result in zip(images, results):
            for cls_boxes in bbox_results(result):
                for box in cls_boxes:
                    if len(box) >= 5:
                        writer.writerow([image["file_name"], float(box[4])])
                        rows += 1

    meta = {
        "ann": str(Path(args.ann).resolve()),
        "pkl": str(Path(args.pkl).resolve()),
        "csv": str(out_path.resolve()),
        "eval_images": len(images),
        "rows": rows,
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] wrote {out_path} rows={rows}")


if __name__ == "__main__":
    main()
