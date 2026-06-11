#!/usr/bin/env python3
"""Stage 3C: external neutral/distractor stress test for DAQ.

The external set is deliberately used only as negatives. Candidate selection and
recall thresholds come from D-Fire calibration/holdout data, then those frozen
decisions are applied to external neutral images.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage2_daq_oneclick as base
import stage2_daq_strong as strong


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_tokens(text):
    return [t.strip().lower().replace(" ", "_") for t in str(text).split(",") if t.strip()]


def norm_part(text):
    return str(text).strip().lower().replace(" ", "_")


def find_neutral_images(root, neutral_tokens, limit=0):
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"[error] external root not found: {root}")

    found = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_parts = [norm_part(p) for p in path.relative_to(root).parts[:-1]]
        if any(any(tok in part for tok in neutral_tokens) for part in rel_parts):
            found.append(str(path.resolve()))

    if not found:
        # Fallback for classification datasets that encode class in filenames.
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = norm_part(path.stem)
            if any(tok in stem for tok in neutral_tokens):
                found.append(str(path.resolve()))

    found = sorted(dict.fromkeys(found))
    if limit and limit > 0:
        found = found[:limit]
    return found


def write_labels(path, imgs, label="none"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for img in imgs:
            writer.writerow([img, label])


def export_rtdetr_predictions(weights, imgs, out_csv, imgsz, conf, max_det, workers_note=""):
    try:
        import torch
        from ultralytics import RTDETR
    except Exception as exc:
        raise SystemExit(f"[error] torch/ultralytics unavailable: {exc}")

    dev = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == 0 else "CPU"
    print(
        f"[export] weights={weights} images={len(imgs)} device={dev}({gpu}) "
        f"conf>={conf} max_det={max_det} {workers_note}".strip()
    )
    model = RTDETR(str(weights))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for idx, img in enumerate(imgs, start=1):
            result = model.predict(
                img,
                conf=conf,
                max_det=max_det,
                imgsz=imgsz,
                device=dev,
                verbose=False,
            )[0]
            if result.boxes is not None and len(result.boxes):
                for c in result.boxes.conf.detach().cpu().tolist():
                    writer.writerow([img, float(c)])
                    n_rows += 1
            if idx % 100 == 0 or idx == len(imgs):
                print(f"  ...{idx}/{len(imgs)} rows={n_rows}")
    print(f"[export] wrote {out_csv} rows={n_rows}")


def load_prediction_csv(path, valid_imgs):
    valid = set(valid_imgs)
    dets = defaultdict(list)
    rows = 0
    unknown = 0
    bad = 0
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[error] prediction CSV missing: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "conf"}.issubset(reader.fieldnames):
            raise SystemExit(f"[error] {path} must contain image,conf columns")
        for row_idx, row in enumerate(reader, start=2):
            img = row["image"].strip()
            try:
                conf = float(row["conf"])
            except Exception:
                bad += 1
                if bad <= 5:
                    print(f"[warn] {path}:{row_idx} non-numeric conf: {row.get('conf')!r}")
                continue
            if not np.isfinite(conf) or conf < 0.0 or conf > 1.0:
                bad += 1
                if bad <= 5:
                    print(f"[warn] {path}:{row_idx} invalid conf: {conf}")
                continue
            if img not in valid:
                unknown += 1
                if unknown <= 5:
                    print(f"[warn] {path}:{row_idx} unknown image ignored: {img}")
                continue
            dets[img].append(conf)
            rows += 1
    for img in dets:
        dets[img].sort(reverse=True)
    return dets, {"rows": rows, "images_with_dets": len(dets), "unknown": unknown, "bad": bad}


def score_matrix(dets, imgs, max_det):
    scores = np.zeros((len(imgs), max_det), dtype=np.float64)
    for row, img in enumerate(imgs):
        vals = np.asarray(dets.get(img, []), dtype=np.float64)
        if vals.size:
            vals = np.sort(vals)[::-1]
            n = min(max_det, vals.size)
            scores[row, :n] = vals[:n]
    return scores


def neg_metrics_at(scores, thr):
    if scores.shape[0] == 0:
        return {"thr": round(float(thr), 4), "FPR": 0.0, "FPPI": 0.0}
    max_scores = scores.max(axis=1)
    return {
        "thr": round(float(thr), 4),
        "FPR": round(float(np.mean(max_scores >= thr)), 4),
        "FPPI": round(float(np.sum(scores >= thr) / max(int(scores.shape[0]), 1)), 4),
    }


def write_score_csv(path, imgs, scores):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "conf"])
        for img, vals in zip(imgs, scores):
            keep = vals[vals > 0.0]
            keep = np.sort(keep)[::-1]
            for conf in keep:
                writer.writerow([img, float(conf)])


def report_table(summary):
    lines = [
        "| target recall fixed on D-Fire | raw ext FPR | DAQ ext FPR | delta FPR | raw ext FPPI | DAQ ext FPPI | delta FPPI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("0.80", "0.85", "0.90", "0.95"):
        raw = summary["external"]["raw_hardneg_at_dfire_thr"].get(target)
        daq = summary["external"]["daq_at_dfire_thr"].get(target)
        if not raw or not daq:
            continue
        lines.append(
            f"| {target} | {raw['FPR']:.4f} | {daq['FPR']:.4f} | {daq['FPR'] - raw['FPR']:+.4f} | "
            f"{raw['FPPI']:.4f} | {daq['FPPI']:.4f} | {daq['FPPI'] - raw['FPPI']:+.4f} |"
        )
    return lines


def verdict(summary, main_target, min_gain, fppi_tol):
    raw = summary["external"]["raw_hardneg_at_dfire_thr"].get(main_target)
    daq = summary["external"]["daq_at_dfire_thr"].get(main_target)
    if not raw or not daq:
        return "FAIL: main target missing"
    if raw["FPR"] <= 0.01 and daq["FPR"] <= 0.01:
        return "INCONCLUSIVE: external neutral set is too easy at the main target"
    fpr_gain = raw["FPR"] - daq["FPR"]
    fppi_ok = daq["FPPI"] <= raw["FPPI"] * (1.0 + fppi_tol)
    if fpr_gain >= min_gain and fppi_ok:
        return "GO: DAQ transfers to external neutral distractors"
    if fpr_gain > 0 and fppi_ok:
        return "BORDERLINE-GO: external FPR improves but margin is modest"
    if fpr_gain > 0:
        return "NO-GO: external FPR improves but FPPI worsens too much"
    return "NO-GO: DAQ does not improve external neutral false alarms"


def write_report(path, summary):
    selected = summary["selected_candidate"]
    lines = [
        "# Stage 3C External Distractor Report",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- source: `{summary['external_source']}`",
        f"- external root: `{summary['external_root']}`",
        f"- external neutral images: `{summary['external']['n_neg']}`",
        f"- selected DAQ: `{strong.variant_name(selected)}`",
        f"- main target: `{summary['main_target']}`",
        f"- D-Fire holdout gate pos_mean: `{summary['gate_stats']['dfire_holdout_pos_mean']:.4f}`",
        f"- D-Fire holdout gate neg_mean: `{summary['gate_stats']['dfire_holdout_neg_mean']:.4f}`",
        f"- external gate mean: `{summary['gate_stats']['external_mean']:.4f}`",
        "",
    ]
    lines.extend(report_table(summary))
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "- DAQ candidate selection uses only D-Fire calibration images.",
            "- D-Fire holdout positives set the recall thresholds; external neutral images are not used to tune thresholds.",
            "- This is an image-level alarm false-positive stress test, not box-level localization validation.",
        ]
    )
    if summary["warnings"]:
        lines.extend(["", "## Warnings"])
        for msg in summary["warnings"]:
            lines.append(f"- {msg}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Stage 3C external neutral distractor stress test")
    ap.add_argument("--external-root", default="external_data/FIRE-SMOKE-DATASET")
    ap.add_argument("--external-source", default="DeepQuestAI Fire-Smoke-Dataset neutral")
    ap.add_argument("--neutral-tokens", default="neutral,normal,negative,none,other,no_fire,no-fire")
    ap.add_argument("--labels", default="data/dfire/eval_labels.csv")
    ap.add_argument("--hardneg-pred", default="preds/hardneg_clean5.csv")
    ap.add_argument("--hardneg-weight", default="runs/detect/runs_tierb/hardneg/weights/best.pt")
    ap.add_argument("--out-dir", default="stage3_daq/external_deepquest")
    ap.add_argument("--external-pred", default="")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--split-salt", default="oneclick")
    ap.add_argument("--main-target", default="0.90", choices=["0.80", "0.85", "0.90", "0.95"])
    ap.add_argument("--lambdas", default="0,0.25,0.50,0.75,1.0,1.5,2.0,3.0,4.0,5.0")
    ap.add_argument("--topks", default="1,2,3,5,10,20,100")
    ap.add_argument("--rank-gammas", default="0,0.5,1.0,1.5")
    ap.add_argument("--floors", default="0")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.02)
    ap.add_argument("--fppi-tol", type=float, default=0.05)
    ap.add_argument("--min-fpr-gain", type=float, default=0.015)
    ap.add_argument("--keep-leaderboard", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-external-images", type=int, default=50)
    ap.add_argument("--force-export", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    external_imgs = find_neutral_images(args.external_root, parse_tokens(args.neutral_tokens), args.limit)
    if len(external_imgs) < args.min_external_images:
        raise SystemExit(
            f"[error] only found {len(external_imgs)} external neutral images under {args.external_root}; "
            f"check --external-root or --neutral-tokens"
        )
    external_labels = out_dir / "external_labels.csv"
    write_labels(external_labels, external_imgs, "none")

    audit = base.Audit()
    labels = base.read_labels(args.labels, audit)
    hardneg_dets = base.read_predictions(args.hardneg_pred, labels, audit, "hardneg")
    if not Path(args.hardneg_weight).exists():
        audit.warn(f"hardneg weight missing: {args.hardneg_weight}; cached external predictions must already exist")
    if not audit.ok():
        audit.print()
        sys.exit(2)

    calib_imgs, holdout_imgs = strong.salted_split(labels, args.calib_frac, args.split_salt)
    if set(calib_imgs) & set(holdout_imgs):
        raise SystemExit("[error] split leakage: calibration and holdout overlap")

    x_calib, y_calib = base.build_matrix(labels, hardneg_dets, calib_imgs)
    x_holdout, y_holdout = base.build_matrix(labels, hardneg_dets, holdout_imgs)
    model = base.fit_logreg(x_calib, y_calib, args.steps, args.lr, args.l2)
    gate_calib = base.predict_gate(model, x_calib)
    gate_holdout = base.predict_gate(model, x_holdout)

    calib_pos, calib_scores = strong.conf_matrix(labels, hardneg_dets, calib_imgs, args.max_det)
    holdout_pos, holdout_scores = strong.conf_matrix(labels, hardneg_dets, holdout_imgs, args.max_det)
    raw_calib = strong.eval_score_matrix(calib_pos, calib_scores)
    raw_holdout = strong.eval_score_matrix(holdout_pos, holdout_scores)

    top_rows, all_rows = strong.scan_candidates(calib_scores, calib_pos, gate_calib, raw_calib, args)
    if not top_rows:
        raise SystemExit("[error] no DAQ candidate could be selected")
    selected = top_rows[0]["candidate"]
    daq_holdout_scores = strong.transform_scores(holdout_scores, gate_holdout, **selected)
    daq_holdout = strong.eval_score_matrix(holdout_pos, daq_holdout_scores)
    if selected["topk"] == 1:
        warnings.append("selected topk=1: interpret Stage 3C as scene-level alarm suppression")

    external_pred = Path(args.external_pred) if args.external_pred else out_dir / "external_hardneg_raw.csv"
    if not args.skip_export and (args.force_export or not external_pred.exists()):
        if not Path(args.hardneg_weight).exists():
            raise SystemExit(f"[error] cannot export without hardneg weight: {args.hardneg_weight}")
        export_rtdetr_predictions(
            args.hardneg_weight,
            external_imgs,
            external_pred,
            args.imgsz,
            args.conf,
            args.max_det,
        )
    elif not external_pred.exists():
        raise SystemExit(f"[error] external prediction CSV missing: {external_pred}")
    else:
        print(f"[cache] using existing external predictions: {external_pred}")

    external_dets, external_pred_stats = load_prediction_csv(external_pred, external_imgs)
    external_scores = score_matrix(external_dets, external_imgs, args.max_det)
    x_external = np.vstack([base.features_from_conf(external_dets.get(img, [])) for img in external_imgs])
    gate_external = base.predict_gate(model, x_external)
    daq_external_scores = strong.transform_scores(external_scores, gate_external, **selected)

    raw_external_at = {}
    daq_external_at = {}
    for target, row in raw_holdout["at_fixed_recall"].items():
        if row:
            raw_external_at[target] = neg_metrics_at(external_scores, row["thr"])
    for target, row in daq_holdout["at_fixed_recall"].items():
        if row:
            daq_external_at[target] = neg_metrics_at(daq_external_scores, row["thr"])

    write_score_csv(out_dir / "external_daq.csv", external_imgs, daq_external_scores)
    strong.write_leaderboard(out_dir / "candidate_leaderboard.csv", top_rows, args.main_target)

    gate_stats = {
        "dfire_calib_pos_mean": float(gate_calib[y_calib > 0.5].mean()),
        "dfire_calib_neg_mean": float(gate_calib[y_calib < 0.5].mean()),
        "dfire_holdout_pos_mean": float(gate_holdout[y_holdout > 0.5].mean()),
        "dfire_holdout_neg_mean": float(gate_holdout[y_holdout < 0.5].mean()),
        "external_mean": float(gate_external.mean()),
        "external_median": float(np.median(gate_external)),
        "external_p90": float(np.quantile(gate_external, 0.90)),
    }
    if gate_stats["external_mean"] >= gate_stats["dfire_holdout_pos_mean"]:
        warnings.append("external gate mean is unusually high; inspect whether neutral discovery mixed in fire/smoke images")

    summary = {
        "verdict": "",
        "external_source": args.external_source,
        "external_root": str(Path(args.external_root).resolve()),
        "main_target": args.main_target,
        "selected_candidate": selected,
        "split_salt": args.split_salt,
        "n_calib": len(calib_imgs),
        "n_holdout": len(holdout_imgs),
        "gate_stats": gate_stats,
        "warnings": warnings,
        "dfire_holdout": {"raw_hardneg": raw_holdout, "daq": daq_holdout},
        "external": {
            "n_neg": len(external_imgs),
            "prediction_stats": external_pred_stats,
            "raw_hardneg_at_dfire_thr": raw_external_at,
            "daq_at_dfire_thr": daq_external_at,
        },
        "calibration": {
            "raw_hardneg": raw_calib,
            "selected": top_rows[0],
            "candidate_count": len(all_rows),
        },
        "artifacts": {
            "summary_json": str(out_dir / "stage3_external_summary.json"),
            "report_md": str(out_dir / "STAGE3_EXTERNAL_DISTRACTOR.md"),
            "external_labels": str(external_labels),
            "external_raw": str(external_pred),
            "external_daq": str(out_dir / "external_daq.csv"),
            "candidate_leaderboard": str(out_dir / "candidate_leaderboard.csv"),
        },
    }
    summary["verdict"] = verdict(summary, args.main_target, args.min_fpr_gain, args.fppi_tol)

    with (out_dir / "stage3_external_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(out_dir / "STAGE3_EXTERNAL_DISTRACTOR.md", summary)

    audit.print()
    print("\n=== Stage 3C external distractor result ===")
    print(f"external_neutral={len(external_imgs)} selected={strong.variant_name(selected)}")
    print("\n".join(report_table(summary)))
    print(f"\n[verdict] {summary['verdict']}")
    print(f"[done] wrote {out_dir / 'STAGE3_EXTERNAL_DISTRACTOR.md'}")


if __name__ == "__main__":
    main()
