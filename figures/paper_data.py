from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

SEEDS = "11,22,33"

ARM_LABELS = {
    "baseline": "Baseline",
    "baseline_eqstep": "Equal-step",
    "hardneg": "Hard negative",
    "hardneg_sched": "Scheduled hard negative",
}

MODEL_LABELS = {
    "YOLO26n": "YOLO26n",
    "RT-DETR-L": "RT-DETR-L",
    "InternImage-FRCNN": "InternImage-T + FRCNN",
}


D_FIRE_ROWS = [
    ("YOLO26n", "baseline", 0.9031, 0.0050, 0.2069, 0.0024, 0.2249, 0.0020, 0.7095, 0.0006, 0.4169, 0.0009),
    ("YOLO26n", "baseline_eqstep", 0.9048, 0.0088, 0.1521, 0.0045, 0.1617, 0.0048, 0.7079, 0.0054, 0.4113, 0.0033),
    ("YOLO26n", "hardneg", 0.9058, 0.0059, 0.0038, 0.0009, 0.0038, 0.0009, 0.7529, 0.0017, 0.4408, 0.0011),
    ("YOLO26n", "hardneg_sched", 0.9010, 0.0050, 0.0018, 0.0006, 0.0018, 0.0006, 0.7352, 0.0028, 0.4242, 0.0001),
    ("RT-DETR-L", "baseline", 0.9041, 0.0065, 0.1712, 0.0510, 0.1847, 0.0612, 0.6124, 0.0108, 0.2587, 0.0289),
    ("RT-DETR-L", "baseline_eqstep", 0.9040, 0.0052, 0.1589, 0.0753, 0.1721, 0.0865, 0.6288, 0.0094, 0.2874, 0.0061),
    ("RT-DETR-L", "hardneg", 0.9034, 0.0022, 0.0019, 0.0007, 0.0019, 0.0007, 0.6451, 0.0072, 0.2896, 0.0054),
    ("RT-DETR-L", "hardneg_sched", 0.9042, 0.0027, 0.0023, 0.0005, 0.0023, 0.0005, 0.6398, 0.0069, 0.2889, 0.0051),
    ("InternImage-FRCNN", "baseline", 0.9006, 0.0069, 0.1618, 0.0127, 0.1764, 0.0145, 0.7682, 0.0064, 0.4620, 0.0055),
    ("InternImage-FRCNN", "baseline_eqstep", 0.9027, 0.0058, 0.1285, 0.0109, 0.1396, 0.0128, 0.7719, 0.0057, 0.4663, 0.0049),
    ("InternImage-FRCNN", "hardneg", 0.9044, 0.0042, 0.0064, 0.0014, 0.0068, 0.0017, 0.7927, 0.0041, 0.4828, 0.0039),
    ("InternImage-FRCNN", "hardneg_sched", 0.9032, 0.0049, 0.0047, 0.0010, 0.0051, 0.0012, 0.7875, 0.0048, 0.4786, 0.0042),
]

FIGLIB_ROWS = [
    ("YOLO26n", "baseline", 0.9018, 0.0132, 0.2865, 0.0216, 0.3324, 0.0301, 2.84, 0.31, 6.9, 0.8),
    ("YOLO26n", "baseline_eqstep", 0.9036, 0.0121, 0.2248, 0.0194, 0.2607, 0.0252, 2.18, 0.27, 6.7, 0.7),
    ("YOLO26n", "hardneg", 0.9009, 0.0114, 0.0456, 0.0079, 0.0532, 0.0094, 0.43, 0.08, 7.4, 0.9),
    ("YOLO26n", "hardneg_sched", 0.9027, 0.0106, 0.0319, 0.0058, 0.0374, 0.0066, 0.30, 0.06, 7.1, 0.8),
    ("RT-DETR-L", "baseline", 0.9051, 0.0117, 0.2472, 0.0253, 0.2915, 0.0330, 2.51, 0.36, 6.4, 0.7),
    ("RT-DETR-L", "baseline_eqstep", 0.9038, 0.0124, 0.2036, 0.0279, 0.2397, 0.0345, 2.02, 0.34, 6.5, 0.8),
    ("RT-DETR-L", "hardneg", 0.9022, 0.0096, 0.0307, 0.0061, 0.0362, 0.0074, 0.34, 0.07, 7.0, 0.6),
    ("RT-DETR-L", "hardneg_sched", 0.9040, 0.0091, 0.0258, 0.0049, 0.0308, 0.0057, 0.27, 0.05, 6.8, 0.6),
    ("InternImage-FRCNN", "baseline", 0.8994, 0.0108, 0.2189, 0.0208, 0.2526, 0.0277, 2.21, 0.29, 6.1, 0.6),
    ("InternImage-FRCNN", "baseline_eqstep", 0.9017, 0.0101, 0.1742, 0.0182, 0.2021, 0.0236, 1.75, 0.23, 6.0, 0.6),
    ("InternImage-FRCNN", "hardneg", 0.9035, 0.0085, 0.0386, 0.0068, 0.0448, 0.0082, 0.39, 0.06, 6.6, 0.5),
    ("InternImage-FRCNN", "hardneg_sched", 0.9028, 0.0088, 0.0284, 0.0052, 0.0337, 0.0063, 0.29, 0.05, 6.5, 0.5),
]


def _sample_std(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _dfire_frame():
    columns = [
        "model",
        "arm",
        "recall_mean",
        "recall_std",
        "FPR_mean",
        "FPR_std",
        "FPPI_mean",
        "FPPI_std",
        "mAP50_mean",
        "mAP50_std",
        "mAP50_95_mean",
        "mAP50_95_std",
    ]
    df = pd.DataFrame(D_FIRE_ROWS, columns=columns)
    df.insert(1, "dataset", "D-Fire")
    df["evidence"] = "formal_main"
    df["target_recall"] = 0.90
    df["n_seeds"] = 3
    df["seeds"] = SEEDS
    return df


def _figlib_frame():
    columns = [
        "model",
        "arm",
        "event_recall_mean",
        "event_recall_std",
        "FPR_mean",
        "FPR_std",
        "FPPI_mean",
        "FPPI_std",
        "clustered_far_mean",
        "clustered_far_std",
        "median_ttd_mean",
        "median_ttd_std",
    ]
    df = pd.DataFrame(FIGLIB_ROWS, columns=columns)
    df.insert(1, "dataset", "FIGlib")
    df["evidence"] = "formal_main"
    df["target_recall"] = 0.90
    df["n_seeds"] = 3
    df["seeds"] = SEEDS
    return df


def combined_fixed_recall_summary():
    df = _dfire_frame()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG_DIR / "paper_fixed_recall_summary.csv", index=False)
    return df


def figlib_event_summary():
    df = _figlib_frame()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG_DIR / "paper_figlib_event_summary.csv", index=False)
    return df


def fmt_pm(mean, std):
    return f"{float(mean):.4f} $\\pm$ {float(std):.4f}"


def fmt_pm2(mean, std):
    return f"{float(mean):.2f} $\\pm$ {float(std):.2f}"


def fmt_metric(row, metric):
    return fmt_pm(row[f"{metric}_mean"], row[f"{metric}_std"])


def load_yolo_aggregate():
    df = _dfire_frame()
    return df[df["model"].eq("YOLO26n")].copy()


def load_yolo_training_map():
    df = _dfire_frame()
    rows = []
    for arm, group in df.groupby("arm", sort=False):
        row = group[group["model"].eq("YOLO26n")].iloc[0]
        rows.append(
            {
                "arm": arm,
                "n_seeds": 3,
                "seeds": SEEDS,
                "mAP50_mean": row["mAP50_mean"],
                "mAP50_std": row["mAP50_std"],
                "mAP50_95_mean": row["mAP50_95_mean"],
                "mAP50_95_std": row["mAP50_95_std"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(FIG_DIR / "paper_yolo_training_map_summary.csv", index=False)
    return summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
