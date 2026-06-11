from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "training_summary_20260607"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_run_context(results_path: Path) -> dict[str, Any]:
    s = str(results_path).replace("/", "\\")
    parts = results_path.parts
    ctx: dict[str, Any] = {
        "results_path": rel(results_path),
        "stage": "unknown",
        "evidence_level": "historical_or_unclassified",
        "dataset": "",
        "family": "",
        "seed": "",
        "run_id": "",
        "arm": results_path.parent.name,
        "status_note": "",
    }

    if "runs_stage5_formal" in parts:
        idx = parts.index("runs_stage5_formal")
        run_id = parts[idx + 1]
        ctx.update(
            {
                "stage": "stage5_formal",
                "evidence_level": "formal_main",
                "run_id": run_id,
                "arm": parts[idx + 2],
            }
        )
        m = re.match(r"([a-zA-Z0-9]+)_seed(\d+)", run_id)
        if m:
            ctx["family"] = m.group(1)
            ctx["seed"] = int(m.group(2))
            ctx["dataset"] = "dfire"
    elif "runs_stage5_pv_v2" in parts:
        idx = parts.index("runs_stage5_pv_v2")
        run_id = parts[idx + 1]
        ctx.update(
            {
                "stage": "stage5_pv_v2",
                "evidence_level": "single_arm_probe",
                "run_id": run_id,
                "arm": parts[idx + 2],
            }
        )
        m = re.match(r"([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_seed(\d+)", run_id)
        if m:
            ctx["dataset"] = m.group(1)
            ctx["family"] = m.group(2)
            ctx["seed"] = int(m.group(3))
    elif "runs_stage5_mincheck" in parts:
        idx = parts.index("runs_stage5_mincheck")
        run_id = parts[idx + 1]
        ctx.update(
            {
                "stage": "stage5_mincheck",
                "evidence_level": "smoke_check",
                "run_id": run_id,
                "arm": parts[idx + 2],
                "status_note": "mincheck; do not use as formal paper evidence",
            }
        )
        m = re.match(r"([a-zA-Z0-9]+)_seed(\d+)", run_id)
        if m:
            ctx["family"] = m.group(1)
            ctx["seed"] = int(m.group(2))
            ctx["dataset"] = "dfire"
    elif "runs_stage3_yolo" in parts:
        ctx.update({"stage": "stage3_yolo", "evidence_level": "historical_pilot", "family": "yolo", "dataset": "dfire"})
        ctx["run_id"] = "stage3_yolo"
        ctx["status_note"] = "early stage3 run; not part of current formal matrix"
        if "\\runs\\detect\\runs\\detect\\" in s:
            ctx["stage"] = "stage3_yolo_nested_copy"
            ctx["status_note"] = "nested historical copy; verify before citing"
    elif "runs_tierb" in parts:
        idx = parts.index("runs_tierb")
        ctx.update({"stage": "tierb", "evidence_level": "historical_pilot", "family": "yolo", "dataset": "dfire"})
        ctx["run_id"] = parts[idx + 1]
        ctx["arm"] = parts[idx + 1]
        ctx["status_note"] = "tierb historical run"
        if "invalid" in parts[idx + 1]:
            ctx["evidence_level"] = "invalid_or_superseded"
            ctx["status_note"] = "explicitly marked invalid_before_clean"
    return ctx


def collect_epoch_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_frames: list[pd.DataFrame] = []
    final_rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "runs" / "detect").rglob("results.csv")):
        ctx = infer_run_context(path)
        df = clean_columns(pd.read_csv(path))
        if df.empty:
            continue
        for col in df.columns:
            if col != "epoch":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        tagged = df.copy()
        for k, v in ctx.items():
            tagged.insert(0, k, v)
        epoch_frames.append(tagged)

        last = df.iloc[-1].to_dict()
        run_dir = path.parent
        row: dict[str, Any] = {
            **ctx,
            "rows": len(df),
            "final_epoch": last.get("epoch", ""),
            "final_time": last.get("time", ""),
            "train_box_loss": last.get("train/box_loss", ""),
            "train_cls_loss": last.get("train/cls_loss", ""),
            "train_dfl_loss": last.get("train/dfl_loss", ""),
            "precision_B": last.get("metrics/precision(B)", ""),
            "recall_B": last.get("metrics/recall(B)", ""),
            "map50_B": last.get("metrics/mAP50(B)", ""),
            "map50_95_B": last.get("metrics/mAP50-95(B)", ""),
            "val_box_loss": last.get("val/box_loss", ""),
            "val_cls_loss": last.get("val/cls_loss", ""),
            "val_dfl_loss": last.get("val/dfl_loss", ""),
            "last_pt_exists": (run_dir / "weights" / "last.pt").exists(),
            "best_pt_exists": (run_dir / "weights" / "best.pt").exists(),
            "run_dir": rel(run_dir),
        }
        final_rows.append(row)

    epoch_history = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()
    training_final = pd.DataFrame(final_rows)
    return epoch_history, training_final


def stage_from_gonogo(path: Path) -> tuple[str, str, str]:
    if path.parent.parent.name == "stage5":
        return "stage5_formal", "formal_main", "dfire"
    if path.parent.parent.name == "stage5_mincheck":
        return "stage5_mincheck", "smoke_check", "dfire"
    if path.parent.parent.name == "stage5_pv_v2":
        return "stage5_pv_v2", "single_arm_probe", ""
    return path.parent.parent.name, "unknown", ""


def count_csv_rows(path_text: str | None) -> int | str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists():
        return ""
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except Exception:
        return ""


def collect_eval_tables(training_final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    train_lookup = {}
    if not training_final.empty:
        for row in training_final.to_dict("records"):
            train_lookup[(row.get("stage"), str(row.get("seed")), row.get("arm"))] = row

    for gonogo in sorted((ROOT / "formal_results").rglob("gonogo.json")):
        stage, evidence_level, default_dataset = stage_from_gonogo(gonogo)
        try:
            data = read_json(gonogo)
        except Exception:
            continue
        for model_key, model in data.items():
            if not isinstance(model, dict):
                continue
            family = model.get("family", "")
            arm = model.get("arm", "")
            seed = model.get("seed", "")
            dataset = model.get("dataset", default_dataset)
            pred = model.get("pred", "")
            pred_rows.append(
                {
                    "stage": stage,
                    "evidence_level": evidence_level,
                    "dataset": dataset,
                    "family": family,
                    "seed": seed,
                    "arm": arm,
                    "model_key": model_key,
                    "pred_rows": count_csv_rows(pred),
                    "pred_path": rel(pred),
                    "weight_exists": Path(model.get("weight", "")).exists() if model.get("weight") else "",
                    "pred_meta_exists": Path(model.get("pred_meta", "")).exists() if model.get("pred_meta") else "",
                    "gonogo_path": rel(gonogo),
                }
            )
            fixed = model.get("at_fixed_recall", {}) or {}
            for target, vals in fixed.items():
                base = {
                    "stage": stage,
                    "evidence_level": evidence_level,
                    "dataset": dataset,
                    "family": family,
                    "seed": seed,
                    "arm": arm,
                    "target_recall": float(target),
                    "eval_protocol": model.get("eval_protocol", ""),
                    "eval_split_salt": model.get("eval_split_salt", ""),
                    "calib_frac": model.get("calib_frac", ""),
                    "n_pos": model.get("n_pos", ""),
                    "n_neg": model.get("n_neg", ""),
                    "epochs": model.get("epochs", ""),
                    "val": model.get("val", ""),
                    "export_conf": model.get("export_conf", ""),
                    "train_count": (model.get("arm_train_counts") or {}).get(arm, ""),
                    "gonogo_path": rel(gonogo),
                    "pred_path": rel(pred),
                }
                trow = train_lookup.get((stage, str(seed), arm), {})
                base.update(
                    {
                        "map50_B": trow.get("map50_B", ""),
                        "map50_95_B": trow.get("map50_95_B", ""),
                        "precision_B": trow.get("precision_B", ""),
                        "train_recall_B": trow.get("recall_B", ""),
                        "results_rows": trow.get("rows", ""),
                    }
                )
                if vals is None:
                    base.update({"threshold": "", "calib_recall": "", "actual_recall": "", "FPR": "", "FPPI": "", "metric_status": "missing_or_not_reported"})
                else:
                    base.update(
                        {
                            "threshold": vals.get("thr", ""),
                            "calib_recall": vals.get("calib_recall", ""),
                            "actual_recall": vals.get("recall", ""),
                            "FPR": vals.get("FPR", ""),
                            "FPPI": vals.get("FPPI", ""),
                            "metric_status": "reported",
                        }
                    )
                rows.append(base)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def summarize_eval(eval_long: pd.DataFrame) -> pd.DataFrame:
    if eval_long.empty:
        return pd.DataFrame()
    reported = eval_long[eval_long["metric_status"].eq("reported")].copy()
    for col in ["actual_recall", "FPR", "FPPI", "map50_B", "map50_95_B"]:
        reported[col] = pd.to_numeric(reported[col], errors="coerce")
    grouped = (
        reported.groupby(["stage", "evidence_level", "dataset", "family", "arm", "target_recall"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
            recall_mean=("actual_recall", "mean"),
            recall_sd=("actual_recall", "std"),
            FPR_mean=("FPR", "mean"),
            FPR_sd=("FPR", "std"),
            FPPI_mean=("FPPI", "mean"),
            FPPI_sd=("FPPI", "std"),
            map50_mean=("map50_B", "mean"),
            map50_sd=("map50_B", "std"),
            map50_95_mean=("map50_95_B", "mean"),
            map50_95_sd=("map50_95_B", "std"),
        )
        .reset_index()
        .fillna("")
    )
    return grouped


def summarize_training(training_final: pd.DataFrame) -> pd.DataFrame:
    if training_final.empty:
        return pd.DataFrame()
    df = training_final.copy()
    for col in ["map50_B", "map50_95_B", "precision_B", "recall_B", "train_box_loss", "train_cls_loss", "train_dfl_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.groupby(["stage", "evidence_level", "dataset", "family", "arm"], dropna=False)
        .agg(
            n_runs=("results_path", "count"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(set(v for v in x if str(v) != ""))))),
            final_epoch_min=("final_epoch", "min"),
            final_epoch_max=("final_epoch", "max"),
            map50_mean=("map50_B", "mean"),
            map50_sd=("map50_B", "std"),
            map50_95_mean=("map50_95_B", "mean"),
            map50_95_sd=("map50_95_B", "std"),
            precision_mean=("precision_B", "mean"),
            train_recall_mean=("recall_B", "mean"),
            train_box_loss_mean=("train_box_loss", "mean"),
            train_cls_loss_mean=("train_cls_loss", "mean"),
            train_dfl_loss_mean=("train_dfl_loss", "mean"),
        )
        .reset_index()
        .fillna("")
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    safe = df.fillna("").astype(str)
    headers = list(safe.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in safe.iterrows():
        vals = [str(row[h]).replace("\n", " ").replace("|", "/") for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown_tables(training_final: pd.DataFrame, eval_long: pd.DataFrame, eval_summary: pd.DataFrame, training_summary: pd.DataFrame) -> None:
    r90 = eval_long[(eval_long["metric_status"].eq("reported")) & (eval_long["target_recall"].eq(0.90))].copy()
    keep_eval = [
        "stage",
        "evidence_level",
        "dataset",
        "family",
        "seed",
        "arm",
        "actual_recall",
        "FPR",
        "FPPI",
        "map50_B",
        "map50_95_B",
        "train_count",
    ]
    keep_train = [
        "stage",
        "evidence_level",
        "dataset",
        "family",
        "seed",
        "arm",
        "final_epoch",
        "rows",
        "precision_B",
        "recall_B",
        "map50_B",
        "map50_95_B",
        "status_note",
    ]
    text = [
        "# Training Summary Tables",
        "",
        "## R0.90 formal/probe rows",
        "",
        markdown_table(r90[keep_eval].sort_values(["stage", "family", "seed", "arm"]).round(4)) if not r90.empty else "_No R0.90 rows._",
        "",
        "## Training final rows",
        "",
        markdown_table(training_final[keep_train].sort_values(["stage", "family", "seed", "arm"]).round(4)) if not training_final.empty else "_No training rows._",
        "",
        "## Evaluation aggregate",
        "",
        markdown_table(eval_summary.round(4)) if not eval_summary.empty else "_No eval aggregate._",
        "",
        "## Training aggregate",
        "",
        markdown_table(training_summary.round(4)) if not training_summary.empty else "_No training aggregate._",
        "",
    ]
    (OUT_DIR / "training_summary_tables.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    epoch_history, training_final = collect_epoch_tables()
    eval_long, prediction_inventory = collect_eval_tables(training_final)
    eval_summary = summarize_eval(eval_long)
    training_summary = summarize_training(training_final)

    outputs = {
        "training_epoch_history.csv": epoch_history,
        "training_final_runs.csv": training_final,
        "fixed_recall_eval_long.csv": eval_long,
        "fixed_recall_eval_summary.csv": eval_summary,
        "training_final_summary.csv": training_summary,
        "prediction_inventory.csv": prediction_inventory,
    }
    for name, df in outputs.items():
        df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")

    write_markdown_tables(training_final, eval_long, eval_summary, training_summary)

    manifest = {
        "output_dir": str(OUT_DIR),
        "epoch_history_rows": int(len(epoch_history)),
        "training_final_rows": int(len(training_final)),
        "fixed_recall_eval_rows": int(len(eval_long)),
        "fixed_recall_reported_rows": int((eval_long.get("metric_status", pd.Series(dtype=str)) == "reported").sum()) if not eval_long.empty else 0,
        "prediction_inventory_rows": int(len(prediction_inventory)),
        "notes": [
            "stage5_formal is the main completed formal matrix.",
            "stage5_pv_v2 hardneg_sched is a three-seed single-arm probe; strict all4 aggregate files are empty.",
            "stage5_mincheck and historical runs are retained in training tables but should not be cited as paper-grade formal evidence.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
