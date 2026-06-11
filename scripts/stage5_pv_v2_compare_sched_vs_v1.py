#!/usr/bin/env python3
"""Compare available Stage5-PV v2 hardneg_sched runs with Stage 5 v1 YOLO arms."""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("0.80", "0.85", "0.90", "0.95")
REF_ARMS = ("baseline", "baseline_eqstep", "hardneg")
FAMILY_MAP = {"yolo26n": "yolo"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def iter_sched_runs(roots):
    seen = set()
    for root in roots:
        for gonogo_path in sorted(root.glob("*_seed*/gonogo.json")):
            key = gonogo_path.resolve()
            if key in seen:
                continue
            seen.add(key)
            data = read_json(gonogo_path)
            for rep in data.values():
                if isinstance(rep, dict) and rep.get("arm") == "hardneg_sched":
                    yield gonogo_path.parent, rep


def find_v1_reps(v1_root, family, seed):
    v1_family = FAMILY_MAP.get(family, family)
    path = v1_root / f"{v1_family}_seed{seed}" / "gonogo.json"
    if not path.exists():
        return path, {}
    data = read_json(path)
    reps = {}
    for rep in data.values():
        if isinstance(rep, dict) and rep.get("arm") in REF_ARMS:
            reps[rep["arm"]] = rep
    return path, reps


def same_eval_files(a, b):
    keys = ("labels_md5", "eval_calib_md5", "eval_test_md5", "calib_frac")
    return all(str(a.get(k)) == str(b.get(k)) for k in keys)


def safe_get(row, key):
    if not row:
        return ""
    value = row.get(key)
    return "" if value is None else value


def artifact_status(run_dir, sched_rep):
    arm = sched_rep.get("arm", "hardneg_sched")
    pred_meta_path = first_existing(
        [
            Path(sched_rep.get("pred_meta", "")),
            run_dir / f"{arm}.pred_meta.json",
        ]
    )
    pred_path = first_existing(
        [
            Path(sched_rep.get("pred", "")),
            run_dir / f"{arm}.csv",
        ]
    )
    weight_path = first_existing(
        [
            Path(sched_rep.get("weight", "")),
            inferred_weight_path(run_dir, arm),
        ]
    )
    results_path = first_existing(
        [
            weight_path.parents[1] / "results.csv" if len(weight_path.parents) >= 2 else Path(""),
            inferred_run_arm_dir(run_dir, arm) / "results.csv",
        ]
    )
    meta_path = run_dir / "run_meta.json"

    status = {
        "pred_exists": pred_path.exists(),
        "pred_meta_exists": pred_meta_path.exists(),
        "weight_exists": weight_path.exists(),
        "results_exists": results_path.exists(),
        "run_meta_exists": meta_path.exists(),
        "n_images": "",
        "n_rows": "",
        "results_lines": "",
        "epochs": sched_rep.get("epochs", ""),
        "val": sched_rep.get("val", ""),
        "export_conf": sched_rep.get("export_conf", ""),
        "train_count": "",
    }
    if pred_meta_path.exists():
        pred_meta = read_json(pred_meta_path)
        status["n_images"] = pred_meta.get("n_images", "")
        status["n_rows"] = pred_meta.get("n_rows", "")
    if results_path.exists():
        status["results_lines"] = sum(1 for _ in results_path.open("r", encoding="utf-8", errors="ignore"))
    counts = sched_rep.get("arm_train_counts") or {}
    status["train_count"] = counts.get("hardneg_sched", "")
    return status


def first_existing(paths):
    clean = [path for path in paths if path and str(path)]
    for path in clean:
        if path.exists():
            return path
    return clean[0] if clean else Path("")


def inferred_run_root(formal_dir):
    formal_root_name = formal_dir.parent.name
    if not formal_root_name.startswith("stage5_pv_v2"):
        return Path("")
    return ROOT / "runs" / "detect" / f"runs_{formal_root_name}"


def inferred_run_arm_dir(formal_dir, arm):
    run_root = inferred_run_root(formal_dir)
    return run_root / formal_dir.name / arm if run_root else Path("")


def inferred_weight_path(formal_dir, arm):
    return inferred_run_arm_dir(formal_dir, arm) / "weights" / "last.pt"


def build_rows(pv2_roots, v1_root):
    rows = []
    run_rows = []
    for run_dir, sched_rep in iter_sched_runs(pv2_roots):
        dataset = sched_rep.get("dataset", "dfire")
        family = sched_rep.get("family", "")
        seed = sched_rep.get("seed", "")
        v1_path, v1_reps = find_v1_reps(v1_root, family, seed)
        status = artifact_status(run_dir, sched_rep)
        run_rows.append(
            {
                "dataset": dataset,
                "family": family,
                "seed": seed,
                "pv2_source": str(run_dir),
                "v1_source": str(v1_path),
                "available_v1_refs": ",".join(sorted(v1_reps)),
                "eval_files_match_v1_hardneg": same_eval_files(sched_rep, v1_reps.get("hardneg", {})),
                **status,
            }
        )
        sched_metrics = sched_rep.get("at_fixed_recall", {})
        for target in TARGETS:
            sched_item = sched_metrics.get(target) or {}
            for ref_arm in REF_ARMS:
                ref_rep = v1_reps.get(ref_arm)
                ref_item = (ref_rep or {}).get("at_fixed_recall", {}).get(target) or {}
                rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "seed": seed,
                        "target_recall": target,
                        "arm": "hardneg_sched",
                        "reference_arm": ref_arm,
                        "arm_recall": safe_get(sched_item, "recall"),
                        "reference_recall": safe_get(ref_item, "recall"),
                        "arm_FPR": safe_get(sched_item, "FPR"),
                        "reference_FPR": safe_get(ref_item, "FPR"),
                        "delta_FPR": (
                            float(sched_item["FPR"]) - float(ref_item["FPR"])
                            if sched_item and ref_item
                            else ""
                        ),
                        "arm_FPPI": safe_get(sched_item, "FPPI"),
                        "reference_FPPI": safe_get(ref_item, "FPPI"),
                        "delta_FPPI": (
                            float(sched_item["FPPI"]) - float(ref_item["FPPI"])
                            if sched_item and ref_item
                            else ""
                        ),
                        "pv2_source": str(run_dir),
                        "v1_source": str(v1_path),
                    }
                )
    return rows, run_rows


def fmt(value):
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def write_summary(path, rows, run_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage5-PV v2 hardneg_sched vs Stage 5 v1",
        "",
        f"Available hardneg_sched runs: {len(run_rows)}",
        "",
        "This report is a single-arm ablation summary. It does not replace the strict",
        "Stage5-PV v2 all4 aggregation gate.",
        "",
        "## Artifact Check",
        "",
        "| dataset | family | seed | source | epochs | val | train count | pred rows | eval images | results lines | eval files match v1 hardneg |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for run in run_rows:
        lines.append(
            "| {dataset} | {family} | {seed} | {source} | {epochs} | {val} | {train_count} | "
            "{n_rows} | {n_images} | {results_lines} | {match} |".format(
                dataset=run["dataset"],
                family=run["family"],
                seed=run["seed"],
                source=run["pv2_source"],
                epochs=run["epochs"],
                val=run["val"],
                train_count=run["train_count"],
                n_rows=run["n_rows"],
                n_images=run["n_images"],
                results_lines=run["results_lines"],
                match=run["eval_files_match_v1_hardneg"],
            )
        )

    lines.extend(
        [
            "",
            "## R0.90 Comparison",
            "",
            "| dataset | family | seed | reference | sched recall | reference recall | sched FPR | reference FPR | delta FPR | sched FPPI | reference FPPI | delta FPPI |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [r for r in rows if r["target_recall"] == "0.90"]:
        lines.append(
            f"| {row['dataset']} | {row['family']} | {row['seed']} | {row['reference_arm']} | "
            f"{fmt(row['arm_recall'])} | {fmt(row['reference_recall'])} | "
            f"{fmt(row['arm_FPR'])} | {fmt(row['reference_FPR'])} | {fmt(row['delta_FPR'])} | "
            f"{fmt(row['arm_FPPI'])} | {fmt(row['reference_FPPI'])} | {fmt(row['delta_FPPI'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- Complete as an available single-arm run when epochs=300, val=false, results lines=301, and last.pt exists.",
            "- Use the R0.90 hardneg_sched-vs-hardneg row as the main added-arm probe signal.",
            "- Do not promote this to a final multi-seed PV v2 claim until all required seeds are merged, or the full all4 v2 matrix is run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_roots(text):
    return [ROOT / item.strip() for item in text.split(",") if item.strip()]


def main():
    ap = argparse.ArgumentParser(description="Compare hardneg_sched against Stage 5 v1 YOLO arms")
    ap.add_argument("--pv2-roots", default="formal_results/stage5_pv_v2,formal_results/stage5_pv_v2_singlearm")
    ap.add_argument("--v1-root", default="formal_results/stage5")
    ap.add_argument("--out-csv", default="formal_results/stage5_pv_v2/stage5_pv_v2_hardneg_sched_vs_v1.csv")
    ap.add_argument("--out-runs", default="formal_results/stage5_pv_v2/stage5_pv_v2_hardneg_sched_run_check.csv")
    ap.add_argument("--out-md", default="formal_results/stage5_pv_v2/HARDNEG_SCHED_SINGLEARM_SUMMARY.md")
    args = ap.parse_args()

    rows, run_rows = build_rows(parse_roots(args.pv2_roots), ROOT / args.v1_root)
    write_csv(ROOT / args.out_csv, rows)
    write_csv(ROOT / args.out_runs, run_rows)
    write_summary(ROOT / args.out_md, rows, run_rows)
    print(f"[done] sched_runs={len(run_rows)} rows={len(rows)} summary={ROOT / args.out_md}")


if __name__ == "__main__":
    main()
