#!/usr/bin/env python3
"""Aggregate Stage5-PV v2 runs with dataset-aware strict gates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARM_ORDER = ("baseline", "baseline_eqstep", "hardneg", "hardneg_sched")
FORMAL_PROTOCOL = "calibration_threshold_test_report"
MAX_EXPORT_CONF = 1e-3
HEADLINE_TARGET = "0.90"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def as_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def required_arms(require_eqstep, require_sched):
    arms = {"baseline", "hardneg"}
    if require_eqstep:
        arms.add("baseline_eqstep")
    if require_sched:
        arms.add("hardneg_sched")
    return arms


def read_optional_json(path):
    if not path or not Path(path).exists():
        return None, [f"missing {Path(path).name if path else 'pred_meta'}"]
    try:
        return read_json(path), []
    except Exception as exc:
        return None, [f"cannot read {Path(path).name}: {exc}"]


def metadata_issues(run_dir, dataset, family, seed, reps, req_arms, formal_epochs):
    meta, issues = read_optional_json(run_dir / "run_meta.json")
    if meta is None:
        return issues
    if meta.get("dataset") != dataset:
        issues.append(f"run_meta dataset={meta.get('dataset')} expected={dataset}")
    if meta.get("family") != family:
        issues.append(f"run_meta family={meta.get('family')} expected={family}")
    if as_int(meta.get("seed")) != seed:
        issues.append(f"run_meta seed={meta.get('seed')} expected={seed}")
    if meta.get("mode") != "all":
        issues.append(f"run_meta mode={meta.get('mode')} expected=all")
    if as_int(meta.get("epochs")) != formal_epochs:
        issues.append(f"run_meta epochs={meta.get('epochs')} expected={formal_epochs}")
    if meta.get("val") is not False:
        issues.append(f"run_meta val={meta.get('val')} expected=false")
    if not req_arms.issubset(set(meta.get("arms") or [])):
        issues.append(f"run_meta arms missing {','.join(sorted(req_arms - set(meta.get('arms') or [])))}")

    counts = meta.get("arm_train_counts") or {}
    if "baseline_eqstep" in req_arms and counts.get("baseline_eqstep") != counts.get("hardneg"):
        issues.append("baseline_eqstep train_count must equal hardneg train_count")
    if "hardneg_sched" in req_arms and counts.get("hardneg_sched") != counts.get("hardneg"):
        issues.append("hardneg_sched train_count must equal hardneg train_count")

    reps_by_arm = {rep.get("arm"): rep for rep in reps}
    for arm in sorted(req_arms):
        rep = reps_by_arm.get(arm)
        if not rep:
            continue
        if rep.get("dataset") != dataset:
            issues.append(f"{arm} dataset={rep.get('dataset')} expected={dataset}")
        if rep.get("runner_mode") != "all":
            issues.append(f"{arm} runner_mode={rep.get('runner_mode')} expected=all")
        if as_int(rep.get("epochs")) != formal_epochs:
            issues.append(f"{arm} epochs={rep.get('epochs')} expected={formal_epochs}")
        if rep.get("val") is not False:
            issues.append(f"{arm} val={rep.get('val')} expected=false")
        rep_conf = as_float(rep.get("export_conf"))
        if rep_conf is None or rep_conf > MAX_EXPORT_CONF:
            issues.append(f"{arm} export_conf={rep_conf} exceeds {MAX_EXPORT_CONF}")
        weight = rep.get("weight")
        if weight and Path(str(weight)).name.lower() != "last.pt":
            issues.append(f"{arm} weight is not last.pt: {weight}")
        pred_meta, pred_issues = read_optional_json(rep.get("pred_meta"))
        issues.extend(f"{arm} {issue}" for issue in pred_issues)
        if pred_meta:
            if pred_meta.get("dataset") != dataset:
                issues.append(f"{arm} pred_meta dataset mismatch")
            if pred_meta.get("pred") != rep.get("pred"):
                issues.append(f"{arm} pred_meta pred mismatch")
            if pred_meta.get("weight") != weight:
                issues.append(f"{arm} pred_meta weight mismatch")
    return issues


def collect(result_root, req_arms, formal_epochs):
    rows = []
    statuses = []
    for path in sorted(Path(result_root).glob("*_seed*/gonogo.json")):
        data = read_json(path)
        reps = [rep for rep in data.values() if isinstance(rep, dict) and rep.get("arm") in ARM_ORDER]
        if not reps:
            continue
        dataset = reps[0].get("dataset") or path.parent.name.split("_", 1)[0]
        family = reps[0].get("family") or "unknown"
        seed = as_int(reps[0].get("seed"), -1)
        arms = {rep["arm"] for rep in reps}
        protocols = {rep.get("eval_protocol", "missing") for rep in reps}
        split_fingerprints = {
            "|".join(
                [
                    str(rep.get("labels_md5", "missing")),
                    str(rep.get("eval_calib_md5", "missing")),
                    str(rep.get("eval_test_md5", "missing")),
                    str(rep.get("eval_split_salt", "missing")),
                    str(rep.get("calib_frac", "missing")),
                ]
            )
            for rep in reps
        }
        reachable = {
            rep["arm"]: sorted(target for target, item in rep.get("at_fixed_recall", {}).items() if item)
            for rep in reps
        }
        meta_issues = metadata_issues(path.parent, dataset, family, seed, reps, req_arms, formal_epochs)
        complete = req_arms.issubset(arms)
        formal = protocols == {FORMAL_PROTOCOL}
        split_ok = len(split_fingerprints) == 1 and "missing" not in next(iter(split_fingerprints), "missing")
        headline = complete and all(HEADLINE_TARGET in reachable.get(arm, []) for arm in req_arms)
        included = complete and formal and split_ok and not meta_issues and headline
        status = {
            "dataset": dataset,
            "family": family,
            "seed": seed,
            "source": str(path),
            "arms": ",".join(sorted(arms)),
            "eval_protocols": ",".join(sorted(protocols)),
            "split_fingerprint": next(iter(split_fingerprints), "missing"),
            "complete": complete,
            "headline_reachable": headline,
            "metadata_ok": not meta_issues,
            "metadata_issues": "; ".join(meta_issues),
            "included": included,
        }
        statuses.append(status)
        if not included:
            continue
        for rep in reps:
            for target, item in rep.get("at_fixed_recall", {}).items():
                if not item:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "seed": seed,
                        "arm": rep["arm"],
                        "target_recall": target,
                        "threshold": item["thr"],
                        "calib_recall": item.get("calib_recall", item["recall"]),
                        "actual_recall": item["recall"],
                        "FPR": item["FPR"],
                        "FPPI": item["FPPI"],
                        "split_fingerprint": status["split_fingerprint"],
                        "source": str(path),
                    }
                )
    return rows, statuses


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["family"], row["arm"], row["target_recall"])].append(row)
    out = []
    for (dataset, family, arm, target), group in sorted(groups.items()):
        fprs = np.array([float(r["FPR"]) for r in group])
        fppis = np.array([float(r["FPPI"]) for r in group])
        recalls = np.array([float(r["actual_recall"]) for r in group])
        out.append(
            {
                "dataset": dataset,
                "family": family,
                "arm": arm,
                "target_recall": target,
                "n_seeds": len(group),
                "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda x: x["seed"])),
                "FPR_mean": float(fprs.mean()),
                "FPR_std": float(fprs.std(ddof=1)) if len(fprs) > 1 else 0.0,
                "FPPI_mean": float(fppis.mean()),
                "FPPI_std": float(fppis.std(ddof=1)) if len(fppis) > 1 else 0.0,
                "recall_mean": float(recalls.mean()),
                "recall_std": float(recalls.std(ddof=1)) if len(recalls) > 1 else 0.0,
            }
        )
    return out


def paired_deltas(rows):
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["dataset"], row["family"], row["seed"], row["target_recall"])][row["arm"]] = row
    out = []
    comparisons = (("hardneg", "baseline"), ("hardneg", "baseline_eqstep"), ("hardneg_sched", "hardneg"))
    for (dataset, family, seed, target), arms in sorted(by_key.items()):
        for arm, ref_arm in comparisons:
            if arm not in arms or ref_arm not in arms:
                continue
            cur = arms[arm]
            ref = arms[ref_arm]
            out.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "seed": seed,
                    "target_recall": target,
                    "arm": arm,
                    "reference_arm": ref_arm,
                    "delta_FPR": float(cur["FPR"]) - float(ref["FPR"]),
                    "delta_FPPI": float(cur["FPPI"]) - float(ref["FPPI"]),
                    "arm_FPR": cur["FPR"],
                    "reference_FPR": ref["FPR"],
                    "arm_FPPI": cur["FPPI"],
                    "reference_FPPI": ref["FPPI"],
                }
            )
    return out


def strict_issues(statuses, required_datasets, required_families, required_seeds, min_seeds):
    issues = []
    included = [s for s in statuses if s["included"]]
    for dataset in required_datasets:
        fps = {s["split_fingerprint"] for s in included if s["dataset"] == dataset}
        if len(fps) > 1:
            issues.append(f"{dataset}: split fingerprint mismatch ({len(fps)} variants)")
        for family in required_families:
            seeds = {s["seed"] for s in included if s["dataset"] == dataset and s["family"] == family}
            missing = set(required_seeds) - seeds
            if missing:
                issues.append(f"{dataset}/{family}: missing seeds {','.join(map(str, sorted(missing)))}")
            if len(seeds) < min_seeds:
                issues.append(f"{dataset}/{family}: complete seeds {len(seeds)}/{min_seeds}")
    return issues


def write_summary(path, agg, statuses, deltas, issues):
    lines = [
        "# Stage5-PV v2 Summary",
        "",
        f"**included runs**: {sum(1 for s in statuses if s['included'])}",
        f"**excluded/incomplete runs**: {sum(1 for s in statuses if not s['included'])}",
        "",
    ]
    if issues:
        lines.extend(["## Strict Issues", ""])
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")
    lines.extend(
        [
            "## R0.90 Summary",
            "",
            "| dataset | family | arm | seeds | FPR | FPPI | recall |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in [r for r in agg if r["target_recall"] == HEADLINE_TARGET]:
        lines.append(
            f"| {row['dataset']} | {row['family']} | {row['arm']} | {row['seeds']} | "
            f"{row['FPR_mean']:.4f} ± {row['FPR_std']:.4f} | "
            f"{row['FPPI_mean']:.4f} ± {row['FPPI_std']:.4f} | "
            f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f} |"
        )
    lines.extend(["", "## R0.90 Paired Deltas", ""])
    lines.extend(["| dataset | family | seed | arm | reference | delta FPR | delta FPPI |", "|---|---|---:|---|---|---:|---:|"])
    for row in [r for r in deltas if r["target_recall"] == HEADLINE_TARGET]:
        lines.append(
            f"| {row['dataset']} | {row['family']} | {row['seed']} | {row['arm']} | "
            f"{row['reference_arm']} | {row['delta_FPR']:.4f} | {row['delta_FPPI']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Aggregate Stage5-PV v2")
    ap.add_argument("--result-root", default="formal_results/stage5_pv_v2")
    ap.add_argument("--require-eqstep", action="store_true")
    ap.add_argument("--require-sched", action="store_true")
    ap.add_argument("--formal-epochs", type=int, default=300)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--required-seeds", default="11,22,33")
    ap.add_argument("--required-datasets", default="dfire,dfs")
    ap.add_argument("--required-families", default="yolo26n,rtdetr")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    root = ROOT / args.result_root
    req_arms = required_arms(args.require_eqstep, args.require_sched)
    rows, statuses = collect(root, req_arms, args.formal_epochs)
    agg = aggregate(rows)
    deltas = paired_deltas(rows)
    write_csv(root / "stage5_pv_v2_run_status.csv", statuses)
    write_csv(root / "stage5_pv_v2_metrics.csv", rows)
    write_csv(root / "stage5_pv_v2_aggregate.csv", agg)
    write_csv(root / "stage5_pv_v2_paired_deltas.csv", deltas)
    issues = strict_issues(
        statuses,
        [x.strip() for x in args.required_datasets.split(",") if x.strip()],
        [x.strip() for x in args.required_families.split(",") if x.strip()],
        [int(x.strip()) for x in args.required_seeds.split(",") if x.strip()],
        args.min_seeds,
    )
    write_summary(root / "STAGE5_PV_V2_SUMMARY.md", agg, statuses, deltas, issues)
    if args.strict and issues:
        raise SystemExit(2)
    print(f"[done] rows={len(rows)} summary={root / 'STAGE5_PV_V2_SUMMARY.md'}")


if __name__ == "__main__":
    main()
