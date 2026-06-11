#!/usr/bin/env python3
"""汇总 Stage 5 正式重复实验。"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARM_ORDER = ("baseline", "baseline_eqstep", "hardneg")
REQUIRED_MAIN_ARMS = ("baseline", "hardneg")
HEADLINE_TARGET = "0.90"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


FORMAL_PROTOCOL = "calibration_threshold_test_report"
FORMAL_EPOCHS = 300
FORMAL_SEEDS = (11, 22, 33)
MAX_EXPORT_CONF = 1e-3
EXPECTED_TRAIN_COUNTS = {
    "baseline": 7651,
    "baseline_eqstep": 14109,
    "hardneg": 14109,
}


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


def read_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None, [f"missing {path.name}"]
    try:
        return read_json(path), []
    except Exception as exc:
        return None, [f"cannot read {path.name}: {exc}"]


def formal_metadata_issues(path, family, seed, reps, require_eqstep):
    meta, issues = read_optional_json(path.parent / "run_meta.json")
    required_arms = set(REQUIRED_MAIN_ARMS)
    if require_eqstep:
        required_arms.add("baseline_eqstep")
    if meta is None:
        return issues

    if meta.get("family") != family:
        issues.append(f"run_meta family={meta.get('family')} expected={family}")
    if as_int(meta.get("seed")) != seed:
        issues.append(f"run_meta seed={meta.get('seed')} expected={seed}")
    if meta.get("mode") != "all":
        issues.append(f"run_meta mode={meta.get('mode')} expected=all")
    if as_int(meta.get("epochs")) != FORMAL_EPOCHS:
        issues.append(f"run_meta epochs={meta.get('epochs')} expected={FORMAL_EPOCHS}")
    if meta.get("val") is not False:
        issues.append(f"run_meta val={meta.get('val')} expected=false")
    meta_arms = set(meta.get("arms") or [])
    if not required_arms.issubset(meta_arms):
        issues.append(f"run_meta arms missing {','.join(sorted(required_arms - meta_arms))}")

    counts = meta.get("arm_train_counts") or {}
    for arm in sorted(required_arms):
        expected = EXPECTED_TRAIN_COUNTS.get(arm)
        if expected is not None and counts.get(arm) != expected:
            issues.append(f"{arm} train_count={counts.get(arm)} expected={expected}")

    exported_weights = meta.get("exported_weights") or {}
    meta_conf = as_float(meta.get("export_conf", meta.get("conf")))
    if meta_conf is None or meta_conf > MAX_EXPORT_CONF:
        issues.append(f"run_meta export_conf={meta_conf} exceeds {MAX_EXPORT_CONF}")

    reps_by_arm = {rep.get("arm"): rep for rep in reps}
    for arm in sorted(required_arms):
        rep = reps_by_arm.get(arm)
        if not rep:
            continue
        rep_weight = rep.get("weight")
        if exported_weights.get(arm) != rep_weight:
            issues.append(f"{arm} weight mismatch between gonogo and run_meta")
        if rep_weight and Path(str(rep_weight)).name.lower() != "last.pt":
            issues.append(f"{arm} weight is not last.pt under val=false: {rep_weight}")
        if rep.get("runner_mode") != "all":
            issues.append(f"{arm} runner_mode={rep.get('runner_mode')} expected=all")
        if as_int(rep.get("epochs")) != FORMAL_EPOCHS:
            issues.append(f"{arm} epochs={rep.get('epochs')} expected={FORMAL_EPOCHS}")
        if rep.get("val") is not False:
            issues.append(f"{arm} val={rep.get('val')} expected=false")
        rep_conf = as_float(rep.get("export_conf"))
        if rep_conf is None or rep_conf > MAX_EXPORT_CONF:
            issues.append(f"{arm} export_conf={rep_conf} exceeds {MAX_EXPORT_CONF}")

        pred_meta_path = rep.get("pred_meta")
        pred_meta, pred_issues = read_optional_json(pred_meta_path) if pred_meta_path else (None, ["missing pred_meta"])
        issues.extend(f"{arm} {issue}" for issue in pred_issues)
        if pred_meta:
            if pred_meta.get("pred") != rep.get("pred"):
                issues.append(f"{arm} pred_meta pred mismatch")
            if pred_meta.get("weight") != rep_weight:
                issues.append(f"{arm} pred_meta weight mismatch")
            pred_conf = as_float(pred_meta.get("export_conf"))
            if pred_conf is None or pred_conf > MAX_EXPORT_CONF:
                issues.append(f"{arm} pred_meta export_conf={pred_conf} exceeds {MAX_EXPORT_CONF}")
    return issues


def collect(result_root, allow_partial=False, require_eqstep=False, allow_legacy_protocol=False):
    rows = []
    run_status = []
    for path in sorted(Path(result_root).glob("*_seed*/gonogo.json")):
        data = read_json(path)
        family, seed_text = path.parent.name.rsplit("_seed", 1)
        seed = int(seed_text)
        reps = []
        for key, rep in data.items():
            if isinstance(rep, dict) and rep.get("arm") in ARM_ORDER:
                reps.append(rep)
        arms_present = {rep["arm"] for rep in reps}
        reachable_by_arm = {
            rep["arm"]: sorted(target for target, item in rep.get("at_fixed_recall", {}).items() if item)
            for rep in reps
        }
        protocols_present = sorted({rep.get("eval_protocol", "missing") for rep in reps})
        formal_protocol = all(rep.get("eval_protocol") == FORMAL_PROTOCOL for rep in reps)
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
        split_ok = len(split_fingerprints) == 1 and "missing" not in next(iter(split_fingerprints), "missing")
        required = set(REQUIRED_MAIN_ARMS)
        if require_eqstep:
            required.add("baseline_eqstep")
        complete = required.issubset(arms_present)
        protocol_ok = formal_protocol or allow_legacy_protocol
        metadata_issues = formal_metadata_issues(path, family, seed, reps, require_eqstep)
        metadata_ok = not metadata_issues
        headline_reachable = complete and all(HEADLINE_TARGET in reachable_by_arm.get(arm, []) for arm in required)
        protocol_included = (complete or allow_partial) and protocol_ok and split_ok and metadata_ok
        status = {
            "family": family,
            "seed": seed,
            "source": str(path),
            "has_baseline": "baseline" in arms_present,
            "has_baseline_eqstep": "baseline_eqstep" in arms_present,
            "has_hardneg": "hardneg" in arms_present,
            "eval_protocols": ",".join(protocols_present),
            "reachable_targets": ";".join(f"{arm}:{','.join(vals)}" for arm, vals in sorted(reachable_by_arm.items())),
            "has_formal_protocol": formal_protocol,
            "split_fingerprint": next(iter(split_fingerprints), "missing"),
            "has_split_fingerprint": split_ok,
            "has_formal_metadata": metadata_ok,
            "metadata_issues": "; ".join(metadata_issues),
            "headline_reachable": headline_reachable,
            "included_for_protocol": protocol_included,
            "included": protocol_included and headline_reachable,
        }
        run_status.append(status)
        if not status["included"]:
            continue
        for rep in reps:
            arm_name = rep["arm"]
            family = rep.get("family") or family
            seed = int(rep.get("seed") or seed)
            for target, item in rep["at_fixed_recall"].items():
                if not item:
                    continue
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "arm": arm_name,
                        "target_recall": target,
                        "threshold": item["thr"],
                        "calib_recall": item.get("calib_recall", item["recall"]),
                        "actual_recall": item["recall"],
                        "FPR": item["FPR"],
                        "FPPI": item["FPPI"],
                        "run_complete": complete,
                        "split_fingerprint": status["split_fingerprint"],
                        "source": str(path),
                    }
                )
    return rows, run_status


def completeness_for(rows):
    by_family_target = defaultdict(lambda: {arm: set() for arm in ARM_ORDER})
    for row in rows:
        by_family_target[(row["family"], row["target_recall"])][row["arm"]].add(row["seed"])
    out = {}
    for key, arms in by_family_target.items():
        baseline = arms["baseline"]
        baseline_eqstep = arms["baseline_eqstep"]
        hardneg = arms["hardneg"]
        paired_main = baseline & hardneg
        paired_eqstep = baseline_eqstep & hardneg
        paired_all = baseline & baseline_eqstep & hardneg
        full = baseline == baseline_eqstep == hardneg and len(paired_all) > 0
        out[key] = {
            "baseline": baseline,
            "baseline_eqstep": baseline_eqstep,
            "hardneg": hardneg,
            "paired_main": paired_main,
            "paired_eqstep": paired_eqstep,
            "paired_all": paired_all,
            "text": (
                "full"
                if full
                else (
                    f"partial baseline={len(baseline)} baseline_eqstep={len(baseline_eqstep)} "
                    f"hardneg={len(hardneg)} all3={len(paired_all)}"
                )
            ),
        }
    return out


def aggregate(rows, require_eqstep=False):
    groups = defaultdict(list)
    completeness = completeness_for(rows)
    for row in rows:
        groups[(row["family"], row["arm"], row["target_recall"])].append(row)
    out = []
    for (family, arm, target), group in sorted(groups.items()):
        comp = completeness[(family, target)]
        if require_eqstep or arm == "baseline_eqstep":
            paired_seeds = comp["paired_all"]
        else:
            paired_seeds = comp["paired_main"]
        group = [row for row in group if row["seed"] in paired_seeds]
        if not group:
            continue
        fprs = np.array([float(r["FPR"]) for r in group], dtype=float)
        fppis = np.array([float(r["FPPI"]) for r in group], dtype=float)
        recalls = np.array([float(r["actual_recall"]) for r in group], dtype=float)
        calib_recalls = np.array([float(r["calib_recall"]) for r in group], dtype=float)
        out.append(
            {
                "family": family,
                "arm": arm,
                "target_recall": target,
                "n_seeds": len(group),
                "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda x: x["seed"])),
                "paired_seeds": ",".join(str(seed) for seed in sorted(paired_seeds)),
                "completeness": completeness[(family, target)]["text"],
                "FPR_mean": float(fprs.mean()),
                "FPR_std": float(fprs.std(ddof=1)) if len(fprs) > 1 else 0.0,
                "FPPI_mean": float(fppis.mean()),
                "FPPI_std": float(fppis.std(ddof=1)) if len(fppis) > 1 else 0.0,
                "recall_mean": float(recalls.mean()),
                "recall_std": float(recalls.std(ddof=1)) if len(recalls) > 1 else 0.0,
                "calib_recall_mean": float(calib_recalls.mean()),
                "calib_recall_std": float(calib_recalls.std(ddof=1)) if len(calib_recalls) > 1 else 0.0,
            }
        )
    return out


def paired_deltas(rows):
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["family"], row["seed"], row["target_recall"])
        by_key[key][row["arm"]] = row
    out = []
    for (family, seed, target), arms in sorted(by_key.items()):
        if "hardneg" not in arms:
            continue
        hardneg = arms["hardneg"]
        for ref_arm in ("baseline", "baseline_eqstep"):
            if ref_arm not in arms or "hardneg" not in arms:
                continue
            ref = arms[ref_arm]
            fpr_delta = float(hardneg["FPR"]) - float(ref["FPR"])
            fppi_delta = float(hardneg["FPPI"]) - float(ref["FPPI"])
            out.append(
                {
                    "family": family,
                    "seed": seed,
                    "target_recall": target,
                    "reference_arm": ref_arm,
                    "reference_FPR": ref["FPR"],
                    "hardneg_FPR": hardneg["FPR"],
                    "delta_FPR": fpr_delta,
                    "reference_FPPI": ref["FPPI"],
                    "hardneg_FPPI": hardneg["FPPI"],
                    "delta_FPPI": fppi_delta,
                    "hardneg_improves_FPR": fpr_delta < 0,
                    "hardneg_improves_FPPI": fppi_delta < 0,
                }
            )
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_mean_std(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def write_summary(path, rows, agg, run_status, deltas, require_eqstep, min_seeds):
    r90 = [r for r in agg if r["target_recall"] == "0.90"]
    r90_deltas = [r for r in deltas if r["target_recall"] == "0.90"]
    incomplete = []
    for r in run_status:
        ok = r["has_baseline"] and r["has_hardneg"]
        if require_eqstep:
            ok = ok and r["has_baseline_eqstep"]
        ok = ok and r["has_formal_protocol"]
        ok = ok and r["has_split_fingerprint"]
        ok = ok and r.get("has_formal_metadata")
        ok = ok and r.get("headline_reachable")
        if not ok:
            incomplete.append(r)
    lines = [
        "# Stage 5A 随机种子重复实验汇总",
        "",
        f"**已收集 run 数**：{len(set((r['family'], r['seed']) for r in rows))}",
        f"**不完整 run 数**：{len(incomplete)}",
        f"**正式 seed 门槛**：每个模型族 `{min_seeds}` 个完整 seeds",
        f"**equal-step 要求**：`{require_eqstep}`",
        "",
        "## R0.90 汇总",
        "",
        "| 模型族 | arm | seeds | paired seeds | 完整性 | test FPR | test FPPI | test recall | calib recall |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    if not rows or incomplete:
        lines.insert(2, "**警告**：本文件包含 0 个正式纳入 run 或存在不完整 run；它不是正式主表，不能用于论文结论。")
        lines.insert(3, "")
    for row in r90:
        lines.append(
            f"| {row['family']} | {row['arm']} | {row['seeds']} | {row['paired_seeds']} | {row['completeness']} | "
            f"{fmt_mean_std(row['FPR_mean'], row['FPR_std'])} | "
            f"{fmt_mean_std(row['FPPI_mean'], row['FPPI_std'])} | "
            f"{fmt_mean_std(row['recall_mean'], row['recall_std'])} | "
            f"{fmt_mean_std(row['calib_recall_mean'], row['calib_recall_std'])} |"
        )
    lines.extend(
        [
            "",
            "## R0.90 Paired Deltas",
            "",
            "| 模型族 | seed | reference | reference FPR | hardneg FPR | ΔFPR | reference FPPI | hardneg FPPI | ΔFPPI |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in r90_deltas:
        lines.append(
            f"| {row['family']} | {row['seed']} | {row['reference_arm']} | {float(row['reference_FPR']):.4f} | "
            f"{float(row['hardneg_FPR']):.4f} | {row['delta_FPR']:.4f} | "
            f"{float(row['reference_FPPI']):.4f} | {float(row['hardneg_FPPI']):.4f} | {row['delta_FPPI']:.4f} |"
        )
    if incomplete:
        lines.extend(["", "## 不完整 Run", ""])
        for item in incomplete:
            missing = []
            if not item["has_baseline"]:
                missing.append("baseline")
            if require_eqstep and not item["has_baseline_eqstep"]:
                missing.append("baseline_eqstep")
            if not item["has_hardneg"]:
                missing.append("hardneg")
            if not item["has_formal_protocol"]:
                missing.append(f"formal eval_protocol ({item.get('eval_protocols') or 'none'})")
            if not item["has_split_fingerprint"]:
                missing.append("split fingerprint")
            if not item.get("has_formal_metadata"):
                missing.append(f"formal metadata ({item.get('metadata_issues') or 'unknown issue'})")
            if not item.get("headline_reachable"):
                missing.append(f"reachable {HEADLINE_TARGET} for all required arms")
            lines.append(f"- `{item['family']}_seed{item['seed']}` 缺少 {', '.join(missing)}，默认未纳入正式汇总。")
    lines.extend(
        [
            "",
            "## 判读规则",
            "",
            "- 如果同一模型族内 hardneg 的 R0.90 test FPR/FPPI 均值稳定低于 baseline，则 C2 得到重复实验支持。",
            "- 如果 hardneg 也低于 baseline_eqstep，则硬负样本收益不能只归因于更多 optimizer steps。",
            "- 同时查看 paired deltas；如果某个 seed 的 hardneg 反而升高，正文必须如实报告，不能只报均值。",
            f"- 如果 seed 数小于 {min_seeds}，只能作为补充证据，不能写成正式 3-seed 主结论。",
            "- 默认只纳入 baseline、baseline_eqstep 与 hardneg 都完成的 run；若某个 family 缺 arm，需要补齐后再写跨 arm 结论。",
            f"- 默认只纳入 `eval_protocol={FORMAL_PROTOCOL}` 的 run；旧 in-sample 结果不能写入正式汇总。",
            "- 默认只纳入带有同一 full/calibration/test split fingerprint 的 run；跨机器 split 不一致时 strict 汇总会失败。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_main_outputs(result_root):
    for name in (
        "STAGE5_REPEAT_SUMMARY.md",
        "stage5_repeat_metrics.csv",
        "stage5_repeat_aggregate.csv",
        "stage5_paired_deltas.csv",
    ):
        path = result_root / name
        if path.exists():
            path.unlink()


def write_failure_summary(path, issues):
    lines = [
        "# Stage 5A 随机种子重复实验汇总失败",
        "",
        "严格汇总未通过；本文件不是正式主表，不能用于论文结论。",
        "",
        "## 阻断问题",
        "",
    ]
    for issue in issues:
        lines.append(f"- {issue}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strict_issues(run_status, rows, required_families, min_seeds, require_eqstep, required_seeds, allow_extra_seeds):
    issues = []
    by_family = defaultdict(set)
    for item in run_status:
        ok = item["has_baseline"] and item["has_hardneg"]
        if require_eqstep:
            ok = ok and item["has_baseline_eqstep"]
        ok = ok and item["has_formal_protocol"]
        ok = ok and item["has_split_fingerprint"]
        ok = ok and item.get("has_formal_metadata")
        ok = ok and item.get("headline_reachable")
        if ok:
            by_family[item["family"]].add(item["seed"])
    fingerprints = {
        item["split_fingerprint"]
        for item in run_status
        if item["included"] and item.get("has_split_fingerprint")
    }
    if len(fingerprints) > 1:
        issues.append(f"split fingerprint mismatch across included runs: {len(fingerprints)} variants")
    completeness = completeness_for(rows)
    for family in required_families:
        seeds = by_family.get(family, set())
        expected = set(required_seeds)
        if expected:
            missing = expected - seeds
            extra = seeds - expected
            if missing:
                issues.append(f"{family}: missing required seeds {','.join(map(str, sorted(missing)))}")
            if extra and not allow_extra_seeds:
                issues.append(f"{family}: unexpected formal seeds {','.join(map(str, sorted(extra)))}")
        if len(seeds) < min_seeds:
            issues.append(f"{family}: complete seeds {len(seeds)}/{min_seeds} ({','.join(map(str, sorted(seeds))) or 'none'})")
        headline = completeness.get((family, HEADLINE_TARGET))
        if headline:
            reachable = headline["paired_all"] if require_eqstep else headline["paired_main"]
        else:
            reachable = set()
        if len(reachable) < min_seeds:
            issues.append(
                f"{family}: reachable paired {HEADLINE_TARGET} seeds "
                f"{len(reachable)}/{min_seeds} ({','.join(map(str, sorted(reachable))) or 'none'})"
            )
    return issues


def main():
    ap = argparse.ArgumentParser(description="汇总 Stage 5 正式重复实验")
    ap.add_argument("--result-root", default="formal_results/stage5")
    ap.add_argument("--allow-partial", action="store_true", help="允许把缺 baseline/hardneg 的 run 纳入汇总")
    ap.add_argument("--allow-legacy-protocol", action="store_true", help="允许旧 in-sample eval_protocol 纳入汇总；正式论文不要使用")
    ap.add_argument("--allow-empty", action="store_true", help="允许在没有 gonogo.json 时写出空汇总")
    ap.add_argument("--require-eqstep", action="store_true", help="要求 baseline_eqstep arm 完整后才纳入正式汇总")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--required-seeds", default="11,22,33")
    ap.add_argument("--allow-extra-seeds", action="store_true")
    ap.add_argument("--required-families", default="yolo,rtdetr")
    ap.add_argument("--strict", action="store_true", help="正式矩阵不满足时返回非零状态")
    args = ap.parse_args()
    result_root = ROOT / args.result_root
    rows, run_status = collect(result_root, args.allow_partial, args.require_eqstep, args.allow_legacy_protocol)
    write_csv(result_root / "stage5_run_status.csv", run_status)
    if not rows and not args.allow_empty:
        if run_status:
            issue = (
                f"找到 {len(run_status)} 个 run，但没有任何可达固定召回 operating point；"
                "请检查 gonogo.json 中 at_fixed_recall 是否全为 null。"
            )
            issues = [issue]
            for item in run_status:
                reasons = []
                if not item.get("has_formal_metadata"):
                    reasons.append(f"formal metadata: {item.get('metadata_issues') or 'unknown issue'}")
                if not item.get("headline_reachable"):
                    reasons.append(f"missing reachable {HEADLINE_TARGET} for all required arms")
                if reasons:
                    issues.append(f"{item['family']}_seed{item['seed']}: " + " | ".join(reasons))
            clear_main_outputs(result_root)
            write_failure_summary(result_root / "STAGE5_REPEAT_SUMMARY.md", issues)
            raise SystemExit(
                f"[错误] {issue}"
            )
        raise SystemExit(f"[错误] 没有找到 Stage 5 run：{result_root}\\*_seed*\\gonogo.json")
    incomplete = []
    for r in run_status:
        ok = r["has_baseline"] and r["has_hardneg"]
        if args.require_eqstep:
            ok = ok and r["has_baseline_eqstep"]
        ok = ok and r["has_formal_protocol"]
        ok = ok and r["has_split_fingerprint"]
        ok = ok and r.get("has_formal_metadata")
        ok = ok and r.get("headline_reachable")
        if not ok:
            incomplete.append(r)
    if incomplete and not args.allow_partial:
        print(f"[warn] excluded incomplete runs={len(incomplete)}; see stage5_run_status.csv")
    required_families = [x.strip() for x in args.required_families.split(",") if x.strip()]
    required_seeds = [int(x.strip()) for x in args.required_seeds.split(",") if x.strip()]
    issues = strict_issues(
        run_status,
        rows,
        required_families,
        args.min_seeds,
        args.require_eqstep,
        required_seeds,
        args.allow_extra_seeds,
    )
    for issue in issues:
        print(f"[warn] formal matrix incomplete: {issue}")
    if args.strict and issues:
        clear_main_outputs(result_root)
        write_failure_summary(result_root / "STAGE5_REPEAT_SUMMARY.md", issues)
        raise SystemExit(2)
    agg = aggregate(rows, args.require_eqstep)
    deltas = paired_deltas(rows)
    write_csv(result_root / "stage5_repeat_metrics.csv", rows)
    write_csv(result_root / "stage5_repeat_aggregate.csv", agg)
    write_csv(result_root / "stage5_paired_deltas.csv", deltas)
    write_summary(result_root / "STAGE5_REPEAT_SUMMARY.md", rows, agg, run_status, deltas, args.require_eqstep, args.min_seeds)
    print(f"[done] runs={len(set((r['family'], r['seed']) for r in rows))} rows={len(rows)}")
    print(f"[done] wrote {result_root / 'STAGE5_REPEAT_SUMMARY.md'}")


if __name__ == "__main__":
    main()
