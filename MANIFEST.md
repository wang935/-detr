# 输出清单

**当前日期**：2026-06-01  
**工作目录**：`D:\detr_Q3`  
**当前论文主线**：面向火焰/烟雾报警的固定召回误报率控制

说明：早期 `20260530` 和 `20260601-103000` 时间戳文件保留为历史快照；当前写作和后续实验以本清单中“当前主线”文件为准。`paper_assets/` 中的 Stage 4 主表和图只能作为历史探索证据，正式论文 claim 必须等待 Stage 5 strict 汇总后重算。

## 当前主线材料

| 文件 | 用途 |
|---|---|
| `refine-logs/FINAL_PROPOSAL.md` | 中文最终方案：把论文主线从 DAQ-DETR 主方法转为“固定召回误报控制 + 硬负样本训练 + DETR query 分析”。 |
| `refine-logs/FINAL_PROPOSAL_20260601_ALARM_CONTROL_PIVOT.md` | 同一最终方案的时间戳版本，便于追踪转向节点。 |
| `refine-logs/RESULT_TO_CLAIM_20260601.md` | 中文结果到论文主张判断：Stage 4 只作为探索性证据，正式 claim 等 Stage 5 重跑。 |
| `refine-logs/EXPERIMENT_PLAN.md` | 中文实验计划：Stage 4/5 的表格、定性图、重复实验和写作路线。 |
| `refine-logs/EXPERIMENT_PLAN_20260601_ALARM_CONTROL_PIVOT.md` | 同一实验计划的时间戳版本。 |
| `refine-logs/EXPERIMENT_TRACKER.md` | 中文实验追踪表：记录 Tier B、Stage 2、Stage 3C/3D、Stage 4A/4B 的状态。 |
| `refine-logs/STAGE5_CLAUDE_REVIEW_SUMMARY.md` | Stage 5 Claude Code 审查总结；新版 Round 5 因 401 未完成。 |
| `refine-logs/FORMAL_EXPERIMENT_DESIGN.md` | Stage 5 正式实验设计：每族 3 seeds、300 epochs、equal-step 控制、DAQ 消融、固定召回指标、统计口径和写作边界。 |
| `refine-logs/EXPERIMENT_BRIDGE_HANDOFF.md` | Stage 5 从实验设计到正式运行的本地 bridge handoff：实现状态、sanity、运行顺序和结果检查清单。 |
| `refine-logs/EXTERNAL_NEGATIVE_SOURCE_PLAN_20260602.md` | 第二外部负样本源计划：选择 BoWFire non-fire / fire-like negatives，并规定 Stage 5 后的 external FPR/FPPI 评估边界。 |
| `stage3_daq/STAGE3C_3D_SUMMARY.md` | 中文 Stage 3C/3D 结果总结：外部中性负样本压力测试与 YOLO 基线结果。 |
| `stage3_daq/external_deepquest/STAGE3_EXTERNAL_DISTRACTOR.md` | 中文外部中性负样本压力测试报告：说明 DAQ 为什么不能作为当前主贡献。 |
| `paper_assets/STAGE4A_CLAIM_SUMMARY.md` | 中文 Stage 4A 论文主张摘要；当前仅作历史探索草稿，不能直接用于正式 claim。 |
| `paper_assets/TABLE_MAIN_ALARM_METRICS.md` | 中文 Stage 4 探索主表：固定召回下 FPR/FPPI 对比；正式主表需由 Stage 5 重算。 |
| `paper_assets/TABLE_EXTERNAL_NEUTRAL.md` | 中文外部中性负样本表：外部干扰物误报表现。 |
| `paper_assets/fig_r090_fpr_fppi.png` | R0.90 下 FPR/FPPI 历史探索图；正式图需由 Stage 5 重算。 |
| `paper_assets/qualitative/STAGE4B_QUALITATIVE_SUMMARY.md` | 中文 Stage 4B 定性误报图组摘要。 |
| `paper_assets/qualitative/contact_all.jpg` | 四类定性误报样例总拼图。 |
| `paper_assets/EVIDENCE_PACKAGE_AUDIT.md` | 中文证据包自动审计报告，当前状态为通过。 |

## 当前主线脚本

| 文件 | 用途 |
|---|---|
| `scripts/stage4_paper_assets.py` | 生成中文 Stage 4A 表格、摘要和 R0.90 图。 |
| `scripts/stage4b_qualitative.py` | 生成中文 Stage 4B 定性误报样例、CSV 清单和拼图。 |
| `scripts/audit_evidence_package.py` | 自动审计证据包完整性、数值一致性和文档状态。 |
| `scripts/stage5_formal_runner.py` | Stage 5 正式随机种子重复实验 runner，支持 baseline/baseline_eqstep/hardneg 三臂。 |
| `scripts/stage5_aggregate_repeats.py` | 汇总 Stage 5 多 seed 结果并生成均值/标准差表与 paired delta。 |
| `scripts/stage5_external_negatives.py` | Stage 5 strict 汇总后，对 BoWFire 等外部 negative-only 图像源复用 D-Fire 阈值并报告 external FPR/FPPI。 |
| `scripts/stage3_external_distractor.py` | Stage 3C 外部中性负样本/干扰物压力测试。 |
| `scripts/stage3_yolo_baseline.py` | Stage 3D YOLO26n 基线训练、导出和评估。 |
| `scripts/stage3_multisplit.py` | Stage 3A 多划分 DAQ 稳定性实验。 |
| `scripts/stage3_budget_ablation.py` | Stage 3B query budget 消融实验。 |

## 关键结果位置

| 文件或目录 | 内容 |
|---|---|
| `gonogo_clean5.json` | RT-DETR 基线与硬负样本训练的固定召回误报评估。 |
| `stage2_daq/strong/` | Stage 2 强化版 DAQ 的 D-Fire holdout 结果。 |
| `stage3_daq/external_deepquest/stage3_external_summary.json` | Stage 3C 外部中性负样本评估结果。 |
| `preds_stage3_yolo/yolo_gonogo.json` | Stage 3D YOLO26n 基线与硬负样本训练结果。 |
| `runs/detect/runs_stage3_yolo/` | YOLO26n 训练输出。 |

## 3060 运行材料

| 文件或目录 | 用途 |
|---|---|
| `stage3_3060/README_CMD.md` | 中文 3 台 RTX 3060 分工运行说明。 |
| `stage3_3060/*.bat` | 每台 3060 可直接执行的 Windows 启动脚本。 |
| `STAGE3_3060_COMMANDS.md` | 根目录下的 Stage 3 命令速查。 |
| `stage5_3060/README_CMD.md` | 中文 Stage 5 正式实验三台 3060 启动说明。 |
| `stage5_3060/TEST_README_CMD.md` | Stage 5 正式训练前的最小环境测试说明。 |
| `stage5_3060/00_create_or_check_env.bat` | 每台 3060 的 Stage 5 conda 环境创建/检查入口。 |
| `stage5_3060/TEST_*.bat` | Stage 5 mincheck 测试脚本，输出到独立测试目录。 |
| `stage5_3060/100_external_bowfire.bat` | Stage 5 strict 汇总后运行的 BoWFire 第二外部负样本源压力测试脚本。 |
| `stage5_3060/PC3b_rtdetr_seed33.bat` | RT-DETR-L seed 33 的 Stage 5 正式运行脚本。 |
| `refine-logs/CLAUDE_STAGE5_REVIEW_ROUND1.md` | Claude Code 第 1 轮 Stage 5 审查记录。 |
| `refine-logs/CLAUDE_STAGE5_REVIEW_ROUND2.md` | Claude Code 第 2 轮 Stage 5 审查记录。 |
| `refine-logs/CLAUDE_STAGE5_REVIEW_ROUND3.md` | Claude Code 第 3 轮 Stage 5 审查记录。 |
| `refine-logs/CLAUDE_STAGE5_REVIEW_ROUND4.md` | Claude Code 最终确认记录。 |
| `refine-logs/CLAUDE_STAGE5_REVIEW_ROUND5.md` | 新版 3-seed/300-epoch/equal-step 矩阵的 Claude 审查请求与 401 失败记录。 |
| `data/D-Fire/` | 用于便携运行的 D-Fire 本地镜像。 |
| `data/dfire_local/` | 使用本地路径的数据配置。 |
| `data/dfire_local/dfire_posonly_equalstep.yaml` | Stage 5 equal-step 控制数据配置，由预检/runner 自动生成。 |
| `data/dfire_local/train_posonly_equalstep.txt` | Stage 5 equal-step 正样本重复列表，长度与 hardneg train 相同。 |

## 历史材料

| 文件或目录 | 说明 |
|---|---|
| `idea-stage/` | 早期 idea-discovery 报告和候选方向。当前只作历史参考。 |
| `outputs/` | 早期用户版报告输出。当前论文主线以 `refine-logs/` 和 `paper_assets/` 为准。 |
| `MANIFEST_20260530-205415.md` | 最早输出清单历史快照。 |
| `PILOT_QUERY_README.md` | Tier A / query probe 的早期说明。 |
| `PROJECT_HANDOFF.md` | 早期交接说明。 |
| `SETUP_3060_TIER_A.md` | Tier A 3060 环境说明。 |
# 2026-06-03 Stage 5 five-host launch update

| File or directory | Purpose |
|---|---|
| `STAGE5_5HOST_COMMANDS.md` | Quick command sheet for 3 x RTX 3060 plus 2 x RTX 3080 Ti. |
| `stage5_5hosts/README_CMD.md` | Full five-host Stage 5 launch instructions, mincheck flow, merge, and aggregation steps. |
| `stage5_5hosts/00_create_or_check_env.bat` | Per-host environment bootstrap/check; installs Miniconda if missing, creates conda env `daq`, installs CUDA PyTorch cu121 and Stage 5 dependencies, and verifies CUDA. |
| `stage5_5hosts/00_check_stage5_assets.bat` | Per-host code, CUDA, weight, dataset, and Stage 5 smoke precheck. |
| `stage5_5hosts/TEST_*.bat` | Per-host 1-epoch mincheck scripts writing only to `formal_results/stage5_mincheck` and `runs/detect/runs_stage5_mincheck`. |
| `stage5_5hosts/H*.bat` | Formal five-host training scripts: three YOLO seeds on RTX 3060 and three RT-DETR seeds on RTX 3080 Ti. |
| `stage5_3060_mirror_env/` | Mirror-based setup/mincheck/formal wrappers for Ye Hailong H02 and Sun Hao H03 RTX 3060 hosts. |
| `stage5_5hosts/99_aggregate_stage5.bat` | Strict Stage 5 formal result aggregation for seeds `11,22,33`. |
| `refine-logs/STAGE5_5HOST_TRAINING_PLAN_20260603.md` | Updated training allocation and environment policy for the five-host hardware configuration. |
| `stage5_h20_server/run_stage5_pv_v2_full_queue.sh` | H20 full Stage5-PV v2 queue for `dfire,dfs x yolo26n,rtdetr x seeds 11,22,33 x all4 arms x 300 epochs`; fails early until DFS is prepared. |

# 2026-06-07 Fixed-Recall Q1 idea-discovery update

| File or directory | Purpose |
|---|---|
| `idea-stage/FIXED_RECALL_Q1_IDEA_REPORT_20260607-1324.md` | Timestamped main idea-discovery report for SCI-Q1 fixed-recall operating-regime scenarios. |
| `idea-stage/FIXED_RECALL_Q1_IDEA_REPORT.md` | Fixed-name main report; canonical current view of the 5 refined candidates. |
| `idea-stage/FIXED_RECALL_Q1_IDEA_REPORT.html` | Rendered single-file HTML view of the main report. |
| `idea-stage/FIXED_RECALL_Q1_IDEA_CANDIDATES_20260607-1324.md` | Timestamped detailed candidate file with fields, risks, counter-experiments, scores, and go/no-go rules. |
| `idea-stage/FIXED_RECALL_Q1_IDEA_CANDIDATES.md` | Fixed-name detailed candidate file. |
| `idea-stage/FIXED_RECALL_Q1_DATASET_MATRIX_20260607-1324.md` | Timestamped public dataset matrix for the 5 candidates. |
| `idea-stage/FIXED_RECALL_Q1_DATASET_MATRIX.md` | Fixed-name public dataset matrix. |
| `idea-stage/FIXED_RECALL_Q1_CLAUDE_REVIEW_20260607-1324.md` | Timestamped Claude Opus 4.8 two-round review and adoption record. |
| `idea-stage/FIXED_RECALL_Q1_CLAUDE_REVIEW.md` | Fixed-name Claude review and adoption record. |

# 2026-06-07 Fixed-Recall Q1 top-2 continuation

| File or directory | Purpose |
|---|---|
| `idea-stage/FIXED_RECALL_Q1_TOP2_EXPERIMENT_PLAN_20260607-1345.md` | Timestamped top-2 experiment plan for RACO-Wildfire and XRayThreat-RC, revised after Claude review. |
| `idea-stage/FIXED_RECALL_Q1_TOP2_EXPERIMENT_PLAN.md` | Fixed-name top-2 experiment plan; current version emphasizes denominator audits before training. |
| `idea-stage/FIXED_RECALL_Q1_TOP2_EXPERIMENT_PLAN.html` | Rendered HTML view of the top-2 experiment plan. |
| `idea-stage/FIXED_RECALL_Q1_TOP2_DECISION_BRIEF_20260607-1345.md` | Timestamped decision brief revised after Claude review. |
| `idea-stage/FIXED_RECALL_Q1_TOP2_DECISION_BRIEF.md` | Fixed-name decision brief; no primary Q1 paper selected until denominator audits pass. |
| `idea-stage/FIXED_RECALL_Q1_TOP2_DECISION_BRIEF.html` | Rendered HTML view of the decision brief. |
| `idea-stage/FIXED_RECALL_Q1_TOP2_CLAUDE_REVIEW_20260607-1345.md` | Timestamped Claude Opus 4.8 top-2 review record. |
| `idea-stage/FIXED_RECALL_Q1_TOP2_CLAUDE_REVIEW.md` | Fixed-name top-2 Claude review and adoption record. |
| `scripts/fixed_recall_eval_general.py` | Generic fixed-recall evaluator supporting arbitrary positive/negative labels and optional event/bag/scan/procedure unit columns. |
| `scripts/fixed_recall_label_audit.py` | Label/unit/split audit tool for fixed-recall dataset denominator checks and CI sanity. |
| `idea-stage/fixed_recall_general_dfire_smoketest.json` | D-Fire smoke-test output for the generic evaluator. |
| `idea-stage/fixed_recall_general_dfire_smoketest.csv` | CSV summary of the generic evaluator D-Fire smoke test. |
| `idea-stage/fixed_recall_label_audit_dfire.json` | D-Fire label audit JSON output. |
| `idea-stage/fixed_recall_label_audit_dfire.md` | D-Fire label audit Markdown output. |

# 2026-06-07 Stage5 third-model allocation (restored to YOLO26n)

| File or directory | Purpose |
|---|---|
| `stage5_5hosts/README_CMD.md` | Three 3060/2x3080Ti five-host runbook: YOLO26n seeds 11,22,33 and RT-DETR-L seeds 11,22 are active in the main plan. |
| `refine-logs/STAGE5_INTERNIMAGE_DFIRE_ALLOCATION_20260607.md` | Archive roll-back note showing third-model restoration: InternImage disabled, active seeds are H02=22, H03=33 (H01=11). |
| `stage5_internimage/README_CMD.md` | InternImage D-Fire command sheet is archived and marked as deprecated; do not use for active Stage 5 allocation. |
| `refine-logs/STAGE5_5HOST_TRAINING_PLAN_20260603.md` | Host mapping table for active five-host Stage 5 run (H02->seed22 YOLO, H03->seed33 YOLO; RT-DETR-L on H04/H05). |

# 2026-06-07 Fixed-Recall Q1 denominator audit

| File or directory | Purpose |
|---|---|
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_AUDIT_20260607-1420.md` | Timestamped denominator audit for XRayThreat-RC and RACO-Wildfire data feasibility. |
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_AUDIT.md` | Fixed-name denominator audit; current source for no-training/data-denominator decision. |
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_AUDIT.html` | Rendered HTML view of the denominator audit. |
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_CLAUDE_REVIEW_20260607-1420.md` | Timestamped Claude Opus 4.8 review of the denominator audit. |
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_CLAUDE_REVIEW.md` | Fixed-name Claude review and adoption record for denominator audit. |
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_CLAUDE_REVIEW.html` | Rendered HTML view of the denominator Claude review. |
| `idea-stage/denominator_link_probe_20260607.json` | Lightweight URL probe artifact; PowerShell web probing was inconclusive and not used as data unavailability evidence. |

# 2026-06-07 Fixed-Recall Q1 denominator-continuation package

| File or directory | Purpose |
|---|---|
| `scripts/figlib_metadata_probe.py` | Lightweight FIgLib/HPWREN public directory probe; reads sequence index, tar index, sample frame offsets, and HEAD metadata without downloading full archives. |
| `scripts/figlib_manifest_builder.py` | Full FIgLib manifest builder; enumerates all public sequence directory pages into fixed-recall-compatible CSV without downloading image payloads. |
| `scripts/figlib_split_audit.py` | FIgLib split/leakage audit script; builds event-family fold plan and quantifies station/fire-family leakage risk from the manifest. |
| `scripts/xray_denominator_probe.py` | Lightweight XRay denominator probe; fetches official SIXray `data_list.txt` and HEAD-checks small metadata artifacts without downloading image data. |
| `idea-stage/RACO_DENOMINATOR_AUDIT.json` | Machine-readable FIgLib metadata evidence from the probe: 503 public sequence directories, sample offset structure, station heuristic, and tar HEAD status. |
| `idea-stage/RACO_DENOMINATOR_AUDIT.md` | Human-readable RACO-Wildfire denominator audit; current evidence passes the denominator and split gates for a pilot but keeps visual purity, FIgLib calibration, positive recall, TTD, and FAR as required next gates. |
| `idea-stage/RACO_DENOMINATOR_AUDIT.html` | Rendered HTML view of the RACO denominator audit. |
| `idea-stage/figlib_manifest.csv` | Full public-directory FIgLib frame manifest: 38,547 frame rows, event units, pre-event negative frame/minute units, offsets, station/camera fields, and split keys. |
| `idea-stage/FIGLIB_MANIFEST_AUDIT.json` | Machine-readable full FIgLib manifest audit: 502 non-empty sequences, 501 positive event units, 16,340 negative frame/minute units, and 269.54 conservative negative span hours. |
| `idea-stage/FIGLIB_MANIFEST_AUDIT.md` | Human-readable full FIgLib manifest audit and first denominator-size gate result. |
| `idea-stage/FIGLIB_MANIFEST_AUDIT.html` | Rendered HTML view of the full FIgLib manifest audit. |
| `idea-stage/figlib_label_audit.json` | Fixed-recall label/unit audit for the FIgLib manifest using `eval_unit`; confirms 0 mixed units. |
| `idea-stage/figlib_label_audit.md` | Human-readable FIgLib label/unit audit. |
| `idea-stage/figlib_label_audit.html` | Rendered HTML view of the FIgLib label/unit audit. |
| `idea-stage/FIGLIB_SPLIT_AUDIT.json` | Machine-readable split/leakage audit: 40 stations, 296 legacy `date|fire_name` groups, 290 primary fire-family keys, 83 multi-station legacy groups, and balanced 5-fold plan. |
| `idea-stage/FIGLIB_SPLIT_AUDIT.md` | Human-readable split/leakage audit and fold plan for FIgLib. |
| `idea-stage/FIGLIB_SPLIT_AUDIT.html` | Rendered HTML view of the split/leakage audit. |
| `idea-stage/FIGLIB_EVENT_FAMILY_FOLDS.csv` | Frozen 5-fold event-family assignment for FIgLib using primary fire-family keys; each fold has about 100 positive event units and 55-59 negative proxy hours. |
| `idea-stage/FIGLIB_MANIFEST_CLAUDE_REVIEW.md` | Claude Opus 4.8 review of the full FIgLib manifest gate; verdict MINOR and confirms denominator-size PASS for pilot after reconciliations. |
| `idea-stage/FIGLIB_MANIFEST_CLAUDE_REVIEW.html` | Rendered HTML view of the FIgLib manifest Claude review. |
| `scripts/figlib_visual_purity_sample.py` | Visual-purity sampler for FIgLib pre-onset negatives; stratifies by frozen folds and samples frames closest to the -300s boundary. |
| `idea-stage/FIGLIB_VISUAL_PURITY_SAMPLE.csv` | Row-level 50-frame visual-purity sample with download status and blank manual labels. |
| `idea-stage/FIGLIB_VISUAL_PURITY_SAMPLE.json` | Machine-readable visual-purity sample summary; 49 of 50 images downloaded. |
| `idea-stage/FIGLIB_VISUAL_PURITY_SAMPLE.md` | Human-readable visual-purity sample report and manual labeling codebook. |
| `idea-stage/FIGLIB_VISUAL_PURITY_SAMPLE.html` | Rendered HTML view of the visual-purity sample report. |
| `idea-stage/FIGLIB_VISUAL_PURITY_CONTACT_SHEET.jpg` | Contact sheet for the downloaded pre-onset visual-purity sample. |
| `idea-stage/figlib_visual_purity_images/` | Downloaded sampled FIgLib pre-onset frames for manual purity labeling. |
| `idea-stage/FIGLIB_VISUAL_PURITY_PRELIM_LABELS.csv` | Preliminary row-level visual labels from contact-sheet plus selected full-image inspection; 42 clean, 6 ambiguous, 2 bad_image, 0 contaminated. |
| `idea-stage/FIGLIB_VISUAL_PURITY_PRELIM_SUMMARY.json` | Machine-readable summary of preliminary visual-purity labels. |
| `idea-stage/FIGLIB_VISUAL_PURITY_PRELIM_AUDIT.md` | Human-readable preliminary visual-purity audit with caveats and sensitivity-use guidance. |
| `idea-stage/FIGLIB_VISUAL_PURITY_PRELIM_AUDIT.html` | Rendered HTML view of the preliminary visual-purity audit. |
| `scripts/figlib_visual_purity_expanded_sample.py` | Expanded multi-offset visual-purity sampler for FIgLib pre-onset negatives; stratifies by frozen folds and offset targets. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_SAMPLE.csv` | 100-row expanded visual-purity sample: 20 rows per fold and 25 rows per offset target (-300, -600, -900, -1200 sec). |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_SAMPLE.json` | Machine-readable expanded visual-purity sample summary; 100/100 images downloaded. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_SAMPLE.md` | Human-readable expanded visual-purity sample report and manual labeling codebook. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_SAMPLE.html` | Rendered HTML view of the expanded visual-purity sample report. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_CONTACT_SHEET.jpg` | Contact sheet for the expanded 100-image visual-purity sample. |
| `idea-stage/figlib_visual_purity_expanded_images/` | Downloaded expanded visual-purity sample images. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_PRELIM_AUDIT.json` | Machine-readable expanded contact-sheet preliminary screen; no obvious localized smoke/flame at contact-sheet scale. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_PRELIM_AUDIT.md` | Human-readable expanded contact-sheet preliminary screen; not final row-level adjudication. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_PRELIM_AUDIT.html` | Rendered HTML view of the expanded preliminary visual screen. |
| `scripts/figlib_visual_purity_label_report.py` | Converts a sampled FIgLib visual-purity CSV plus conservative screen-status index lists into row-level screen labels and a summary report. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_CODEX_SCREEN_LABELS.csv` | Row-level Codex contact-sheet screen labels for the 100-image expanded visual-purity sample: 72 clean, 28 ambiguous, 0 contaminated, 0 bad image. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_CODEX_SCREEN_LABELS.json` | Machine-readable summary of the expanded row-level contact-sheet screen and use policy. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_CODEX_SCREEN_LABELS.md` | Human-readable expanded row-level visual-purity screen report; explicitly not final human adjudication. |
| `idea-stage/FIGLIB_VISUAL_PURITY_EXPANDED_CODEX_SCREEN_LABELS.html` | Rendered HTML view of the expanded row-level visual-purity screen report. |
| `scripts/figlib_score_sample.py` | Runs or reuses existing formal YOLO predictions on the FIgLib visual-purity sample and summarizes D-Fire-transferred false-alarm sanity with Wilson confidence intervals. |
| `idea-stage/FIGLIB_SCORE_SAMPLE_YOLO_SEED22.csv` | Prediction rows for formal YOLO26n seed22 baseline, baseline_eqstep, and hardneg arms on 49 downloaded FIgLib pre-onset images. |
| `idea-stage/FIGLIB_SCORE_SAMPLE_YOLO_SEED22.json` | Machine-readable score-sample summary, threshold alarm rates, D-Fire threshold provenance, and Wilson confidence intervals. |
| `idea-stage/FIGLIB_SCORE_SAMPLE_YOLO_SEED22.md` | Human-readable score-sample report; R0.90 results are baseline 1/49, baseline_eqstep 0/49, hardneg 1/49 all-downloaded alarms, but arm ranking is explicitly unsupported at n=49. |
| `idea-stage/FIGLIB_SCORE_SAMPLE_YOLO_SEED22.html` | Rendered HTML view of the score-sample report. |
| `idea-stage/FIGLIB_SCORE_SAMPLE_CLAUDE_REVIEW.md` | Claude Opus 4.8 score-sample review record; verdict MINOR, with adopted edits removing arm-ranking language and adding CI/threshold caveats. |
| `idea-stage/FIGLIB_SCORE_SAMPLE_CLAUDE_REVIEW.html` | Rendered HTML view of the score-sample Claude review. |
| `scripts/figlib_raco_frozen_eval.py` | Frozen-fold RACO evaluator/protocol script for FIgLib: calibration recall LCB thresholding, early event detection, TTD, clustered false-alarm FAR, and conservative pre-event span-hour denominator. |
| `idea-stage/RACO_FROZEN_FOLD_PROTOCOL.json` | Machine-readable frozen-fold RACO protocol denominator summary: 501 positive event units, 16,340 negative units, 269.54 conservative hours, and 5 folds. |
| `idea-stage/RACO_FROZEN_FOLD_PROTOCOL.md` | Human-readable frozen-fold RACO protocol locking units, split rule, denominators, and claim gate before full detector scoring. |
| `idea-stage/RACO_FROZEN_FOLD_PROTOCOL.html` | Rendered HTML view of the frozen-fold RACO protocol. |
| `idea-stage/RACO_FROZEN_FOLD_SYNTHETIC_ORACLE_SCORES.csv` | Synthetic oracle score table covering all 36,454 positive/negative manifest images; software smoke-test input only. |
| `idea-stage/RACO_FROZEN_FOLD_SYNTHETIC_ORACLE_EVAL.json` | Machine-readable strict-coverage synthetic evaluator smoke test; verifies 501/501 event recall and 0 FAR over 269.54 hours. |
| `idea-stage/RACO_FROZEN_FOLD_SYNTHETIC_ORACLE_EVAL.md` | Human-readable synthetic evaluator smoke test; explicitly not a detector result. |
| `idea-stage/RACO_FROZEN_FOLD_SYNTHETIC_ORACLE_EVAL.html` | Rendered HTML view of the synthetic evaluator smoke test. |
| `idea-stage/RACO_FROZEN_FOLD_HARD_SYNTHETIC_SCORES.csv` | Hard synthetic score table with partial early recall and clustered false alarms; software smoke-test input only. |
| `idea-stage/RACO_FROZEN_FOLD_HARD_SYNTHETIC_EVAL.json` | Machine-readable hard synthetic evaluator test; verifies LCB gating, clustered FAR, and unreachable-target reporting. |
| `idea-stage/RACO_FROZEN_FOLD_HARD_SYNTHETIC_EVAL.md` | Human-readable hard synthetic evaluator test; target 0.90 passes LCB, target 0.95 is unreachable by LCB in all folds. |
| `idea-stage/RACO_FROZEN_FOLD_HARD_SYNTHETIC_EVAL.html` | Rendered HTML view of the hard synthetic evaluator test. |
| `idea-stage/RACO_FROZEN_FOLD_CLAUDE_REVIEW.md` | Claude Opus 4.8 stress-review record for the frozen-fold RACO protocol; initial MAJOR conditional, follow-up MINOR after metric-gate fixes. |
| `idea-stage/RACO_FROZEN_FOLD_CLAUDE_REVIEW.html` | Rendered HTML view of the frozen-fold RACO Claude review. |
| `idea-stage/RACO_FROZEN_FOLD_PREREGISTRATION.json` | Machine-readable frozen scoring-gate pre-registration: targets, LCB rule, early window, debounce, denominator, artifact contract, and FAR population check. |
| `idea-stage/RACO_FROZEN_FOLD_PREREGISTRATION.md` | Human-readable pre-registration freezing RACO scoring/evaluation degrees of freedom before full detector inference. |
| `idea-stage/RACO_FROZEN_FOLD_PREREGISTRATION.html` | Rendered HTML view of the frozen-fold pre-registration. |
| `scripts/figlib_raco_score_manifest.py` | Full-manifest FIgLib scoring runner with dry-run planning, image-level true-zero score persistence, detection-level raw rows, cache/resume support, concurrent download workers, and fail-on-error default. |
| `scripts/figlib_raco_throughput_benchmark.py` | Cached bounded throughput benchmark for the FIgLib RACO scoring runner; measures batch-size variants and writes a pre-flight full-scoring gate report. |
| `scripts/figlib_raco_preflight_gates.py` | Computes full-scoring statistical preflight gates for the frozen `calib_lcb` fixed-recall targets from manifest event-unit counts. |
| `scripts/figlib_url_preflight.py` | Concurrent HEAD / range-GET reachability preflight for selected FIgLib frame URLs before fail-on-error full scoring. |
| `idea-stage/RACO_FULL_SCORING_DRYRUN_YOLO_SEED22.json` | Machine-readable dry-run plan for scoring 36,454 positive/negative FIgLib images with three YOLO seed22 arms. |
| `idea-stage/RACO_FULL_SCORING_DRYRUN_YOLO_SEED22.md` | Human-readable dry-run scoring plan and artifact persistence contract. |
| `idea-stage/RACO_FULL_SCORING_DRYRUN_YOLO_SEED22.html` | Rendered HTML view of the YOLO seed22 full-scoring dry-run plan. |
| `idea-stage/RACO_FULL_SCORING_COMMANDS.md` | Prepared PowerShell command sheet for full three-arm YOLO seed22 FIgLib engineering scoring and strict-coverage evaluation; not yet executed. |
| `idea-stage/RACO_FULL_SCORING_COMMANDS.html` | Rendered HTML view of the full scoring command sheet. |
| `idea-stage/RACO_FULL_SCORING_FAILURE_POLICY.md` | Pre-registered unreachable-image and run-failure policy for the first full engineering scoring attempt. |
| `idea-stage/RACO_FULL_SCORING_FAILURE_POLICY.html` | Rendered HTML view of the full-scoring failure policy. |
| `idea-stage/RACO_FULL_SCORING_PREFLIGHT_GATES.json` | Machine-readable statistical attainability preflight; R0.90/R0.95 pass with 400-401 calibration positive event units per held-out fold. |
| `idea-stage/RACO_FULL_SCORING_PREFLIGHT_GATES.md` | Human-readable full-scoring preflight gate report for `calib_lcb` event-count attainability. |
| `idea-stage/RACO_FULL_SCORING_PREFLIGHT_GATES.html` | Rendered HTML view of the full-scoring preflight gate report. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT_SMOKETEST.csv` | 100-row URL preflight smoke-test results; 100/100 selected URLs passed. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT_SMOKETEST.json` | Machine-readable URL preflight smoke-test summary. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT_SMOKETEST.md` | Human-readable URL preflight smoke-test report. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT_SMOKETEST.html` | Rendered HTML view of the URL preflight smoke-test report. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT.csv` | Full selected-manifest URL reachability results: 36,454 checked, 0 failures. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT.json` | Machine-readable full URL reachability preflight summary. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT.md` | Human-readable full URL reachability preflight report. |
| `idea-stage/RACO_FULL_SCORING_URL_PREFLIGHT.html` | Rendered HTML view of the full URL reachability preflight report. |
| `idea-stage/RACO_FULL_SCORING_OPERATIONAL_PREFLIGHT.md` | Operational preflight report: all three YOLO seed22 arms load on GPU device 0; D: free space is sufficient for projected FIgLib cache. |
| `idea-stage/RACO_FULL_SCORING_OPERATIONAL_PREFLIGHT.html` | Rendered HTML view of the full-scoring operational preflight report. |
| `idea-stage/RACO_FULL_SCORING_GPU_MODEL_LOAD_SMOKETEST_*` | One-image, three-arm GPU smoke-test plan, selected manifest, image-score CSV, and detection CSV artifacts. |
| `idea-stage/RACO_SCORING_DOWNLOAD_WORKERS_SMOKETEST_*` | Four-image scoring-runner smoke test proving `--download-workers` path preserves successful image-level and detection-level outputs. |
| `idea-stage/RACO_FULL_SCORING_RUN_STATUS.md` | Active full YOLO seed22 FIgLib scoring run status; launched PID 7448 with initial rows observed. |
| `idea-stage/RACO_FULL_SCORING_RUN_STATUS.html` | Rendered HTML view of the active full scoring run status. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_STDOUT.log` | Stdout log for the active full scoring run. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_STDERR.log` | Stderr log for the active full scoring run. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_IMAGE_SCORES.csv` | Active full scoring image-level output CSV; incomplete until process finishes and row coverage is verified. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_DETECTIONS.csv` | Active full scoring detection-level output CSV; incomplete until process finishes. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_PLAN.json` | Machine-readable full-scoring selected-image contract generated by the prepared command: 36,454 images and three YOLO seed22 arms. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_PLAN.md` | Human-readable full-scoring plan generated by dry-run command, with fold/label counts and persistence contract. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_PLAN.html` | Rendered HTML view of the full-scoring plan. |
| `idea-stage/RACO_FULL_SCORING_YOLO_SEED22_SELECTED_MANIFEST.csv` | Exact 36,454-row FIgLib positive/negative image universe selected for the prepared full three-arm scoring run. |
| `scripts/figlib_raco_scoring_pilot_report.py` | Summarizes bounded real FIgLib scoring pilots, including image-row coverage, true-zero preservation, detection rows, and fixed-recall evaluator reachability. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_DRYRUN.json` | Machine-readable dry-run plan for the 20-image YOLO seed22 real scoring pilot. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_DRYRUN.md` | Human-readable dry-run plan for the bounded real scoring pilot. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_DRYRUN.html` | Rendered HTML view of the pilot dry-run plan. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_SELECTED_MANIFEST.csv` | Selected 20-row manifest for the bounded real scoring pilot: 2 positive and 2 negative images per fold. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_IMAGE_SCORES.csv` | Real image-level pilot scores for baseline, baseline_eqstep, and hardneg arms; 60 model-image rows with true zero scores preserved. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_DETECTIONS.csv` | Raw detection rows from the bounded real scoring pilot. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_EVAL.json` | Machine-readable strict-coverage frozen evaluator output for the bounded pilot; targets are unreachable due tiny calibration sample. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_EVAL.md` | Human-readable strict-coverage evaluator output for the bounded pilot. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_EVAL.html` | Rendered HTML view of the bounded pilot evaluator output. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_REPORT.json` | Machine-readable bounded real scoring pilot report. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_REPORT.md` | Human-readable bounded real scoring pilot report; validates plumbing but makes no arm-quality claim. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_REPORT.html` | Rendered HTML view of the bounded real scoring pilot report. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_SMOKE_TESTS.json` | Machine-readable scoring-runner smoke tests: idempotency, failure injection, plus-sign URL/cache, and selected-manifest balance. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_SMOKE_TESTS.md` | Human-readable scoring-runner smoke tests required by Claude pilot review. |
| `idea-stage/RACO_SCORING_PILOT_YOLO_SEED22_SMOKE_TESTS.html` | Rendered HTML view of the scoring-runner smoke tests. |
| `idea-stage/RACO_SCORING_FAILURE_INJECTION_MANIFEST.csv` | One-row bad-URL manifest for scoring-runner failure injection. |
| `idea-stage/RACO_SCORING_FAILURE_INJECTION_IMAGE_SCORES.csv` | Failure-injection image-level output; contains one `scored_ok=0` row with persisted error. |
| `idea-stage/RACO_SCORING_FAILURE_INJECTION_DETECTIONS.csv` | Failure-injection detection-level output; header only as expected. |
| `idea-stage/RACO_SCORING_PILOT_CLAUDE_REVIEW.md` | Claude Opus 4.8 review/adoption record for the bounded real scoring pilot; verdict MINOR. |
| `idea-stage/RACO_SCORING_PILOT_CLAUDE_REVIEW.html` | Rendered HTML view of the scoring-pilot Claude review. |
| `idea-stage/RACO_PREFULL_SCORING_CLAUDE_REVIEW.md` | Claude Opus 4.8 pre-full-scoring audit and adoption record; conditional GO after failure-policy and LCB preflight blockers were cleared. |
| `idea-stage/RACO_PREFULL_SCORING_CLAUDE_REVIEW.html` | Rendered HTML view of the pre-full-scoring Claude review/adoption record. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BASELINE_DRYRUN.json` | Machine-readable dry-run plan for 10-image GPU throughput smoke test. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BASELINE_IMAGE_SCORES.csv` | Image-level scores from 10-image GPU throughput smoke test. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BASELINE_DETECTIONS.csv` | Detection rows from 10-image GPU throughput smoke test. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BASELINE_REPORT.json` | Machine-readable conservative throughput estimate for one-image-at-a-time GPU scoring. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BASELINE_REPORT.md` | Human-readable throughput smoke report; estimates 14.78 h for 3-arm full scoring without batching. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BASELINE_REPORT.html` | Rendered HTML view of the throughput smoke report. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BATCH_GATE.json` | Machine-readable cached batch throughput gate; recommends `RECT_BATCH16` and estimates 7.85 h cached inference for three YOLO seed22 arms. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BATCH_GATE.md` | Human-readable cached batch throughput gate for batch sizes 1/8/16/32 plus rect batch16. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BATCH_GATE.html` | Rendered HTML view of the cached batch throughput gate. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_BATCH_GATE_*` | Per-variant dry-run, selected-manifest, image-score, and detection CSV artifacts for the cached batch throughput gate. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_100IMG_GATE.json` | Machine-readable 100-image batch throughput gate; expands the FIgLib cache from 20 to 100 images with 0 errors and recommends `RECT_BATCH16`. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_100IMG_GATE.md` | Human-readable 100-image batch throughput gate separating cache-expansion timing from cached rect batch16 inference timing. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_100IMG_GATE.html` | Rendered HTML view of the 100-image batch throughput gate. |
| `idea-stage/RACO_SCORING_THROUGHPUT_GPU_100IMG_GATE_*` | Per-variant dry-run, selected-manifest, image-score, and detection CSV artifacts for the 100-image throughput gate. |
| `idea-stage/XRAY_DENOMINATOR_AUDIT.json` | Machine-readable XRayThreat-RC evidence state: PIDray/SIXray/SIXray-D/OPIXray public metadata, access blockers, and fixed-recall denominator gates. |
| `idea-stage/XRAY_DENOMINATOR_AUDIT.md` | Human-readable XRayThreat-RC denominator audit; current evidence keeps XRay as access-required conditional GO. |
| `idea-stage/XRAY_DENOMINATOR_AUDIT.html` | Rendered HTML view of the XRay denominator audit. |
| `idea-stage/FIXED_RECALL_Q1_DATA_REQUEST_TEMPLATES.md` | Request/access templates and checklists for SIXray-D, original SIXray, and FIgLib next-step metadata work. |
| `idea-stage/FIXED_RECALL_Q1_DATA_REQUEST_TEMPLATES.html` | Rendered HTML view of the request/access templates. |
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_CONTINUATION_CLAUDE_REVIEW.md` | Claude Opus 4.8 continuation review record for the RACO/XRay denominator audits, including MINOR verdict and adopted edits. |
| `idea-stage/FIXED_RECALL_Q1_DENOMINATOR_CONTINUATION_CLAUDE_REVIEW.html` | Rendered HTML view of the denominator-continuation Claude review. |

# 2026-06-11 paper-writing pipeline

| File or directory | Purpose |
|---|---|
| `paper/.aris/assurance.txt` | Workflow 3 assurance marker resolved to `draft` for this run. |
| `PAPER_PLAN_20260611-102758.md` | Timestamped paper plan generated from `NARRATIVE_REPORT.md`. |
| `PAPER_PLAN_20260611-103434_REVIEWED.md` | Timestamped reviewed paper plan after Claude Opus 4.8 edits. |
| `PAPER_PLAN.md` | Fixed-name current paper plan for the recall-anchored fire/smoke evaluation manuscript. |
| `paper/reviews/claude_plan_review_20260611-103434.md` | Claude Opus 4.8 paper-plan review record and adopted edits. |
| `figures/gen_fig1_fixed_recall_protocol.py` | Reproducible generator for the fixed-recall protocol diagram. |
| `figures/gen_fig2_r090_false_alarm_burden.py` | Reproducible generator for the R0.90 FPR grouped bar chart. |
| `figures/gen_fig3_yolo_target_recall_curve.py` | Reproducible generator for the YOLO26n target-recall/FPR curve. |
| `figures/gen_fig4_evidence_completion_map.py` | Reproducible generator for the evidence completion roadmap. |
| `figures/gen_paper_tables.py` | Reproducible generator for paper LaTeX table snippets and summary CSV files. |
| `figures/paper_plot_style.py` | Shared Python-only Nature-style plot settings; figures export PDF/SVG/TIFF. |
| `figures/fig1_fixed_recall_protocol.pdf` | Generated protocol diagram for the paper. |
| `figures/fig2_r090_false_alarm_burden.pdf` | Generated R0.90 false-alarm burden chart. |
| `figures/fig3_yolo_target_recall_curve.pdf` | Generated YOLO26n target-recall/FPR curve. |
| `figures/fig4_evidence_completion_map.pdf` | Generated evidence-completion roadmap figure. |
| `figures/fig*.svg` | Editable SVG exports for the generated paper figures. |
| `figures/fig*.tiff` | High-resolution TIFF exports for the generated paper figures. |
| `figures/table_main_r090.tex` | LaTeX main R0.90 fixed-recall result table. |
| `figures/table_ablation_deltas.tex` | LaTeX R0.90 ablation-delta table. |
| `figures/table_training_budget.tex` | LaTeX formal YOLO26n training-budget table; excludes the scheduled probe because it uses a separate scheduling convention. |
| `figures/table_map_vs_fpr.tex` | LaTeX YOLO26n AP/mAP versus R0.90 FPR table. |
| `figures/table_completion_gate.tex` | LaTeX roadmap/completion-gate table. |
| `figures/gen_split_overlap_check.py` | Split-overlap checker for path-level train/test disjointness evidence. |
| `figures/paper_split_overlap_check.csv` | Split-overlap output used to state zero path-level overlap for current D-Fire split files. |
| `figures/latex_includes.tex` | LaTeX snippets for generated paper figures. |
| `paper/main.tex` | Main IEEE-style LaTeX manuscript entrypoint. |
| `paper/main.pdf` | Compiled six-page draft manuscript PDF. |
| `paper/sections/` | Manuscript section files written from `NARRATIVE_REPORT.md`. |
| `paper/references.bib` | Draft bibliography used by the compiled paper. |
| `paper/figures/` | Paper-local copies of generated figures and LaTeX table snippets consumed by `main.tex`. |
| `paper/reviews/claude_draft_review_20260611-104545.md` | Claude Opus 4.8 draft review record and adopted fixes. |
| `paper/reviews/claude_claim_patch_recheck_20260611-110416.md` | Claude Opus 4.8 narrow post-fix claim-audit recheck; PASS for patched WARN items. |
| `paper/reviews/claude_update_review_20260611-113253.md` | Claude Opus 4.8 review for the refreshed narrative update; stale-PDF blocker resolved by rebuilding `main.pdf`. |
| `paper/reviews/claude_claim_audit_20260611-113630.md` | Claude Opus 4.8 draft-scope claim audit for the refreshed narrative update; verdict PASS. |
| `paper/PAPER_CLAIM_AUDIT.md` | Human-readable claim audit summary for this draft-assurance run. |
| `paper/PAPER_CLAIM_AUDIT.json` | Machine-readable claim audit summary with hashes and reviewer thread IDs. |
| `paper/.aris/traces/paper-claim-audit/20260611_run01/evidence_packet.md` | Evidence packet supplied to the narrow Claude post-fix audit recheck. |
| `paper/PAPER_WRITING_PIPELINE_REPORT.md` | End-to-end paper-writing pipeline report, verification results, and remaining gates. |
| `paper/preview/page-*.png` | Rendered PDF preview pages used for final visual spot checks. |
