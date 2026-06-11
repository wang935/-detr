# DAQ-DETR 项目交接（历史快照，不是当前主线）

> 状态说明：本文件保留为 2026-05-30 左右的早期交接材料，里面的“DAQ-DETR 主方法”和“SCI 二区”目标已经不是当前论文主线。当前主线以 `MANIFEST.md`、`refine-logs/FINAL_PROPOSAL.md`、`refine-logs/FORMAL_EXPERIMENT_DESIGN.md` 为准：固定召回误报控制 + 硬负样本训练为主，DETR query/DAQ 只作为分析项，当前目标按 SCI 三区风险评估推进。

下面每一节对应 Claude Project 创建页的一个填写项，可直接复制粘贴。

---

## ① 项目名称（Project name）

```
DAQ-DETR：火焰烟雾低误报检测（SCI 二区）
```

---

## ② 项目描述（Description，一句话）

```
基于 RT-DETR 的火焰/烟雾检测研究，核心是"火焰相似干扰物"误报抑制 + 评测协议，目标投 SCI 二区。
```

---

## ③ 自定义指令（Custom instructions，整段粘贴）

```
你在协助一个计算机视觉科研项目，目标是写一篇 SCI 二区可投的火焰/烟雾目标检测论文。

【方向】DAQ-DETR（主）：Distractor-Aware Query Calibration —— 假设 RT-DETR 的 object
query 置信度"分布"里带有信号，能区分真·火/烟 与 火焰相似干扰物（夕阳、车灯、红橙物、
反光、云），据此在固定召回率下降低误报。FPQ-DETR（备选）：火焰先验引导 query selection。

【创新点定位（必须守住，别撞车）】单个模块都不新（matched/unmatched 校准、火焰误报抑制、
相似负样本各自都有人做过）。能站住的是组合 + 一句尖锐主张：火灾检测只报 mAP 不够，必须在
火焰相似干扰物上报"固定召回下的 FPR/FPPI"；我们建这个基准+协议，并把 DETR 的 unmatched
query 分布训成面向误报的校准门，效果优于 hard-negative+调阈值，而这个 query 级信号 YOLO 没有。
三层贡献按抗撞强度：①干扰物基准+评测协议（最稳）②面向误报的 query 校准门（差异点锁在"抑误报"
这个目标，而非通用可靠性估计）③YOLO 差异化叙事。

【kill-gate（最关键，先打）】若"hard-negative 训练 + 调阈值"就能把误报降到和 DAQ 差不多低，
则 DAQ 校准头没必要，方向应转 FPQ-DETR。这是 go/no-go 命门，必须先用实验验证。

【工作风格】回答用中文，简洁直接、少废话。涉及当前事实先联网查证。对结论保持诚实：区分
"已验证"和"假设"，不夸大。凡是写论文要用的数据/数字/引用，先研究核实再动笔。

【机器与环境】两台带 GPU 的 Windows 主机：一台 RTX 5070 Ti(12G)，一台 RTX 3060。
注意 3060 主机默认 python 是 MSYS2 版（无 pip、跑不了 torch GPU），必须用 conda 或
python.org 版 + CUDA 版 torch(cu121)。深度学习实验在本地 GPU 跑，不在云端跑。

【数据】D-Fire 数据集（YOLO 格式，0=smoke 1=fire；fire≈5822 smoke≈5867 none≈9838），
3060 主机路径 D:\fire\D-Fire。None 图含云/夕阳等"像火不是火"场景，直接用作误报评测负样本。
工作目录 D:\detr_Q3。
```

---

## ④ 项目知识（Project knowledge，把这些文件上传到知识库）

建议上传 `D:\detr_Q3` 下：

- `PROJECT_HANDOFF.md`（本文件，总纲）
- `PILOT_QUERY_README.md`（创新点定位 + Tier A/B 操作手册）
- `CODEX_PROMPT_tier_a.md`（给 Codex 的 Tier A 交接 prompt）
- `idea-stage/IDEA_REPORT.md`（完整选题报告 + 文献格局）
- `idea-stage/IDEA_CANDIDATES.md`（两个方向精简版）
- `refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`（提案与实验计划）
- `tier_a_3060.py`、`tier_b_3060.py`（两层全流程脚本）
- `pilot_outputs/pilot_results.csv`（首轮 pilot 数据）

---

## ⑤ 当前进度（粘进知识库或首条消息）

```
【已完成】
- 选题与文献调研：确认 DAQ-DETR 为主、FPQ-DETR 备选；查清近两年 RT-DETR 火焰/烟雾论文
  已较拥挤（RT-DETR-Smoke、IFS-DETR、CSDETR 等），方法创新撞车风险高，故走"基准+协议"路线。
- 首轮 sample-level pilot：证明 COCO 版 RT-DETR/YOLO 在火/烟图上乱报无关类、暖色先验
  单独不可靠（要挡住 1 张干扰物会漏掉 4/6 真火）—— 仅证明"问题真实"，未触及核心假设。
- Tier A 脚本：抽取 RT-DETR 300 个 query 的置信度分布、算火 vs 干扰物可分性。
  首次在 COCO 权重 + 仅 1 张干扰物上跑通流水线（结果不可用，符合预期）。
- 已从 D-Fire 抽好 160 张（40 火+40 烟+80 None）到 pilot_images_dfire，准备重跑 Tier A。
- Tier B 全脚本写好（install/prep/train/export/eval 五子命令），含 FPPI/FPR 评测，已自测逻辑。

【进行中 / 待办】
- Tier A 收尾：在 3060 上用 160 张真样本重跑 query 探针，判读分布是否重叠（预期重叠→
  动机性结论：需微调+DAQ）。卡点：3060 默认 python 是 MSYS2 版，需先换 conda+CUDA torch，
  已写好 Codex 交接 prompt。
- Tier B go/no-go：D-Fire 上微调 RT-DETR baseline arm 与 hardneg arm，比固定召回下 FPR/FPPI，
  执行 kill-gate 判定。
- 若 GO：实现 DAQ 校准头（Stage 2）；若 NO-GO：转 FPQ-DETR。

【诚实提醒】到目前为止只证明了"选题成立"，方法是否有效要等 Tier B 微调后的对比实验。
```

---

## ⑥ 关键文献（备查）

- RT-DETR-Smoke (MDPI Fire 2025): https://www.mdpi.com/2571-6255/8/5/170
- DETR 预测可靠性/object-level 校准 (arXiv 2412.01782): https://arxiv.org/html/2412.01782v2
- Uncertainty-aware DETR (arXiv 2507.14855): https://arxiv.org/html/2507.14855v1
- Early Fire/Smoke 综述 (MDPI Appl. Sci. 2025): https://www.mdpi.com/2076-3417/15/18/10255
- D-Fire 数据集: https://github.com/gaia-solutions-on-demand/DFireDataset
```
