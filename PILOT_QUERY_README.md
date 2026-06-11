# DAQ-DETR — 创新点定位 + 本地 pilot 操作手册

**日期**: 2026-05-30  ·  **硬件**: 本地 RTX 5070 Ti (12 GB)  ·  **状态**: 选题成立，待本地 go/no-go

---

## 一、你的创新点到底在哪里（必须钉死，否则撞车）

我查过 2024–2026 文献，**你的三个原料单独拿出来都不新**：
- matched/unmatched query 校准 → 通用 DETR-UQ 已做（arXiv 2412.01782 等）；
- 火焰误报抑制 → RT-DETR-Smoke 自带 uncertainty query selection，还有 VQGAN+InnMPD-IoU 处理 "flame-like"；
- 火焰相似负样本 → 部分数据集已收录 hard negatives。

所以创新点**不能是"某个模块"**，必须是下面这套**组合 + 一个尖锐主张**：

> **主张**：火灾检测只报 mAP 是不够的；必须在"火焰相似干扰物"上报告 **固定召回率下的 FPR / FPPI**。
> 我们构建该干扰物基准与评测协议，并把 DETR 的 unmatched-query 分布训练成一个**面向误报的校准门**，
> 在固定召回下比 hard-negative 训练 + 调阈值更低误报——而这个 query 级信号是 anchor-based YOLO 不具备的。

三层贡献，按"抗撞车强度"排序：

1. **干扰物基准 + 评测协议（最稳）**：DFS `other` + D-Fire `none` + 网络挖掘的 sunset/车灯/反光/红橙物，配 FPR@recall、FPPI、跨数据集（train D-Fire → test DFS）。现有论文几乎都不报这些指标，这是别人没占的位。
2. **面向误报的 DETR query 校准门（中等）**：与通用 UQ 不同——目标不是"估可靠性"，而是"用显式干扰物监督抑制误报且保召回"。差异点锁死在 *objective* 上。
3. **YOLO 差异化（定位）**：query set 结构是 DETR 区别于 YOLO 的技术抓手，论文叙事据此展开。

**一句话卖点**（投稿用）：*A fire-like-distractor false-alarm benchmark and a DETR query-calibration gate that lowers FPR at fixed recall beyond hard-negative training — a query-level signal anchor-based detectors lack.*

---

## 二、已经跑出的一个支撑结论（来自现有 pilot 数据，无需重跑）

把"暖色先验"当成最朴素的火焰检测器，在现有 8 张图上：

| 图类型 | warm_pixel_ratio |
|---|---|
| 真火 fire_02 | 0.211 |
| 真火 fire_01 / 03 / 04 | 0.0004 / 0.0016 / **0.000** |
| 烟 smoke_07 / 10 | 0.000 / 0.069 |
| **干扰物 distractor_11** | **0.058** |

→ 若把阈值设到能挡住该干扰物（>0.0585），**6 张真火/烟里有 4 张会被漏掉**。
**暖色先验无法分开火与干扰物**——干扰物比 3/4 的真火得分还高。这正面支撑"问题真实、简单方法不够"的论文前提。
（样本量极小，仅作动机，不作结论。）

---

## 三、本地 pilot（在你的 5070 Ti 上跑）

### Tier A — 仪器验证，现在就能跑（几分钟，用现有 rtdetr-l.pt）

```bash
cd D:\detr_Q3
pip install ultralytics            # 若未装
python scripts\pilot_query_signal.py --weights rtdetr-l.pt --images pilot_images --out pilot_outputs\query_probe
```

作用：抽取 RT-DETR 的 300 个 query 的**置信度分布**，输出 `query_features.csv` + `query_separability.json`，
并打印 fire/smoke vs distractor 在各分布特征上的 gap。
**注意**：rtdetr-l.pt 是 COCO 权重、无 fire 类，这一步只证明"信号可抽取、分布有结构"，**不是真信号**。
跑通即说明流水线 OK，可进入 Tier B。

### Tier B — 真正的 go/no-go（核心，需要 D-Fire，约几小时）

1. **数据**：下载 D-Fire（约 21,527 张，含 fire/smoke/none）→ 转 Ultralytics 格式放 `D:\detr_Q3\data\dfire`。
   另备干扰物测试集：DFS `other` + D-Fire none + 少量网络 sunset/车灯/反光。
2. **训练 RT-DETR-R18（不要用 L，12 GB 吃力）**：
   ```bash
   yolo detect train model=rtdetr-r18.yaml data=data\dfire\dfire.yaml imgsz=640 epochs=60 batch=8 device=0
   ```
3. **三个对照**（这一步决定论文成立与否）：
   - baseline + 调阈值；
   - baseline + hard-negative 训练（**最强简单基线**）；
   - baseline + DAQ 校准门。
4. **导出每个模型的预测为 `image,conf` CSV**，再跑评测协议：
   ```bash
   python scripts\fppi_fpr_eval.py --pred hardneg_preds.csv --pred2 daq_preds.csv --labels labels.csv --out gonogo.json
   ```
   `labels.csv` 列为 `image,label`，label ∈ {fire,smoke,distractor,none}。

### 决策规则（你自己定的 kill condition，必须先打）

- DAQ 在 **recall=0.90** 下 FPR / FPPI **明显低于** hard-negative+调阈值 → **GO**，全力做 DAQ-DETR。
- 两者差距 ~0 → **KILL**：query 校准门没必要，转 FPQ-DETR（但 FPQ 须证明 query-bias 严格优于 prior-as-input，否则也撞 RT-DETR-Smoke）。

---

## 四、产物清单

| 文件 | 作用 |
|---|---|
| `scripts/pilot_query_signal.py` | Tier-A：抽取 RT-DETR query 分布、算可分性（现有权重可跑） |
| `scripts/fppi_fpr_eval.py` | 评测协议核心：固定召回下的 FPR/FPPI + 双模型 go/no-go 对比（已自测通过） |
| `PILOT_QUERY_README.md` | 本文件 |

文献依据见 `outputs/` 下既有报告与本仓库根目录历史文档。
