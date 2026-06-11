# 固定相机火焰识别：1 区强线设计（锁定约束版）

**日期**：2026-06-09  **视角**：CVPR/ICCV/ECCV/TPAMI 审稿人 + 实验方法顾问  
**锁定约束**：固定摄像头火焰识别；300ep / 4 arms / 3 压力；模型固定 = {yolo26n, rt_detr_l, InternImage-T+Faster R-CNN(box-only)}；数据固定 = {D-Fire, FIGlib}；压力固定 = {S1 光学退化, S2 干扰物注入, S3 时序/运营}；非硬性不可行不得换。

---

## 0. 故事线主线（先给评审一眼能懂的叙事）

**一句话问题定义**：固定相机火焰检测真正失败的不是"看不见火"，而是——**在保证不漏报的高固定召回算子点上，误报负担爆炸、且告警在时序上不稳定**。社区把它当成一个"检测 AP 已解决"的问题，真正未解决的是"固定召回下的可部署误报/延迟控制"，而这种失败会沿成像退化→语义混淆→时序抖动三条链放大。

**方法主张（三模型异构为何覆盖三类核心风险）**：三个检测器不是冗余对照，而是**张成"部署风险空间"的三个正交基**，各占一个极点：

| 模型 | 架构范式 / 归纳偏置 | 部署画像 | 主要承载的风险类型 |
|---|---|---|---|
| yolo26n | 单阶段 anchor-free / 局部卷积、小容量 | 边缘轻量、高吞吐 | **检出率**（小容量在 S1 退化下 recall 先崩）+ 工程可部署性上限 |
| rt_detr_l | 集合预测 transformer / 全局注意力、无 NMS | 中量 | **误报率/语义判别**（全局上下文区分火 vs 火焰相似干扰物；但 query 置信度外部迁移差——DAQ 已实测外部失败 0.073→0.237） |
| InternImage-T + Faster R-CNN(box) | 两阶段 proposal / 可变形卷积 DCNv3、大感受野强 backbone | 重量 | **定位精度**（AP75、bbox 误差）+ S2 下 RPN 对干扰物的 proposal 污染 |

三者机制不同 → 它们的**失败签名可预测、可归因**，把"bake-off"升级成"**架构 × 失败模式交互**"的科学问题。

**压力场景逻辑（S1/S2/S3 ↔ 三类失败链，且为串联因果链）**：

- **S1 光学退化 ↔ 外观退化失败链**：输入分布漂移 → 特征响应衰减 → 低对比/夜光下漏检（recall 塌）。
- **S2 干扰物注入 ↔ 语义混淆失败链**：暖色/反光/烟团触发"假阳性语义" → 高置信误检（FPR 升、precision 塌）。
- **S3 时序/运营 ↔ 时序决策失败链**：单帧噪声/抖动 → 突发聚簇误报 + 告警延迟权衡（TTD vs FAR）。

三链**串联**：外观退化制造弱特征 → 弱特征更易被语义干扰俘获 → 在时序上放大为聚簇误报。这条因果链就是评审能复述的"故事"。

**为何构成 1 区可辩护的科学问题**：(1) 可证伪——"检测器架构族在三条固定相机失败链上是否有系统、可预测的脆弱性签名，统一算子点协议能否暴露并部分闭合差距"，对应 H1–H5；(2) 非单数据集——D-Fire 域内 + FIGlib 跨域事件级 + leave-station-out；(3) 非只报 mAP——固定召回 FPR/FPPI + 事件级聚簇 FAR/hour + TTD，这些 mAP 反映不了；(4) 机制可解释——注意力/proposal/感受野可归因，不是黑箱比分。

> 中心假设（统领全文，每个 arm/压力都回指它）：**H0：固定相机火焰检测的可部署性瓶颈是"固定召回下的误报负担与时序稳定性"，且该瓶颈的脆弱性是架构条件化的、可被统一算子点协议量化与部分闭合的。**

---

## 1. 1 区发表性风险审计（1–10，针对锁定后的升级线）

| 维度 | 分 | 风险说明 | 可修复动作 |
|---|---:|---|---|
| 创新性 | **6** | 无全新架构/loss；"benchmark+characterization"类要 1 区需 sharp insight，否则被读成比拼 | 把"架构×失败模式交互"做成**可证伪、有机制归因**的发现（如 transformer 在 S2 系统优于单阶段 X 点，但 query 外部不可迁移——有数据支撑）；加一个**轻量统一算子点控制层**作最小方法贡献 |
| 泛化性 | **7** | D-Fire→FIGlib + LOSO 是真泛化；但 FIGlib pre-event 负时长是 proxy，仅 2 数据集 | 站点级 leave-station-out + 事件族 fold + 保守口径；BoWFire/DeepQuest 仅作 image-level negative 补充压力（不进 headline） |
| 鲁棒性 | **7** | 三条参数化压力链是卖点；但合成退化与真实退化有 gap | S1/S2 用**物理可解释模型**（大气散射、色温匹配）+ 在 FIGlib 真实退化帧取小 real 子集**锚定**合成有效性 |
| 可复现性 | **9** | 现有协议（calib/test 互斥、md5、deterministic、last.pt、strict 聚合、预注册 gate）已是顶级 | 唯一缺口=跨框架口径（Ultralytics vs MMDet）→ 统一导出预测 CSV + 同一评测器 |
| 统计显著性 | **6** | n=3 seed 对 1 区偏少；事件级样本相关（同站点/事件族） | 事件族 **bootstrap CI + paired test + Holm 校正 + 效应量(Cohen's d/Cliff's δ) + MDE**；必要时 seed 升到 5 |
| 实验可完成性 | **6** | 36 run 训练（RT-DETR ~50h、InternImage 更重）+ FIGlib 全量 + 3 压力多参数=大工程 | headline 锁 R0.90；**压力评测在冻结权重上做（不重训）**；InternImage 超预算降 2 seed 补充；双 GPU 并行 |

**综合 1 区就绪度 ≈ 6.5/10 = "有条件的 1 区 / 强 Q1 边界"**（较上一版纯 hardneg 线的 4.2 显著提升）。

---

## 2. 当前主线薄弱点 + 最少改动修复清单（只给必要项）

| # | 薄弱点 | 最少改动（不发散） |
|---|---|---|
| P1 | "三模型比拼"易被读成 bake-off | 用**一句可证伪中心假设 H0** 统领；每个 arm/压力都回指 H0 |
| P2 | 纯比较缺方法贡献 | 加一个**轻量统一固定召回算子点控制层**（per-group / site-time 条件阈值，frozen-score 后处理），对比 CRC/LTT；**零重训** |
| P3 | FIGlib 负时长 proxy（1 区命门） | 聚簇 FAR 分 "per pre-event-proxy-hour" 与 "per-frame FP" 两口径分开报 + 保守措辞；能补连续 non-event 序列则解除 |
| P4 | 合成退化可信度被质疑 | FIGlib 真实雾/夜/逆光帧取小 real 子集，验证合成 severity 与真实退化指标变化**方向一致**（anchoring） |
| P5 | 跨框架口径不可比 | 所有模型导出统一预测 CSV(image,conf,bbox)，用**同一评测器**算全部指标；不把各框架内置 val 数字放进主表 |
| P6 | n=3 统计功效 | 事件族 bootstrap + 预注册判定规则 + 报 MDE；主结论只在通过规则时声明 |

---

## 3. 模型/数据集/压力角色定义 + 为何三模型非同质

**三模型非同质性的三条正交轴**（这是"非 bake-off"的核心论证）：

- **架构范式轴**：单阶段(yolo26n) / 集合预测(rt_detr_l) / 两阶段(InternImage-FRCNN)。
- **归纳偏置轴**：局部卷积小容量 / 全局注意力无 NMS / 可变形卷积+proposal 大感受野。
- **部署画像轴**：边缘轻量高吞吐 / 中量 transformer / 重量强 backbone。

每个模型是**不同轴上的极点**，因此三者三角化"部署风险空间"，而非复制同一点。这保证消融能区分"是架构机制差异还是随机差异"。

**数据集角色**：

| 数据集 | 角色 | 提供 | 主指标 |
|---|---|---|---|
| D-Fire | 主任务训练 + 域内算子点 | 火/烟框、域内硬负样本(none/distractor)、calib/test 互斥池 | precision/recall/mAP(补充)、FPR/FPPI@R{0.80–0.95} |
| FIGlib | 跨域 + 事件级（1 区抬升点） | 固定相机野火序列、站点元数据、事件/前事件窗口、事件族 fold | 事件 recall@R0.90/0.95、聚簇 FAR/hour、TTD、bootstrap CI |

**压力角色**：S1/S2/S3 = 三条失败链的**可控注入器**，各对应不同主检验指标（见 §5），在冻结权重上施加，不改变训练。

---

## 4. 4 个 arms 精确定义（claim 结构 + 科学问题 + 预期方向）

> 说明：你 pipeline 里 4 个**训练条件** = {baseline, baseline_eqstep, hardneg, hardneg_sched}；下表把它们映射到论文的 4 个 **claim 角色**。统一算子点控制层是叠加在所有 arm 之上的**评测轴**，不算第 5 个训练 arm。

| Arm | 训练条件 | claim 角色 | 科学问题 | 预期对比结论方向 |
|---|---|---|---|---|
| **A 基线** | baseline（仅正样本） | 检出能力 + 误报底盘锚 | 无任何误报抑制时，三架构的固定召回误报负担与检出上限？ | FPR 普遍高（D-Fire R0.90：YOLO 实测 0.207）；架构差异已现（预期 transformer 语义判别略优、单阶段在 S1 下 recall 先掉） |
| **B 压力** | hardneg（+硬负样本/干扰物暴露） | 处理臂（在 S1/S2/S3 上评） | 训练期负样本暴露 + 压力下，误报是否跨架构下降、检出是否保持？ | FPR 大降（YOLO 实测 0.207→**0.004**，3 seed 无反例）；S2 注入下抑制幅度**架构相关** |
| **C 消融** | baseline_eqstep（步数对齐）+ hardneg_sched（负样本剂量↑） | 混淆隔离 + 剂量响应 | 收益来自负样本语义还是数据量/步数？剂量-响应与正样本覆盖代价？ | hardneg **优于 eqstep**（语义有效，实测 0.152→0.004）；sched 呈剂量曲线但 unique 正覆盖↓(0.7377) |
| **D 最终主张** | 取最优配置**冻结分数** + 统一算子点控制层 | 可部署最终主张（跨域事件级） | 固定事件召回下，统一控制层能否跨域、跨架构稳定降聚簇 FAR 且不增 TTD，并击败 CRC/LTT？ | 控制器显著优于 global/per-group/CRC/LTT，FIGlib LOSO 跨站点稳定 → headline claim |

---

## 5. 三个外部压力的参数化方案

> 通则：所有压力施加在**测试/评测池**与 FIGlib 序列上（不污染训练正样本框）；severity/比例做 sweep，headline 取中度档，附 sweep 曲线；三模型用**完全相同**的压力参数。

### S1 — 大气/光学退化压力（外观退化链）

| 退化类型 | 物理/实现模型 | 参数范围（severity 1→5） | 采样比例 |
|---|---|---|---|
| 雾化/散射 | Koschmieder：I=J·t+A(1−t), t=e^(−βd) | β∈[0.4,3.0], airlight A∈[0.7,1.0] | 单类消融各 ~20%，混合档等比例 |
| 低对比 | 增益+直方图压缩 | gain α∈[0.5,0.9] | 同上 |
| 夜光漂移 | 色温偏移 + gamma + 低光泊松噪声 | CCT 2000–3500K, γ∈[1.5,3.0] | 同上 |
| 散焦/运动模糊 | defocus disk / motion kernel | r∈[1,7]px, L∈[5,25]px | 同上 |
| 传感器噪声+压缩 | 高斯+JPEG | σ∈[5,40]/255, JPEG Q∈[20,90] | 同上 |

- 采样：每张图生成 s∈{0,1,2,3,4,5}；**headline = s3（中度）**，附 severity-sweep。
- 预期评价变化：recall 随 s 单调降（yolo26n 最快）；mAP@.5:.95、AP50/AP75 单调降；bbox 误差升；FPR 可能轻升。
- **主检验指标：Recall 保持率（recall@s / recall@clean）、mAP@.5:.95、AP50/AP75、bbox 误差。**

### S2 — 火焰相似干扰物注入压力（语义混淆链，针对误报）

| 干扰物 | 构造方式 | 关键参数 |
|---|---|---|
| 光源反光（车灯/路灯） | 暖色高亮 blob + 镜头眩光 sprite，色温匹配火焰 1800–2200K | 强度 I∈[0.6,1.0]，光晕 σ |
| 热源抖动 | 火焰相似时序闪烁纹理（灯/反光亮度抖动） | 频率 f∈[2,10]Hz，幅度 ±20% |
| 烟团纹理 | Perlin/Worley 噪声合成云/雾团，半透明叠加 | α∈[0.3,0.7]（贴到 non-fire 背景） |
| 反射体 | 金属/玻璃/水面镜面高光 + 暖色反射 patch | 高光强度、反射率 |

- 注入比例：负样本图中注入率 ρ∈{10%,30%,50%}；每图密度 d∈{1,2,4} patch；**只注入负样本(none)与 FIGlib pre-event 帧，绝不注入正样本框区域**。
- 误报验证指标：固定召回 R0.90 下 **FPR/FPPI**、precision、**distractor-conditional FPR**（仅注入图上的误报率）。
- **主检验指标：Precision/FPR、FPPI、distractor-FPR。** 预期：hardneg/控制器大幅压低 distractor-FPR；rt_detr_l 因全局上下文预期 S2 上 FPR 最低；FRCNN 的 RPN 可能被反光 proposal 污染（中等）；yolo26n 最脆弱。

### S3 — 时序/运营压力（时序决策链：聚簇告警 + 延迟）

| 维度 | 策略/参数 |
|---|---|
| 窗口大小 | W∈{5,15,30} 帧 |
| 平滑策略 | 滑动多数投票 k-of-n；EMA λ∈[0.3,0.7]；连续命中去抖 persistence k∈{2,3,5} |
| 抖动容忍 | 小仿射 ±2px、曝光闪烁 ±10%、丢帧率∈{0,5%,10%} |
| 延迟预算 | {50ms, 100ms, 200ms} 三档，各模型实测单帧延迟标注可达性 |

- 告警稳定性指标：**聚簇 FAR/hour**、**TTD（首次正确告警延迟）**、告警 flicker（状态翻转次数/min）、事件 recall。
- **主检验指标：告警延迟(TTD)、聚簇 FAR/hour、事件 recall、flicker。** 预期：去抖/平滑降聚簇 FAR 但增 TTD（权衡）；控制器在固定事件 recall 下取 **FAR–TTD Pareto 最优**；延迟预算下 yolo26n 可部署性最好（易满足 50ms），rt_detr_l ~100ms，InternImage-FRCNN 可能 >200ms 需降尺寸。

**压力↔主指标映射总表**：S1→Recall 保持率/mAP@.5:.95/AP50/AP75/bbox 误差；S2→Precision-FPR/FPPI/distractor-FPR；S3→TTD/聚簇 FAR-hour/事件 recall/flicker。

---

## 6. 统一训练与评测协议

**训练设置**：

| 项 | yolo26n | rt_detr_l | InternImage-T + FRCNN(box) |
|---|---|---|---|
| 输入尺寸 | 640 | 640 | 短边 800/长边≤1333（fallback 1000×600） |
| batch | 16（上限~32） | 4（上限~8@12G） | 2（上限~2@12G；降尺寸可到4） |
| 优化器 | SGD(auto) | AdamW | AdamW lr1e-4 + layer-decay |
| LR 策略 | cosine + warmup | cosine + warmup | step([0.75,0.9]×total) + linear warmup 500it |
| AMP | ✓ | ✓ | ✓ |
| 多卡 | 单卡足够 | 单卡 | 优先 DDP；否则单卡 |
| 显存预估 | 4–6 GB | 10–11 GB | 10–12 GB |
| 时间/300ep | ~8–12 h | ~50 h（实测 608 s/ep） | ~60–90 h（须实测） |

**早停判据**：**无 val 早停**（`val=false` 保 eval split 干净），固定 300ep + 导出 `last.pt`；"早停"语义替换为训练日志监控不发散（loss NaN/不收敛 → 记录并按 §9 触发替换条款）。

**同口径验证集划分**：单一 eval pool → calib/test 互斥（md5 指纹）；FIGlib 按**事件族 leave-station-out** fold；跨模型/seed/arm 用完全相同 split。

**统一比较口径（关键，跨框架可比）**：

1. **统一导出**：所有模型导出 `(image, conf, x, y, w, h)` 预测 CSV，`export_conf≤0.001`、`max_det=300`、原图分辨率。
2. **同一评测器**：用同一脚本在导出 CSV 上算 COCO 口径 mAP/AP50/AP75 + FPR/FPPI + 事件级指标；**不把各框架内置 val 数字放进主表**（Ultralytics `mAP50-95`=COCO AP、`mAP50`=AP50，对齐 IoU/area 后可比）。
3. **同一阈值策略**：calib 正样本上选达目标 recall 的最高 conf 阈值（R0.80/0.85/0.90/0.95，**R0.90 headline**）。
4. **同一 NMS**：YOLO/FRCNN 用 NMS IoU=0.7 per-class；RT-DETR 原生无 NMS → 论文显式注明各自后处理（架构固有差异，不强行抹平，但**记录并在导出阶段统一 max_det/conf**）。
5. **同一时序后处理**：S3 的窗口/平滑/去抖参数对三模型完全一致。

**算力现实与可完成性**：36 训练 run（3 模型×3 seed×4 arm）中 RT-DETR+InternImage 占大头（单卡估 >800h）→ 用 5070Ti+3060 双机并行不同 seed/model；**1 区增量（控制层/CRC/LTT/FIGlib 事件评测/三压力）全部在冻结权重上后处理，不增训练**；InternImage 超预算则降 2 seed 作补充结果（按规则只写补充、不写 3-seed 主结论）。

---

## 7. 最小可提交主线（可直接起草方法+实验）

### 7.1 主表 Table 1（模型 × 数据集 × 压力 × 指标）— schema

行：{yolo26n, rt_detr_l, InternImage-FRCNN} × {D-Fire, FIGlib} × {clean, S1, S2, S3}  
列：Recall@R0.90 | Precision | FPR | FPPI | mAP@.5:.95 | AP50 | AP75 | 事件 recall(FIGlib) | 聚簇 FAR/hour(FIGlib) | TTD(FIGlib)  
（已测锚点：D-Fire R0.90 YOLO baseline FPR 0.207 / hardneg 0.004；其余单元格为待跑，按 §4 预期方向填。）

### 7.2 消融表 Table 2

(a) 4 arms × 3 模型 的 R0.90 FPR/FPPI（mean±std）；(b) 控制方法轴：global / per-group / hardneg+thr / CRC / LTT / **ours**；(c) S2 注入比例 ρ 剂量行；(d) S1 severity 行；(e) S3 延迟预算档行。

### 7.3 显著性表 Table 3 + 判定规则

每个主张（hardneg<baseline、hardneg<eqstep、ours<CRC、ours<LTT、架构 A<架构 B 在某压力）报：paired Δ + **bootstrap 95% CI** + p（paired t 或 Wilcoxon）+ 效应量（Cohen's d / Cliff's δ）+ MDE。

**预注册判定规则（全满足才声明主张）**：
1. paired Δ 的 bootstrap 95% CI **不含 0**；
2. p<0.05（**Holm–Bonferroni** 跨 recall targets/baselines/datasets 校正后）；
3. 效应量 |d|≥0.8 或 |Cliff's δ|≥0.474（large）；
4. 实际差值 ≥ **MPD**（最小实际差，预设：ΔFPR≥0.02 / ΔFAR≥0.1 per hour / ΔTTD 不劣化超 1 帧）；
5. 3 seed 全同向、无反例。  
任一不满足 → 降级表述（"均值下降但 CI 触 0 / 存在 seed 反例"）。

### 7.4 失败案例分析图列表

定量：(F1) S1 severity-sweep（recall vs s，三模型）；(F2) S2 distractor-FPR vs 注入比例 ρ；(F3) S3 FAR–TTD Pareto 前沿；(F4) 三架构失败模式雷达图（S1/S2/S3 脆弱性）。  
定性：(F5) S1 低对比/夜光漏检对比；(F6) S2 火焰相似误检（反光/烟团，谁误报）；(F7) S3 聚簇告警时间线（去抖前后）；(F8) FIGlib LOSO 最差站点失败案例。

### 7.5 可复现字段

seed {11,22,33(,44,55)}；超参（§6 表）；随机源（torch/cuda/numpy seed、deterministic=True、cudnn.benchmark=False）；训练日志键值（epoch, loss 分量, lr, time）；结果文件名（`formal_results/.../<family>_seed<>/<arm>.csv`、`run_meta.json`、`*.pred_meta.json`、`gonogo.json`、`stage5_*_summary.md`、`figlib_raco_*.json`）。

---

## 8. 强制结论项

### 8.1 ≤200 字摘要：为什么之前主线薄弱

之前主线的命题是"加硬负样本能在固定召回降火/烟误报"。D-Fire 上证据干净（FPR 0.207→0.004，3 seed 无反例）、可复现性几近满分，但作为 1 区命题有三处硬伤：一，命题本身是检测领域既有常识（加背景负样本降误报是官方建议），不构成新知识；二，唯一架构创新 DAQ 在外部负样本上实测变差（0.073→0.237）、已降级，方法贡献归零；三，主效应只在单一 D-Fire，外部仅 107 张负样本，无跨域、跨相机、事件级证据。结果是结构扎实但贡献类型偏弱的二/三区论文——可复现性救不了"不新+不泛化"。升级线通过"架构×失败模式"科学问题 + FIGlib 事件级跨域 + 统一算子点控制把它抬到有条件 1 区。

### 8.2 实验矩阵（可直接给审稿人）

| 轴 | 取值 | 数量 |
|---|---|---|
| 检测器族 | yolo26n / rt_detr_l / InternImage-T+FRCNN(box) | 3 |
| 训练 arm | baseline / baseline_eqstep / hardneg / hardneg_sched | 4 |
| seed | 11 / 22 / 33（功效不足升 44/55） | 3(–5) |
| 数据集角色 | D-Fire(训练+域内表) / FIGlib(事件级 headline) | 2 |
| 算子点控制 | global / per-group / hardneg+thr / CRC / LTT / **ours** | 6 |
| recall target | 0.80 / 0.85 / **0.90(headline)** / 0.95 | 4 |
| 外部压力 | S1 退化 / S2 干扰物 / S3 时序 | 3 |

训练 run = 3×4×3 = **36（300ep）**；压力×控制×模型×数据 = 冻结权重后处理网格（不增训练）；FIGlib 事件族 LOSO。

### 8.3 结果呈现框架（表名/图名）

- 主表：**Table 1 — Cross-Architecture Operating-Point & Event-Level Benchmark under S1/S2/S3**（模型×数据×压力×指标）。
- 域内子表：**Table 1b — D-Fire Fixed-Recall FPR/FPPI @R{0.80–0.95}**。
- 消融图/表：**Table 2 — Arm & Control-Method Ablation**；**Fig. F1 severity-sweep**；**Fig. F2 distractor-FPR vs ρ**；**Fig. F3 FAR–TTD Pareto**；**Fig. F4 architecture failure radar**。
- 显著性表：**Table 3 — Paired Δ with Bootstrap 95% CI, Holm-corrected p, Effect Size & MDE**。
- 失败分析：**Fig. F5–F8（定性）**；**附录 Leakage Audit Table（站点/事件族/近邻帧去重 + split md5）**。

### 8.4 最终结论：达 1 区的边界条件

**这套锁定线在以下边界条件全部满足时，有机会达 1 区（CVPR/ICCV/ECCV/TPAMI-benchmark 量级）；否则落强 Q1 期刊（TIP/PR/TPAMI minor）。**

1. **负样本/干扰物构造合理性**：S1/S2 用物理可解释模型，并在 FIGlib 真实退化帧上 real-anchoring（方向一致）。
2. **跨场景验证充分**：FIGlib **leave-station-out** + 事件族 fold；负时长 proxy 与 per-frame FP 分开报、口径可辩护。
3. **统计功效达标**：事件族 bootstrap CI + 效应量 + MDE，且通过 §7.3 五条规则；n=3 功效不足时升到 5。
4. **方法贡献成立**：统一算子点控制层在固定（事件）召回下**显著击败 CRC/LTT**，跨架构跨站点稳定。
5. **发现非平凡**：架构×失败模式交互有**机制归因**（注意力/proposal/感受野），不是黑箱比分。

---

## 9. 仍不足之处点名 + 替代（仅显式触发条件下提交）

| 对象 | 触发条件（硬性不可行） | 可执行替代 | 替换原因 | 损失成本 |
|---|---|---|---|---|
| **InternImage-T + FRCNN** | 12G 卡上 800×1333 / bs2 **显存溢出** 或 90h 内**不收敛** | ResNet-50-FRCNN @1000×600 | 保留"非 YOLO/非 DETR 两阶段 CNN"族，砍显存 | backbone 现代性↓，AP 预期↓1–3 点，"强 backbone"卖点弱化 |
| **FIGlib** | 全量数据**不可恢复** 或负时长口径**无法辩护** | (a) PyroNear/其他 wildfire camera 库补跨域；(b) 退守只报 per-frame FP、不报 FAR/hour | proxy 口径不可辩护 | (a) 需新 manifest+审计、延期；(b) 部署叙事弱化 |
| **统计功效** | 预注册 MDE > MPD（n=3 检不出真差） | seed 升到 5 | 功效不足 | 训练量 ×1.67 |
| **D-Fire** | 无（主训练集合理）；若被质疑单一来源 | BoWFire/DeepQuest 作 image-level negative 补充压力 | 仅补充、非 headline | 轻微 |

> 触发条款守则：以上替代**默认不启用**；只有命中对应硬性不可行、且附 VRAM/收敛/功效证据日志时才提交。**任何情况下不得把第三模型回退为 YOLO/DETR 变体。**

---

### 一句话最终判断

锁定这套配置后，主线已从"单数据集 hardneg 经验观察"升级为"**架构条件化失败模式表征 + 固定相机事件级算子点控制**"——科学问题可证伪、跨域有 FIGlib、指标超越 mAP、统计可严谨。**这是有条件的 1 区**：边界全过冲顶会，过半数过强 Q1，方法贡献（控制层）是把它从"好 benchmark"推过"1 区"门槛的关键杠杆。
