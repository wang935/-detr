# 固定相机火焰识别主线：1 区可发表性审计与主线升级方案

**日期**：2026-06-09  
**视角**：CVPR / ICCV / ECCV / TPAMI 级审稿人 + 实验设计顾问  
**审计对象**：`D:\detr_Q3` 现有 Stage-5 / PV-v2 主线（固定召回误报控制 + 硬负样本训练）  
**硬约束**：固定摄像头野火识别；3 模型 / 2 数据集 / 4 arms / 300 epoch / 3 外部压力；已确认 yolo26n、rt_detr_l；已确认 D-Fire；第三模型可非 YOLO/DETR；默认不许换回 YOLO/DETR 族。

> 一句话结论（先给）：**现有主线"硬负样本在固定召回下降低误报"在 D-Fire 上证据干净、可复现性强，但作为命题它是检测领域的既有常识，泛化与方法新颖性不足，按现框架是 SCI 二区/三区，不是 1 区。** 你自己 idea-stage 里的 RACO-Wildfire（FIGlib 事件级固定召回控制）才是 1 区形状，且能复用现有全部资产。下文给出把现有资产"重新挂载"到 1 区命题上的最小改动路线。

---

## 0. 我从你文件夹里读到的事实基线（审计前提，全部可核对）

| 事实 | 来源 | 对 1 区判定的意义 |
|---|---|---|
| YOLO26n R0.90：baseline FPR 0.2069±0.0024 → hardneg **0.0038±0.0009**；eqstep 0.1521 → hardneg 0.0038；3 seed 无反例 | `formal_results/stage5/STAGE5_REPEAT_SUMMARY.md` | 效应巨大且干净，但这是"加负样本降误报"的**已知现象** |
| RT-DETR R0.90：pos-only FPR 0.5358 → hardneg 0.1056；YOLO 0.3704 → 0.0571 | `paper_assets/STAGE4A_CLAIM_SUMMARY.md` | 跨家族一致，支撑"family-agnostic"，但仍是同一已知现象 |
| DAQ（query 校准头，原唯一方法创新）外部中性负样本 FPR 0.0730 → **0.2370（变差）** | `STAGE4A_CLAIM_SUMMARY.md` | 唯一架构级创新**外部失败**，已被降级为分析项 → 方法新颖性归零 |
| 主效应只在 **D-Fire 单一 eval pool**；外部仅 BoWFire 107 张 `not_fire` 负样本 | `FORMAL_EXPERIMENT_DESIGN.md` §8.5 | 泛化证据极薄，1 区最大短板 |
| 第二检测数据集 DFS-FIRE-SMOKE **未在盘上**；DeepQuest Fire-Smoke 是**分类集**（明确禁止用于检测指标） | `EXPERIMENT_TRACKER_PV_V2.md` Gate Log | "2 数据集"目前只有 1 个真正可用于检测训练 |
| 第三模型 InternImage-T + **Mask R-CNN** 已搭好，但只做了 `max_epochs=1` smoke test；config 强制 `with_mask=True`/`gt_masks` | `runs/internimage/.../baseline/*_min.py` | D-Fire 只有框、无 mask → **标签不匹配，必须改 Faster R-CNN** |
| FIGlib/HPWREN：manifest、事件族 fold、label/visual-purity 审计、RACO 冻结评分脚本、YOLO-seed22 评分样本**都已存在**；但全量影像与冻结 RACO 评测**未跑** | `idea-stage/FIGLIB_*`、`scripts/figlib_raco_*` | 1 区路线的关键路径，已完成 70% 前置 |
| RT-DETR ~608 s/epoch（≈50 h/300ep）；YOLO26n 远快；InternImage 1333×800 / samples_per_gpu=1 最慢 | `outputs/exp_data/detr/.../results.csv` | 36 次 300ep 训练是真实算力瓶颈 |
| 4 arms：baseline / baseline_eqstep / hardneg / hardneg_sched（sched `unique_positive_coverage=0.7377`） | `EXPERIMENT_PLAN_PV_V2.md` | sched 牺牲正样本覆盖，须按"剂量+覆盖代价"解读 |

**关键洞察（决定全文走向）**：1 区升级的增量（控制器对比 CRC/LTT、FIGlib 事件级评测）**几乎都是在已冻结的 detector 输出上做后处理**，不会让 36 次训练翻倍——所以升级在算力上是可行的。

---

## 1. 现有主线 1 区可发表性风险审计（打分，满分 10）

针对**现框架**（D-Fire 上"hardneg 在固定召回降误报" + PV 动机）：

| 维度 | 分数 | 判据（审稿人会怎么说） |
|---|---:|---|
| **创新性** | **3/10** | 命题 = "加背景/负样本图降低误报"，是 Ultralytics 官方建议、检测领域教科书级常识；"报固定召回 FPR/FPPI 而非只报 mAP"在行人检测/医学 CAD 早有 FROC/FPPI 传统；唯一架构创新 DAQ 已外部失败被降级。无存活的方法贡献。 |
| **泛化性** | **3/10** | 主效应单数据集（D-Fire）；外部仅 107 张负样本；DFS 缺盘；"PV-motivated"是动机改标签、无 PV 数据。跨域/跨相机/事件级证据为零。 |
| **鲁棒性** | **4/10** | 3 个外部压力尚未落地；目前只有干净实验室条件下的算子点；无 corruption/OOD/时序压力曲线。 |
| **可复现性** | **8/10** | **真正的强项**：calib/test 互斥、md5 指纹、deterministic、last.pt、export_conf≤0.001、strict 聚合拒绝不一致 split、预注册 go/no-go。优于绝大多数已发火检测论文。 |
| **统计显著性** | **5/10** | 3 seed + mean±std + paired delta + 无反例，效应量极大（in-domain 安全）；但显式放弃严谨显著性，n=3、单数据集、无 bootstrap CI、无事件级推断。推断射程有限。 |
| **实验可完成性** | **8/10** | YOLO 3 seed 已完成、RT-DETR 在跑、基建成熟。风险点：InternImage Mask R-CNN（标签错配 + 3060 上慢）与 FIGlib 全量数据。 |

**综合 1 区就绪度：约 4.2/10。** 定位：**结构完整、诚实、可复现的二区/三区应用论文**（Fire / Sensors / ESWA / PRL / IEEE Access 量级）。**按现框架达不到 1 区**——不是质量问题，是"贡献类型"问题：可复现性满分救不了"命题本身不新 + 泛化不足"。

---

## 2. 薄弱环节 + 必须修正的最小改动

| # | 薄弱环节 | 必须修正的最小改动（不推翻已有资产） |
|---|---|---|
| W1 | **命题不新**：headline 是已知常识 | 把贡献从"hardneg 有用"重铸为"**给定冻结检测器，在固定（事件）召回下的算子点控制器，击败 per-group 阈值/CRC/LTT/learned-threshold**"。hardneg 降级为**训练期一个 arm**，不是论点。 |
| W2 | **单数据集泛化** | 加一个**域真正不同、且有序列/事件结构**的第二检测集 = **FIGlib（固定相机野火）**，做 leave-station-out / 事件族 fold。BoWFire 永远只当 negative-only 压力。 |
| W3 | **没有击败强基线的方法** | 在每个 arm 的冻结分数上加 **CRC（Conformal Risk Control）+ LTT（Learn-then-Test）+ per-group 阈值** 作对照；论文控制器必须在固定召回下带 CI 地赢它们。（后处理，算力便宜） |
| W4 | **mAP 表口径**：val pool 与 calib 重叠 | 保持现状——只当 detector 质量**补充表**，不进 headline（你已标注）。 |
| W5 | **指标射程**：D-Fire FPR/FPPI ≠ 部署 FAR | headline 指标换成 FIGlib 上**事件召回 + 聚簇 FAR/hour + TTD**；D-Fire FPR/FPPI 保留为"域内算子点负担"。 |

> 这 5 条里，W1+W2+W3 是 1 区门槛；W4/W5 是表述边界。**最小改动 = 不动 36 次训练，只重写命题 + 增 FIGlib 评测 + 增 3 个后处理控制基线。**

---

## 3. 模型层面审视：够不够 1 区？候选与最优替换

**够不够**：YOLO26n（单阶段）+ RT-DETR-L（set-based transformer）两族足以支撑"现象跨家族"。但第三族能把"算子点控制 detector-agnostic"从 2 点变 3 点、显著加固。问题只在**选哪个第三模型**，以及**修掉现有 Mask R-CNN 的标签错配**。

**第三模型候选（≤3，附可审核理由）**：

| 候选 | 架构差异 | 优点 | 致命点 / 成本 |
|---|---|---|---|
| **A. InternImage-T + Faster R-CNN（box-only，去 mask）** ★最优 | 两阶段 anchor + 可变形卷积(DCNv3) CNN，与 YOLO/DETR 都不同 | 复用已搭好的 InternImage/MMDet 管线与权重；**box-only 与 D-Fire/FIGlib 标签匹配**；比 Mask R-CNN 轻 | 1333×800 在 3060 上仍偏慢，需 batch=1~2 |
| B. ResNet-50 + Faster R-CNN（box-only） | 同为两阶段 CNN | 最稳、审稿人最熟、最省显存 | backbone 较旧，"现代性"弱于 InternImage |
| C. Co-DETR / DINO / Deformable-DETR | DETR 族 | 强 | **违反"第三模型默认非 DETR"且与 RT-DETR 重叠** → 否决 |

**最优 = A：InternImage-T backbone + Faster R-CNN head（box-only，丢弃 mask 分支）。**

**替换原因（可审核，不只给名字）**：
1. **标签一致性（硬错配，必须修）**：现 config `roi_head` 带 `mask_head`、pipeline `LoadAnnotations(with_mask=True)` 与 `gt_masks`，但 D-Fire 是 YOLO 框标注、**无实例 mask**。Mask R-CNN 的 mask 分支无监督信号——要么伪造 box→mask（引入噪声、审稿人红线），要么改 Faster R-CNN。改 FRCNN 是唯一干净解。
2. **架构正交性**：FRCNN 提供"两阶段 + proposal + DCNv3"族，与 YOLO（单阶段 anchor-free）、RT-DETR（集合预测 transformer）三足正交 → "控制器与检测器架构无关"从 2 证据点升到 3。
3. **工程增量最小**：InternImage/MMDet 代码、DCNv3、2 类适配、seed split transfer 包（`transfer_packages/stage5_internimage_*`）都已存在；只需删 mask 头、关 `with_mask`。
4. **算力可控**：去 mask 分支降显存与时延，缓解 3060 风险。

**替换触发条款（遵守硬约束 #7）**：仅当 InternImage-T FRCNN 在 3060 上 **1333×800 显存溢出或 300ep 内不收敛** 时，**唯一允许**降级为 **ResNet-50 FRCNN @ ≤1000×600**——理由：仍保留"非 YOLO/非 DETR 两阶段 CNN"族、同时砍显存；必须附 VRAM/throughput 证据日志。**不得回退到任何 YOLO/DETR 变体。**

---

## 4. 数据集层面审视：分工 + 负样本覆盖修复

**两数据集明确分工**：

| 角色 | 数据集 | 职责 | 指标 |
|---|---|---|---|
| **主任务训练 + 域内算子点** | **D-Fire** | 3 族检测器训练；提供正样本 + 域内硬负样本(none/distractor)；域内固定召回 FPR/FPPI 主表（沿用 calib/test 互斥协议） | precision/recall/mAP（补充）、FPR/FPPI@R{0.80..0.95} |
| **跨域 + 事件级验证（1 区抬升点）** | **FIGlib/HPWREN** | 固定相机野火序列；leave-station-out / 事件族 fold；事件级召回 + 聚簇 FAR + 预警延迟 | 事件召回@R0.90/0.95、聚簇 FAR/hour、TTD、bootstrap CI |

**负样本覆盖不足——修复策略（逐源）**：
1. **D-Fire `none`(3380) 是泛化负样本，火焰相似干扰物不足** → 从 none 中策展/挖掘 **fire-like distractor 子集**（夕阳、车灯、红橙物、反光、低云），单列为"distractor"负样本类做误报压力（与外部压力 2 对齐）。
2. **BoWFire 107 `not_fire`** → 体量小、静态图：**严格只当 external negative-only 压力**，只报 external FPR/FPPI，**绝不**写成 recall/mAP/部署 FAR（你已有此规则，保持）。
3. **FIGlib pre-event 窗口是负时长代理，非连续真负小时**（你自评 80/100 的核心风险）→ **分开报告**："聚簇 FAR / pre-event-proxy-hour" 与 "per-frame FP" 各算各的，措辞保守；**若能恢复连续 non-event 序列**补真负小时，则风险解除。
4. **DeepQuest Fire-Smoke = 分类集** → 不进任何检测指标；最多做 image-level 正/负 sanity 压力。
5. **DFS-FIRE-SMOKE 未在盘** → 二选一：(a) 用 `scripts/stage5_pv_v2_prepare_dfs.py` 从 VOC XML 备好，作**额外**跨源负/正压力；(b) 直接以 FIGlib 为 headline 第二集、DFS 仅选配。**推荐 (b)**：FIGlib 域匹配（固定相机）远强于 DFS。

---

## 5. 三个外部压力（可参数化 / 可合成 / 无需额外硬件）

| 压力 | 触发场景 | 实施方式（增强/采样/损伤强度） | 预期检验目标（方向性） |
|---|---|---|---|
| **S1 大气/光学退化** | 雾霾、黄昏低光、镜头眩光、雨雪沾镜、传感器噪声——固定室外相机常态 | 测试池上合成 corruption，**severity s∈{1..5}**：大气散射雾、亮度/gamma（晨昏）、运动+散焦模糊、高斯/shot 噪声、JPEG 压缩 | 固定算子点下**召回保持率 vs s**；FPR/FPPI 随退化的**上升速率**；mAP 单调下降。主张：控制器比固定/CRC 阈值更能在退化下守住事件召回并压住聚簇 FAR 增速。 |
| **S2 火焰相似干扰物注入（误报压力）** | 夕阳辉光、车头/尾灯、红橙工业物、镜面反光、似烟低云——野火典型误报源 | 负样本上合成/合成贴片：色温匹配色块、眩光精灵、云纹理，**密度 d（patch/图）+ 混合 α**；或直接采样策展的 distractor 负样本 | distractor-重负样本上 **FPR/FPPI 上升**；**召回必须不掉（解耦）**。方向：hardneg arm 与控制器应大幅抑制 distractor FPR；若召回下降 → 标记过度抑制。 |
| **S3 时序/运营压力（聚簇告警 + 延迟）** | 持续良性运动（飘云、晃树、相机抖动/PTZ 微动）造成突发误报；早窗预警要求 | FIGlib 序列上注入时序抖动（丢帧、小仿射、曝光闪烁），滑窗评估告警**聚簇**；去抖参数 **k（连续帧）、窗口 W、抖动幅度** | **聚簇 FAR/hour**（非逐帧 FP）；固定事件召回下 **TTD 不得变坏**；抖动下定位漂移（中心误差）。主张：控制器降聚簇 FAR 而不增 TTD；纯阈值会顾此失彼。 |

三者全部参数化、纯合成或纯采样、复用现有 eval 池与 FIGlib 序列，**零额外硬件**。

---

## 6. 4 个 arms 的最强信息增益排布

arms 固定为 baseline / baseline_eqstep / hardneg / hardneg_sched。**最强信息增益排布 = 把它们当成"训练期负样本暴露的剂量阶梯"，再在每个 arm 的冻结分数上叠加"算子点控制方法轴"**：

| arm | 角色（信息增益） | 回答的问题 |
|---|---|---|
| 1. baseline（仅正样本） | 地板 / 无负样本暴露对照 | 不加负样本时误报底盘多高 |
| 2. baseline_eqstep（重复正样本，步数对齐） | **最高信息量的混淆控制**：隔离"步数/数据预算"与"负样本语义" | hardneg 的赢是不是只因训练步更多？没有它 hardneg 不可解释 |
| 3. hardneg（正 + 硬负样本） | 主处理 | 负样本语义本身是否降误报 |
| 4. hardneg_sched（负样本比例↑至 0.60） | **剂量-响应**臂（带正样本覆盖代价 0.7377） | 更多负样本暴露是否单调降 FP，还是触及"正样本覆盖塌缩"拐点 |

**两条正交轴一起报**（这是 1 区信息所在）：
- **训练 arm 轴**（上表 4 臂）：回答"训练期负样本暴露有没有用 + 剂量曲线"。
- **算子点控制方法轴**（在冻结 detector 上后处理）：global thr / per-group thr / **CRC / LTT** / learned-threshold / **本文控制器**——回答"给定冻结检测器，控制器是否在固定召回下击败统计基线"。

> 警示：hardneg_sched 降低 unique 正样本覆盖（0.7377），必须作为"剂量↑但覆盖↓"的**权衡曲线**报告，不能写成"越多越好"。

---

## 7. 三模型 300 epoch 统一训练预算

| 项 | YOLO26n | RT-DETR-L | InternImage-T FRCNN（box-only） |
|---|---|---|---|
| batch | 16 | 4 | 2（显存不足降 1） |
| imgsz | 640 | 640 | 1333×800（降级 1000×600） |
| 优化器 | SGD/auto | AdamW | AdamW lr 1e-4 + layer-decay |
| LR 策略 | cosine + warmup | cosine + warmup | step([0.75,0.9]×总)+linear warmup |
| AMP | ✓ | ✓ | ✓ |
| 预估显存 | ~4–6 GB | ~10–11 GB | ~10–12 GB |
| 预估时长/300ep | ~8–12 h | ~50 h（实测 608 s/ep） | 最慢，~60–90 h（须实测） |
| 早停 | **无 val 早停**（`val=false` 保持 eval 干净），固定 300ep + last.pt | 同左 | 同左 |
| 跨框架对齐 | epoch=对同一 train list 的轮次 | 同左 | MMDet 的 epoch 语义不同 → **按"等价图像呈现次数"对齐，不按 wall-clock** |

**统一评测口径（所有模型/seed 强制一致）**：同一 eval 池、calib/test 互斥同指纹、recall targets {0.80,0.85,0.90,0.95}、**R0.90 为 headline**；域内 D-Fire 报 FPR/FPPI，跨域 FIGlib 报事件召回/聚簇 FAR/TTD；seeds {11,22,33}；导出 last.pt、export_conf≤0.001、deterministic。

**算力现实**：36 次 300ep 训练（3 模型×3 seed×4 arm）中 RT-DETR+InternImage 占大头（单 GPU 估 >800 h）。**可完成性策略**：(a) 用 5070Ti+3060 双机并行；(b) headline 只锁 R0.90，其余 recall 当稳健性附录；(c) 若 InternImage 超预算，降为 **2 seed 补充结果**（按 §7.1 规则只能写补充、不写 3-seed 主结论）；(d) **1 区增量（CRC/LTT/控制器/FIGlib 事件评测）全部是冻结分数上的后处理，不增训练**。

---

## 8. 最小可交付主线（method + 实验，去掉无关备选）

> **题目骨架**：*Recall-Constrained Operating-Point Control for Fixed-Camera Early Fire/Smoke Detection*（固定相机早期火/烟检测的召回约束算子点控制）。

- **方法**：标定集上拟合的控制器，输入冻结检测器的逐帧火险分 + 站点/时段条件特征，输出"保持目标（事件）召回、最小化聚簇 FAR/hour"的决策门；hardneg 训练作为训练期一个 arm，不是方法本身。
- **数据**：D-Fire（训练 + 域内算子点表）；FIGlib（事件级跨域 headline）。
- **模型**：YOLO26n、RT-DETR-L、InternImage-T FRCNN（box-only）。
- **要击败的基线**：global thr、per-group thr、hardneg+thr、CRC、LTT、learned-threshold。
- **外部压力**：S1 光学退化 / S2 干扰物注入 / S3 时序聚簇。
- **产出**：域内固定召回表、FIGlib 事件级表（事件召回 R0.90/0.95 + 聚簇 FAR/hour + TTD + CI）、4-arm 剂量消融、控制方法消融、3 条压力曲线、泄漏审计。

这条主线**不丢任何已确认资产**（4 arms、3 模型、D-Fire、300ep、3 压力全部在内），只是把它们挂到一个能赢强基线、有跨域事件级证据的命题上。

---

## 9. 强制输出

### 9.1 最终推荐主线三元组（模型 × 数据集 × 外部压力）

**完整 headline 矩阵**：
{YOLO26n, RT-DETR-L, InternImage-T FRCNN} × {D-Fire(训练+域内), FIGlib(事件级)} × {S1 光学退化, S2 干扰物注入, S3 时序聚簇}

**若只能锁一个"核心三元组"**（用于最强单点叙事）：  
**RT-DETR-L × FIGlib × S2 干扰物注入** —— transformer 检测器 + 固定相机野火事件评测 + 火焰相似误报压力，最直击"野火误报"痛点与"控制器降 FAR"主张。

### 9.2 替换建议汇总（每条附可审核理由）

| 替换 | 从 → 到 | 可审核理由 |
|---|---|---|
| 第三模型头 | Mask R-CNN → **Faster R-CNN（box-only）** | D-Fire/FIGlib 无实例 mask，mask 头无监督信号；现 config `with_mask=True`+`gt_masks` 是标签错配红线 |
| 第二数据集 | DeepQuest 分类集 → **FIGlib（检测/事件）** | DeepQuest 仅图像级分类、无框，无法算检测/事件指标 |
| 第二数据集（备选） | DFS-FIRE-SMOKE | 未在盘；域匹配弱于 FIGlib；保留为选配跨源压力，非 headline |
| 命题框架 | "PV-motivated 部署" → **固定相机野火/事件级控制** | 无 PV 数据，PV 是改标签；FIGlib 提供真实固定相机野火事件 |
| 原方法 | DAQ 作主方法 → **降级为 DETR query 域内分析** | 实测外部 FPR 0.0730→0.2370 变差，外部泛化失败 |
| 第三模型降级（仅触发时） | InternImage-T FRCNN → ResNet-50 FRCNN @≤1000×600 | 仅当 3060 显存溢出/不收敛；保留非 YOLO/DETR 两阶段族；须附 VRAM 日志 |

### 9.3 可直接执行实验清单（有前后顺序）

1. **修第三模型**：InternImage config 删 mask 头、关 `with_mask`/`gt_masks`，改 Faster R-CNN，2 类；1 epoch smoke test 过 → 记录 VRAM/throughput（决定是否触发 §9.2 降级）。
2. **备 FIGlib 全量数据**：跑 `figlib_url_preflight.py` → `figlib_manifest_builder.py` → `figlib_split_audit.py`；冻结事件族 fold（leave-station-out）、做泄漏键审计（站点/事件族/近邻帧去重）。
3. **训练 36 run**：3 模型 × seed{11,22,33} × 4 arm × 300ep（双机并行；YOLO 先行、RT-DETR/InternImage 排队）；全程 `val=false`、last.pt、export_conf≤0.001、记 md5/run_meta。
4. **域内导出 + 固定召回表**：D-Fire calib/test 互斥，导出预测，算 R{0.80..0.95} 的 FPR/FPPI（沿用 `stage5` 协议）。
5. **冻结分数后处理（1 区核心，便宜）**：在每个 arm 的冻结分上实现并对比 global/per-group thr、**CRC、LTT**、learned-threshold、**本文控制器**。
6. **FIGlib 事件级评测**：`figlib_raco_score_manifest.py` → `figlib_raco_frozen_eval.py`，报事件召回@R0.90/0.95、聚簇 FAR/hour、TTD；pre-event-proxy 与 per-frame FP 分开报。
7. **三压力曲线**：S1/S2/S3 各扫参数，报召回保持/FPR 上升/聚簇 FAR/TTD 方向性。
8. **统计**：事件族 bootstrap CI + paired Δ（控制器 vs 各基线）+ 3-seed mean±std + 无反例检查 + Holm–Bonferroni 多重校正。
9. **汇总与写作**：主表、消融图、显著性表、泄漏审计、限制与失败分析（含 DAQ 外部失败、pre-event proxy 风险）。

### 9.4 预注册 / 复现字段

| 字段 | 取值 / 规则 |
|---|---|
| **seeds** | 11, 22, 33（每族每 arm 必须 3 个完整 run；不足只能写补充） |
| **训练** | epochs=300, val=false, deterministic=true, amp=true, 导出 last.pt, export_conf≤0.001 |
| **split** | full/calib/test 三文件 md5 指纹；事件族 fold（站点级 leave-station-out）；strict 聚合拒绝指纹不一致 |
| **headline 指标** | 域内：FPR/FPPI@R0.90；跨域 FIGlib：事件召回@R0.90/0.95、聚簇 FAR/hour、TTD |
| **显著性检验** | 事件族 **bootstrap 95% CI**（事件召回 & 聚簇 FAR）；控制器 vs 每个基线的 **paired ΔFAR bootstrap CI**；多 recall/多基线用 **Holm–Bonferroni** |
| **判定规则** | 仅当 ① paired ΔFAR 的 95% CI 不含 0，且 ② 3 seed 全部同向、无反例，且 ③ 校正后仍显著 → 才可写"控制器在固定召回下显著降 FAR"；任一不满足 → 降级为"均值下降但…" |
| **禁写** | SOTA 检测器、部署级真实 FAR、单 D-Fire split 证明外部泛化、把 proxy 负时长写成真负小时、DAQ 泛化可靠 |

---

## 摘要：为什么当前主线不够强（≤200 字）

当前主线的命题是"加硬负样本能在固定召回下降低火/烟误报"。D-Fire 上证据干净（FPR 0.207→0.004，3 seed 无反例）、可复现性几乎满分。但作为 1 区命题它有三处硬伤：一，命题本身是检测领域既有常识（加背景负样本降误报是官方建议），不构成新知识；二，唯一的架构创新 DAQ 在外部负样本上实测变差、已被降级，方法贡献归零；三，主效应只在单一数据集 D-Fire，外部仅 107 张负样本，无跨域、跨相机、事件级证据，"PV 动机"只是改标签而无 PV 数据。结果是一篇结构扎实但贡献类型偏弱的二/三区应用论文——可复现性救不了"不新 + 不泛化"。

---

## 附 A：可直接交付审稿人的实验矩阵

| 轴 | 取值 | 数量 |
|---|---|---|
| 检测器族 | YOLO26n / RT-DETR-L / InternImage-T FRCNN | 3 |
| 训练 arm | baseline / baseline_eqstep / hardneg / hardneg_sched | 4 |
| seed | 11 / 22 / 33 | 3 |
| 数据集角色 | D-Fire（训练+域内表） / FIGlib（事件级 headline） | 2 |
| 算子点控制方法 | global / per-group / hardneg+thr / CRC / LTT / learned / **本文控制器** | 7 |
| recall target | 0.80 / 0.85 / 0.90(headline) / 0.95 | 4 |
| 外部压力 | S1 退化 / S2 干扰物 / S3 时序 | 3 |

训练 run = 3×4×3 = **36**（300ep）；控制方法 × arm × 模型 × 数据集 = 后处理网格（不增训练）；事件族 fold 做 leave-station-out。

## 附 B：可直接交付写作的结果呈现框架

**主表（Table 1）**——FIGlib 事件级，R0.90：行=控制方法×检测器族，列=事件召回 / 聚簇 FAR/hour / TTD / 95% CI；加 D-Fire 域内 FPR/FPPI 子表。  
**主表（Table 2）**——域内 D-Fire 固定召回：4 arm × 3 族 × R{0.80..0.95} 的 FPR/FPPI（mean±std）。  
**消融图 1**——4-arm 剂量曲线：x=负样本暴露（baseline→eqstep→hardneg→sched），y=R0.90 FPR/FPPI，叠正样本覆盖代价轴。  
**消融图 2**——控制方法对比：固定召回曲线（recall 0.80–0.98）下各方法的聚簇 FAR，本文控制器 vs CRC/LTT。  
**消融图 3**——三压力曲线：S1 severity / S2 distractor 密度 / S3 抖动幅度 对召回保持、FPR、聚簇 FAR、TTD 的方向性影响。  
**显著性表（Table 3）**——控制器 vs 每个基线的 paired ΔFAR + bootstrap 95% CI + Holm 校正 p；逐 seed 明细 + 无反例标记。  
**泄漏审计表（附录）**——站点/事件族/近邻帧去重前后规模；split 指纹 md5。  
**失败分析（必写）**——DAQ 外部失败（0.073→0.237）、FIGlib pre-event proxy 口径、hardneg_sched 正样本覆盖代价。

---

## 最终结论：是否达 1 区口径及边界条件

**按现框架（D-Fire 上 hardneg 降误报）：否，是二/三区。**

**按本方案重铸（召回约束算子点控制 + FIGlib 事件级 + 击败 CRC/LTT + 3 模型 3 压力）：可冲 1 区，但是"有条件的 1 区"（约 75–82/100，与你 idea-stage 自评一致）。** 边界条件（全部满足才成立）：

1. **FIGlib 全量 + 事件级评测真正跑通**，且 leave-station-out 下控制器带 CI 地赢 CRC/LTT（不只赢朴素阈值）。
2. **负时长口径站得住**：pre-event proxy 与 per-frame FP 分开报，或补连续真负小时；否则聚簇 FAR/hour 的部署解读必须保守。
3. **第三模型标签错配修掉**（Mask→Faster R-CNN），三族都有 3 seed（InternImage 至少 2 seed 补充）。
4. **方法贡献是"控制器"而非"hardneg"**；hardneg 仅作 arm；DAQ 仅作分析。
5. **统计达事件族 bootstrap CI + 多重校正**，不是只报 mean±std。

**未达上述任一 → 回落二区**（仍是好论文，但不是 1 区）。

### 可发表性短板点名 + 替代

- **InternImage-T + Mask R-CNN**：mask 标签错配 → 换 **Faster R-CNN（box-only）**；超预算再降 ResNet-50 FRCNN@≤1000×600。
- **DeepQuest Fire-Smoke 分类集**：无框 → **排除**出检测指标，仅 image-level sanity。
- **DFS-FIRE-SMOKE**：未在盘、域匹配弱 → **以 FIGlib 替代为 headline 第二集**，DFS 选配。
- **BoWFire（107 负样本）**：体量小、静态 → 仅 **external negative-only 压力**，不进 headline。
- **FIGlib pre-event 负时长**：proxy 非真负小时（你自评 80/100 的命门）→ **分开报 + 保守措辞**，能补连续真负则解除。
- **DAQ**：外部失败 → 永久降级为 DETR query 域内机制分析。
