# 可融合方法提案：固定召回约束下的固定相机火/烟检测（RACO × OPAL）

**日期**：2026-06-10　**定位**：本提案给出 `CV_Q1_MAINLINE_DESIGN_20260609.md` §4 Arm-D 中留空的"统一算子点控制层（ours）"的具体方法定义（方案1），及训练轴的可选增强（方案2）。零资产丢弃：36 训练 run、4 arms、3 模型、D-Fire/FIGlib、S1–S3 全部保留。

**事实锚点（已测 vs 待跑，写作时必须区分）**
- 已测：YOLO26n D-Fire R0.90 FPR baseline 0.2069±0.0024 → hardneg 0.0038±0.0009（eqstep 0.1521；3 seed 无反例）；RT-DETR pos-only 0.5358 → hardneg 0.1056；DAQ 外部失败 0.0730→0.2370。
- 待跑：FIGlib 全量冻结评分、全部控制方法轴、S1–S3 压力、InternImage-FRCNN 修复后训练。
- FIGlib 负时长 proxy = 269.54 h（仅 proxy 口径，与 per-frame FP 分开报）。

**实测锚点（2026-06-10 在本仓库冻结分数上复算；脚本 `scripts/crc_anchor_demo.py`、`scripts/crc_boot_demo.py`）**

协议复现校验：seed11 baseline R0.90 → τ=0.5245 / recall 0.9028 / FPR 0.2095，与 `gonogo.json` 逐位一致；以下数字均在该协议上计算（YOLO 3 seed；RT-DETR pv_v2 仅 seed 11/22 在盘，按 2-seed 报）。

1. **组异质性（方案1组条件控制的实测动机）**：全局 τ(R0.90) 下，test 集分类召回——YOLO smoke 0.837 / fire 0.971（gap 0.13），RT-DETR smoke 0.851 / fire 0.959（gap 0.11），baseline 与 hardneg、全部 seed 一致。即名义 0.90 聚合召回下，**早预警最关键的 smoke 类只有 ~0.84**。图2 动机已有域内实测版本。
2. **小标定集违约率（H1 机制域内已可证）**：500 次标定子采样、test 实现召回 < 0.90 的违约率：

   | Ncal | PR-pick 违约 / FPR | CRC 违约 / FPR | LTT(δ=0.05) 违约 / FPR |
   |---|---|---|---|
   | 2006(全) | 0.0% / 0.0036 | 0.0% / 0.0036 | 0.0% / 0.0036 |
   | 500 | 45.6% / 0.0030 | 38.2% / 0.0032 | **1.4% / 0.0042** |
   | 200 | 49.0% / 0.0028 | 39.6% / 0.0031 | **1.8% / 0.0054** |
   | 100 | 54.8% / 0.0026 | 42.4% / 0.0032 | **1.8% / 0.0084** |

   （YOLO hardneg seed11；baseline 与 RT-DETR 同模式。）FIGlib LOSO 每 fold 标定事件量正处于 100–500 量级 → **headline 协议应预注册 LTT(δ=0.05)**；其 FPR 代价在 hardneg 分数上仅 +0.001~0.003。
3. **smoke 锚定 CRC（最难组担保）**：τ 由 smoke 正样本单独标定——hardneg 上近乎免费：YOLO smoke 召回 0.899±0.002、FPR 0.0038→**0.0059**；RT-DETR smoke 0.913、FPR **0.0050**。baseline 上昂贵（YOLO FPR 0.2095→0.2558）。**训练轴×控制轴交互的直接证据：组保证的价格取决于训练出的分数分离度** → 双轴叙事成立。
4. **证书价格 ∝ 分数分离度（OPAL 联动的域内代理）**：Ncal=100 时 LTT 的 FPR 代价 YOLO hardneg +0.006 vs RT-DETR hardneg +0.030——分离度差的分数为证书付更高价；OPAL 的"认证效率"增益由此在域内即可量化（Δ认证FPR@等保证）。
5. **工程注记**：现导出 CSV 无 class 列（class-agnostic max conf）——真 per-class/组条件阈值需在导出协议中加 class 列（小改）；smoke 锚定单阈值不受影响。

---

## 方案 1（可融合，优先级最高，A 类）

### 1.1 方法定义

**方法名**：**RACO — Recall-Anchored Conformal Operating-point control**（与仓库现有 `figlib_raco_*` 脚本命名连续；中文：召回锚定的保形算子点控制）。

**核心机制**：在冻结检测器分数上，把"固定召回"从一个超参选择问题改写为**分布无关、有限样本的风险控制问题**，分三层：

(i) **事件级 TTD 预算风险**。对事件 e（站点 s(e)，首次可见烟时刻 t0(e)），告警规则 A_λ 由配置 λ = (τ_g, k, n) 定义：滑窗 n 帧内 ≥k 帧的帧级分 s_t = max_j conf_{t,j} 超过组条件阈值 τ_{g(t)} 才触发。事件损失：

  L_e(λ) = 1[ T_alarm(e; λ) > t0(e) + Δ ]　　(1)

即"Δ 分钟预算内未确认告警记一次漏报"。L_e 对 τ 单调非减、对 k 单调非减——满足 CRC 单调性。Δ∈{5,10,20} min，headline Δ=10。Δ=∞ 退化为普通事件召回，作对照列。

(ii) **CRC 阈值（单调维）**。给定标定事件集 {e_1..e_N}（来自 K−1 个站点），经验风险 R̂_N(τ) = (1/N)Σ L_e(τ)，取

  τ̂ = sup{ τ : (N·R̂_N(τ) + 1)/(N+1) ≤ α }　　(2)

则对可交换的新事件 E[L(τ̂)] ≤ α，即**事件召回 ≥ 1−α 的有限样本保证**（α=0.10 ↔ R0.90；0.05/0.15 ↔ 0.95/0.85 对照）。

(iii) **可迁移分组 + 层级收缩 + LTT 联合认证（非单调维）**。
- 分组 g 不用站点 ID（LOSO 下测试站未见过），用**可迁移协变量**：g = 昼/夜（太阳高度角）× 场景上下文簇（背景帧 embedding 的 k-means 簇，新站点可直接归簇）。这是对标准 Mondrian/组条件 CRC 的关键改造点。
- 小组风险收缩：R̃_g(τ) = w_g·R̂_g(τ) + (1−w_g)·R̂_pool(τ)，w_g = n_g/(n_g+κ)。收缩破坏 CRC 精确性 → 收缩产生的候选配置交给 LTT 重新检验，validity 由检验恢复。
- **LTT**：对候选网格 λ ∈ Λ（τ_g × k × n，含收缩候选），检验 H_λ: R(λ) > α，Hoeffding–Bentkus p 值：

  p_λ = min( e·P[Bin(N,α) ≤ ⌈N·R̂_N(λ)⌉], exp(−N·h1(min(R̂,α),α)) )　　(3)

  按"预期 FAR 升序"的 fixed-sequence 做 FWER ≤ δ（δ=0.05）筛出可行集 Λ̂，最终 λ* = argmin_{λ∈Λ̂} FÂR/hour(λ)。输出保证：P( EventRecall(λ*) ≥ 1−α ) ≥ 1−δ。
- 样本量约束（写进 limitation）：组级证书非平凡要求 n_g ≥ (1−α)/α（α=0.10 → ≥9 事件/组；α=0.05 → ≥19）；不足的组回退 pooled 证书 + 经验诊断。

**为什么在固定召回约束下天然有用**
1. 固定召回本来就是风险约束而非指标偏好；CRC/LTT 正是"约束风险、最优化效率（FAR/hour、TTD）"的标准机器——把现有"calib 上 PR 曲线选阈值"升级为带证书的选择，且**零重训**。
2. 证书只依赖分数的序统计量，**对置信度失准免疫**——直接回收 DAQ 外部失败（0.073→0.237）作为动机证据："校准头会跨域失效，分布无关证书不会"。
3. LOSO 下"名义召回 vs 实现召回"的违约率本身可测——朴素阈值法预期显著违约，RACO 预期贴线——这给出论文的 money figure（图2）。

### 1.2 论文可执行主线

**标题候选**
- *Certify, Don't Hope: Recall-Certified Operating-Point Control for Fixed-Camera Wildfire Smoke Detection*
- *Recall-Anchored Conformal Alarm Control for Early Fire Detection under Site Shift*
- *Distribution-Free Operating Points: Event-Level Recall Guarantees with Minimal False Alarms*

**研究主问题**：在站点级分布偏移（LOSO）下，能否对任意冻结检测器给出有限样本的事件级召回（含 TTD 预算）保证，同时把聚簇 FAR/hour 压到统计基线之下？

**假设**
- H1（有效性）：LOSO 下 RACO 的实现事件召回违约率 ≤ 名义水平（binomial 带内），而 PR 曲线选点 / learned-threshold 的违约率显著超标。
- H2（效率）：等认证召回（R0.90, Δ=10min）下，RACO 的 FAR/hour 显著低于 global-CRC 与 LTT-τ-only（paired bootstrap 95% CI 不含 0，Holm 校正），且 TTD 中位数劣化 ≤ 1 帧（1 min）。

**技术创新点（≤4）**
1. 事件级、TTD 预算内的保形风险定义（式1）——首次把"预警及时性"写进可认证损失（现有 conformal detection 工作均为 per-image/box 召回）。
2. LOSO 可迁移分组的组条件认证（场景簇 + 昼夜，而非站点 ID）+ 小组层级收缩再认证。
3. (τ_g, k, n) 联合配置空间上的 LTT/FWER 搜索，输出"认证算子流形"并以 FAR/hour 为效率目标。
4. 检测器无关性：同一证书框架横跨 YOLO26n / RT-DETR-L / InternImage-FRCNN 三个架构族，并用 DAQ 失败案例论证免校准的必要性。

**可复现实验协议**
- 数据：D-Fire（calib/test 互斥、md5 指纹，组=类别×尺度 bin，损失=per-GT-box miss@IoU0.5）；FIGlib（事件族 leave-station-out fold，冻结 manifest）。
- 划分：标定=K−1 站点的事件，测试=留出站点；D-Fire 沿用现协议。
- seeds：{11,22,33}（训练侧）；控制层为确定性后处理，另设标定重采样 ×100 评估证书方差。
- 指标：实现事件召回与违约率、FAR/hour（pre-event-proxy 口径）与 per-frame FP（双口径分开报）、precision@R、TTD 中位/p90、D-Fire FPR/FPPI@R{0.85,0.90,0.95}。
- 显著性：事件族 cluster bootstrap 95% CI（10k 重采样）、per-fold paired Δ + permutation/Wilcoxon、Holm–Bonferroni（跨 R0×基线）、Cliff's δ、MDE；沿用 §7.3 五条预注册判定规则。

### 1.3 与现有 pipeline 兼容性

**最小改动**：训练/推理代码零改动。控制层是纯后处理（numpy，约 300–500 LOC）：输入统一预测 CSV（image, conf, bbox）+ 元数据，输出 λ* 与证书报告。

**需新增组件**
1. 场景上下文分组器：每站点取 pre-event 背景帧 → 冻结 backbone embedding → k-means（k∈{4,8}）；新站点推断期归簇。
2. 事件聚合器：帧分 → 事件序列（`figlib_raco_score_manifest.py` 已有 70%）。
3. CRC/LTT 标定器 + 违约审计器（输出图2 数据）。
4. D-Fire 侧 per-GT-box miss 损失提取（现有 stage5 评测器小改）。

**与三模型融合（逐项）**
- YOLO26n：现有导出协议（export_conf≤0.001、max_det=300、NMS IoU 0.7）即满足输入要求；帧分 = top-1 conf。
- RT-DETR-L：无 NMS，取 top-300 query 分；其外部失准恰是卖点——RACO 只用序信息；可选温度缩放仅影响效率不影响有效性（作消融行）。
- InternImage-FRCNN（box-only 修复后）：post-NMS 分数同 CSV schema；若降级 ResNet-50-FRCNN，控制层不受影响（检测器无关性论证反而更强）。

**最小可行 3-seed 矩�阵（全部后处理，不增训练）**
{YOLO26n, RT-DETR-L} × {baseline, hardneg} × 控制轴{global-PR-pick, global-CRC, group-CRC+收缩, LTT-τ-only, RACO} × R0{0.85,0.90,0.95} × seeds{11,22,33}，FIGlib LOSO 全 fold + D-Fire 域内表；InternImage 以 2-seed 补充列。计算量 ≈ CSV 扫描级。

**能否直接进论文：高。** 它就是主线设计里 Arm-D 等待的"ours"，且击败对象（CRC/LTT 朴素版）已内置为基线。

**预期增益（3）**：① 补上 W3"无可击败强基线的方法贡献"，是 1 区门槛的关键杠杆；② 图2（名义 vs 实现召回违约）预期视觉冲击强、可证伪；③ 把 FIGlib proxy 口径争议转化为"效率目标的两种估计"，措辞可辩护。
**风险（3）**：① 站点不可交换 → 保证只在事件边际意义下成立，LOSO 个别离群站点可能违约——必须如实报告并靠可迁移分组缓解；② 每 fold 标定事件量决定证书松紧（(N+1)α 量化），α=0.05 时小组证书可能过保守 → FAR 抬升，需收缩+回退；③ 与 Andéol & Mossina "Conformal Object Detection by Sequential Risk Control"（arXiv 2505.24038）及 2026 野火疏散 CRC（arXiv 2603.22331）的撞车风险——差异化锚点：事件级+TTD 预算损失、FAR/hour 效率目标、LOSO 可迁移分组、固定相机视频流（彼为 per-image box 召回 / 蔓延制图）。相关工作必须正面引用并对表。

---

## 方案 2（可融合，B 类，训练轴增强）

### 2.1 方法定义

**方法名**：**OPAL — Operating-Point-Aligned Learning**（算子点对齐训练：高召回区 partial-AUC 代理损失 + 同相机干扰物课程）。

**核心机制**：mAP 训练把梯度摊在整条 PR 曲线上，而部署只用 R≥0.90 一个点。OPAL 把梯度集中到"最难负样本 vs 最弱正样本"的决策边界：

(i) **高召回区 pAUC 排序代理**。批内正集 P =（匹配到 GT 的候选分），负集 N =（负样本/干扰物图上的未匹配候选分），取 top-k 难负（k/|N|=β，对应单向 pAUC 的 FPR∈[0,β] 区段）：

  L_pAUC = (1/(|P|·k)) Σ_{p∈P} Σ_{n∈TopK(N)} [ m − s_p + s_n ]²₊　　(4)

(ii) **召回锚（对偶上升）**：min_θ L_det + γ·L_pAUC + μ_t·(R0 − T̂PR_soft(θ))，μ_{t+1} = [μ_t + η(R0 − R̂_calib)]₊ ——pAUC 压 FP 时防召回塌缩。

(iii) **同相机干扰物课程**：负样本源 = FIGlib 训练 fold 站点的 pre-event 帧（同机位、同光学、零标注成本的"最难负样本"）+ D-Fire none 中策展的 fire-like distractor；注入率 ρ(t) = ρ_max·min(1, t/T_w) 线性升温，并加**覆盖闸门：unique positive coverage ≥ 0.95**（直接修复 hardneg_sched 已测的 0.7377 覆盖塌缩）。

**为什么在固定召回约束下天然有用**：固定召回算子点的 FPR 由"τ(R0) 附近的分数重叠区"决定；式(4) 恰好最大化该区间的分离度 → 直接抬升 precision@R0.90；同时与方案1 形成闭环——分数分离度提高 ⇒ LTT 可行集 Λ̂ 中更高的 τ 可过检 ⇒ **同一证书下 FAR/hour 更低**（"认证效率"可量化为 Δτ_feasible / ΔFAR）。

### 2.2 论文可执行主线（作为从属贡献；亦可独立成文）

**标题候选**（独立成文时）
- *Training Where You Operate: Operating-Point-Aligned Learning for High-Recall Fire Detection*
- *Same-Camera Hard Negatives for Free: Distractor Curriculum from Pre-Event Wildfire Footage*

**研究主问题**：把训练目标对齐到部署召回区并使用同相机零成本难负样本，能否在固定（认证）召回下系统性降低跨域 FAR？

**假设**
- H1：等召回（R0.90）下，OPAL 相对 hardneg 在 FIGlib LOSO 的 FAR/hour 与 S2 distractor-FPR 显著下降（域内 D-Fire 已近地板 0.0038，不作主战场）。
- H2：OPAL 在覆盖闸门约束下不损失事件召回与 TTD（解耦检验），且增益与方案1 叠加（认证效率提升）。

**技术创新点（≤4）**
1. 检测器内的高召回区 pAUC 代理（匹配候选 vs top-k 未匹配候选的排序损失），适配 TAL 分配（YOLO）与匈牙利匹配（DETR）两种正负定义。
2. "pre-event 帧 = 同相机免费难负样本"的课程化使用（数据侧创新，FIGlib 结构天然支持，fold-安全）。
3. 带覆盖闸门的剂量调度，把已测的"剂量↑覆盖↓"权衡（0.7377）变成受控变量。
4. 训练增益 → 认证效率增益的量化链路（与 RACO 联动）。

**协议**：同方案1 的数据/fold/指标/显著性；训练新增条件 "opal"（=式4+锚）与 "opal_curr"（+课程），300ep、val=false、last.pt、同导出协议；主终点预注册为 FIGlib LOSO FAR/hour@认证R0.90 与 S2 distractor-FPR（避免域内地板效应稀释）。

### 2.3 与现有 pipeline 兼容性

**最小训练改动**
- YOLO26n：自定义 loss 插件（TAL 分配后收集正/未匹配候选 cls 分，加式4 与 μ 项，~150 LOC）。
- RT-DETR-L：匈牙利匹配后 matched query logits = P、top-k unmatched = N，同损失项（一对一匹配使 P/N 定义最干净）。
- InternImage-FRCNN：仅作迁移验证（ROI head logits），默认不训 3 seed（省算力）。

**需新增组件**：distractor manifest 构建器（FIGlib pre-event 采样，按 fold 隔离防泄漏）；ρ(t) 采样器（复用 hardneg 基建）；覆盖监控（unique coverage ≥0.95 gate 写入 run_meta）；μ_t 日志。

**最小可行 3-seed 矩阵**：YOLO26n × {hardneg(已有), opal, opal_curr} × seeds{11,22,33} = 新增 6 个 300ep run（~60–120 h，5070Ti 可承担）；通过 MPD 后再决定 RT-DETR × opal_curr × 3 seed（~150 h，3060/排队）。评测全走冻结导出 + RACO 控制层。

**能否直接进论文：中。** 作为"训练轴 × 控制轴"双轴叙事中的训练轴增强为高价值；独立撑 1 区为中（pAUC 思想在分类/行人检测有先例，应用级创新）。

**预期增益（3）**：① 给审稿人"除后处理外的训练侧贡献"，堵"只是校准包装"的攻击；② 主攻 hardneg 尚未证明的跨域/压力场景（RT-DETR 域内残留 0.1056 也有压缩空间）；③ 同相机免费难负样本是审稿人记得住的工程亮点。
**风险（3）**：① D-Fire 域内地板效应（0.0038）→ 增益必须在 FIGlib/S2 上找，若 < MPD 则按预注册降级为消融（无害退出路径）；② 对偶上升不稳 → 备选固定 μ 调度；③ 新增 6–15 个 300ep run 触碰算力预算与"4 arms 锁定"表述——写作时定位为"训练轴扩展条件"，不动原 4-arm 结论。

---

## 一篇文章的统一主线与实验图表清单

**统一叙事（一句话）**：固定相机火/烟检测的部署瓶颈不是 mAP，而是"固定事件召回下的误报负担与告警及时性"；本文给出（a）分布无关、事件级、TTD 预算内的召回认证控制层 RACO（对任意冻结检测器、显著优于 CRC/LTT 朴素版），与（b）把训练梯度对齐到部署召回区的 OPAL，二者叠加给出"认证召回、最小告警"的完整配方，并在 D-Fire + FIGlib LOSO、三架构族、S1–S3 压力下系统验证。

**贡献排布**：贡献1 = RACO（headline，方案1）；贡献2 = OPAL 训练轴（方案2，YOLO 3-seed 起步）；贡献3 = 架构×失败模式表征（原主线 H0 保留，DAQ 失败作免校准动机）；hardneg 4-arm = 训练轴剂量阶梯（非论点）。

**表清单**
- **表1（headline）**FIGlib LOSO 事件级 @R0.90/Δ10min：行 = 控制方法{global-PR, learned-thr, global-CRC, group-CRC, LTT-τ, RACO} × 检测器{YOLO26n, RT-DETR-L, (InternImage 补充)}；列 = 实现事件召回[95% CI]、违约率、FAR/hour(proxy 口径)、per-frame FP、TTD 中位/p90。
- **表2（域内+训练轴）**D-Fire 固定召回：训练条件{baseline, eqstep, hardneg, hardneg_sched, opal, opal_curr} × 3 模型 × R{0.85,0.90,0.95} 的 FPR/FPPI（mean±std, 3 seed），附认证效率列（可行 τ 与认证 precision）。
- **表3（显著性）**每条主张（RACO<global-CRC、RACO<LTT-τ、opal_curr<hardneg、H1 违约率对比）：paired Δ + cluster bootstrap 95% CI + Holm 校正 p + Cliff's δ + MDE + 3-seed 同向标记。
- 附录表：泄漏审计（站点/事件族/近邻帧 + split md5）、组样本量与证书松紧（n_g vs (N+1)α 量化）、269.54h proxy 口径说明。

**图清单**
- **图1**方法总览：冻结检测器 → 统一 CSV → 可迁移分组 → CRC/LTT 联合认证 → 告警状态机；OPAL 在训练侧的插入点；S1–S3 注入位置。
- **图2（money figure）**名义 vs 实现事件召回：LOSO 各 fold 散点 + binomial 可接受带；PR-pick/learned-thr 落带外、RACO 贴带——H1 的直接可视化（域内实测预览已成立：Ncal=200 时 PR 违约 49% vs LTT 1.8%，见实测锚点 §2）。
- **图3**效率前沿：FAR/hour vs 认证召回（0.80–0.95）各方法曲线；inset：(k,n) 扫描的 TTD–FAR Pareto。
- **图4**压力鲁棒性：固定认证配置下 S1 severity / S2 注入率 ρ / S3 抖动对实现召回与 FAR 漂移的曲线（含 shift-aware 标定消融）；右侧定性面板：OPAL 前后干扰物误报案例与分数分布位移。

**最终最推荐组合**：**RACO（方案1）为唯一 headline 方法贡献 + OPAL 以 YOLO26n 3-seed 最小矩阵进入训练轴 + 时序确认 (k,n) 并入 RACO 配置空间**。理由：方案1 零训练成本、直接落在现有设计的空位上、撞车面可控；方案2 以"无害退出"方式增量加入。
**替代组合**：若 FIGlib 全量影像不可恢复（触发 §9 条款）→ RACO 退守 D-Fire box-level 组条件认证 + PyroNear/PyroSDIS 补跨域，OPAL 升为共同主角，目标降至 TPAMI-minor/TIP/PR；若 OPAL 未过 MPD → 折叠为消融，主线不受损。

**新颖性碰撞核查（检索于 2026-06-10）**
- Andéol & Mossina et al., *Conformal Object Detection by Sequential Risk Control*（arXiv 2505.24038）：最近邻，per-image/box 召回损失 + 序贯风险控制（铁路）。差异：事件级 TTD 预算损失、FAR/hour 效率目标、LOSO 可迁移分组、固定相机视频流。必须正面对表。
- *Conformal Risk Control for Safety-Critical Wildfire Evacuation Mapping*（arXiv 2603.22331, 2026-03）：CRC 用于野火蔓延/疏散制图（非相机检测、无 TTD/事件流），证明该思路在野火域的时效性，引用并划界。
- FIgLib & SmokeyNet（arXiv 2112.08598）：基准与 TTD 指标来源，作对照系统。
- pAUC 优化系（arXiv 2203.01505、2502.11570、1310.0900 行人检测）：OPAL 的方法学源头，差异在检测器内正负集定义（TAL/匈牙利）与同相机课程。

**参考链接**：arxiv.org/abs/2505.24038 · arxiv.org/abs/2603.22331 · arxiv.org/abs/2112.08598 · arxiv.org/abs/2203.01505 · arxiv.org/abs/2502.11570 · Angelopoulos et al., *Conformal Risk Control*（arXiv 2208.02814）· *Learn then Test*（arXiv 2110.01052）
