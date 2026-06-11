# 3060 环境搭建 + Tier A 运行手册

**目标**: 在 RTX 3060 主机上打通 CUDA torch 环境，跑完 Tier A query 探针（160 张 D-Fire 真样本）。
**核心坑**: 3060 主机默认 `python` 是 MSYS2 版——无 pip、跑不了 GPU torch。**全程不要用它**，用下面的 conda 环境。

---

## Step 0 — 先确认是不是中了 MSYS2 陷阱

打开 **Anaconda Prompt**（不是 MSYS2/Git-Bash 终端），运行：

```bat
where python
```

若输出里出现 `C:\msys64\...\python.exe` 或没有 conda 路径 → 就是这个问题，照下面做。

---

## Step 1 — 装 Miniconda（已装过可跳过）

下载并安装 Miniconda（Windows x64）: https://docs.conda.io/en/latest/miniconda.html
装完用 **Anaconda Prompt** 操作，别用系统 cmd / MSYS2。

---

## Step 2 — 建并激活干净环境（Python 3.10）

```bat
conda create -n daq python=3.10 -y
conda activate daq
python -c "import sys; print(sys.executable)"
```

最后一行**必须**打印 `...\miniconda3\envs\daq\python.exe`。
如果还指向 msys64 → 说明没激活成功，检查是不是在 MSYS2 终端里。

---

## Step 3 — 装 CUDA 版 torch (cu121) + 依赖　⚠ 国内必须换镜像

3060 是 Ampere，cu121 完全支持、最稳。**官方源 `download.pytorch.org` 在国内直连会断流（2.4GB 拖不完），必须走国内镜像。**

已激活 daq 环境、在 `D:\detr_Q3` 下，依次跑：

```bat
:: torch 走上海交大镜像（官方源的镜像，解析逻辑一样，服务器在国内）
pip install torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --timeout 120 --retries 10
:: 其余依赖走默认的清华源即可（小包，快）
pip install ultralytics matplotlib numpy
```

> 顺序很重要：torch 先装好，再装 ultralytics，pip 才不会去清华源拉 CPU 版 torch 覆盖。
> **不要用** `python tier_a_3060.py --install`——它写死了官方源，正是会断流那条。

**若 SJTU 也慢，换阿里云镜像（需带版本号）**：

```bat
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

> cu124 / cu126 也支持 3060 但要求驱动更新；cu121 驱动门槛最低（Win 驱动 ≥ 527），先用它。装完若报驱动太旧，再升级 NVIDIA 驱动或换 cu124。

---

## Step 4 — 验证 GPU 可用（关键，必须 True）

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

期望类似: `2.x.x True NVIDIA GeForce RTX 3060`。
若 `False` → torch 装成了 CPU 版或驱动问题，别往下跑，先解决。

---

## Step 5 — 跑 Tier A 探针（160 张样本已就位）

样本 `pilot_images_dfire`（40 火 + 40 烟 + 80 None）和权重 `rtdetr-l.pt` 都已在 `D:\detr_Q3`，**不用再抽样**。直接：

```bat
cd /d D:\detr_Q3
python tier_a_3060.py --images pilot_images_dfire --weights rtdetr-l.pt --out pilot_outputs\query_probe_dfire
```

> ⚠ **不要**加 `--dfire-root`——那会重新抽样并覆盖已选好的 160 张。

输出在 `pilot_outputs\query_probe_dfire\`：
`query_features.csv` / `query_separability.json` / `query_vectors.npz` / `query_hist.png`

---

## Step 6 — 把结果发回来判读

回传这两个文件即可：

- `pilot_outputs\query_probe_dfire\query_separability.json`
- `pilot_outputs\query_probe_dfire\query_hist.png`

---

## 判读预期（先说在前头，避免误读）

`rtdetr-l.pt` 是 COCO 权重、**没有 fire/smoke 类**。这一步衡量的是"像不像 COCO 物体"，不是"像不像火"。
**预期看到火/烟 与 干扰物的 query 分布重叠、各 gap 都很小** → 这正是动机性结论：现成 DETR 分不开，**必须微调 + DAQ**。
它**不能**证明 DAQ 成立——那是 Tier B（D-Fire 微调）的事。所以无论分布重不重叠，Tier A 都不改变"要不要做 DAQ"，只为论文 motivation 段落提供素材。

---

## 附注：为什么 Tier A 放 3060 而不是 5070 Ti

5070 Ti 是 Blackwell 架构（sm_120），需要更新的 cu128 / torch≥2.7，**cu121 在它上面跑不了**。
3060 走成熟的 cu121 栈最省事，所以本阶段用 3060 是对的。后面 Tier B 微调若要用 5070 Ti，再单独配 cu128 环境。
