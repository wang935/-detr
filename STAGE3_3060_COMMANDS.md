# Stage 3 三台 RTX 3060 启动命令

## 时间判断

Stage 3A 和 Stage 3B 不长，主要是读已有 CSV 做后处理，通常几分钟内完成。

真正长的是 Stage 3D：YOLO 重新训练、导出预测、评估。Stage 3C 如果外部数据已下载，主要是 RT-DETR 推理外部 neutral 图，通常比重新训练短。

## 每台机器先跑

```bat
cd /d D:\detr_Q3
stage3_3060\00_create_or_check_env.bat
stage3_3060\01_check_assets.bat
```

## 3060 一号机：Stage 3A 多划分稳定性

```bat
cd /d D:\detr_Q3
stage3_3060\PC1_stage3A_multisplit.bat
```

结果：

```text
stage3_daq\multisplit\STAGE3_MULTISPLIT_SUMMARY.md
stage3_daq\multisplit\stage3_multisplit_summary.csv
stage3_daq\multisplit\stage3_multisplit_summary.json
```

## 3060 二号机：Stage 3B query-budget 消融

```bat
cd /d D:\detr_Q3
stage3_3060\PC2_stage3B_ablation.bat
```

结果：

```text
stage3_daq\ablation\STAGE3_BUDGET_ABLATION.md
stage3_daq\ablation\stage3_budget_ablation.csv
stage3_daq\ablation\stage3_budget_ablation.json
```

## Stage 3C：外部 neutral 干扰物

先确保外部数据解压到：

```text
external_data\FIRE-SMOKE-DATASET
```

然后运行：

```bat
cd /d D:\detr_Q3
stage3_3060\PC2b_stage3C_external_deepquest.bat
```

结果：

```text
stage3_daq\external_deepquest\STAGE3_EXTERNAL_DISTRACTOR.md
stage3_daq\external_deepquest\stage3_external_summary.json
stage3_daq\external_deepquest\external_hardneg_raw.csv
stage3_daq\external_deepquest\external_daq.csv
```

## 3060 三号机：Stage 3D YOLO baseline

先做一个不训练的 smoke test：

```bat
cd /d D:\detr_Q3
stage3_3060\PC3_stage3D_yolo_smoke_test.bat
```

确认 OK 后启动训练、导出、评估：

```bat
cd /d D:\detr_Q3
stage3_3060\PC3_stage3D_yolo_baseline.bat
```

结果：

```text
runs\detect\runs_stage3_yolo\baseline\weights\best.pt
runs\detect\runs_stage3_yolo\hardneg\weights\best.pt
preds_stage3_yolo\baseline.csv
preds_stage3_yolo\hardneg.csv
preds_stage3_yolo\yolo_gonogo.json
```

如果显存不够，编辑：

```text
stage3_3060\PC3_stage3D_yolo_baseline.bat
```

把：

```bat
--batch 16
```

改成：

```bat
--batch 8
```

或：

```bat
--batch 4
```

## 已经放进 D:\detr_Q3 的内容

- `data\D-Fire`：D-Fire 本地镜像，约 33.57 GB。
- `data\dfire_local`：本地路径版 YAML/TXT/labels。
- `scripts\stage3_multisplit.py`：Stage 3A。
- `scripts\stage3_budget_ablation.py`：Stage 3B。
- `scripts\stage3_external_distractor.py`：Stage 3C。
- `scripts\stage3_yolo_baseline.py`：Stage 3D。
- `stage3_3060\*.bat`：三台 3060 的启动脚本。
