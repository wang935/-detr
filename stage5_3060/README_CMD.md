# Stage 5 正式实验 3060 启动说明

本阶段目标是生成正式论文主表级别的重复实验，而不是快速 pilot。

核心矩阵：

1. 两个模型族：YOLO26n 与 RT-DETR-L。
2. 每个模型族 3 个 seeds：11、22、33。
3. 每个 seed 跑 3 个 arm：`baseline`、`baseline_eqstep`、`hardneg`。
4. 每个 arm 训练 300 epochs，并显式 `--val false`。
5. RT-DETR 的 DAQ strong 与 DAQ ablation 只作为可选诊断，在正式主实验完成后单独运行。

所有机器的项目路径必须统一为 `D:\detr_Q3`，否则合并后的绝对路径追溯会变困难。

## 每台机器先检查环境

第一次在某台机器上运行，先执行：

```bat
cd /d D:\detr_Q3
stage5_3060\00_create_or_check_env.bat
```

这个脚本会检查或创建本机 conda 环境 `daq`，并安装/检查 `torch`、`ultralytics`、`numpy`。conda 环境本体不在 `D:\detr_Q3` 目录里，而在每台电脑自己的 Anaconda/Miniconda 环境目录下。

## 每台机器正式任务前运行

```bat
cd /d D:\detr_Q3
stage5_3060\00_check_stage5_assets.bat
```

预检会自动生成 equal-step 数据配置：

```text
data\dfire_local\train_posonly_equalstep.txt
data\dfire_local\dfire_posonly_equalstep.yaml
```

`baseline_eqstep` 只重复正样本，使每个 epoch 的样本数与 `hardneg` 相同；它用于控制 optimizer-step 数量，不加入无火/无烟负样本。

预检还会自动生成并校验固定召回评估用的 calibration/test label 文件：

```text
data\dfire_local\eval_calib_labels.csv
data\dfire_local\eval_test_labels.csv
```

正式评估阈值只在 `eval_calib_labels.csv` 的正样本上选择；test recall、FPR 和 FPPI 只在 `eval_test_labels.csv` 上报告。不要在正式实验中添加 `--legacy-in-sample-eval`。

## 三台 3060 第一轮

### 1 号机：YOLO26n seed 11

```bat
cd /d D:\detr_Q3
stage5_3060\PC1_yolo_seed11.bat
```

### 2 号机：YOLO26n seed 22

```bat
cd /d D:\detr_Q3
stage5_3060\PC2_yolo_seed22.bat
```

### 3 号机：RT-DETR-L seed 11

```bat
cd /d D:\detr_Q3
stage5_3060\PC3_rtdetr_seed11.bat
```

## 第二轮补齐

### 1 号机：YOLO26n seed 33

```bat
cd /d D:\detr_Q3
stage5_3060\PC1b_yolo_seed33.bat
```

### 2 号机：RT-DETR-L seed 22

```bat
cd /d D:\detr_Q3
stage5_3060\PC2b_rtdetr_seed22.bat
```

### 3 号机：RT-DETR-L seed 33

```bat
cd /d D:\detr_Q3
stage5_3060\PC3b_rtdetr_seed33.bat
```

RT-DETR 正式 `.bat` 只运行 Stage 5 主实验；训练、导出、评估完成后，如需做 query/DAQ 诊断，再手动运行：

```text
stage5_3060\DAQ_rtdetr_seed11.bat
stage5_3060\DAQ_rtdetr_seed22.bat
stage5_3060\DAQ_rtdetr_seed33.bat
```

以上 DAQ 脚本只作为 RT-DETR within-dataset diagnostic；正式 `PC*_rtdetr*.bat` 不自动调用 DAQ，DAQ 失败不能污染主实验成功状态。

## 第二外部负样本源：BoWFire

为了补强“外部负样本误报压力测试”，Stage 5 strict 汇总通过后可额外评估 BoWFire non-fire / fire-like negative 图像。

先将 BoWFire 解压到：

```text
external_data\BoWFireDataset\dataset\img\
```

然后运行：

```bat
cd /d D:\detr_Q3
stage5_3060\100_external_bowfire.bat
```

该评估不会重新选择阈值；它复用每个正式 Stage 5 arm 在 D-Fire calibration split 上得到的固定召回阈值，并只在 BoWFire `not_fire*.png` 负样本上报告 external FPR/FPPI。BoWFire 结果只能作为外部 negative-only stress test，不能写成外部 recall、mAP 或部署真实误报率。

## 汇总结果

三台机器跑完后，把各自的 `formal_results\stage5\*` 和 `runs\detect\runs_stage5_formal\*` 合并回主机同一路径，然后运行：

```bat
cd /d D:\detr_Q3
stage5_3060\99_aggregate_stage5.bat
```

汇总脚本使用严格门槛：

```text
--require-eqstep --min-seeds 3 --required-seeds 11,22,33 --strict
```

如果任何模型族少于 3 个完整 seeds，或任何 seed 缺少 `baseline`、`baseline_eqstep`、`hardneg` 任一 arm，汇总会返回非零状态。不要用不完整 run 写均值 ± 标准差。

汇总脚本默认只纳入 `eval_protocol=calibration_threshold_test_report` 的结果；旧 in-sample 协议或缺少协议标记的 `gonogo.json` 会被排除。
汇总脚本还会检查 split fingerprint；三台机器的 full/calibration/test label md5、`calib_frac` 和 `eval_split_salt` 必须一致，否则 strict 汇总失败。

输出：

```text
formal_results\stage5\STAGE5_REPEAT_SUMMARY.md
formal_results\stage5\stage5_repeat_metrics.csv
formal_results\stage5\stage5_repeat_aggregate.csv
formal_results\stage5\stage5_paired_deltas.csv
formal_results\stage5\stage5_run_status.csv
```

## 重要规则

- 不要覆盖已有正式实验目录。脚本默认检测到已有训练目录或预测文件会退出。
- 如确认要重跑，手动给对应命令添加 `--overwrite`，并记录原因。
- 所有实验必须使用 `data\dfire_local\eval_labels.csv` 作为预测导出的完整 eval pool，并使用自动生成的 `eval_calib_labels.csv` / `eval_test_labels.csv` 做固定召回评估。
- 项目根目录必须已有 `yolo26n.pt` 和 `rtdetr-l.pt`；预检会在训练前检查权重文件。
- 所有正式训练都使用 `--val false`，不启用训练期验证，避免 D-Fire eval split 参与 checkpoint 选择。
- 正式训练使用 `--val false` 时，runner 强制使用 `last.pt` 导出预测，并在 `run_meta.json` / `gonogo.json` 记录实际导出权重路径。
- `--workers 4` 可按 CPU 情况调低，例如低配 CPU 改成 `--workers 2`。
- strict 汇总会读取 `run_meta.json` 和 `*.pred_meta.json`，要求 `mode=all`、`epochs=300`、`val=false`、seed 精确为 `11,22,33`、导出权重为 `last.pt`、训练列表计数匹配，并要求正式导出 `--conf` 不高于 `0.001`。
- 正式导出默认使用 `--conf 0.001`；如果手动传入更高 `--conf`，runner 会拒绝运行，避免高召回 operating point 被导出阶段截断。
- 默认启用 AMP；如果某台机器 CUDA/驱动异常，可以在命令中添加 `--amp false`。
- 当前重复实验仍是报警级 FPR/FPPI，不是框级定位贡献，也不是检测器 SOTA 训练。D-Fire calibration/test 结果不能写成部署真实误报率或外部泛化证明。
