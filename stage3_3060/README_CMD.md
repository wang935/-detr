# Stage 3 RTX 3060 Command Sheet

## Is Stage 3 Long?

Stage 3A and Stage 3B are not long. They use existing prediction CSV files and mostly run on CPU:

- Stage 3A multi-split: about several minutes for 6 splits.
- Stage 3B query-budget ablation: about 1 minute.

The long part is Stage 3D:

- YOLO/RT-DETR extra training.
- Re-exporting predictions on thousands of eval images.
- External distractor evaluation after adding DFS/web negatives.

## What Is Inside `D:\detr_Q3`

This folder now contains the needed local assets for RTX 3060 machines:

- Code: `scripts\stage3_multisplit.py`, `scripts\stage3_budget_ablation.py`, `scripts\stage3_yolo_baseline.py`
- Existing RT-DETR predictions: `preds\baseline_clean5.csv`, `preds\hardneg_clean5.csv`
- Existing RT-DETR weights: `runs\detect\runs_tierb\*\weights\best.pt`
- Local D-Fire mirror: `data\D-Fire`
- Local D-Fire configs: `data\dfire_local`
- 3060 launch scripts: `stage3_3060\*.bat`

If you copy this whole `D:\detr_Q3` folder to another 3060 machine at the same path, the Stage 3 scripts should have the code, predictions, weights, and dataset they need.

## First Command On Every 3060

Open `cmd`, then run:

```bat
cd /d D:\detr_Q3
stage3_3060\00_create_or_check_env.bat
stage3_3060\01_check_assets.bat
```

The first script installs/checks the `daq` environment. The second script performs a short smoke test.

## Three-Machine Split

### 3060 Machine 1: Stage 3A Multi-Split Stability

```bat
cd /d D:\detr_Q3
stage3_3060\PC1_stage3A_multisplit.bat
```

Output:

```text
stage3_daq\multisplit\STAGE3_MULTISPLIT_SUMMARY.md
stage3_daq\multisplit\stage3_multisplit_summary.csv
stage3_daq\multisplit\stage3_multisplit_summary.json
```

### 3060 Machine 2: Stage 3B Query-Budget Ablation

```bat
cd /d D:\detr_Q3
stage3_3060\PC2_stage3B_ablation.bat
```

Output:

```text
stage3_daq\ablation\STAGE3_BUDGET_ABLATION.md
stage3_daq\ablation\stage3_budget_ablation.csv
stage3_daq\ablation\stage3_budget_ablation.json
```

### 3060 Machine 3: Stage 3D YOLO Baseline

```bat
cd /d D:\detr_Q3
stage3_3060\PC3_stage3D_yolo_baseline.bat
```

Output:

```text
runs\detect\runs_stage3_yolo\baseline\weights\best.pt
runs\detect\runs_stage3_yolo\hardneg\weights\best.pt
preds_stage3_yolo\baseline.csv
preds_stage3_yolo\hardneg.csv
preds_stage3_yolo\yolo_gonogo.json
```

This is the longest task. If batch 16 gives out-of-memory on a 3060, edit the `.bat` and change `--batch 16` to `--batch 8` or `--batch 4`.

## If You Only Have One 3060

Run in this order:

```bat
cd /d D:\detr_Q3
stage3_3060\PC1_stage3A_multisplit.bat
stage3_3060\PC2_stage3B_ablation.bat
stage3_3060\PC3_stage3D_yolo_baseline.bat
```

## How To Read Results

Continue the DAQ paper only if:

- Stage 3A: mean R=0.90 FPR improves by at least 0.015.
- Stage 3A: FPPI does not increase.
- Stage 3B: `gate_topk1` beats `topk1_only`.
- Stage 3D: DAQ/RT-DETR still has a meaningful false-alarm advantage over simple YOLO threshold tuning.

