# Stage5-PV v2 Command Sheet

This folder is isolated from the running Stage 5 v1 jobs. It writes to:

```text
formal_results\stage5_pv_v2
runs\detect\runs_stage5_pv_v2
```

Do not stop or edit the current v1 YOLO jobs.

## Preflight

```bat
cd /d D:\detr_Q3
stage5_pv_v2\00_check_stage5_pv_v2_assets.bat
```

This compiles the v2 scripts and smoke-checks D-Fire metadata. It may create:

```text
data\stage5_pv_v2\dfire\dfire_posonly_equalstep.yaml
data\stage5_pv_v2\dfire\dfire_hardneg_sched.yaml
data\stage5_pv_v2\dfire\train_posonly_equalstep.txt
data\stage5_pv_v2\dfire\train_hardneg_sched.txt
data\stage5_pv_v2\dfire\hardneg_sched_audit.json
```

## Prepare DFS

Download or clone `DFS-FIRE-SMOKE-Dataset` to:

```text
external_data\DFS-FIRE-SMOKE-Dataset
```

Then run:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\01_prepare_dfs.bat
```

The script refuses classification-only folders and requires VOC XML annotations.

## One-Epoch Minchecks

D-Fire:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\TEST_dfire_mincheck.bat
```

DFS after preparation:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\TEST_dfs_mincheck.bat
```

Mincheck outputs are not paper results.

## Formal Examples

D-Fire, YOLO26n, seed 11, four arms:

```bat
cd /d D:\detr_Q3
call stage5_pv_v2\_activate_env.bat
python scripts\stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 11 --mode all --arm all4 --epochs 300 --batch 16 --workers 4 --device 0 --val false
```

D-Fire, YOLO26n, hardneg_sched only, seed 22:

```bat
cd /d D:\detr_Q3
call stage5_pv_v2\_activate_env.bat
python scripts\stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 22 --mode all --arm hardneg_sched --epochs 300 --batch 16 --workers 4 --device 0 --val false --run-root runs\detect\runs_stage5_pv_v2_singlearm --out-root formal_results\stage5_pv_v2_singlearm
```

The single-arm command writes to scratch roots so it cannot collide with the
full formal matrix for the same dataset/family/seed.

3-host single-arm continuation for the added `hardneg_sched` arm:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\H01_yolo26n_seed11_hardneg_sched.bat
stage5_pv_v2\H02_yolo26n_seed22_hardneg_sched.bat
stage5_pv_v2\H03_yolo26n_seed33_hardneg_sched.bat
```

After merging the single-arm outputs back to one host, compare them against
the completed Stage 5 v1 YOLO arms:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\101_compare_hardneg_sched_vs_v1.bat
```

This writes:

```text
formal_results\stage5_pv_v2\HARDNEG_SCHED_SINGLEARM_SUMMARY.md
formal_results\stage5_pv_v2\stage5_pv_v2_hardneg_sched_vs_v1.csv
formal_results\stage5_pv_v2\stage5_pv_v2_hardneg_sched_run_check.csv
```

DFS, YOLO26n, seed 11, four arms:

```bat
cd /d D:\detr_Q3
call stage5_pv_v2\_activate_env.bat
python scripts\stage5_pv_v2_runner.py --dataset dfs --family yolo26n --seed 11 --mode all --arm all4 --epochs 300 --batch 16 --workers 4 --device 0 --val false
```

RT-DETR-L uses `--family rtdetr`. On H20, use batch 54 unless memory
measurements force a smaller value.

## Aggregation

Preview:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\99_aggregate_pv_v2_preview.bat
```

Final strict aggregation after the full matrix is merged:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\99_aggregate_pv_v2_strict.bat
```

## Conventional Metrics

Run only after `last.pt` exists:

```bat
cd /d D:\detr_Q3
stage5_pv_v2\100_map_metrics_pv_v2.bat
```
