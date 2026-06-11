# Stage 5 5-Host Command Sheet

Updated hardware: 3 x RTX 3060 plus 2 x RTX 3080 Ti.

Host mapping:

| Host name | GPU | Task id |
|---|---|---|
| 我的主机 | RTX 3060 | H01 |
| 叶海龙的主机 | RTX 3060 | H02 |
| 孙浩的主机 | RTX 3060 | H03 |
| 陈慧的主机 | RTX 3080 Ti | H04 |
| 王建斌的主机 | RTX 3080 Ti | H05 |

Keep the project path exactly:

```bat
D:\detr_Q3
```

The scientific matrix is unchanged: YOLO26n and RT-DETR-L, seeds `11,22,33`,
three arms per seed (`baseline`, `baseline_eqstep`, `hardneg`), `300` epochs,
`--val false`, strict aggregation after all six complete runs.

## First Run On Every Host

Run these three commands on each of the five hosts before formal training:

```bat
cd /d D:\detr_Q3
stage5_5hosts\00_create_or_check_env.bat
stage5_5hosts\00_check_stage5_assets.bat
```

`00_create_or_check_env.bat` creates or checks conda env `daq`. If no
Anaconda/Miniconda exists, it installs Miniconda for the current Windows user,
then installs CUDA PyTorch cu121, Ultralytics, NumPy, Matplotlib, and PyYAML.
The final check runs a real CUDA tensor operation and exits non-zero if CUDA is
not usable.

## Minimum Experiment Test Per Host

These are pre-flight minchecks only. They write to:

```text
formal_results\stage5_mincheck\
runs\detect\runs_stage5_mincheck\
```

They use `epochs=1`, `--limit 256`, `--overwrite`, and do not enter the paper
results.

| Host | GPU | Mincheck command |
|---|---|---|
| 我的主机 | RTX 3060 | `stage5_5hosts\TEST_H01_3060_yolo_seed11_min.bat` |
| 叶海龙的主机 | RTX 3060 | `stage5_5hosts\TEST_H02_3060_yolo_seed22_min.bat` |
| 孙浩的主机 | RTX 3060 | `stage5_5hosts\TEST_H03_3060_yolo_seed33_min.bat` |
| 陈慧的主机 | RTX 3080 Ti | `stage5_5hosts\TEST_H04_3080ti_rtdetr_seed11_min.bat` |
| 王建斌的主机 | RTX 3080 Ti | `stage5_5hosts\TEST_H05_3080ti_rtdetr_seed22_min.bat` |

Optional before the second RT-DETR round:

```bat
cd /d D:\detr_Q3
stage5_5hosts\TEST_3080ti_rtdetr_seed33_min.bat
```

After copying all mincheck outputs back to one host, run:

```bat
cd /d D:\detr_Q3
stage5_5hosts\TEST_aggregate_min.bat
```

`TEST_aggregate_min.bat` is a structure check. A warning about unreachable
fixed-recall points is expected for 1 epoch plus `--limit 256`.

## Formal Training Round 1

Run these five tasks in parallel:

| Host | GPU | Formal command | Model | Seed |
|---|---|---|---|---:|
| 我的主机 | RTX 3060 | `stage5_5hosts\H01_3060_yolo_seed11.bat` | YOLO26n | 11 |
| 叶海龙的主机 | RTX 3060 | `stage5_5hosts\H02_3060_yolo_seed22.bat` | YOLO26n | 22 |
| 孙浩的主机 | RTX 3060 | `stage5_5hosts\H03_3060_yolo_seed33.bat` | YOLO26n | 33 |
| 陈慧的主机 | RTX 3080 Ti | `stage5_5hosts\H04_3080ti_rtdetr_seed11.bat` | RT-DETR-L | 11 |
| 王建斌的主机 | RTX 3080 Ti | `stage5_5hosts\H05_3080ti_rtdetr_seed22.bat` | RT-DETR-L | 22 |

## Formal Training Round 2

After either RTX 3080 Ti host becomes free, run the last RT-DETR seed on 陈慧的主机 or 王建斌的主机:

```bat
cd /d D:\detr_Q3
stage5_5hosts\H04b_3080ti_rtdetr_seed33.bat
```

The filename says H04b, but 王建斌的主机 can run it too if it finishes first. The
output directory is keyed by `rtdetr_seed33`, not by host name.

## Optional RT-DETR Diagnostics

Run these only after the matching RT-DETR formal seed has finished:

```bat
cd /d D:\detr_Q3
stage5_5hosts\DAQ_rtdetr_seed11.bat
stage5_5hosts\DAQ_rtdetr_seed22.bat
stage5_5hosts\DAQ_rtdetr_seed33.bat
```

DAQ outputs are within-dataset diagnostics and are not required for strict
Stage 5 success.

## Aggregation

After merging every host's:

```text
formal_results\stage5\*
runs\detect\runs_stage5_formal\*
```

back into the same path on one host, run:

```bat
cd /d D:\detr_Q3
stage5_5hosts\99_aggregate_stage5.bat
```

The strict gate remains:

```text
--require-eqstep --min-seeds 3 --required-seeds 11,22,33 --strict
```

## External Negative-Only Stress Test

After strict Stage 5 aggregation succeeds:

```bat
cd /d D:\detr_Q3
stage5_5hosts\100_external_bowfire.bat
```

## If A Host Fails

- If CUDA is unavailable, update the NVIDIA driver first, then rerun
  `00_create_or_check_env.bat`.
- If YOLO batch 16 OOMs on a small-memory 3060, reduce only that host script to
  `--batch 8` and record the reason. Do not mix RT-DETR batch sizes across
  RT-DETR seeds unless a rerun note explains it.
- Formal scripts intentionally do not pass `--overwrite`; an existing formal
  output directory stops the run instead of silently replacing results.
- After an interrupted formal run, remove only that run's matching
  `runs\detect\runs_stage5_formal\<family>_seed<seed>` and
  `formal_results\stage5\<family>_seed<seed>` directories before relaunching,
  and record the rerun reason.
