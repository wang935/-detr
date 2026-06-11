# Stage 5 5-Host Quick Commands

Use this updated plan for 3 x RTX 3060 plus 2 x RTX 3080 Ti.

Every host first:

```bat
cd /d D:\detr_Q3
stage5_5hosts\00_create_or_check_env.bat
stage5_5hosts\00_check_stage5_assets.bat
```

Minimum test per host:

```bat
:: 我的主机 (RTX 3060, H01)
cd /d D:\detr_Q3
stage5_5hosts\TEST_H01_3060_yolo_seed11_min.bat

:: 叶海龙的主机 (RTX 3060, H02)
cd /d D:\detr_Q3
stage5_5hosts\TEST_H02_3060_yolo_seed22_min.bat

:: 孙浩的主机 (RTX 3060, H03)
cd /d D:\detr_Q3
stage5_5hosts\TEST_H03_3060_yolo_seed33_min.bat

:: 陈慧的主机 (RTX 3080 Ti, H04)
cd /d D:\detr_Q3
stage5_5hosts\TEST_H04_3080ti_rtdetr_seed11_min.bat

:: 王建斌的主机 (RTX 3080 Ti, H05)
cd /d D:\detr_Q3
stage5_5hosts\TEST_H05_3080ti_rtdetr_seed22_min.bat
```

Formal training round 1:

```bat
:: 我的主机 (RTX 3060, H01)
cd /d D:\detr_Q3
stage5_5hosts\H01_3060_yolo_seed11.bat

:: 叶海龙的主机 (RTX 3060, H02)
cd /d D:\detr_Q3
stage5_5hosts\H02_3060_yolo_seed22.bat

:: 孙浩的主机 (RTX 3060, H03)
cd /d D:\detr_Q3
stage5_5hosts\H03_3060_yolo_seed33.bat

:: 陈慧的主机 (RTX 3080 Ti, H04)
cd /d D:\detr_Q3
stage5_5hosts\H04_3080ti_rtdetr_seed11.bat

:: 王建斌的主机 (RTX 3080 Ti, H05)
cd /d D:\detr_Q3
stage5_5hosts\H05_3080ti_rtdetr_seed22.bat
```

Formal training round 2, on whichever RTX 3080 Ti becomes free first (陈慧的主机 or 王建斌的主机):

```bat
cd /d D:\detr_Q3
stage5_5hosts\H04b_3080ti_rtdetr_seed33.bat
```

After merging all hosts back to one copy of `D:\detr_Q3`:

```bat
cd /d D:\detr_Q3
stage5_5hosts\99_aggregate_stage5.bat
```
