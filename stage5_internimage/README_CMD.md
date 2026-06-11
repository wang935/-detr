# Stage5 InternImage Folder (Deprecated)

This folder is archival. The D-Fire third-model allocation has been rolled back to **YOLO26n**.

## Active allocation now used

- H01 (my host): YOLO26n seed 11
- H02 (Ye Hailong): YOLO26n seed 22
- H03 (Sun Hao): YOLO26n seed 33

Run these files from `stage5_5hosts`:

- `stage5_5hosts\\H01_3060_yolo_seed11.bat`
- `stage5_5hosts\\H02_3060_yolo_seed22.bat`
- `stage5_5hosts\\H03_3060_yolo_seed33.bat`

If you still need InternImage for ad-hoc validation, use:

- `00_create_or_check_env.bat`
- `00_repair_mmcv_full.bat` / `00_repair_mmcv_full_310.bat`
- `01_prepare_dfire_coco.bat`

Those files are kept for reference and recovery, and are **not** part of the active
Stage 5 formal run chain.
