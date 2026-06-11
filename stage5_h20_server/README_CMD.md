# H20 Server Command Sheet

Target hardware: one Linux server with 4 x NVIDIA H20-3e, 32 CPU cores, and
128 GB RAM. This replaces the old split plan that used 3 x RTX 3060 plus
2 x RTX 3080 Ti.

Keep the remote project root consistent, for example:

```bash
/data/detr_Q3
```

## 1. Sync From This Windows Workspace

After you have the SSH target, run this locally from PowerShell:

```powershell
Set-Location D:\detr_Q3
powershell -ExecutionPolicy Bypass -File stage5_h20_server\03_sync_to_h20.ps1 -Server user@host -RemoteRoot /data/detr_Q3
```

If the server requires a port:

```powershell
Set-Location D:\detr_Q3
powershell -ExecutionPolicy Bypass -File stage5_h20_server\03_sync_to_h20.ps1 -Server user@host -Port 2222 -RemoteRoot /data/detr_Q3
```

The helper uses `scp -r`, so large `data\D-Fire` transfer can take time. If the
dataset is already on the server, copy only the code and keep the same remote
directory shape under `/data/detr_Q3`.

## 2. Build The Environment On The Server

```bash
ssh user@host
cd /data/detr_Q3
bash stage5_h20_server/00_bootstrap_h20_env.sh
```

Expected final line:

```text
[ok] H20 Stage 5 environment is ready: daq310
```

The bootstrap creates conda env `daq310`, installs CUDA PyTorch from cu124
first, falls back to cu121 if needed, installs Ultralytics without replacing
the CUDA torch wheel, and runs a real CUDA tensor operation on every visible
GPU.

## 3. Preflight Assets And Path Rewrite

```bash
cd /data/detr_Q3
bash stage5_h20_server/01_check_h20_assets.sh
```

Expected final line:

```text
[ok] H20 assets and GPU smoke checks passed.
```

This rewrites copied `D:\detr_Q3\...` data paths into the server root before
smoke checks. It is safe to rerun on the server.

## 4. Stage 5 v1 Mincheck

```bash
cd /data/detr_Q3
bash stage5_h20_server/run_stage5_v1_mincheck_4gpu.sh
```

This runs all six one-epoch checks across the H20 server and writes only to:

```text
formal_results/stage5_mincheck
runs/detect/runs_stage5_mincheck
```

## 5. Stage 5 v1 Formal Queue

```bash
cd /data/detr_Q3
bash stage5_h20_server/run_stage5_v1_all_queue.sh
```

Queue layout:

| GPU | Worker |
|---:|---|
| 0 | RT-DETR-L seed 11, all three arms, 300 epochs, batch 54 |
| 1 | RT-DETR-L seed 22, all three arms, 300 epochs, batch 54 |
| 2 | RT-DETR-L seed 33, all three arms, 300 epochs, batch 54 |
| 3 | YOLO26n seed 11 -> 22 -> 33, all three arms, 300 epochs, batch 16 |

The formal output roots stay unchanged:

```text
runs/detect/runs_stage5_formal
formal_results/stage5
```

The script uses `screen` when available and falls back to `nohup`. Logs are in:

```text
logs/h20
```

## 6. Aggregation And External Stress Test

After every formal worker finishes:

```bash
cd /data/detr_Q3
bash stage5_h20_server/run_stage5_v1_aggregate.sh
bash stage5_h20_server/run_stage5_v1_external_bowfire.sh
```

## 7. Stage5-PV v2

D-Fire mincheck:

```bash
cd /data/detr_Q3
bash stage5_h20_server/run_stage5_pv_v2_dfire_mincheck_4gpu.sh
```

Full formal queue:

```bash
cd /data/detr_Q3
bash stage5_h20_server/run_stage5_pv_v2_full_queue.sh
```

This runs `dfire,dfs x yolo26n,rtdetr x seeds 11,22,33 x all4 arms x
300 epochs`. It uses `YOLO_BATCH=16` and `RTDETR_BATCH=54` by default on H20;
override those environment variables only if memory or runtime measurements
force a change. `yolo26s` is intentionally excluded from the formal matrix.
The H20 launchers also default `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and
`NUMEXPR_NUM_THREADS=4` so four RT-DETR jobs can initialize without CPU thread
oversubscription; override them only when profiling shows a better setting.
Before launching parallel training waves, the queue runs CPU smoke checks once
per dataset to generate shared equal-step and scheduled-hard-negative lists
serially.

Strict aggregation after the full PV v2 matrix exists:

```bash
cd /data/detr_Q3
bash stage5_h20_server/run_stage5_pv_v2_aggregate.sh
```

DFS still needs a VOC-XML detection dataset under:

```text
external_data/DFS-FIRE-SMOKE-Dataset
```

Then run:

```bash
python scripts/stage5_pv_v2_prepare_dfs.py --source external_data/DFS-FIRE-SMOKE-Dataset
```

The full queue will generate `dfs_posonly_equalstep.yaml` from the prepared DFS
files before parallel training starts.

## Monitor

```bash
screen -ls
tail -f logs/h20/*.log
nvidia-smi
```

Formal scripts still refuse to overwrite existing formal output directories.
After an interrupted run, remove only the matching
`runs/detect/runs_stage5_formal/<family>_seed<seed>` and
`formal_results/stage5/<family>_seed<seed>` directories before relaunching.
For Stage5-PV v2, remove only the matching
`runs/detect/runs_stage5_pv_v2/<dataset>_<family>_seed<seed>` and
`formal_results/stage5_pv_v2/<dataset>_<family>_seed<seed>` directories.
