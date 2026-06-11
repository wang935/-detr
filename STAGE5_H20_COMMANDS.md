# Stage 5 H20 Quick Commands

New target: one remote Linux server, 4 x NVIDIA H20-3e.

Gemini/VirtAI platform image:

```text
hpc.chzu.edu.cn:32402/tcvmpvo5lnx3/detr_q3:cu124-lab
```

This lab image includes SSH, JupyterLab, TensorBoard, and the Stage 5
PyTorch/Ultralytics training environment. Default service ports are 22, 8888,
and 6006.

Gemini/VirtAI paths:

```text
/gemini/code      code package
/gemini/data-1    D-Fire dataset mount
/gemini/pretrain  yolo26n.pt and rtdetr-l.pt
/gemini/output    training outputs
```

Stage5-PV v2 uses one YOLO family (`yolo26n.pt`) and one non-YOLO family
(`rtdetr-l.pt`); `yolo26s.pt` is not part of the formal matrix.

Gemini/VirtAI start:

```bash
cd /gemini/code
bash stage5_h20_server/gemini_stage5_check.sh
bash stage5_h20_server/gemini_stage5_mincheck.sh
bash stage5_h20_server/gemini_stage5_formal.sh
```

Gemini/VirtAI outputs:

```text
/gemini/output/detr_Q3_results/runs/detect/runs_stage5_formal
/gemini/output/detr_Q3_results/formal_results/stage5
/gemini/output/detr_Q3_results/logs/h20
```

Local sync after SSH is known:

```powershell
Set-Location D:\detr_Q3
powershell -ExecutionPolicy Bypass -File stage5_h20_server\03_sync_to_h20.ps1 -Server user@host -RemoteRoot /data/detr_Q3
```

Server bootstrap:

```bash
cd /data/detr_Q3
bash stage5_h20_server/00_bootstrap_h20_env.sh
bash stage5_h20_server/01_check_h20_assets.sh
```

Stage 5 v1:

```bash
bash stage5_h20_server/run_stage5_v1_mincheck_4gpu.sh
bash stage5_h20_server/run_stage5_v1_all_queue.sh
```

After completion:

```bash
bash stage5_h20_server/run_stage5_v1_aggregate.sh
bash stage5_h20_server/run_stage5_v1_external_bowfire.sh
```

Stage5-PV v2:

```bash
bash stage5_h20_server/run_stage5_pv_v2_dfire_mincheck_4gpu.sh
bash stage5_h20_server/run_stage5_pv_v2_full_queue.sh
bash stage5_h20_server/run_stage5_pv_v2_aggregate.sh
```

The full queue serially generates shared equal-step/scheduled hard-negative
lists before parallel training waves, then runs
`dfire,dfs x yolo26n,rtdetr x seeds 11,22,33 x all4 arms x 300 epochs`.

The full queue is `dfire,dfs x yolo26n,rtdetr x seeds 11,22,33 x all4
arms x 300 epochs`. Prepare DFS first with
`scripts/stage5_pv_v2_prepare_dfs.py`; otherwise the queue exits before
launching partial jobs.

RT-DETR D-Fire 12-task queue:

`bash stage5_h20_server/run_stage5_pv_v2_rtdetr_dfire_12_tmux.sh`

默认行为：

- `AUTO_FILL_GPU=1`（默认）——4 张 H20 全部填满，任一任务结束后在对应 GPU 上立刻补新任务，减少空档。
- `ALLOW_RESUME=0`（默认）——严格检查：已有 `formal_results`/`runs` 目录会直接退出，不覆盖。

恢复行为（`ALLOW_RESUME=1`）：

- 对每个 arm 扫描 `runs/detect/runs_stage5_pv_v2/dfire_rtdetr_seed<seed>/<arm>/weights/last.pt`、`results.csv`、训练日志和 `run_meta.json`，判断是否已完整完成。
- 已完成 arm 输出：
  - `[skip] label=seed<seed>_<arm> reason=existing_done`
- 已完整导出结果输出：
  - `[skip] seed=<seed> reason=existing_export_complete`
- 未完成 arm 会按队列自动补齐，不改写已完成结果。

回退开关：

- `AUTO_FILL_GPU=0` 回退到按 seed 逐波次执行（每个 seed 4 个 arm 并行，完成后再跑下一个 seed），便于对照旧逻辑排障。

脚本日志关键格式：

- `[launch] label=... gpu=... log=...`
- `[done] label=... rc=...`
- `[error] label=... rc=... log=...`
- `[skip] label=... reason=existing_done`
- `[skip] seed=... reason=existing_export_complete`

导出时机：

- 某个 seed 的 4 个 arm 都成功完成后才触发该 seed 的 `export+eval`，不是等到固定“整个波次”尾部。

Monitor:

```bash
screen -ls
tail -f logs/h20/*.log
nvidia-smi
```

Gemini/VirtAI monitor:

```bash
nvidia-smi
tail -f /gemini/output/detr_Q3_results/logs/h20/*.log
```
