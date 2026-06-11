# Stage 5 最小环境测试命令

这些脚本只用于启动前验证，不进入论文结果。

测试输出隔离在：

```text
formal_results\stage5_mincheck\
runs\detect\runs_stage5_mincheck\
```

测试设置：

- `epochs=1`
- `--limit 256`
- `--arm all3`
- `--val false`
- `--overwrite` 只作用于 mincheck 测试目录

注意：`--limit 256` 只导出 eval pool 的前 256 张图，但 evaluation 仍会读取完整 calibration/test labels。因此 mincheck 指标可能显示 target recall not reachable；这是预期行为。mincheck 只验证环境、训练、导出、评估 JSON 结构和协议字段，不作为论文指标。

每台机器第一次测试前先运行：

```bat
cd /d D:\detr_Q3
stage5_3060\00_create_or_check_env.bat
```

推荐最小覆盖：

- PC1 跑 YOLO mincheck。
- PC2 跑 YOLO mincheck 和 RT-DETR mincheck，因为它后续会接两类任务。
- PC3 跑 RT-DETR mincheck。

通过标准：对应 `.bat` 返回 0，并且生成的 `gonogo.json` / `run_meta.json` 包含 `eval_protocol=calibration_threshold_test_report`、split md5 和实际导出权重路径。由于 1 epoch + `--limit 256` 可能无法达到 R0.90，`TEST_aggregate_min.bat` 只作为聚合脚本结构检查；若提示无可达 fixed-recall operating point，不代表正式 300-epoch 实验失败。
