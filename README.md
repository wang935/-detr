# detr_Q3 Training Model Code

This is a source-only export of the training/model code from `D:\detr_Q3`.

## Included

- YOLO/RT-DETR stage scripts and one-click Windows entrypoints.
- Stage 3 and Stage 5 local/multi-host training launchers.
- Stage 5 PV-v2, InternImage, and H20 server helper scripts.
- Evaluation, aggregation, plotting, and dataset-preparation utilities.
- Selected dataset config/list files needed by the launch scripts.
- Third-party InternImage source code under `external_code/InternImage`; keep its upstream MIT license when redistributing.

## Excluded

The export intentionally excludes generated or heavy artifacts:

- Full datasets and image/label trees.
- Model weights and checkpoints such as `.pt`, `.pth`, `.onnx`, and engine files.
- Training outputs, logs, predictions, formal result folders, and transfer packages.
- Python caches, temporary folders, wheels, and local environment mirrors.

To run the training scripts, restore datasets and pretrained weights to the paths expected by the command README files, or edit the YAML/path files under `data/`.

## Useful Entry Points

- `stage5_5hosts/README_CMD.md`: formal Stage 5 host/seed command map.
- `stage5_pv_v2/README_CMD.md`: PV-v2 hard-negative schedule commands.
- `stage5_internimage/README_CMD.md`: InternImage D-Fire training commands.
- `stage5_h20_server/README_CMD.md`: H20 server workflow.
- `scripts/stage5_formal_runner.py`: Stage 5 formal runner.
- `scripts/stage5_pv_v2_runner.py`: Stage 5 PV-v2 runner.
