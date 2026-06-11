#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-detr-q3-h20:cu124}"

docker run --rm --gpus all "$IMAGE_TAG" python3 - <<'PY'
import torch
import ultralytics
import numpy

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("ultralytics", ultralytics.__version__)
print("numpy", numpy.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("device_count", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(idx, torch.cuda.get_device_name(idx))
PY
