# H20 Docker Image

This is an environment-only image for the DETR Q3 Stage 5 H20 server. It does
not include code, datasets, or model weights. Keep those under:

```text
/data/detr_Q3
```

## Build

```bash
cd /data/detr_Q3/stage5_h20_server/docker
bash build_h20_image.sh
```

Image tag:

```text
detr-q3-h20:cu124
```

Lab/service image tag:

```text
hpc.chzu.edu.cn:32402/tcvmpvo5lnx3/detr_q3:cu124-lab
```

Base image:

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
```

## Smoke Test

```bash
bash test_h20_image.sh
```

## Save As Uploadable Tar

```bash
bash save_h20_image.sh
```

This writes:

```text
/data/detr_Q3/transfer_packages/detr-q3-h20-cu124.tar
/data/detr_Q3/transfer_packages/detr-q3-h20-cu124.tar.sha256
```

## Push Lab Image

```bash
cd /data/detr_Q3/stage5_h20_server/docker
docker login hpc.chzu.edu.cn:32402
bash push_h20_lab_image.sh
```

The lab image starts SSH, JupyterLab, and TensorBoard by default.

Default ports:

```text
SSH          22
JupyterLab  8888
TensorBoard 6006
```

Useful environment variables:

```text
SSH_PUBLIC_KEY   optional public key appended to /root/.ssh/authorized_keys
ROOT_PASSWORD    optional root password for SSH password login
JUPYTER_TOKEN    optional fixed token; generated at startup if omitted
PROJECT_ROOT     default /gemini/code
RESULT_ROOT      default /gemini/output/detr_Q3_results
```

Service logs are written under:

```text
/var/log/h20-services
```

## Run Training Inside The Image

If you use raw Docker:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v /data/detr_Q3:/data/detr_Q3 \
  detr-q3-h20:cu124 \
  bash
```

Inside the container:

```bash
cd /data/detr_Q3
bash stage5_h20_server/01_check_h20_assets.sh
bash stage5_h20_server/run_stage5_v1_mincheck_4gpu.sh
bash stage5_h20_server/run_stage5_v1_all_queue.sh
```
