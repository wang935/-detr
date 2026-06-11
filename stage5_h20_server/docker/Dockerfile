FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    PROJECT_ROOT=/data/detr_Q3 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        htop \
        iproute2 \
        nano \
        net-tools \
        openssh-server \
        procps \
        tini \
        tmux \
        vim \
        wget \
    && mkdir -p /run/sshd /root/.ssh \
    && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip "setuptools<81" wheel

COPY requirements-h20.txt /tmp/requirements-h20.txt
RUN python -m pip install -r /tmp/requirements-h20.txt \
    && python -m pip install jupyterlab tensorboard ipywidgets \
    && python -m pip install --no-deps polars ultralytics-thop ultralytics \
    && python -m pip install "setuptools<81" \
    && rm -f /tmp/requirements-h20.txt

COPY start_h20_services.sh /usr/local/bin/start_h20_services.sh

RUN chmod +x /usr/local/bin/start_h20_services.sh \
    && mkdir -p /tmp/matplotlib /tmp/ultralytics /gemini/code /gemini/output /var/log/h20-services \
    && chmod -R 777 /tmp/matplotlib /tmp/ultralytics /var/log/h20-services

EXPOSE 22 8888 6006
WORKDIR /data/detr_Q3
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start_h20_services.sh"]
