FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

ARG PIP_INDEX_URL=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/rocnovo_lightning/

RUN if [ -n "$PIP_INDEX_URL" ]; then \
    pip config set global.index-url "$PIP_INDEX_URL"; \
    fi && \
    pip install --no-cache-dir -r /workspace/rocnovo_lightning/requirements.txt

COPY . /workspace/rocnovo_lightning/

WORKDIR /workspace/rocnovo_lightning

CMD ["/bin/bash"]