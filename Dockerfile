FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple

WORKDIR /workspace

COPY requirements.txt /workspace/rocnovo_lightning/
RUN pip install --no-cache-dir -r /workspace/rocnovo_lightning/requirements.txt

COPY . /workspace/rocnovo_lightning/

WORKDIR /workspace/rocnovo_lightning

CMD ["/bin/bash"]