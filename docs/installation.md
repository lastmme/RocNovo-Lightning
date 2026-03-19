# Environment Setup

This project is developed and tested with **Python 3.11**, **PyTorch 2.6.0**, and **CUDA 12.4**.

You can set up the environment either locally using pip or via Docker (recommended for reproducibility).

## Local Installation
For convenience, we provide a `requirements.txt` file. You can easily install all dependencies in your local Python environment:

```sh
pip install -r requirements.txt
```

## Docker Installation (Recommended)
We highly recommend using Docker to avoid environment conflicts. You can either build the image yourself using the provided Dockerfile or use our pre-built image.

### Run the container

```sh
docker run -itd \
  --gpus all \
  --name rocnovo_lightning \
  --shm-size=64g
  rocnovo_lightning:latest
```

If you are using Visual Studio Code, we recommend that you install the Dev Containers extension to attach VS Code to the running container.