# RocNovo-Lightning

Lightning version of RocNovo. The original repository is [RocNovo](https://github.com/lastmme/RocNovo).

## Installation

We provide a brief description of the environment installation and configuration in [installation.md](./docs/installation.md).

You can choose to manage your virtual environment using either Conda or uv.

Option A: Using Conda
First, create and activate a conda virtual environment, then install the package:

```bash
conda create -n rocnovo-lightning python=3.11
conda activate rocnovo-lightning
pip install .
```

Option B: Using uv
First, create and activate a uv virtual environment, then install the package:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install .
```

Run the following command to check if the installation was successful:
```bash
rocnovo --help
```

```bash
Usage: rocnovo [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  denovo
  train
```

The command-line arguments configuration here is the same as in `main.py`.

## Usage

### Model Training

Before training the model, you need to convert the data into the corresponding format. You can check data_processing for details.

Once the conversion is complete and you obtain the hdf5 files, you need to prepare the configuration files for training. Model training is divided into two stages: the clip stage and the denovo stage.

#### Clip Stage

For the clip stage, you need to prepare the clip training configuration file.

Below is the configuration file we used when training on the massiveKB dataset:

You need to change the paths below to your own paths. We provide a detailed introduction to the parameters in the [train_clip](./tutorials/03_train_clip.ipynb).

```bash
rocnovo train \
--stage clip \
--config ./configs/clip.yaml \
--log_dir ./outputs/clip
```

#### Denovo Stage

For the denovo stage, you need to prepare the denovo training configuration file. An additional parameter, `clip_checkpoint_path`, is added to specify the checkpoint file from the clip training.

You can check the [train_denovo](./tutorials/04_train_denovo.ipynb) for details.

If you do not want to execute the command line twice, you can refer to the code in `end2end_train.py` to run it all at once. It will scan the checkpoint files based on the target storage path of the clip stage and automatically load the optimal checkpoint file from the clip training.

### Model Inference

If you want to perform inference, you need to prepare the inference configuration file.

You can check the [inference](./tutorials/05_inference.ipynb) for details.

If your data does not have labels, you can change the mode parameter below to "denovo", and then run the inference.

```bash
rocnovo denovo \
--config ./configs/inference.yaml \
--log_dir ./outputs/inference
```

## Benchmark

We provide comprehensive benchmark results for inference speed, as well as amino acid-level and peptide-level metrics in [benchmark](./docs/benchmarks.md), evaluating both zero-shot generalization on nine species V1 and V2 datasets, as well as the standard in-domain performance on NovoBench.

## Contact

If you have any questions or suggestions, please contact us:

- Peng Xiong: pengx@mail.ustc.edu.cn
- Hongtao Xu: xht020521@mail.ustc.edu.cn