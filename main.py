from typing import Literal

import click

from rocnovo.common.io import normalize_path
from rocnovo.common.logger import set_logger_dir
from rocnovo.module.clip import train as clip_train
from rocnovo.module.denovo import train as denovo_train
from rocnovo.inference.inference import predict

@click.command()
@click.argument("mode", type=click.Choice(["train", "predict"]))
@click.option("--config", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--log_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--stage", type=click.Choice(["clip", "denovo"]), default="clip")
def main(
    mode: Literal["train", "predict"],
    config: str,
    log_dir: str,
    stage: Literal["clip", "denovo"]="clip"
):
    config_path = normalize_path(config)
    log_dir_path = normalize_path(log_dir)
    set_logger_dir(log_dir_path)

    if mode == "train":
        if stage == "clip":
            clip_train(config_path)
        elif stage == "denovo":
            denovo_train(config_path)
    
    elif mode == "predict":
        predict(config_path)

if __name__ == "__main__":
    main()
