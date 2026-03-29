from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from functools import wraps

import click

from rocnovo.common.io import normalize_path
from rocnovo.common.logger import set_logger_dir
from rocnovo.module.clip import train as clip_train
from rocnovo.module.denovo import train as denovo_train
from rocnovo.inference.inference import predict

@dataclass(frozen=True)
class SharedParams:
    config: Path
    log_dir: Path

def with_shared_params(func):
    @click.option(
        "--config", 
        type=click.Path(exists=True, file_okay=True, dir_okay=False),
        required=True,
        help="config file path"
    )
    @click.option(
        "--log_dir", 
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        required=True,
        help="logging directory"
    )
    @wraps(func)
    def wrapper(config: Path, log_dir: Path, *args, **kwargs):
        config = normalize_path(config)
        log_dir = normalize_path(log_dir)
        set_logger_dir(log_dir)
        params = SharedParams(config=config, log_dir=log_dir)
        return func(params, *args, **kwargs)
    
    return wrapper

@click.group()
def main():
    pass

@main.command()
@click.option("--stage", type=click.Choice(["clip", "denovo"]), default="clip", help="train stage")
@with_shared_params
def train(params: SharedParams, stage: Literal["clip", "denovo"]="clip"):
    if stage == "clip":
        clip_train(params.config)
    elif stage == "denovo":
        denovo_train(params.config)
    else:
        raise ValueError(f"Unknown stage: {stage}")

@main.command()
@with_shared_params
def denovo(params: SharedParams):
    predict(params.config)

if __name__ == "__main__":
    main()
