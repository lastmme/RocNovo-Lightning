from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
import tempfile

import click

from rocnovo.common.io import load_yaml, normalize_path
from rocnovo.data.io import format_to_hdf5
from rocnovo.common.logger import set_logger_dir, redirect_lightning_logs_to_loguru
from rocnovo.module.clip import train as clip_train
from rocnovo.module.denovo import train as denovo_train
from rocnovo.inference.inference import predict as _predict, DenovoPredictor

RAW_MS_EXTENSIONS = {".mgf", ".mzml", ".mzxml"}

@dataclass(frozen=True)
class SharedParams:
    config: Path
    log_dir: Path

def with_shared_params(func):
    @click.option(
        "-c", "--config",
        type=click.Path(exists=True, file_okay=True, dir_okay=False),
        required=True,
        help="config file path"
    )
    @click.option(
        "-l", "--log_dir",
        type=click.Path(file_okay=False, dir_okay=True),
        default=".",
        show_default=True,
        help="logging directory; a 'logs' subfolder will be created here"
    )
    @wraps(func)
    def wrapper(config: Path, log_dir: Path, *args, **kwargs):
        config = normalize_path(config)
        log_dir = normalize_path(log_dir)
        set_logger_dir(log_dir)
        redirect_lightning_logs_to_loguru()
        params = SharedParams(config=config, log_dir=log_dir)
        return func(params, *args, **kwargs)
    
    return wrapper

@click.group()
def main():
    pass

@main.command()
@click.option("-s", "--stage", type=click.Choice(["clip", "denovo"]), default="clip", help="train stage")
@with_shared_params
def train(params: SharedParams, stage: Literal["clip", "denovo"]="clip"):
    if stage == "clip":
        clip_train(params.config)
    elif stage == "denovo":
        denovo_train(params.config)
    else:
        raise ValueError(f"Unknown stage: {stage}")

def resolve_ms_inputs(
    ms_file: Path | None,
    ms_dir: Path | None,
    hdf5_workdir: Path,
    annotated: bool=False
):
    """Convert raw MS files to HDF5 and return the list of HDF5 paths."""
    raw_files: list[Path] = []
    if ms_dir is not None:
        ms_dir = normalize_path(ms_dir)
        for path in sorted(ms_dir.rglob("*")):
            if path.suffix.lower() in RAW_MS_EXTENSIONS:
                raw_files.append(path)
    
    elif ms_file is not None:
        raw_files.append(normalize_path(ms_file))
    else:
        return []

    hdf5_paths: list[Path] = []
    for raw_path in raw_files:
        format_to_hdf5(
            raw_path,
            hdf5_workdir,
            annotated=annotated
        )
        hdf5_path = hdf5_workdir / raw_path.with_suffix(".hdf5").name
        hdf5_paths.append(hdf5_path)
    
    return hdf5_paths

@main.command()
@with_shared_params
@click.option(
    "-p", "--checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    help="denovo checkpoint path"
)
@click.option(
    "-C", "--clip-checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    help="clip checkpoint path"
)
@click.option(
    "-f", "--ms-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    help="raw mass spectrometry file (.mgf/.mzml/.mzxml)"
)
@click.option(
    "-d", "--ms-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="directory containing raw mass spectrometry files"
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
    help="output directory for result CSV files"
)
@click.option(
    "-n", "--num-beams",
    type=int,
    required=False,
    help="beam width; 0 means greedy search"
)
@click.option(
    "-x", "--max-len",
    type=int,
    required=False,
    help="maximum decode length"
)
@click.option(
    "-w", "--hdf5-workdir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
    help="working directory for HDF5 files converted from raw MS files"
)
@click.option(
    "-m", "--mode",
    type=click.Choice(["denovo", "eval"]),
    required=False,
    help="inference mode: denovo or eval"
)
def predict(
    params: SharedParams,
    checkpoint: Path | None,
    clip_checkpoint: Path | None,
    ms_file: Path | None,
    ms_dir: Path | None,
    output_dir: Path | None,
    num_beams: int | None,
    max_len: int | None,
    hdf5_workdir: Path | None,
    mode: Literal["denovo", "eval"] | None
):
    config = load_yaml(params.config)

    if checkpoint is not None:
        config["checkpoint_path"] = str(normalize_path(checkpoint))
    
    if clip_checkpoint is not None:
        config["clip_checkpoint_path"] = str(normalize_path(clip_checkpoint))
    
    if output_dir is not None:
        output_dir = normalize_path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        config["output_dir"] = str(output_dir)
    
    if num_beams is not None:
        config["prediction"]["num_beams"] = num_beams
    
    if max_len is not None:
        config["prediction"]["max_len"] = max_len

    if hdf5_workdir is None:
        hdf5_workdir = config.get("hdf5_workdir")
    
    if hdf5_workdir is None:
        hdf5_workdir = tempfile.gettempdir()
    
    hdf5_workdir = normalize_path(hdf5_workdir)
    hdf5_workdir.mkdir(parents=True, exist_ok=True)

    if mode is not None:
        config["mode"] = mode

    mode_value = config.get("mode")
    if mode_value is None:
        raise ValueError("mode must be specified in config or via --mode")

    input_paths = resolve_ms_inputs(
        ms_file,
        ms_dir,
        hdf5_workdir,
        mode_value == "eval"
    )

    if input_paths:
        predictor = DenovoPredictor(config)
        output_dir_path = normalize_path(config["output_dir"])
        output_dir_path.mkdir(parents=True, exist_ok=True)
        for hdf5_path in input_paths:
            output_csv = output_dir_path / f"{hdf5_path.stem}.csv"
            predictor.predict_file(hdf5_path, output_csv, hdf5_path.stem)
    else:
        _predict(config)

if __name__ == "__main__":
    main()
