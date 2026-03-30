import re
import gc
import subprocess
from pathlib import Path

import optuna
import optuna.importance
from optuna.trial import Trial
import torch
import pandas as pd
from ruamel.yaml import YAML

from rocnovo.common.logger import logger, set_logger_dir
from rocnovo.common.io import normalize_path

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

def search_best_denovo_model(denovo_model_save_folder: Path):
    def extract_metrics(file_path):
        match = re.search(r"step=(\d+).*accuracy=(\d+(?:\.\d+)?)", file_path.name)
        if match:
            accuracy = float(match.group(2))
            return accuracy
        
        return -1.0
    
    ckpt_files = list(denovo_model_save_folder.glob("*.ckpt"))
    metrics = [extract_metrics(file_path) for file_path in ckpt_files]
    return max(metrics)

def objective(trial: Trial, config_path: Path, base_save_dir: Path):
    gc.collect()
    torch.cuda.empty_cache()
    n_epoch = trial.suggest_int('n_epoch', 10, 50, step=5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.01, 0.02, 0.05, 0.1, 0.2])
    
    desc = f"lr: {lr}, n_epoch: {n_epoch}, warmup_ratio: {warmup_ratio}"
    logger.info(desc)

    trial_index = trial.number
    trial_save_dir = base_save_dir.joinpath(f"trial_{trial_index}")
    trial_save_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.load(config_path)
    
    config["data"]["train_batch_size"] = 192
    config["trainer"]["validation_steps"] = 500
    config["optimizer"]["lr"] = lr
    config["trainer"]["max_epochs"] = n_epoch
    config["trainer"]["summarywriter_folder"] = str(trial_save_dir.joinpath("tb"))
    config["trainer"]["model_save_folder"] = str(trial_save_dir.joinpath("checkpoints"))
    config["scheduler"]["warmup_steps"] = warmup_ratio
    yaml.dump(config, trial_save_dir.joinpath("denovo.yaml"))
    
    subprocess.run(
        [
            "python", "main.py", "train",
            "--stage", "denovo",
            "--config", f"{str(trial_save_dir.joinpath('denovo.yaml'))}",
            "--log_dir", f"{str(base_save_dir)}"
        ],
        check=True
    )
    logger.info(f"train done, trial_save_dir: {trial_save_dir}")
    metric = search_best_denovo_model(trial_save_dir.joinpath("checkpoints"))
    logger.info(f"best accuracy: {metric}")
    return metric

if __name__ == '__main__':
    base_denovo_config_path = normalize_path("./yamls/novobench/denovo/hc_pt.yaml")
    base_save_dir = normalize_path("./outputs/hyperparameter_search")
    base_save_dir.mkdir(parents=True, exist_ok=True)
    set_logger_dir(base_save_dir)

    study = optuna.create_study(study_name="rocnovo", direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial,
            base_denovo_config_path,
            base_save_dir,
        ),
        n_trials=50,
        show_progress_bar=True,
    )
    params = study.best_params
    logger.info(params)
    df: pd.DataFrame = study.trials_dataframe(
        attrs=('number', 'value', 'params', 'state')
    )
    logger.info(f"\n{df}")
    df.to_csv(f"./outputs/rocnovo_params.tsv", sep='\t')
    importance = optuna.importance.get_param_importances(study)
    for param, importance_value in importance.items():
        logger.info(f"Parameter: {param}, Importance: {importance_value:.4f}")