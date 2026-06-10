from typing import Any, Union, Literal, Optional
from jinja2 import Template
from dataclasses import dataclass

import yaml

@dataclass
class SchedulerConfig:
    enabled: bool=True
    n_cycles: int=1
    lr_decay_factor: float=1.0
    warmup_steps: int | float=0.1

@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float

@dataclass
class TrainerConfig:
    max_epochs: int
    validation_steps: int | float
    model_save_folder: str
    summarywriter_folder: str
    devices: Union[list[int], int]
    task_name: str="task"
    random_seed: int=42
    save_top_k: Optional[int]=None
    distributed: str="auto"
    mode: Literal["full_state", "weight_only"]="weight_only"
    checkpoint_path: Optional[str]=None
    grad_norm_clip: Optional[float]=None
    show_progress_bar: bool=False
    grad_scaler_enable: bool=True
    gradient_accumulation_steps: int=1
    is_max: bool=True
    evaluate_metric_name: str="top1_acc"

class Config:
    def __init__(self, path: str):
        with open(path, "r") as f:
            template: Template = Template(f.read())
            self.config = yaml.safe_load(template.render())
            self._convert_optim_beta_type()
    
    def __getitem__(self, name: str) -> Any:
        return self.config[name]

    def __repr__(self):
        return repr(self.config)

    def dict(self):
        return self.config

    def _convert_optim_beta_type(self):
        if "betas" in self.config["optimizer"]:
            self.config["optimizer"]["betas"] = tuple(self.config["optimizer"]["betas"])