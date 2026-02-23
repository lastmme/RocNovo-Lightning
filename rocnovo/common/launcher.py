from dataclasses import asdict
from abc import ABC, abstractmethod

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import AdamW
from tqdm import tqdm

import rocnovo.config.train as train_configs
from rocnovo.data.dataloaders import BaseDataModule
from rocnovo.common.scheduler import get_restart_cosine_decay_scheduler_with_warmup

class BaseModule(ABC, LightningModule):
    """Lightning Base Class"""
    def __init__(self, config: dict):
        super().__init__()
        self.config: dict = config
        self.model_config: dict = config["model"]
        self.trainer_config = train_configs.TrainerConfig(**config["trainer"])
        self.optimizer_config = train_configs.OptimizerConfig(**config["optimizer"])
        self.scheduler_config = train_configs.SchedulerConfig(**config["scheduler"])
        self.save_hyperparameters(config, logger=False)
        self.init_model()
    
    @abstractmethod
    def init_model(self):
        pass

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        self.should_skip_lr_scheduler_step = False
        scaler = getattr(self.trainer.strategy.precision_plugin, "scaler", None)
        if scaler:
            scale_before_step = scaler.get_scale()
        optimizer.step(closure=optimizer_closure)
        if scaler:
            scale_after_step = scaler.get_scale()
            self.should_skip_lr_scheduler_step = scale_before_step > scale_after_step

    def lr_scheduler_step(self, scheduler, metric):
        if self.should_skip_lr_scheduler_step:
            return

        scheduler.step()
    
    def configure_optimizers(self):
        optimizer = AdamW(
            (param for param in self.parameters() if param.requires_grad), 
            **asdict(self.optimizer_config)
        )

        if self.scheduler_config.enabled:
            total_steps = self.trainer.estimated_stepping_batches
            print(f"total_steps = {total_steps}")
            scheduler = get_restart_cosine_decay_scheduler_with_warmup(
                optimizer,
                self.scheduler_config.warmup_steps,
                total_steps,
                self.scheduler_config.n_cycles,
                self.scheduler_config.lr_decay_factor
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "cosine_lr" 
                }
            }
        else:
            return optimizer
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, False)
        return loss

    @abstractmethod
    def common_step(self, batch, batch_idx, train: bool=True):
        pass

    def on_train_epoch_start(self):
        current_epoch = self.current_epoch + 1
        total_epochs = self.trainer.max_epochs
        description = f"Train Epochs[{current_epoch}/{total_epochs}]"
        
        if hasattr(self.trainer, "progress_bar_callback"):
            pbar: tqdm = self.trainer.progress_bar_callback.train_progress_bar
            if pbar is not None:
                pbar.set_description(description)

def create_tb_checkpoint_dir(trainer_config: train_configs.TrainerConfig):
    Path(trainer_config.summarywriter_folder).mkdir(parents=True, exist_ok=True)
    Path(trainer_config.model_save_folder).mkdir(parents=True, exist_ok=True)

def get_unique_log_dir(base: str) -> str:
    if not os.path.exists(base):
        return base

    i = 1
    while os.path.exists(f"{base}_{i}"):
        i += 1
    
    return f"{base}_{i}"

def build_trainer(
    trainer_config: train_configs.TrainerConfig,
    mode: Literal["train", "test"]="train"
):
    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_config.model_save_folder,
        filename=f"checkpoints_{{step}}_{{val_{trainer_config.evaluate_metric_name}:.4f}}",
        monitor="val_" + trainer_config.evaluate_metric_name,
        save_top_k=trainer_config.save_top_k if trainer_config.save_top_k is not None else 1,
        mode="max" if trainer_config.is_max else "min"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_" + trainer_config.evaluate_metric_name,
        min_delta=0.00,
        verbose=False,
        patience=50,
        mode="max" if trainer_config.is_max else "min"
    )
    trainer = pl.Trainer(
        max_epochs=trainer_config.max_epochs,
        enable_progress_bar=trainer_config.show_progress_bar,
        log_every_n_steps=10,
        accelerator="gpu",
        val_check_interval=trainer_config.validation_steps,
        devices=trainer_config.devices,
        strategy=trainer_config.distributed if mode == "train" else "auto",
        precision="bf16-mixed" if trainer_config.grad_scaler_enable else "32-true",
        gradient_clip_val=trainer_config.grad_norm_clip if trainer_config.grad_norm_clip is not None else None,
        accumulate_grad_batches=trainer_config.gradient_accumulation_steps,
        default_root_dir=trainer_config.model_save_folder,
        logger=TensorBoardLogger(
            name=None,
            save_dir=get_unique_log_dir(
                os.path.join(
                    trainer_config.summarywriter_folder,
                    trainer_config.task_name
                )
            ),
            version=""
        ) if mode == "train" else False,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    return trainer

def concat_results(results):
    preds = []
    labels = []

    for batch in results:
        preds.append(batch["pred"])
        labels.append(batch["label"])
    
    return {
        "preds": np.concatenate(preds, axis=0).squeeze(),
        "labels": np.concatenate(labels, axis=0).squeeze()
    }

def pipeline(
    module: BaseModule,
    trainer_config: train_configs.TrainerConfig,
    data_module: BaseDataModule,
    mode: Literal["train", "test", "prediction"]="train"
):
    pl.seed_everything(trainer_config.random_seed)
    torch.set_float32_matmul_precision("medium")
    create_tb_checkpoint_dir(trainer_config)
    trainer = build_trainer(
        trainer_config,
        mode
    )
    results = None
    if mode == "train":
        trainer.fit(
            module,
            datamodule=data_module,
            ckpt_path=trainer_config.checkpoint_path if trainer_config.mode == "full_state" else None
        )

    elif mode == "test":
        # 多卡预测可以考虑在 dataset 中返回 idx 进行处理
        # 得到结果后，再手动进行去重
        trainer.test(module, datamodule=data_module)
        results = concat_results(trainer.model.test_results)
    else:
        raise ValueError(f"mode {mode} not supported")

    return results