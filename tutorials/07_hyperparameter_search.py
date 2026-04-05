"""
lr: 0.0009862194965627857, n_epoch: 50, warmup_ratio: 0.1
best accuracy: 0.4565

| Dataset / Species | AA Precision | AA Recall | Peptide Precision | Peptide Recall | PTM Precision | PTM Recall | Full Accuracy | Curve AUC |
| :---------------- | :----------: | :-------: | :---------------: | :------------: | :-----------: | :--------: | :-----------: | :-------: |
| **HC PT**         |    0.676     |   0.675   |       0.519       |     0.519      |     0.764     |   0.792    |     0.518     |   0.478   |
"""
import re
import gc
import warnings
from dataclasses import asdict
from pathlib import Path
from packaging import version

import optuna
import optuna.importance
from optuna.trial import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
with optuna._imports.try_import() as _imports:
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback

import pandas as pd
from ruamel.yaml import YAML
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage

from rocnovo.common.logger import logger, set_logger_dir
from rocnovo.common.io import normalize_path, load_yaml
from rocnovo.module.denovo import Denovo
import rocnovo.config.data as data_config
import rocnovo.config.tokenizer as tokenizer_config
from rocnovo.config.train import TrainerConfig
from rocnovo.tokenizer.spectrum import SpectrumTokenizer
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer
from rocnovo.data.dataloaders import BiDirectDeNovoDataLoaderModule

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

denovo_yaml = """
aug:
  enabled: false

clip_checkpoint_path: "/data2/xp/RocNovo-Lightning/outputs/checkpoints/hc_pt_clip/checkpoints_step=17082_val_top1_accuracy=0.9036.ckpt"

tokenizer:
  spectrum:
    n_top_peaks: 300
    min_mz: 50.0
    max_mz: 4500.0
    min_intensity: 0.01
    remove_precursor_tol: 2.0

  peptide:
    reverse: true
    residues: "massivekb"

model:
  spectrum:
    hidden_size: 512
    n_head: 8
    n_layers: 9
    dropout: 0.18
    dim_feedforward: 1024
    activation: "gelu"
  peptide:
    n_vocab: 29
    hidden_size: 512
    n_head: 8
    n_layers: 9
    dropout: 0.18
    dim_feedforward: 1024
    activation: "gelu"

loss:
  label_smoothing: 0.05

optimizer:
  lr: 3.0e-4
  weight_decay: 1.0e-4

scheduler:
  enabled: true
  n_cycles: 1
  warmup_steps: 0.05
  lr_decay_factor: 10.0

trainer:
  is_max: true
  evaluate_metric_name: "total_accuracy"
  task_name: "hc_pt_denovo"
  random_seed: 1256
  save_top_k: 3
  max_epochs: 30
  devices: [0, 1]
  grad_norm_clip: 1.5
  validation_steps: 1000
  show_progress_bar: true
  grad_scaler_enable: true
  distributed: "ddp_spawn" # not ddp
  gradient_accumulation_steps: 1
  summarywriter_folder: "/data2/xp/RocNovo-Lightning/outputs/tb"
  model_save_folder: "/data2/xp/RocNovo-Lightning/outputs/checkpoints/hc_pt_denovo"

data:
  train_path: "/data2/xp/data/novobench/train_hdf5/hc_pt.hdf5"
  val_path: "/data2/xp/data/novobench/valid_hdf5/hc_pt.hdf5"
  test_path: "/data2/xp/data/novobench/test_hdf5/hc_pt.hdf5"
  train_batch_size: 32
  val_batch_size: 256
  test_batch_size: 256
  n_workers: 4
"""

_EPOCH_KEY = "ddp_pl:epoch"
_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_PRUNED_KEY = "ddp_pl:pruned"

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``lightning.pytorch.LightningModule.training_step`` or
            ``lightning.pytorch.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.


    .. note::
        For the distributed data parallel training, the version of PyTorchLightning needs to be
        higher than or equal to v1.6.0. In addition, :class:`~optuna.study.Study` should be
        instantiated with RDB storage.


    .. note::
        If you would like to use PyTorchLightningPruningCallback in a distributed training
        environment, you need to evoke ``PyTorchLightningPruningCallback.check_pruned()``
        manually so that :class:`~optuna.exceptions.TrialPruned` is properly handled.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        _imports.check()
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_fit_start(self, trainer: Trainer, pl_module: "pl.LightningModule") -> None:
        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if self.is_ddp_backend:
            if version.parse(pl.__version__) < version.parse(  # type: ignore[attr-defined]
                "1.6.0"
            ):
                raise ValueError("PyTorch Lightning>=1.6.0 is required in DDP.")
            # If it were not for this block, fitting is started even if unsupported storage
            # is used. Note that the ValueError is transformed into ProcessRaisedException inside
            # torch.
            if not (
                isinstance(self._trial.study._storage, _CachedStorage)
                and isinstance(self._trial.study._storage._backend, RDBStorage)
            ):
                raise ValueError(
                    "optuna_integration.PyTorchLightningPruningCallback"
                    " supports only optuna.storages.RDBStorage in DDP."
                )
            # It is necessary to store intermediate values directly in the backend storage because
            # they are not properly propagated to main process due to cached storage.
            # TODO(Shinichi) Remove intermediate_values from system_attr after PR #4431 is merged.
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(
                    self._trial._trial_id,
                    _INTERMEDIATE_VALUE,
                    dict(),
                )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Trainer calls `on_validation_end` for sanity check. Therefore, it is necessary to avoid
        # calling `trial.report` multiple times at epoch 0. For more details, see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        epoch = pl_module.current_epoch
        should_stop = False

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        # Determine if the trial should be terminated in a DDP.
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()

            # Update intermediate value in the storage.
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _INTERMEDIATE_VALUE, intermediate_values
            )

        # Terminate every process if any world process decides to stop.
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _PRUNED_KEY, True)
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _EPOCH_KEY, epoch)

    def check_pruned(self) -> None:
        """Raise :class:`optuna.TrialPruned` manually if pruned.

        Currently, ``intermediate_values`` are not properly propagated between processes due to
        storage cache. Therefore, necessary information is kept in ``trial.system_attrs`` when the
        trial runs in a distributed situation. Please call this method right after calling
        ``lightning.pytorch.Trainer.fit()``.
        If a callback doesn't have any backend storage for DDP, this method does nothing.
        """

        _trial_id = self._trial._trial_id
        _study = self._trial.study
        # Confirm if storage is not InMemory in case this method is called in a non-distributed
        # situation by mistake.
        if not isinstance(_study._storage, _CachedStorage):
            return

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)

        # Confirm if DDP backend is used in case this method is called from a non-DDP situation by
        # mistake.
        if intermediate_values is None:
            return
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = _trial_system_attrs.get(_EPOCH_KEY)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

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

def objective(trial: Trial, config: dict, base_save_dir: Path):
    gc.collect()
    torch.cuda.empty_cache()
    n_epoch = trial.suggest_int('n_epoch', 20, 50, step=5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.01, 0.02, 0.05, 0.1, 0.2])
    
    desc = f"lr: {lr}, n_epoch: {n_epoch}, warmup_ratio: {warmup_ratio}"
    logger.info(desc)

    trial_index = trial.number + 1
    trial_save_dir = base_save_dir.joinpath(f"trial_{trial_index}")
    trial_save_dir.mkdir(parents=True, exist_ok=True)
    
    config["data"]["train_batch_size"] = 192
    config["trainer"]["validation_steps"] = 500
    config["optimizer"]["lr"] = lr
    config["trainer"]["max_epochs"] = n_epoch
    config["trainer"]["summarywriter_folder"] = str(trial_save_dir.joinpath("tb"))
    config["trainer"]["model_save_folder"] = str(trial_save_dir.joinpath("checkpoints"))
    config["scheduler"]["warmup_steps"] = warmup_ratio
    yaml.dump(config, trial_save_dir.joinpath("denovo.yaml"))
    # avoid ScalerInt type error
    config = load_yaml(trial_save_dir.joinpath("denovo.yaml"))
    trainer_config = TrainerConfig(**config["trainer"])
    spectrum_tokenizer_config = tokenizer_config.SpectrumTokenizerConfig(**config["tokenizer"]["spectrum"])
    peptide_tokenizer_config = tokenizer_config.PTMTokenizerConfig(**config["tokenizer"]["peptide"])
    spectrum_tokenizer = SpectrumTokenizer(**asdict(spectrum_tokenizer_config))
    spectrum_tokenizer.disable_aug()
    peptide_tokenizer = PTMPeptideTokenizer(**asdict(peptide_tokenizer_config))
    
    data_module = BiDirectDeNovoDataLoaderModule(
        data_config.DataConfig(**config["data"]),
        spectrum_tokenizer,
        peptide_tokenizer
    )
    module = Denovo(config)
    
    pl.seed_everything(trainer_config.random_seed)
    torch.set_float32_matmul_precision("medium")
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
        patience=10,
        mode="max" if trainer_config.is_max else "min"
    )
    pruning_callback = PyTorchLightningPruningCallback(
        trial,
        monitor="val_" + trainer_config.evaluate_metric_name
    )
    tb_logger = TensorBoardLogger(
        name=None,
        save_dir=config["trainer"]["summarywriter_folder"],
        version=""
    )
    trainer = pl.Trainer(
        max_epochs=trainer_config.max_epochs,
        enable_progress_bar=trainer_config.show_progress_bar,
        log_every_n_steps=10,
        accelerator="gpu",
        val_check_interval=trainer_config.validation_steps,
        devices=trainer_config.devices,
        strategy="ddp_spawn",
        precision="bf16-mixed" if trainer_config.grad_scaler_enable else "32-true",
        gradient_clip_val=trainer_config.grad_norm_clip,
        accumulate_grad_batches=trainer_config.gradient_accumulation_steps,
        default_root_dir=trainer_config.model_save_folder,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, pruning_callback]
    )
    logger.info(f"Trial {trial.number}, trainer.fit ...")
    trainer.fit(module, datamodule=data_module)
    pruning_callback.check_pruned()
    
    metric = search_best_denovo_model(trial_save_dir.joinpath("checkpoints"))
    logger.info(f"train done, trial_save_dir: {trial_save_dir}")
    logger.info(f"best accuracy: {metric}")
    return metric

if __name__ == '__main__':
    base_denovo_config = yaml.load(denovo_yaml)
    base_save_dir = normalize_path("./outputs/hyperparameter_search")
    base_save_dir.mkdir(parents=True, exist_ok=True)
    set_logger_dir(base_save_dir)

    db_path = base_save_dir.joinpath("hcpt_optuna.db")
    storage_url = f"sqlite:///{db_path}"
    # pip install optuna-dashboard
    logger.info(f"use SQLite to store Optuna states: {storage_url}")
    logger.info("use the command-line to view the metrics in the web: optuna-dashboard " + storage_url)
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=1,
        n_warmup_steps=1
    )
    study = optuna.create_study(
        study_name="rocnovo",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        pruner=pruner
    )
    logger.info(f"current Study finish the number of Trial: {len(study.trials)}")
    study.optimize(
        lambda trial: objective(
            trial,
            base_denovo_config,
            base_save_dir,
        ),
        n_trials=10,
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