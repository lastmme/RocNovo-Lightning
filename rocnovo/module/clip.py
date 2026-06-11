from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn as dist_nn

import rocnovo.config.data as data_config
import rocnovo.config.tokenizer as tokenizer_config
from rocnovo.common.launcher import pipeline
from rocnovo.tokenizer.spectrum import SpectrumTokenizer
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer
from rocnovo.common.io import normalize_path
from rocnovo.common.logger import logger
from rocnovo.common.launcher import BaseModule
from rocnovo.config.train import Config, TrainerConfig
from rocnovo.config.aug import AugmentationConfig
from rocnovo.components.encoders import ClipSpectrumEncoder
from rocnovo.components.decoders import ClipPeptideDecoder
from rocnovo.data.dataloaders import DeNovoDataLoaderModule

class Clip(BaseModule):
    def __init__(self, config: dict):
        super().__init__(config)
    
    def init_model(self):
        self.spectrum_encoder = ClipSpectrumEncoder(
            **self.model_config["spectrum"]
        )
        self.peptide_decoder = ClipPeptideDecoder(
            **self.model_config["peptide"]
        )
        self.global_spectrum = nn.Linear(
            self.model_config["spectrum"]["hidden_size"],
            1
        )
        learnable_logits_scale = self.model_config.get("learnable_logits_scale", True)
        logits_scale = self.model_config["logits_scale"]
        if learnable_logits_scale:
            self.logits_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / logits_scale)
            )
        else:
            self.register_buffer(
                "logits_scale", 
                torch.ones([]) * np.log(1 / logits_scale)
            )

    def encode_spectrum(self, spectra: data_config.Spectra):
        pkt, mask = self.spectrum_encoder(spectra)  # (B, L + 1, D)
        return pkt, mask
    
    def encode_peptide(self, tokens: torch.LongTensor):
        tgt = self.peptide_decoder(tokens)  # (B, L, D)
        return tgt
    
    def repr_spectrum(self, spectra: data_config.Spectra):
        pkt, mask = self.encode_spectrum(spectra)
        # pkt: (B, L + 1, D), mask: (B, L + 1)
        weight_scores: torch.FloatTensor = self.global_spectrum(pkt)  # (B, L + 1, 1)
        weight_scores = weight_scores.squeeze(-1)  # (B, L + 1)
        weight_scores = weight_scores.masked_fill(mask, float("-inf"))  # (B, L + 1)
        weights = torch.softmax(weight_scores, dim=1)  # (B, L + 1)
        pkt = torch.matmul(weights.unsqueeze(1), pkt).squeeze(1)  # (B, D)
        return pkt  # (B, D)

    def repr_peptide(self, peptide: data_config.Peptide):
        tgt = self.encode_peptide(peptide.tokens)
        cols = torch.sum(peptide.mask, dim=1) - 1
        rows = torch.arange(peptide.mask.shape[0], device=peptide.tokens.device)
        return tgt[rows, cols]  # (B, D)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    @torch.amp.autocast("cuda", enabled=False)
    def anchor_dot_contrast(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        logits_exp: torch.Tensor
    ):
        batch_size = a.shape[0]
        device = a.device
        a = a.float()
        b = b.float()
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        logits = torch.matmul(a, b.T) * logits_exp
        rank = self.global_rank
        targets: torch.LongTensor = torch.arange(
            batch_size,
            dtype=torch.long,
            device=device
        ) + rank * batch_size
        loss = F.cross_entropy(logits, targets)
        acc = (logits.argmax(dim=-1) == targets).float().mean()
        return loss, logits, acc
    
    def dist_nn_all_gather(self, data: torch.Tensor):
        if not torch.distributed.is_initialized():
            return data
        
        return torch.concat(dist_nn.all_gather(data), dim=0)

    def forward(self, spectra: data_config.Spectra, peptides: data_config.Peptide):
        pkt_features = self.repr_spectrum(spectra)
        tgt_features = self.repr_peptide(peptides)
        gathered_pkt_features = self.dist_nn_all_gather(pkt_features)
        gathered_tgt_features = self.dist_nn_all_gather(tgt_features)
        
        loss_p2t, sim_p2t, p2t_acc = self.anchor_dot_contrast(
            pkt_features,
            gathered_tgt_features,
            self.logits_scale.exp()
        )
        loss_t2p, sim_t2p, tp2_acc = self.anchor_dot_contrast(
            tgt_features,
            gathered_pkt_features,
            self.logits_scale.exp()
        )
        loss = (loss_p2t + loss_t2p) / 2
        top1_acc = (p2t_acc + tp2_acc) / 2
        return loss, sim_p2t, sim_t2p, top1_acc
        
    def common_step(self, batch: data_config.TrainBatch, batch_idx: int, train: bool=True):
        loss, _, _, top1_acc = self.forward(batch.spectra, batch.peptide)
        kwargs = {
            "on_epoch": True,
            "on_step": True if train else False,
            "sync_dist": True
        }
        prefix = "train_" if train else "val_"
        self.log(f"{prefix}loss", loss, prog_bar=train, **kwargs)
        self.log(f"{prefix}{self.trainer_config.evaluate_metric_name}", top1_acc, **kwargs)
        return loss

def train(config_path: str):
    config_path = normalize_path(config_path)

    logger.debug(f"Start to train clip model with config: {config_path}")

    config = Config(config_path)
    trainer_config = TrainerConfig(**config["trainer"])
    checkpoint_path = trainer_config.checkpoint_path
    mode = trainer_config.mode
    if checkpoint_path is not None:
        logger.debug(f"Resuming training from checkpoint: {trainer_config.checkpoint_path}, mode: {trainer_config.mode}")
        module = Clip.load_from_checkpoint(
            trainer_config.checkpoint_path,
            map_location="cpu"
        )
        if trainer_config.mode == "full_state":
            trainer_config = module.trainer_config
            if trainer_config.checkpoint_path is None:
                trainer_config.checkpoint_path = checkpoint_path
                trainer_config.mode = mode
    else:
        module = Clip(config.dict())
        logger.debug(f"Start to train clip model from scratch with config")
    
    aug_config = AugmentationConfig(**config["aug"])
    spectrum_tokenizer_config = tokenizer_config.SpectrumTokenizerConfig(**config["tokenizer"]["spectrum"])
    peptide_tokenizer_config = tokenizer_config.PTMTokenizerConfig(**config["tokenizer"]["peptide"])
    
    spectrum_tokenizer = SpectrumTokenizer(**asdict(spectrum_tokenizer_config))
    if aug_config.enabled:
        spectrum_tokenizer.set_aug_config(aug_config)
    
    logger.debug(f"Train with augmentation: {aug_config.enabled}")
    peptide_tokenizer = PTMPeptideTokenizer(**asdict(peptide_tokenizer_config))
    data_module = DeNovoDataLoaderModule(
        data_config.DataConfig(**config["data"]),
        spectrum_tokenizer,
        peptide_tokenizer
    )
    pipeline(
        module,
        trainer_config,
        data_module,
        "train"
    )