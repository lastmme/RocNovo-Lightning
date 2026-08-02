from typing import Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import rocnovo.config.data as data_config
import rocnovo.config.model as model_config
import rocnovo.config.tokenizer as tokenizer_config
from rocnovo.config.aug import AugmentationConfig
from rocnovo.config.train import Config, TrainerConfig
from rocnovo.tokenizer.spectrum import SpectrumTokenizer
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SPECIAL_TOKENS, PAD
from rocnovo.common.io import normalize_path
from rocnovo.common.logger import logger
from rocnovo.common.launcher import pipeline, BaseModule
from rocnovo.components.encoders import RoPESpectrumEncoder
from rocnovo.components.decoders import BiDirectRopeDecoder
from rocnovo.module.clip import Clip
from rocnovo.data.dataloaders import BiDirectDeNovoDataLoaderModule

class Denovo(BaseModule):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        self.clip = Clip.load_from_checkpoint(
            self.config["clip_checkpoint_path"],
            map_location="cpu"
        )
        self.clip.freeze()
        
        self.clip_spectrum_adapter = nn.Sequential(
            nn.Linear(
                self.clip.model_config["spectrum"]["hidden_size"],
                self.model_config["peptide"]["dim_feedforward"]
            ),
            nn.ReLU(),
            nn.Dropout(self.model_config["spectrum"]["dropout"]),
            nn.Linear(
                self.model_config["peptide"]["dim_feedforward"],
                self.model_config["peptide"]["hidden_size"]
            )
        )
        self.spectrum_encoder = RoPESpectrumEncoder(**self.model_config["spectrum"])
        self.peptide_decoder = BiDirectRopeDecoder(
            **self.model_config["peptide"],
            mem_hidden_size=self.model_config["spectrum"]["hidden_size"]
        )
    
    def encode_spectrum(self, spectra: data_config.Spectra):
        mem_hidden_states, mem_attention_mask = self.spectrum_encoder(spectra)
        return mem_hidden_states, mem_attention_mask

    def prefill(self, spectra: data_config.Spectra):
        prompt_hidden_states = self.clip.repr_spectrum(spectra)
        prompt_hidden_states = self.clip_spectrum_adapter(prompt_hidden_states)
        return prompt_hidden_states

    def step(
        self,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        mem_hidden_states: Optional[torch.FloatTensor]=None,
        mem_attention_mask: Optional[torch.BoolTensor]=None,
        cache: Optional[model_config.Cache]=None,
        cache_reverse: Optional[model_config.Cache]=None,
        cache_return_config: model_config.OutputConfig=model_config.default_output_config,
        precursor: Optional[data_config.Precursor]=None,
    ):
        output: model_config.DecoderOutput
        output_reverse: model_config.DecoderOutput

        if cache is not None and not cache.has_self_cache():
            step_tokens = tokens
        else:
            step_tokens = tokens[:, [-1]]
        
        if cache_reverse is not None and not cache_reverse.has_self_cache():
            step_tokens_reverse = tokens_reverse
        else:
            step_tokens_reverse = tokens_reverse[:, [-1]]

        output, output_reverse = self.peptide_decoder.decode_with_cache(
            step_tokens,
            step_tokens_reverse,
            mem_hidden_states,
            mem_attention_mask,
            cache,
            cache_reverse,
            cache_return_config,
            precursor
        )
        return output, output_reverse

    def ce_loss(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        V = logits.shape[-1]
        logits_flatten = logits.reshape(-1, V)
        labels_flatten = labels.reshape(-1)
        return F.cross_entropy(
            logits_flatten,
            labels_flatten,
            ignore_index=SPECIAL_TOKENS[PAD],
            reduction="mean",
            label_smoothing=self.config["loss"]["label_smoothing"]
        )

    def full_peptide_match_accuracy(self, logits: torch.FloatTensor, labels: torch.LongTensor, mask: torch.BoolTensor):
        pred = torch.argmax(logits, dim=-1)
        matches = ((pred == labels) | ~mask).all(dim=-1)
        return torch.mean(matches.float())

    def forward(
        self,
        spectra: data_config.Spectra,
        peptide: data_config.Peptide,
        peptide_reverse: data_config.Peptide
    ):
        mem_hidden_states, mem_attention_mask = self.encode_spectrum(spectra)
        prompt_hidden_states = self.prefill(spectra)
        output: model_config.DecoderOutput
        output_reverse: model_config.DecoderOutput
        output, output_reverse = self.peptide_decoder(
            peptide.tokens[:, :-1],
            peptide_reverse.tokens[:, :-1],
            spectra.precursor,
            mem_hidden_states,
            mem_attention_mask,
            prompt_hidden_states
        )
        loss = 0.5 * self.ce_loss(
            output.logits,
            peptide.tokens[:, 1:]
        ) + 0.5 * self.ce_loss(
            output_reverse.logits,
            peptide_reverse.tokens[:, 1:]
        )

        accuracy = self.full_peptide_match_accuracy(
            output.logits,
            peptide.tokens[:, 1:],
            peptide.mask[:, 1:]
        )
        accuracy_reverse = self.full_peptide_match_accuracy(
            output_reverse.logits,
            peptide_reverse.tokens[:, 1:],
            peptide_reverse.mask[:, 1:]
        )
        return loss, accuracy, accuracy_reverse

    def common_step(self, batch: data_config.BidirectTrainBatch, batch_idx: int, train: bool=True):
        loss, accuracy, accuracy_reverse = self.forward(batch.spectra, batch.peptide, batch.peptide_reverse)
        kwargs = {
            "on_epoch": True,
            "on_step": True if train else False,
            "sync_dist": True
        }
        prefix = "train_" if train else "val_"
        self.log(f"{prefix}loss", loss, prog_bar=train, **kwargs)
        self.log(f"{prefix}accuracy", accuracy, **kwargs)
        self.log(f"{prefix}accuracy_reverse", accuracy_reverse, **kwargs)
        self.log(f"{prefix}{self.trainer_config.evaluate_metric_name}", (accuracy + accuracy_reverse) / 2, **kwargs)
        return loss

def train(config_path: str):
    logger.debug(f"Start to train denovo model with config: {config_path}")
    config_path = normalize_path(config_path)
    config = Config(config_path)
    trainer_config = TrainerConfig(**config["trainer"])
    checkpoint_path = trainer_config.checkpoint_path
    mode = trainer_config.mode
    if checkpoint_path is not None:
        logger.debug(f"Load checkpoint from: {trainer_config.checkpoint_path}, mode: {trainer_config.mode}")
        module = Denovo.load_from_checkpoint(
            trainer_config.checkpoint_path,
            map_location="cpu"
        )
        if trainer_config.mode == "full_state":
            trainer_config = module.trainer_config
            if trainer_config.checkpoint_path is None:
                trainer_config.checkpoint_path = checkpoint_path
                trainer_config.mode = mode
    else:
        module = Denovo(config.dict())
        logger.debug(f"Start to train denovo model from scratch with config: {config}")
    
    spectrum_tokenizer_config = tokenizer_config.SpectrumTokenizerConfig(**config["tokenizer"]["spectrum"])
    peptide_tokenizer_config = tokenizer_config.PTMTokenizerConfig(**config["tokenizer"]["peptide"])
    
    aug_config = AugmentationConfig(**config["aug"])
    spectrum_tokenizer = SpectrumTokenizer(**asdict(spectrum_tokenizer_config))
    if aug_config.enabled:
        spectrum_tokenizer.set_aug_config(aug_config)
    
    peptide_tokenizer = PTMPeptideTokenizer(**asdict(peptide_tokenizer_config))
    data_module = BiDirectDeNovoDataLoaderModule(
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