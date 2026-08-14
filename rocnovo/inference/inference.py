import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
from pathlib import Path
from typing import Literal
from dataclasses import asdict, replace, fields

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rocnovo.common.io import load_yaml, normalize_path
from rocnovo.common.logger import logger
from rocnovo.config.model import OutputConfig, Cache
from rocnovo.config.model import OutputConfig
import rocnovo.config.tokenizer as tokenizer_config
from rocnovo.common.utils import load_denovo_from_checkpoint
from rocnovo.config.data import DataConfig, BidirectTrainBatch, InferenceBatch
from rocnovo.config.inference import InferenceConfig, DenovoResult, EvalResult, ExportMixin
from rocnovo.inference.beam_search import beam_search
from rocnovo.inference.greedy_search import greedy_search
from rocnovo.inference.utils import (
    _prepare_token_metadata,
    SpecialTokens,
    parse_device,
    estimate_analytical_batch_size,
    _log_eval_summary,
    resolve_bidirectional_predictions,
)
from rocnovo.module.denovo import Denovo
from rocnovo.common.logger import logger
from rocnovo.tokenizer.spectrum import SpectrumTokenizer
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SOS, PAD, SPECIAL_TOKENS
from rocnovo.data.dataloaders import BiDirectDeNovoDataLoaderModule, SpectraDataLoaderModule

@torch.no_grad()
def generation_streaming(
    model: Denovo,
    mode: Literal["denovo", "eval"],
    data_module: SpectraDataLoaderModule | BiDirectDeNovoDataLoaderModule,
    peptide_tokenizer: PTMPeptideTokenizer,
    inference_config: InferenceConfig,
    device: torch.device,
    output_csv: Path,
    progressbar_desc: str = "Prediction",
    species_name: str = "",
):
    model.eval()
    pred_peptides: list[str] = []
    gt_peptides: list[str] = []
    pred_scores: list[float] = []
    first_batch = True

    metadata = _prepare_token_metadata(
        peptide_tokenizer,
        device,
        SpecialTokens(
            SPECIAL_TOKENS[SOS],
            SPECIAL_TOKENS[PAD]
        )
    )

    autocast_enabled = inference_config.gradscaling_enabled and device.type == "cuda"
    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled):
        for _, batch in enumerate(tqdm(data_module.test_dataloader(), desc=progressbar_desc, dynamic_ncols=True)):
            batch: BidirectTrainBatch | InferenceBatch
            batch = batch.to(device)
            batch_size = batch.spectra.mz.shape[0]
            scan_ids = [""] * batch_size if batch.scan_id is None else batch.scan_id

            mem_hidden_states, mem_attention_mask = model.encode_spectrum(batch.spectra)
            prompt_hidden_states = model.prefill(batch.spectra)
            
            tokens = torch.full(
                (batch_size, 1),
                SPECIAL_TOKENS[SOS],
                dtype=torch.long,
                device=device
            )
            tokens_reverse = torch.full(
                (batch_size, 1),
                SPECIAL_TOKENS[SOS],
                dtype=torch.long,
                device=device
            )
            
            init_output, init_output_reverse = model.peptide_decoder(
                tokens,
                tokens_reverse,
                batch.spectra.precursor,
                mem_hidden_states,
                mem_attention_mask,
                prompt_hidden_states,
                cache_return_config=OutputConfig(
                    return_cross_cache=inference_config.use_cross_cache,
                    return_self_cache=inference_config.use_self_cache,
                )
            )
        
            init_log_probs = F.log_softmax(init_output.logits[:, -1, :], dim=-1)
            init_rev_log_probs = F.log_softmax(init_output_reverse.logits[:, -1, :], dim=-1)
            
            init_cache = init_output.cache
            init_rev_cache = init_output_reverse.cache
            # If no cache was requested from the prefill call, create a minimal
            # cache that only tracks the current sequence length for position
            # encoding in subsequent decoding steps.
            if init_cache is None:
                init_cache = Cache(
                    past_length=tokens.size(1),
                    prompt_hidden_states=prompt_hidden_states,
                )
            elif prompt_hidden_states is not None:
                init_cache = replace(
                    init_cache,
                    prompt_hidden_states=prompt_hidden_states,
                )

            if init_rev_cache is None:
                init_rev_cache = Cache(
                    past_length=tokens.size(1),
                    prompt_hidden_states=prompt_hidden_states,
                )
            elif prompt_hidden_states is not None:
                init_rev_cache = replace(
                    init_rev_cache,
                    prompt_hidden_states=prompt_hidden_states,
                )

            if not inference_config.use_cross_cache:
                init_cache = init_cache.without_cross_cache()
                init_rev_cache = init_rev_cache.without_cross_cache()

            if not inference_config.use_self_cache:
                init_cache = init_cache.without_self_cache()
                init_rev_cache = init_rev_cache.without_self_cache()

            if inference_config.num_beams == 0:
                searched_result = greedy_search(
                    model,
                    batch.spectra.precursor,
                    metadata,
                    mem_hidden_states,
                    mem_attention_mask,
                    init_log_probs,
                    init_rev_log_probs,
                    init_cache,
                    init_rev_cache,
                    peptide_tokenizer,
                    inference_config
                )
            else:
                searched_result = beam_search(
                    model,
                    batch.spectra.precursor,
                    mem_hidden_states,
                    mem_attention_mask,
                    init_log_probs,
                    init_rev_log_probs,
                    init_cache,
                    init_rev_cache,
                    peptide_tokenizer,
                    inference_config
                )

            item = DenovoResult(
                searched_result.fwd_mz,
                searched_result.rev_mz,
                searched_result.fwd_score,
                searched_result.fwd_peptide,
                searched_result.rev_score,
                searched_result.rev_peptide,
                batch.spectra.precursor.mz.cpu().numpy(),
                batch.spectra.precursor.charge.cpu().numpy(),
                scan_ids,
            )
            
            if mode == "eval":
                gt_peps = peptide_tokenizer.detokenize_by_array(batch.peptide.tokens.cpu().numpy())
                item = EvalResult(
                    **asdict(item),
                    gt_peptide=gt_peps
                )
                gt_peptides.extend(gt_peps)

            resolved = resolve_bidirectional_predictions(item)

            ExportMixin.stream_batch_to_csv(
                output_csv,
                item,
                resolved,
                mode,
                write_header=first_batch,
            )
            first_batch = False

            pred_peptides.extend(resolved.pred_peptide)
            pred_scores.extend(resolved.pred_score.tolist())

    if mode == "eval" and pred_peptides:
        _log_eval_summary(
            pred_peptides,
            gt_peptides,
            pred_scores,
            peptide_tokenizer,
            species_name
        )

class DenovoPredictor:
    def __init__(self, config: dict, device: torch.device | None = None):
        self.config = config
        self.mode = config["mode"]
        if self.mode != "denovo" and self.mode != "eval":
            raise ValueError(f"mode must be either 'denovo' or 'eval', but got {self.mode}")

        if device is None:
            device = parse_device(config["device"])
        
        self.device = device

        logger.debug(f"Start to load predictor for {config['task_name']} with mode: {self.mode}")
        self.model, self.config = load_denovo_from_checkpoint(
            config["checkpoint_path"],
            config,
            device
        )
        self.inference_config = InferenceConfig(**self.config["prediction"])
        logger.debug(f"spectrum tokenizer config: {self.config['tokenizer']['spectrum']}")
        logger.debug(f"peptide tokenizer config: {self.config['tokenizer']['peptide']}")
        spectrum_tokenizer_config = tokenizer_config.SpectrumTokenizerConfig(**self.config["tokenizer"]["spectrum"])
        peptide_tokenizer_config = tokenizer_config.PTMTokenizerConfig(**self.config["tokenizer"]["peptide"])

        self.spectrum_tokenizer = SpectrumTokenizer(**asdict(spectrum_tokenizer_config))
        self.spectrum_tokenizer.disable_aug()
        self.peptide_tokenizer = PTMPeptideTokenizer(**asdict(peptide_tokenizer_config))
        self.model.eval()

    def _build_data_module(self, hdf5_path: Path):
        estimated_batch_size = estimate_analytical_batch_size(
            self.model,
            self.inference_config.num_beams,
            self.peptide_tokenizer.vocab_size,
            self.inference_config.max_len + 1,
            self.config["tokenizer"]["spectrum"]["n_top_peaks"],
            self.inference_config.gradscaling_enabled,
            self.device,
            self.inference_config.use_cross_cache,
            self.inference_config.use_self_cache,
        )
        logger.debug(f"num_beams >= 1 means number of beams in beam search, 0 means greedy search")
        logger.debug(f"Estimated prediction batch size: {estimated_batch_size} with num_beams: {self.inference_config.num_beams}")

        data_config = DataConfig(**self.config["data"])
        data_config = replace(
            data_config,
            test_path=str(hdf5_path),
            test_batch_size=estimated_batch_size
        )
        if self.mode == "eval":
            return BiDirectDeNovoDataLoaderModule(
                data_config,
                self.spectrum_tokenizer,
                self.peptide_tokenizer
            )
        else:
            return SpectraDataLoaderModule(
                data_config,
                self.spectrum_tokenizer
            )

    def predict_file(self, hdf5_path: str | Path, output_csv: str | Path, task_name: str | None = None):
        hdf5_path = normalize_path(hdf5_path)
        output_csv = normalize_path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if output_csv.exists():
            output_csv.unlink()

        desc = task_name or hdf5_path.stem
        info = f"greedy search" if self.inference_config.num_beams == 0 else f"beam search beam size: {self.inference_config.num_beams}"
        logger.info(f"Start to predict {desc} {info}")
        pl.seed_everything(0)
        data_module = self._build_data_module(hdf5_path)
        start = time.time()
        generation_streaming(
            self.model,
            self.mode,
            data_module,
            self.peptide_tokenizer,
            self.inference_config,
            self.device,
            output_csv,
            progressbar_desc=desc,
            species_name=desc,
        )
        end = time.time()
        logger.info(f"Task {desc} End! It consumes {end - start:.3f} seconds")
        return output_csv

def predict(config: str | Path | dict):
    if isinstance(config, (str, Path)):
        config = load_yaml(normalize_path(config))

    mode = config["mode"]
    if mode != "denovo" and mode != "eval":
        raise ValueError(f"mode must be either 'denovo' or 'eval', but got {mode}")

    device = parse_device(config["device"])
    predictor = DenovoPredictor(config, device)

    output_dir = normalize_path(predictor.config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir.joinpath(f"{predictor.config['task_name']}.csv")
    hdf5_path = predictor.config["data"]["test_path"]

    predictor.predict_file(
        hdf5_path,
        output_csv,
        predictor.config["task_name"]
    )