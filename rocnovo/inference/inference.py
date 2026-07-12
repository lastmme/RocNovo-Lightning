import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path
from typing import Literal
from dataclasses import asdict, replace

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
from rocnovo.config.inference import InferenceConfig, DenovoResult, EvalResult
from rocnovo.inference.beam_search import beam_search
from rocnovo.inference.greedy_search import greedy_search
from rocnovo.inference.utils import (
    post_process,
    _prepare_token_metadata,
    SpecialTokens,
    parse_device,
    estimate_analytical_batch_size
)
from rocnovo.module.denovo import Denovo
from rocnovo.common.logger import logger
from rocnovo.tokenizer.spectrum import SpectrumTokenizer
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SOS, PAD, SPECIAL_TOKENS
from rocnovo.data.dataloaders import BiDirectDeNovoDataLoaderModule, SpectraDataLoaderModule

@torch.no_grad()
def generation(
    model: Denovo,
    mode: Literal["denovo", "eval"],
    data_module: SpectraDataLoaderModule | BiDirectDeNovoDataLoaderModule,
    peptide_tokenizer: PTMPeptideTokenizer,
    inference_config: InferenceConfig,
    device: torch.device="cuda:0",
    progressbar_desc: str="Prediction"
):    
    results: list[DenovoResult | EvalResult] = []
    metadata = _prepare_token_metadata(
        peptide_tokenizer,
        device,
        SpecialTokens(
            SPECIAL_TOKENS[SOS],
            SPECIAL_TOKENS[PAD]
        )
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=inference_config.gradscaling_enabled):        
        for _, batch in enumerate(tqdm(data_module.test_dataloader(), desc=progressbar_desc, dynamic_ncols=True)):
            batch: BidirectTrainBatch | InferenceBatch
            batch = batch.to(device)
            batch_size = batch.spectra.mz.shape[0]
            
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
                batch.spectra.precursor.charge.cpu().numpy()
            )
            
            if mode == "eval":
                item = EvalResult(
                    **asdict(item),
                    gt_peptide=peptide_tokenizer.detokenize_by_array(batch.peptide.tokens.cpu().numpy())
                )
            
            results.append(item)
    
    if not results:
        return []
    
    return post_process(results)

def predict(config: str | Path | dict):
    if isinstance(config, (str, Path)):
        config_path = normalize_path(config)
        config = load_yaml(config_path)
    
    mode = config["mode"]
    if mode != "denovo" and mode != "eval":
        raise ValueError(f"mode must be either 'denovo' or 'eval', but got {mode}")

    logger.debug(f"Start to predict {config['task_name']} with mode: {mode}")
    device = parse_device(config["device"])
    model, config = load_denovo_from_checkpoint(
        config["checkpoint_path"],
        config,
        device
    )
    inference_config = InferenceConfig(**config["prediction"])
    logger.debug(f"spectrum tokenizer config: {config['tokenizer']['spectrum']}")
    logger.debug(f"peptide tokenizer config: {config['tokenizer']['peptide']}")
    spectrum_tokenizer_config = tokenizer_config.SpectrumTokenizerConfig(**config["tokenizer"]["spectrum"])
    peptide_tokenizer_config = tokenizer_config.PTMTokenizerConfig(**config["tokenizer"]["peptide"])

    spectrum_tokenizer = SpectrumTokenizer(**asdict(spectrum_tokenizer_config))
    spectrum_tokenizer.disable_aug()
    
    peptide_tokenizer = PTMPeptideTokenizer(**asdict(peptide_tokenizer_config))
    estimated_batch_size = estimate_analytical_batch_size(
        model,
        inference_config.num_beams,
        peptide_tokenizer.vocab_size,
        inference_config.max_len + 1,
        config["tokenizer"]["spectrum"]["n_top_peaks"],
        inference_config.gradscaling_enabled,
        device,
        inference_config.use_cross_cache,
        inference_config.use_self_cache,
    )
    logger.debug(f"num_beams >= 1 means number of beams in beam search, 0 means greedy search")
    logger.debug(f"Estimated prediction batch size: {estimated_batch_size} with num_beams: {inference_config.num_beams}")

    data_config = DataConfig(**config["data"])
    data_config = replace(
        data_config,
        test_batch_size=estimated_batch_size
    )
    if mode == "eval":
        data_module = BiDirectDeNovoDataLoaderModule(
            data_config,
            spectrum_tokenizer,
            peptide_tokenizer
        )
    else:
        data_module = SpectraDataLoaderModule(
            data_config,
            spectrum_tokenizer
        )
    
    info = f"greedy search" if inference_config.num_beams == 0 else f"beam search beam size: {inference_config.num_beams}" 
    logger.info(f"Start to predict {config['task_name']} {info}")
    pl.seed_everything(0)
    start = time.time()
    result = generation(
        model,
        mode,
        data_module,
        peptide_tokenizer,
        inference_config,
        parse_device(config["device"]),
        config["task_name"]
    )
    end = time.time()
    logger.info(f"Task End! It consumes {end - start:.3f} seconds")
    if not result:
        print("No result found")
        return

    output_dir = normalize_path(config["output_dir"])
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    result.to_csv(output_dir.joinpath(f"{config['task_name']}.csv"), index=False)