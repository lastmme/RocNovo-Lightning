import copy
from dataclasses import dataclass, fields, asdict

import numpy as np
import torch

from rocnovo.common.io import normalize_path
from rocnovo.module.denovo import Denovo
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, ISOTOPE
from rocnovo.config.inference import DenovoResult, EvalResult, DenovoGlobalResult, EvalGlobalResult, ResolvedPrediction

@dataclass(frozen=True)
class SpecialTokens:
    sos: int
    pad: int

    def __iter__(self):
        return iter(asdict(self).items())

@dataclass(frozen=True)
class MetaData:
    idx2masses: torch.FloatTensor
    is_aa_neg_token: torch.BoolTensor
    is_n_term_token: torch.BoolTensor
    is_special_token: torch.BoolTensor
    isotope_offsets: torch.FloatTensor
    min_neg_mass: float

def _prepare_token_metadata(
    tokenizer: PTMPeptideTokenizer,
    device: torch.device,
    special_tokens: SpecialTokens,
    max_isotope: int=1,
):
    vocab_size = tokenizer.vocab_size
    # idx -> mass
    idx2masses = torch.zeros(vocab_size)
    for aa, mass in tokenizer.masses.items():
        idx2masses[tokenizer.vocab2idx[aa]] = mass
    
    # neg mass token
    is_aa_neg_token = torch.zeros(vocab_size, dtype=torch.bool)
    for aa, mass in tokenizer.masses.items():
        if mass < 0:
            is_aa_neg_token[tokenizer.vocab2idx[aa]] = True
    
    if is_aa_neg_token.any():
        min_neg_mass = idx2masses[is_aa_neg_token].min().item()
    else:
        min_neg_mass = 0.0
    
    # N-term token
    is_n_term_token = torch.zeros(vocab_size, dtype=torch.bool)
    for aa in tokenizer.masses:
        if aa.startswith(("+", "-")):
            is_n_term_token[tokenizer.vocab2idx[aa]] = True
    
    # sos/pad token
    is_special_token = torch.zeros(vocab_size, dtype=torch.bool)
    for _, t in special_tokens:
        is_special_token[t] = True
    
    # isotope offset mass
    isotope_offsets = torch.tensor([i * ISOTOPE for i in range(max_isotope + 1)])

    return MetaData(
        idx2masses.to(device),
        is_aa_neg_token.to(device),
        is_n_term_token.to(device),
        is_special_token.to(device),
        isotope_offsets.to(device),
        min_neg_mass
    )

def resolve_bidirectional_predictions(merged_batch: DenovoResult | EvalResult):
    fwd_scores = merged_batch.fwd_score
    rev_scores = merged_batch.rev_score
    
    fwd_peps = np.array(merged_batch.fwd_peptide, dtype=object)
    rev_peps = np.array(merged_batch.rev_peptide, dtype=object)

    fwd_peps = np.where(fwd_scores == -2.0, "", fwd_peps)
    rev_peps = np.where(rev_scores == -2.0, "", rev_peps)

    fwd_wins = fwd_scores >= rev_scores
    fwd_wins = np.where((fwd_peps == "") & (rev_peps != ""), False, fwd_wins)
    fwd_wins = np.where((rev_peps == "") & (fwd_peps != ""), True, fwd_wins)

    pred_mzs = np.where(fwd_wins, merged_batch.fwd_mz, merged_batch.rev_mz)
    pred_peptides = np.where(fwd_wins, fwd_peps, rev_peps).tolist()
    pred_scores = np.where(fwd_wins, fwd_scores, rev_scores)
    directions = np.where(fwd_wins, "fwd", "rev").tolist()
    
    return ResolvedPrediction(
        pred_mzs,
        pred_peptides,
        pred_scores,
        directions
    )

def merge_batches(batches: list[DenovoResult | EvalResult]):
    batch_type = type(batches[0])
    merged_args = {}
    for f in fields(batch_type):
        values = [getattr(b, f.name) for b in batches]
        if isinstance(values[0], np.ndarray):
            merged_args[f.name] = np.concatenate(values, axis=0)
        elif isinstance(values[0], list):
            merged_args[f.name] = [item for sublist in values for item in sublist]
        else:
            raise TypeError(f"Unsupported field type for merging: {type(values[0])}")
    
    return batch_type(**merged_args)

def post_process(batches: list[DenovoResult | EvalResult]):
    if not batches:
        raise ValueError("The batch list is empty.")
    
    merged_raw_batch = merge_batches(batches)
    resolved_preds = resolve_bidirectional_predictions(merged_raw_batch)

    if isinstance(merged_raw_batch, EvalResult):
        return EvalGlobalResult(
            resolved_preds.pred_mz,
            resolved_preds.pred_peptide,
            resolved_preds.pred_score,
            resolved_preds.direction,
            merged_raw_batch.fwd_score,
            merged_raw_batch.fwd_peptide,
            merged_raw_batch.rev_score,
            merged_raw_batch.rev_peptide,
            merged_raw_batch.precursor_mz,
            merged_raw_batch.charge,
            merged_raw_batch.gt_peptide
        )
    
    return DenovoGlobalResult(
        resolved_preds.pred_mz,
        resolved_preds.pred_peptide,
        resolved_preds.pred_score,
        resolved_preds.direction,
        merged_raw_batch.fwd_score,
        merged_raw_batch.fwd_peptide,
        merged_raw_batch.rev_score,
        merged_raw_batch.rev_peptide,
        merged_raw_batch.precursor_mz,
        merged_raw_batch.charge
    )

def _deep_merge_dicts(default_dict: dict, override_dict: dict) -> dict:
    merged = copy.deepcopy(default_dict)
    for k, v in override_dict.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    
    return merged

def parse_device(device_input: int | str):
    if isinstance(device_input, int):
        device_str = f"cuda:{device_input}"
    elif isinstance(device_input, str):
        device_str = device_input.strip().lower()
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
    else:
        raise TypeError(f"Unsupported device input type: {type(device_input)}. Expected int or str.")

    try:
        device = torch.device(device_str)
    except RuntimeError as e:
        raise ValueError(f"Invalid device string: '{device_str}'. PyTorch error: {e}")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device_str}', but CUDA is not available on this machine.")
        
        if device.index is not None:
            device_count = torch.cuda.device_count()
            if device.index >= device_count:
                raise ValueError(
                    f"Requested CUDA index {device.index} is out of bounds. "
                    f"Found only {device_count} CUDA device(s) on this machine."
                )

    return device

def estimate_analytical_batch_size(
    model: Denovo,
    num_beams: int,
    vocab_size: int,
    max_length: int,
    num_peaks: int=300,
    gradscaler_enabled: bool=True,
    device: torch.device = torch.device("cuda:0")
):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    n_byte = 2 if gradscaler_enabled else 4
    actual_beam_size = max(1, num_beams)
    free_vram, _ = torch.cuda.mem_get_info(device)
    
    static_overhead = 1.5 * (1024**3)
    usable_vram = max(0, free_vram - static_overhead)
    target_vram = usable_vram * 0.95 
    
    encoder_hidden_size = model.model_config["spectrum"]["hidden_size"]
    encoder_ffn_dim = model.model_config["spectrum"]["dim_feedforward"]
    encoder_heads = model.model_config["spectrum"]["n_head"]
    
    decoder_hidden_size = model.model_config["peptide"]["hidden_size"]
    decoder_num_layers = len(model.peptide_decoder.decoder.layers)
    
    mem_states_bytes = actual_beam_size * num_peaks * (encoder_hidden_size * n_byte)
    kvcache_bytes = 2 * actual_beam_size * decoder_num_layers * 2 * max_length * (decoder_hidden_size * n_byte)
    
    persistent_memory = mem_states_bytes + kvcache_bytes
    attn_matrix_bytes = encoder_heads * ((num_peaks + 1) ** 2) * 4
    ffn_bytes = (num_peaks + 1) * encoder_ffn_dim * n_byte
    
    single_encoder_peak = attn_matrix_bytes + ffn_bytes
    activation_peak = single_encoder_peak * 1.5

    bidirectional_beam_buffer = 0
    if num_beams >= 1:
        S = actual_beam_size
        L = max_length + 2
        V = vocab_size

        bytes_cur_tokens = S * L * 8             # [B, S, L] int64
        bytes_cur_scores = S * L * V * 4         # [B, S, L, V] float32
        bytes_cur_cumsum = S * 4                 # [B, S] float32

        bytes_his_tokens = S * L * L * 8         # [B, S, L, L] int64
        bytes_his_scores = S * L * 4             # [B, S, L] float32

        bytes_flags = S * 4 * 1                  # 4 * [B * S] bool

        single_dir_buffer_bytes = (
            bytes_cur_tokens + bytes_cur_scores + bytes_cur_cumsum + 
            bytes_his_tokens + bytes_his_scores + bytes_flags
        )
        
        bidirectional_beam_buffer = single_dir_buffer_bytes * 2

    total_bytes_per_batch = persistent_memory + activation_peak + bidirectional_beam_buffer
    memory_per_batch_with_margin = total_bytes_per_batch * 1.15
    max_batch_size = int(target_vram // memory_per_batch_with_margin)
    return max(1, max_batch_size)

def load_denovo_from_checkpoint(
    ckpt_path: str,
    inference_config_overrides: dict,
    device: torch.device
) -> "Denovo":
    ckpt_path = normalize_path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" in checkpoint:
        original_config = checkpoint["hyper_parameters"]
    else:
        raise KeyError("Could not find 'config' in the checkpoint hyper_parameters.")
    
    merged_config = _deep_merge_dicts(original_config, inference_config_overrides)
    model = Denovo.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        config=merged_config
    )
    
    return model.to(device).eval(), merged_config