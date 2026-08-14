from dataclasses import dataclass, fields, asdict, replace

import torch
import numpy as np

from rocnovo.common.logger import logger
from rocnovo.module.denovo import Denovo
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, ISOTOPE, SOS, SPECIAL_TOKENS
from rocnovo.config.inference import DenovoResult, EvalResult, ResolvedPrediction
from rocnovo.config.data import Precursor, Spectra
from rocnovo.config.model import OutputConfig, Cache
from rocnovo.metrics.abnovobench import aa_match_batch, aa_match_metrics

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

def _make_dummy_spectra(batch_size: int, num_peaks: int, device: torch.device):
    """Create deterministic dummy spectra for memory probing."""
    mz = torch.linspace(100.0, 2000.0, num_peaks, device=device)[None, :].expand(
        batch_size, -1
    )
    intensity = torch.rand(batch_size, num_peaks, device=device)
    intensity = intensity / intensity.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    mask = torch.ones(batch_size, num_peaks, dtype=torch.bool, device=device)
    precursor = Precursor(
        mass=torch.full((batch_size,), 2000.0, device=device),
        charge=torch.full((batch_size,), 2, dtype=torch.int32, device=device),
        mz=torch.full((batch_size,), 1000.0, device=device),
    )
    return Spectra(mz=mz, intensity=intensity, mask=mask, precursor=precursor)

def _measure_decode_peak_memory(
    model: Denovo,
    batch_size: int,
    num_peaks: int,
    max_length: int,
    num_beams: int,
    use_cross_cache: bool,
    use_self_cache: bool,
    gradscaler_enabled: bool,
    device: torch.device,
    max_decode_steps: int | None = None,
) -> int:
    """Run encode + prefill + decode steps and return peak allocated bytes.

    By default this simulates the full decoding length (``max_length`` steps),
    because the peak memory in autoregressive decoding occurs when the KV cache
    has grown to its maximum length. Pass ``max_decode_steps`` to override the
    number of decoding steps for faster but less accurate probes.
    """
    spectra = _make_dummy_spectra(batch_size, num_peaks, device)
    actual_beam_size = max(1, num_beams)
    decode_steps = max_length if max_decode_steps is None else max_decode_steps

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=gradscaler_enabled
    ):
        mem_hidden_states, mem_attention_mask = model.encode_spectrum(spectra)
        prompt_hidden_states = model.prefill(spectra)

        sos_token = SPECIAL_TOKENS[SOS]
        tokens = torch.full(
            (batch_size, 1), sos_token, dtype=torch.long, device=device
        )
        tokens_reverse = torch.full(
            (batch_size, 1), sos_token, dtype=torch.long, device=device
        )

        init_output, init_output_reverse = model.peptide_decoder(
            tokens,
            tokens_reverse,
            spectra.precursor,
            mem_hidden_states,
            mem_attention_mask,
            prompt_hidden_states,
            cache_return_config=OutputConfig(
                return_cross_cache=use_cross_cache,
                return_self_cache=use_self_cache,
            ),
        )

        init_cache = init_output.cache
        init_rev_cache = init_output_reverse.cache
        
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

        if not use_cross_cache:
            init_cache = init_cache.without_cross_cache()
            init_rev_cache = init_rev_cache.without_cross_cache()
        
        if not use_self_cache:
            init_cache = init_cache.without_self_cache()
            init_rev_cache = init_rev_cache.without_self_cache()

        beam_cache = init_cache.repeat_for_beam(actual_beam_size)
        beam_rev_cache = init_rev_cache.repeat_for_beam(actual_beam_size)
        beam_precursor = spectra.precursor.repeat_beamsize(actual_beam_size)

        if num_beams >= 1:
            step_tokens = torch.full(
                (batch_size * actual_beam_size, 1),
                sos_token,
                dtype=torch.long,
                device=device,
            )
            step_tokens_reverse = step_tokens.clone()
            step_mask = torch.repeat_interleave(
                mem_attention_mask, actual_beam_size, dim=0
            )
            if use_cross_cache:
                step_mem_hidden = None
            else:
                step_mem_hidden = torch.repeat_interleave(
                    mem_hidden_states, actual_beam_size, dim=0
                )
        else:
            step_tokens = tokens
            step_tokens_reverse = tokens_reverse
            step_mask = mem_attention_mask
            step_mem_hidden = mem_hidden_states

        # Simulate the full decoding length so that the peak memory reflects the
        # maximum KV cache size. Memory usage during generation is dominated by
        # the cache growing with each step, so a single-step probe severely
        # underestimates the true requirement.
        full_tokens = step_tokens
        full_tokens_reverse = step_tokens_reverse
        for step_idx in range(decode_steps):
            if use_self_cache:
                model_input_tokens = step_tokens
                model_input_tokens_reverse = step_tokens_reverse
            else:
                # Without self KV cache the decoder recomputes self attention
                # over the whole prefix, so the input grows by one each step.
                if step_idx == 0:
                    full_tokens = step_tokens
                    full_tokens_reverse = step_tokens_reverse
                else:
                    full_tokens = torch.cat([full_tokens, step_tokens], dim=1)
                    full_tokens_reverse = torch.cat(
                        [full_tokens_reverse, step_tokens_reverse], dim=1
                    )
                model_input_tokens = full_tokens
                model_input_tokens_reverse = full_tokens_reverse

            output, output_reverse = model.step(
                model_input_tokens,
                model_input_tokens_reverse,
                step_mem_hidden,
                step_mask,
                beam_cache,
                beam_rev_cache,
                cache_return_config=OutputConfig(
                    return_cross_cache=use_cross_cache,
                    return_self_cache=use_self_cache,
                ),
                precursor=beam_precursor,
            )
            beam_cache = output.cache
            beam_rev_cache = output_reverse.cache
            # For memory estimation the exact next token does not matter; using
            # argmax keeps the probe deterministic and avoids importing search
            # logic into this utility.
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_reverse = output_reverse.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            step_tokens = next_token
            step_tokens_reverse = next_token_reverse

    peak = torch.cuda.max_memory_allocated(device)
    # Explicitly free the large probe tensors so they don't affect the next probe.
    del spectra, mem_hidden_states, mem_attention_mask, prompt_hidden_states
    del init_output, init_output_reverse, beam_cache, beam_rev_cache
    torch.cuda.empty_cache()
    return peak

def _estimate_analytical_memory(
    model: Denovo,
    num_beams: int,
    vocab_size: int,
    max_length: int,
    num_peaks: int,
    use_cross_cache: bool,
    use_self_cache: bool,
    gradscaler_enabled: bool,
    batch_size: int,
) -> int:
    """Return the dimensionally estimated peak memory (bytes) for a given batch size."""
    n_byte = 2 if gradscaler_enabled else 4
    S = max(1, num_beams)
    P = num_peaks
    L = max_length
    B = batch_size

    spectrum_cfg = model.model_config["spectrum"]
    peptide_cfg = model.model_config["peptide"]
    clip_cfg = model.clip.model_config["spectrum"]

    H_s = spectrum_cfg["hidden_size"]
    H_p = peptide_cfg["hidden_size"]
    F_p = peptide_cfg["dim_feedforward"]
    n_layers_p = peptide_cfg["n_layers"]
    n_head_p = peptide_cfg["n_head"]
    d_head_p = H_p // n_head_p
    H_clip_s = clip_cfg["hidden_size"]
    V = peptide_cfg.get("n_vocab", vocab_size)

    # Persistent memory.
    persistent = 0
    persistent += 2 * B * (P + 1) * H_clip_s * n_byte
    persistent += 2 * B * H_clip_s * n_byte
    persistent += 2 * B * H_p * n_byte
    persistent += 2 * B * (P + 1) * H_s * n_byte
    if use_cross_cache:
        # Cross KV cache is stored for every decoder layer and for both
        # directions (forward + reverse). Each layer keeps K and V of shape
        # [B, n_head, P+1, d_head] = B * (P+1) * H_s elements.
        persistent += 4 * n_layers_p * B * (P + 1) * H_s * n_byte
    
    if use_self_cache:
        # Self KV cache for both directions.
        persistent += 4 * n_layers_p * B * S * L * H_p * n_byte

    # Activation peak.
    activation = 0
    activation = max(activation, 4 * B * S * L * F_p * n_byte)
    activation = max(activation, 2 * B * S * n_head_p * L * L * n_byte)
    activation = max(activation, 2 * B * S * n_head_p * L * (P + 1) * n_byte)
    if not use_cross_cache:
        activation = max(
            activation,
            4 * B * S * n_head_p * (P + 1) * d_head_p * n_byte,
        )

    # Beam search buffers.
    beam_buffer = 0
    if S > 1:
        per_direction = (
            8 * B * S * L
            + 4 * B * S * L * V
            + 4 * B * S
            + 8 * B * S * L * L
            + 4 * B * S * L
            + 4 * B * S
        )
        beam_buffer = 2 * per_direction

    return persistent + activation + beam_buffer

def estimate_analytical_batch_size(
    model: Denovo,
    num_beams: int,
    vocab_size: int,
    max_length: int,
    num_peaks: int = 300,
    gradscaler_enabled: bool = True,
    device: torch.device = torch.device("cuda:0"),
    use_cross_cache: bool = True,
    use_self_cache: bool = True,
    static_overhead_gb: float = 1.5,
    safety_margin: float = 0.10,
    validation_safety_factor: float = 0.85,
    oom_reduction_factor: float = 0.9,
    alignment: int = 8,
) -> int:
    if device.type != "cuda":
        return 64

    torch.cuda.empty_cache()
    free_vram, _ = torch.cuda.mem_get_info(device)

    static_overhead = static_overhead_gb * (1024 ** 3)
    target_vram = max(0, free_vram - static_overhead) * (1.0 - safety_margin)

    logger.debug(
        f"Estimating batch size: free_vram={free_vram/1e9:.2f}GB, "
        f"target_vram={target_vram/1e9:.2f}GB, num_beams={num_beams}, "
        f"use_cross_cache={use_cross_cache}, use_self_cache={use_self_cache}"
    )

    # Calibrate the analytical formula with two small probes (batch sizes 2 and 4).
    # The formula captures the per-batch scaling of persistent buffers and dominant
    # activations, but misses fixed costs and some super-linear effects. Fitting a
    # line through two measured points yields a more reliable slope than a single
    # point, where fixed overhead would dominate.
    calibration_batches = (2, 4)
    measured_peaks = []
    for calibration_batch in calibration_batches:
        try:
            peak = _measure_decode_peak_memory(
                model,
                batch_size=calibration_batch,
                num_peaks=num_peaks,
                max_length=max_length,
                num_beams=num_beams,
                use_cross_cache=use_cross_cache,
                use_self_cache=use_self_cache,
                gradscaler_enabled=gradscaler_enabled,
                device=device,
            )
        except RuntimeError as e:
            logger.warning(
                f"Calibration probe (batch={calibration_batch}) OOMed: {e}. "
                "Falling back to batch size 1."
            )
            return 1

        measured_peaks.append(peak)

    analytical_peaks = [
        _estimate_analytical_memory(
            model,
            num_beams,
            vocab_size,
            max_length,
            num_peaks,
            use_cross_cache,
            use_self_cache,
            gradscaler_enabled,
            b,
        )
        for b in calibration_batches
    ]

    # Fit measured_peak = fixed + slope * batch_size to the two probes.
    measured_slope = (
        measured_peaks[1] - measured_peaks[0]
    ) / (calibration_batches[1] - calibration_batches[0])
    analytical_slope = (
        analytical_peaks[1] - analytical_peaks[0]
    ) / (calibration_batches[1] - calibration_batches[0])
    # The analytical slope is usually too small because it misses per-step
    # activation growth. Use the measured slope directly; it already reflects the
    # true per-batch memory cost at small scale.
    slope_per_batch = max(measured_slope, analytical_slope)
    fixed_overhead = max(0, measured_peaks[0] - slope_per_batch * calibration_batches[0])

    logger.debug(
        f"Calibration: measured_slope={measured_slope/1e6:.3f}MB/batch, "
        f"analytical_slope={analytical_slope/1e6:.3f}MB/batch, "
        f"fixed_overhead={fixed_overhead/1e9:.3f}GB"
    )

    if slope_per_batch <= 0:
        logger.warning("Non-positive slope; using conservative default.")
        candidate = alignment
    else:
        candidate = int(
            (target_vram * validation_safety_factor - fixed_overhead) / slope_per_batch
        )
    
    candidate = max(1, (candidate // alignment) * alignment)
    logger.debug(f"Calibrated estimate before validation: {candidate}")

    # Validate with synthetic extreme data: 300 peaks and full max_length decode.
    # The probe must run the complete decoding length so that the KV cache reaches
    # its maximum size; this is the worst-case memory scenario.
    while candidate >= 1:
        try:
            peak = _measure_decode_peak_memory(
                model,
                batch_size=candidate,
                num_peaks=num_peaks,
                max_length=max_length,
                num_beams=num_beams,
                use_cross_cache=use_cross_cache,
                use_self_cache=use_self_cache,
                gradscaler_enabled=gradscaler_enabled,
                device=device,
            )
            logger.debug(
                f"Validated batch size {candidate}: peak={peak/1e9:.3f}GB"
            )
            return candidate
        
        except RuntimeError as e:
            logger.warning(
                f"Validation OOM for batch size {candidate}: {e}. "
                f"Reducing by {oom_reduction_factor:.2f} and retrying."
            )
            candidate = int(candidate * oom_reduction_factor)
            candidate = max(1, (candidate // alignment) * alignment)

    return 1

def _format_three_line_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a Markdown-style three-line table as a string."""
    col_widths = [max(len(str(rows[i][j])) for i in range(len(rows))) for j in range(len(headers))]
    for j, h in enumerate(headers):
        col_widths[j] = max(col_widths[j], len(h))

    def fmt(cells):
        return "| " + " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(cells)) + " |"

    sep = "|" + "|".join("-" * (col_widths[j] + 2) for j in range(len(headers))) + "|"
    lines = [fmt(headers), sep, fmt(rows[0])]
    return "\n".join(lines)

def _log_eval_summary(
    pred_peptides: list[str],
    gt_peptides: list[str],
    pred_scores: list[float],
    peptide_tokenizer: PTMPeptideTokenizer,
    species_name: str = "",
):
    """Compute evaluation metrics and print a three-line grid to the log."""
    ptm_list = ["C+57.021", "M+15.995", "N+0.984", "Q+0.984"]

    batch, n_pep, n_aa_true, n_aa_pred, n_ptm_true, n_ptm_pred, n_pep_pred_non_empty = aa_match_batch(
        gt_peptides,
        pred_peptides,
        peptide_tokenizer.masses,
        ptm_list,
        0.5,
        0.1,
        "best",
    )
    metrics = aa_match_metrics(
        batch,
        n_pep,
        n_aa_true,
        n_aa_pred,
        n_ptm_true,
        n_ptm_pred,
        n_pep_pred_non_empty,
        pred_scores,
    )
    full_accuracy = sum(p == g for p, g in zip(pred_peptides, gt_peptides)) / max(len(gt_peptides), 1)

    headers = [
        "aa_precision",
        "aa_recall",
        "curve_auc",
        "pep_precision",
        "pep_recall",
        "ptm_precision",
        "ptm_recall",
        "full_accuracy",
    ]
    values = [metrics[h] for h in headers[:-1]] + [full_accuracy]
    row = [f"{v:.6f}" for v in values]
    table = _format_three_line_table(headers, [row])
    logger.info(f"Evaluation summary for {species_name or 'dataset'}:\n{table}")
