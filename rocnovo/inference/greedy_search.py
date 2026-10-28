import torch
import torch.nn.functional as F

from rocnovo.module.denovo import Denovo
from rocnovo.config.data import Precursor
from rocnovo.config.inference import InferenceConfig, SearchResult
from rocnovo.config.model import OutputConfig, Cache
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SOS, PAD, H2O, PROTON, SPECIAL_TOKENS
from rocnovo.inference.utils import MetaData
from rocnovo.inference.beam_search import _format_token_logprobs


def _finalize_greedy(
    final_tokens: torch.LongTensor,
    final_tokens_reverse: torch.LongTensor,
    logprobs_view: torch.FloatTensor,
    logprobs_reverse_view: torch.FloatTensor,
    scores: torch.FloatTensor,
    scores_reverse: torch.FloatTensor,
    lengths: torch.FloatTensor,
    lengths_reverse: torch.FloatTensor,
    current_step: int,
    original_precursor: Precursor,
    metadata: MetaData,
    peptide_tokenizer: PTMPeptideTokenizer,
    inference_config: InferenceConfig
):
    final_scores = torch.exp(scores / lengths)
    final_scores_reverse = torch.exp(scores_reverse / lengths_reverse)
    fwd_mass = metadata.idx2masses[final_tokens].sum(dim=1) + H2O
    rev_mass = metadata.idx2masses[final_tokens_reverse].sum(dim=1) + H2O
    fwd_mz = torch.where(
        original_precursor.charge > 0,
        fwd_mass / original_precursor.charge + PROTON,
        fwd_mass
    )
    rev_mz = torch.where(
        original_precursor.charge > 0,
        rev_mass / original_precursor.charge + PROTON,
        rev_mass
    )

    # Mass-fit flags on GPU, same isotope-aware ppm test as the beam search QC
    charges = original_precursor.charge.to(torch.long)
    iso_shift = metadata.isotope_offsets / charges[:, None]
    target_mz = original_precursor.mz[:, None]

    def _isotopic_mass_fit(pred_mz: torch.FloatTensor):
        mass_errors = torch.abs(
            (pred_mz[:, None] - (target_mz - iso_shift)) / target_mz * 1e6
        )
        return (mass_errors < inference_config.mass_tolerance).any(dim=1)

    fwd_fit = _isotopic_mass_fit(fwd_mz)
    rev_fit = _isotopic_mass_fit(rev_mz)

    # Entry count per direction: first special token after position 0, or the
    # full span for beams that never emitted a stop
    def _token_lengths(tokens: torch.LongTensor):
        special = (tokens == SPECIAL_TOKENS[SOS]) | (tokens == SPECIAL_TOKENS[PAD])
        special[:, 0] = False
        has_stop = special.any(dim=-1)
        stop_step = torch.argmax(special.long(), dim=-1)
        return torch.where(has_stop, stop_step, current_step).clamp_min(1)

    fwd_lengths = _token_lengths(final_tokens)
    rev_lengths = _token_lengths(final_tokens_reverse)
    fwd_token_logprobs = _format_token_logprobs(
        logprobs_view.cpu().numpy(), fwd_lengths.cpu().numpy()
    )
    rev_token_logprobs = _format_token_logprobs(
        logprobs_reverse_view.cpu().numpy(), rev_lengths.cpu().numpy()
    )

    return SearchResult(
        fwd_mz.cpu().numpy(),
        rev_mz.cpu().numpy(),
        fwd_fit.cpu().numpy(),
        rev_fit.cpu().numpy(),
        final_scores.cpu().numpy(),
        peptide_tokenizer.detokenize_by_array(final_tokens.cpu().numpy()),
        final_scores_reverse.cpu().numpy(),
        peptide_tokenizer.reverse_detokenize_by_array(final_tokens_reverse.cpu().numpy()),
        fwd_token_logprobs,
        rev_token_logprobs
    )

def greedy_search(
    model: Denovo,
    precursor: Precursor,
    metadata: MetaData,
    mem_hidden_states: torch.FloatTensor,
    mem_attention_mask: torch.BoolTensor,
    init_log_probs: torch.FloatTensor,
    init_rev_log_probs: torch.FloatTensor,
    init_cache: Cache,
    init_rev_cache: Cache,
    peptide_tokenizer: PTMPeptideTokenizer,
    inference_config: InferenceConfig
):
    device = mem_hidden_states.device
    batch_size = mem_hidden_states.shape[0]
    max_length = inference_config.max_len + 1
    original_precursor = precursor

    all_tokens = torch.full(
        (batch_size, max_length + 1),
        SPECIAL_TOKENS[SOS],
        dtype=torch.long,
        device=device
    )
    all_tokens_reverse = torch.full(
        (batch_size, max_length + 1),
        SPECIAL_TOKENS[SOS],
        dtype=torch.long,
        device=device
    )
    all_logprobs = torch.zeros(batch_size, max_length + 1, device=device)
    all_logprobs_reverse = torch.zeros(batch_size, max_length + 1, device=device)
    
    scores = torch.zeros(batch_size, device=device, dtype=torch.float)
    scores_reverse = torch.zeros(batch_size, device=device, dtype=torch.float)
    lengths = torch.zeros(batch_size, device=device, dtype=torch.float)
    lengths_reverse = torch.zeros(batch_size, device=device, dtype=torch.float)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    finished_reverse = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    active_idx = torch.arange(batch_size, device=device)

    step_log_probs = init_log_probs
    step_log_probs_rev = init_rev_log_probs
    cache = init_cache
    cache_reverse = init_rev_cache
    step_needs_mem = not init_cache.has_cross_cache()
    if not step_needs_mem:
        mem_hidden_states = None

    current_step = 0
    for step_idx in range(1, max_length + 1):
        current_step = step_idx

        next_score, next_token = torch.max(step_log_probs, dim=-1)
        next_score_rev, next_token_rev = torch.max(step_log_probs_rev, dim=-1)

        is_fwd_active = ~finished[active_idx]
        is_rev_active = ~finished_reverse[active_idx]

        scores[active_idx] += next_score * is_fwd_active.float()
        scores_reverse[active_idx] += next_score_rev * is_rev_active.float()
        lengths[active_idx] += is_fwd_active.float()
        lengths_reverse[active_idx] += is_rev_active.float()

        all_tokens[active_idx[is_fwd_active], step_idx] = next_token[is_fwd_active]
        all_tokens_reverse[active_idx[is_rev_active], step_idx] = next_token_rev[is_rev_active]
        all_logprobs[active_idx[is_fwd_active], step_idx] = next_score[is_fwd_active]
        all_logprobs_reverse[active_idx[is_rev_active], step_idx] = next_score_rev[is_rev_active]

        finished[active_idx] |= (next_token == SPECIAL_TOKENS[SOS])
        finished_reverse[active_idx] |= (next_token_rev == SPECIAL_TOKENS[SOS])

        if finished.all() and finished_reverse.all():
            break

        still_active = ~(finished[active_idx] & finished_reverse[active_idx])
        if not still_active.all():
            active_idx = active_idx[still_active]
            next_token = next_token[still_active]
            next_token_rev = next_token_rev[still_active]
            if mem_hidden_states is not None:
                mem_hidden_states = mem_hidden_states[still_active]
            mem_attention_mask = mem_attention_mask[still_active]
            precursor = precursor.filter_by_mask(still_active)
            cache = cache.filter_by_mask(still_active)
            cache_reverse = cache_reverse.filter_by_mask(still_active)

        step_cache_config = OutputConfig(
            return_cross_cache=cache.has_cross_cache(),
            return_self_cache=cache.has_self_cache(),
        )
        if cache.has_self_cache():
            curr_tokens = next_token.unsqueeze(-1)
            curr_tokens_rev = next_token_rev.unsqueeze(-1)
        else:
            # Without self KV cache we must recompute self attention over the
            # full prefix generated so far.
            curr_tokens = all_tokens[active_idx, :current_step + 1]
            curr_tokens_rev = all_tokens_reverse[active_idx, :current_step + 1]
        
        output, output_reverse = model.step(
            curr_tokens,
            curr_tokens_rev,
            mem_hidden_states,
            mem_attention_mask,
            cache,
            cache_reverse,
            step_cache_config,
            precursor
        )
        cache = output.cache
        cache_reverse = output_reverse.cache
        
        step_log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)
        step_log_probs_rev = F.log_softmax(output_reverse.logits[:, -1, :], dim=-1)

    final_tokens = all_tokens[:, :current_step + 1]
    final_tokens_reverse = all_tokens_reverse[:, :current_step + 1]
    return _finalize_greedy(
        final_tokens,
        final_tokens_reverse,
        all_logprobs[:, :current_step + 1],
        all_logprobs_reverse[:, :current_step + 1],
        scores,
        scores_reverse,
        lengths,
        lengths_reverse,
        current_step,
        original_precursor,
        metadata,
        peptide_tokenizer,
        inference_config,
    )
