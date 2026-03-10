import torch
import torch.nn.functional as F

from rocnovo.module.denovo import Denovo
from rocnovo.config.data import Precursor
from rocnovo.config.inference import InferenceConfig, SearchResult
from rocnovo.config.model import OutputConfig, Cache
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SOS, H2O, PROTON, SPECIAL_TOKENS
from rocnovo.inference.utils import MetaData

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

        finished[active_idx] |= (next_token == SPECIAL_TOKENS[SOS])
        finished_reverse[active_idx] |= (next_token_rev == SPECIAL_TOKENS[SOS])

        if finished.all() and finished_reverse.all():
            break

        still_active = ~(finished[active_idx] & finished_reverse[active_idx])
        if not still_active.all():
            active_idx = active_idx[still_active]
            next_token = next_token[still_active]
            next_token_rev = next_token_rev[still_active]
            mem_hidden_states = mem_hidden_states[still_active]
            mem_attention_mask = mem_attention_mask[still_active]
            cache = cache.filter_by_mask(still_active)
            cache_reverse = cache_reverse.filter_by_mask(still_active)

        curr_tokens = next_token.unsqueeze(-1)
        curr_tokens_rev = next_token_rev.unsqueeze(-1)
        output, output_reverse = model.step(
            curr_tokens,
            curr_tokens_rev,
            mem_hidden_states,
            mem_attention_mask,
            cache,
            cache_reverse,
            OutputConfig(
                return_cache=True
            )
        )
        cache = output.cache
        cache_reverse = output_reverse.cache
        
        step_log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)
        step_log_probs_rev = F.log_softmax(output_reverse.logits[:, -1, :], dim=-1)

    final_tokens = all_tokens[:, :current_step + 1]
    final_tokens_reverse = all_tokens_reverse[:, :current_step + 1]
    final_scores = torch.exp(scores / lengths)
    final_scores_reverse = torch.exp(scores_reverse / lengths_reverse)
    fwd_mass = metadata.idx2masses[final_tokens].sum(dim=1) + H2O
    rev_mass = metadata.idx2masses[final_tokens_reverse].sum(dim=1) + H2O
    fwd_mz = torch.where(
        precursor.charge > 0,
        fwd_mass / precursor.charge + PROTON,
        fwd_mass
    )
    rev_mz = torch.where(
        precursor.charge > 0,
        rev_mass / precursor.charge + PROTON,
        rev_mass
    )

    return SearchResult(
        fwd_mz.cpu().numpy(),
        rev_mz.cpu().numpy(),
        final_scores.cpu().numpy(),
        peptide_tokenizer.detokenize_by_array(final_tokens.cpu().numpy()),
        final_scores_reverse.cpu().numpy(),
        peptide_tokenizer.reverse_detokenize_by_array(final_tokens_reverse.cpu().numpy())
    )