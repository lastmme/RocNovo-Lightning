from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
import einops

from rocnovo.config.model import Cache
from rocnovo.module.denovo import Denovo
from rocnovo.config.data import Precursor
from rocnovo.config.model import OutputConfig
from rocnovo.config.inference import InferenceConfig, SearchResult
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SOS, PAD, H2O, PROTON
from rocnovo.inference.utils import  _prepare_token_metadata, SpecialTokens

"""
    B: Batch Size
    S: Beam Size
    L: Max Length
    V: Vocab Size
    T: Current Step
"""

@dataclass(frozen=True)
class Size:
    batch_size: int
    beam_size: int
    min_len: int
    max_len: int
    vocab_size: int
    mass_tolerance: float

@dataclass(frozen=True)
class MetaData:
    idx2masses: torch.FloatTensor
    is_aa_neg_token: torch.BoolTensor
    is_n_term_token: torch.BoolTensor
    is_special_token: torch.BoolTensor
    isotope_offsets: torch.FloatTensor
    min_neg_mass: float

@dataclass
class Current:
    tokens: torch.LongTensor  # [B, S, L]
    scores: torch.FloatTensor # [B, S, L, V]
    cumsum_scores: torch.FloatTensor # [B, S]

@dataclass
class History:
    tokens: torch.LongTensor  # [B, S, L, L]
    scores: torch.FloatTensor # [B, S, L]

@dataclass
class Flag:
    active: torch.BoolTensor # [B * S, ]
    finished: torch.BoolTensor # [B * S, ]
    discarded: torch.BoolTensor # [B * S, ]
    fits: torch.BoolTensor # [B * S, ]

    def zero_(self):
        self.finished.zero_()
        self.discarded.zero_()
        self.fits.zero_()
    
    def update(self):
        self.finished |= self.discarded
        self.active = ~self.finished

@dataclass(frozen=True)
class BestResult:
    tokens: torch.LongTensor  # [B, S, L]
    scores: torch.FloatTensor # [B, S, L]

@dataclass(frozen=True)
class BirdirectBestResult:
    fwd: BestResult
    rev: BestResult

@dataclass(frozen=True)
class NextInput:
    fwd_tokens: torch.LongTensor # [B, 1]
    rev_tokens: torch.LongTensor # [B, 1]
    active: torch.BoolTensor # [B,]
    fwd_cache: Cache
    rev_cache: Cache

def _prepare_current_info(
    B: int, S: int, L: int, V: int,
    device: torch.device | str,
    pad_token: int,
    sos_token: int,
):
    tokens = torch.full(
        (B, S, L),
        fill_value=pad_token,
        dtype=torch.long,
        device=device
    )
    tokens[:, :, 0] = sos_token

    scores = torch.full(
        (B, S, L, V),
        fill_value=float("-inf"),
        device=device
    )
    scores[:, :, 0, :] = 0.0

    cumsum_scores = torch.zeros(
        (B, S),
        device=device,
    )

    return Current(
        tokens,
        scores,
        cumsum_scores
    )

def _prepare_history_info(
    B: int, S: int, L: int,
    device: torch.device | str,
    pad_token: int
):
    tokens = torch.full(
        (B, S, L, L),
        fill_value=pad_token,
        dtype=torch.long,
        device=device
    )
    scores = torch.full(
        (B, S, L),
        fill_value=-2.0,
        device=device
    )

    return History(
        tokens,
        scores,
    )

def _prepare_beamflag(
    B: int, S: int,
    device: torch.device | str
):
    return Flag(
        torch.ones(B * S, dtype=torch.bool, device=device),
        torch.zeros(B * S, dtype=torch.bool, device=device),
        torch.zeros(B * S, dtype=torch.bool, device=device),
        torch.zeros(B * S, dtype=torch.bool, device=device),
    )

class BeamStat:
    def __init__(
        self,
        tokenizer: PTMPeptideTokenizer,
        precursors: Precursor,
        config: InferenceConfig,
        is_reverse: bool=False
    ):        
        batch_size = precursors.charge.shape[0]
        device = precursors.charge.device
        
        self.size = Size(
            batch_size,
            config.num_beams,
            config.min_len,
            config.max_len,
            tokenizer.vocab_size,
            config.mass_tolerance
        )
        self.flag = _prepare_beamflag(
            batch_size,
            config.num_beams,
            device
        )
        special_tokens = SpecialTokens(
            tokenizer.vocab2idx[SOS],
            tokenizer.vocab2idx[PAD]
        )
        self.metadata = _prepare_token_metadata(
            tokenizer,
            device,
            special_tokens,
            config.max_isotope
        )
        self.current = _prepare_current_info(
            batch_size,
            config.num_beams,
            config.max_len,
            tokenizer.vocab_size,
            device,
            special_tokens.pad,
            special_tokens.sos,
        )
        self.history = _prepare_history_info(
            batch_size,
            config.num_beams,
            config.max_len,
            device,
            special_tokens.pad
        )
        self.special_tokens = special_tokens
        self.device = device

        self.precursors = precursors.repeat_beamsize(config.num_beams)
        self.tokenizer = tokenizer
        self.config = config
        self.is_reverse = self.tokenizer.reverse if not is_reverse else (not self.tokenizer.reverse)
        self.current_step = 1
        self.cache = None

    @property
    def active(self):
        return self.flag.active.clone()

    def _apply_quality_control(self):
        BS = self.size.batch_size * self.size.beam_size
        tokens = einops.rearrange(
            self.current.tokens,
            "B S L -> (B S) L"
        )
        current_step_tokens = tokens[:, self.current_step]
        self.flag.zero_()

        self.flag.finished = (current_step_tokens == self.special_tokens.sos)
        self.flag.discarded = (current_step_tokens == self.special_tokens.pad)

        if self.current_step > 2:
            # Check multiple N-term and internal N-term
            if self.is_reverse:
                col = torch.full(
                    (BS,),
                    self.current_step,
                    device=self.device
                )
                col[self.flag.finished] = self.current_step - 1
                row = torch.arange(BS, device=self.device)

                is_n_term_last = self.metadata.is_n_term_token[
                    tokens[row, col]
                ]
                is_n_term_last_prev = self.metadata.is_n_term_token[
                    tokens[row, col - 1]
                ]
                multiple_n_term = is_n_term_last & is_n_term_last_prev
                
                mask = torch.arange(self.size.max_len, device=self.device)[None, :] < col[:, None]
                internal_n_term = self.metadata.is_n_term_token[tokens] & mask
                internal_n_term = internal_n_term.any(dim=1)

                self.flag.discarded |= (multiple_n_term | internal_n_term)
            else:
                mask = torch.arange(self.size.max_len, device=self.device)[None, :] > 1
                internal_n_term = self.metadata.is_n_term_token[tokens] & mask
                internal_n_term = internal_n_term.any(dim=1)
                self.flag.discarded |= internal_n_term

        if self.current_step < 2:
            return
        
        # [B * S, L]
        pred_tokens = tokens[:, 1:self.current_step + 1]
        peptide_length = self.current_step - self.flag.finished.to(torch.long)
        # [B * S]
        peptide_cur_masses = self.metadata.idx2masses[pred_tokens].sum(dim=1) + H2O
        precursor_charges = self.precursors.charge.to(torch.long)
        target_mz = self.precursors.mz
        pred_mz = torch.where(
            precursor_charges > 0,
            peptide_cur_masses / precursor_charges + PROTON,
            peptide_cur_masses
        )
        # [B * S, max_isotope]
        iso_shift = self.metadata.isotope_offsets / precursor_charges[:, None]
        mass_errors = torch.abs(
            (
                pred_mz[:, None] - (target_mz[:, None] - iso_shift)
            )
            / target_mz[:, None] * 1e6
        )
        fits_mass = (mass_errors < self.config.mass_tolerance).any(dim=1)

        exceeds_mass = torch.zeros_like(self.flag.finished, dtype=torch.bool)
        if not self.flag.finished.all():
            neg_masses = peptide_cur_masses + self.metadata.min_neg_mass
            neg_mz = torch.where(
                precursor_charges > 0,
                neg_masses / precursor_charges + PROTON,
                neg_masses
            )
            neg_errors = (
                (neg_mz[:, None] - (target_mz[:, None] - iso_shift))
                / target_mz[:, None]
                * 1e6
            )
            exceeds_mass |= ~self.flag.finished & (neg_errors > self.config.mass_tolerance).all(dim=1)

        self.flag.fits = self.flag.finished & fits_mass & (peptide_length >= self.size.min_len)
        self.flag.discarded |= self.flag.finished & (peptide_length < self.size.min_len)
        self.flag.finished |= exceeds_mass

    def _write_history(self):
        tokens = einops.rearrange(
            self.current.tokens,
            "B S L -> (B S) L"
        )
        written_idx = self.flag.finished & (~self.flag.discarded)
        if not written_idx.any():
            return
        
        written_idx = written_idx.nonzero(as_tuple=True)[0]
        row = written_idx // self.size.beam_size
        col = written_idx % self.size.beam_size
        mean_scores = self.current.cumsum_scores[row, col] / self.current_step
        exp_scores = torch.exp(mean_scores)
        written_scores = torch.where(
            self.flag.fits[written_idx],
            exp_scores,
            exp_scores - 1
        )
        
        written_tokens = tokens[written_idx]
        if self.current_step + 1 < self.size.max_len:
            written_tokens[:, self.current_step + 1:] = self.special_tokens.sos
        
        self.history.scores[row, col, self.current_step] = written_scores
        self.history.tokens[row, col, self.current_step, :] = written_tokens

    def get_best_peptide(self):
        scores = einops.rearrange(
            self.history.scores,
            "B S L -> B (S L)"
        )
        tokens = einops.rearrange(
            self.history.tokens,
            "B S T L -> B (S T) L"
        )
        best_scores, best_tokens_idx = torch.max(scores, dim=1)
        batch_idx = torch.arange(scores.shape[0], device=self.device)
        best_tokens = tokens[batch_idx, best_tokens_idx, :]
        return BestResult(best_tokens, best_scores)

    def _write_current(self):
        cumsum_scores = einops.repeat(
            self.current.cumsum_scores,
            "B S -> B S V",
            V=self.size.vocab_size
        )
        step_cumsum_scores = cumsum_scores + self.current.scores[:, :, self.current_step, :]
        step_cumsum_scores = einops.rearrange(
            step_cumsum_scores,
            "B S V -> B (S V)"
        )
        # [B, S]
        topk_scores, topk_idx = torch.topk(
            step_cumsum_scores,
            k=self.size.beam_size,
            dim=-1
        )
        row = topk_idx // self.size.vocab_size
        col = topk_idx % self.size.vocab_size
        row = einops.rearrange(
            row,
            "B S -> (B S)"
        )
        batch_idx = einops.repeat(
            torch.arange(self.size.batch_size, device=self.device),
            "B -> (B S)",
            S=self.size.beam_size
        )
        self.current.tokens[:, :, :self.current_step] = einops.rearrange(
            # [B * S, T]
            self.current.tokens[batch_idx, row, :self.current_step],
            "(B S) L -> B S L",
            S=self.size.beam_size
        )
        self.current.tokens[:, :, self.current_step] = col
        self.current.cumsum_scores = topk_scores

        batch_is_active = self.flag.active.view(self.size.batch_size, self.size.beam_size).any(dim=1)
        dead_batches = ~batch_is_active
        if dead_batches.any():
            self.current.tokens[dead_batches, :, self.current_step] = self.special_tokens.pad
        
        if hasattr(self, '_temp_active_cache') and self._temp_active_cache is not None:
            global_order = batch_idx * self.size.beam_size + row
            global_to_active_map = torch.clamp(torch.cumsum(self.flag.active.long(), dim=0) - 1, min=0)
            active_order = global_to_active_map[global_order]
            self.cache = self._temp_active_cache.reorder(active_order)
            self._temp_active_cache = None

    def update(self, active_log_logits: torch.FloatTensor, active_cache: Cache):
        active_idx = self.flag.active.nonzero(as_tuple=True)[0]
        row = active_idx // self.size.beam_size
        col = active_idx % self.size.beam_size
        # [B * S, V]
        self.current.scores[row, col, self.current_step, :] = active_log_logits
        
        self._temp_active_cache = active_cache

        self._write_current()
        self._apply_quality_control()
        self._write_history()
        self.flag.update()
        self.current_step += 1

    def init_state(
        self,
        log_logits: torch.FloatTensor,
        cache: Cache,
    ):
        self.current.scores[:, :, self.current_step, :] = einops.repeat(
            log_logits,
            "B V -> B S V",
            S=self.size.beam_size,
        )
        topk_scores, topk_idx = torch.topk(
            log_logits,
            k=self.size.beam_size,
            dim=-1
        )
        # [B, S]
        self.current.tokens[:, :, self.current_step] = topk_idx
        # [B, S]
        self.current.cumsum_scores = topk_scores

        self.cache = cache.repeat_for_beam(self.size.beam_size)

        self._apply_quality_control()
        self._write_history()
        self.flag.update()
        self.current_step += 1

    def get_next_input(self):
        tokens = einops.rearrange(
            self.current.tokens[:, :, [self.current_step - 1]],
            "B S L -> (B S) L"
        )
        return tokens

class BirdirectBeamStat:
    def __init__(
        self,
        tokenizer: PTMPeptideTokenizer,
        precursors: Precursor,
        config: InferenceConfig
    ):
        self.fwd_stat = BeamStat(
            tokenizer,
            precursors,
            config,
            False
        )
        self.rev_stat = BeamStat(
            tokenizer,
            precursors,
            config,
            True
        )
    
    def init_state(
        self,
        fwd_log_logits: torch.FloatTensor,
        rev_log_logits: torch.FloatTensor,
        fwd_cache: Cache,
        rev_cache: Cache,
    ):
        self.fwd_stat.init_state(
            fwd_log_logits,
            fwd_cache
        )

        self.rev_stat.init_state(
            rev_log_logits,
            rev_cache
        )
    
    def is_done(self):
        return (not self.fwd_stat.active.any()) & (not self.rev_stat.active.any())

    def update(
        self,
        fwd_active_log_logits: torch.FloatTensor,
        fwd_active_cache: Cache,
        rev_active_log_logits: torch.FloatTensor,
        rev_active_cache: Cache
    ):
        active_final = self.fwd_stat.active | self.rev_stat.active
        fwd_mask = self.fwd_stat.active[active_final]
        rev_mask = self.rev_stat.active[active_final]
        if self.fwd_stat.active.any():
            self.fwd_stat.update(
                fwd_active_log_logits[fwd_mask],
                fwd_active_cache.filter_by_mask(fwd_mask)
            )
        
        if self.rev_stat.active.any():
            self.rev_stat.update(
                rev_active_log_logits[rev_mask],
                rev_active_cache.filter_by_mask(rev_mask)
            )
    
    def get_next_input(self):
        fwd_tokens = self.fwd_stat.get_next_input()
        rev_tokens = self.rev_stat.get_next_input()
        active_final = self.fwd_stat.active | self.rev_stat.active

        return NextInput(
            fwd_tokens[active_final],
            rev_tokens[active_final],
            active_final,
            self.fwd_stat.cache.filter_by_mask(active_final),
            self.rev_stat.cache.filter_by_mask(active_final),
        )
    
    def get_best_peptide(self):
        fwd_result = self.fwd_stat.get_best_peptide()
        rev_result = self.rev_stat.get_best_peptide()
        return BirdirectBestResult(fwd_result, rev_result)

def beam_search(
    model: Denovo,
    precursors: Precursor,
    mem_hidden_states: torch.FloatTensor,
    mem_attention_mask: torch.BoolTensor,
    init_log_probs: torch.FloatTensor,
    init_rev_log_probs: torch.FloatTensor,
    init_cache: Cache,
    init_rev_cache: Cache,
    peptide_tokenizer: PTMPeptideTokenizer,
    inference_config: InferenceConfig
):
    stat = BirdirectBeamStat(
        peptide_tokenizer,
        precursors,
        replace(
            inference_config,
            max_len=inference_config.max_len + 2,
        ),
    )
    stat.init_state(
        init_log_probs,
        init_rev_log_probs,
        init_cache,
        init_rev_cache
    )
    
    mem_hidden_states = torch.repeat_interleave(mem_hidden_states, inference_config.num_beams, dim=0)
    mem_attention_mask = torch.repeat_interleave(mem_attention_mask, inference_config.num_beams, dim=0)
    
    for _ in range(inference_config.max_len):
        if stat.is_done():
            break
        
        input = stat.get_next_input()
        output, output_reverse = model.step(
            input.fwd_tokens,
            input.rev_tokens,
            mem_hidden_states[input.active],
            mem_attention_mask[input.active],
            input.fwd_cache,
            input.rev_cache,
            OutputConfig(return_cache=True)
        )
        
        stat.update(
            F.log_softmax(output.logits[:, -1, :], dim=-1),
            output.cache,
            F.log_softmax(output_reverse.logits[:, -1, :], dim=-1),
            output_reverse.cache
        )
    
    result = stat.get_best_peptide()
    fwd_mass = stat.fwd_stat.metadata.idx2masses[result.fwd.tokens].sum(dim=1) + H2O
    rev_mass = stat.rev_stat.metadata.idx2masses[result.rev.tokens].sum(dim=1) + H2O
    fwd_mz = torch.where(
        precursors.charge > 0,
        fwd_mass / precursors.charge + PROTON,
        fwd_mass
    )
    rev_mz = torch.where(
        precursors.charge > 0,
        rev_mass / precursors.charge + PROTON,
        rev_mass
    )
    
    return SearchResult(
        fwd_mz.cpu().numpy(),
        rev_mz.cpu().numpy(),
        result.fwd.scores.cpu().numpy(),
        peptide_tokenizer.detokenize_by_array(result.fwd.tokens.cpu().numpy()),
        result.rev.scores.cpu().numpy(),
        peptide_tokenizer.reverse_detokenize_by_array(result.rev.tokens.cpu().numpy()),
    )
