from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
import einops
import numpy as np

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

LOGPROB_SENTINEL = -1e4

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
    token_logprobs: torch.FloatTensor # [B, S, L]
    cumsum_scores: torch.FloatTensor # [B, S]

@dataclass
class History:
    tokens: torch.LongTensor  # [B, S, L, L]
    logprobs: torch.FloatTensor # [B, S, L, L]
    mass_fits: torch.BoolTensor # [B, S, L]

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
    mass_fits: torch.BoolTensor # [B]
    logprobs: torch.FloatTensor # [B, L], per-position log-prob of the chosen tokens
    lengths: torch.LongTensor # [B], emitted entries incl. the terminating SOS

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

    token_logprobs = torch.zeros(
        (B, S, L),
        device=device,
    )

    return Current(
        tokens,
        scores,
        token_logprobs,
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
    logprobs = torch.full(
        (B, S, L, L),
        fill_value=LOGPROB_SENTINEL,
        device=device
    )
    mass_fits = torch.zeros(
        (B, S, L),
        dtype=torch.bool,
        device=device
    )

    return History(
        tokens,
        logprobs,
        mass_fits,
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
        self._temp_active_cache = None
        self._parent_global = None

    @property
    def active(self):
        return self.flag.active.clone()

    def _apply_quality_control(self, tokens: torch.LongTensor):
        BS = self.size.batch_size * self.size.beam_size
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

        # [B * S, L]
        pred_tokens = tokens[:, 1:self.current_step + 1]
        is_special = (pred_tokens == self.special_tokens.sos) | (pred_tokens == self.special_tokens.pad)
        peptide_length = (~is_special).sum(dim=1)
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

    def _write_history(self, tokens: torch.LongTensor):
        written_idx = self.flag.finished & (~self.flag.discarded)
        # Avoid a CPU-GPU sync from .any().item(); if nothing is written the
        # subsequent indexing operations produce empty tensors and the writes
        # are no-ops.
        written_idx = written_idx.nonzero(as_tuple=True)[0]
        if written_idx.numel() == 0:
            return

        row = written_idx // self.size.beam_size
        col = written_idx % self.size.beam_size
        written_tokens = tokens[written_idx]
        if self.current_step + 1 < self.size.max_len:
            written_tokens[:, self.current_step + 1:] = self.special_tokens.sos

        self.history.tokens[row, col, self.current_step, :] = written_tokens
        # Slot t stores the taken log-probs of steps 1..t in its first t positions
        self.history.logprobs[row, col, self.current_step, :self.current_step] = (
            self.current.token_logprobs[row, col, 1:self.current_step + 1]
        )
        # Written entries always satisfy min_len, so flag.fits marks mass fit
        self.history.mass_fits[row, col, self.current_step] = self.flag.fits[written_idx]

    def get_best_peptide(self):
        # Aggregate: score = exp(mean(logp)), the only exp in the whole pipeline
        logprobs = self.history.logprobs
        B, S, T, L = logprobs.shape
        steps = torch.arange(T, device=self.device)
        pos_mask = torch.arange(L, device=self.device)[None, :] < steps[:, None]  # [T, L]
        step_sums = (logprobs * pos_mask[None, None]).sum(dim=-1)  # [B, S, T]
        scores = torch.exp(step_sums / steps.clamp_min(1))
        scores[:, :, 0] = 0.0

        scores = einops.rearrange(scores, "B S T -> B (S T)")
        tokens = einops.rearrange(self.history.tokens, "B S T L -> B (S T) L")
        mass_fits = einops.rearrange(self.history.mass_fits, "B S T -> B (S T)")

        # Prefer mass-fitting candidates when any exists, otherwise rank by score
        has_fit = mass_fits.any(dim=1)
        fitting_only = torch.where(mass_fits, scores, torch.full_like(scores, -1.0))
        best_scores, best_idx = torch.max(
            torch.where(has_fit[:, None], fitting_only, scores),
            dim=1,
        )
        batch_idx = torch.arange(scores.shape[0], device=self.device)
        final_tokens = tokens[batch_idx, best_idx, :]
        best_mass_fits = mass_fits[batch_idx, best_idx].clone()

        # Per-token log-probs of the chosen candidates, aligned to token
        # positions (position 0 and unused positions stay 0). A history slot
        # t keeps the log-probs of steps 1..t in its first t entries.
        flat_logprobs = einops.rearrange(self.history.logprobs, "B S T L -> B (S T) L")
        slot_steps = best_idx % T
        compact = flat_logprobs[batch_idx, best_idx]  # [B, L]
        pos = torch.arange(L, device=self.device)
        final_logprobs = torch.zeros_like(compact)
        final_logprobs[:, 1:] = torch.where(
            pos[None, 1:] <= slot_steps[:, None],
            compact[:, :-1],
            torch.zeros_like(compact[:, :-1]),
        )
        final_lengths = slot_steps.clone()

        if (best_scores == 0).any():
            self._empty_history_fallback(final_tokens, best_scores, best_mass_fits, final_logprobs, final_lengths)

        return BestResult(final_tokens, best_scores, best_mass_fits, final_logprobs, final_lengths)

    def _empty_history_fallback(
        self,
        final_tokens: torch.LongTensor,
        best_scores: torch.FloatTensor,
        best_mass_fits: torch.BoolTensor,
        final_logprobs: torch.FloatTensor,
        final_lengths: torch.LongTensor,
    ):
        """Overwrite empty-history samples with the best live beam holding
        valid tokens, in place and per sample."""
        empty_history = best_scores == 0
        current_tokens = self.current.tokens[empty_history]  # [N, S, L]
        current_scores = self.current.cumsum_scores[empty_history]  # [N, S]
        valid_mask = (
            (current_tokens != self.special_tokens.sos)
            & (current_tokens != self.special_tokens.pad)
        )
        has_valid_tokens = valid_mask.any(dim=-1)  # [N, S]
        selectable_scores = torch.where(
            has_valid_tokens,
            current_scores,
            torch.tensor(float("-inf"), dtype=current_scores.dtype, device=current_scores.device),
        )
        best_beam_scores, best_beam_idx = torch.max(selectable_scores, dim=1)

        n_idx = torch.arange(current_tokens.shape[0], device=self.device)
        has_valid = has_valid_tokens[n_idx, best_beam_idx]

        # Derive the finish step from tokens: first special token after position 0
        special_after_start = (
            (current_tokens == self.special_tokens.sos)
            | (current_tokens == self.special_tokens.pad)
        )
        special_after_start[:, :, 0] = False
        has_stop = special_after_start.any(dim=-1)  # [N, S]
        stop_step = torch.argmax(special_after_start.long(), dim=-1)  # [N, S]
        # Number of choices made by the beam: positions 1..stop_step with a
        # terminator, or 1..current_step-1 when it never finished. Used both
        # as the score normaliser and the log-prob entry count.
        beam_steps = torch.where(has_stop, stop_step, self.current_step - 1).clamp_min(1)

        selected_scores = torch.exp(best_beam_scores / beam_steps[n_idx, best_beam_idx])

        empty_indices = empty_history.nonzero(as_tuple=True)[0]
        valid_empty_indices = empty_indices[has_valid]
        final_tokens[valid_empty_indices] = current_tokens[n_idx[has_valid], best_beam_idx[has_valid]]
        best_scores[valid_empty_indices] = selected_scores[has_valid]
        best_mass_fits[valid_empty_indices] = False

        # Rebuild the log-prob row from the live beam state
        final_logprobs[valid_empty_indices] = 0.0
        sel_token_logprobs = self.current.token_logprobs[empty_history][n_idx[has_valid], best_beam_idx[has_valid]]  # [M, L]
        sel_steps = beam_steps[n_idx, best_beam_idx][has_valid]  # [M]
        final_lengths[empty_indices] = 0
        final_lengths[valid_empty_indices] = sel_steps
        pos = torch.arange(final_logprobs.shape[-1], device=self.device)
        fill_mask = pos[None, 1:] <= sel_steps[:, None]
        final_logprobs[valid_empty_indices, 1:] = torch.where(
            fill_mask,
            sel_token_logprobs[:, 1:],
            torch.zeros_like(sel_token_logprobs[:, 1:]),
        )

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
        # Single divmod call instead of separate // and %.
        row = torch.div(topk_idx, self.size.vocab_size, rounding_mode="floor")
        col = topk_idx - row * self.size.vocab_size
        row = einops.rearrange(
            row,
            "B S -> (B S)"
        )
        batch_idx = einops.repeat(
            torch.arange(self.size.batch_size, device=self.device),
            "B -> (B S)",
            S=self.size.beam_size
        )

        # On dead batches topk indices are arbitrary, keep prefix and step token
        batch_is_active = self.flag.active.view(self.size.batch_size, self.size.beam_size).any(dim=1)

        new_prefix = einops.rearrange(
            # [B * S, T]
            self.current.tokens[batch_idx, row, :self.current_step],
            "(B S) L -> B S L",
            S=self.size.beam_size
        )
        self.current.tokens[:, :, :self.current_step] = torch.where(
            batch_is_active[:, None, None],
            new_prefix,
            self.current.tokens[:, :, :self.current_step],
        )
        self.current.tokens[:, :, self.current_step] = torch.where(
            batch_is_active[:, None],
            col,
            torch.full_like(col, self.special_tokens.pad),
        )

        # Take the chosen token's log-prob directly from the step's log-softmax
        step_logprobs = einops.rearrange(
            self.current.scores[:, :, self.current_step, :],
            "B S V -> B (S V)"
        )
        token_logprobs = step_logprobs.gather(1, topk_idx)  # [B, S]
        new_taken_prefix = einops.rearrange(
            self.current.token_logprobs[batch_idx, row, :self.current_step],
            "(B S) L -> B S L",
            S=self.size.beam_size
        )
        self.current.token_logprobs[:, :, :self.current_step] = torch.where(
            batch_is_active[:, None, None],
            new_taken_prefix,
            self.current.token_logprobs[:, :, :self.current_step],
        )
        self.current.token_logprobs[:, :, self.current_step] = torch.where(
            batch_is_active[:, None],
            token_logprobs,
            torch.zeros_like(token_logprobs),
        )

        # Samples that have already terminated should keep their last beam scores
        # so the final fallback returns a consistent score regardless of when
        # other samples in the same batch finish.
        self.current.cumsum_scores = torch.where(
            batch_is_active[:, None],
            topk_scores,
            self.current.cumsum_scores,
        )

        # Parent slot of every beam slot; consumed by BirdirectBeamStat to
        # compact the caches in one gather
        self._parent_global = batch_idx * self.size.beam_size + row

    def update(self, active_log_logits: torch.FloatTensor, active_cache: Cache):
        active_idx = self.flag.active.nonzero(as_tuple=True)[0]
        row = active_idx // self.size.beam_size
        col = active_idx % self.size.beam_size
        # [B * S, V]
        self.current.scores[row, col, self.current_step, :] = active_log_logits
        
        self._temp_active_cache = active_cache

        self._write_current()
        # Rearrange once and reuse for quality control and history writing.
        tokens = einops.rearrange(self.current.tokens, "B S L -> (B S) L")
        self._apply_quality_control(tokens)
        self._write_history(tokens)
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
        self.current.token_logprobs[:, :, self.current_step] = topk_scores

        self.cache = cache.repeat_for_beam(self.size.beam_size)

        tokens = einops.rearrange(self.current.tokens, "B S L -> (B S) L")
        self._apply_quality_control(tokens)
        self._write_history(tokens)
        self.flag.update()
        self.current_step += 1

    def get_next_input(self):
        # Cache reordering is applied in _write_current, so self.cache is always
        # in the current B*S logical order.  This method only prepares the input
        # tokens.

        if self.cache is not None and not self.cache.has_self_cache():
            # Without self KV cache we must recompute self attention over the
            # full prefix generated so far. current_step has already been
            # advanced, so the prefix is everything before it.
            tokens = einops.rearrange(
                self.current.tokens[:, :, :self.current_step],
                "B S L -> (B S) L"
            )
        else:
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

        # Caches start B*S-ordered; compact to the post-init union rows
        B = self.fwd_stat.size.batch_size
        S = self.fwd_stat.size.beam_size
        self._compact_caches(torch.ones(B * S, dtype=torch.bool, device=self.fwd_stat.device))

    def is_done(self):
        return (not self.fwd_stat.active.any()) & (not self.rev_stat.active.any())

    def _compact_caches(self, prev_union: torch.BoolTensor):
        # Re-gather both caches in one pass to the next step's union rows:
        # each live slot pulls its parent's row, replacing the three separate
        # materializations (union filter, per-direction filter, B*S re-expand)
        next_union = self.fwd_stat.active | self.rev_stat.active
        if not next_union.any():
            for stat in (self.fwd_stat, self.rev_stat):
                if stat._temp_active_cache is not None:
                    stat.cache = stat._temp_active_cache
                    stat._temp_active_cache = None
            return

        # Row index of every global slot in the current (prev_union-ordered) cache
        cur_row = torch.clamp(torch.cumsum(prev_union.long(), dim=0) - 1, min=0)
        next_slots = next_union.nonzero(as_tuple=True)[0]
        for stat in (self.fwd_stat, self.rev_stat):
            if stat._temp_active_cache is not None:
                src = cur_row[stat._parent_global][next_slots]
                stat.cache = stat._temp_active_cache.reorder(src)
                stat._temp_active_cache = None
            else:
                # Direction idle this step: keep each slot's own row
                src = cur_row[next_slots]
                stat.cache = stat.cache.reorder(src)

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
                fwd_active_cache
            )

        if self.rev_stat.active.any():
            self.rev_stat.update(
                rev_active_log_logits[rev_mask],
                rev_active_cache
            )

        self._compact_caches(active_final)

    def get_next_input(self):
        fwd_tokens = self.fwd_stat.get_next_input()
        rev_tokens = self.rev_stat.get_next_input()
        active_final = self.fwd_stat.active | self.rev_stat.active

        return NextInput(
            fwd_tokens[active_final],
            rev_tokens[active_final],
            active_final,
            self.fwd_stat.cache,
            self.rev_stat.cache,
        )
    
    def get_best_peptide(self):
        fwd_result = self.fwd_stat.get_best_peptide()
        rev_result = self.rev_stat.get_best_peptide()
        return BirdirectBestResult(fwd_result, rev_result)

def _format_token_logprobs(logprobs: np.ndarray, lengths: np.ndarray):
    # Entries cover generated steps 1..lengths[i] incl. the terminating SOS;
    # validity comes from the recorded length, not from the log-prob value
    return [
        ";".join(f"{lp:.6f}" for lp in lps[1:lengths[i] + 1])
        for i, lps in enumerate(logprobs)
    ]

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
    
    # With the cross cache active the decoder never reads the memory hidden
    # states, skip the beam repeat and the per-step gather entirely
    step_needs_mem = not init_cache.has_cross_cache()
    if step_needs_mem:
        mem_hidden_states = torch.repeat_interleave(mem_hidden_states, inference_config.num_beams, dim=0)
    mem_attention_mask = torch.repeat_interleave(mem_attention_mask, inference_config.num_beams, dim=0)
    
    for _ in range(inference_config.max_len):
        if stat.is_done():
            break
        
        input = stat.get_next_input()
        # Respect the cache flags inherited from the prefill cache: if a cache
        # type was stripped because it is disabled in InferenceConfig, do not
        # recompute/store it during beam search.
        step_cache_config = OutputConfig(
            return_cross_cache=input.fwd_cache.has_cross_cache(),
            return_self_cache=input.fwd_cache.has_self_cache(),
        )

        step_mem = mem_hidden_states[input.active] if step_needs_mem else None
        output, output_reverse = model.step(
            input.fwd_tokens,
            input.rev_tokens,
            step_mem,
            mem_attention_mask[input.active],
            input.fwd_cache,
            input.rev_cache,
            step_cache_config,
            stat.fwd_stat.precursors[input.active],
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

    fwd_token_logprobs = _format_token_logprobs(
        result.fwd.logprobs.cpu().numpy(), result.fwd.lengths.cpu().numpy()
    )
    rev_token_logprobs = _format_token_logprobs(
        result.rev.logprobs.cpu().numpy(), result.rev.lengths.cpu().numpy()
    )
    fwd_peptides = peptide_tokenizer.detokenize_by_array(result.fwd.tokens.cpu().numpy())
    rev_peptides = peptide_tokenizer.reverse_detokenize_by_array(result.rev.tokens.cpu().numpy())

    return SearchResult(
        fwd_mz.cpu().numpy(),
        rev_mz.cpu().numpy(),
        result.fwd.mass_fits.cpu().numpy(),
        result.rev.mass_fits.cpu().numpy(),
        result.fwd.scores.cpu().numpy(),
        fwd_peptides,
        result.rev.scores.cpu().numpy(),
        rev_peptides,
        fwd_token_logprobs,
        rev_token_logprobs,
    )
