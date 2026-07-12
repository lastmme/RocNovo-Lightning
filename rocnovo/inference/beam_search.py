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
    mass_fits = torch.zeros(
        (B, S, L),
        dtype=torch.bool,
        device=device
    )

    return History(
        tokens,
        scores,
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

        # Per-beam mass-fit flag for the most recent quality-control step.
        # It is updated in _apply_quality_control and consumed in _write_history.
        self._mass_fits = torch.zeros(
            batch_size * config.num_beams,
            dtype=torch.bool,
            device=device
        )

        # Step at which each beam finished (generated SOS). Used by the empty-
        # history fallback in get_best_peptide to normalise scores consistently
        # with entries written to history.
        self._finished_step = torch.zeros(
            batch_size,
            config.num_beams,
            dtype=torch.long,
            device=device,
        )

        self.precursors = precursors.repeat_beamsize(config.num_beams)
        self.tokenizer = tokenizer
        self.config = config
        self.is_reverse = self.tokenizer.reverse if not is_reverse else (not self.tokenizer.reverse)
        self.current_step = 1
        self.cache = None
        self._temp_active_cache = None

    @property
    def active(self):
        return self.flag.active.clone()

    def _record_finished_steps(self):
        """Record the current step for beams that just became inactive."""
        was_active = self.flag.active.view(
            self.size.batch_size, self.size.beam_size
        ).clone()
        self.flag.update()
        newly_finished = was_active & (~self.flag.active.view(
            self.size.batch_size, self.size.beam_size
        ))
        self._finished_step[newly_finished] = self.current_step

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

        # Keep the per-beam mass-fit flag for later history selection.
        # The final ``flag.fits`` additionally requires min_len; for the
        # get_best_peptide fallback we want to relax the min_len constraint
        # while still preferring mass-compliant candidates.
        self._mass_fits = fits_mass
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
        # Record whether the written entry satisfies the mass tolerance
        # regardless of the min_len requirement.
        self.history.mass_fits[row, col, self.current_step] = self._mass_fits[written_idx]

    def get_best_peptide(self):
        scores = einops.rearrange(
            self.history.scores,
            "B S L -> B (S L)"
        )
        tokens = einops.rearrange(
            self.history.tokens,
            "B S T L -> B (S T) L"
        )
        mass_fits = einops.rearrange(
            self.history.mass_fits,
            "B S L -> B (S L)"
        )

        # Highest-scoring candidate (legacy behaviour).
        best_scores, best_tokens_idx = torch.max(scores, dim=1)
        batch_idx = torch.arange(scores.shape[0], device=self.device)
        best_tokens = tokens[batch_idx, best_tokens_idx, :]

        # If the top candidate is not mass-compliant, fall back to the best
        # mass-compliant candidate in history.  The fallback intentionally does
        # not enforce min_len, but it does require mass fit (and token validity,
        # which is already guaranteed for written history entries).

        # To avoid overriding high-quality short predictions whose mass is
        # slightly outside the strict beam-search tolerance (e.g. correct
        # length-5 peptides), only trigger the fallback when the top candidate
        # is itself poor (mean log-prob below a threshold).
        top_fits_mass = mass_fits[batch_idx, best_tokens_idx]
        # Recover mean log-prob from the stored score.
        # mass-fit entries: score = exp(mean)
        # non-mass-fit entries: score = exp(mean) - 1
        eps = 1e-6
        top_mean = torch.where(
            top_fits_mass,
            torch.log(best_scores.clamp_min(eps)),
            torch.log((best_scores + 1.0).clamp_min(eps)),
        )
        # Never-written history slots have score -2.0; treat them as -inf mean.
        top_mean = torch.where(best_scores > -1.9, top_mean, float("-inf"))
        needs_fallback = ~top_fits_mass & (top_mean < -0.5)
        if not needs_fallback.any():
            return BestResult(best_tokens, best_scores)

        # Mask non-mass-fitting entries to -inf so torch.max picks the best
        # compliant one per batch.
        fallback_scores = torch.where(
            mass_fits,
            scores,
            torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device),
        )
        fallback_scores, fallback_tokens_idx = torch.max(fallback_scores, dim=1)

        # Use the fallback only when a mass-fitting candidate actually exists
        # (i.e. its score is not the -inf sentinel).  The default history score
        # is -2.0 for never-written slots, so any real written entry is larger.
        fallback_exists = fallback_scores > -1.9
        use_fallback = needs_fallback & fallback_exists

        final_idx = torch.where(use_fallback, fallback_tokens_idx, best_tokens_idx)
        final_scores = torch.where(use_fallback, fallback_scores, best_scores)
        final_tokens = tokens[batch_idx, final_idx, :]

        # Last-resort fallback: if history contains no written entry for a sample
        # (score is still the default -2.0), use the best current beam state so
        # that we never return a completely empty peptide when the model has
        # produced any real tokens at all.
        empty_history = final_scores <= -1.9
        if empty_history.any():
            current_tokens = self.current.tokens[empty_history]  # [N, S, L]
            current_scores = self.current.cumsum_scores[empty_history]  # [N, S]
            finished_steps = self._finished_step[empty_history]  # [N, S]
            # Count real peptide tokens per beam, ignoring SOS/PAD.
            valid_mask = (
                (current_tokens != self.special_tokens.sos)
                & (current_tokens != self.special_tokens.pad)
            )
            valid_count = valid_mask.any(dim=-1)  # [N, S]
            # Penalise beams with no valid tokens so they are only chosen when
            # there is no alternative.
            selectable_scores = torch.where(
                valid_count,
                current_scores,
                torch.tensor(float("-inf"), dtype=current_scores.dtype, device=current_scores.device),
            )
            best_beam_scores, best_beam_idx = torch.max(selectable_scores, dim=1)
            # Only overwrite when at least one beam has a valid token.
            empty_batch_idx = torch.arange(best_beam_idx.shape[0], device=best_beam_idx.device)
            has_valid = valid_count[empty_batch_idx, best_beam_idx].any()
            if has_valid:
                selected_tokens = current_tokens[empty_batch_idx, best_beam_idx, :]
                selected_finished_step = finished_steps[empty_batch_idx, best_beam_idx]
                # Use the step at which the beam finished to normalise the score,
                # matching the denominator used when writing history. Beams that
                # never finished (e.g., hit max_len) fall back to current_step.
                selected_finished_step = torch.where(
                    selected_finished_step > 0,
                    selected_finished_step,
                    self.current_step,
                )
                # Score mirrors the non-fit history writing (exp(mean_score) - 1).
                mean_score = best_beam_scores / selected_finished_step
                selected_scores = torch.exp(mean_score) - 1.0
                final_tokens[empty_history] = selected_tokens
                final_scores[empty_history] = selected_scores
        
        return BestResult(final_tokens, final_scores)

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
        self.current.tokens[:, :, :self.current_step] = einops.rearrange(
            # [B * S, T]
            self.current.tokens[batch_idx, row, :self.current_step],
            "(B S) L -> B S L",
            S=self.size.beam_size
        )
        self.current.tokens[:, :, self.current_step] = col

        batch_is_active = self.flag.active.view(self.size.batch_size, self.size.beam_size).any(dim=1)
        # Samples that have already terminated should keep their last beam scores
        # so the final fallback returns a consistent score regardless of when
        # other samples in the same batch finish.
        self.current.cumsum_scores = torch.where(
            batch_is_active[:, None],
            topk_scores,
            self.current.cumsum_scores,
        )

        dead_batches = ~batch_is_active
        # Avoid .any().item() sync; masked assignment is a no-op when the mask
        # is all False.
        self.current.tokens[dead_batches, :, self.current_step] = self.special_tokens.pad
        
        if self._temp_active_cache is not None:
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
        # Rearrange once and reuse for quality control and history writing.
        tokens = einops.rearrange(self.current.tokens, "B S L -> (B S) L")
        self._apply_quality_control(tokens)
        self._write_history(tokens)
        self._record_finished_steps()
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

        tokens = einops.rearrange(self.current.tokens, "B S L -> (B S) L")
        self._apply_quality_control(tokens)
        self._write_history(tokens)
        self._record_finished_steps()
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
        # Respect the cache flags inherited from the prefill cache: if a cache
        # type was stripped because it is disabled in InferenceConfig, do not
        # recompute/store it during beam search.
        step_cache_config = OutputConfig(
            return_cross_cache=input.fwd_cache.has_cross_cache(),
            return_self_cache=input.fwd_cache.has_self_cache(),
        )

        output, output_reverse = model.step(
            input.fwd_tokens,
            input.rev_tokens,
            mem_hidden_states[input.active],
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
    
    return SearchResult(
        fwd_mz.cpu().numpy(),
        rev_mz.cpu().numpy(),
        result.fwd.scores.cpu().numpy(),
        peptide_tokenizer.detokenize_by_array(result.fwd.tokens.cpu().numpy()),
        result.rev.scores.cpu().numpy(),
        peptide_tokenizer.reverse_detokenize_by_array(result.rev.tokens.cpu().numpy()),
    )
