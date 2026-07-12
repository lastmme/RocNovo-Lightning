from typing import Optional
from dataclasses import dataclass, replace

import torch

@dataclass(frozen=True)
class KVCache:
    key: torch.Tensor # [batch, n_head, seq, d_head]
    value: torch.Tensor # [batch, n_head, seq, d_head]
    
    def past_length(self):
        return self.key.shape[2]
    
    def repeat_for_beam(self, beam_size: int):
        return replace(
            self,
            key=self.key.repeat_interleave(beam_size, dim=0),
            value=self.value.repeat_interleave(beam_size, dim=0),
        )
    
    def filter_by_mask(self, mask: torch.BoolTensor):
        return replace(
            self,
            key=self.key[mask],
            value=self.value[mask]
        )
    
    def reorder(self, new_order: torch.Tensor):
        idx = new_order.to(self.key.device)
        return replace(
            self,
            key=self.key[idx],
            value=self.value[idx]
        )

@dataclass(frozen=True)
class AttentionOutput:
    atten_output: torch.Tensor
    atten_score: torch.Tensor=None
    past_key_value: Optional[KVCache]=None
    cross_key_value: Optional[KVCache]=None

@dataclass(frozen=True)
class LayerOutput:
    last_hidden_states: torch.Tensor
    past_key_value: Optional[KVCache]=None
    cross_key_value: Optional[KVCache]=None

@dataclass(frozen=True)
class Cache:
    kv_cache: list[KVCache]
    cross_kv_cache: Optional[list[KVCache]]=None
    cross_kv_index: Optional[torch.LongTensor]=None
    past_length: int=0
    # Prompt hidden states from the CLIP spectrum prefill.  Kept in the cache
    # so that when self KV cache is disabled we can recompute self attention
    # over the full prefix [prompt, precursor, tokens...] exactly as in prefill.
    prompt_hidden_states: Optional[torch.FloatTensor]=None

    def get_ith(self, idx: int):
        if self.kv_cache is None:
            raise ValueError("kv_cache is not available")
        
        return self.kv_cache[idx]
    
    def has_self_cache(self) -> bool:
        return self.kv_cache is not None and len(self.kv_cache) > 0

    def has_cross_cache(self) -> bool:
        return self.cross_kv_cache is not None and len(self.cross_kv_cache) > 0

    def without_self_cache(self):
        """Return a new Cache with self-attention KV cache removed."""
        return replace(
            self,
            kv_cache=None,
        )

    def without_cross_cache(self):
        """Return a new Cache with cross-attention KV cache removed."""
        return replace(
            self,
            cross_kv_cache=None,
            cross_kv_index=None,
        )

    def repeat_for_beam(self, beam_size: int):
        if beam_size <= 0:
            raise ValueError("beam_size must be > 0")

        new_kv_cache = None
        if self.kv_cache is not None:
            new_kv_cache = []
            for i, kv in enumerate(self.kv_cache):
                if kv.key.size(0) != kv.value.size(0):
                    raise ValueError(f"kv.key and kv.value batch dim mismatch at layer {i}")

                new_kv_cache.append(kv.repeat_for_beam(beam_size))

        new_cross_kv_index = None
        if self.cross_kv_cache is not None:
            device = self.cross_kv_cache[0].key.device
            batch_size = self.cross_kv_cache[0].key.size(0)
            new_cross_kv_index = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(batch_size, beam_size)
                .reshape(batch_size * beam_size)
            )

        new_prompt = None
        if self.prompt_hidden_states is not None:
            new_prompt = self.prompt_hidden_states.repeat_interleave(beam_size, dim=0)

        return replace(
            self,
            kv_cache=new_kv_cache,
            cross_kv_cache=self.cross_kv_cache,
            cross_kv_index=new_cross_kv_index,
            prompt_hidden_states=new_prompt,
        )
    
    def filter_by_mask(self, mask: torch.BoolTensor):
        new_kv_cache = None
        if self.kv_cache is not None:
            new_kv_cache = [kv.filter_by_mask(mask) for kv in self.kv_cache]

        new_cross_kv_index = None
        if self.cross_kv_index is not None:
            new_cross_kv_index = self.cross_kv_index[mask]

        new_prompt = None
        if self.prompt_hidden_states is not None:
            new_prompt = self.prompt_hidden_states[mask]

        return replace(
            self,
            kv_cache=new_kv_cache,
            cross_kv_cache=self.cross_kv_cache,
            cross_kv_index=new_cross_kv_index,
            prompt_hidden_states=new_prompt,
        )

    def reorder(self, new_order: torch.Tensor):
        if new_order.dim() != 1:
            raise ValueError("new_order must be a 1D LongTensor")

        new_kv_cache = None
        if self.kv_cache is not None:
            new_kv_cache = [kv.reorder(new_order) for kv in self.kv_cache]

        new_cross_kv_index = None
        if self.cross_kv_index is not None:
            new_cross_kv_index = self.cross_kv_index[new_order]

        new_prompt = None
        if self.prompt_hidden_states is not None:
            new_prompt = self.prompt_hidden_states[new_order]

        return replace(
            self,
            kv_cache=new_kv_cache,
            cross_kv_cache=self.cross_kv_cache,
            cross_kv_index=new_cross_kv_index,
            prompt_hidden_states=new_prompt,
        )

@dataclass(frozen=True)
class OutputConfig:
    return_hidden_states: bool=False
    return_cross_cache: bool=False
    return_self_cache: bool=False

@dataclass(frozen=True)
class Output:
    last_hidden_states: torch.Tensor
    hidden_states: list[torch.Tensor]
    cache: Optional[Cache]=None

@dataclass(frozen=True)
class DecoderOutput(Output):
    logits: Optional[torch.Tensor]=None

default_output_config = OutputConfig()