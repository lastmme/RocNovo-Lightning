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
            key=self.key.repeat_interleave(
                beam_size,
                dim=0
            ).clone(),
            value=self.value.repeat_interleave(
                beam_size,
                dim=0
            ).clone()
        )
    
    def filter_by_mask(self, mask: torch.BoolTensor):
        return replace(
            self,
            key=self.key[mask].clone(),
            value=self.value[mask].clone()
        )
    
    def reorder(self, new_order: torch.Tensor):
        idx = new_order.to(self.key.device)
        return replace(
            self,
            key=torch.index_select(
                self.key,
                0,
                idx
            ).clone(),
            value=torch.index_select(
                self.value,
                0,
                idx
            ).clone()
        )

@dataclass(frozen=True)
class AttentionOutput:
    atten_output: torch.Tensor
    atten_score: torch.Tensor=None
    past_key_value: Optional[KVCache]=None

@dataclass(frozen=True)
class LayerOutput:
    last_hidden_states: torch.Tensor
    past_key_value: Optional[KVCache]=None

@dataclass(frozen=True)
class Cache:
    kv_cache: list[KVCache]
    
    @property
    def past_length(self):
        return self.kv_cache[0].past_length()

    def get_ith(self, idx: int):
        return self.kv_cache[idx]
    
    def repeat_for_beam(self, beam_size: int):
        if beam_size <= 0:
            raise ValueError("beam_size must be > 0")
        
        new_kv_cache: list[KVCache] = []
        for i, kv in enumerate(self.kv_cache):
            if kv.key.size(0) != kv.value.size(0):
                raise ValueError(f"kv.key and kv.value batch dim mismatch at layer {i}")
            
            new_kv_cache.append(kv.repeat_for_beam(beam_size))

        return replace(
            self,
            kv_cache=new_kv_cache
        )
    
    def filter_by_mask(self, mask: torch.BoolTensor):
        new_kv_cache = [kv.filter_by_mask(mask) for kv in self.kv_cache]
        return replace(
            self,
            kv_cache=new_kv_cache
        )

    def reorder(self, new_order: torch.Tensor):
        if new_order.dim() != 1:
            raise ValueError("new_order must be a 1D LongTensor")
        
        new_kv_cache = [
            kv.reorder(new_order)
            for kv in self.kv_cache
        ]
        
        return replace(
            self,
            kv_cache=new_kv_cache
        )

@dataclass(frozen=True)
class OutputConfig:
    return_hidden_states: bool=False
    return_cache: bool = False

@dataclass(frozen=True)
class Output:
    last_hidden_states: torch.Tensor
    hidden_states: list[torch.Tensor]
    cache: Optional[Cache]=None

@dataclass(frozen=True)
class DecoderOutput(Output):
    logits: Optional[torch.Tensor] = None

default_output_config = OutputConfig()