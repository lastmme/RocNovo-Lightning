from typing import Optional
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class KVCache:
    key: torch.Tensor # [batch, n_head, seq, d_head]
    value: torch.Tensor # [batch, n_head, seq, d_head]
    
    def past_length(self):
        return self.key.shape[2]

    def update(self, key: torch.Tensor, value: torch.Tensor):
        self.key = torch.concat([self.key, key], dim=2)
        self.value = torch.concat([self.value, value], dim=2)

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
    
    def update(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        for kv_cache, key, value in zip(self.kv_cache, keys, values):
            kv_cache.update(key, value)

@dataclass(frozen=True)
class OutputConfig:
    return_hidden_states: bool=False
    return_cache: bool = False

@dataclass(frozen=True)
class Output:
    last_hidden_states: torch.Tensor
    hidden_states: list[torch.Tensor]
    cache: Optional[Cache]=None

default_output_config = OutputConfig()