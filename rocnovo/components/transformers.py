from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import rocnovo.config.model as model_config
import rocnovo.config.data as data_config
from rocnovo.components.float_encoder import FloatEncoder

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.fc1(x)) * self.fc2(x)

def get_activations(activation: str, **kwargs):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "swilu":
        return SwiGLU(**kwargs)
    else:
        raise ValueError(f"activation {activation} not in the expected set")

def prepare_for_scores(hidden_states: torch.FloatTensor, n_head: int, d_head: int):
    # [batch, seq_len, d]
    new_shape = hidden_states.shape[:-1] + (n_head, d_head)
    hidden_states = hidden_states.view(*new_shape)
    return hidden_states.permute(0, 2, 1, 3)

def recover_hidden_states(hidden_states: torch.FloatTensor):
    # n_head * head_size
    hidden_size = hidden_states.shape[1] * hidden_states.shape[-1]
    hidden_states = hidden_states.transpose(1, 2).contiguous()
    shapes = hidden_states.shape[:-2]
    hidden_states = hidden_states.view(*shapes, hidden_size)
    return hidden_states

@torch.amp.autocast(device_type="cuda", enabled=False)
def _apply_rotary_pos_emb(hidden_states: torch.FloatTensor, sinusoidal_pos: tuple[torch.FloatTensor, torch.FloatTensor]):
    cos_emb, sin_emb = sinusoidal_pos
    front, behind = hidden_states.chunk(2, dim=-1)
    if cos_emb.dim() == 3:
        cos_emb = cos_emb[:, None, :, :]
        sin_emb = sin_emb[:, None, :, :]
    
    rotated_front = front * cos_emb - behind * sin_emb
    # [batch_size, n_head, seq_len, d // 2]
    rotated_behind = behind * cos_emb + front * sin_emb
    # [batch_size, n_head, seq_len, d]
    outputs = torch.cat([rotated_front, rotated_behind], dim=-1)
    return outputs.to(hidden_states.dtype)

class PeakRotaryPositionalEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, min_wavelength: float, max_wavelength: float):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("For RoPE, hidden_size must be even.")
        
        if min_wavelength <= 0 or max_wavelength <= 0 or min_wavelength >= max_wavelength:
            raise ValueError("Wavelengths must be positive and min_wavelength < max_wavelength.")

        freq_min = 1.0 / max_wavelength
        freq_max = 1.0 / min_wavelength

        num_freqs = hidden_size // 2
        exponent = torch.arange(num_freqs, dtype=torch.float)
        if num_freqs > 1:
            exponent /= num_freqs - 1

        scale = freq_max / freq_min
        inv_freq = freq_min * (scale ** exponent)
        self.register_buffer("inv_freq", inv_freq)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, positions: torch.LongTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        freqs = torch.einsum("bl,d->bld", positions, self.inv_freq)
        return (freqs.cos(), freqs.sin())

class WordEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int
    ):
        super().__init__()
        self.mass_encoder = FloatEncoder(
            hidden_size // 2,
            0.001,
            10000
        )
        self.charge_encoder = nn.Embedding(
            10,
            hidden_size - hidden_size // 2
        )
        self.aa_encoder = nn.Embedding(
            vocab_size,
            hidden_size, 
            padding_idx=pad_token_id
        )

    def embed(self, tokens: torch.LongTensor):
        return self.aa_encoder(tokens)

    def forward(
        self,
        tokens: torch.LongTensor,
        precursor: data_config.Precursor,
        prompt_hidden_states: Optional[torch.FloatTensor]=None
    ):
        mass_hidden_states = self.mass_encoder(precursor.mass[:, None])
        charge_hidden_states = self.charge_encoder(precursor.charge.int() - 1)
        charge_hidden_states = charge_hidden_states[:, None, :]
        precursor_hidden_states = torch.cat(
            [mass_hidden_states, charge_hidden_states],
            dim=-1
        )
        
        aa_hidden_states = self.embed(tokens)
        aa_hidden_states[:, 0, :] = precursor_hidden_states.squeeze(dim=1)
        hidden_states = aa_hidden_states
        # 插入 prompt_hidden_states
        if prompt_hidden_states is not None:
            if prompt_hidden_states.dim() == 2:
                prompt_hidden_states = prompt_hidden_states[:, None, :]
            
            hidden_states = torch.cat(
                [prompt_hidden_states, hidden_states],
                dim=1
            )
        
        return hidden_states

class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        dropout: float,
        is_causal: bool=True
    ):
        """
            is_causal 为 True 时默认是 causal attention
            forward 输入的 mask 会被忽略
        """
        super().__init__()
        if hidden_size % n_head != 0:
            raise ValueError(f"hidden_size must be divisible by n_head, but got {hidden_size} and {n_head}")

        self.is_causal = is_causal
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.d_head = hidden_size // n_head

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        attention_mask: Optional[torch.BoolTensor]=None,
        past_key_value: Optional[model_config.KVCache]=None,
        cache_return_config: model_config.OutputConfig=model_config.default_output_config
    ):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q = prepare_for_scores(q, self.n_head, self.d_head)
        k = prepare_for_scores(k, self.n_head, self.d_head)
        v = prepare_for_scores(v, self.n_head, self.d_head)
        q = _apply_rotary_pos_emb(q, pos_emb)
        k = _apply_rotary_pos_emb(k, pos_emb)
        if past_key_value is not None:
            k = torch.cat([past_key_value.key, k], dim=2)
            v = torch.cat([past_key_value.value, v], dim=2)
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
        
        q_len = q.size(-2)
        k_len = k.size(-2)
        is_decoding_step = (q_len == 1 and k_len > 1)
        atten_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask if not self.is_causal else None,
            is_causal=self.is_causal and not is_decoding_step,
            dropout_p=self.dropout if self.training else 0.0
        )
        atten_output = recover_hidden_states(atten_output)

        return model_config.AttentionOutput(
            atten_output=atten_output,
            past_key_value=model_config.KVCache(
                key=k,
                value=v,
            ) if cache_return_config.return_cache else None
        )

class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mem_hidden_size: int,
        n_head: int,
        dropout: float
    ):
        super().__init__()
        if hidden_size % n_head != 0:
            raise ValueError(f"hidden_size must be divisible by n_head, but got {hidden_size} and {n_head}")

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(mem_hidden_size, hidden_size)
        self.value = nn.Linear(mem_hidden_size, hidden_size)
        self.dropout = dropout
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.d_head = hidden_size // n_head

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        mem_hidden_states: torch.FloatTensor,
        mem_attention_mask: Optional[torch.BoolTensor]=None,
    ):
        q = self.query(hidden_states)
        k = self.key(mem_hidden_states)
        v = self.value(mem_hidden_states)
        q = prepare_for_scores(q, self.n_head, self.d_head)
        k = prepare_for_scores(k, self.n_head, self.d_head)
        v = prepare_for_scores(v, self.n_head, self.d_head)

        if mem_attention_mask is not None:
            if mem_attention_mask.dim() == 2:
                mem_attention_mask = mem_attention_mask[:, None, None, :]
            elif mem_attention_mask.dim() == 3:
                mem_attention_mask = mem_attention_mask[:, None, :, :]
        
        atten_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mem_attention_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        atten_output = recover_hidden_states(atten_output)

        return model_config.AttentionOutput(atten_output=atten_output)

class SelfOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.FloatTensor, input_tensor: torch.FloatTensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor

class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        attn_module: CrossAttention | SelfAttention
    ):
        super().__init__()
        self.attention = attn_module
        self.output = SelfOutput(hidden_size, dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_emb: tuple[torch.FloatTensor, torch.FloatTensor]=None,
        mem_hidden_states: Optional[torch.FloatTensor]=None,
        mem_attention_mask: Optional[torch.BoolTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        past_key_value: Optional[model_config.KVCache]=None,
        cache_return_config: model_config.OutputConfig=model_config.default_output_config
    ):
        normed_hidden_states = self.norm(hidden_states)
        if isinstance(self.attention, CrossAttention):
            attention_outputs: model_config.AttentionOutput = self.attention(
                hidden_states=normed_hidden_states,
                mem_hidden_states=mem_hidden_states,
                mem_attention_mask=mem_attention_mask
            )
        elif isinstance(self.attention, SelfAttention):
            attention_outputs: model_config.AttentionOutput = self.attention(
                hidden_states=normed_hidden_states,
                pos_emb=pos_emb,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_return_config=cache_return_config
            )
        
        output = attention_outputs.atten_output
        output = self.output(output, hidden_states)

        return model_config.AttentionOutput(
            atten_output=output,
            past_key_value=attention_outputs.past_key_value
        )

class Intermediate(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        intermediate_size: int,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu"
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self._activation = get_activations(activation, hidden_size=intermediate_size)

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self._activation(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.FloatTensor, input_tensor: torch.FloatTensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor

class FFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu"
    ):
        super().__init__()
        self.intermediate = Intermediate(
            hidden_size,
            intermediate_size,
            activation
        )

        self.output = Output(
            hidden_size,
            intermediate_size,
            dropout
        )
    
    def forward(self, hidden_states: torch.FloatTensor):
        intermediate_states = self.intermediate(hidden_states)
        output = self.output(intermediate_states, hidden_states)
        return output

class EncoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu"
    ):
        super().__init__()
        attn = SelfAttention(
            hidden_size,
            n_head,
            dropout,
            is_causal=False
        )
        
        self.self_attn = Attention(
            hidden_size,
            dropout,
            attn
        )

        self.ffn = FFN(
            hidden_size,
            intermediate_size,
            dropout,
            activation
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        attention_mask: Optional[torch.BoolTensor]=None
    ):
        self_attention_outputs: model_config.AttentionOutput = self.self_attn(
            hidden_states=hidden_states,
            pos_emb=pos_emb,
            attention_mask=attention_mask
        )
        hidden_states = self_attention_outputs.atten_output
        hidden_states = self.ffn(hidden_states)
        
        return model_config.LayerOutput(
            last_hidden_states=hidden_states,
            past_key_value=self_attention_outputs.past_key_value
        )

class Encoder(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_layer: int,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu"
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(
                n_head,
                hidden_size,
                intermediate_size,
                dropout,
                activation,
            ) for _ in range(n_layer)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        attention_mask: Optional[torch.BoolTensor]=None,
        return_hidden_states: bool=False
    ):
        all_hidden_states = []
        for _, layer in enumerate(self.layers):
            layer_outputs: model_config.LayerOutput = layer(
                hidden_states=hidden_states,
                pos_emb=pos_emb,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs.last_hidden_states
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        return model_config.Output(
            last_hidden_states=hidden_states,
            hidden_states=all_hidden_states if return_hidden_states else None,
            cache=None
        )

class DecoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        hidden_size: int,
        mem_hidden_size: int,
        intermediate_size: int,
        dropout: float,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu"
    ):
        super().__init__()
        causal_self_attn = SelfAttention(
            hidden_size,
            n_head,
            dropout,
            is_causal=True
        )
        self.self_attn = Attention(
            hidden_size,
            dropout,
            causal_self_attn
        )
        cross_attn = CrossAttention(
            hidden_size,
            mem_hidden_size,
            n_head,
            dropout
        )
        self.cross_attn = Attention(
            hidden_size,
            dropout,
            cross_attn
        )
        self.ffn = FFN(
            hidden_size,
            intermediate_size,
            dropout,
            activation
        )
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        mem_hidden_states: torch.FloatTensor,
        mem_attention_mask: torch.BoolTensor,
        past_key_value: Optional[model_config.KVCache]=None,
        cache_return_config: model_config.OutputConfig=model_config.default_output_config
    ):
        self_attention_outputs: model_config.AttentionOutput = self.self_attn(
            hidden_states=hidden_states,
            pos_emb=pos_emb,
            past_key_value=past_key_value,
            cache_return_config=cache_return_config
        )
        hidden_states = self_attention_outputs.atten_output
        cross_attention_outputs: model_config.AttentionOutput = self.cross_attn(
            hidden_states=hidden_states,
            mem_hidden_states=mem_hidden_states,
            mem_attention_mask=mem_attention_mask
        )
        hidden_states = cross_attention_outputs.atten_output
        hidden_states = self.ffn(hidden_states)
        return model_config.LayerOutput(
            last_hidden_states=hidden_states,
            past_key_value=self_attention_outputs.past_key_value
        )

class Decoder(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_layer: int,
        hidden_size: int,
        mem_hidden_size: int,
        intermediate_size: int,
        dropout: float,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu"
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(
                n_head,
                hidden_size,
                mem_hidden_size,
                intermediate_size,
                dropout,
                activation,
            ) for _ in range(n_layer)
        )
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        mem_hidden_states: torch.FloatTensor,
        mem_attention_mask: Optional[torch.BoolTensor]=None,
        cache: Optional[model_config.Cache]=None,
        cache_return_config: model_config.OutputConfig=model_config.default_output_config
    ):
        key_values_cache = []
        all_hidden_states = []
        for idx, layer in enumerate(self.layers):
            layer_outputs: model_config.LayerOutput = layer(
                hidden_states=hidden_states,
                pos_emb=pos_emb,
                mem_hidden_states=mem_hidden_states,
                mem_attention_mask=mem_attention_mask,
                past_key_value=cache.kv_cache[idx] if cache is not None else None,
                cache_return_config=cache_return_config
            )
            hidden_states = layer_outputs.last_hidden_states
            if cache_return_config.return_cache:
                key_values_cache.append(layer_outputs.past_key_value)
            
            if cache_return_config.return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        return model_config.Output(
            last_hidden_states=hidden_states,
            hidden_states=all_hidden_states if cache_return_config.return_hidden_states else None,
            cache=model_config.Cache(
                kv_cache=key_values_cache
            ) if cache_return_config.return_cache else None
        )