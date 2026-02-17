from typing import Optional, Literal

import torch
import torch.nn as nn
import einops

from rocnovo.components.transformers import WordEmbedding, PeakRotaryPositionalEmbeddings, Decoder
from rocnovo.components.encoders import PositionalEncoder
from rocnovo.tokenizer.peptide import CANONICAL, SPECIAL_TOKENS, PAD
import rocnovo.config.model as model_config
from rocnovo.config.data import Peptide, Precursor

class ClipPeptideDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int=len(CANONICAL) + len(SPECIAL_TOKENS),
        hidden_size: int=128,
        n_head: int=8,
        dim_feedforward: int=1024,
        n_layers: int=1,
        dropout: float=0
    ):
        super().__init__()
        self.aa_encoder = torch.nn.Embedding(
            n_vocab,
            hidden_size,
            padding_idx=SPECIAL_TOKENS[PAD],
        )

        self.pos_encoder = PositionalEncoder(hidden_size)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
            norm=torch.nn.LayerNorm(hidden_size),
        )
        self.ln_final = nn.LayerNorm(hidden_size)
        self.projection = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, tokens: torch.LongTensor):
        hidden_states = self.aa_encoder(tokens)
        hidden_states = self.pos_encoder(hidden_states)
        seq_len = hidden_states.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(hidden_states.device)
        latent = self.transformer_encoder(hidden_states, mask=mask, is_causal=True)
        latent = self.ln_final(latent)
        latent = self.projection(latent)
        return latent

class BiDirectRopeDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int=len(CANONICAL) + len(SPECIAL_TOKENS),
        hidden_size: int=128,
        mem_hidden_size: int=128,
        n_head: int=8,
        dim_feedforward: int=1024,
        n_layers: int=1,
        dropout: float=0,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu"
    ):
        super().__init__()
        self.word_embedding = WordEmbedding(
            n_vocab,
            hidden_size,
            SPECIAL_TOKENS[PAD]
        )
        self.d_k = hidden_size // n_head
        self.rotary_emb = PeakRotaryPositionalEmbeddings(
            self.d_k,
            1,
            1000
        )
        self.decoder = Decoder(
            n_head,
            n_layers,
            hidden_size,
            mem_hidden_size,
            dim_feedforward,
            dropout,
            activation
        )
        self.decoder_reverse = Decoder(
            n_head,
            n_layers,
            hidden_size,
            mem_hidden_size,
            dim_feedforward,
            dropout,
            activation
        )
        self.lm_head = nn.Linear(hidden_size, n_vocab)
        self.lm_head_reverse = nn.Linear(hidden_size, n_vocab)
    
    def calculate_pos_emb(self, start: int, end: int, batch_size: int, device: torch.device):
        positions = torch.arange(
            start,
            end,
            device=device,
            dtype=torch.float
        ).unsqueeze(0)  # (1, L+1)
        positions = einops.repeat(positions, "1 L -> B L", B=batch_size)
        pos_emb = self.rotary_emb(positions)
        return pos_emb

    def decode_with_cache(
        self,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        mem_hidden_states: torch.FloatTensor,
        mem_attention_mask: Optional[torch.BoolTensor]=None,
        cache: Optional[model_config.Cache]=None,
        cache_reverse: Optional[model_config.Cache]=None,
        cache_return_config: model_config.OutputConfig=model_config.default_output_config
    ):
        hidden_states = self.word_embedding.embed(tokens)
        hidden_states_reverse = self.word_embedding.embed(tokens_reverse)
        start_pos = 0
        if cache is not None:
            start_pos = cache.kv_cache[0].shape[-2]
        
        pos_emb = self.calculate_pos_emb(
            start_pos,
            hidden_states.size(1) + start_pos,
            hidden_states.size(0),
            hidden_states.device
        )
        output: model_config.Output = self.decoder(
            hidden_states,
            pos_emb,
            mem_hidden_states,
            mem_attention_mask,
            cache,
            cache_return_config
        )
        output_reverse: model_config.Output = self.decoder_reverse(
            hidden_states_reverse,
            pos_emb,
            mem_hidden_states,
            mem_attention_mask,
            cache_reverse,
            cache_return_config
        )
        logits = self.lm_head(output.last_hidden_states)
        logits_reverse = self.lm_head_reverse(output_reverse.last_hidden_states)
        return (
            model_config.DecoderOutput(
                output.last_hidden_states,
                output.hidden_states,
                output.cache,
                logits
            ),
            model_config.DecoderOutput(
                output_reverse.last_hidden_states,
                output_reverse.hidden_states,
                output_reverse.cache,
                logits_reverse
            )
        )

    def forward(
        self,
        tokens: torch.LongTensor,
        tokens_reverse: torch.LongTensor,
        precursor: Precursor,
        mem_hidden_states: torch.FloatTensor,
        mem_attention_mask: Optional[torch.BoolTensor]=None,
        prompt_hidden_states: torch.FloatTensor=None
    ):
        hidden_states = self.word_embedding(
            tokens,
            precursor,
            prompt_hidden_states
        )
        hidden_states_reverse = self.word_embedding(
            tokens_reverse,
            precursor,
            prompt_hidden_states
        )
        
        pos_emb = self.calculate_pos_emb(
            0,
            hidden_states.size(1),
            hidden_states.size(0),
            hidden_states.device
        )

        output: model_config.Output = self.decoder(
            hidden_states,
            pos_emb,
            mem_hidden_states,
            mem_attention_mask
        )
        output_reverse: model_config.Output = self.decoder_reverse(
            hidden_states_reverse,
            pos_emb,
            mem_hidden_states,
            mem_attention_mask
        )

        logits = self.lm_head(output.last_hidden_states)
        logits_reverse = self.lm_head_reverse(output_reverse.last_hidden_states)
        
        n_skip = 0
        if prompt_hidden_states is not None:
            n_skip = prompt_hidden_states.size(1)
            if prompt_hidden_states.dim() == 2:
                n_skip = 1
        
        return (
            model_config.DecoderOutput(
                output.last_hidden_states,
                output.hidden_states,
                output.cache,
                logits[:, n_skip:, :],
            ),
            model_config.DecoderOutput(
                output_reverse.last_hidden_states,
                output_reverse.hidden_states,
                output_reverse.cache,
                logits_reverse[:, n_skip:, :],
            )
        )
