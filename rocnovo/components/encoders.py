from typing import Union, Iterable, Literal

import torch
import torch.nn as nn
import einops

from rocnovo.config.data import Spectra
import rocnovo.config.model as model_config
from rocnovo.components.float_encoder import FloatEncoder
from rocnovo.components.transformers import get_activations, Encoder, PeakRotaryPositionalEmbeddings

class PositionalEncoder(FloatEncoder):
    """The positional encoder for sequences.

    Parameters
    ----------
    hidden_size : int
        The number of features to output.
    min_wavelength : float, optional
        The shortest wavelength in the geometric progression.
    max_wavelength : float, optional
        The longest wavelength in the geometric progression.
    """

    def __init__(self,
        hidden_size: int,
        min_wavelength: float=1,
        max_wavelength: float=10000
    ):
        """Initialize the MzEncoder"""
        super().__init__(
            hidden_size=hidden_size,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(self, X: torch.FloatTensor):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X

class ClipPeakEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mz_encoder = FloatEncoder(
            hidden_size,
            0.001,
            10000
        )
        self.intensity_encoder = torch.nn.Linear(1, hidden_size, bias=False)

    def forward(self, mz: torch.FloatTensor, intensity: torch.FloatTensor):
        if intensity.dim() == 2:
            intensity = intensity.unsqueeze(2)
        
        encoded = self.mz_encoder(mz)
        # print(f"clip_peak_encoder.encoded: {encoded.dtype}")
        intensity = self.intensity_encoder(intensity)
        return encoded + intensity

class MultiFeedForwardModule(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, Iterable[int]],
        output_size: int,
        *,
        activation: Literal["relu", "selu", "gelu"]="gelu",
        dropout: float=0.1,
        dropout_last_layer: bool=True,
    ):
        super(MultiFeedForwardModule, self).__init__()
        self._activation = get_activations(activation, hidden_size=hidden_size)

        if not hasattr(hidden_size, "__iter__"):
            if hidden_size is None:
                hidden_size = [output_size]
            else:
                hidden_size = [hidden_size]

        self._layers = []
        layer_dims = [input_size] + hidden_size + [output_size]

        for i in range(1, len(layer_dims) - 1):
            self._layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            self._layers.append(self._activation)
            self._layers.append(nn.Dropout(dropout))

        self._layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        if dropout_last_layer:
            self._layers.append(nn.Dropout(dropout))
        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return self._layers(x)

class ClipSpectrumEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int=128,
        n_head: int=8,
        dim_feedforward: int=1024,
        n_layers: int=1,
        dropout: float=0.1
    ):
        super().__init__()
        self.peak_encoder = ClipPeakEncoder(hidden_size)
        self.precursor_encoder = FloatEncoder(
            hidden_size,
            0.001,
            10000
        )
        self.precursor_projection = MultiFeedForwardModule(
            hidden_size,
            dim_feedforward,
            hidden_size,
            dropout=dropout,
            activation="gelu"
        )

        # transformer encoder
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
            norm=nn.LayerNorm(hidden_size),
        )

    def forward(self, spectra: Spectra):
        device = spectra.mz.device
        batch_size = spectra.mz.shape[0]
        peak_repr = self.peak_encoder(spectra.mz, spectra.intensity)  # (B, L, D)
        
        precursors = spectra.precursor
        precursor_mz = precursors.mz.unsqueeze(-1)
        precursor_repr = self.precursor_encoder(precursor_mz)  # (B, 1, D)
        precursor_repr = self.precursor_projection(precursor_repr)  # (B, 1, D)
        
        precursor_mask = torch.zeros((batch_size, 1), dtype=bool).to(device)  # (B, 1)
        mask = torch.cat([precursor_mask, ~spectra.mask], dim=1)  # (B, L + 1)
        spectra_repr = torch.cat([precursor_repr, peak_repr], dim=1)  # (B, L + 1, D)
        spectra_repr = self.transformer_encoder(spectra_repr, src_key_padding_mask=mask)  # (B, L + 1, D)
        return spectra_repr, mask

class RoPEPeakEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        """Initialize the MzEncoder"""
        super().__init__()
        self.mz_encoder = FloatEncoder(
            hidden_size // 2,
            min_wavelength=0.001,
            max_wavelength=10000,
        )

        self.intensity_encoder = FloatEncoder(
            hidden_size - hidden_size // 2,
            min_wavelength=1e-6,
            max_wavelength=1
        )

    def forward(self, mz: torch.FloatTensor, intensity: torch.FloatTensor):
        encoded_mz = self.mz_encoder(mz)  # (B, L, D/2)
        encoded_intensity = self.intensity_encoder(intensity)  # (B, L, D/2)
        encoded = torch.cat([encoded_mz, encoded_intensity], dim=-1)
        return encoded

class RoPESpectrumEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_layers: int,
        dropout: float,
        dim_feedforward: int,
        activation: Literal["relu", "selu", "gelu", "silu", "swilu"]="swilu",
    ):
        super().__init__()
        self.peak_encoder = RoPEPeakEncoder(hidden_size)
        self.precursor_encoder = FloatEncoder(
            hidden_size,
            0.001,
            10000
        )
        self.precursor_projection = MultiFeedForwardModule(
            hidden_size,
            dim_feedforward,
            hidden_size,
            dropout=dropout,
            activation=activation
        )
        self.d_k = hidden_size // n_head
        self.rotary_emb = PeakRotaryPositionalEmbeddings(
            self.d_k,
            0.001,
            10000
        )
        self.encoder = Encoder(
            n_head,
            n_layers,
            hidden_size,
            dim_feedforward,
            dropout,
            activation
        )
    
    def forward(self, spectra: Spectra):
        device = spectra.mz.device
        batch_size = spectra.mz.shape[0]
        peak_repr = self.peak_encoder(spectra.mz, spectra.intensity)  # (B, L, D)
        
        precursors = spectra.precursor
        precursor_mz = precursors.mz.unsqueeze(-1)
        precursor_repr = self.precursor_encoder(precursor_mz)  # (B, 1, D)
        precursor_repr = self.precursor_projection(precursor_repr)  # (B, 1, D)
        
        precursor_mask = torch.ones((batch_size, 1), dtype=bool).to(device)  # (B, 1)
        mask = torch.cat([precursor_mask, spectra.mask], dim=1)  # (B, L + 1)
        spectra_repr = torch.cat([precursor_repr, peak_repr], dim=1)  # (B, L + 1, D)

        mz_positions = spectra.mz  # (B, L)
        precursor_positions = torch.zeros(batch_size, 1, device=device)
        positions = torch.cat([precursor_positions, mz_positions], dim=1)
        pos_emb = self.rotary_emb(positions)
        output: model_config.Output = self.encoder(
            spectra_repr,
            pos_emb,
            mask
        )
        return output.last_hidden_states, mask