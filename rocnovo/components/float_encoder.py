import math

import numpy as np
import torch
import torch.nn as nn

class FloatEncoder(nn.Module):
    """Encode floating point values using sine and cosine waves.

    Parameters
    ----------
    hidden_size : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(
        self,
        hidden_size,
        min_wavelength: float=0.001,
        max_wavelength: float=10000
    ):
        """Initialize the MassEncoder"""
        super().__init__()

        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        # Get dimensions for equations:
        d_sin = math.ceil(hidden_size / 2)
        d_cos = hidden_size - d_sin

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, hidden_size).float() - d_sin) / (d_cos - 1)
        sin_term = base * (scale ** sin_exp)
        cos_term = base * (scale ** cos_exp)

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X: torch.FloatTensor):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_masses, hidden_size)
            The encoded features for the mass spectra.
        """
        if X.dim() == 2:
            X = X[:, :, None]
        # print(f"float_encoder.forward: {X.dtype}")
        # print(f"float_encoder.forward: {self.sin_term.dtype}")
        # print(f"float_encoder.forward: {self.cos_term.dtype}")

        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)