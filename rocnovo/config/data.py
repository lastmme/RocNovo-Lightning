import os
from dataclasses import dataclass, replace

import torch

@dataclass(frozen=True)
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    n_workers: int

@dataclass(frozen=True)
class Precursor:
    mass: torch.FloatTensor
    charge: torch.IntTensor
    mz: torch.FloatTensor
    
    def to(self, device: torch.device):
        return replace(
            self,
            mass=self.mass.to(device),
            charge=self.charge.to(device),
            mz=self.mz.to(device),
        )

@dataclass(frozen=True)
class Spectra:
    mz: torch.FloatTensor
    intensity: torch.FloatTensor
    mask: torch.BoolTensor
    precursor: Precursor
    
    def to(self, device: torch.device):
        return replace(
            self,
            mz=self.mz.to(device),
            intensity=self.intensity.to(device),
            mask=self.mask.to(device),
            precursor=self.precursor.to(device),
        )

@dataclass(frozen=True)
class Peptide:
    tokens: torch.LongTensor
    mask: torch.BoolTensor
    def to(self, device: torch.device):
        return replace(
            self,
            tokens=self.tokens.to(device),
            mask=self.mask.to(device)
        )

@dataclass(frozen=True)
class TrainBatch:
    spectra: Spectra
    peptide: Peptide
    def to(self, device: torch.device):
        return replace(
            self,
            spectra=self.spectra.to(device),
            peptide=self.peptide.to(device)
        )

@dataclass(frozen=True)
class BidirectTrainBatch(TrainBatch):
    peptide_reverse: Peptide

    def to(self, device: torch.device):
        return replace(
            self,
            spectra=self.spectra.to(device),
            peptide=self.peptide.to(device),
            peptide_reverse=self.peptide_reverse.to(device)
        )

@dataclass(frozen=True)
class InferenceBatch:
    spectra: Spectra
    def to(self, device: torch.device):
        return replace(
            self,
            spectra=self.spectra.to(device)
        )