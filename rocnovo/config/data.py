from dataclasses import dataclass, replace

import torch
import einops

@dataclass
class SpectrumItem:
    spectrum: torch.Tensor
    precursor_mz: float
    precursor_charge: int
    scan_id: str

@dataclass
class DeNovoItem(SpectrumItem):
    peptide_tokens: torch.LongTensor

@dataclass
class BiDirectDeNovoItem(SpectrumItem):
    peptide_tokens: torch.LongTensor
    peptide_tokens_reverse: torch.LongTensor

@dataclass
class DBSearchItem(BiDirectDeNovoItem):
    real_spectrum_idx: int

@dataclass(frozen=True)
class DataConfig:
    train_path: str=""
    val_path: str=""
    test_path: str=""
    train_batch_size: int=0
    val_batch_size: int=0
    test_batch_size: int=0
    n_workers: int=0

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
            mz=self.mz.to(device)
        )
    
    def repeat_beamsize(self, S: int):
        return replace(
            self,
            mass=einops.repeat(self.mass, "B -> (B S)", S=S),
            charge=einops.repeat(self.charge, "B -> (B S)", S=S),
            mz=einops.repeat(self.mz, "B -> (B S)", S=S),
        )

    def filter_by_mask(self, mask: torch.BoolTensor):
        return replace(
            self,
            mass=self.mass[mask],
            charge=self.charge[mask],
            mz=self.mz[mask],
        )

    def __getitem__(self, item):
        return replace(
            self,
            mass=self.mass[item],
            charge=self.charge[item],
            mz=self.mz[item],
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
    scan_id: list[str] | None = None

    def to(self, device: torch.device):
        return replace(
            self,
            spectra=self.spectra.to(device),
            peptide=self.peptide.to(device),
            peptide_reverse=self.peptide_reverse.to(device)
        )

@dataclass(frozen=True)
class DBSearchBatch(TrainBatch):
    peptide_reverse: Peptide
    spectrum_id: torch.LongTensor
    scan_id: list[str] | None = None

    def to(self, device: torch.device):
        return replace(
            self,
            spectra=self.spectra.to(device),
            peptide=self.peptide.to(device),
            peptide_reverse=self.peptide_reverse.to(device),
            spectrum_id=self.spectrum_id.to(device)
        )

@dataclass(frozen=True)
class InferenceBatch:
    spectra: Spectra
    scan_id: list[str] | None = None

    def to(self, device: torch.device):
        return replace(
            self,
            spectra=self.spectra.to(device)
        )