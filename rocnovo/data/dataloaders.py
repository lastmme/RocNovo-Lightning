from copy import copy
from pathlib import Path
from typing import Callable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from rocnovo.common.io import normalize_path
from rocnovo.config.data import BidirectTrainBatch, TrainBatch, InferenceBatch, Spectra, Peptide, Precursor, DataConfig, DBSearchBatch
from rocnovo.data.datasets import DeNovoStream, SpectrumStream, BiDirectDeNovoStream, PeptideMetadataStream, DBSearchDataset
from rocnovo.tokenizer.spectrum import SpectrumTokenizer
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SPECIAL_TOKENS, PAD, PROTON

def prepare_spectra(spectra, precursor_mzs, precursor_charges):
    spectra = pad_sequence(spectra, batch_first=True)
    mask = torch.sum(spectra, dim=-1).bool()
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON) * precursor_charges
    return InferenceBatch(
        Spectra(
            spectra[:, :, 0],
            spectra[:, :, 1],
            mask,
            Precursor(
                precursor_masses,
                precursor_charges,
                precursor_mzs
            )
        )
    )

def spectra_collate_fn(batch: list[tuple[torch.Tensor, float, int]]):
    spectra, precursor_mzs, precursor_charges = list(zip(*batch))
    return prepare_spectra(
        spectra,
        precursor_mzs,
        precursor_charges
    )

def denovo_collate_fn(batch: list[tuple[torch.Tensor, float, int, torch.LongTensor]]):
    spectra, precursor_mzs, precursor_charges, peptide_tokens = list(zip(*batch))
    inference_batch = prepare_spectra(spectra, precursor_mzs, precursor_charges)
    peptide_tokens = pad_sequence(peptide_tokens, batch_first=True, padding_value=SPECIAL_TOKENS[PAD])
    peptide_mask = (peptide_tokens != SPECIAL_TOKENS[PAD])
    return TrainBatch(
        inference_batch.spectra,
        Peptide(
            peptide_tokens,
            peptide_mask
        )
    )

def bidirect_denovo_collate_fn(batch: list[tuple[torch.Tensor, float, int, torch.LongTensor, torch.LongTensor]]):
    spectra, precursor_mzs, precursor_charges, peptide_tokens, peptide_tokens_reverse = list(zip(*batch))
    inference_batch = prepare_spectra(spectra, precursor_mzs, precursor_charges)
    peptide_tokens = pad_sequence(peptide_tokens, batch_first=True, padding_value=SPECIAL_TOKENS[PAD])
    peptide_tokens_reverse = pad_sequence(peptide_tokens_reverse, batch_first=True, padding_value=SPECIAL_TOKENS[PAD])
    peptide_mask = (peptide_tokens != SPECIAL_TOKENS[PAD])
    peptide_reverse_mask = (peptide_tokens_reverse != SPECIAL_TOKENS[PAD])
    return BidirectTrainBatch(
        inference_batch.spectra,
        Peptide(
            peptide_tokens,
            peptide_mask
        ),
        Peptide(
            peptide_tokens_reverse,
            peptide_reverse_mask
        )
    )

def dbsearch_collate_fn(batch: list[tuple]):
    base_batch = [item[:5] for item in batch]
    processed_batch = bidirect_denovo_collate_fn(base_batch)
    spectrum_ids = [item[5] for item in batch]
    
    return DBSearchBatch(
        processed_batch.spectra,
        processed_batch.peptide,
        processed_batch.peptide_reverse,
        torch.tensor(spectrum_ids, dtype=torch.long)
    )

class BaseDataModule(LightningDataModule):
    allow_zero_length_dataloader_with_multiple_devices = False
    _log_hyperparams=True
    custom_collatefn: Callable=None
    train_dataset: Dataset=None
    val_dataset: Dataset=None
    test_dataset: Dataset=None
    data_config: DataConfig=None
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.data_config.train_batch_size,
            collate_fn=self.custom_collatefn,
            num_workers=self.data_config.n_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.data_config.val_batch_size,
            collate_fn=self.custom_collatefn,
            num_workers=self.data_config.n_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.data_config.test_batch_size,
            collate_fn=self.custom_collatefn,
            num_workers=self.data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )

class SpectraDataLoaderModule(BaseDataModule):
    def __init__(self, data_config: DataConfig, spectrum_tokenizer: SpectrumTokenizer):
        self.data_config = data_config
        self.custom_collatefn = spectra_collate_fn
        
        self.spectrum_tokenizer = spectrum_tokenizer
        self.spectrum_tokenizer.disable_aug()
        
        if self.data_config.train_path != "" and Path(self.data_config.train_path).exists():
            self.train_dataset = SpectrumStream(
                self.data_config.train_path,
                self.spectrum_tokenizer
            )
        
        if self.data_config.val_path != "" and Path(self.data_config.val_path).exists():
            self.val_dataset = SpectrumStream(
                self.data_config.val_path,
                spectrum_tokenizer
            )
        
        if self.data_config.test_path != "" and Path(self.data_config.test_path).exists():
            self.test_dataset = SpectrumStream(
                self.data_config.test_path,
                spectrum_tokenizer
            )

class DeNovoDataLoaderModule(BaseDataModule):
    def __init__(self, data_config: DataConfig, spectrum_tokenizer: SpectrumTokenizer, peptide_tokenizer: PTMPeptideTokenizer):
        self.data_config = data_config
        self.custom_collatefn = denovo_collate_fn
        
        self.spectrum_tokenizer = spectrum_tokenizer
        spectrum_tokenizer_val = copy(spectrum_tokenizer)
        spectrum_tokenizer_val.disable_aug()
        
        self.peptide_tokenizer = peptide_tokenizer
        if self.data_config.train_path != "" and Path(self.data_config.train_path).exists():
            self.train_dataset = DeNovoStream(
                self.data_config.train_path,
                self.spectrum_tokenizer,
                self.peptide_tokenizer
            )
        
        if self.data_config.val_path != "" and Path(self.data_config.val_path).exists():
            self.val_dataset = DeNovoStream(
                self.data_config.val_path,
                spectrum_tokenizer_val,
                self.peptide_tokenizer
            )
        
        if self.data_config.test_path != "" and Path(self.data_config.test_path).exists():
            self.test_dataset = DeNovoStream(
                self.data_config.test_path,
                spectrum_tokenizer_val,
                self.peptide_tokenizer
            )

class BiDirectDeNovoDataLoaderModule(BaseDataModule):
    def __init__(self, data_config: DataConfig, spectrum_tokenizer: SpectrumTokenizer, peptide_tokenizer: PTMPeptideTokenizer):
        self.data_config = data_config
        self.custom_collatefn = bidirect_denovo_collate_fn
        
        self.spectrum_tokenizer = spectrum_tokenizer
        self.spectrum_tokenizer.disable_aug()
        
        self.peptide_tokenizer = peptide_tokenizer
        if self.data_config.train_path != "" and Path(self.data_config.train_path).exists():
            self.train_dataset = BiDirectDeNovoStream(
                self.data_config.train_path,
                self.spectrum_tokenizer,
                self.peptide_tokenizer
            )
        
        if self.data_config.val_path != "" and Path(self.data_config.val_path).exists():
            self.val_dataset = BiDirectDeNovoStream(
                self.data_config.val_path,
                self.spectrum_tokenizer,
                self.peptide_tokenizer
            )
        
        if self.data_config.test_path != "" and Path(self.data_config.test_path).exists():
            self.test_dataset = BiDirectDeNovoStream(
                self.data_config.test_path,
                self.spectrum_tokenizer,
                self.peptide_tokenizer
            )

class DBSearchDataLoaderModule(BaseDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        spectrum_tokenizer: SpectrumTokenizer,
        peptide_tokenizer: PTMPeptideTokenizer,
        stage1_score_file: Path | str
    ):
        self.data_config = data_config
        self.custom_collatefn = dbsearch_collate_fn
        
        self.spectrum_tokenizer = spectrum_tokenizer
        self.spectrum_tokenizer.disable_aug()
        self.peptide_tokenizer = peptide_tokenizer
        self.stage1_score_file = normalize_path(stage1_score_file)
        
        self.test_dataset = DBSearchDataset(
            SpectrumStream(
                data_config.test_path,
                spectrum_tokenizer
            ),
            self.peptide_tokenizer,
            self.stage1_score_file
        )
    
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass