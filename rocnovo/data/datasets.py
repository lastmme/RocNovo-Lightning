import h5py
from pathlib import Path
from functools import lru_cache

from torch.utils.data import Dataset

from rocnovo.common.io import normalize_path
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer
from rocnovo.tokenizer.spectrum import SpectrumTokenizer

class SpectrumStream(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        spectrum_tokenizer: SpectrumTokenizer
    ):
        super().__init__()
        self.h5_path = h5_path
        file_handle = h5py.File(h5_path, "r")
        dataset_handle = file_handle["0"]
        self._n_spectra = dataset_handle.attrs["n_spectra"]
        self._raw_path = dataset_handle.attrs["path"]
        self._n_peaks = dataset_handle.attrs["n_peaks"]
        file_handle.close()

        self.stream_handle = None
        self.spectrum_tokenizer = spectrum_tokenizer

    def __len__(self):
        return self._n_spectra
    
    @property
    def n_spectra(self):
        return self._n_spectra

    @property
    def raw_path(self):
        return self._raw_path
    
    @property
    def n_peaks(self):
        return self._n_peaks
    
    def __getitem__(self, idx: int):
        if self.stream_handle is None:
            self.stream_handle = h5py.File(self.h5_path, "r")["0"]
        
        start_offset = self.stream_handle["metadata"][idx]["offset"]
        precursor_mz = self.stream_handle["metadata"][idx]["precursor_mz"]
        precursor_charge = self.stream_handle["metadata"][idx]["precursor_charge"]
        if idx == self.n_spectra - 1:
            stop_offset = self.n_peaks
        else:
            stop_offset = self.stream_handle["metadata"][idx + 1]["offset"]

        peaks = self.stream_handle["spectra"][start_offset:stop_offset]
        mz_array = peaks["mz_array"]
        int_array = peaks["intensity_array"]
        spectrum = self.spectrum_tokenizer.tokenize(
            mz_array,
            int_array,
            precursor_mz,
            precursor_charge,
        )
        return spectrum, precursor_mz, precursor_charge

class DeNovoStream(SpectrumStream):
    def __init__(
        self,
        h5_path: str | Path,
        spectrum_tokenizer: SpectrumTokenizer,
        peptide_tokenizer: PTMPeptideTokenizer,
    ):
        super().__init__(h5_path, spectrum_tokenizer)
        self.peptide_tokenizer = peptide_tokenizer
    
    def __getitem__(self, idx: int):
        spectrum, precursor_mz, precursor_charge = super().__getitem__(idx)
        peptide = self.stream_handle["annotations"][idx].decode()
        peptide_tokens = self.peptide_tokenizer.tokenize(peptide)
        return spectrum, precursor_mz, precursor_charge, peptide_tokens

class BiDirectDeNovoStream(SpectrumStream):
    def __init__(
        self,
        h5_path: str | Path,
        spectrum_tokenizer: SpectrumTokenizer,
        peptide_tokenizer: PTMPeptideTokenizer,
    ):
        super().__init__(h5_path, spectrum_tokenizer)
        self.peptide_tokenizer = peptide_tokenizer
    
    def __getitem__(self, idx: int):
        spectrum, precursor_mz, precursor_charge = super().__getitem__(idx)
        peptide = self.stream_handle["annotations"][idx].decode()
        peptide_tokens = self.peptide_tokenizer.tokenize(peptide)
        peptide_tokens_reverse = self.peptide_tokenizer.reverse_tokenize(peptide)
        return spectrum, precursor_mz, precursor_charge, peptide_tokens, peptide_tokens_reverse

class PeptideMetadataStream(Dataset):
    def __init__(self, peptide_metadata_file: str | Path):
        super().__init__()
        self.peptide_metadata_file = peptide_metadata_file
        self._h5_file = None 
        with h5py.File(peptide_metadata_file, 'r') as f:
            self.length = f['mass'].shape[0]

    def _get_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.peptide_metadata_file, 'r')
        return self._h5_file

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int):
        h5f = self._get_file()
        return {
            "id": idx + 1,
            "mass": h5f['mass'][idx],
            "modified_peptide": h5f['modified_peptide'][idx].decode(),
            "peptide": h5f['peptide'][idx].decode(),
            "protein_id": h5f['protein_id'][idx],
            "is_decoy": h5f['is_decoy'][idx]
        }

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()

class DBSearchDataset(Dataset):
    def __init__(
        self,
        spectrum_stream: SpectrumStream,
        peptide_tokenizer: PTMPeptideTokenizer,
        stage1_score_file: str | Path
    ):
        super().__init__()
        self.spectrum_stream = spectrum_stream
        self.peptide_tokenizer = peptide_tokenizer
        self.stage1_score_file = normalize_path(stage1_score_file)
        self._h5_file = None
        
        with h5py.File(self.stage1_score_file, 'r') as f:
            self.length = f['spectrum_id'].shape[0]

    def _get_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.stage1_score_file, 'r')
        return self._h5_file

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int):
        h5f = self._get_file()
        real_spectrum_idx = h5f["spectrum_id"][idx]
        modified_peptide = h5f["metadata"][idx]["modified_peptide"].decode()
        
        spectrum, precursor_mz, precursor_charge = self.spectrum_stream[real_spectrum_idx]
        peptide_tokens = self.peptide_tokenizer.tokenize(modified_peptide)
        peptide_tokens_reverse = self.peptide_tokenizer.reverse_tokenize(modified_peptide)
        
        return spectrum, precursor_mz, precursor_charge, peptide_tokens, peptide_tokens_reverse, real_spectrum_idx
    
    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()