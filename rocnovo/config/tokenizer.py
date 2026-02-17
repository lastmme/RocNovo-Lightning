from dataclasses import dataclass

@dataclass
class SpectrumTokenizerConfig:
    n_top_peaks: int=300
    min_mz: float=50.0
    max_mz: float=4500.0
    min_intensity: float=0.01
    remove_precursor_tol: float=2.0

@dataclass
class PTMTokenizerConfig:
    max_len: int=128
    reverse: bool=True
    residues: str="massivekb"