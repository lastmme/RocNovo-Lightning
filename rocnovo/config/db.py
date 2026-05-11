from typing import Literal, get_args
from dataclasses import dataclass, field

from pyteomics.parser import expasy_rules

FIXED_MODS = ["C:C+57.021"]
VAR_MODS = ["M:M+15.995", "N:N+0.984", "Q:Q+0.984", "nterm:+42.011", "nterm:+43.006", "nterm:-17.027", "nterm:+43.006-17.027"] 
DigestionType = Literal["full", "semi", "non-specific"]

@dataclass(frozen=True)
class DigestConfig:
    enzyme: str="trypsin"
    digestion: DigestionType="full"
    missed_cleavages: int=0
    min_peptide_len: int=6
    max_peptide_len: int=100
    max_mods: int=1
    fixed_mods: list[str]=field(default_factory=lambda: list(FIXED_MODS))
    var_mods: list[str]=field(default_factory=lambda: list(VAR_MODS))

    def __post_init__(self):
        if self.enzyme not in expasy_rules:
            raise ValueError(f"Unsupported enzyme: '{self.enzyme}'. Available options: {list(expasy_rules.keys())}")
        
        if self.min_peptide_len > self.max_peptide_len:
            raise ValueError(
                f"Length conflict: min_peptide_len ({self.min_peptide_len}) "
                f"cannot be greater than max_peptide_len ({self.max_peptide_len})."
            )
        
        valid_digestions = get_args(DigestionType)
        if self.digestion not in valid_digestions:
            raise ValueError(
                f"The digestion setting '{self.digestion}' is not supported. "
                f"Must be one of {valid_digestions}."
            )
        
        if not isinstance(self.fixed_mods, list) or not isinstance(self.var_mods, list):
            raise TypeError("Both fixed_mods and var_mods must be of type list.")

@dataclass(frozen=True)
class Peptide:
    protein_id: str
    peptide: str
    is_decoy: bool=False

@dataclass(frozen=True)
class BucketConfig:
    bin_size: int=50_000
    min_mass: float=400.0
    max_mass: float=6_000.0

    def __post_init__(self):
        if self.min_mass > self.max_mass:
            raise ValueError("min_mass must be less than or equal to max_mass.")
@dataclass(frozen=True)
class DecoyConfig:
    decoy_prefix: str="DECOY_"
    generate_decoy: bool=True
    decoy_strategy: Literal["reverse", "shuffle", "fused"]="reverse"