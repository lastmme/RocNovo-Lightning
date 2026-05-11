from typing import Literal
from dataclasses import dataclass, fields

import numpy.typing as npt
import pandas as pd

@dataclass(frozen=True)
class InferenceConfig:
    num_beams: int=10 # 0 means greedy search
    min_len: int=6
    max_len: int=80
    max_isotope: int=1
    mass_tolerance: float=50.0
    gradscaling_enabled: bool=True

@dataclass(frozen=True)
class RawSearchResult:
    fwd_score: npt.NDArray
    fwd_peptide: list[str]
    rev_score: npt.NDArray
    rev_peptide: list[str]

@dataclass(frozen=True)
class BaseDenovoResult(RawSearchResult):
    precursor_mz: npt.NDArray
    charge: npt.NDArray

@dataclass(frozen=True)
class BaseEvalResult(BaseDenovoResult):
    gt_peptide: list[str]

@dataclass(frozen=True)
class TheoreticalMz:
    fwd_mz: npt.NDArray
    rev_mz: npt.NDArray

@dataclass(frozen=True)
class ResolvedPrediction:
    pred_mz: npt.NDArray
    pred_peptide: list[str]
    pred_score: npt.NDArray
    direction: list[Literal["fwd", "rev"]]

@dataclass(frozen=True)
class SearchResult(RawSearchResult, TheoreticalMz):
    pass

@dataclass(frozen=True)
class DenovoResult(BaseDenovoResult, TheoreticalMz):
    pass

@dataclass(frozen=True)
class EvalResult(BaseEvalResult, TheoreticalMz):
    pass

class ExportMixin:
    def to_dataframe(self) -> pd.DataFrame:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return pd.DataFrame(data)

    def to_csv(self, path: str, index: bool=False, **kwargs):
        df = self.to_dataframe()
        df.to_csv(path, index=index, **kwargs)

@dataclass(frozen=True)
class DenovoGlobalResult(ExportMixin, BaseDenovoResult, ResolvedPrediction):
    pass

@dataclass(frozen=True)
class EvalGlobalResult(ExportMixin, BaseEvalResult, ResolvedPrediction):
    pass