import csv
from pathlib import Path
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
    use_cross_cache: bool=True
    use_self_cache: bool=True

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
    scan_id: list[str]

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

    @staticmethod
    def stream_batch_to_csv(
        path: str | Path,
        batch_result: "DenovoResult | EvalResult",
        resolved_preds: ResolvedPrediction,
        mode: Literal["denovo", "eval"],
        write_header: bool = False,
    ):
        """Stream one batch of predictions to a CSV file.

        The row layout mirrors ``post_process``: resolved prediction columns
        come first, followed by the raw forward/reverse outputs and precursor
        metadata. ``eval`` mode additionally writes ``gt_peptide``.
        """
        path = Path(path)
        batch_size = len(resolved_preds.pred_peptide)
        is_eval = mode == "eval"

        base_fieldnames = [
            "scan_id",
            "pred_mz",
            "pred_peptide",
            "pred_score",
            "direction",
            "fwd_score",
            "fwd_peptide",
            "rev_score",
            "rev_peptide",
            "precursor_mz",
            "charge",
        ]
        fieldnames = base_fieldnames + (["gt_peptide"] if is_eval else [])

        file_mode = "w" if write_header else "a"
        with open(path, file_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for i in range(batch_size):
                row = {
                    "scan_id": batch_result.scan_id[i],
                    "pred_mz": float(resolved_preds.pred_mz[i]),
                    "pred_peptide": resolved_preds.pred_peptide[i],
                    "pred_score": float(resolved_preds.pred_score[i]),
                    "direction": resolved_preds.direction[i],
                    "fwd_score": float(batch_result.fwd_score[i]),
                    "fwd_peptide": batch_result.fwd_peptide[i],
                    "rev_score": float(batch_result.rev_score[i]),
                    "rev_peptide": batch_result.rev_peptide[i],
                    "precursor_mz": float(batch_result.precursor_mz[i]),
                    "charge": int(batch_result.charge[i]),
                }
                if is_eval:
                    row["gt_peptide"] = batch_result.gt_peptide[i]
                
                writer.writerow(row)
