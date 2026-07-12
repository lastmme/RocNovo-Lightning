from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from rocnovo.common.logger import logger
from rocnovo.config.data import Peptide, DBSearchBatch, DataConfig
from rocnovo.config.model import DecoderOutput
from rocnovo.config.db import DigestConfig, BucketConfig, DecoyConfig, ExecutionConfig
from rocnovo.config.inference import InferenceConfig
from rocnovo.data.dataloaders import (
    DBSearchDataLoaderModule,
    prepare_spectra,
)
from rocnovo.data.datasets import SpectrumStream
from rocnovo.tokenizer.peptide import (
    PTMPeptideTokenizer,
    PAD,
    SPECIAL_TOKENS,
    PROTON,
    ISOTOPE,
)
from rocnovo.module.denovo import Denovo
from rocnovo.common.io import normalize_path
from rocnovo.common.digest import digest

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _numba_filter_candidates(
    candidate_masses: npt.NDArray[np.float32],
    target_masses: npt.NDArray[np.float32],
    tolerances: npt.NDArray[np.float32]
):
    n = candidate_masses.shape[0]
    m = target_masses.shape[0]

    valid_mask = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        mass = candidate_masses[i]
        for j in range(m):
            diff = abs(mass - target_masses[j])
            if diff <= tolerances[j]:
                valid_mask[i] = True
                break
    
    return valid_mask

@dataclass
class CandidateSet:
    """A contiguous set of candidate peptides for one or more spectra."""
    candidate_ids: npt.NDArray[np.int64]
    modified_peptides: npt.NDArray[np.object_]
    peptides: npt.NDArray[np.object_]
    protein_ids: npt.NDArray[np.object_]
    is_decoys: npt.NDArray[np.bool_]

    def __len__(self):
        return self.candidate_ids.shape[0]

    def is_empty(self):
        return self.candidate_ids.shape[0] == 0

@dataclass
class PSMRecord:
    """Best PSM record for a single spectrum."""
    spectrum_id: int
    score: float
    metadata: npt.NDArray
    """
        self.meta_dtype = np.dtype([
            ("modified_peptide", dt_str),
            ("peptide", dt_str),
            ("protein_id", dt_str),
            ("is_decoy", np.bool_)
        ])
    """

class HDF5Stage1Buffer:
    def __init__(self, h5_file: h5py.File, capacity: int=100_000):
        self.capacity = capacity
        self.size = 0
        self.global_offset = 0
        self.h5_file = h5_file
        self.spectrum_ids = np.zeros(capacity, dtype=np.int64)
        self.modified_peptides = np.empty(capacity, dtype=object)
        self.peptides = np.empty(capacity, dtype=object)
        self.protein_ids = np.empty(capacity, dtype=object)
        self.is_decoys = np.zeros(capacity, dtype=bool)
        self.scores = np.zeros(capacity, dtype=np.float32)
        chunk_size = (capacity,)
        dt_str = h5py.string_dtype(encoding="utf-8")
        self.meta_dtype = np.dtype(
            [
                ("modified_peptide", dt_str),
                ("peptide", dt_str),
                ("protein_id", dt_str),
                ("is_decoy", np.bool_),
            ]
        )
        self.ds_spec = h5_file.create_dataset(
            "spectrum_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=chunk_size,
        )
        self.ds_score = h5_file.create_dataset(
            "score",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=chunk_size,
        )
        self.ds_metadata = h5_file.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=self.meta_dtype,
            chunks=chunk_size,
        )

    def extend(
        self,
        spectrum_ids: npt.NDArray[np.int64],
        modified_peptides: npt.NDArray[np.object_],
        peptides: npt.NDArray[np.object_],
        protein_ids: npt.NDArray[np.object_],
        is_decoys: npt.NDArray[np.bool_],
        scores: npt.NDArray[np.float32],
    ):
        n = len(spectrum_ids)
        if self.size + n > self.capacity:
            self.flush()

        if n > self.capacity:
            self._write_direct(
                spectrum_ids,
                modified_peptides,
                peptides,
                protein_ids,
                is_decoys,
                scores,
            )
            return

        self.spectrum_ids[self.size : self.size + n] = spectrum_ids
        self.modified_peptides[self.size : self.size + n] = modified_peptides
        self.peptides[self.size : self.size + n] = peptides
        self.protein_ids[self.size : self.size + n] = protein_ids
        self.is_decoys[self.size : self.size + n] = is_decoys
        self.scores[self.size : self.size + n] = scores
        self.size += n

        if self.size >= self.capacity:
            self.flush()

    def _write_direct(
        self,
        spec_ids: npt.NDArray[np.int64],
        mod_peps: npt.NDArray[np.object_],
        peps: npt.NDArray[np.object_],
        prot_ids: npt.NDArray[np.object_],
        is_decoys: npt.NDArray[np.bool_],
        scs: npt.NDArray[np.float32],
    ):
        n = len(spec_ids)
        new_size = self.global_offset + n

        self.ds_spec.resize((new_size,))
        self.ds_score.resize((new_size,))
        self.ds_metadata.resize((new_size,))

        self.ds_spec[self.global_offset : new_size] = spec_ids
        self.ds_score[self.global_offset : new_size] = scs
        temp_struct = np.empty(n, dtype=self.meta_dtype)
        temp_struct["modified_peptide"] = mod_peps
        temp_struct["peptide"] = peps
        temp_struct["protein_id"] = prot_ids
        temp_struct["is_decoy"] = is_decoys
        self.ds_metadata[self.global_offset : new_size] = temp_struct
        self.global_offset = new_size

    def flush(self):
        if self.size == 0:
            return

        self._write_direct(
            self.spectrum_ids[: self.size],
            self.modified_peptides[: self.size],
            self.peptides[: self.size],
            self.protein_ids[: self.size],
            self.is_decoys[: self.size],
            self.scores[: self.size],
        )
        self.clear()

    def clear(self):
        self.size = 0
        self.modified_peptides.fill(None)
        self.peptides.fill(None)
        self.protein_ids.fill(None)

    def __len__(self):
        return self.size

class HDF5Stage2Buffer:
    def __init__(self, capacity: int=100_000):
        self.capacity = capacity
        self.size = 0
        self.scores = np.zeros(capacity, dtype=np.float32)
        self.spectrum_ids = np.zeros(capacity, dtype=np.int64)

    def extend(self, scores: npt.NDArray[np.float32], spectrum_ids: np.ndarray[np.int64]):
        n = len(scores)
        self.scores[self.size : self.size + n] = scores
        self.spectrum_ids[self.size : self.size + n] = spectrum_ids
        self.size += n

    def flush_to_hdf5(self, h5_file: h5py.File, start_offset: int):
        if self.size == 0:
            return 0

        end_offset = start_offset + self.size
        h5_file["score"][start_offset:end_offset] = self.scores[: self.size]
        h5_file["spectrum_id"][start_offset:end_offset] = self.spectrum_ids[: self.size]
        
        written_size = self.size
        self.size = 0
        return written_size

class MassIndexDB:
    """Light-weight mass index that can be forked to worker processes."""
    def __init__(self, h5_file: str | Path | h5py.File):
        if isinstance(h5_file, (str, Path)):
            with h5py.File(h5_file, "r") as f:
                self._load(f)
        else:
            self._load(h5_file)

    def _load(self, h5_file: h5py.File):
        self.attrs = {
            "rocnovo_min_mass": h5_file.attrs["rocnovo_min_mass"],
            "rocnovo_bin_width": h5_file.attrs["rocnovo_bin_width"],
        }
        self.bucket_bounds = h5_file["bucket_bounds"][:]

    def __getitem__(self, key: str):
        if key == "bucket_bounds":
            return self.bucket_bounds
        
        raise KeyError(key)

class CandidateSearcher:
    @staticmethod
    def _target_mzs_and_masses(
        precursor_mz: float,
        precursor_charge: int,
        max_isotope: int,
    ):
        """Compute target m/z values and corresponding neutral masses.

        Isotope shifts are expressed in m/z domain using the precursor charge.
        """
        target_mzs = precursor_mz - np.arange(max_isotope + 1) * ISOTOPE / precursor_charge
        target_masses = (target_mzs - PROTON) * precursor_charge
        return target_mzs, target_masses

    @staticmethod
    def get_candidate_ids(
        precursor_mz: float,
        precursor_charge: int,
        is_ppm: bool,
        mass_tolerance: float,
        max_isotope: int,
        mass_index: MassIndexDB,
        mass_data: npt.NDArray[np.float64],
    ):
        """Return candidate row IDs whose precursor m/z matches within tolerance.

        ppm tolerance is computed in the m/z domain using the precursor charge,
        i.e. ``|candidate_mz - target_mz| <= target_mz * ppm * 1e-6``.
        Only the pre-loaded mass array is touched; no peptide metadata is read.
        """
        if precursor_charge <= 0:
            return np.array([], dtype=np.int64)

        min_mass = mass_index.attrs["rocnovo_min_mass"]
        bin_width = mass_index.attrs["rocnovo_bin_width"]
        bucket_bounds = mass_index["bucket_bounds"]
        num_buckets = bucket_bounds.shape[0]

        target_mzs, target_masses = CandidateSearcher._target_mzs_and_masses(
            precursor_mz, precursor_charge, max_isotope
        )
        b_indices = ((target_masses - min_mass) // bin_width).astype(np.int64)
        offsets = np.array([-1, 0, 1], dtype=np.int64)
        all_b_indices = np.ravel(b_indices[:, None] + offsets)
        valid_b_mask = (all_b_indices >= 0) & (all_b_indices < num_buckets)

        target_buckets = np.unique(all_b_indices[valid_b_mask])
        if target_buckets.size == 0:
            return np.array([], dtype=np.int64)

        bounds = bucket_bounds[target_buckets]
        valid_bounds = bounds[(bounds[:, 0] != -1) & (bounds[:, 1] != -1)]
        if valid_bounds.size == 0:
            return np.array([], dtype=np.int64)

        merged_slices = []
        cur_start, cur_end = valid_bounds[0]
        for start, end in valid_bounds[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged_slices.append((cur_start, cur_end))
                cur_start, cur_end = start, end

        merged_slices.append((cur_start, cur_end))

        chunks = []
        for start, end in merged_slices:
            chunks.append(
                (
                    np.arange(start, end, dtype=np.int64),
                    np.asarray(mass_data[start:end]),
                )
            )

        if len(chunks) == 1:
            c_ids, c_masses = chunks[0]
        else:
            c_ids = np.concatenate([c[0] for c in chunks])
            c_masses = np.concatenate([c[1] for c in chunks])

        candidate_mzs = c_masses / precursor_charge + PROTON

        if is_ppm:
            tolerances = target_mzs * mass_tolerance * 1e-6
        else:
            tolerances = np.full_like(target_mzs, mass_tolerance)

        valid_mask = _numba_filter_candidates(
            candidate_mzs,
            target_mzs,
            tolerances,
        )

        return c_ids[valid_mask]

class CandidateIDHDF5Buffer:
    """Per-worker buffer for writing (spectrum_id, candidate_id) pairs into a temporary HDF5 file.

    Only the two integer identifiers are persisted; peptide metadata is fetched
    on demand from ``peptides_metadata.hdf5`` during later stages. This keeps the
    temporary candidate files small and avoids duplicating large string columns.
    """

    def __init__(self, h5_file: h5py.File, capacity: int=100_000):
        self.capacity = capacity
        self.size = 0
        self.global_offset = 0
        self.h5_file = h5_file
        self.spectrum_ids = np.zeros(capacity, dtype=np.int64)
        self.candidate_ids = np.zeros(capacity, dtype=np.int64)

        chunk_size = (capacity,)
        self.ds_spec = h5_file.create_dataset(
            "spectrum_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=chunk_size,
        )
        self.ds_cand = h5_file.create_dataset(
            "candidate_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=chunk_size,
        )

    def extend(
        self,
        spectrum_ids: npt.NDArray[np.int64],
        candidate_ids: npt.NDArray[np.int64],
    ):
        n = len(spectrum_ids)
        if n == 0:
            return

        if self.size + n > self.capacity:
            self.flush()

        if n > self.capacity:
            self._write_direct(spectrum_ids, candidate_ids)
            return

        end = self.size + n
        self.spectrum_ids[self.size : end] = spectrum_ids
        self.candidate_ids[self.size : end] = candidate_ids
        self.size = end

        if self.size >= self.capacity:
            self.flush()

    def _write_direct(
        self,
        spectrum_ids: npt.NDArray[np.int64],
        candidate_ids: npt.NDArray[np.int64],
    ):
        n = len(spectrum_ids)
        new_size = self.global_offset + n

        self.ds_spec.resize((new_size,))
        self.ds_cand.resize((new_size,))

        self.ds_spec[self.global_offset : new_size] = spectrum_ids
        self.ds_cand[self.global_offset : new_size] = candidate_ids
        self.global_offset = new_size

    def flush(self):
        if self.size == 0:
            return

        self._write_direct(
            self.spectrum_ids[: self.size],
            self.candidate_ids[: self.size],
        )
        self.clear()

    def clear(self):
        self.size = 0

class CandidateMerger:
    """Merge per-worker candidate HDF5 files into one contiguous file.

    The merged file preserves the property that rows belonging to the same
    spectrum_id are stored contiguously.
    """

    @staticmethod
    def merge(tmp_dir: Path, output_file: Path):
        tmp_dir = normalize_path(tmp_dir)
        output_file = normalize_path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(tmp_dir.glob("candidates_*.hdf5"))
        if not chunk_files:
            raise RuntimeError(f"No candidate chunk files found in {tmp_dir}")

        # Skip empty chunks and sort the rest by the first spectrum_id so that
        # the merge keeps each spectrum's candidates contiguous.
        def _first_spectrum_id(path: Path):
            with h5py.File(path, "r") as f:
                if f["spectrum_id"].shape[0] == 0:
                    return -1

                return int(f["spectrum_id"][0])

        chunk_files = [p for p in chunk_files if _first_spectrum_id(p) >= 0]
        chunk_files.sort(key=_first_spectrum_id)

        if not chunk_files:
            raise RuntimeError(f"All candidate chunk files in {tmp_dir} are empty")

        total_rows = 0
        file_lengths = []
        for path in chunk_files:
            with h5py.File(path, "r") as f:
                n = f["spectrum_id"].shape[0]
                file_lengths.append(n)
                total_rows += n

        chunk_size = (min(100_000, max(total_rows, 1)),)

        with h5py.File(output_file, "w") as out:
            out.create_dataset(
                "spectrum_id",
                shape=(total_rows,),
                dtype=np.int64,
                chunks=chunk_size,
            )
            out.create_dataset(
                "candidate_id",
                shape=(total_rows,),
                dtype=np.int64,
                chunks=chunk_size,
            )
            out.attrs["n_rows"] = total_rows

            offset = 0
            for path, n in tqdm(
                zip(chunk_files, file_lengths),
                total=len(chunk_files),
                desc="Merging candidate files",
            ):
                with h5py.File(path, "r") as src:
                    out["spectrum_id"][offset : offset + n] = src["spectrum_id"][:]
                    out["candidate_id"][offset : offset + n] = src["candidate_id"][:]
                
                offset += n

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return output_file

def _split_ranges(n: int, n_workers: int):
    if n_workers <= 0:
        n_workers = 1
    
    n_workers = min(n_workers, max(1, n))
    chunk_size = n // n_workers
    remainder = n % n_workers
    ranges = []
    start = 0
    for i in range(n_workers):
        end = start + chunk_size + (1 if i < remainder else 0)
        ranges.append((start, end))
        start = end
    
    return ranges

def _generate_candidates_worker(
    worker_id: int,
    spec_start: int,
    spec_end: int,
    spectra_file: Path,
    peptide_metadata_file: Path,
    mass_mmap_file: Path,
    mass_tolerance: float,
    max_isotope: int,
    tmp_dir: Path,
    flush_batch_size: int,
):
    spectra_file = normalize_path(spectra_file)
    peptide_metadata_file = normalize_path(peptide_metadata_file)
    mass_mmap_file = normalize_path(mass_mmap_file)
    tmp_dir = normalize_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = tmp_dir.joinpath(f"candidates_{worker_id:05d}.hdf5")

    mass_index = MassIndexDB(peptide_metadata_file)
    # Memory-map the shared mass array instead of loading it into each worker.
    mass_data = np.load(mass_mmap_file, mmap_mode="r")

    with h5py.File(spectra_file, "r") as spec_h5, h5py.File(
        chunk_path, "w"
    ) as out_h5:
        spec_meta = spec_h5["0"]["metadata"]
        buffer = CandidateIDHDF5Buffer(out_h5, capacity=flush_batch_size)

        for spec_id in range(spec_start, spec_end):
            precursor_mz = float(spec_meta[spec_id]["precursor_mz"])
            precursor_charge = int(spec_meta[spec_id]["precursor_charge"])

            candidate_ids = CandidateSearcher.get_candidate_ids(
                precursor_mz,
                precursor_charge,
                True,
                mass_tolerance,
                max_isotope,
                mass_index,
                mass_data,
            )
            if candidate_ids.size == 0:
                continue

            buffer.extend(
                np.full(candidate_ids.size, spec_id, dtype=np.int64),
                candidate_ids,
            )

        buffer.flush()

    return chunk_path

class CandidateGenerator:
    def __init__(
        self,
        peptide_metadata_file: str | Path,
        output_dir: str | Path,
        mass_tolerance: float,
        max_isotope: int,
        n_workers: int | None=None,
        flush_batch_size: int=100_000,
        progress_bar: bool=True,
        overwrite: bool=False,
    ):
        self.peptide_metadata_file = normalize_path(peptide_metadata_file)
        self.output_dir = normalize_path(output_dir)
        self.mass_tolerance = mass_tolerance
        self.max_isotope = max_isotope
        self.n_workers = n_workers if n_workers is not None else max(1, os.cpu_count() - 2)
        self.flush_batch_size = flush_batch_size
        self.progress_bar = progress_bar
        self.overwrite = overwrite

    def generate(self, spectra_file: str | Path):
        spectra_file = normalize_path(spectra_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.output_dir.joinpath("stage1_candidates.hdf5")
        tmp_dir = self.output_dir.joinpath("tmp_candidates")
        mass_mmap_file = self.output_dir.joinpath("peptide_mass.mmap.npy")

        if output_file.exists() and not self.overwrite:
            logger.warning(f"Candidate file {output_file} already exists, skip.")
            return output_file

        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Pre-extract the peptide mass array into a memory-mapped file that can be
        # shared across candidate-generation workers without duplicating the full
        # peptide metadata in every process.
        if not mass_mmap_file.exists() or self.overwrite:
            logger.info(f"Building shared mass mmap at {mass_mmap_file}...")
            with h5py.File(self.peptide_metadata_file, "r") as f:
                masses = f["metadata"]["mass"][:]
            
            mass_mmap = np.lib.format.open_memmap(
                mass_mmap_file,
                mode="w+",
                dtype=np.float64,
                shape=masses.shape,
            )
            mass_mmap[:] = masses
            del mass_mmap
            logger.info("Shared mass mmap ready.")

        with h5py.File(spectra_file, "r") as spec_h5:
            n_spectra = int(spec_h5["0"].attrs["n_spectra"])

        ranges = _split_ranges(n_spectra, self.n_workers)
        worker_args = [
            (
                wid,
                start,
                end,
                spectra_file,
                self.peptide_metadata_file,
                mass_mmap_file,
                self.mass_tolerance,
                self.max_isotope,
                tmp_dir,
                self.flush_batch_size,
            )
            for wid, (start, end) in enumerate(ranges)
        ]

        logger.info(
            f"Generating candidates for {n_spectra} spectra using {len(ranges)} workers..."
        )
        with ProcessPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [executor.submit(_generate_candidates_worker, *args) for args in worker_args]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Candidate generation workers",
                disable=not self.progress_bar,
            ):
                future.result()

        logger.info(f"Merging candidate chunks into {output_file}...")
        CandidateMerger.merge(tmp_dir, output_file)
        with h5py.File(output_file, "a") as out:
            out.attrs["spectra_path"] = str(spectra_file)
        
        logger.info("Candidate generation complete.")
        return output_file

class DBSearcher:
    def __init__(
        self,
        fasta_path: str | Path,
        output_dir: str | Path,
        model: Denovo,
        device: torch.device,
        digest_config: DigestConfig,
        bucket_config: BucketConfig,
        decoy_config: DecoyConfig,
        spectrum_tokenizer,
        peptide_tokenizer: PTMPeptideTokenizer,
        inference_config: InferenceConfig,
        exec_config: ExecutionConfig,
    ):
        self.fasta_path = normalize_path(fasta_path)
        self.output_dir = normalize_path(output_dir)
        self.model = model
        self.device = device
        self.spectrum_tokenizer = spectrum_tokenizer
        self.peptide_tokenizer = peptide_tokenizer
        self.digest_config = digest_config
        self.bucket_config = bucket_config
        self.decoy_config = decoy_config
        self.inference_config = inference_config
        self.exec_config = exec_config

    def digest_fasta(self):
        return digest(
            self.digest_config,
            self.fasta_path,
            self.output_dir,
            self.peptide_tokenizer,
            self.bucket_config,
            self.decoy_config,
            self.exec_config.progress_bar,
            self.exec_config.n_workers,
            self.exec_config.worker_batch_size,
            self.exec_config.sort_buffer_size,
            self.exec_config.flush_batch_size,
            self.exec_config.overwrite,
        )

    def embed_peptide(self, peptide_metadata_file: str | Path, batch_size: int = 1024):
        peptide_metadata_file = normalize_path(peptide_metadata_file)
        if not peptide_metadata_file.exists():
            raise FileNotFoundError(
                f"The peptide metadata file {peptide_metadata_file} does not exist."
            )

        save_file = peptide_metadata_file.with_name("embedding.npy")
        if save_file.exists() and not self.exec_config.overwrite:
            logger.warning("The peptide embedding file already exists, skip.")
            return save_file

        with h5py.File(peptide_metadata_file, "r") as f:
            n_rows = f.attrs["n_rows"]
        
        hidden_size = self.model.clip.config["model"]["peptide"]["hidden_size"]
        embeddings_mmap = np.lib.format.open_memmap(
            save_file,
            "w+",
            np.float32,
            shape=(n_rows, hidden_size),
        )
        n_batch = (n_rows + batch_size - 1) // batch_size
        with h5py.File(peptide_metadata_file, "r") as h5_file, torch.no_grad(), torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.inference_config.gradscaling_enabled,
        ):
            for i in tqdm(
                range(n_batch),
                total=n_batch,
                desc="Embedding Peptides",
                disable=not self.exec_config.progress_bar,
                dynamic_ncols=True,
            ):
                row_ids = np.arange(i * batch_size, min((i + 1) * batch_size, n_rows))
                seqs = h5_file["metadata"][row_ids]["modified_peptide"]
                token_ids = [
                    self.peptide_tokenizer.tokenize(seq.decode()) for seq in seqs
                ]
                token_ids = pad_sequence(
                    token_ids,
                    batch_first=True,
                    padding_value=SPECIAL_TOKENS[PAD],
                )
                mask = token_ids != SPECIAL_TOKENS[PAD]
                embeddings: torch.Tensor = self.model.clip.repr_peptide(
                    Peptide(token_ids, mask).to(self.device)
                )
                embeddings_mmap[row_ids] = embeddings.float().cpu().numpy()

        return save_file

    def generate_candidates(self, spectra_file: str | Path):
        peptide_metadata_file = self.output_dir.joinpath("peptides_metadata.hdf5")
        if not peptide_metadata_file.exists():
            raise FileNotFoundError(
                f"Peptide metadata file {peptide_metadata_file} not found. "
                "Run digest_fasta() first."
            )

        generator = CandidateGenerator(
            peptide_metadata_file=peptide_metadata_file,
            output_dir=self.output_dir,
            mass_tolerance=self.inference_config.mass_tolerance,
            max_isotope=self.inference_config.max_isotope,
            n_workers=self.exec_config.n_workers,
            flush_batch_size=self.exec_config.flush_batch_size,
            progress_bar=self.exec_config.progress_bar,
            overwrite=self.exec_config.overwrite,
        )
        return generator.generate(spectra_file)

    def stage1_scoring(
        self,
        candidate_file: str | Path,
        peptide_embed_file: str | Path,
    ):
        candidate_file = normalize_path(candidate_file)
        peptide_embed_file = normalize_path(peptide_embed_file)

        if not candidate_file.exists():
            raise FileNotFoundError(f"Candidate file {candidate_file} does not exist.")

        if not peptide_embed_file.exists():
            raise FileNotFoundError(
                f"Peptide embedding file {peptide_embed_file} does not exist."
            )
        
        save_file = candidate_file.with_name("stage1_score.hdf5")
        if save_file.exists() and not self.exec_config.overwrite:
            logger.warning(f"Stage1 score file {save_file} already exists, skip.")
            return save_file

        if not self.exec_config.use_clip:
            self._copy_candidates_as_score(candidate_file, save_file)
            return save_file

        return self._clip_prescreen(candidate_file, peptide_embed_file, save_file)

    def _copy_candidates_as_score(
        self,
        candidate_file: Path,
        save_file: Path,
    ):
        peptide_metadata_file = self.output_dir.joinpath("peptides_metadata.hdf5")
        with h5py.File(candidate_file, "r") as src, h5py.File(
            peptide_metadata_file, "r"
        ) as meta_h5, h5py.File(save_file, "w") as out:
            n_rows = src["spectrum_id"].shape[0]
            chunk_size = (min(100_000, max(n_rows, 1)),)
            out.create_dataset(
                "spectrum_id",
                data=src["spectrum_id"][:],
                dtype=np.int64,
                chunks=chunk_size,
            )
            out.create_dataset(
                "score",
                shape=(n_rows,),
                dtype=np.float32,
                chunks=chunk_size,
            )
            out["score"][:] = 0.0

            candidate_ids = src["candidate_id"][:]
            order = np.argsort(candidate_ids)
            sorted_ids = candidate_ids[order]
            meta_sorted = meta_h5["metadata"][sorted_ids]
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))
            meta = meta_sorted[inv_order]

            dt_str = h5py.string_dtype(encoding="utf-8")
            meta_dtype = np.dtype(
                [
                    ("modified_peptide", dt_str),
                    ("peptide", dt_str),
                    ("protein_id", dt_str),
                    ("is_decoy", np.bool_),
                ]
            )
            out.create_dataset("metadata", data=meta, dtype=meta_dtype, chunks=chunk_size)
            out.attrs["n_rows"] = n_rows

    def _clip_prescreen(
        self,
        candidate_file: Path,
        peptide_embed_file: Path,
        save_file: Path,
    ):
        if self.exec_config.clip_embed_in_memory:
            logger.info("Loading peptide embeddings into host memory...")
            embeddings_all = np.load(peptide_embed_file)
            logger.info(f"Peptide embeddings loaded: {embeddings_all.shape}")
            embeddings_mmap = None
        else:
            logger.info("Using memory-mapped peptide embeddings (external I/O).")
            embeddings_mmap = np.load(peptide_embed_file, mmap_mode="r")
            embeddings_all = None

        def _fetch_embeddings(indices: npt.NDArray[np.int64]):
            if embeddings_all is not None:
                return embeddings_all[indices]

            # memory-mapped I/O: sort indices for sequential read, then restore order
            order = np.argsort(indices)
            sorted_indices = indices[order]
            sorted_emb = embeddings_mmap[sorted_indices]
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))
            return sorted_emb[inv_order]

        with h5py.File(candidate_file, "r") as cf:
            all_spec_ids = cf["spectrum_id"][:]
            spectra_path = cf.attrs.get("spectra_path", "")
            if not spectra_path:
                raise ValueError(
                    f"Candidate file {candidate_file} does not store 'spectra_path'."
                )

            spectra_path = normalize_path(spectra_path)
            if not spectra_path.exists():
                raise FileNotFoundError(
                    f"Spectra file {spectra_path} recorded in candidate file does not exist."
                )

        unique_specs, counts = np.unique(all_spec_ids, return_counts=True)
        spec_starts = np.cumsum(np.concatenate(([0], counts[:-1])))
        spec_to_start = {int(sid): int(start) for sid, start in zip(unique_specs, spec_starts)}
        spec_to_count = {int(sid): int(cnt) for sid, cnt in zip(unique_specs, counts)}

        n_spectra = int(unique_specs[-1]) + 1 if unique_specs.size > 0 else 0
        spectra_batch_size = self.exec_config.clip_spectra_batch_size
        topk = self.exec_config.topk

        logits_scale = self.model.clip.logits_scale.exp()

        peptide_metadata_file = self.output_dir.joinpath("peptides_metadata.hdf5")

        def _fetch_metadata(indices: npt.NDArray[np.int64]):
            """Read peptide metadata by candidate_id with sequential I/O."""
            order = np.argsort(indices)
            sorted_ids = indices[order]
            sorted_meta = meta_h5["metadata"][sorted_ids]
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))
            return sorted_meta[inv_order]

        with h5py.File(candidate_file, "r") as cf, h5py.File(
            peptide_metadata_file, "r"
        ) as meta_h5, h5py.File(save_file, "w") as h5_out, torch.no_grad(), torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.inference_config.gradscaling_enabled,
        ):
            flush_threshold = max(self.exec_config.flush_batch_size, 10_000)
            buffer = HDF5Stage1Buffer(h5_out, capacity=flush_threshold)

            spec_stream = SpectrumStream(spectra_path, self.spectrum_tokenizer)

            n_batches = (n_spectra + spectra_batch_size - 1) // max(spectra_batch_size, 1)
            for batch_idx in tqdm(
                range(n_batches),
                total=n_batches,
                desc="CLIP prescreening",
                disable=not self.exec_config.progress_bar,
                dynamic_ncols=True,
            ):
                chunk_start = batch_idx * spectra_batch_size
                chunk_end = min(chunk_start + spectra_batch_size, n_spectra)

                spectra_list = []
                precursor_mzs = []
                precursor_charges = []
                spec_ids_chunk = []

                for sid in range(chunk_start, chunk_end):
                    if sid not in spec_to_count:
                        continue
                    
                    spectrum, precursor_mz, precursor_charge = spec_stream[sid]
                    spectra_list.append(spectrum)
                    precursor_mzs.append(precursor_mz)
                    precursor_charges.append(precursor_charge)
                    spec_ids_chunk.append(sid)

                if not spectra_list:
                    continue

                batch = prepare_spectra(
                    spectra_list,
                    precursor_mzs,
                    precursor_charges
                ).to(self.device)
                spectra_emb = self.model.clip.repr_spectrum(batch.spectra).float()
                spectra_emb = F.normalize(spectra_emb, p=2, dim=-1)

                cand_counts = [spec_to_count[sid] for sid in spec_ids_chunk]
                total_cands = sum(cand_counts)
                cand_ids_chunk = np.empty(total_cands, dtype=np.int64)

                offset = 0
                for sid in spec_ids_chunk:
                    start = spec_to_start[sid]
                    cnt = spec_to_count[sid]
                    cand_ids_chunk[offset : offset + cnt] = cf["candidate_id"][
                        start : start + cnt
                    ]
                    offset += cnt

                pep_emb = (
                    torch.from_numpy(_fetch_embeddings(cand_ids_chunk).copy())
                    .pin_memory()
                    .to(self.device, non_blocking=True)
                )
                pep_emb = F.normalize(pep_emb, p=2, dim=-1)

                cand_offset = 0
                for j, sid in enumerate(spec_ids_chunk):
                    cnt = cand_counts[j]
                    if cnt == 0:
                        continue

                    scores = spectra_emb[[j]] @ pep_emb[
                        cand_offset : cand_offset + cnt
                    ].T  # [1, cnt]
                    logits = scores * logits_scale
                    probs = F.softmax(logits, dim=-1)[0]  # [cnt]

                    current_k = min(topk, cnt)
                    topk_probs, topk_indices = torch.topk(
                        probs, k=current_k, dim=-1
                    )
                    topk_indices_np = topk_indices.cpu().numpy()

                    topk_global_ids = cand_ids_chunk[
                        cand_offset + topk_indices_np
                    ]
                    meta = _fetch_metadata(topk_global_ids)

                    buffer.extend(
                        np.full(current_k, sid, dtype=np.int64),
                        meta["modified_peptide"],
                        meta["peptide"],
                        meta["protein_id"],
                        meta["is_decoy"],
                        topk_probs.cpu().numpy(),
                    )

                    cand_offset += cnt

            buffer.flush()

        return save_file

    def stage2_scoring(
        self,
        datamodule: DBSearchDataLoaderModule,
        strategy: Literal["average", "max", "min"]="average",
    ):
        def pll(logits: torch.Tensor, targets: torch.LongTensor, mask: torch.BoolTensor):
            log_probs = F.log_softmax(logits, dim=-1)
            step_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            scores = (step_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
            return torch.exp(scores)

        if strategy == "average":
            file_name = "stage2_score_average.hdf5"
        elif strategy == "max":
            file_name = "stage2_score_maximum.hdf5"
        elif strategy == "min":
            file_name = "stage2_score_minimum.hdf5"

        save_file = self.output_dir.joinpath(file_name)
        if save_file.exists() and not self.exec_config.overwrite:
            logger.warning(f"The stage2 score file {save_file} already exists, skip.")
            return save_file

        dataloader = datamodule.test_dataloader()
        with h5py.File(save_file, "w") as h5_file:
            total_rows = len(dataloader.dataset)
            chunk_size = (min(100_000, max(total_rows, 1)),)
            h5_file.create_dataset(
                "score", shape=(total_rows,), dtype=np.float32, chunks=chunk_size
            )
            h5_file.create_dataset(
                "spectrum_id", shape=(total_rows,), dtype=np.int64, chunks=chunk_size
            )
            h5_file.attrs["n_rows"] = total_rows
            flush_threshold = max(datamodule.data_config.test_batch_size, 100_000)
            with h5py.File(datamodule.stage1_score_file, "r") as h5_in:
                if "metadata" in h5_in:
                    h5_in.copy("metadata", h5_file)

            buffer = HDF5Stage2Buffer(capacity=flush_threshold)
            offset = 0

            with torch.no_grad(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=self.inference_config.gradscaling_enabled,
            ):
                for batch in tqdm(
                    dataloader,
                    desc="Stage2 Scoring",
                    total=len(dataloader),
                    dynamic_ncols=True,
                    disable=not self.exec_config.progress_bar,
                ):
                    batch: DBSearchBatch = batch.to(self.device)
                    mem_hidden_states, mem_attention_mask = self.model.encode_spectrum(
                        batch.spectra
                    )
                    prompt_hidden_states = self.model.prefill(batch.spectra)
                    output: DecoderOutput
                    output_reverse: DecoderOutput
                    output, output_reverse = self.model.peptide_decoder(
                        batch.peptide.tokens[:, :-1],
                        batch.peptide_reverse.tokens[:, :-1],
                        batch.spectra.precursor,
                        mem_hidden_states,
                        mem_attention_mask,
                        prompt_hidden_states,
                    )
                    forward_scores = pll(
                        output.logits,
                        batch.peptide.tokens[:, 1:],
                        batch.peptide.mask[:, 1:],
                    )
                    reverse_scores = pll(
                        output_reverse.logits,
                        batch.peptide_reverse.tokens[:, 1:],
                        batch.peptide_reverse.mask[:, 1:],
                    )
                    if strategy == "average":
                        final_scores = (forward_scores + reverse_scores) / 2
                    elif strategy == "max":
                        final_scores = torch.maximum(forward_scores, reverse_scores)
                    else:
                        final_scores = torch.minimum(forward_scores, reverse_scores)

                    scores_np = final_scores.cpu().numpy()
                    spec_ids_np = batch.spectrum_id.cpu().numpy()

                    if buffer.size + len(scores_np) > buffer.capacity:
                        offset += buffer.flush_to_hdf5(h5_file, offset)

                    buffer.extend(scores_np, spec_ids_np)

                if buffer.size > 0:
                    buffer.flush_to_hdf5(h5_file, offset)

        return save_file

    @staticmethod
    def get_best_psms(stage2_score_file: str | Path):
        best_psms: dict[int, PSMRecord] = {}
        partition_size = 10_000
        with h5py.File(stage2_score_file, "r") as file:
            n_total = file["score"].size
            for start in tqdm(
                range(0, n_total, partition_size),
                desc="Reading DB Search Results",
            ):
                end = min(start + partition_size, n_total)
                scores = file["score"][start:end]
                spectrum_ids = file["spectrum_id"][start:end]
                metadatas = file["metadata"][start:end]
                for spec_id, score, meta in zip(spectrum_ids, scores, metadatas):
                    spec_id = int(spec_id)
                    existing = best_psms.get(spec_id)
                    if existing is not None and score <= existing.score:
                        continue

                    best_psms[spec_id] = PSMRecord(
                        spectrum_id=spec_id,
                        score=float(score),
                        metadata=meta,
                    )

        return list(best_psms.values())

    @staticmethod
    def export_top1_results_tsv(
        stage2_score_file: str | Path,
        stage1_score_file: str | Path,
        spectra_file: str | Path,
        output_tsv: str | Path,
        fdr_threshold: float | None = None
    ):
        stage2_score_file = normalize_path(stage2_score_file)
        stage1_score_file = normalize_path(stage1_score_file)
        spectra_file = normalize_path(spectra_file)
        output_tsv = normalize_path(output_tsv)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(stage2_score_file, "r") as s2, h5py.File(
            stage1_score_file, "r"
        ) as s1, h5py.File(spectra_file, "r") as spec_f:
            spec_meta_all = spec_f["0"]["metadata"][:]
            scan_ids = np.array(
                [x.decode() for x in spec_meta_all["scan_id"]], dtype=object
            )
            precursor_mzs = spec_meta_all["precursor_mz"]
            precursor_charges = spec_meta_all["precursor_charge"]

            spectrum_ids = s2["spectrum_id"][:]
            denovo_scores = s2["score"][:]
            clip_scores = s1["score"][:]
            meta = s2["metadata"][:]

            df = pd.DataFrame(
                {
                    "spectrum_id": spectrum_ids,
                    "scan_id": scan_ids[spectrum_ids],
                    "precursor_mz": precursor_mzs[spectrum_ids],
                    "precursor_charge": precursor_charges[spectrum_ids],
                    "modified_peptide": [x.decode() for x in meta["modified_peptide"]],
                    "peptide": [x.decode() for x in meta["peptide"]],
                    "protein_id": [x.decode() for x in meta["protein_id"]],
                    "is_decoy": meta["is_decoy"],
                    "clip_score": clip_scores,
                    "denovo_score": denovo_scores,
                }
            )
        
        df = df.sort_values("denovo_score", ascending=False).drop_duplicates(
            "spectrum_id", keep="first"
        )
        df = df.sort_values("denovo_score", ascending=False).reset_index(drop=True)

        cum_decoys = df["is_decoy"].cumsum()
        cum_targets = (~df["is_decoy"]).cumsum()
        df["fdr"] = cum_decoys / np.maximum(cum_targets, 1)
        df["q_value"] = df["fdr"][::-1].cummin()[::-1]

        if fdr_threshold is not None:
            df = df[(df["q_value"] <= fdr_threshold) & (~df["is_decoy"])]

        df.to_csv(output_tsv, sep="\t", index=False)
        logger.info(f"Top1 denovo results exported to {output_tsv}")
        return output_tsv

    def search(
        self,
        spectra_file: str | Path,
        stage2_batch_size: int = 512,
    ):
        spectra_file = normalize_path(spectra_file)

        peptide_metadata_file = self.digest_fasta()
        peptide_embed_file = self.embed_peptide(peptide_metadata_file)
        candidate_file = self.generate_candidates(spectra_file)
        stage1_score_file = self.stage1_scoring(candidate_file, peptide_embed_file)

        datamodule = DBSearchDataLoaderModule(
            DataConfig(
                test_path=str(spectra_file),
                test_batch_size=stage2_batch_size,
                n_workers=self.exec_config.n_workers,
            ),
            self.spectrum_tokenizer,
            self.peptide_tokenizer,
            stage1_score_file,
        )

        return self.stage2_scoring(datamodule, "average")
