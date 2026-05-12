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
from rocnovo.config.data import Peptide, InferenceBatch, DBSearchBatch
from rocnovo.config.model import DecoderOutput
from rocnovo.config.db import DigestConfig, BucketConfig, DecoyConfig, ExecutionConfig
from rocnovo.config.inference import InferenceConfig
from rocnovo.data.dataloaders import SpectraDataLoaderModule, DBSearchDataLoaderModule
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, SPECIAL_TOKENS, PAD, PROTON
from rocnovo.module.denovo import Denovo
from rocnovo.common.io import normalize_path
from rocnovo.common.digest import digest

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _numba_filter_candidates(candidate_masses, target_masses, tolerances):
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
        dt_str = h5py.string_dtype(encoding='utf-8')
        self.meta_dtype = np.dtype([
            ("modified_peptide", dt_str),
            ("peptide", dt_str),
            ("protein_id", dt_str),
            ("is_decoy", np.bool_)
        ])
        self.ds_spec = h5_file.create_dataset("spectrum_id", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=chunk_size)
        self.ds_score = h5_file.create_dataset("score", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=chunk_size)
        self.ds_metadata = h5_file.create_dataset("metadata", shape=(0,), maxshape=(None,), dtype=self.meta_dtype, chunks=chunk_size)
    
    def extend(self, spectrum_ids, modified_peptides, peptides, protein_ids, is_decoys, scores):
        n = len(spectrum_ids)
        if self.size + n > self.capacity:
            self.flush()
        
        if n > self.capacity:
            self._write_direct(spectrum_ids, modified_peptides, peptides, protein_ids, is_decoys, scores)
            return

        self.spectrum_ids[self.size:self.size + n] = spectrum_ids
        self.modified_peptides[self.size:self.size + n] = modified_peptides
        self.peptides[self.size:self.size + n] = peptides
        self.protein_ids[self.size:self.size + n] = protein_ids
        self.is_decoys[self.size:self.size + n] = is_decoys
        self.scores[self.size:self.size + n] = scores
        self.size += n
        
        if self.size >= self.capacity:
            self.flush()

    def _write_direct(self, spec_ids, mod_peps, peps, prot_ids, is_decoys, scs):
        n = len(spec_ids)
        new_size = self.global_offset + n
        
        self.ds_spec.resize((new_size,))
        self.ds_score.resize((new_size,))
        self.ds_metadata.resize((new_size,))
        
        self.ds_spec[self.global_offset:new_size] = spec_ids
        self.ds_score[self.global_offset:new_size] = scs
        temp_struct = np.empty(n, dtype=self.meta_dtype)
        temp_struct["modified_peptide"] = mod_peps
        temp_struct["peptide"] = peps
        temp_struct["protein_id"] = prot_ids
        temp_struct["is_decoy"] = is_decoys
        self.ds_metadata[self.global_offset:new_size] = temp_struct
        self.global_offset = new_size

    def flush(self):
        if self.size == 0:
            return
        
        self._write_direct(
            self.spectrum_ids[:self.size], 
            self.modified_peptides[:self.size],
            self.peptides[:self.size],
            self.protein_ids[:self.size],
            self.is_decoys[:self.size],
            self.scores[:self.size]
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

    def extend(self, scores: np.ndarray, spectrum_ids: np.ndarray):
        n = len(scores)
        self.scores[self.size:self.size + n] = scores
        self.spectrum_ids[self.size:self.size + n] = spectrum_ids
        self.size += n

    def flush_to_hdf5(self, h5_file: h5py.File, start_offset: int) -> int:
        if self.size == 0:
            return 0
        
        end_offset = start_offset + self.size
        h5_file["score"][start_offset:end_offset] = self.scores[:self.size]
        h5_file["spectrum_id"][start_offset:end_offset] = self.spectrum_ids[:self.size]
        
        written_size = self.size
        self.size = 0
        return written_size

class InMemoryCandidateDB:
    def __init__(self, h5_file: h5py.File):
        self.attrs = {
            "rocnovo_min_mass": h5_file.attrs["rocnovo_min_mass"],
            "rocnovo_bin_width": h5_file.attrs["rocnovo_bin_width"]
        }
        self.data = {
            "bucket_bounds": h5_file["bucket_bounds"][:],
            "metadata": h5_file["metadata"][:]
        }

    def __getitem__(self, key: str):
        return self.data[key]

class CandidateSearcher:
    @staticmethod
    def get_candidates(
        precursor_mass: float,
        is_ppm: bool,
        mass_tolerance: float,
        max_isotope: int,
        db: h5py.File | InMemoryCandidateDB
    ):  
        def _empty_res():
            return {
                "candidate_ids": np.array([], dtype=np.int64),
                "modified_peptides": np.array([], dtype=object),
                "peptides": np.array([], dtype=object),
                "protein_ids": np.array([], dtype=object),
                "is_decoys": np.array([], dtype=bool)
            }
        
        min_mass = db.attrs["rocnovo_min_mass"]
        bin_width = db.attrs["rocnovo_bin_width"]
        bucket_bounds = db["bucket_bounds"]
        num_buckets = bucket_bounds.shape[0]
        
        target_masses = precursor_mass - np.arange(max_isotope + 1) * PROTON
        b_indices = ((target_masses - min_mass) // bin_width).astype(np.int64)
        offsets = np.array([-1, 0, 1], dtype=np.int64)
        all_b_indices = np.ravel(b_indices[:, None] + offsets)
        valid_b_mask = (all_b_indices >= 0) & (all_b_indices < num_buckets)
        
        target_buckets = np.unique(all_b_indices[valid_b_mask])
        if target_buckets.size == 0:
            return _empty_res()

        bounds = bucket_bounds[target_buckets]
        valid_bounds = bounds[(bounds[:, 0] != -1) & (bounds[:, 1] != -1)]
        if valid_bounds.size == 0:
            return _empty_res()

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
            chunks.append((
                np.arange(start, end, dtype=np.int64),
                db["metadata"][start:end]
            ))

        if len(chunks) == 1:
            c_ids, c_meta = chunks[0]
        else:
            c_ids = np.concatenate([c[0] for c in chunks])
            c_meta = np.concatenate([c[1] for c in chunks])
        
        if is_ppm:
            tolerances = target_masses * mass_tolerance * 1e-6
        else:
            tolerances = np.full_like(target_masses, mass_tolerance)
        
        valid_mask = _numba_filter_candidates(
            c_meta["mass"],
            target_masses,
            tolerances
        )
        
        if not np.any(valid_mask):
            return _empty_res()

        valid_meta = c_meta[valid_mask]
        
        return {
            "candidate_ids": c_ids[valid_mask],
            "modified_peptides": valid_meta["modified_peptide"],
            "peptides": valid_meta["peptide"],
            "protein_ids": valid_meta["protein_id"],
            "is_decoys": valid_meta["is_decoy"]
        }

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
        tokenizer: PTMPeptideTokenizer,
        inference_config: InferenceConfig,
        exec_config: ExecutionConfig
    ):
        self.fasta_path = normalize_path(fasta_path)
        self.output_dir = normalize_path(output_dir)
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
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
            self.tokenizer,
            self.bucket_config,
            self.decoy_config,
            self.exec_config.progress_bar,
            self.exec_config.n_workers,
            self.exec_config.worker_batch_size,
            self.exec_config.sort_buffer_size,
            self.exec_config.flush_batch_size,
            self.exec_config.overwrite
        )

    def embed_peptide(self, peptide_metadata_file: str | Path, batch_size: int=4096):
        peptide_metadata_file = normalize_path(peptide_metadata_file)
        if not peptide_metadata_file.exists():
            raise FileNotFoundError(f"The file {peptide_metadata_file} is not existed.")

        save_file = peptide_metadata_file.with_name(f"embedding.npy")
        if save_file.exists() and not self.exec_config.overwrite:
            logger.warning(f"The Peptide Embedding file is existed, skip.")
            return save_file

        n_rows = h5py.File(peptide_metadata_file).attrs["n_rows"]
        hidden_size = self.model.clip.config["model"]["peptide"]["hidden_size"]
        embeddings_mmap = np.lib.format.open_memmap(
            save_file,
            "w+",
            np.float32,
            shape=(n_rows, hidden_size)
        )
        n_batch = (n_rows + batch_size - 1) // batch_size
        with h5py.File(peptide_metadata_file, "r") as h5_file, \
        torch.no_grad(), \
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.inference_config.gradscaling_enabled):
            for i in tqdm(
                range(n_batch),
                total=n_batch,
                desc="Embedding Peptides",
                disable=not self.exec_config.progress_bar,
                dynamic_ncols=True
            ):
                row_ids = np.array(list(range(i * batch_size, min((i + 1) * batch_size, n_rows))))
                seqs = h5_file["metadata"][row_ids]["modified_peptide"]
                token_ids = [
                    self.tokenizer.tokenize(seq.decode())
                    for seq in seqs
                ]
                token_ids = pad_sequence(token_ids, batch_first=True, padding_value=SPECIAL_TOKENS[PAD])
                mask = token_ids != SPECIAL_TOKENS[PAD]
                embeddings: torch.Tensor = self.model.clip.repr_peptide(
                    Peptide(
                        token_ids,
                        mask
                    ).to(self.device)
                )
                embeddings_mmap[row_ids] = embeddings.float().cpu().numpy()

        return save_file

    def stage1_scoring(
        self,
        datamodule: DBSearchDataLoaderModule,
        peptide_embed_file: Path | str,
        peptide_metadata_file: Path | str,
        in_memory: bool=False
    ):
        @torch.amp.autocast(device_type="cuda", enabled=False)
        def fp32_matmul(a: torch.Tensor, b: torch.Tensor):
            return a @ b.T

        peptide_embed_file = normalize_path(peptide_embed_file)
        peptide_metadata_file = normalize_path(peptide_metadata_file)
        
        if not peptide_embed_file.exists():
            raise FileNotFoundError(f"The peptide embedding file {peptide_embed_file} is not existed.")

        if not peptide_metadata_file.exists():
            raise FileNotFoundError(f"The peptide metadata file {peptide_embed_file} is not existed.")

        save_file = peptide_embed_file.with_name("stage1_score.hdf5")

        if save_file.exists() and not self.exec_config.overwrite:
            logger.warning(f"The stage1 score file {save_file} is existed, skip.")
            return save_file

        embeddings_mmap = np.load(peptide_embed_file, mmap_mode="r")
        dataloader = datamodule.test_dataloader()
        flush_threshold = max(self.exec_config.flush_batch_size, 10_000)

        with h5py.File(peptide_metadata_file, "r") as h5_in, \
        h5py.File(save_file, "w") as h5_out, \
        torch.no_grad(), \
        torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.inference_config.gradscaling_enabled
        ):
            if in_memory:
                logger.info("Loading candidate metadata into memory...")
                search_db = InMemoryCandidateDB(h5_in)
            else:
                search_db = h5_in

            buffer = HDF5Stage1Buffer(h5_out, capacity=flush_threshold)
            for i, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                dynamic_ncols=True,
                desc="Stage1 Scoring",
                disable=not self.exec_config.progress_bar
            ):
                batch: DBSearchBatch = batch.to(self.device)
                spectra_embeddings = self.model.clip.repr_spectrum(batch.spectra).float()
                spectra_embeddings = F.normalize(spectra_embeddings, p=2, dim=-1)
                
                batch_size = spectra_embeddings.shape[0]
                batch_candidates_metadata = []
                unique_candidates = set()
                precursor_masses = batch.spectra.precursor.mass.cpu().numpy()
                for j in range(batch_size):
                    candidate_metadata: dict[str, npt.NDArray] = CandidateSearcher.get_candidates(
                        precursor_masses[j],
                        True,
                        self.inference_config.mass_tolerance,
                        self.inference_config.max_isotope,
                        search_db
                    )
                    batch_candidates_metadata.append(candidate_metadata)
                    
                    if candidate_metadata["candidate_ids"].size > 0:
                        unique_candidates.update(candidate_metadata["candidate_ids"].tolist())
                
                if not unique_candidates:
                    continue
                
                retrieved_ids = np.sort(np.fromiter(unique_candidates, dtype=np.int64))
                batch_peptide_embeddings_cpu = torch.from_numpy(
                    embeddings_mmap[retrieved_ids].copy()
                ).pin_memory()
                for j in range(batch_size):
                    if candidate_metadata["candidate_ids"].size == 0:
                        continue

                    spectrum_id = i * dataloader.batch_size + j
                    candidate_metadata = batch_candidates_metadata[j]
                    candidate_ids = candidate_metadata["candidate_ids"]

                    local_indices = np.searchsorted(retrieved_ids, candidate_ids)
                    current_gpu_embeddings = batch_peptide_embeddings_cpu[local_indices].to(self.device, non_blocking=True)
                    current_gpu_embeddings = F.normalize(current_gpu_embeddings, p=2, dim=-1)
                    
                    scores = fp32_matmul(spectra_embeddings[[j]], current_gpu_embeddings)
                    current_k = min(self.exec_config.topk, scores.shape[-1])
                    topk_scores, topk_indices = torch.topk(scores, k=current_k, dim=-1)
                    topk_idx_np = topk_indices[0].cpu().numpy()
                    buffer.extend(
                        np.full(current_k, spectrum_id, dtype=np.int64),
                        candidate_metadata["modified_peptides"][topk_idx_np],
                        candidate_metadata["peptides"][topk_idx_np],
                        candidate_metadata["protein_ids"][topk_idx_np],
                        candidate_metadata["is_decoys"][topk_idx_np],
                        topk_scores[0].cpu().numpy()
                    )
            
            buffer.flush()

        return save_file

    def stage2_scoring(self, datamodule: DBSearchDataLoaderModule, strategy: Literal["average", "max", "min"]="average"):
        def pll(logits: torch.Tensor, targets: torch.LongTensor, mask: torch.BoolTensor):
            log_probs = F.log_softmax(logits, dim=-1)
            # [B, S, 1] -> [B, S]
            step_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            # [B]
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
            logger.warning(f"The stage2 score file {save_file} is existed, skip.")
            return save_file
        
        dataloader = datamodule.test_dataloader()
        with h5py.File(save_file, "w") as h5_file:
            total_rows = len(dataloader.dataset)
            chunk_size = (min(100_000, max(total_rows, 1)),)
            h5_file.create_dataset("score", shape=(total_rows,), dtype=np.float32, chunks=chunk_size)
            h5_file.create_dataset("spectrum_id", shape=(total_rows,), dtype=np.int64, chunks=chunk_size)
            h5_file.attrs["n_rows"] = total_rows
            flush_threshold = max(datamodule.data_config.test_batch_size, 100_000)
            with h5py.File(datamodule.stage1_score_file, "r") as h5_in:
                if "metadata" in h5_in:
                    h5_in.copy("metadata", h5_file)
            
            buffer = HDF5Stage2Buffer(capacity=flush_threshold)
            offset = 0

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.inference_config.gradscaling_enabled):
                for batch in tqdm(dataloader, desc="Stage2 Scoring", total=len(dataloader), dynamic_ncols=True, disable=not self.exec_config.progress_bar):
                    batch: DBSearchBatch = batch.to(self.device)
                    mem_hidden_states, mem_attention_mask = self.model.encode_spectrum(batch.spectra)
                    prompt_hidden_states = self.model.prefill(batch.spectra)
                    output: DecoderOutput
                    output_reverse: DecoderOutput
                    output, output_reverse = self.model.peptide_decoder(
                        batch.peptide.tokens[:, :-1],
                        batch.peptide_reverse.tokens[:, :-1],
                        batch.spectra.precursor,
                        mem_hidden_states,
                        mem_attention_mask,
                        prompt_hidden_states
                    )
                    forward_scores = pll(
                        output.logits,
                        batch.peptide.tokens[:, 1:],
                        batch.peptide.mask[:, 1:]
                    )
                    reverse_scores = pll(
                        output_reverse.logits,
                        batch.peptide_reverse.tokens[:, 1:],
                        batch.peptide_reverse.mask[:, 1:]
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

    def get_best_psms(stage2_score_file: str | Path):
        best_psms = {}
        partition_size = 10_000
        with h5py.File(stage2_score_file, "r") as file:
            n_total = file["score"].size
            for start in tqdm(range(0, n_total, partition_size), desc="Reading DB Search Results"):
                end = min(start + partition_size, n_total)
                scores = file["score"][start:end]
                spectrum_ids = file["spectrum_id"][start:end]
                metadatas = file["metadata"][start:end]
                for spec_id, score, meta in zip(spectrum_ids, scores, metadatas):
                    existing = best_psms.get(spec_id)
                    if existing is not None and score <= existing[0]:
                        continue
                    
                    best_psms[spec_id] = (score, meta)
        
        return best_psms

    def search(self):
        pass

def control_fdr(best_psms: dict[str, tuple[float, npt.NDArray]], fdr_threshold: float=0.01):
    n_psms = len(best_psms)
    print(f"n_psm: {n_psms}")
    spectrum_ids = np.empty(n_psms, dtype=object)
    scores = np.empty(n_psms, dtype=np.float32)

    modified_peptides = np.empty(n_psms, dtype=object)
    peptides = np.empty(n_psms, dtype=object)
    protein_ids = np.empty(n_psms, dtype=object)
    is_decoy_list = np.empty(n_psms, dtype=bool)
    for i, (spec_id, (score, meta)) in enumerate(best_psms.items()):
        mod_pep = meta["modified_peptide"].decode()
        pep = meta["peptide"].decode()
        prot_id = meta["protein_id"].decode()
        is_decoy = meta["is_decoy"]
        spectrum_ids[i] = spec_id
        scores[i] = score
        modified_peptides[i] = mod_pep
        peptides[i] = pep
        protein_ids[i] = prot_id
        is_decoy_list[i] = is_decoy
    
    df = pd.DataFrame({
        "spectrum_id": spectrum_ids,
        "score": scores,
        "is_decoy": is_decoy_list,
        "modified_peptide": modified_peptides,
        "peptide": peptides,
        "protein": protein_ids
    })
    df.sort_values(by="score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    cum_decoys = df["is_decoy"].cumsum()
    cum_targets = (~df["is_decoy"]).cumsum()
    df["fdr"] = cum_decoys / np.maximum(cum_targets, 1)
    df["q_value"] = df["fdr"][::-1].cummin()[::-1]
    valid_psms = df[(df["q_value"] <= fdr_threshold) & (~df["is_decoy"])]
    score_cutoff = valid_psms["score"].min() if not valid_psms.empty else None
    
    print(f"Total PSMs Evaluated: {len(df)}")
    print(f"Passed {fdr_threshold*100}% FDR: {len(valid_psms)} targets")
    if score_cutoff is not None:
        print(f"Score Cutoff: {score_cutoff:.4f}")
    else:
        print("Warning: No PSMs passed the strict FDR threshold.")
    
    return valid_psms, score_cutoff