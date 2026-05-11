import os
from pathlib import Path
from typing import Optional

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from rocnovo.config.data import Peptide, InferenceBatch, DBSearchBatch
from rocnovo.config.model import DecoderOutput
from rocnovo.config.db import DigestConfig, BucketConfig, DecoyConfig
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
        self.spectrum_ids = np.zeros(capacity, dtype=np.int64)
        self.peptide_ids = np.zeros(capacity, dtype=np.int64)
        self.scores = np.zeros(capacity, dtype=np.float32)
        chunk_size = (capacity,)
        self.h5_file = h5_file
        self.ds_spec = h5_file.create_dataset('spectrum_id', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=chunk_size, compression='lzf')
        self.ds_pep = h5_file.create_dataset('peptide_id', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=chunk_size, compression='lzf')
        self.ds_score = h5_file.create_dataset('score', shape=(0,), maxshape=(None,), dtype=np.float32, chunks=chunk_size, compression='lzf')

    def extend(self, spectrum_ids: list[int], peptide_ids: list[int], scores: npt.NDArray):
        n = len(spectrum_ids)
        if self.size + n > self.capacity:
            self.flush()
        
        if n > self.capacity:
            self._write_direct(spectrum_ids, peptide_ids, scores)
            return

        self.spectrum_ids[self.size : self.size + n] = spectrum_ids
        self.peptide_ids[self.size : self.size + n] = peptide_ids
        self.scores[self.size : self.size + n] = scores
        self.size += n
        if self.size >= self.capacity:
            self.flush()

    def _write_direct(self, spec_ids, pep_ids, scs):
        n = len(spec_ids)
        new_size = self.global_offset + n
        self.ds_spec.resize((new_size,))
        self.ds_pep.resize((new_size,))
        self.ds_score.resize((new_size,))
        self.ds_spec[self.global_offset : new_size] = spec_ids
        self.ds_pep[self.global_offset : new_size] = pep_ids
        self.ds_score[self.global_offset : new_size] = scs
        
        self.global_offset = new_size

    def flush(self):
        if self.size == 0:
            return
        
        self._write_direct(
            self.spectrum_ids[:self.size], 
            self.peptide_ids[:self.size], 
            self.scores[:self.size]
        )
        self.size = 0

    def __len__(self):
        return self.size

class DBSearcher:
    def __init__(
        self,
        fasta_path: str | Path,
        output_dir: str | Path,
        model: Denovo,
        spectrum_stream: SpectraDataLoaderModule,
        device: torch.device,
        digest_config: DigestConfig,
        bucket_config: BucketConfig,
        decoy_config: DecoyConfig,
        tokenizer: PTMPeptideTokenizer,
        inference_config: InferenceConfig,
        topk: int=10,
        n_workers: Optional[int]=None,
        worker_batch_size: int=5_000,
        sort_buffer_size: int=10_000,
        progress_bar: bool=True
    ):
        self.fasta_path = normalize_path(fasta_path)
        self.output_dir = normalize_path(output_dir)
        self.spectrum_stream = spectrum_stream
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.digest_config = digest_config
        self.bucket_config = bucket_config
        self.decoy_config = decoy_config
        self.topk = topk
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.inference_config = inference_config
        self.written_batch_size = self.spectrum_stream.data_config.test_batch_size
        self.worker_batch_size = worker_batch_size
        self.sort_buffer_size = sort_buffer_size
        self.progress_bar = progress_bar

    def digest_fasta(self):
        return digest(
            self.digest_config,
            self.fasta_path,
            self.output_dir,
            self.tokenizer,
            self.bucket_config,
            self.decoy_config,
            self.progress_bar,
            self.n_workers,
            self.worker_batch_size,
            self.sort_buffer_size
        )

    def embed_peptide(self, peptide_metadata_file: str | Path):
        peptide_metadata_file = normalize_path(peptide_metadata_file)
        if not peptide_metadata_file.exists():
            raise FileNotFoundError(f"The file {peptide_metadata_file} is not existed.")

        save_file = peptide_metadata_file.with_name(f"embedding.npy")
        if save_file.exists():
            return save_file

        num_rows = h5py.File(peptide_metadata_file).attrs["num_rows"]
        hidden_size = self.model.clip.config["model"]["peptide"]["hidden_size"]
        embeddings_mmap = np.lib.format.open_memmap(
            save_file,
            "w+",
            np.float32,
            shape=(num_rows, hidden_size)
        )
        batch_size = self.spectrum_stream.data_config.test_batch_size
        n_batch = (num_rows + batch_size - 1) // batch_size
        with h5py.File(peptide_metadata_file, "r") as h5_file, \
        torch.no_grad(), \
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.inference_config.gradscaling_enabled):
            for i in tqdm(
                range(n_batch),
                total=n_batch,
                desc="Embedding Peptides",
                disable=not self.progress_bar,
                dynamic_ncols=True
            ):
                row_ids = np.array(list(range(i * batch_size, min((i + 1) * batch_size, num_rows))))
                seqs = h5_file["modified_peptide"][row_ids]
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
                embeddings_mmap[row_ids - 1] = embeddings.float().cpu().numpy()

        return save_file
    
    @staticmethod
    def get_candidates(precursor_mass: float, is_ppm: bool, mass_tolerance: float, max_isotope: int, h5_file: h5py.File):  
        min_mass = h5_file.attrs['rocnovo_min_mass']
        bin_width = h5_file.attrs['rocnovo_bin_width']
        bucket_bounds = h5_file['bucket_bounds']
        num_buckets = bucket_bounds.shape[0]

        target_masses = np.array([precursor_mass + i * PROTON for i in range(max_isotope + 1)])
        target_buckets = set()
        
        for mass in target_masses:
            b_idx = int((mass - min_mass) // bin_width)
            for idx in (b_idx - 1, b_idx, b_idx + 1):
                if 0 <= idx < num_buckets:
                    target_buckets.add(idx)
        
        slices_to_read = []
        for b_idx in sorted(list(target_buckets)):
            start, end = bucket_bounds[b_idx]
            if start != -1 and end != -1:
                slices_to_read.append((start, end))

        if not slices_to_read:
            return np.array([], dtype=np.int64)

        merged_slices = [list(slices_to_read[0])]
        for start, end in slices_to_read[1:]:
            if start == merged_slices[-1][1]:
                merged_slices[-1][1] = end
            else:
                merged_slices.append([start, end])

        candidate_ids = []
        candidate_masses = []

        for start, end in merged_slices:
            candidate_masses.append(h5_file['mass'][start:end])
            candidate_ids.append(np.arange(start, end, dtype=np.int64))

        candidate_masses = np.concatenate(candidate_masses)
        candidate_ids = np.concatenate(candidate_ids)

        if is_ppm:
            tolerances = target_masses * mass_tolerance * 1e-6
        else:
            tolerances = np.full_like(target_masses, mass_tolerance)
        
        valid_mask = _numba_filter_candidates(candidate_masses, target_masses, tolerances)
        
        if np.any(valid_mask):
            return candidate_ids[valid_mask]
        else:
            return np.array([], dtype=np.int64)

    def stage1_scoring(self, peptide_embed_file: Path | str, peptide_metadata_path: Path | str):
        peptide_embed_file = normalize_path(peptide_embed_file)
        peptide_metadata_path = normalize_path(peptide_metadata_path)
        
        save_file = peptide_embed_file.with_name("stage1_score.hdf5")
        embeddings_mmap = np.load(peptide_embed_file, "r")
        
        @torch.amp.autocast(device_type="cuda", enabled=False)
        def fp32_matmul(a: torch.Tensor, b: torch.Tensor):
            return a @ b.T

        dataloader = self.spectrum_stream.test_dataloader()
        flush_threshold = max(self.written_batch_size, 10_000)
        with h5py.File(peptide_metadata_path, 'r') as h5_in, \
        h5py.File(save_file, 'w') as h5_out, \
        torch.no_grad(), \
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.inference_config.gradscaling_enabled):
            buffer = HDF5Stage1Buffer(h5_out, capacity=flush_threshold)
            for i, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                dynamic_ncols=True,
                desc="Stage1 Scoring",
                disable=not self.progress_bar
            ):
                batch: InferenceBatch = batch.to(self.device)
                spectra_embeddings = self.model.clip.repr_spectrum(batch.spectra).float()
                spectra_embeddings = F.normalize(spectra_embeddings, p=2, dim=-1)
                batch_candidate_lists = []
                unique_candidates = set()
                
                for j in range(spectra_embeddings.shape[0]):
                    candidate_ids = self.get_candidates(
                        batch.spectra.precursor.mass[j].item(),
                        True,
                        self.inference_config.mass_tolerance,
                        self.inference_config.max_isotope,
                        h5_in
                    )
                    batch_candidate_lists.append(candidate_ids)
                    unique_candidates.update(candidate_ids)
                
                if not unique_candidates:
                    continue
                
                retrieved_ids = np.sort(list(unique_candidates))
                id_to_local_idx = {pid: idx for idx, pid in enumerate(retrieved_ids)}
                
                batch_peptide_embeddings_cpu = torch.tensor(
                    embeddings_mmap[retrieved_ids - 1], dtype=torch.float32
                ).pin_memory()
                
                for j in range(spectra_embeddings.shape[0]):
                    spectrum_id = i * self.spectrum_stream.data_config.test_batch_size + j
                    candidate_ids = np.array(batch_candidate_lists[j])
                    local_indices = [id_to_local_idx[pid] for pid in candidate_ids]
                    if not local_indices:
                        continue
                    
                    current_gpu_embeddings = batch_peptide_embeddings_cpu[local_indices].to(self.device, non_blocking=True)
                    current_gpu_embeddings = F.normalize(current_gpu_embeddings, p=2, dim=-1)
                    scores = fp32_matmul(spectra_embeddings[[j]], current_gpu_embeddings)
                    current_k = min(self.topk, scores.shape[-1])
                    topk_scores, topk_indices = torch.topk(scores, k=current_k, dim=-1)
                    matched_peptide_ids = candidate_ids[topk_indices[0].cpu().numpy()]
                    buffer.extend(
                        [spectrum_id] * current_k,
                        matched_peptide_ids.tolist(),
                        topk_scores[0].cpu().numpy()
                    )
               
            buffer.flush()

        return save_file

    def stage2_scoring(self, datamodule: DBSearchDataLoaderModule):
        def pll(logits: torch.Tensor, targets: torch.LongTensor, mask: torch.BoolTensor):
            log_probs = F.log_softmax(logits, dim=-1)
            # [B, S, 1] -> [B, S]
            step_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            # [B]
            scores = (step_log_probs * mask).sum(dim=-1)
            return scores

        save_file = self.output_dir.joinpath("stage2_score.hdf5")
        if save_file.exists():
            return save_file

        dataloader = datamodule.test_dataloader()
        
        with h5py.File(save_file, "w") as h5_file:
            total_rows = len(dataloader.dataset)
            chunk_size = (min(100_000, max(total_rows, 1)),)
            h5_file.create_dataset('score', shape=(total_rows,), dtype=np.float32, chunks=chunk_size, compression='lzf')
            h5_file.create_dataset('spectrum_id', shape=(total_rows,), dtype=np.int64, chunks=chunk_size, compression='lzf')
            h5_file.create_dataset('peptide_id', shape=(total_rows,), dtype=np.int64, chunks=chunk_size, compression='lzf')
            
            h5_file.attrs["n_rows"] = total_rows
            offset = 0
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Stage2 Scoring", total=len(dataloader), dynamic_ncols=True, disable=not self.progress_bar):
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
                        batch.peptide.tokens[:, :-1],
                        batch.peptide.mask[:, :-1]
                    )
                    reverse_scores = pll(
                        output_reverse.logits,
                        batch.peptide_reverse.tokens[:, 1:],
                        batch.peptide_reverse.mask[:, 1:]
                    )
                    final_scores = (forward_scores + reverse_scores) / 2
                    
                    current_batch_size = final_scores.shape[0]
                    end_idx = offset + current_batch_size
                    h5_file['score'][offset:end_idx] = final_scores.cpu().numpy()
                    h5_file['spectrum_id'][offset:end_idx] = batch.spectrum_id.cpu().numpy()
                    h5_file['peptide_id'][offset:end_idx] = batch.peptide_id.cpu().numpy()
                    offset += current_batch_size
        
        return save_file

    def search(self):
        pass