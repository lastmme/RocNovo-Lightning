import re
import os
import time
import string
import shutil
import heapq
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Generator, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import h5py
import numpy as np
from pyteomics.parser import isoforms, cleave
from pyteomics import fasta
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from rocnovo.config.db import FIXED_MODS, VAR_MODS, DigestConfig, Peptide, BucketConfig, DecoyConfig
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, CANONICAL
from rocnovo.common.logger import logger

@dataclass(frozen=True)
class ExternalSortBatchData:
    mass: list[float]
    modified_peptides: list[str]
    peptide: list[str]
    protein_id: list[str]
    is_decoy: list[bool]

class HDF5Buffer:
    def __init__(self, capacity: int=100_000):
        self.capacity = capacity
        self.size = 0
        self.masses = np.zeros(capacity, dtype=np.float64)
        self.modified_peptides = np.empty(capacity, dtype=object)
        self.peptides = np.empty(capacity, dtype=object)
        self.protein_ids = np.empty(capacity, dtype=object)
        self.is_decoys = np.zeros(capacity, dtype=np.bool_)

    def append_row(self, mass: float, data_pool: ExternalSortBatchData, ptr: int):
        self.masses[self.size] = mass
        self.modified_peptides[self.size] = data_pool.modified_peptides[ptr]
        self.peptides[self.size] = data_pool.peptide[ptr]
        self.protein_ids[self.size] = data_pool.protein_id[ptr]
        self.is_decoys[self.size] = data_pool.is_decoy[ptr]
        self.size += 1

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0

    def flush_to_hdf5(self, h5_file: h5py.File, start_offset: int):
        if self.size == 0:
            return
        
        end_offset = start_offset + self.size
        h5_file['mass'][start_offset:end_offset] = self.masses[:self.size]
        h5_file['modified_peptide'][start_offset:end_offset] = self.modified_peptides[:self.size]
        h5_file['peptide'][start_offset:end_offset] = self.peptides[:self.size]
        h5_file['protein_id'][start_offset:end_offset] = self.protein_ids[:self.size]
        h5_file['is_decoy'][start_offset:end_offset] = self.is_decoys[:self.size]
        self.clear()

def _construct_mods_dict(
    allowed_fixed_mods: list[str]=FIXED_MODS,
    allowed_var_mods: list[str]=VAR_MODS
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    swap_map: dict[str, str] = {}
    fixed_mods_map: dict[str, Any] = {}
    var_mods_map: dict[str, Any] = {}

    alias_pool = string.ascii_letters  # a-z, A-Z

    def _map_mods(mod_list: list[str], target_map: dict[str, Any]):
        if len(mod_list) > len(alias_pool):
            raise ValueError(f"The length of mod list is {len(mod_list)}, out of the acceptable length {len(alias_pool)}.")
        
        for i, mod in enumerate(mod_list):
            aa, mod_aa = mod.split(":", 1)
            mod_id = alias_pool[i]
            
            if aa.lower() == "nterm":
                target_map[f"{mod_id}-"] = True
                swap_map[f"{mod_id}-"] = mod_aa
            else:
                target_map[mod_id] = [aa]
                swap_map[f"{mod_id}{aa}"] = mod_aa

    _map_mods(allowed_fixed_mods, fixed_mods_map)
    _map_mods(allowed_var_mods, var_mods_map)

    return fixed_mods_map, var_mods_map, swap_map

def non_specific_digestion(
    config: DigestConfig,
    header: str,
    protein_seq: str,
    valid_aa: set[str],
    decoy_prefix: str="DECOY_",
):
    protein_id = header.split()[0]
    skipped = 0
    for i in range(len(protein_seq)):
        for j in range(i + config.min_peptide_len, min(i + config.max_peptide_len, len(protein_seq)) + 1):
            peptide = protein_seq[i:j]
            if any(aa not in valid_aa for aa in peptide):
                skipped += 1
            else:
                yield Peptide(
                    protein_id,
                    peptide,
                    protein_id.startswith(decoy_prefix)
                ), skipped
                skipped = 0

def synchro_digestion(
    config: DigestConfig,
    header: str,
    protein_seq: str,
    valid_aa: set[str],
    decoy_prefix: str="DECOY_",
):
    protein_id = header.split()[0]
    peptides = cleave(
        protein_seq,
        rule=config.enzyme,
        missed_cleavages=config.missed_cleavages,
        semi=config.digestion=="semi"
    )
    skipped = 0
    for peptide in peptides:
        if config.min_peptide_len <= len(peptide) <= config.max_peptide_len:
            if any(aa not in valid_aa for aa in peptide):
                skipped += 1
            else:
                yield Peptide(
                    protein_id,
                    peptide,
                    protein_id.startswith(decoy_prefix)
                ), skipped
                skipped = 0

def _digest(
    fasta_path: Path,
    config: DigestConfig,
    valid_aa: set[str],
    decoy_prefix: str="DECOY_",
):
    logger.debug(f"Using the {config.digestion} digestion rule.")
    for header, protein_seq in fasta.read(str(fasta_path)):
        if config.digestion == "non-specific":
            yield from non_specific_digestion(
                config,
                header,
                protein_seq,
                valid_aa,
                decoy_prefix
            )
        else:
            yield from synchro_digestion(
                config,
                header,
                protein_seq,
                valid_aa,
                decoy_prefix
            )

def decoy(fasta_path: Path, output_dir: Path, decoy_prefix: str="DECOY_", mode: Literal["reverse", "shuffle", "fused"]="reverse", progress_bar: bool=True):
    total = None
    if progress_bar:
        total = sum(1 for _ in fasta.read(str(fasta_path)))
    
    db_file = output_dir.joinpath(f"{fasta_path.stem}_with_decoy{fasta_path.suffix}")
    if db_file.exists():
        logger.warning(f"File {db_file} already exists, skipping.")
        return db_file

    shutil.copy(fasta_path, db_file)
    with tqdm(total=total, desc="Generating decoy sequences", unit=" sequences", disable=not progress_bar, dynamic_ncols=True) as pbar:
        with open(fasta_path, "r") as input_f, open(db_file, "a") as output_f:
            for header, protein_seq in fasta.read(input_f):
                decoy_protein_seq = fasta.decoy_sequence(
                    protein_seq,
                    mode=mode
                )
                decoy_header = f"{decoy_prefix}{header}"
                output_f.write('>' + decoy_header.replace('\n', '\n;') + '\n')
                for i in range(0, len(decoy_protein_seq), 60):
                    output_f.write(''.join([('%s\n' % decoy_protein_seq[i:i+60])]))
                
                pbar.update(1)
    
    return db_file

def aggregate_and_resolve_collisions(peptides_generator: Generator[tuple[Peptide, int], Any, None], progress_bar: bool=True):
    # we may use the stream to aggregate the peptides to avoid memory issues.
    logger.info("Aggregating shared peptides and resolving Target-Decoy collisions ...")
    peptide_map: dict[str, dict[Literal["target", "decoy"], set[str]]] = defaultdict(lambda: defaultdict(set))
    total_skipped = 0

    for pep, skipped in tqdm(peptides_generator, desc="Aggregating peptides", dynamic_ncols=True, disable=not progress_bar):
        total_skipped += skipped

        if pep.is_decoy:
            peptide_map[pep.peptide]["decoy"].add(pep.protein_id)
        else:
            peptide_map[pep.peptide]["target"].add(pep.protein_id)

    logger.info(f"Aggregation complete. Found {len(peptide_map)} unique base sequences. Starting PTM expansion ...")
    
    first_yield = True
    for seq, type_map in peptide_map.items():
        # Pass the accumulated skipped count only on the first yield so the downstream pbar is accurate
        skipped_to_yield = total_skipped if first_yield else 0
        first_yield = False
        
        targets = type_map["target"]
        decoys = type_map["decoy"]
        
        if targets:
            # TARGET WINS: If targets exist, we completely ignore any decoys for this sequence
            prot_str = ";".join(sorted(targets))
            yield Peptide(prot_str, seq, False), skipped_to_yield
        elif decoys:
            # PURE DECOY: Only exists in the decoy database
            prot_str = ";".join(sorted(decoys))
            yield Peptide(prot_str, seq, True), skipped_to_yield

def process_peptide_batch(
    batch_data: list[Peptide],
    temp_chunk_path: Path,
    max_mods: int,
    tokenizer: PTMPeptideTokenizer,
    fixed_mods: dict[str, Any],
    var_mods: dict[str, Any],
    swap_map: dict[str, str],
    bucket_config: BucketConfig
):
    swap_regex = re.compile("(%s)" % "|".join(map(re.escape, swap_map.keys()))) if swap_map else None
    
    local_buffer = []
    for peptide in batch_data:
        for isoform in isoforms(
            peptide.peptide,
            fixed_mods=fixed_mods,
            variable_mods=var_mods,
            max_mods=max_mods
        ):
            if swap_regex:
                modified_peptide = swap_regex.sub(lambda x: swap_map[x.group()], isoform)
            else:
                modified_peptide = isoform
                
            seq_mass = tokenizer.cal_seq_mass(modified_peptide)
            if bucket_config.min_mass > seq_mass or seq_mass > bucket_config.max_mass:
                continue
            
            local_buffer.append({
                "mass": seq_mass,
                "modified_peptide": modified_peptide,
                "peptide": peptide.peptide,
                "protein_id": peptide.protein_id,
                "is_decoy": peptide.is_decoy
            })
    
    if not local_buffer:
        return 0

    table = pa.Table.from_pylist(local_buffer)
    sorted_table = table.sort_by([("mass", "ascending")])
    
    pq.write_table(sorted_table, temp_chunk_path, compression="snappy")
    return len(local_buffer)

def write_digested_peptides(
    peptides_generator: Generator[tuple[Peptide, int], Any, None],
    output_dir: Path,
    tokenizer: PTMPeptideTokenizer,
    max_mods: int,
    fixed_mods: dict[str, Any],
    var_mods: dict[str, Any],
    swap_map: dict[str, str],
    bucket_config: BucketConfig,
    progress_bar: bool=True,
    num_workers: Optional[int]=None,
    worker_batch_size: int=5_000,
    sort_buffer_size: int=10_000
):
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_file = output_dir.joinpath("peptides_metadata.hdf5")
    writing_output_file = output_dir.joinpath("writing_peptides.hdf5")
    temp_sort_dir = output_dir.joinpath("tmp_parallel_chunks")
    
    if final_output_file.exists():
        logger.info(f"The peptides table is already stored.")
        schema_metadata = h5py.File(final_output_file).attrs
        logging_buffer = ["Bucket information is"]
        for key, value in schema_metadata.items():
            logging_buffer.append(f"{key.decode}: {value.decode}")
        
        logger.info(f"{''.join(logging_buffer)}")
        return final_output_file
    
    shutil.rmtree(temp_sort_dir, ignore_errors=True)
    temp_sort_dir.mkdir(exist_ok=True)

    if num_workers is None:
        num_workers = os.cpu_count()

    logger.info(f"Parallel digesting with {num_workers} workers...")

    chunk_files = []
    total_rows = 0
    total_skipped_count = 0
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        current_batch = []
        chunk_idx = 0
        
        with tqdm(peptides_generator, desc="Submitting tasks", disable=not progress_bar, dynamic_ncols=True) as pbar:
            for peptide, skipped_count in pbar:
                total_skipped_count += skipped_count
                current_batch.append(peptide)
                if len(current_batch) >= worker_batch_size:
                    chunk_path = temp_sort_dir / f"chunk_{chunk_idx + 1:05d}.parquet"
                    futures.append(executor.submit(
                        process_peptide_batch,
                        current_batch.copy(),
                        chunk_path,
                        max_mods,
                        tokenizer,
                        fixed_mods,
                        var_mods,
                        swap_map,
                        bucket_config
                    ))
                    current_batch.clear()
                    chunk_idx += 1
            
            if len(current_batch) > 0:
                chunk_path = temp_sort_dir / f"chunk_{chunk_idx + 1:05d}.parquet"
                futures.append(executor.submit(
                    process_peptide_batch,
                    current_batch,
                    chunk_path,
                    max_mods,
                    tokenizer,
                    fixed_mods,
                    var_mods,
                    swap_map,
                    bucket_config
                ))
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Waiting for tasks", disable=not progress_bar, dynamic_ncols=True)):
            rows = future.result()
            if rows > 0:
                total_rows += rows
                chunk_files.append(temp_sort_dir / f"chunk_{i + 1:05d}.parquet")
        
        logger.info(f"Total skipped peptides: {total_skipped_count}")

    if total_rows == 0:
        logger.warning("No peptides generated.")
        return final_output_file

    end = time.time()
    logger.info(f"Total digesting time: {end - start:.2f} seconds.")

    logger.info(f"K-way merging {len(chunk_files)} sorted chunks...")
    min_heap = []
    iterators = [pq.ParquetFile(f).iter_batches(sort_buffer_size) for f in chunk_files]
    batches: list[Optional[ExternalSortBatchData]] = [None] * len(chunk_files)
    pointers = [0] * len(chunk_files)
    batch_limits = [0] * len(chunk_files)

    def initial_batch_worker(idx: int):
        try:
            batch_arrow = next(iterators[idx])
            data = ExternalSortBatchData(
                mass=batch_arrow["mass"].to_pylist(),
                modified_peptides=batch_arrow["modified_peptide"].to_pylist(),
                peptide=batch_arrow["peptide"].to_pylist(),
                protein_id=batch_arrow["protein_id"].to_pylist(),
                is_decoy=batch_arrow["is_decoy"].to_pylist()
            )
            return (data.mass[0], idx, data, len(data.mass))
        except StopIteration:
            return None
        
    start = time.time()
    with ThreadPoolExecutor(max_workers=min(len(chunk_files), num_workers)) as pool:
        results = list(pool.map(initial_batch_worker, range(len(chunk_files))))
        
        for res in results:
            if res:
                first_mass, idx, data, limit = res
                batches[idx] = data
                batch_limits[idx] = limit
                heapq.heappush(min_heap, (first_mass, idx))
    
    end = time.time()
    logger.debug(f"Initializing heap in {end - start:.2f} seconds.")

    def refill_batch_sync(idx: int):
        try:
            batch_arrow = next(iterators[idx])
            batches[idx] = ExternalSortBatchData(
                mass=batch_arrow["mass"].to_pylist(),
                modified_peptides=batch_arrow["modified_peptide"].to_pylist(),
                peptide=batch_arrow["peptide"].to_pylist(),
                protein_id=batch_arrow["protein_id"].to_pylist(),
                is_decoy=batch_arrow["is_decoy"].to_pylist()
            )
            pointers[idx] = 0
            batch_limits[idx] = len(batches[idx].mass)
            heapq.heappush(min_heap, (batches[idx].mass[0], idx))
            return True

        except StopIteration:
            batches[idx] = None
            return False
    
    bucket_bin_width = (bucket_config.max_mass - bucket_config.min_mass) / bucket_config.bin_size
    num_buckets = bucket_config.bin_size + 1
    bucket_bounds = np.full((num_buckets, 2), -1, dtype=np.int64)
    buffer = HDF5Buffer(capacity=100_000)
    global_offset = 0
    current_bucket = -1
    dt_str = h5py.string_dtype(encoding='utf-8')
    with h5py.File(writing_output_file, 'w') as h5_file:
        chunk_size = (min(100_000, max(total_rows, 1)),)
        h5_file.create_dataset('mass', shape=(total_rows,), dtype=np.float64, chunks=chunk_size, compression='lzf')
        h5_file.create_dataset('modified_peptide', shape=(total_rows,), dtype=dt_str, chunks=chunk_size, compression='lzf')
        h5_file.create_dataset('peptide', shape=(total_rows,), dtype=dt_str, chunks=chunk_size, compression='lzf')
        h5_file.create_dataset('protein_id', shape=(total_rows,), dtype=dt_str, chunks=chunk_size, compression='lzf')
        h5_file.create_dataset('is_decoy', shape=(total_rows,), dtype=np.bool_, chunks=chunk_size, compression='lzf')
        
        h5_file.attrs["rocnovo_min_mass"] = bucket_config.min_mass
        h5_file.attrs["rocnovo_max_mass"] = bucket_config.max_mass
        h5_file.attrs["rocnovo_bin_size"] = bucket_config.bin_size
        h5_file.attrs["rocnovo_bin_width"] = bucket_bin_width
        h5_file.attrs["rocnovo_db_version"] = "1.0.0"
        h5_file.attrs["num_rows"] = total_rows
        with tqdm(total=total_rows, desc="Merging & Bucketing", disable=not progress_bar, dynamic_ncols=True) as pbar:
            while min_heap:
                mass: float
                chunk_idx: int
                mass, chunk_idx = heapq.heappop(min_heap)
                ptr = pointers[chunk_idx]
                data_pool = batches[chunk_idx]
                bucket_id = int((mass - bucket_config.min_mass) // bucket_bin_width)
                bucket_id = max(0, min(bucket_id, num_buckets - 1))

                if bucket_id != current_bucket:
                    if current_bucket != -1:
                        bucket_bounds[current_bucket, 1] = global_offset
                    
                    bucket_bounds[bucket_id, 0] = global_offset
                    current_bucket = bucket_id
                
                buffer.append_row(mass, data_pool, ptr)
                if len(buffer) >= buffer.capacity:
                    buffer.flush_to_hdf5(h5_file, global_offset + 1 - buffer.size)
                
                global_offset += 1
                pointers[chunk_idx] += 1

                if pointers[chunk_idx] < batch_limits[chunk_idx]:
                    next_ptr = pointers[chunk_idx]
                    heapq.heappush(min_heap, (data_pool.mass[next_ptr], chunk_idx))
                else:
                    refill_batch_sync(chunk_idx)

                pbar.update(1)

            if len(buffer) > 0:
                buffer.flush_to_hdf5(h5_file, global_offset - buffer.size)

            if current_bucket != -1:
                bucket_bounds[current_bucket, 1] = global_offset
        
        h5_file.create_dataset('bucket_bounds', data=bucket_bounds, compression='lzf')

    writing_output_file.rename(final_output_file)
    shutil.rmtree(temp_sort_dir, ignore_errors=True)
    return final_output_file

def digest(
    config: DigestConfig,
    fasta_path: Path,
    output_dir: Path,
    tokenizer: PTMPeptideTokenizer,
    bucket_config: BucketConfig,
    decoy_config: DecoyConfig,
    progress_bar: bool=True,
    num_workers: Optional[int]=None,
    worker_batch_size: int=5_000,
    sort_buffer_size: int=10_000
):
    logger.debug(f"Starting digestion with config: {config}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")
    db_file = fasta_path
    if decoy_config.generate_decoy:
        logger.warning("No decoy FASTA provided. Using the default decoy generation method (reverse sequence).")
        db_file = decoy(
            fasta_path,
            output_dir,
            decoy_config.decoy_prefix,
            decoy_config.decoy_strategy,
            progress_bar
        )
    
    logger.debug(f"Digesting {db_file}")
    valid_aa = set(c[0] for c in CANONICAL.keys() if c[0].isalpha())
    raw_peptides_generator = _digest(
        db_file,
        config,
        valid_aa,
        decoy_config.decoy_prefix
    )

    unique_peptides_generator = aggregate_and_resolve_collisions(
        raw_peptides_generator,
        progress_bar
    )

    fixed_mods, var_mods, swap_map = _construct_mods_dict(
        config.fixed_mods,
        config.var_mods
    )
    logger.debug(f"Fixed mods: {fixed_mods}")
    logger.debug(f"Var mods: {var_mods}")
    logger.debug(f"Swap map: {swap_map}")

    peptide_metadata_file = write_digested_peptides(
        unique_peptides_generator,
        output_dir,
        tokenizer,
        config.max_mods,
        fixed_mods,
        var_mods,
        swap_map,
        bucket_config,
        progress_bar,
        num_workers,
        worker_batch_size,
        sort_buffer_size
    )
    return peptide_metadata_file