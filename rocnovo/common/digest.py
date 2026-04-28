import re
import os
import time
import string
import shutil
import heapq
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal, Generator, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from pyteomics.parser import isoforms, cleave
from pyteomics import fasta
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from rocnovo.config.db import FIXED_MODS, VAR_MODS, DigestConfig, Peptide, BucketConfig
from rocnovo.tokenizer.peptide import PTMPeptideTokenizer, CANONICAL
from rocnovo.common.logger import logger

@dataclass(frozen=True)
class ExternalSortBatchData:
    mass: list[float]
    modified_peptides: list[str]
    peptide: list[str]
    protein_id: list[str]
    is_decoy: list[bool]

@dataclass
class Buffer:
    ids: list[int] = field(default_factory=list)
    bucket_ids: list[int] = field(default_factory=list)
    masses: list[float] = field(default_factory=list)
    modified_peptides: list[str] = field(default_factory=list)
    peptides: list[str] = field(default_factory=list)
    protein_ids: list[str] = field(default_factory=list)
    is_decoys: list[bool] = field(default_factory=list)

    def append_row(self, global_id: int, bucket_id: int, mass: float, data_pool: ExternalSortBatchData, ptr: int):
        self.ids.append(global_id)
        self.bucket_ids.append(bucket_id)
        self.masses.append(mass)
        self.modified_peptides.append(data_pool.modified_peptides[ptr])
        self.peptides.append(data_pool.peptide[ptr])
        self.protein_ids.append(data_pool.protein_id[ptr])
        self.is_decoys.append(data_pool.is_decoy[ptr])

    def __len__(self):
        return len(self.ids)

    def clear(self):
        self.ids.clear()
        self.bucket_ids.clear()
        self.masses.clear()
        self.modified_peptides.clear()
        self.peptides.clear()
        self.protein_ids.clear()
        self.is_decoys.clear()

    def to_table(self, schema: pa.Schema) -> pa.Table:
        return pa.Table.from_arrays(
            [
                self.ids, self.bucket_ids, self.masses, 
                self.modified_peptides, self.peptides, 
                self.protein_ids, self.is_decoys
            ],
            schema=schema
        )

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

def process_peptide_batch(
    batch_data: list[Peptide],
    temp_chunk_path: Path,
    max_mods: int,
    tokenizer: PTMPeptideTokenizer,
    fixed_mods: dict[str, Any],
    var_mods: dict[str, Any],
    swap_map: dict[str, str],
    bucket_config: BucketConfig
) -> int:
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
    worker_batch_size: int=10_000,
    sort_buffer_size: int=10_000
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_file = output_dir.joinpath("peptides.parquet")
    writing_output_file = output_dir.joinpath("writing_peptides.parquet")
    temp_sort_dir = output_dir.joinpath("tmp_parallel_chunks")
    
    if final_output_file.exists():
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
        
        for i, future in enumerate(as_completed(futures)):
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
    final_schema = pa.schema([
        ("id", pa.int64()),
        ("bucket_id", pa.int32()),
        ("mass", pa.float64()),
        ("modified_peptide", pa.string()),
        ("peptide", pa.string()),
        ("protein_id", pa.string()),
        ("is_decoy", pa.bool_())
    ])
    
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

    def refill_batch_sync(idx: int) -> bool:
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
    
    global_id = 1
    buffer = Buffer()
    bucket_bin_width = (bucket_config.max_mass - bucket_config.min_mass) / bucket_config.bin_size
    
    with pq.ParquetWriter(writing_output_file, final_schema, compression='snappy') as writer:
        with tqdm(total=total_rows, desc="Merging & Bucketing", disable=not progress_bar, dynamic_ncols=True) as pbar:
            while min_heap:
                mass: float
                chunk_idx: int
                mass, chunk_idx = heapq.heappop(min_heap)
                ptr = pointers[chunk_idx]
                data_pool = batches[chunk_idx]
                bucket_id = int((mass - bucket_config.min_mass) // bucket_bin_width)
                
                buffer.append_row(
                    global_id,
                    bucket_id,
                    mass,
                    data_pool,
                    ptr
                )
                global_id += 1
                pointers[chunk_idx] += 1

                if len(buffer) >= 100_000:
                    table = buffer.to_table(final_schema)
                    writer.write_table(table)
                    buffer.clear()

                if pointers[chunk_idx] < batch_limits[chunk_idx]:
                    next_ptr = pointers[chunk_idx]
                    heapq.heappush(min_heap, (data_pool.mass[next_ptr], chunk_idx))
                else:
                    refill_batch_sync(chunk_idx)

                pbar.update(1)

            if buffer:
                table = buffer.to_table(final_schema)
                writer.write_table(table)

    writing_output_file.rename(final_output_file)
    shutil.rmtree(temp_sort_dir, ignore_errors=True)
    return final_output_file

def digest(
    config: DigestConfig,
    fasta_path: Path,
    output_dir: Path,
    tokenizer: PTMPeptideTokenizer,
    bucket_config: BucketConfig,
    decoy_prefix: str="DECOY_",
    generate_decoy: bool=True,
    decoy_strategy: Literal["reverse", "shuffle", "fused"]="reverse",
    progress_bar: bool=True,
    num_workers: Optional[int]=None,
    worker_batch_size: int=50_000,
    sort_buffer_size: int=10_000
):
    logger.debug(f"Starting digestion with config: {config}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")
    db_file = fasta_path
    if generate_decoy:
        logger.warning("No decoy FASTA provided. Using the default decoy generation method (reverse sequence).")
        db_file = decoy(
            fasta_path,
            output_dir,
            decoy_prefix,
            decoy_strategy,
            progress_bar
        )
    
    logger.debug(f"Digesting {db_file}")
    valid_aa = set(c[0] for c in CANONICAL.keys() if c[0].isalpha())
    peptides_generator = _digest(
        db_file,
        config,
        valid_aa,
        decoy_prefix
    )

    fixed_mods, var_mods, swap_map = _construct_mods_dict(
        config.fixed_mods,
        config.var_mods
    )
    logger.debug(f"Fixed mods: {fixed_mods}")
    logger.debug(f"Var mods: {var_mods}")
    logger.debug(f"Swap map: {swap_map}")

    peptide_metadata_file = write_digested_peptides(
        peptides_generator,
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