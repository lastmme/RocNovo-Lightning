import json
import pickle
from typing import Any
from pathlib import Path

import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def normalize_path(p: str | Path):
    if isinstance(p, str):
        p = Path(p)
    
    return p.absolute().resolve()

def load_yaml(yaml_path: str | Path) -> dict:
    yaml_path = normalize_path(yaml_path)
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def load_pkl(file_path: Path | str) -> dict:
    file_path = normalize_path(file_path)
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    
    return obj

def save_pkl(file_path: Path | str, obj: Any):
    file_path = normalize_path(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def save_json(file_path: Path | str, obj: Any):
    file_path = normalize_path(file_path)
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4)

def read_fasta(path: Path, n: int=1):
    if n <= 0:
        raise ValueError(f"n must be greater than 0, but got {n}")
    
    sequences: list[str] = []
    record: SeqRecord
    for record in SeqIO.parse(str(normalize_path(path)), "fasta"):
        sequences.append(str(record.seq))
    
    return sequences[:n]

def write_fasta(path: Path, seq: list[str]):
    path = normalize_path(path)
    for i, s in enumerate(seq):
        record = SeqRecord(
            Seq(s),
            id=f"seq_{i + 1}",
        )
        
        SeqIO.write(
            record,
            str(path),
            "fasta"
        )