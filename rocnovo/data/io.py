import re
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
from typing import Generator, Optional, Set, Iterable

import numpy as np
import numpy.typing as npt
import h5py
from tqdm import tqdm
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from pyteomics.mgf import MGF

from rocnovo.common.io import normalize_path
from rocnovo.common.logger import logger

@dataclass
class BufferItem:
    scan_id: str
    precursor_mz: float
    charge: int
    mz: npt.NDArray[np.float32]
    intensity: npt.NDArray[np.float32]

@dataclass
class AnnotatedBufferItem(BufferItem):
    annotated: str

@dataclass
class BatchBuffer:
    scan_id: list[str] = field(default_factory=list)
    precursor_mz: list[float] = field(default_factory=list)
    charge: list[int] = field(default_factory=list)
    mz: list[npt.NDArray[np.float32]] = field(default_factory=list)
    intensity: list[npt.NDArray[np.float32]] = field(default_factory=list)

    def append(self, item: BufferItem):
        for f in fields(item):
            getattr(self, f.name).append(getattr(item, f.name))

    def clear(self):
        for f in fields(self):
            getattr(self, f.name).clear()

    def __len__(self):
        return len(self.scan_id)

@dataclass
class AnnotatedBatchBuffer(BatchBuffer):
    annotated: list[str] = field(default_factory=list)

BASECHARGE = set(range(1, 11))

class BaseParser(ABC):
    def __init__(
        self,
        ms_data_file: str | Path,
        ms_level: int=2,
        valid_charge: Optional[Iterable[int]]=None,
        annotations: bool=False
    ):
        self.path = normalize_path(ms_data_file)
        if not self.path.exists():
            raise FileNotFoundError(f"The file path {self.path} is not existed.")

        self.ms_level = ms_level
        self.valid_charge = set(valid_charge) if valid_charge else None
        self.annotations = annotations

    @abstractmethod
    def open(self) -> MzML | MGF | MzXML:
        pass

    @abstractmethod
    def pre_traverse(self) -> Generator[int, None, None]:
        pass

    @abstractmethod
    def parse_spectrum(self, spectrum: dict) -> BufferItem | AnnotatedBufferItem:
        pass

    def iter_spectra(self) -> Generator[BufferItem | AnnotatedBufferItem, None, None]:
        skipped = 0
        with self.open() as spectra:
            for spectrum in spectra:
                try:
                    parsed = self.parse_spectrum(spectrum)
                    if parsed is not None:
                        yield parsed
                    else:
                        skipped += 1
                
                except (IndexError, KeyError, ValueError):
                    skipped += 1
                    continue
        
        logger.warning(f"Skip {skipped} spectra.")
        

class MgfParser(BaseParser):
    def open(self):
        return MGF(str(self.path))

    def pre_traverse(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            in_spectrum = False
            charge = 0
            current_peaks = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line == "BEGIN IONS":
                    in_spectrum = True
                    charge = 0
                    current_peaks = 0
                elif line == "END IONS":
                    if in_spectrum and (self.valid_charge is None or charge in self.valid_charge):
                        yield current_peaks
                    in_spectrum = False
                elif in_spectrum:
                    if '=' in line:
                        if line.startswith('CHARGE='):
                            c_str = line.split('=')[1].strip('+ -')
                            if c_str.isdigit():
                                charge = int(c_str)
                    elif line[0].isdigit():
                        current_peaks += 1

    def parse_spectrum(self, spectrum: dict):
        params = spectrum.get("params", {})
        
        precursor_mz = float(params.get("pepmass", [0.0])[0])
        precursor_charge = int(params.get("charge", [0])[0])

        if self.valid_charge and precursor_charge not in self.valid_charge:
            return None

        scan_id_str = str(params.get("scans", ""))
        if not scan_id_str:
            title = str(params.get("title", ""))
            match = re.search(r'scan[=\s]+(\w+)', title, re.IGNORECASE)
            if match:
                scan_id_str = match.group(1)

        base_kwargs = {
            "scan_id": scan_id_str,
            "precursor_mz": precursor_mz,
            "charge": precursor_charge,
            "mz": spectrum["m/z array"],
            "intensity": spectrum["intensity array"],
        }

        if self.annotations:
            return AnnotatedBufferItem(**base_kwargs, annotated=str(params.get("seq", "")))
        
        return BufferItem(**base_kwargs)

class MzMLParser(BaseParser):
    def open(self):
        return MzML(str(self.path))
    
    def pre_traverse(self) -> Generator[int, None, None]:
        context = ET.iterparse(str(self.path), events=("start", "end"))
        _, root = next(context)
        
        for event, elem in context:
            if event == "end" and elem.tag.split('}')[-1] == 'spectrum':
                ms_level = None
                charge = 0
                
                for child in elem.iter():
                    tag = child.tag.split('}')[-1]
                    if tag == 'cvParam':
                        acc = child.get('accession')
                        if acc == 'MS:1000511':
                            ms_level = int(child.get('value', 0))
                        elif acc == 'MS:1000041':
                            charge = int(child.get('value', 0))
                
                if ms_level == self.ms_level and (self.valid_charge is None or charge in self.valid_charge):
                    yield int(elem.get('defaultArrayLength', 0))
                
                elem.clear()
                root.clear()

    def parse_spectrum(self, spectrum: dict):
        if spectrum.get("ms level") != self.ms_level:
            return None

        precursors = spectrum.get("precursorList", {}).get("precursor", [])
        if not precursors:
            return None

        ion = precursors[0].get("selectedIonList", {}).get("selectedIon", [])
        if not ion:
            return None

        precursor_mz = float(ion[0].get("selected ion m/z", 0.0))
        precursor_charge = int(ion[0].get("charge state", 0))

        if self.valid_charge and precursor_charge not in self.valid_charge:
            return None

        scan_id_str = ""
        match = re.search(r'scan[=\s]+(\w+)', spectrum.get("id", ""), re.IGNORECASE)
        if match:
            scan_id_str = match.group(1)

        base_kwargs = {
            "scan_id": scan_id_str,
            "precursor_mz": precursor_mz,
            "charge": precursor_charge,
            "mz": spectrum["m/z array"],
            "intensity": spectrum["intensity array"],
        }
        
        return BufferItem(**base_kwargs)

class MzXMLParser(BaseParser):
    def open(self):
        return MzXML(str(self.path))

    def pre_traverse(self) -> Generator[int, None, None]:
        context = ET.iterparse(str(self.path), events=("start", "end"))
        _, root = next(context)
        
        for event, elem in context:
            if event == "end" and elem.tag.split('}')[-1] == 'scan':
                ms_level = int(elem.get('msLevel', 0))
                peaks_count = int(elem.get('peaksCount', 0))
                charge = 0
                
                for child in elem.iter():
                    if child.tag.split('}')[-1] == 'precursorMz':
                        charge = int(child.get('precursorCharge', 0))
                        break
                        
                if ms_level == self.ms_level and (self.valid_charge is None or charge in self.valid_charge):
                    yield peaks_count
                    
                elem.clear()
                root.clear()

    def parse_spectrum(self, spectrum: dict):
        if spectrum.get("msLevel") != self.ms_level:
            return None

        precursors = spectrum.get("precursorMz", [])
        if not precursors:
            return None

        precursor_mz = float(precursors[0].get("precursorMz", 0.0))
        precursor_charge = int(precursors[0].get("precursorCharge", 0))

        if self.valid_charge and precursor_charge not in self.valid_charge:
            return None

        scan_id_str = spectrum.get("num", "0")
        base_kwargs = {
            "scan_id": scan_id_str,
            "precursor_mz": precursor_mz,
            "charge": precursor_charge,
            "mz": spectrum["m/z array"],
            "intensity": spectrum["intensity array"],
        }

        return BufferItem(**base_kwargs)

def format_to_hdf5(
    ms_data_path: str | Path,
    output_dir: str | Path,
    ms_level: int=2,
    valid_charge: Set[int]=BASECHARGE,
    annotated: bool=False,
    batch_size: int=10_000
):
    ms_data_path = normalize_path(ms_data_path)
    ext = ms_data_path.suffix.lower()
    annotated = annotated & (ext == '.mgf')
    
    if ext == '.mgf':
        parser = MgfParser(ms_data_path, ms_level, valid_charge, annotated)
    elif ext == '.mzml':
        parser = MzMLParser(ms_data_path, ms_level, valid_charge, False)
    elif ext == '.mzxml':
        parser = MzXMLParser(ms_data_path, ms_level, valid_charge, False)
    else:
        raise ValueError(f"Unsupported extension {ext}, expected .mgf, .mzml, or .mzxml")
    
    logger.info(f"Pre-scanning {ms_data_path.name} to calculate total sizes...")
    n_spectra = 0
    n_peaks = 0
    start = time.time()
    for peak_num in tqdm(parser.pre_traverse(), "Pre-traverse"):
        n_spectra += 1
        n_peaks += peak_num
    
    end = time.time()
    logger.info(f"Pre-traverse operation consumes {end - start:.2f} seconds.")

    meta_dt = np.dtype([
        ("precursor_mz", np.float32),
        ("precursor_charge", np.uint8),
        ("offset", np.uint64),
        ("scan_id", h5py.string_dtype(encoding='utf-8')),
    ])
    
    spec_dt = np.dtype([
        ("mz_array", np.float32),
        ("intensity_array", np.float32),
    ])

    output_dir = normalize_path(output_dir)
    hdf5_path = output_dir.joinpath(ms_data_path.name).with_suffix(".hdf5")
    logger.info(f"The hdf5 file will be stored in the {output_dir}, dst file is {hdf5_path}")
    with h5py.File(hdf5_path, "w") as index:
        group = index.create_group("0")
        group.attrs["path"] = str(ms_data_path)
        group.attrs["ms_level"] = ms_level
        group.attrs["n_spectra"] = n_spectra
        group.attrs["n_peaks"] = n_peaks
        group.attrs["annotated"] = annotated
        meta_ds = group.create_dataset(
            "metadata",
            shape=(n_spectra,),
            dtype=meta_dt,
        )
        spec_ds = group.create_dataset(
            "spectra",
            shape=(n_peaks,),
            dtype=spec_dt
        )
        anno_ds = None

        if annotated:
            anno_ds = group.create_dataset(
                "annotations",
                shape=(n_spectra,),
                dtype=h5py.string_dtype(encoding='utf-8')
            )

        buffer = AnnotatedBatchBuffer() if annotated else BatchBuffer()

        current_spectra_idx = 0
        current_peaks_idx = 0
        def flush_batch():
            nonlocal current_spectra_idx, current_peaks_idx
            if len(buffer) == 0: 
                return

            n_new_meta = len(buffer)
            lengths = np.array([len(arr) for arr in buffer.mz], dtype=np.uint64)
            offsets = current_peaks_idx + np.cumsum(np.insert(lengths[:-1], 0, 0))
            
            m_arr = np.empty(n_new_meta, dtype=meta_dt)
            m_arr["precursor_mz"] = buffer.precursor_mz
            m_arr["precursor_charge"] = buffer.charge
            m_arr["offset"] = offsets
            m_arr["scan_id"] = buffer.scan_id

            s_arr = np.empty(lengths.sum(), dtype=spec_dt)
            s_arr["mz_array"] = np.concatenate(buffer.mz)
            s_arr["intensity_array"] = np.concatenate(buffer.intensity)
            
            n_new_peaks = len(s_arr)
            meta_ds[current_spectra_idx : current_spectra_idx + n_new_meta] = m_arr
            spec_ds[current_peaks_idx : current_peaks_idx + n_new_peaks] = s_arr

            if annotated and anno_ds is not None:
                a_arr = np.array(buffer.annotated, dtype=object)
                anno_ds[current_spectra_idx : current_spectra_idx + n_new_meta] = a_arr

            current_spectra_idx += n_new_meta
            current_peaks_idx += n_new_peaks
            buffer.clear()
        
        for item in tqdm(parser.iter_spectra(), desc="Extracting Spectra", total=n_spectra, dynamic_ncols=True):
            buffer.append(item)
            if len(buffer) >= batch_size:
                flush_batch()

        flush_batch()