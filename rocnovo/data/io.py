from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from pyteomics.mgf import MGF
import h5py

class BaseParser(ABC):
    """A base parser class to inherit from.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level,
        valid_charge=None,
        id_type="scan",
    ):
        """Initialize the BaseParser"""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level
        self.valid_charge = None if valid_charge is None else set(valid_charge)
        self.id_type = id_type
        self.offset = None
        self.precursor_mz = []
        self.precursor_charge = []
        self.scan_id = []
        self.mz_arrays = []
        self.intensity_arrays = []

    @abstractmethod
    def open(self):
        """Open the file as an iterable"""
        pass

    @abstractmethod
    def parse_spectrum(self, spectrum):
        pass

    def read(self):
        n_skipped = 0
        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                try:
                    self.parse_spectrum(spectrum)
                except (IndexError, KeyError, ValueError):
                    n_skipped += 1
        
        self.precursor_mz = np.array(self.precursor_mz, dtype=np.float64)
        self.precursor_charge = np.array(
            self.precursor_charge,
            dtype=np.uint8,
        )

        self.scan_id = np.array(self.scan_id)

        # Build the index
        sizes = np.array([0] + [s.shape[0] for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()
        self.mz_arrays = np.concatenate(self.mz_arrays).astype(np.float64)
        self.intensity_arrays = np.concatenate(self.intensity_arrays).astype(np.float32)
        return self

    @property
    def n_spectra(self):
        """The number of spectra"""
        return self.offset.shape[0]

    @property
    def n_peaks(self):
        """The number of peaks in the file."""
        return self.mz_arrays.shape[0]


class MgfParser(BaseParser):
    """Parse mass spectra from an MGF file.

    Parameters
    ----------
    ms_data_file : str or Path
        The MGF file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level=2,
        valid_charge=None,
        annotations=False,
    ):
        """Initialize the MgfParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            id_type="index",
        )
        self.annotations = [] if annotations else None
        self._counter = -1

    def open(self):
        """Open the MGF file for reading"""
        return MGF(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in MGF format.
        """
        self._counter += 1

        if self.ms_level > 1:
            precursor_mz = float(spectrum["params"]["pepmass"][0])
            precursor_charge = int(spectrum["params"].get("charge", [0])[0])
        else:
            precursor_mz, precursor_charge = None, 0

        if self.annotations is not None:
            self.annotations.append(spectrum["params"].get("seq"))

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(spectrum["m/z array"])
            self.intensity_arrays.append(spectrum["intensity array"])
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(self._counter)
        else:
            raise ValueError("Invalid precursor charge")

BASECHARGE = [i for i in range(1, 11)]

def MgfToHdf5(mgf_path: str, hdf5_path: str, ms_level: int = 2, valid_charge: list[int] = BASECHARGE, annotated=False):
    parser = MgfParser(mgf_path, ms_level, valid_charge, annotated)
    parser.read()

    meta_types = [
        ("precursor_mz", np.float32),
        ("precursor_charge", np.uint8),
        ("offset", np.uint64),
        ("scan_id", np.uint32),
    ]
    metadata = np.empty(parser.n_spectra, dtype=meta_types)
    metadata["precursor_mz"] = parser.precursor_mz
    metadata["precursor_charge"] = parser.precursor_charge
    metadata["offset"] = parser.offset
    metadata["scan_id"] = parser.scan_id

    spectrum_types = [
        ("mz_array", np.float64),
        ("intensity_array", np.float32),
    ]

    spectra = np.zeros(parser.n_peaks, dtype=spectrum_types)
    spectra["mz_array"] = parser.mz_arrays
    spectra["intensity_array"] = parser.intensity_arrays

    with h5py.File(hdf5_path, "w") as index:
        index.attrs["ms_level"] = ms_level
        index.attrs["n_spectra"] = parser.n_spectra
        index.attrs["n_peaks"] = parser.n_peaks
        index.attrs["annotated"] = annotated

        group_index = len(index)
        group = index.create_group(str(group_index))
        group.attrs["path"] = str(mgf_path)
        group.attrs["n_spectra"] = parser.n_spectra
        group.attrs["n_peaks"] = parser.n_peaks
        group.attrs["id_type"] = parser.id_type

        # Add the datasets:
        group.create_dataset(
            "metadata",
            data=metadata,
        )

        group.create_dataset(
            "spectra",
            data=spectra,
        )

        try:
            group.create_dataset(
                "annotations",
                data=parser.annotations,
                dtype=h5py.string_dtype(),
            )
        except (KeyError, AttributeError):
            pass