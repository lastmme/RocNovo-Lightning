"""Amino acid masses and other useful mass spectrometry calculations"""

import re
from math import inf
from typing import Union, Literal

import torch
import numpy as np
import numpy.typing as npt

PAD = "PAD"
SOS = "SOS"

SPECIAL_TOKENS = {PAD: 0, SOS: 1}

CANONICAL = {
    "G": 57.021463735,
    "A": 71.037113805,
    "S": 87.032028435,
    "P": 97.052763875,
    "V": 99.068413945,
    "T": 101.047678505,
    "C+57.021": 103.009184505 + 57.02146,
    "L": 113.084064015,
    "I": 113.084064015,
    "N": 114.042927470,
    "D": 115.026943065,
    "Q": 128.058577540,
    "K": 128.094963050,
    "E": 129.042593135,
    "M": 131.040484645,
    "H": 137.058911875,
    "F": 147.068413945,
    # "U": 150.953633405,
    "R": 156.101111050,
    "Y": 163.063328575,
    "W": 186.079312980,
    # "O": 237.147726925,
}

MASSIVEKB = {
    # N-terminal mods:
    "+42.011": 42.010565,  # Acetylation
    "+43.006": 43.005814,  # Carbamylation
    "-17.027": -17.026549,  # NH3 loss
    "+43.006-17.027": (43.006814 - 17.026549),
    # AA mods:
    "M+15.995": CANONICAL["M"] + 15.994915,  # Met Oxidation
    "N+0.984": CANONICAL["N"] + 0.984016,  # Asn Deamidation
    "Q+0.984": CANONICAL["Q"] + 0.984016,  # Gln Deamidation
}

# Constants
H = 1.007825035
O = 15.99491463
N = 14.003074
H2O = 2 * H + O
PROTON = 1.00727646688
NH3 = 3 * H + N
ISOTOPE = 1.00335

class PTMPeptideTokenizer:
    """A simple class for calculating peptide masses

    Parameters
    ----------
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    # Modfications found in MassIVE-KB

    def __init__(
        self,
        residues: Union[dict[str, float], Literal["canonical", "massivekb"]] = "canonical",
        max_len: int = inf,
        reverse: bool = False,
    ):
        self.max_len = max_len
        self.reverse = reverse
        if isinstance(residues, dict):
            self.masses = residues
        else:
            if residues == "canonical":
                self.masses = CANONICAL.copy()
            elif residues == "massivekb":
                self.masses = CANONICAL.copy()
                self.masses.update(MASSIVEKB)

        self.initial_dictionary()
        self.init_mass_array()
        self.init_vocab_array()
        self._first_valid_token_idx = None
        self._vocab_mass_tensor = torch.from_numpy(self._vocab_mass_array).float()

    def initial_dictionary(self):
        vocabs = list(self.masses.keys())

        self._vocab2idx = {vocab: idx + len(SPECIAL_TOKENS) for idx, vocab in enumerate(vocabs)}
        self._vocab2idx.update(SPECIAL_TOKENS)

        self._idx2vocab = {idx: vocab for vocab, idx in self._vocab2idx.items()}

    def init_mass_array(self):
        self._vocab_mass_array = np.zeros((len(self._idx2vocab),))
        for idx, vocab in self._idx2vocab.items():
            if vocab in self.masses:
                self._vocab_mass_array[idx] = self.masses[vocab]

    def init_vocab_array(self):
        self._vocab_array = [""] * len(self._idx2vocab)
        for idx, vocab in self._idx2vocab.items():
            self._vocab_array[idx] = vocab

        self._vocab_array = np.array(self._vocab_array)

    @property
    def vocab2idx(self):
        return self._vocab2idx.copy()

    @property
    def idx2vocab(self):
        return self._idx2vocab.copy()

    @property
    def vocab_size(self):
        return len(self.idx2vocab)

    @property
    def first_valid_token_idx(self):
        if self._first_valid_token_idx is not None:
            return self._first_valid_token_idx

        for idx in range(self.vocab_size):
            if self._idx2vocab[idx] not in SPECIAL_TOKENS:
                self._first_valid_token_idx = idx
                break

        return self._first_valid_token_idx

    def detokenize_by_array(self, tokens_seq: npt.NDArray):
        """
        Parameters:
        ---
        tokens_seq: [batch_size, max_len]

        Returns:
        ---
        sequences: [batch_size, max_len]
        """
        tokenized_sequences = self._vocab_array[tokens_seq]
        sos_mask = tokenized_sequences == SOS

        sequences = []

        for mask, seq in zip(sos_mask, tokenized_sequences):
            sos_positions = np.where(mask)[0]
            if len(sos_positions) >= 2:
                start = sos_positions[0] + 1
                end = sos_positions[1]
            elif len(sos_positions) == 1:
                start = sos_positions[0] + 1
                end = len(seq)
            else:
                # 没有 <SOS>，直接返回整个序列
                start = 0
                end = len(seq)
            seq_tojoin = seq[start:end]
            if self.reverse:
                seq_tojoin = seq_tojoin[::-1]
            sequences.append("".join(seq_tojoin))

        return sequences
    
    def reverse_detokenize_by_array(self, tokens_seq: npt.NDArray):
        """
        Parameters:
        ---
        tokens_seq: [batch_size, max_len]

        Returns:
        ---
        sequences: [batch_size, max_len]
        """
        tokenized_sequences = self._vocab_array[tokens_seq]
        sos_mask = tokenized_sequences == SOS

        sequences = []

        for mask, seq in zip(sos_mask, tokenized_sequences):
            sos_positions = np.where(mask)[0]
            if len(sos_positions) >= 2:
                start = sos_positions[0] + 1
                end = sos_positions[1]
            elif len(sos_positions) == 1:
                start = sos_positions[0] + 1
                end = len(seq)
            else:
                # 没有 <SOS>，直接返回整个序列
                start = 0
                end = len(seq)
            seq_tojoin = seq[start:end]
            if not self.reverse:
                seq_tojoin = seq_tojoin[::-1]
            sequences.append("".join(seq_tojoin))

        return sequences


    def tokenize(self, seq: str):
        seq = seq.replace("I", "L")
        seq = re.split(r"(?<=.)(?=[A-Z])", seq)
        if self.reverse:
            seq = seq[::-1]

        # 如果超出最大分析长度, 则进行截断
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        seq = [SOS] + seq + [SOS]
        return torch.LongTensor([self.vocab2idx[item] for item in seq])

    def reverse_tokenize(self, seq: str):
        seq = seq.replace("I", "L")
        seq = re.split(r"(?<=.)(?=[A-Z])", seq)
        if not self.reverse:
            seq = seq[::-1]
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        seq = [SOS] + seq + [SOS]
        return torch.LongTensor([self.vocab2idx[item] for item in seq])

    def get_valid_tokens(self, tokens: torch.LongTensor):
        sos_token_id = self.vocab2idx[SOS]
        if isinstance(tokens, torch.Tensor):
            sos_positions = (tokens == sos_token_id).nonzero(as_tuple=True)[0]
        elif isinstance(tokens, np.ndarray):
            sos_positions = (tokens == sos_token_id).nonzero()[0]

        if len(sos_positions) >= 2:
            start = sos_positions[0] + 1
            end = sos_positions[1]
        elif len(sos_positions) == 1:
            start = sos_positions[0] + 1
            end = len(tokens)
        else:
            # 没有 <SOS>，直接返回整个序列
            start = 0
            end = len(tokens)

        return tokens[start:end]

    def detokenize(self, tokens: torch.LongTensor):
        valid_tokens = self.get_valid_tokens(tokens)

        if self.reverse:
            valid_tokens = valid_tokens.flip(0)

        return [self.idx2vocab[int(item)] for item in valid_tokens]

    def reverse_detokenize(self, tokens: torch.LongTensor):
        valid_tokens = self.get_valid_tokens(tokens)

        if not self.reverse:
            valid_tokens = valid_tokens.flip(0)

        return [self.idx2vocab[int(item)] for item in valid_tokens]

    def detokenize_seqence(self, tokens: torch.LongTensor):

        return "".join(self.detokenize(tokens))

    def tokens2mz(self, tokens: torch.LongTensor, charge: torch.LongTensor):
        seq = self.detokenize_seqence(tokens)
        mz = self.cal_seq_mz(seq, int(charge))
        return mz

    def tokens2mass(self, tokens: torch.LongTensor):
        seq = self.detokenize_seqence(tokens)
        mass = self.cal_seq_mass(seq)
        return mass

    def mass2mz(self, mass: float, charge: int):
        return mass / charge + PROTON

    def mz2mass(self, mz: float, charge: int):
        return (mz - PROTON) * charge

    def cal_seq_mz(self, seq: str, charge: int):
        return self._mass(seq, charge)

    def cal_seq_mass(self, seq: str):
        return self._mass(seq)

    def array2mass(self, tokens_seq: npt.NDArray):
        return np.sum(self._vocab_mass_array[tokens_seq], axis=1)
    
    def _mass(self, seq, charge=None):
        """Calculate a peptide's mass or m/z.

        Parameters
        ----------
        seq : list or str
            The peptide sequence, using tokens defined in ``self.residues``.
        charge : int, optional
            The charge used to compute m/z. Otherwise the neutral peptide mass
            is calculated

        Returns
        -------
        float
            The computed mass or m/z.
        """
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + H2O
        if charge is not None:
            calc_mass = (calc_mass / charge) + PROTON

        return calc_mass

    def tokens_to_prefix_masses_torch(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        mass_lookup = self._vocab_mass_tensor.to(tokens_seq.device)
        mass_seq = mass_lookup[tokens_seq]
        prefix_mass_seq = torch.cumsum(mass_seq, dim=1)

        return prefix_mass_seq

    def fragment_masses_to_loss_masses_torch(
        self, Fragment_masses: torch.Tensor, loss_type: Literal["noloss", "H2O", "NH3"] = "noloss"
    ):
        if loss_type == "noloss":
            return Fragment_masses
        elif loss_type == "H2O":
            return Fragment_masses - H2O
        elif loss_type == "NH3":
            return Fragment_masses - NH3