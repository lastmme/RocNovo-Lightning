import re
from collections.abc import Iterable

import numpy as np
from tqdm import tqdm
from spectrum_utils.utils import mass_diff
from sklearn.metrics import auc

def split_peptide(peptide: str) -> list[str]:
    peptide = peptide.replace("I", "L")
    parts = re.split(r"(?<=.)(?=[A-Z])", peptide)
    return parts

def aa_match_prefix(
    p1: list[str], p2: list[str], aa_dict: dict[str,float], ptm_list: list[str],
    cum_thresh: float = 0.5, ind_thresh: float = 0.1
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """
    Match prefix of two peptides by cumulative mass. Return arrays of matches and full-match flag.
    """
    max_len = max(len(p1), len(p2))
    matches = np.zeros(max_len, bool)
    ptm1 = np.zeros(max_len, bool)
    ptm2 = np.zeros(max_len, bool)
    i1 = i2 = 0
    cum1 = cum2 = 0.0
    while i1 < len(p1) and i2 < len(p2):
        m1 = aa_dict.get(p1[i1], 0)
        m2 = aa_dict.get(p2[i2], 0)
        if abs(mass_diff(cum1 + m1, cum2 + m2, True)) < cum_thresh:
            idx = max(i1, i2)
            if abs(mass_diff(m1, m2, True)) < ind_thresh:
                matches[idx] = True
                ptm1[idx] = p1[i1] in ptm_list
                ptm2[idx] = p2[i2] in ptm_list
            cum1 += m1; cum2 += m2; i1 += 1; i2 += 1
        elif cum2 + m2 > cum1 + m1:
            cum1 += m1; i1 += 1
        else:
            cum2 += m2; i2 += 1
    full = matches.all()
    return matches, full, ptm1, ptm2


def aa_match(
    p1: list[str], p2: list[str], aa_dict: dict[str,float], ptm_list: list[str],
    cum_thresh: float = 0.5, ind_thresh: float = 0.1, mode: str = "best"
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """
    Compute AA-level matches for two peptides. Uses prefix and then suffix matching if needed.
    """
    matches, full, ptm1, ptm2 = aa_match_prefix(p1, p2, aa_dict, ptm_list, cum_thresh, ind_thresh)
    if full:
        return matches, True, ptm1, ptm2
    i1, i2 = len(p1)-1, len(p2)-1
    stop = np.argwhere(~matches)[0][0]
    cum1 = cum2 = 0.0
    while i1 >= stop and i2 >= stop:
        m1 = aa_dict.get(p1[i1], 0); m2 = aa_dict.get(p2[i2], 0)
        if abs(mass_diff(cum1 + m1, cum2 + m2, True)) < cum_thresh:
            idx = max(i1, i2)
            if abs(mass_diff(m1, m2, True)) < ind_thresh:
                matches[idx] = True
                ptm1[idx] = p1[i1] in ptm_list
                ptm2[idx] = p2[i2] in ptm_list
            cum1 += m1; cum2 += m2; i1 -= 1; i2 -= 1
        elif cum2 + m2 > cum1 + m1:
            cum1 += m1; i1 -= 1
        else:
            cum2 += m2; i2 -= 1
    return matches, matches.all(), ptm1, ptm2


def aa_match_batch(
    gts: Iterable[str], preds: Iterable[str], aa_dict: dict[str,float],
    ptm_list: list[str], cum_thresh: float, ind_thresh: float, mode: str
) -> tuple[list[tuple[np.ndarray,bool,np.ndarray,np.ndarray]], int, int, int, int, int]:
    """
    Batch comparison of ground-truth and predicted peptide lists.

    Returns:
    - batch of (matches, full_match, ptm1, ptm2)
    - number of peptides
    - total true AA count
    - total predicted AA count
    - total true PTM count
    - total predicted PTM count
    """
    batch = []
    n_pep = len(list(gts))
    n_aa_true = sum(len(split_peptide(x)) for x in gts)
    n_aa_pred = 0
    n_ptm_true = n_ptm_pred = 0
    for gt, pr in tqdm(zip(gts, preds), total=len(gts)):
        seq_gt = split_peptide(gt)
        seq_pr = split_peptide(pr) if isinstance(pr, str) else []
        n_aa_pred += len(seq_pr)
        n_ptm_true += sum(aa in ptm_list for aa in seq_gt)
        n_ptm_pred += sum(aa in ptm_list for aa in seq_pr)
        if not seq_pr:
            batch.append((np.zeros(len(seq_gt), bool), False,
                          np.zeros(len(seq_gt), bool), np.zeros(len(seq_gt), bool)))
        else:
            batch.append(aa_match(seq_gt, seq_pr, aa_dict, ptm_list, cum_thresh, ind_thresh, mode))
    return batch, n_pep, n_aa_true, n_aa_pred, n_ptm_true, n_ptm_pred

def aa_match_metrics(
    batch: list[tuple[np.ndarray,bool,np.ndarray,np.ndarray]],
    n_pep_true: int, n_aa_true: int, n_aa_pred: int,
    n_ptm_true: int, n_ptm_pred: int, scores: list[float]
) -> dict[str, float]:
    """
    Compute evaluation metrics:
    - aa_precision, aa_recall
    - pep_precision, pep_recall
    - ptm_precision, ptm_recall
    - peptide-level PR AUC
    """
    # AA
    n_aa_corr = sum(m[0].sum() for m in batch)
    aa_precision = n_aa_corr / (n_aa_pred + 1e-8)
    aa_recall = n_aa_corr / (n_aa_true + 1e-8)
    # Peptide
    n_pep_corr = sum(m[1] for m in batch)
    pep_precision = n_pep_corr / (len(batch) + 1e-8)
    pep_recall = n_pep_corr / (n_pep_true + 1e-8)
    # PTM
    ptm_recall = sum(m[2].sum() for m in batch) / (n_ptm_true + 1e-8)
    ptm_precision = sum(m[3].sum() for m in batch) / (n_ptm_pred + 1e-8)
    # PR AUC
    bools = [m[1] for m in batch]
    combined = sorted(zip(scores, bools), key=lambda x: x[0], reverse=True)
    sorted_bools = [b for _, b in combined]
    prec_curve = np.cumsum(sorted_bools) / np.arange(1, len(sorted_bools) + 1)
    rec_curve = np.cumsum(sorted_bools) / n_pep_true
    curve_auc = auc(rec_curve, prec_curve)
    return {
        'aa_precision': aa_precision,
        'aa_recall': aa_recall,
        'pep_precision': pep_precision,
        'pep_recall': pep_recall,
        'ptm_precision': ptm_precision,
        'ptm_recall': ptm_recall,
        'curve_auc': curve_auc
    }