import torch
import numpy as np
import numpy.typing as npt
import spectrum_utils.spectrum as sus

import rocnovo.config.aug as aug_config

class SpectrumTokenizer:
    def __init__(
        self,
        n_top_peaks: int=150,
        min_mz: float=140.0,
        max_mz: float=2500.0,
        min_intensity: float=0.1,
        remove_precursor_tol: float=2.0,
    ):
        self.n_top_peaks = n_top_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.aug_config = None

    def disable_aug(self):
        self.aug_config = None

    def set_aug_config(self, aug_config: aug_config.AugmentationConfig):
        self.aug_config = aug_config

    def aug(self, mz: npt.NDArray, intensities: npt.NDArray):
        # 遮掩的候选峰
        candidate_indices = np.where(
            (intensities < self.aug_config.removal_intensity_threshold)
        )[0]

        removal_percent = self.aug_config.removal_rate

        removed_indices = np.random.choice(
            candidate_indices,
            int(
                np.floor(
                    removal_percent * len(candidate_indices)
                )
            ),
            replace=False
        )
        indices = np.arange(mz.shape[0], dtype=int)
        if len(removed_indices) > 0:
            indices = np.delete(indices, removed_indices)

        mz, intensities = mz[indices], intensities[indices]

        # 2. 随机对峰强度进行偏移 (每个峰偏移的比例不同，都是随机的)
        intensities = (
            1 - self.aug_config.perturbation_rate *
            2 * (np.random.random(intensities.shape)-0.5)
        ) * intensities

        intensities = intensities / intensities.max()
        return mz, intensities

    def tokenize(
        self,
        mz_array: npt.NDArray,
        int_array: npt.NDArray,
        precursor_mz: float,
        precursor_charge: int,
    ):
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz_array.astype(np.float64),
            int_array.astype(np.float32)
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError
            
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
        
            spectrum.filter_intensity(self.min_intensity, self.n_top_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError

            spectrum.scale_intensity("root", 1)
            mz, intensities = spectrum.mz, spectrum.intensity
            if self.aug_config is not None and self.aug_config.enabled and np.random.random() < self.aug_config.prob:
                mz, intensities = self.aug(mz, intensities)
            
            if self.aug_config is not None and self.aug_config.return_dummy_tensor:
                return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
            else:
                return torch.tensor(np.array([mz, intensities])).T.float()
        
        except ValueError:
            return torch.tensor([[0, 1]]).float()