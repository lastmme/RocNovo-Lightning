from dataclasses import dataclass

@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    prob: float=0.5
    removal_rate: float=0.2
    removal_intensity_threshold: float=0.3
    perturbation_rate: float=0.15
    return_dummy_tensor: bool=True