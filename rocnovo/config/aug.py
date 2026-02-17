from dataclasses import dataclass

@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool  # 是否启用数据增强
    prob: float=0.5  # 数据增强的概率 (random >= p 则不启用)
    removal_rate: float=0.2  # 最多遮掩峰的比例
    removal_intensity_threshold: float=0.3  # 仅遮掩小于该阈值的峰
    perturbation_rate: float=0.15