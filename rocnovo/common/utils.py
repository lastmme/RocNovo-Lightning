import copy

import torch

from rocnovo.common.io import normalize_path
from rocnovo.module.denovo import Denovo

def _deep_merge_dicts(default_dict: dict, override_dict: dict) -> dict:
    merged = copy.deepcopy(default_dict)
    for k, v in override_dict.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    
    return merged

def load_denovo_from_checkpoint(
    ckpt_path: str,
    inference_config_overrides: dict,
    device: torch.device
) -> tuple["Denovo", dict]:
    ckpt_path = normalize_path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" in checkpoint:
        original_config = checkpoint["hyper_parameters"]
    else:
        raise KeyError("Could not find 'config' in the checkpoint hyper_parameters.")
    
    merged_config = _deep_merge_dicts(original_config, inference_config_overrides)
    model = Denovo.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        config=merged_config
    )
    
    return model.to(device).eval(), merged_config