from __future__ import annotations
import torch
import numpy as np

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def flatten_params(model) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def get_delta(global_model, local_model) -> torch.Tensor:
    g = flatten_params(global_model)
    l = flatten_params(local_model)
    return l - g

def robust_center_scale(x: np.ndarray):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # Ensure MAD is meaningful, use std as fallback
    if mad < 1e-6:
        mad = np.std(x) + 1e-6
    return med, mad
