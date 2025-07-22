"""Utilities for reproducibility."""
import random

import numpy as np
import torch


_GLOBAL_SEED = 42


def get_seed() -> int:
    """Return seed value."""
    return _GLOBAL_SEED


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
