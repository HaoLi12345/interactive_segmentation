"""Set all random seeds for reproducibility (R-Code.Seed)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import monai

        monai.utils.set_determinism(seed=seed)
    except ImportError:
        pass
