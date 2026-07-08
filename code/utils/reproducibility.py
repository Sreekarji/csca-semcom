"""
Reproducibility utilities for CSCA experiments.
Set a fixed seed before any experiment to ensure reproducible results.
"""
import os
import random
import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    """Set random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[reproducibility] Seed set to {seed}")


def get_seeds_for_evaluation(n_seeds: int = 3) -> list:
    """Fixed seeds for multi-seed evaluation."""
    return [42, 123, 456][:n_seeds]


if __name__ == "__main__":
    set_seed(42)
    print("Test: random numbers with seed 42:")
    print("  numpy:", np.random.rand(3))
    print("  torch:", torch.rand(3).tolist())
    set_seed(42)
    print("Test: same seed gives same numbers:")
    print("  numpy:", np.random.rand(3))
    print("  torch:", torch.rand(3).tolist())
    print("Reproducibility confirmed.")
