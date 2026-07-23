"""Reward shaping."""
import numpy as np
from cscqi import compute_cscqi, compute_isr

def cscqi_reward(tasks: list, clip: float = 5.0) -> float:
    if not tasks:
        return 0.0
    vals = [compute_cscqi(t["tau_S"], t["vartheta_S"],
                          t["tau_S_int"], t["vartheta_S_int"]) for t in tasks]
    return float(np.clip(np.mean(vals), -clip, clip))

def isr_reward(tasks: list) -> float:
    return compute_isr(tasks)
