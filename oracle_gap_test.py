"""Oracle-gap sanity check.
Run: python oracle_gap_test.py
"""
import os
import sys
import copy
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ["", "code/hdm", "code/channel", "code/evaluation", "code/utils"]:
    p = os.path.join(BASE, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)

from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr

N_STATES        = 100
N_ORACLE_SAMPLES = 64
TPC_LIST        = [1, 2, 4, 10]

def _action_from_bw(bw_vec, n_relays, n_mcs, device):
    n = len(bw_vec)
    return {
        "bandwidth": torch.log(
            torch.tensor(bw_vec, dtype=torch.float, device=device) + 1e-8
        ).unsqueeze(0),
        "relay": torch.zeros(1, n, n_relays, device=device),
        "mcs":   torch.tensor([[[0.0, 1.0, 0.0]] * n], device=device),
    }

def main():
    device = "cpu"
    print(f"{'tpc':>4} {'uniform':>9} {'oracle':>9} {'gap':>9}")
    for tpc in TPC_LIST:
        unis, oracles = [], []
        for s in range(N_STATES):
            np.random.seed(1000 + s)
            env   = MultiCSCAEnvironment(difficulty="medium", tasks_per_csca=tpc)
            state = env.generate_state()
            n     = env.n_tasks

            st  = copy.deepcopy(state)
            uni = compute_isr(env.step(
                _action_from_bw([1.0] * n, env.n_relays, env.n_mcs, device), st
            )["tasks"])
            unis.append(uni)

            best = -1.0
            for k in range(N_ORACLE_SAMPLES):
                np.random.seed(2000 + s * 100 + k)
                bw  = np.random.dirichlet(np.ones(n))
                st  = copy.deepcopy(state)
                isr = compute_isr(env.step(
                    _action_from_bw(bw.tolist(), env.n_relays, env.n_mcs, device), st
                )["tasks"])
                best = max(best, isr)
            oracles.append(best)

        u, o = np.mean(unis), np.mean(oracles)
        print(f"{tpc:>4} {u:>9.3f} {o:>9.3f} {o - u:>+9.3f}")

if __name__ == "__main__":
    main()
