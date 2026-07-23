"""Multi-seed evaluation: HDM vs baselines vs Static.

Each method is evaluated on IDENTICAL states (deep copy) per seed.
Run: python joshi_eval_v2.py
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

from reproducibility import set_seed, get_seeds_for_evaluation
from hdm_trainer import HDMTrainer
from mlp_trainer import MLPTrainer
from train_baselines import BaselineTrainer
from sim_channel import MultiCSCAEnvironment
from ddpm_policy import TPC_TO_IDX
from cscqi import compute_isr

CKPT       = os.path.join(BASE, "checkpoints")
TPC_LIST   = [1, 2, 4, 6, 10]
N_EPISODES = 200

def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path

def _static_action(n_tasks, n_relays, n_mcs, device):
    return {
        "bandwidth": torch.ones(1, n_tasks, device=device),
        "relay":     torch.zeros(1, n_tasks, n_relays, device=device),
        "mcs":       torch.tensor([[[0.0, 1.0, 0.0]] * n_tasks], device=device),
    }

def evaluate_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hdm = HDMTrainer(difficulty="medium")
    hdm.load(_require(os.path.join(CKPT, "hdm_best.pt")))
    methods = {"HDM": hdm}

    for algo in ("sac", "ppo", "ac"):
        p = os.path.join(CKPT, f"{algo}_best.pt")
        if os.path.exists(p):
            t = BaselineTrainer(algo=algo, difficulty="medium",
                                hdm_checkpoint=os.path.join(CKPT, "hdm_best.pt"))
            t.load(p)
            methods[algo.upper()] = t

    mlp_p = os.path.join(CKPT, "mlp_best.pt")
    if os.path.exists(mlp_p):
        m = MLPTrainer(difficulty="medium")
        m.load(mlp_p)
        methods["MLP"] = m

    results = {name: {tpc: [] for tpc in TPC_LIST}
               for name in list(methods) + ["Static"]}

    for seed in get_seeds_for_evaluation(3):
        set_seed(seed)
        for tpc in TPC_LIST:
            cong       = TPC_TO_IDX.get(tpc, 0)
            per_method = {name: [] for name in results}

            for _ in range(N_EPISODES):
                env     = MultiCSCAEnvironment(difficulty="medium",
                                               tasks_per_csca=tpc)
                state   = env.generate_state()
                n_tasks = env.n_tasks
                intents = [[m[1], m[2]] for m in state["SCt"]["message_features"]]

                for name, tr in methods.items():
                    st = copy.deepcopy(state)
                    with torch.no_grad():
                        _, _, memb = tr.han.encode_state(st, intents)
                        if name in ("HDM", "MLP"):
                            act = tr.policy(memb, cong)
                        else:
                            act = tr.policy.act(memb)
                    out = env.step(act, st)
                    per_method[name].append(compute_isr(out["tasks"]))

                st  = copy.deepcopy(state)
                out = env.step(
                    _static_action(n_tasks, env.n_relays, env.n_mcs, device), st)
                per_method["Static"].append(compute_isr(out["tasks"]))

            for name in results:
                results[name][tpc].append(float(np.mean(per_method[name])))

    print("\n=== ISR (mean±std over 3 seeds, 200 eps/tpc) ===")
    header = "method   " + "  ".join(f"tpc{t:>2}" for t in TPC_LIST)
    print(header)
    for name in results:
        row = [
            f"{np.mean(results[name][t]):.3f}±{np.std(results[name][t]):.3f}"
            for t in TPC_LIST
        ]
        print(f"{name:8s} " + "  ".join(row))

    if "HDM" in results and "Static" in results:
        base  = np.mean([np.mean(results["Static"][t]) for t in TPC_LIST])
        hdm_m = np.mean([np.mean(results["HDM"][t])    for t in TPC_LIST])
        if base > 1e-6:
            print(f"\nHDM vs Static (avg over tpc): {(hdm_m - base) / base * 100:+.2f}%")
    return results

if __name__ == "__main__":
    evaluate_all()
