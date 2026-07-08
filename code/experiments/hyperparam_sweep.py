import sys, os
sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")

import torch
import numpy as np
import csv
from itertools import product
from hdm_trainer import HDMTrainer
from datetime import datetime

LOG_PATH = r"D:\MP2\log.txt"
RESULTS = r"D:\MP2\results\software"
os.makedirs(RESULTS, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# Hyperparameter grid
lr_values = [1e-4, 3e-4, 1e-3]
n_denoising_steps = [4, 6, 8]
hidden_dims = [128, 256]

QUICK_EPISODES = 100

results = []

for lr, N, hd in product(lr_values, n_denoising_steps, hidden_dims):
    log(f"Testing: lr={lr}, N={N}, hidden={hd}")

    try:
        trainer = HDMTrainer(
            n_denoising_steps=N,
            lr_actor=lr,
            lr_critic=lr,
        )

        rewards = []
        for ep in range(QUICK_EPISODES):
            r, cl, al = trainer.train_batch_episode(batch_size=8)
            rewards.append(r)

        final_reward = np.mean(rewards[-20:])
        results.append({
            "lr": lr, "N": N, "hidden": hd,
            "final_reward": final_reward,
            "best_reward": max(rewards),
        })
        log(f"  Result: final={final_reward:.4f}, best={max(rewards):.4f}")

    except Exception as e:
        log(f"  FAILED: {e}")
        results.append({"lr": lr, "N": N, "hidden": hd, "final_reward": -999, "best_reward": -999})

# Sort by final reward
results.sort(key=lambda x: x["final_reward"], reverse=True)

log("")
log("=== SWEEP RESULTS (top 5) ===")
for r in results[:5]:
    log(f"  lr={r['lr']}, N={r['N']}, hidden={r['hidden']}: reward={r['final_reward']:.4f} (best={r['best_reward']:.4f})")

best = results[0]
log(f"")
log(f"BEST CONFIG: lr={best['lr']}, N={best['N']}, hidden={best['hidden']}")
log(f"  final_reward={best['final_reward']:.4f}, best_reward={best['best_reward']:.4f}")
log(f"Use these settings for the final 2000-episode training run.")

# Save results
with open(os.path.join(RESULTS, "hyperparam_sweep.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lr", "N", "hidden", "final_reward", "best_reward"])
    writer.writeheader()
    writer.writerows(results)

log(f"Sweep results saved to {os.path.join(RESULTS, 'hyperparam_sweep.csv')}")
