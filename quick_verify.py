"""Quick verification: HAN+MLP vs Static at tasks_per_csca=10 with hard negative reward."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")
sys.path.insert(0, r"D:\MP2\code\utils")
from reproducibility import set_seed
set_seed(42)

from mlp_trainer import MLPTrainer
trainer = MLPTrainer()
print(f"Training at tasks_per_csca=10, n_tasks={trainer.env.n_tasks}")

rewards = trainer.train(max_episodes=200, checkpoint_every=100)

# Static baseline comparison
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr
env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, tasks_per_csca=10)
static_isrs = []
for _ in range(100):
    state = env.generate_state()
    action = {"bandwidth": torch.ones(1,5)*0.2, "relay": torch.ones(1,5,5)*0.5, "mcs": torch.ones(1,5,3)*0.5}
    result = env.step(action, state)
    static_isrs.append(compute_isr(result["tasks"]))

print(f"\nFinal 50-ep MLP reward: {np.mean(rewards[-50:]):.4f}")
print(f"Static ISR at tasks_per_csca=10: {np.mean(static_isrs):.4f} +/- {np.std(static_isrs):.4f}")
print("If MLP reward > Static: hard negative reward + more tasks created a real learning signal")
