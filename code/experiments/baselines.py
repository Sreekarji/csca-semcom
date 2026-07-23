import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'channel'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_cscqi, compute_isr


def train_baseline(baseline, env, han, n_episodes=300, n_tasks=5, n_relays=5, n_mcs=3):
    """
    Generic training loop for SAC/AC/PPO baselines.
    Uses same environment and reward as HDM for fair comparison.
    han: the trained HANNetwork used to get state embeddings (frozen).
    """
    han.eval()
    device = baseline.device

    for ep in range(n_episodes):
        state = env.generate_state()
        with torch.no_grad():
            graph_emb, _, _ = han.encode_state(state)

        action = baseline.actor(graph_emb)

        bw = action[:, :n_tasks]
        relay = action[:, n_tasks:n_tasks + n_tasks * n_relays].reshape(1, n_tasks, n_relays)
        mcs = action[:, n_tasks + n_tasks * n_relays:].reshape(1, n_tasks, n_mcs)
        parsed = {"bandwidth": bw, "relay": relay, "mcs": mcs}

        result = env.step(parsed, state)
        tasks = result["tasks"]

        reward_val = float(np.mean([
            compute_cscqi(t["tau_S"], t["vartheta_S"],
                          t["tau_S_int"], t["vartheta_S_int"])
            for t in tasks
        ]))
        reward_tensor = torch.tensor([[reward_val]], dtype=torch.float, device=device)

        # Proper REINFORCE: maximize reward by gradient through action
        action_for_loss = baseline.actor(graph_emb)
        loss = -(reward_tensor.detach() * action_for_loss).mean()
        baseline.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(baseline.actor.parameters(), 1.0)
        baseline.opt.step()

        if ep % 50 == 0:
            isr = compute_isr(tasks)
            print(f"  [{baseline.__class__.__name__}] ep {ep}/{n_episodes} reward={reward_val:.3f} isr={isr:.3f}")


class SACBaseline:
    def __init__(self, action_dim=45, device=None, state_dim=256):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        ).to(self.device)
        self.opt = optim.Adam(self.actor.parameters(), lr=3e-4)

    def get_action(self, state_emb):
        with torch.no_grad():
            return self.actor(state_emb)


class ACBaseline:
    def __init__(self, action_dim=45, device=None, state_dim=256):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Sigmoid()
        ).to(self.device)
        self.opt = optim.Adam(self.actor.parameters(), lr=1e-3)

    def get_action(self, state_emb):
        with torch.no_grad():
            return self.actor(state_emb)


class PPOBaseline:
    def __init__(self, action_dim=45, device=None, state_dim=256):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        ).to(self.device)
        self.opt = optim.Adam(self.actor.parameters(), lr=3e-4)

    def get_action(self, state_emb):
        with torch.no_grad():
            return self.actor(state_emb)


class StaticBaseline:
    """Uniform BW allocation — the simplest baseline."""
    def __init__(self, action_dim=45, device=None, state_dim=256):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.action_dim = action_dim

    def get_action(self, state_emb):
        return torch.ones(1, self.action_dim, device=self.device) * 0.5
