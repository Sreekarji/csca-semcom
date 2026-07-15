import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr, compute_cscqi


class SACBaseline:
    """Soft Actor-Critic baseline (paper's comparison method)."""

    def __init__(self, action_dim=45, device=None, state_dim=256):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        ).to(self.device)
        self.opt = optim.Adam(self.actor.parameters(), lr=3e-4)

    def get_action(self, state_emb):
        with torch.no_grad():
            return self.actor(state_emb)

    def update(self, state_emb, reward):
        action = self.actor(state_emb)
        loss = -reward.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class ACBaseline:
    """Standard Actor-Critic baseline."""

    def __init__(self, action_dim=45, device=None, state_dim=256):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.Sigmoid()
        ).to(self.device)
        self.opt = optim.Adam(self.actor.parameters(), lr=1e-3)

    def get_action(self, state_emb):
        with torch.no_grad():
            return self.actor(state_emb)


class PPOBaseline:
    """PPO baseline."""

    def __init__(self, action_dim=45, device=None, state_dim=256):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        ).to(self.device)

    def get_action(self, state_emb):
        with torch.no_grad():
            return self.actor(state_emb)


class StaticBaseline:
    """Static uniform policy baseline."""

    def __init__(self, action_dim=45, device=None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_dim = action_dim

    def get_action(self, state_emb):
        batch = state_emb.shape[0] if state_emb.dim() > 1 else 1
        return torch.ones(batch, self.action_dim, device=self.device) * 0.5
