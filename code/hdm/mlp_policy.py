"""
MLP Actor-Critic policy using HAN graph embeddings.
Replaces DDPM with a simple 3-layer MLP for stable training.
This isolates HAN's contribution from DDPM training complexity.

Architecture:
- State: HAN graph embedding GL_t [256-dim] + per-task message embeddings [n_tasks x 256-dim]
- Actor: MLP(GL_t concat mean(message_embs)) -> action [45-dim]
- Critic: MLP(GL_t concat action) -> value [1-dim]
"""

import torch
import torch.nn as nn
import numpy as np


class MLPActor(nn.Module):
    """
    Simple MLP actor replacing DDPM.
    Takes graph embedding + task embeddings -> communication policy.
    """
    def __init__(
        self,
        graph_emb_dim: int = 256,
        task_emb_dim: int = 256,
        action_dim: int = 45,
        hidden_dim: int = 256,
        n_tasks: int = 5,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.action_dim = action_dim

        # Task-specific BW head
        # Input: per-task embedding [256] -> BW fraction [1]
        self.task_bw_head = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Global relay + MCS head
        # Input: graph embedding [256] -> relay+MCS actions
        relay_mcs_dim = action_dim - n_tasks
        self.global_head = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, relay_mcs_dim),
        )

        # Softmax temperature for BW allocation
        self.bw_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, graph_emb: torch.Tensor,
                message_embs: torch.Tensor = None):
        """
        Args:
            graph_emb: [batch, 256] graph embedding from HAN
            message_embs: [n_tasks, 256] per-task embeddings from HAN
        Returns:
            action: [batch, 45] communication policy
        """
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        # Task-specific BW allocation using per-task embeddings
        if message_embs is not None:
            # [n_tasks, 1] -> softmax -> [1, n_tasks]
            bw_logits = self.task_bw_head(message_embs)  # [n_tasks, 1]
            bw_alloc = torch.softmax(
                bw_logits.squeeze(-1) / self.bw_temperature.abs(), dim=0
            ).unsqueeze(0)  # [1, n_tasks]
        else:
            # Uniform if no message embeddings
            bw_alloc = torch.ones(
                graph_emb.shape[0], self.n_tasks, device=graph_emb.device
            ) / self.n_tasks

        # Global relay + MCS
        relay_mcs = torch.sigmoid(self.global_head(graph_emb))

        # Combine
        action = torch.cat([bw_alloc, relay_mcs], dim=-1)
        return action

    def parse_action(self, action, n_tasks=5, n_relays=5, n_mcs=3):
        bw = action[:, :n_tasks]
        relay = action[:, n_tasks:n_tasks + n_tasks * n_relays].reshape(
            action.shape[0], n_tasks, n_relays
        )
        mcs = action[:, n_tasks + n_tasks * n_relays:].reshape(
            action.shape[0], n_tasks, n_mcs
        )
        return {"bandwidth": bw, "relay": relay, "mcs": mcs}


class MLPCritic(nn.Module):
    def __init__(self, state_dim: int = 256, action_dim: int = 45,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor, action: torch.Tensor):
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if graph_emb.shape[0] != action.shape[0]:
            if graph_emb.shape[0] == 1:
                graph_emb = graph_emb.expand(action.shape[0], -1)
            elif action.shape[0] == 1:
                action = action.expand(graph_emb.shape[0], -1)
        return self.net(torch.cat([graph_emb, action], dim=-1))


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = MLPActor(action_dim=45, n_tasks=5).to(DEVICE)
    critic = MLPCritic(action_dim=45).to(DEVICE)

    graph_emb = torch.randn(1, 256, device=DEVICE)
    msg_embs = torch.randn(5, 256, device=DEVICE)

    action = actor(graph_emb, message_embs=msg_embs)
    value = critic(graph_emb, action)

    print(f"Action shape: {action.shape}")
    print(f"BW allocation: {action[0, :5].detach().cpu().numpy().round(3)}")
    print(f"BW sum: {action[0, :5].sum().item():.4f} (should be ~1.0)")
    print(f"Value: {value.item():.4f}")
    print("MLPActor test passed.")
