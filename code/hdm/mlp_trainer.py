"""
MLP Trainer — HAN + MLP Actor-Critic training.
Replaces DDPM policy with simple MLP for stable training.
Tests whether HAN graph attention provides benefit independent of DDPM complexity.

Actor loss: DDPG style -Q(s,a) (simple and stable)
Critic loss: MSE (RW_acc - V(s))^2
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from datetime import datetime
from collections import deque

sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")
sys.path.insert(0, r"D:\MP2\code\utils")

from mlp_policy import MLPActor, MLPCritic
from han_network import HANNetwork
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_cscqi, compute_isr
from reproducibility import set_seed

LOG_PATH = r"D:\MP2\log.txt"
CHECKPOINT_PATH = r"D:\MP2\results\software\checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class MLPReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_emb, action, reward, next_state_emb):
        self.buffer.append({
            "state": state_emb.detach().cpu(),
            "action": action.detach().cpu(),
            "reward": torch.tensor([[reward]], dtype=torch.float),
            "next_state": next_state_emb.detach().cpu(),
        })

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = torch.cat([b["state"] for b in batch], dim=0).to(DEVICE)
        actions = torch.cat([b["action"] for b in batch], dim=0).to(DEVICE)
        rewards = torch.cat([b["reward"] for b in batch], dim=0).to(DEVICE)
        next_states = torch.cat([b["next_state"] for b in batch], dim=0).to(DEVICE)
        return {"state": states, "action": actions, "reward": rewards, "next_state": next_states}

    def __len__(self):
        return len(self.buffer)


class CurriculumScheduler:
    def __init__(self):
        self.phases = {
            "easy":   {"n_cscas": 5, "n_relays": 5},
            "medium": {"n_cscas": 5, "n_relays": 5},
        }

    def get_env_params(self, episode):
        if episode < 300:
            return self.phases["easy"]
        else:
            return self.phases["medium"]


class MLPTrainer:
    """
    HAN + MLP Actor-Critic trainer.
    Same HAN architecture as HDM trainer, but uses MLP instead of DDPM.
    """

    def __init__(self):
        self.device = DEVICE
        self.n_tasks = 5
        self.n_relays = 5
        self.n_mcs = 3
        self.action_dim = self.n_tasks + self.n_tasks * self.n_relays + self.n_tasks * self.n_mcs

        # HAN — SAME as HDM trainer
        self.han = HANNetwork(
            hidden_channels=256, num_heads=8, num_layers=3,
            n_cscas=self.n_tasks, n_relays=self.n_relays,
            n_messages=self.n_tasks, n_base_stations=self.n_tasks,
        ).to(self.device)

        # MLP actor — replaces DDPM
        self.actor = MLPActor(
            graph_emb_dim=256, task_emb_dim=256,
            action_dim=self.action_dim, hidden_dim=256,
            n_tasks=self.n_tasks,
        ).to(self.device)

        # MLP critic
        self.critic = MLPCritic(
            state_dim=256, action_dim=self.action_dim, hidden_dim=256,
        ).to(self.device)

        # Optimizers
        self.opt_actor = optim.Adam(
            list(self.han.parameters()) + list(self.actor.parameters()),
            lr=1e-3
        )
        self.opt_critic = optim.Adam(
            list(self.han.parameters()) + list(self.critic.parameters()),
            lr=1e-3
        )

        # Environment
        self.env = MultiCSCAEnvironment(n_cscas=self.n_tasks, n_relays=self.n_relays)
        self.curriculum = CurriculumScheduler()
        self.replay_buffer = MLPReplayBuffer(capacity=10000)

        # Training params
        self.gamma = 0.95
        self.batch_size = 256
        self.min_buffer_size = 64
        self.max_grad_norm = 1.0
        self.current_episode = 0
        self.reward_history = []
        self.best_isr = 0.0

        log(f"MLPTrainer initialized on {self.device}")

    def _generate_intent_vectors(self):
        intent_vectors = []
        for _ in range(self.n_tasks):
            urgency = np.random.uniform(0.4, 0.7)
            quality = np.random.uniform(0.4, 0.7)
            intent_vectors.append([urgency, quality])
        return intent_vectors

    def train_episode(self):
        """Single episode training with DDPG-style actor loss."""
        self.han.train()
        self.actor.train()
        self.critic.train()
        self.current_episode += 1

        # Generate intent
        intent_vectors = self._generate_intent_vectors()
        params = self.curriculum.get_env_params(self.current_episode)
        system_state = self.env.generate_state()
        system_state['SCt']['delay_intents'] = [max(0.1, (1.0 - iv[0]) * 5.0) for iv in intent_vectors]
        system_state['SCt']['quality_intents'] = [iv[1] for iv in intent_vectors]

        # HAN encoding
        graph_emb, node_embs, message_embs = self.han.encode_state(
            system_state, intent_vectors=intent_vectors
        )

        # Actor: generate action
        action = self.actor(graph_emb, message_embs=message_embs)

        # Parse action into dict format expected by environment
        action_np = action.detach().cpu().numpy()[0]
        bw = action_np[:self.n_tasks]
        relay = action_np[self.n_tasks:self.n_tasks + self.n_tasks * self.n_relays].reshape(1, self.n_tasks, self.n_relays)
        mcs = action_np[self.n_tasks + self.n_tasks * self.n_relays:].reshape(1, self.n_tasks, self.n_mcs)
        parsed_action = {
            "bandwidth": bw.reshape(1, self.n_tasks),
            "relay": relay,
            "mcs": mcs,
        }

        # Environment step
        result = self.env.step(parsed_action, system_state)
        tasks = result["tasks"]

        # Compute CSCQI reward (Eq. 17)
        cscqi_values = []
        for t in tasks:
            cscqi_values.append(compute_cscqi(
                t["tau_S"], t["vartheta_S"],
                t["tau_S_int"], t["vartheta_S_int"],
                w_tau=0.5, w_vartheta=0.5,
            ))
        reward_t = float(np.clip(np.mean(cscqi_values), -5.0, 5.0))
        isr = compute_isr(tasks)

        # Critic update
        value_pred = self.critic(graph_emb.detach(), action.detach())
        value_target = torch.tensor([[reward_t]], device=self.device, dtype=torch.float)
        critic_loss = nn.MSELoss()(value_pred, value_target)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()

        # Actor update — DDPG style: maximize Q(s,a)
        action_new = self.actor(graph_emb, message_embs=message_embs)
        q_value = self.critic(graph_emb.detach(), action_new)
        actor_loss = -q_value.mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.han.parameters()) + list(self.actor.parameters()),
            self.max_grad_norm
        )
        self.opt_actor.step()

        # Store in replay buffer (for batch training later if needed)
        # Use dummy next_state (same as state for simplicity)
        self.replay_buffer.push(graph_emb, action, reward_t, graph_emb)

        self.reward_history.append(reward_t)
        return reward_t, float(critic_loss.item()), float(actor_loss.item()), isr

    def train(self, max_episodes=500, checkpoint_every=100):
        log(f"Starting MLP training for {max_episodes} episodes on {self.device}")
        log("Architecture: HAN (3 layers, 256-dim) + MLP Actor (replacing DDPM)")

        for ep in range(1, max_episodes + 1):
            reward, critic_loss, actor_loss, isr = self.train_episode()

            if ep % 50 == 0 or ep == 1:
                avg_reward = np.mean(self.reward_history[-50:]) if self.reward_history else 0
                log(f"Episode {ep}/{max_episodes} | CSCQI: {reward:.4f} | ISR: {isr:.3f} | "
                    f"Critic: {critic_loss:.4f} | Actor: {actor_loss:.4f}")

            if ep % checkpoint_every == 0:
                ckpt = {
                    "han": self.han.state_dict(),
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "episode": ep,
                }
                path = os.path.join(CHECKPOINT_PATH, f"mlp_ep{ep}.pt")
                torch.save(ckpt, path)
                log(f"Checkpoint saved: {path}")

                if isr > self.best_isr:
                    self.best_isr = isr
                    best_path = os.path.join(CHECKPOINT_PATH, "mlp_best.pt")
                    torch.save(ckpt, best_path)
                    log(f"New best checkpoint: ep{ep}, ISR={isr:.3f}")

        return self.reward_history


if __name__ == "__main__":
    set_seed(42)

    print("=" * 60)
    print("MLP TRAINING — HAN + MLP Actor-Critic")
    print("Paper Table II: 500 episodes, batch=256, LR=0.001")
    print("=" * 60)

    trainer = MLPTrainer()
    rewards = trainer.train(max_episodes=500, checkpoint_every=100)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw', linewidth=0.8)
    plt.plot(range(19, len(rewards)), smoothed, 'r-', linewidth=2, label='Smoothed (20ep)')
    plt.xlabel("Episode")
    plt.ylabel("CSCQI Reward")
    plt.title("MLP Training — HAN + MLP Actor-Critic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(r"D:\MP2\results\software", "mlp_training.png"), dpi=120)
    plt.close()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final reward (last 50): {np.mean(rewards[-50:]):.4f}")
    print(f"Best reward: {max(rewards):.4f}")
    print("=" * 60)
