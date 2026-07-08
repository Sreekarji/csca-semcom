import os
import sys
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from datetime import datetime
from collections import deque

sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")

from ddpm_policy import HDMPolicy, CriticNetwork
from han_network import HANNetwork
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_cscqi
from shaped_reward import compute_shaped_reward

LOG_PATH = r"D:\MP2\log.txt"
RESULTS_PATH = r"D:\MP2\results\software"
CHECKPOINT_PATH = r"D:\MP2\results\software\checkpoints"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class ReplayBuffer:
    """Experience replay buffer for sample-efficient training."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_emb, action, reward, next_state_emb):
        self.buffer.append({
            "state": state_emb.detach().cpu(),
            "action": action.detach().cpu(),
            "reward": torch.tensor([[reward]], dtype=torch.float),
            "next_state": next_state_emb.detach().cpu(),
        })

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            "state": torch.cat([b["state"] for b in batch]).to(DEVICE),
            "action": torch.cat([b["action"] for b in batch]).to(DEVICE),
            "reward": torch.cat([b["reward"] for b in batch]).to(DEVICE),
            "next_state": torch.cat([b["next_state"] for b in batch]).to(DEVICE),
        }

    def __len__(self):
        return len(self.buffer)


class CurriculumScheduler:
    """
    Progressive difficulty scheduler for HDM training.
    Phase 1 (0-200 ep): Easy — loose constraints, short distances
    Phase 2 (200-500 ep): Medium — tighter constraints
    Phase 3 (500+ ep): Hard — paper-equivalent constraints
    """

    def __init__(self):
        self.phase = 1

    def get_difficulty(self, episode: int) -> str:
        if episode < 200:
            return "easy"
        elif episode < 500:
            return "medium"
        else:
            return "hard"

    def get_env_params(self, episode: int) -> dict:
        difficulty = self.get_difficulty(episode)
        if difficulty == "easy":
            return {
                "delay_range": (1.0, 5.0),
                "quality_range": (0.2, 0.6),
                "data_size_range": (1e5, 5e5),
                "distance_range": (0.05, 0.2),
            }
        elif difficulty == "medium":
            return {
                "delay_range": (0.5, 3.0),
                "quality_range": (0.4, 0.8),
                "data_size_range": (5e5, 2e6),
                "distance_range": (0.1, 0.5),
            }
        else:  # hard
            return {
                "delay_range": (0.1, 1.5),
                "quality_range": (0.6, 0.95),
                "data_size_range": (1e6, 5e6),
                "distance_range": (0.2, 1.0),
            }

    def get_phase_name(self, episode: int) -> str:
        difficulty = self.get_difficulty(episode)
        names = {"easy": "Phase 1 (Easy)", "medium": "Phase 2 (Medium)", "hard": "Phase 3 (Hard)"}
        return names[difficulty]


class HDMTrainer:
    """
    Trains HDM using Actor-Critic as defined in Algorithm 2,
    Sun et al. 2026, Section V.
    With experience replay + curriculum learning + shaped reward.
    """

    def __init__(
        self,
        n_cscas: int = 5,
        n_relays: int = 5,
        n_mcs: int = 3,
        n_denoising_steps: int = 6,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        max_grad_norm: float = 1.0,
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_tasks = n_cscas
        self.n_relays = n_relays
        self.n_mcs = n_mcs
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        action_dim = n_cscas + n_cscas * n_relays + n_cscas * n_mcs

        self.han = HANNetwork(
            hidden_channels=128,
            num_heads=8,
            num_layers=2,
            n_cscas=n_cscas,
            n_relays=n_relays,
            n_messages=n_cscas,
            n_base_stations=n_cscas,
        ).to(self.device)

        self.actor = HDMPolicy(
            action_dim=action_dim,
            graph_emb_dim=128,
            n_denoising_steps=n_denoising_steps,
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=128,
            action_dim=action_dim,
        ).to(self.device)

        self.env = MultiCSCAEnvironment(
            n_cscas=n_cscas,
            n_relays=n_relays,
            n_base_stations=n_cscas,
            n_mcs=n_mcs,
            difficulty="hard",
        )

        self.opt_actor = optim.Adam(
            list(self.han.parameters()) + list(self.actor.parameters()),
            lr=lr_actor
        )
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # LR schedulers: warmup for 100 steps, then cosine decay
        self.scheduler_actor = SequentialLR(
            self.opt_actor,
            schedulers=[
                LinearLR(self.opt_actor, start_factor=0.1, end_factor=1.0, total_iters=100),
                CosineAnnealingLR(self.opt_actor, T_max=1900, eta_min=1e-5),
            ],
            milestones=[100]
        )
        self.scheduler_critic = SequentialLR(
            self.opt_critic,
            schedulers=[
                LinearLR(self.opt_critic, start_factor=0.1, end_factor=1.0, total_iters=100),
                CosineAnnealingLR(self.opt_critic, T_max=1900, eta_min=1e-5),
            ],
            milestones=[100]
        )

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.min_buffer_size = 64

        self.reward_history = []
        self.cscqi_history = []
        self.curriculum = CurriculumScheduler()
        self.current_episode = 0

        # Resume from existing checkpoint if available
        resume_ckpt = r"D:\MP2\results\software\checkpoints\hdm_ep5000.pt"
        if not os.path.exists(resume_ckpt):
            resume_ckpt = r"D:\MP2\results\software\checkpoints\hdm_ep500.pt"
        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device, weights_only=False)
            try:
                self.han.load_state_dict(ckpt["han"])
                self.actor.load_state_dict(ckpt["actor"])
                if "critic" in ckpt:
                    self.critic.load_state_dict(ckpt["critic"])
                if "reward_history" in ckpt:
                    self.reward_history = ckpt["reward_history"]
                log(f"Resumed from checkpoint: {resume_ckpt}")
            except Exception as e:
                log(f"Could not resume checkpoint: {e}. Starting fresh.")

        log(f"HDMTrainer initialized on {self.device}")

    def train_episode(self, system_state=None):
        self.han.train()
        self.actor.train()
        self.critic.train()

        # Curriculum: progressive difficulty
        self.current_episode += 1
        if system_state is None:
            params = self.curriculum.get_env_params(self.current_episode)
            system_state = self.env.generate_state_with_params(params)

        # Log phase transitions
        if self.current_episode in [1, 200, 500]:
            phase = self.curriculum.get_phase_name(self.current_episode)
            log(f"Curriculum: now in {phase}")

        # Encode state
        graph_emb, _ = self.han.encode_state(system_state)

        # Generate action
        action = self.actor(graph_emb)

        # Parse action
        bw = action[:, :self.n_tasks]
        relay = action[:, self.n_tasks:self.n_tasks + self.n_tasks * self.n_relays]
        relay = relay.reshape(1, self.n_tasks, self.n_relays)
        mcs = action[:, self.n_tasks + self.n_tasks * self.n_relays:]
        mcs = mcs.reshape(1, self.n_tasks, self.n_mcs)
        parsed_action = {"bandwidth": bw, "relay": relay, "mcs": mcs}

        # Execute through channel
        channel_result = self.env.step(parsed_action, system_state)
        tasks = channel_result["tasks"]

        # Compute shaped reward
        reward_value = compute_shaped_reward(tasks)

        # Get next state and encode
        next_params = self.curriculum.get_env_params(self.current_episode + 1)
        next_state = self.env.generate_state_with_params(next_params)
        next_graph_emb, _ = self.han.encode_state(next_state)

        # Push to replay buffer
        self.replay_buffer.push(graph_emb, action, reward_value, next_graph_emb)

        # If buffer not ready, just collect experience
        if len(self.replay_buffer) < self.min_buffer_size:
            self.reward_history.append(reward_value)
            return reward_value, 0.0, 0.0

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]

        # === CRITIC UPDATE on batch ===
        value_pred = self.critic(states, actions)
        critic_loss = nn.MSELoss()(value_pred, rewards)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()

        # === ACTOR UPDATE on batch ===
        new_actions = self.actor(states)
        value_new = self.critic(states, new_actions)

        # Entropy bonus for exploration
        entropy = -(new_actions * torch.log(new_actions + 1e-8) +
                    (1 - new_actions) * torch.log(1 - new_actions + 1e-8)).mean()
        actor_loss = -value_new.mean() - 0.01 * entropy

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.han.parameters()) + list(self.actor.parameters()),
            self.max_grad_norm
        )
        self.opt_actor.step()

        r = reward_value
        self.reward_history.append(r)
        self.cscqi_history.append(r)
        return r, critic_loss.item(), actor_loss.item()

    def train_batch_episode(self, batch_size: int = 8):
        """Train on a batch of different environment states simultaneously."""
        self.han.train()
        self.actor.train()
        self.critic.train()

        self.current_episode += 1

        # Log phase transitions
        if self.current_episode in [1, 200, 500]:
            phase = self.curriculum.get_phase_name(self.current_episode)
            log(f"Curriculum: now in {phase}")

        all_rewards = []
        all_graph_embs = []
        all_actions = []
        all_next_graph_embs = []

        # Collect experiences from batch_size different states
        for _ in range(batch_size):
            params = self.curriculum.get_env_params(self.current_episode)
            system_state = self.env.generate_state_with_params(params)
            graph_emb, _ = self.han.encode_state(system_state)
            action = self.actor(graph_emb)

            bw = action[:, :self.n_tasks]
            relay = action[:, self.n_tasks:self.n_tasks + self.n_tasks * self.n_relays].reshape(1, self.n_tasks, self.n_relays)
            mcs = action[:, self.n_tasks + self.n_tasks * self.n_relays:].reshape(1, self.n_tasks, self.n_mcs)
            parsed_action = {"bandwidth": bw, "relay": relay, "mcs": mcs}

            channel_result = self.env.step(parsed_action, system_state)
            tasks = channel_result["tasks"]
            reward_value = compute_shaped_reward(tasks)

            all_rewards.append(reward_value)
            all_graph_embs.append(graph_emb)
            all_actions.append(action)

            # Also push to replay buffer
            next_params = self.curriculum.get_env_params(self.current_episode + 1)
            next_state = self.env.generate_state_with_params(next_params)
            next_graph_emb, _ = self.han.encode_state(next_state)
            all_next_graph_embs.append(next_graph_emb)
            self.replay_buffer.push(graph_emb, action, reward_value, next_graph_emb)

        # Stack into batch tensors
        graph_embs = torch.cat(all_graph_embs, dim=0)
        actions = torch.cat(all_actions, dim=0)
        next_graph_embs = torch.cat(all_next_graph_embs, dim=0)
        rewards = torch.tensor(all_rewards, dtype=torch.float, device=self.device).unsqueeze(-1)

        # Eq. 34: cumulative discounted return RW^acc_t = R_t + gamma * V(s_{t+1})
        with torch.no_grad():
            next_values = self.critic(next_graph_embs.detach(), self.actor(next_graph_embs))
            targets = rewards + self.gamma * next_values
            targets = targets.detach()

        # Critic update: L_v = E[(RW^acc_t - V(s_t))^2]  (Eq. 35)
        value_pred = self.critic(graph_embs.detach(), actions.detach())
        critic_loss = nn.MSELoss()(value_pred, targets)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()

        # Actor update: L_pi = -E[Q(s, a)]  (approximation of Eq. 33)
        new_actions = self.actor(graph_embs)
        value_new = self.critic(graph_embs.detach(), new_actions)
        entropy = -(new_actions * torch.log(new_actions + 1e-8) +
                    (1 - new_actions) * torch.log(1 - new_actions + 1e-8)).mean()
        actor_loss = -value_new.mean() - 0.01 * entropy

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.han.parameters()) + list(self.actor.parameters()),
            self.max_grad_norm
        )
        self.opt_actor.step()

        mean_reward = float(np.mean(all_rewards))
        self.reward_history.append(mean_reward)

        # Step LR schedulers
        self.scheduler_actor.step()
        self.scheduler_critic.step()

        return mean_reward, critic_loss.item(), actor_loss.item()

    def train(self, max_episodes: int = 500, checkpoint_every: int = 100):
        log(f"Starting HDM training for {max_episodes} episodes on {self.device}")

        csv_path = os.path.join(RESULTS_PATH, "reward_curve.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "critic_loss", "actor_loss"])

        for ep in range(1, max_episodes + 1):
            reward, c_loss, a_loss = self.train_batch_episode(batch_size=8)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, reward, c_loss, a_loss])

            if ep % 50 == 0:
                current_lr = self.opt_actor.param_groups[0]['lr']
                log(f"Episode {ep}/{max_episodes} | Reward: {reward:.4f} | "
                    f"Critic: {c_loss:.4f} | Actor: {a_loss:.4f} | "
                    f"LR: {current_lr:.6f} | Buffer: {len(self.replay_buffer)}")

            if ep % checkpoint_every == 0:
                ckpt = os.path.join(CHECKPOINT_PATH, f"hdm_ep{ep}.pt")
                torch.save({
                    "episode": ep,
                    "han": self.han.state_dict(),
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "reward_history": self.reward_history,
                }, ckpt)
                log(f"Checkpoint saved: {ckpt}")

        log(f"Training complete. Final reward: {self.reward_history[-1]:.4f}")
        return self.reward_history



if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\MP2\code\utils")
    from reproducibility import set_seed
    set_seed(42)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("HDM TRAINING — 2000 EPISODES (RESUME)")
    print("=" * 60)

    trainer = HDMTrainer(n_denoising_steps=6)

    print(f"Buffer warmup: {trainer.min_buffer_size} episodes")
    print("Verifying actor loss is non-zero after warmup...")
    for i in range(trainer.min_buffer_size + 3):
        r, cl, al = trainer.train_episode()
        if i >= trainer.min_buffer_size:
            print(f"  ep {i+1}: reward={r:.4f}, critic={cl:.4f}, actor={al:.6f}")
            if abs(al) < 1e-6:
                raise ValueError(
                    f"Actor loss is zero at episode {i+1}. "
                    f"Fix the actor update in train_episode() before running full training."
                )
    print("Verification passed. Starting 2000 episode training...")
    print("Checkpoints saved every 200 episodes to:")
    print("  D:\\MP2\\results\\software\\checkpoints\\")
    print("=" * 60)

    rewards = trainer.train(max_episodes=2000, checkpoint_every=200)

    # Plot reward curve
    smoothed_50 = np.convolve(rewards, np.ones(50)/50, mode='valid')
    smoothed_100 = np.convolve(rewards, np.ones(100)/100, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full training curve
    axes[0].plot(rewards, alpha=0.2, linewidth=0.5, color='blue', label='Raw')
    axes[0].plot(range(49, len(rewards)), smoothed_50, 'r-',
                 linewidth=1.5, label='Smoothed (50-ep)', alpha=0.7)
    axes[0].plot(range(99, len(rewards)), smoothed_100, 'g-',
                 linewidth=2, label='Smoothed (100-ep)')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward (CSCQI)")
    axes[0].set_title("HDM Training — Full 5000 Episodes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Convergence region (last 1000 episodes)
    last_1000 = rewards[-1000:]
    smoothed_end = np.convolve(last_1000, np.ones(50)/50, mode='valid')
    axes[1].plot(last_1000, alpha=0.3, linewidth=0.5, color='blue', label='Raw (last 1000)')
    axes[1].plot(range(49, 1000), smoothed_end, 'r-', linewidth=2, label='Smoothed (50-ep)')
    axes[1].set_xlabel("Episode (last 1000 of 5000)")
    axes[1].set_ylabel("Reward (CSCQI)")
    axes[1].set_title("Convergence Region")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"D:\MP2\results\software\hdm_training_5000ep.png", dpi=120)
    plt.close()

    # Print final summary
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final reward (last 100 ep avg):  {np.mean(rewards[-100:]):.4f}")
    print(f"Final reward (last 500 ep avg):  {np.mean(rewards[-500:]):.4f}")
    print(f"Best reward:                     {max(rewards):.4f} at episode {rewards.index(max(rewards))+1}")
    print(f"Convergence check (last 500 std): {np.std(rewards[-500:]):.4f}")
    print(f"  (std < 0.05 suggests convergence)")
    print(f"Reward curve saved to: D:\\MP2\\results\\software\\hdm_training_5000ep.png")
    print("=" * 60)
