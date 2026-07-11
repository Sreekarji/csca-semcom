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
from sim_channel import MultiCSCAEnvironment, HighPressureEnvironment
from cscqi import compute_cscqi, compute_isr
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

    def push(self, state_emb, action, reward, next_state_emb, old_log_prob=None):
        self.buffer.append({
            "state": state_emb.detach().cpu(),
            "action": action.detach().cpu(),
            "reward": torch.tensor([[reward]], dtype=torch.float),
            "next_state": next_state_emb.detach().cpu(),
            "old_log_prob": old_log_prob.detach().cpu() if old_log_prob is not None
                           else torch.tensor([[0.0]]),
        })

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            "state": torch.cat([b["state"] for b in batch]).to(DEVICE),
            "action": torch.cat([b["action"] for b in batch]).to(DEVICE),
            "reward": torch.cat([b["reward"] for b in batch]).to(DEVICE),
            "next_state": torch.cat([b["next_state"] for b in batch]).to(DEVICE),
            "old_log_prob": torch.cat([b["old_log_prob"] for b in batch]).to(DEVICE),
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
        if episode < 300:
            return "easy"
        else:
            return "medium"
        # Phase 3 (Hard) permanently disabled - too resource constrained,
        # degrades training. ep100 was best because it only saw Phase 1.

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
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
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
            hidden_channels=256,
            num_heads=8,
            num_layers=3,
            n_cscas=n_cscas,
            n_relays=n_relays,
            n_messages=n_cscas,
            n_base_stations=n_cscas,
        ).to(self.device)

        self.actor = HDMPolicy(
            action_dim=action_dim,
            graph_emb_dim=256,
            n_denoising_steps=n_denoising_steps,
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=256,
            action_dim=action_dim,
        ).to(self.device)

        self.env = HighPressureEnvironment(
            n_cscas=n_cscas,
            n_relays=n_relays,
            n_base_stations=n_cscas,
            n_mcs=n_mcs,
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
                LinearLR(self.opt_actor, start_factor=0.1, end_factor=1.0, total_iters=10),
                CosineAnnealingLR(self.opt_actor, T_max=490, eta_min=1e-5),
            ],
            milestones=[10]
        )
        self.scheduler_critic = SequentialLR(
            self.opt_critic,
            schedulers=[
                LinearLR(self.opt_critic, start_factor=0.1, end_factor=1.0, total_iters=10),
                CosineAnnealingLR(self.opt_critic, T_max=490, eta_min=1e-5),
            ],
            milestones=[10]
        )

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 256
        self.min_buffer_size = 64

        self.reward_history = []
        self.cscqi_history = []
        self.curriculum = CurriculumScheduler()
        self.current_episode = 0
        self.current_intent_vectors = None

        log(f"HDMTrainer initialized on {self.device}")

    def train_episode(self, system_state=None):
        self.han.train()
        self.actor.train()
        self.critic.train()

        # Curriculum: progressive difficulty
        self.current_episode += 1

        # Generate diverse intent vectors — mix of urgent and non-urgent
        intent_vectors = []
        for i in range(self.n_tasks):
            if i % 3 == 0:
                delay_urgency = np.random.uniform(0.7, 1.0)
                quality_req = np.random.uniform(0.3, 0.6)
            elif i % 3 == 1:
                delay_urgency = np.random.uniform(0.1, 0.4)
                quality_req = np.random.uniform(0.7, 1.0)
            else:
                delay_urgency = np.random.uniform(0.4, 0.7)
                quality_req = np.random.uniform(0.4, 0.7)
            intent_vectors.append([delay_urgency, quality_req])
        self.current_intent_vectors = intent_vectors

        if system_state is None:
            params = self.curriculum.get_env_params(self.current_episode)
            system_state = self.env.generate_state_with_params(params)

        # Update system state to match intent vectors
        system_state["SCt"]["delay_intents"] = [
            max(0.1, (1.0 - iv[0]) * 5.0) for iv in intent_vectors
        ]
        system_state["SCt"]["quality_intents"] = [iv[1] for iv in intent_vectors]

        # Log phase transitions
        if self.current_episode in [1, 200, 500]:
            phase = self.curriculum.get_phase_name(self.current_episode)
            log(f"Curriculum: now in {phase}")

        # Encode state with intent vectors
        graph_emb, node_embs, message_embs = self.han.encode_state(
            system_state, intent_vectors=intent_vectors
        )

        # Generate action and compute log_prob for DDPO-IS
        action, log_prob_current = self.actor.collect_trajectory(
            graph_emb, message_embs=message_embs
        )

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

        # Compute pure CSCQI reward (Eq. 17) — exactly as in paper
        cscqi_values = []
        for t in tasks:
            cscqi_values.append(compute_cscqi(
                t["tau_S"], t["vartheta_S"],
                t["tau_S_int"], t["vartheta_S_int"],
            ))
        reward_value = float(np.clip(np.mean(cscqi_values), -5.0, 5.0))
        # No shaping, no bonuses — pure Eq. 17

        # Get next state and encode
        next_params = self.curriculum.get_env_params(self.current_episode + 1)
        next_state = self.env.generate_state_with_params(next_params)
        next_graph_emb, _, next_message_embs = self.han.encode_state(next_state)

        # Push to replay buffer
        self.replay_buffer.push(graph_emb, action, reward_value, next_graph_emb, old_log_prob=log_prob_current.detach())

        # If buffer not ready, just collect experience
        if len(self.replay_buffer) < self.min_buffer_size:
            self.reward_history.append(reward_value)
            return reward_value, 0.0, 0.0, compute_isr(tasks)

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

        # === DDPO-IS ACTOR LOSS (PPO-style for diffusion) ===
        graph_emb_new, _, message_embs_new = self.han.encode_state(
            system_state, intent_vectors=self.current_intent_vectors
        )
        a_0_new, log_prob_new = self.actor.collect_trajectory(
            graph_emb_new, message_embs=message_embs_new
        )

        if len(self.replay_buffer) >= self.min_buffer_size:
            # Compute importance sampling ratio with buffer samples
            batch = self.replay_buffer.sample(min(8, self.batch_size))
            old_log_prob = batch["old_log_prob"][:8]

            # Compute new log_prob for buffer states
            batch_log_probs = []
            for i in range(min(8, batch["state"].shape[0])):
                single_state = batch["state"][i:i+1]
                _, lp = self.actor.collect_trajectory(single_state)
                batch_log_probs.append(lp)
            new_log_prob_batch = torch.cat(batch_log_probs, dim=0)
            old_log_prob_batch = old_log_prob[:len(batch_log_probs)]

            # Importance sampling ratio (PPO-style)
            log_ratio = new_log_prob_batch - old_log_prob_batch.detach()
            log_ratio = torch.clamp(log_ratio, -2.0, 2.0)
            ratio = torch.exp(log_ratio)

            # Advantage from buffer
            rewards_batch = batch["reward"][:len(batch_log_probs)]
            value_batch = self.critic(
                batch["state"][:len(batch_log_probs)],
                batch["action"][:len(batch_log_probs)]
            )
            advantage = (rewards_batch - value_batch.detach())
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = advantage.clamp(-3.0, 3.0)

            # PPO clipped objective
            eps = 0.2
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
        else:
            # Warmup: simple policy gradient
            value_est = self.critic(graph_emb_new.detach(), a_0_new.detach())
            advantage = torch.tensor([[reward_value]], device=self.device) - value_est.detach()
            actor_loss = -(log_prob_new * advantage.clamp(-1, 1)).mean()

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
        return r, critic_loss.item(), actor_loss.item(), compute_isr(tasks)

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
            # Generate diverse intent vectors
            intent_vectors = []
            for i in range(self.n_tasks):
                if i % 3 == 0:
                    du = np.random.uniform(0.7, 1.0)
                    qr = np.random.uniform(0.3, 0.6)
                elif i % 3 == 1:
                    du = np.random.uniform(0.1, 0.4)
                    qr = np.random.uniform(0.7, 1.0)
                else:
                    du = np.random.uniform(0.4, 0.7)
                    qr = np.random.uniform(0.4, 0.7)
                intent_vectors.append([du, qr])

            params = self.curriculum.get_env_params(self.current_episode)
            system_state = self.env.generate_state_with_params(params)
            system_state["SCt"]["delay_intents"] = [
                max(0.1, (1.0 - iv[0]) * 5.0) for iv in intent_vectors
            ]
            system_state["SCt"]["quality_intents"] = [iv[1] for iv in intent_vectors]

            graph_emb, node_embs, message_embs = self.han.encode_state(
                system_state, intent_vectors=intent_vectors
            )
            action, log_prob_current = self.actor.collect_trajectory(
                graph_emb, message_embs=message_embs
            )

            bw = action[:, :self.n_tasks]
            relay = action[:, self.n_tasks:self.n_tasks + self.n_tasks * self.n_relays].reshape(1, self.n_tasks, self.n_relays)
            mcs = action[:, self.n_tasks + self.n_tasks * self.n_relays:].reshape(1, self.n_tasks, self.n_mcs)
            parsed_action = {"bandwidth": bw, "relay": relay, "mcs": mcs}

            channel_result = self.env.step(parsed_action, system_state)
            tasks = channel_result["tasks"]
            cscqi_values = [
                compute_cscqi(t["tau_S"], t["vartheta_S"],
                              t["tau_S_int"], t["vartheta_S_int"])
                for t in tasks
            ]
            reward_value = float(np.clip(np.mean(cscqi_values), -5.0, 5.0))

            all_rewards.append(reward_value)
            all_graph_embs.append(graph_emb)
            all_actions.append(action)

            # Also push to replay buffer
            next_params = self.curriculum.get_env_params(self.current_episode + 1)
            next_state = self.env.generate_state_with_params(next_params)
            next_graph_emb, _, _ = self.han.encode_state(next_state)
            all_next_graph_embs.append(next_graph_emb)
            self.replay_buffer.push(graph_emb, action, reward_value, next_graph_emb, old_log_prob=log_prob_current.detach())

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

        # === DDPO-SF ACTOR LOSS ===
        last_graph_emb = all_graph_embs[-1]
        last_reward = all_rewards[-1]

        a_0_new, log_prob_new = self.actor.collect_trajectory(
            last_graph_emb, message_embs=None
        )

        val_est = self.critic(last_graph_emb.detach(), a_0_new.detach())
        advantage = torch.tensor([[last_reward]], dtype=torch.float, device=self.device) - val_est.detach()

        if not hasattr(self, 'advantage_mean'):
            self.advantage_mean = 0.0
            self.advantage_std = 1.0
            self.advantage_count = 0
        adv_val = float(advantage.item())
        self.advantage_count += 1
        self.advantage_mean += (adv_val - self.advantage_mean) / self.advantage_count
        self.advantage_std = max(abs(adv_val - self.advantage_mean), 0.1)
        advantage_norm = ((advantage - self.advantage_mean) / (self.advantage_std + 1e-8)).clamp(-5.0, 5.0)

        # Normalize log_prob with running statistics (avoids tanh saturation)
        lp_val = float(log_prob_new.mean().item())
        if not hasattr(self, 'lp_mean'):
            self.lp_mean = 0.0
            self.lp_std = 1.0
            self.lp_count = 0
        self.lp_count += 1
        self.lp_mean += (lp_val - self.lp_mean) / self.lp_count
        self.lp_std = max(abs(lp_val - self.lp_mean), 1.0)
        log_prob_norm = (log_prob_new - self.lp_mean) / (self.lp_std + 1e-8)
        log_prob_norm = log_prob_norm.clamp(-5.0, 5.0)
        actor_loss = -(log_prob_norm * advantage_norm).mean()
        actor_loss = torch.clamp(actor_loss, -10.0, 10.0)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.han.parameters()) + list(self.actor.parameters()),
            self.max_grad_norm
        )
        self.opt_actor.step()

        mean_reward = float(np.mean(all_rewards))
        self.reward_history.append(mean_reward)

        # Compute ISR from the last batch's tasks
        last_tasks = channel_result["tasks"] if 'channel_result' in dir() else []
        isr = compute_isr(last_tasks) if last_tasks else 0.0

        # Step LR schedulers
        self.scheduler_actor.step()
        self.scheduler_critic.step()

        return mean_reward, critic_loss.item(), actor_loss.item(), isr

    def train(self, max_episodes: int = 500, checkpoint_every: int = 100):
        log(f"Starting HDM training for {max_episodes} episodes on {self.device}")

        csv_path = os.path.join(RESULTS_PATH, "reward_curve.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "critic_loss", "actor_loss", "isr"])

        for ep in range(1, max_episodes + 1):
            reward, c_loss, a_loss, isr = self.train_batch_episode(batch_size=8)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, reward, c_loss, a_loss, isr])

            if ep % 50 == 0:
                current_lr = self.opt_actor.param_groups[0]['lr']
                log(f"Episode {ep}/{max_episodes} | CSCQI: {reward:.4f} | ISR: {isr:.3f} | "
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys, os, numpy as np
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, r"D:\MP2\code\utils")
    from reproducibility import set_seed
    set_seed(42)

    print("=" * 60)
    print("HDM TRAINING — DDPO-IS (PPO-style) + Phase 3 disabled")
    print("Paper Table II: 500 episodes, batch=256, LR=0.001")
    print("=" * 60)

    trainer = HDMTrainer(n_denoising_steps=6)

    # Warm up replay buffer
    print("Warming up replay buffer (64 episodes)...")
    for i in range(64):
        trainer.train_episode()
    print(f"Buffer ready: {len(trainer.replay_buffer)} samples")

    # Quick gradient check
    print("Gradient check (5 episodes with training)...")
    for i in range(5):
        r, cl, al, _ = trainer.train_episode()
        print(f"  ep {65+i}: reward={r:.4f}, critic={cl:.4f}, actor={al:.6f}")
        if np.isnan(al):
            print("FATAL: NaN in actor loss")
            sys.exit(1)
    print("Gradient check passed.")

    # Full training
    print("Starting 500 episode training...")
    rewards = trainer.train(max_episodes=500, checkpoint_every=100)

    # Training curve plot
    smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw', linewidth=0.8)
    plt.plot(range(19, len(rewards)), smoothed, 'r-', linewidth=2, label='Smoothed (20ep)')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Return")
    plt.title("HDM Training — DDPO-IS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r"D:\MP2\results\software\hdm_ddpois_training.png", dpi=120)
    plt.close()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final reward (last 50): {np.mean(rewards[-50:]):.4f}")
    print(f"Best reward: {max(rewards):.4f}")
    print(f"Plot saved: results/software/hdm_ddpois_training.png")
    print("=" * 60)

