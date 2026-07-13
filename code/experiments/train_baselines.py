import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")
sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\experiments")

from sim_channel import MultiCSCAEnvironment
from cscqi import compute_cscqi, compute_isr
from han_network import HANNetwork
from baselines import SACBaseline, ACBaseline, PPOBaseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = r"D:\MP2\results\software"
CKPT = r"D:\MP2\results\software\checkpoints"
LOG_PATH = r"D:\MP2\log.txt"
os.makedirs(CKPT, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def train_sac(n_episodes=2000):
    log("Training SAC baseline (v2 — calibrated env, 256/3/8 HAN)...")
    env = MultiCSCAEnvironment(n_cscas=5, n_relays=5)
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)

    action_dim = 45
    actor = nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, action_dim), nn.Sigmoid()
    ).to(DEVICE)

    critic1 = nn.Sequential(
        nn.Linear(256 + action_dim, 256), nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)

    critic2 = nn.Sequential(
        nn.Linear(256 + action_dim, 256), nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_c1 = optim.Adam(critic1.parameters(), lr=3e-4)
    opt_c2 = optim.Adam(critic2.parameters(), lr=3e-4)

    log_alpha = torch.tensor(0.0, requires_grad=True, device=DEVICE)
    opt_alpha = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -action_dim * 0.5

    rewards = []

    for ep in range(1, n_episodes + 1):
        state = env.generate_state()
        graph_emb, _, _ = han.encode_state(state)

        # Sample action with noise for exploration
        action = actor(graph_emb)
        noise = torch.randn_like(action) * 0.1
        action_noisy = (action + noise).clamp(0, 1)

        bw = action_noisy[:, :5]
        relay = action_noisy[:, 5:30].reshape(1, 5, 5)
        mcs = action_noisy[:, 30:].reshape(1, 5, 3)
        parsed = {"bandwidth": bw, "relay": relay, "mcs": mcs}

        result = env.step(parsed, state)
        tasks = result["tasks"]
        cscqi_vals = [compute_cscqi(t["tau_S"], t["vartheta_S"],
                                     t["tau_S_int"], t["vartheta_S_int"]) for t in tasks]
        reward = torch.tensor([[np.mean(cscqi_vals)]], dtype=torch.float, device=DEVICE)

        # Critic update
        with torch.no_grad():
            next_action = actor(graph_emb)
            next_val = torch.min(
                critic1(torch.cat([graph_emb, next_action], dim=-1)),
                critic2(torch.cat([graph_emb, next_action], dim=-1))
            )
            alpha = log_alpha.exp()
            entropy_bonus = -alpha * (next_action * torch.log(next_action + 1e-8)).sum(-1, keepdim=True)
            target_q = reward + 0.99 * (next_val + entropy_bonus)

        q1 = critic1(torch.cat([graph_emb.detach(), action_noisy.detach()], dim=-1))
        q2 = critic2(torch.cat([graph_emb.detach(), action_noisy.detach()], dim=-1))
        c_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        opt_c1.zero_grad()
        opt_c2.zero_grad()
        c_loss.backward()
        opt_c1.step()
        opt_c2.step()

        # Actor update
        action_new = actor(graph_emb)
        q_val = torch.min(
            critic1(torch.cat([graph_emb.detach(), action_new], dim=-1)),
            critic2(torch.cat([graph_emb.detach(), action_new], dim=-1))
        )
        alpha = log_alpha.exp()
        entropy = -(action_new * torch.log(action_new + 1e-8)).sum(-1, keepdim=True)
        a_loss = -(q_val + alpha * entropy).mean()

        opt_actor.zero_grad()
        a_loss.backward()
        opt_actor.step()

        # Alpha update
        alpha_loss = -(log_alpha * (entropy.detach() + target_entropy)).mean()
        opt_alpha.zero_grad()
        alpha_loss.backward()
        opt_alpha.step()

        rewards.append(float(reward.item()))

        if ep % 100 == 0:
            log(f"SAC Episode {ep}/{n_episodes} | Reward: {np.mean(rewards[-50:]):.4f}")

    torch.save({
        "actor": actor.state_dict(),
        "critic1": critic1.state_dict(),
        "critic2": critic2.state_dict(),
    }, os.path.join(CKPT, "sac_v2_trained.pt"))

    log(f"SAC training complete. Final reward: {np.mean(rewards[-50:]):.4f}")
    return actor


def train_ppo(n_episodes=2000):
    log("Training PPO baseline (v2 — calibrated env, 256/3/8 HAN)...")
    env = MultiCSCAEnvironment(n_cscas=5, n_relays=5)
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)

    action_dim = 45
    actor = nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, action_dim), nn.Sigmoid()
    ).to(DEVICE)

    critic = nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)

    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    clip_eps = 0.2
    rewards = []

    for ep in range(1, n_episodes + 1):
        state = env.generate_state()
        graph_emb, _, _ = han.encode_state(state)

        with torch.no_grad():
            old_action = actor(graph_emb)

        bw = old_action[:, :5]
        relay = old_action[:, 5:30].reshape(1, 5, 5)
        mcs = old_action[:, 30:].reshape(1, 5, 3)
        parsed = {"bandwidth": bw, "relay": relay, "mcs": mcs}
        result = env.step(parsed, state)
        tasks = result["tasks"]
        cscqi_vals = [compute_cscqi(t["tau_S"], t["vartheta_S"],
                                     t["tau_S_int"], t["vartheta_S_int"]) for t in tasks]
        reward = torch.tensor([[np.mean(cscqi_vals)]], dtype=torch.float, device=DEVICE)

        # PPO update — detach graph_emb so inner loop doesn't re-use the graph
        graph_det = graph_emb.detach()
        for _ in range(4):
            new_action = actor(graph_det)
            value = critic(graph_det)
            advantage = (reward - value.detach())

            ratio = (new_action / (old_action.detach() + 1e-8))
            ratio = ratio.mean(dim=-1, keepdim=True)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value, reward)
            loss = actor_loss + 0.5 * critic_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            opt.step()

        rewards.append(float(reward.item()))
        if ep % 100 == 0:
            log(f"PPO Episode {ep}/{n_episodes} | Reward: {np.mean(rewards[-50:]):.4f}")

    torch.save({"actor": actor.state_dict()}, os.path.join(CKPT, "ppo_v2_trained.pt"))
    log(f"PPO training complete. Final reward: {np.mean(rewards[-50:]):.4f}")
    return actor


def train_ac(n_episodes=2000):
    log("Training AC baseline (v2 — calibrated env, 256/3/8 HAN)...")
    env = MultiCSCAEnvironment(n_cscas=5, n_relays=5)
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)

    action_dim = 45
    actor = nn.Sequential(
        nn.Linear(256, 128), nn.Tanh(),
        nn.Linear(128, action_dim), nn.Sigmoid()
    ).to(DEVICE)
    critic = nn.Sequential(
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1)
    ).to(DEVICE)

    opt_a = optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = optim.Adam(critic.parameters(), lr=1e-3)
    rewards = []

    for ep in range(1, n_episodes + 1):
        state = env.generate_state()
        graph_emb, _, _ = han.encode_state(state)

        action = actor(graph_emb)
        bw = action[:, :5]
        relay = action[:, 5:30].reshape(1, 5, 5)
        mcs = action[:, 30:].reshape(1, 5, 3)
        parsed = {"bandwidth": bw, "relay": relay, "mcs": mcs}
        result = env.step(parsed, state)
        tasks = result["tasks"]
        cscqi_vals = [compute_cscqi(t["tau_S"], t["vartheta_S"],
                                     t["tau_S_int"], t["vartheta_S_int"]) for t in tasks]
        reward = torch.tensor([[np.mean(cscqi_vals)]], dtype=torch.float, device=DEVICE)

        value = critic(graph_emb.detach())
        advantage = reward - value.detach()

        c_loss = nn.MSELoss()(value, reward)
        opt_c.zero_grad()
        c_loss.backward()
        opt_c.step()

        action_new = actor(graph_emb.detach())
        a_loss = -(advantage * action_new.mean(dim=-1, keepdim=True)).mean()
        opt_a.zero_grad()
        a_loss.backward()
        opt_a.step()

        rewards.append(float(reward.item()))
        if ep % 100 == 0:
            log(f"AC Episode {ep}/{n_episodes} | Reward: {np.mean(rewards[-50:]):.4f}")

    torch.save({"actor": actor.state_dict()}, os.path.join(CKPT, "ac_v2_trained.pt"))
    log(f"AC training complete. Final reward: {np.mean(rewards[-50:]):.4f}")
    return actor

if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\MP2\code\utils")
    from reproducibility import set_seed
    set_seed(42)
    print("=" * 60)
    print("BASELINE TRAINING — SAC, PPO, AC")
    print("2000 episodes each")
    print("=" * 60)
    print("NOTE: Run this AFTER HDM training (hdm_trainer.py) is complete.")
    print("Checkpoints saved to:")
    print("  D:\\MP2\\results\\software\\checkpoints\\sac_v2_trained.pt")
    print("  D:\\MP2\\results\\software\\checkpoints\\ppo_v2_trained.pt")
    print("  D:\\MP2\\results\\software\\checkpoints\\ac_v2_trained.pt")
    print("=" * 60)

    start = datetime.now()

    print("\n[1/3] Training SAC...")
    train_sac(2000)
    print(f"SAC done. Elapsed: {datetime.now() - start}")

    print("\n[2/3] Training PPO...")
    train_ppo(2000)
    print(f"PPO done. Elapsed: {datetime.now() - start}")

    print("\n[3/3] Training AC...")
    train_ac(2000)
    print(f"AC done. Elapsed: {datetime.now() - start}")

    print("\n" + "=" * 60)
    print("ALL BASELINES TRAINED")
    print("Checkpoints saved:")
    print("  D:\\MP2\\results\\software\\checkpoints\\sac_v2_trained.pt")
    print("  D:\\MP2\\results\\software\\checkpoints\\ppo_v2_trained.pt")
    print("  D:\\MP2\\results\\software\\checkpoints\\ac_v2_trained.pt")
    print(f"Total time: {datetime.now() - start}")
    print("=" * 60)
    print("Next step: run final experiments")
