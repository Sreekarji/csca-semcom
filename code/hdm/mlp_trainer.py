"""MLP trainer: HAN encoder + per-task MLP actor (HAN+MLP ablation).

Uses identical per-task credit assignment as HDMTrainer for a fair comparison.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from han_network import HANNetwork
from mlp_policy import MLPActor, TPC_TO_IDX
from ddpm_policy import ValueCritic
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr, compute_cscqi

class MLPTrainer:
    def __init__(self, n_cscas=5, n_relays=5, n_mcs=3, n_base_stations=5,
                 hidden=256, lr=3e-4, device=None, difficulty="medium",
                 tasks_schedule=None, ema_alpha=0.15):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_cscas   = n_cscas
        self.n_relays  = n_relays
        self.n_mcs     = n_mcs
        self.n_bs      = n_base_stations
        self.difficulty       = difficulty
        self.tasks_schedule   = tasks_schedule or [1, 1, 1, 2, 2, 4, 6, 10]
        self.ema_alpha        = ema_alpha
        self.reward_baseline  = 0.0
        self._warmup_count    = 0
        self._warmup_episodes = 50

        self.han    = HANNetwork(hidden, 8, 3, n_cscas, n_relays,
                                 n_cscas, n_base_stations).to(self.device)
        self.policy = MLPActor(hidden, hidden, n_relays, n_mcs).to(self.device)
        self.critic = ValueCritic(hidden).to(self.device)

        self.opt = torch.optim.Adam(
            list(self.han.parameters()) + list(self.policy.parameters()), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def _make_env(self, tpc):
        return MultiCSCAEnvironment(
            n_cscas=self.n_cscas, n_relays=self.n_relays,
            n_base_stations=self.n_bs, n_mcs=self.n_mcs,
            difficulty=self.difficulty, tasks_per_csca=tpc, sigma_s=8.0)

    def _update_baseline(self, reward: float):
        if self._warmup_count < self._warmup_episodes:
            self._warmup_count  += 1
            self.reward_baseline = (
                (self._warmup_count - 1) / self._warmup_count * self.reward_baseline
                + reward / self._warmup_count
            )
        else:
            self.reward_baseline = (
                (1 - self.ema_alpha) * self.reward_baseline
                + self.ema_alpha * reward
            )

    def train_batch_episode(self, batch_size=8, tasks_per_csca=1):
        congestion_idx = TPC_TO_IDX.get(tasks_per_csca, 0)

        all_lp   = []
        all_adv  = []
        vp_list  = []
        vt_list  = []
        ep_rewards = []
        ep_isrs    = []

        for _ in range(batch_size):
            env   = self._make_env(tasks_per_csca)
            state = env.generate_state()
            intents = [[m[1], m[2]] for m in state["SCt"]["message_features"]]
            graph_emb, _, message_embs = self.han.encode_state(state, intents)

            action, per_task_lp = self.policy.collect_trajectory(
                message_embs, congestion_idx)
            out = env.step(action, state)

            task_rewards = np.array([
                compute_cscqi(t["tau_S"], t["vartheta_S"],
                              t["tau_S_int"], t["vartheta_S_int"])
                for t in out["tasks"]
            ], dtype=np.float32)

            r_scalar = float(np.mean(task_rewards))
            self._update_baseline(r_scalar)

            task_adv = task_rewards - self.reward_baseline
            all_lp.append(per_task_lp.squeeze(0))                          # [Nm]
            all_adv.append(torch.tensor(task_adv, dtype=torch.float,
                                        device=self.device))
            vp_list.append(self.critic(graph_emb.detach()).squeeze())
            vt_list.append(torch.tensor(r_scalar, device=self.device))
            ep_rewards.append(r_scalar)
            ep_isrs.append(compute_isr(out["tasks"]))

        lp  = torch.stack(all_lp)    # [B, Nm]
        adv = torch.stack(all_adv)   # [B, Nm]

        adv_flat = adv.reshape(-1)
        adv_std  = adv_flat.std()
        if adv_std > 1e-6:
            adv = (adv - adv_flat.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv_flat.mean()

        actor_loss = -(lp * adv.detach()).mean()
        self.opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.han.parameters()) + list(self.policy.parameters()), 1.0)
        self.opt.step()

        vp          = torch.stack(vp_list)
        vt          = torch.stack(vt_list)
        critic_loss = self.mse(vp, vt)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return (float(np.mean(ep_rewards)), float(critic_loss.item()),
                float(actor_loss.item()), float(np.mean(ep_isrs)), tasks_per_csca)

    @torch.no_grad()
    def evaluate_isr(self, tpc_list=(1, 2, 4, 6, 10), n_episodes=50):
        self.han.eval(); self.policy.eval()
        results = {}
        for tpc in tpc_list:
            congestion_idx = TPC_TO_IDX.get(tpc, 0)
            isrs = []
            for _ in range(n_episodes):
                env   = self._make_env(tpc)
                state = env.generate_state()
                intents = [[m[1], m[2]] for m in state["SCt"]["message_features"]]
                _, _, message_embs = self.han.encode_state(state, intents)
                action = self.policy(message_embs, congestion_idx)
                out    = env.step(action, state)
                isrs.append(compute_isr(out["tasks"]))
            results[tpc] = (float(np.mean(isrs)), float(np.std(isrs)))
        self.han.train(); self.policy.train()
        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "han":             self.han.state_dict(),
            "policy":          self.policy.state_dict(),
            "critic":          self.critic.state_dict(),
            "reward_baseline": self.reward_baseline,
            "warmup_count":    self._warmup_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.han.load_state_dict(ckpt["han"])
        self.policy.load_state_dict(ckpt["policy"])
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
        self.reward_baseline = ckpt.get("reward_baseline", 0.0)
        self._warmup_count   = ckpt.get("warmup_count", self._warmup_episodes)
