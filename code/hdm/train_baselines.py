"""Baseline RL agents: SAC, PPO, AC — with per-task credit assignment.

All three use:
  - Frozen HAN from HDM checkpoint (fair features)
  - Shared-weight per-task Gaussian actor (same interface as HDM/MLP)
  - Per-task advantage = task_cscqi_i - V(s)

SAC adds learnable entropy bonus.
PPO adds clipped surrogate over K epochs.
AC uses pure REINFORCE with critic baseline.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from han_network import HANNetwork
from ddpm_policy import ValueCritic, TPC_TO_IDX
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr, compute_cscqi

_HALF_LOG_2PIE = 0.5 * np.log(2 * np.pi * np.e)

class PerTaskGaussianActor(nn.Module):
    """Shared-weight per-task actor with LEARNABLE-sigma Gaussian BW head."""

    def __init__(self, task_emb_dim=256, hidden=256, n_relays=5, n_mcs=3,
                 bw_sigma=0.3):
        super().__init__()
        self.n_relays     = n_relays
        self.n_mcs        = n_mcs
        self.per_task_out = 1 + n_relays + n_mcs
        self.net = nn.Sequential(
            nn.Linear(task_emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),       nn.ReLU(),
            nn.Linear(hidden, self.per_task_out),
        )
        self.log_sigma = nn.Parameter(torch.full((1,), float(np.log(bw_sigma))))

    def _to_action(self, bw, rest):
        return {
            "bandwidth": bw.unsqueeze(0),
            "relay":     torch.sigmoid(rest[:, :self.n_relays]).unsqueeze(0),
            "mcs":       torch.sigmoid(rest[:, self.n_relays:]).unsqueeze(0),
        }

    @torch.no_grad()
    def act(self, message_embs):
        out = self.net(message_embs)
        return self._to_action(out[:, 0], out[:, 1:])

    @torch.no_grad()
    def sample(self, message_embs):
        """Draw action; return (action, bw_sample [Nm], rest [Nm, ·])."""
        out      = self.net(message_embs)
        bw_mean  = out[:, 0]
        sigma    = torch.exp(self.log_sigma)
        bw       = bw_mean + sigma * torch.randn_like(bw_mean)
        return self._to_action(bw, out[:, 1:]), bw.detach(), out[:, 1:].detach()

    def log_prob_entropy_pertask(self, message_embs, bw_sample):
        """Per-task log-prob [Nm] + scalar entropy; both carry grad."""
        out      = self.net(message_embs)
        bw_mean  = out[:, 0]                             # [Nm]
        sigma    = torch.exp(self.log_sigma)
        # per-task log-prob [Nm]
        per_task_lp = (-0.5 * ((bw_sample - bw_mean) / sigma) ** 2
                       - self.log_sigma.squeeze())       # [Nm]
        entropy     = (_HALF_LOG_2PIE + self.log_sigma).squeeze()  # scalar
        return per_task_lp, entropy

class BaselineTrainer:
    def __init__(self, algo="ac", n_cscas=5, n_relays=5, n_mcs=3,
                 n_base_stations=5, hidden=256, lr=3e-4, device=None,
                 difficulty="medium", hdm_checkpoint=None,
                 ppo_clip=0.2, ppo_epochs=4, sac_alpha=0.05):
        assert algo in ("ac", "ppo", "sac")
        self.algo       = algo
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_cscas    = n_cscas
        self.n_relays   = n_relays
        self.n_mcs      = n_mcs
        self.n_bs       = n_base_stations
        self.difficulty = difficulty
        self.ppo_clip   = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.sac_alpha  = sac_alpha

        self.han = HANNetwork(hidden, 8, 3, n_cscas, n_relays,
                              n_cscas, n_base_stations).to(self.device)
        if hdm_checkpoint is None or not os.path.exists(hdm_checkpoint):
            raise FileNotFoundError(
                f"HDM checkpoint required for shared HAN features: {hdm_checkpoint}")
        ckpt = torch.load(hdm_checkpoint, map_location=self.device)
        self.han.load_state_dict(ckpt["han"])
        for p in self.han.parameters():
            p.requires_grad = False
        self.han.eval()

        self.policy = PerTaskGaussianActor(
            hidden, hidden, n_relays, n_mcs).to(self.device)
        self.critic = ValueCritic(hidden).to(self.device)
        self.opt        = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def _make_env(self, tpc):
        return MultiCSCAEnvironment(
            n_cscas=self.n_cscas, n_relays=self.n_relays,
            n_base_stations=self.n_bs, n_mcs=self.n_mcs,
            difficulty=self.difficulty, tasks_per_csca=tpc, sigma_s=8.0)

    def train_batch_episode(self, batch_size=8, tasks_per_csca=1):
        # ---- Collect transitions ----
        membs_list   = []
        bw_samples   = []
        graph_embs   = []
        task_rews    = []   # list of [Nm] arrays
        ep_isrs      = []
        mean_rewards = []

        for _ in range(batch_size):
            env   = self._make_env(tasks_per_csca)
            state = env.generate_state()
            intents = [[m[1], m[2]] for m in state["SCt"]["message_features"]]
            with torch.no_grad():
                graph_emb, _, message_embs = self.han.encode_state(state, intents)
            action, bw_sample, _ = self.policy.sample(message_embs)
            out = env.step(action, state)

            tr = np.array([
                compute_cscqi(t["tau_S"], t["vartheta_S"],
                              t["tau_S_int"], t["vartheta_S_int"])
                for t in out["tasks"]
            ], dtype=np.float32)

            task_rews.append(tr)
            mean_rewards.append(float(np.mean(tr)))
            ep_isrs.append(compute_isr(out["tasks"]))
            membs_list.append(message_embs.detach())
            bw_samples.append(bw_sample)
            graph_embs.append(graph_emb.detach())

        # ---- Per-task advantages using V(s) critic ----
        ge = torch.cat(graph_embs, dim=0)                     # [B, H]
        with torch.no_grad():
            values = self.critic(ge).squeeze(-1)              # [B]

        # advantages: [B, Nm] = task_reward_i - V(s)
        all_adv = []
        for i in range(batch_size):
            adv_i = torch.tensor(task_rews[i], dtype=torch.float,
                                 device=self.device) - values[i]
            all_adv.append(adv_i)
        adv = torch.stack(all_adv)                            # [B, Nm]

        adv_flat = adv.reshape(-1)
        adv_std  = adv_flat.std()
        if adv_std > 1e-6:
            adv = (adv - adv_flat.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv_flat.mean()
        adv = adv.detach()

        # Old log-probs for PPO ratio (per-task) [B, Nm]
        with torch.no_grad():
            old_lp_list = [
                self.policy.log_prob_entropy_pertask(membs_list[i], bw_samples[i])[0]
                for i in range(batch_size)
            ]
        old_lp = torch.stack(old_lp_list)                    # [B, Nm]

        n_epochs       = self.ppo_epochs if self.algo == "ppo" else 1
        actor_loss_val = 0.0

        for _ in range(n_epochs):
            lp_list = []
            ent_list = []
            for i in range(batch_size):
                lp_i, ent_i = self.policy.log_prob_entropy_pertask(
                    membs_list[i], bw_samples[i])
                lp_list.append(lp_i)
                ent_list.append(ent_i)
            lp  = torch.stack(lp_list)    # [B, Nm]
            ent = torch.stack(ent_list)   # [B] scalar per episode (same sigma)

            if self.algo == "ac":
                actor_loss = -(lp * adv).mean()
            elif self.algo == "ppo":
                ratio      = torch.exp(lp - old_lp)
                unclipped  = ratio * adv
                clipped    = torch.clamp(
                    ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv
                actor_loss = -torch.min(unclipped, clipped).mean()
            else:  # sac
                actor_loss = -(lp * adv).mean() - self.sac_alpha * ent.mean()

            self.opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            actor_loss_val = float(actor_loss.item())

        # ---- Critic update ----
        r_vec       = torch.tensor(mean_rewards, dtype=torch.float,
                                   device=self.device)
        vp          = self.critic(ge).squeeze(-1)
        critic_loss = self.mse(vp, r_vec)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return (float(np.mean(mean_rewards)), float(critic_loss.item()),
                actor_loss_val, float(np.mean(ep_isrs)), tasks_per_csca)

    @torch.no_grad()
    def evaluate_isr(self, tpc_list=(1, 2, 4, 6, 10), n_episodes=50):
        self.policy.eval()
        results = {}
        for tpc in tpc_list:
            isrs = []
            for _ in range(n_episodes):
                env   = self._make_env(tpc)
                state = env.generate_state()
                intents = [[m[1], m[2]] for m in state["SCt"]["message_features"]]
                _, _, message_embs = self.han.encode_state(state, intents)
                out = env.step(self.policy.act(message_embs), state)
                isrs.append(compute_isr(out["tasks"]))
            results[tpc] = (float(np.mean(isrs)), float(np.std(isrs)))
        self.policy.train()
        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "han":    self.han.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "algo":   self.algo,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.han.load_state_dict(ckpt["han"])
        self.policy.load_state_dict(ckpt["policy"])
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
