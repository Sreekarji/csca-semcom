"""
FIX 18: DDPM diffusion policy for the FIX-14c action space (BW + MCS, no relay).
Per-task denoising: each task's 4-dim action chunk (1 BW logit + 3 MCS logits)
is denoised conditioned on (its own message embedding, global graph embedding,
timestep). This preserves the per-task information pathway that makes HDM beat
the baselines, and is task-count agnostic.
Reverse diffusion uses the reparameterization trick end-to-end, so gradients
flow: Q -> action -> denoiser -> message_embs -> HAN, exactly like the MLP path.
Noise schedule: paper Eq. 31-32. Reverse mean: Eq. 30.
"""
import numpy as np
import torch
import torch.nn as nn


class PerTaskDenoiser(nn.Module):
    """eps_theta(a_n^i, n | msg_emb_i, graph_emb) applied to every task i."""
    def __init__(self, task_action_dim=4, graph_emb_dim=256,
                 task_emb_dim=256, hidden_dim=256, n_denoising_steps=6):
        super().__init__()
        self.time_emb = nn.Embedding(n_denoising_steps + 1, hidden_dim)
        cond_dim = graph_emb_dim + task_emb_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(task_action_dim + cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, task_action_dim),
        )
        # Near-zero init so denoiser predicts near-zero noise at init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, a_n, n, graph_emb, message_embs):
        # a_n: [Nt, task_action_dim]; graph_emb: [1, 256]; message_embs: [Nt, 256]
        Nt = a_n.shape[0]
        t = self.time_emb(torch.tensor(n, dtype=torch.long, device=a_n.device))
        t = t.unsqueeze(0).expand(Nt, -1)
        g = graph_emb.expand(Nt, -1)
        return self.net(torch.cat([a_n, g, message_embs, t], dim=-1))


class DDPMActor(nn.Module):
    """Drop-in replacement for MLPActor. forward(graph_emb, message_embs) -> [1, action_dim]."""
    def __init__(self, graph_emb_dim=256, task_emb_dim=256, action_dim=80,
                 hidden_dim=256, n_tasks=20, n_mcs=3, n_denoising_steps=6,
                 beta_min=0.01, beta_max=0.5):
        super().__init__()
        assert action_dim == n_tasks + n_tasks * n_mcs, \
            f"action_dim {action_dim} != n_tasks({n_tasks}) + n_tasks*n_mcs({n_tasks*n_mcs})"
        self.n_tasks = n_tasks
        self.n_mcs = n_mcs
        self.action_dim = action_dim
        self.task_dim = 1 + n_mcs          # per-task: 1 BW logit + n_mcs MCS logits
        self.N = n_denoising_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.denoiser = PerTaskDenoiser(self.task_dim, graph_emb_dim,
                                        task_emb_dim, hidden_dim, n_denoising_steps)
        self.bw_temperature = nn.Parameter(torch.tensor(1.0))
        self._build_schedule()

    def _build_schedule(self):
        # Paper Eq. 31-32
        betas = []
        for n in range(1, self.N + 1):
            beta_n = 1.0 - np.exp(
                -(self.beta_min / self.N)
                - (2 * n - 1) / (2 * self.N ** 2) * (self.beta_max - self.beta_min))
            betas.append(beta_n)
        betas = torch.tensor(betas, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        beta_tilde = torch.zeros_like(betas)
        beta_tilde[0] = betas[0]
        for n in range(1, self.N):
            beta_tilde[n] = ((1.0 - alphas_cumprod[n - 1])
                             / (1.0 - alphas_cumprod[n]) * betas[n])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("beta_tilde", beta_tilde)

    def reverse_diffusion(self, graph_emb, message_embs, deterministic=False):
        device = graph_emb.device
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        a_n = torch.randn(self.n_tasks, self.task_dim, device=device)
        for n in range(self.N, 0, -1):
            beta_n = self.betas[n - 1]
            alpha_n = self.alphas[n - 1]
            abar_n = self.alphas_cumprod[n - 1]
            eps = self.denoiser(a_n, n, graph_emb, message_embs)
            mean = (1.0 / torch.sqrt(alpha_n)) * (
                a_n - (beta_n / torch.sqrt(1.0 - abar_n)) * eps)   # Eq. 30
            if n > 1 and not deterministic:
                a_n = mean + torch.sqrt(self.beta_tilde[n - 1]) * torch.randn_like(a_n)
            else:
                a_n = mean
        return a_n                                                 # [Nt, task_dim]

    def forward(self, graph_emb, message_embs=None, deterministic=False):
        if message_embs is None:
            raise ValueError("DDPMActor requires per-task message_embs from HAN")
        raw = self.reverse_diffusion(graph_emb, message_embs, deterministic)
        bw = torch.softmax(raw[:, 0] / self.bw_temperature.abs().clamp_min(0.1),
                           dim=0).unsqueeze(0)                      # [1, Nt]
        mcs = torch.sigmoid(raw[:, 1:]).reshape(1, -1)              # [1, Nt*n_mcs]
        return torch.cat([bw, mcs], dim=-1)                         # [1, action_dim]


if __name__ == "__main__":
    # Smoke test
    actor = DDPMActor(n_tasks=20, n_mcs=3)
    ge = torch.randn(1, 256)
    me = torch.randn(20, 256)
    out = actor(ge, message_embs=me)
    assert out.shape == (1, 80), f"Shape mismatch: {out.shape}"
    assert abs(out[0, :20].sum().item() - 1.0) < 0.01, f"BW doesn't sum to 1: {out[0,:20].sum()}"
    # Check gradient flow
    loss = -out.sum()
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in actor.denoiser.parameters() if p.grad is not None)
    assert gnorm > 0, f"No gradient! gnorm={gnorm}"
    print(f"DDPMActor smoke test PASSED: shape={out.shape}, BW sum={out[0,:20].sum():.4f}, grad_norm={gnorm:.4f}")
