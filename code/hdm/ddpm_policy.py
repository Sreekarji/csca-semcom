import torch
import torch.nn as nn
import numpy as np


class MLPDenoiser(nn.Module):
    """
    MLP-based denoiser for DDPM policy generation.
    Sun et al. 2026 explicitly uses MLP (not UNet) for faster inference.

    Split architecture:
    - task_bw_head: per-task BW conditioned on message_embs (task-specific)
    - global_head: relay + MCS conditioned on graph_emb (global)
    """

    def __init__(
        self,
        action_dim: int = 45,
        graph_emb_dim: int = 256,
        task_emb_dim: int = 256,
        hidden_dim: int = 256,
        n_denoising_steps: int = 6,
        n_tasks: int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps = n_denoising_steps
        self.n_tasks = n_tasks

        self.time_emb = nn.Embedding(n_denoising_steps + 1, hidden_dim)

        # Task-specific bandwidth head
        # Input: per-task embedding + timestep
        self.task_bw_head = nn.Sequential(
            nn.Linear(task_emb_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # One BW value per task
        )

        # Global policy head for relay + MCS
        relay_mcs_dim = action_dim - n_tasks
        self.global_head = nn.Sequential(
            nn.Linear(action_dim + graph_emb_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, relay_mcs_dim),
        )

    def forward(self, a_n: torch.Tensor, n: int,
                graph_emb: torch.Tensor,
                message_embs: torch.Tensor = None):
        t_emb = self.time_emb(
            torch.tensor(n, device=a_n.device)
        ).unsqueeze(0)  # [1, hidden_dim]

        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        # Task-specific bandwidth allocation
        if message_embs is not None:
            # message_embs: [n_tasks, 128]
            t_emb_expanded = t_emb.expand(message_embs.shape[0], -1)
            task_input = torch.cat([message_embs, t_emb_expanded], dim=-1)
            bw_noise = self.task_bw_head(task_input).T  # [1, n_tasks]
        else:
            bw_noise = a_n[:, :self.n_tasks]

        # Global relay + MCS allocation
        if t_emb.shape[0] != a_n.shape[0]:
            t_emb = t_emb.expand(a_n.shape[0], -1)
        if graph_emb.shape[0] != a_n.shape[0]:
            if graph_emb.shape[0] == 1:
                graph_emb = graph_emb.expand(a_n.shape[0], -1)
            else:
                graph_emb = graph_emb[:a_n.shape[0]]

        global_input = torch.cat([a_n, graph_emb, t_emb], dim=-1)
        relay_mcs_noise = self.global_head(global_input)

        # Combine
        predicted_noise = torch.cat([bw_noise, relay_mcs_noise], dim=-1)
        return predicted_noise


class HDMPolicy(nn.Module):
    """
    HDM: HAN + DDPM policy network.
    Implements Algorithm 2 from Sun et al. 2026.
    Action space: at = {BWt, Πt, Θt}
    """

    def __init__(
        self,
        action_dim: int = 45,
        graph_emb_dim: int = 256,
        n_denoising_steps: int = 6,
        beta_min: float = 0.01,
        beta_max: float = 0.5,
        n_tasks: int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.N = n_denoising_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.denoiser = MLPDenoiser(
            action_dim=action_dim,
            graph_emb_dim=graph_emb_dim,
            n_denoising_steps=n_denoising_steps,
            n_tasks=n_tasks,
        )

        # Precompute noise schedule (Eq. 31-32)
        self._build_schedule()

    def _build_schedule(self):
        betas = []
        for n in range(1, self.N + 1):
            beta_n = 1.0 - np.exp(
                -(self.beta_min / self.N)
                - (2 * n - 1) / (2 * self.N ** 2) * (self.beta_max - self.beta_min)
            )
            betas.append(beta_n)

        betas = torch.tensor(betas, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def collect_trajectory(self, graph_emb: torch.Tensor,
                           message_embs: torch.Tensor = None,
                           batch_size: int = 1):
        """
        Run full denoising trajectory and compute ELBO-based log probability.
        
        DDPO uses the ELBO as a surrogate for log p_θ(a_0):
          ELBO = Σ_{t=1}^{T} KL(q(a_{t-1}|a_t,a_0) || p_θ(a_{t-1}|a_t))
        
        For Gaussians:
          KL = 0.5 * ||μ_q(t) - μ_θ(t)||² / β_t
        
        Gradient flows through μ_θ(t) → denoiser → graph_emb → HAN.
        """
        device = graph_emb.device
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if graph_emb.shape[0] == 1 and batch_size > 1:
            graph_emb = graph_emb.expand(batch_size, -1)

        # Step 1: Generate action via reverse diffusion
        a_n = torch.randn(batch_size, self.action_dim, device=device)

        for n in range(self.N, 0, -1):
            beta_n = self.betas[n - 1]
            alpha_n = self.alphas[n - 1]
            alpha_bar_n = self.alphas_cumprod[n - 1]

            predicted_noise = self.denoiser(
                a_n, n, graph_emb, message_embs=message_embs
            )

            coeff = beta_n / torch.sqrt(1.0 - alpha_bar_n)
            mu_theta = (1.0 / torch.sqrt(alpha_n)) * (a_n - coeff * predicted_noise)

            if n > 1:
                noise = torch.randn_like(a_n)
                a_n = mu_theta + torch.sqrt(beta_n) * noise
            else:
                noise = torch.randn_like(a_n)
                a_n = mu_theta + torch.sqrt(beta_n) * noise

        a_0 = torch.sigmoid(a_n)

        # Step 2: Compute ELBO by re-running denoiser on forward-noised versions of a_0
        elbo = torch.zeros(batch_size, 1, device=device)

        for t in range(1, self.N + 1):
            beta_t = self.betas[t - 1]
            alpha_t = self.alphas[t - 1]
            alpha_bar_t = self.alphas_cumprod[t - 1]
            alpha_bar_prev = self.alphas_cumprod[t - 2] if t > 1 else torch.tensor(1.0, device=device)

            # Forward process: generate a_t from a_0
            eps_t = torch.randn_like(a_0)
            a_t = torch.sqrt(alpha_bar_t) * a_0.detach() + torch.sqrt(1.0 - alpha_bar_t) * eps_t

            # Reverse process mean: μ_θ(t) from denoiser (gradient flows here)
            predicted_noise_t = self.denoiser(
                a_t, t, graph_emb, message_embs=message_embs
            )
            coeff_t = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mu_theta_t = (1.0 / torch.sqrt(alpha_t)) * (a_t - coeff_t * predicted_noise_t)

            # Posterior mean (detached — no gradient through forward process)
            mu_q = (alpha_t * a_0.detach() + torch.sqrt(alpha_bar_prev) * (1.0 - alpha_t) * a_t.detach()) / (1.0 - alpha_bar_t)

            # KL divergence: 0.5 * ||μ_q - μ_θ||² / β_t
            kl_t = 0.5 * torch.mean((mu_q - mu_theta_t) ** 2, dim=-1, keepdim=True) / beta_t
            elbo = elbo + kl_t

        # Normalize by number of steps
        elbo = elbo / self.N

        # Return -elbo as log_prob (higher is better)
        total_log_prob = -elbo

        return a_0, total_log_prob

    def reverse_diffusion(self, graph_emb: torch.Tensor,
                          message_embs: torch.Tensor = None,
                          batch_size: int = 1):
        """
        Reverse denoising process: a_N ~ N(0,I) -> a_0
        Eq. 28-30 from paper.
        """
        device = graph_emb.device
        a_n = torch.randn(batch_size, self.action_dim, device=device)

        for n in range(self.N, 0, -1):
            beta_n = self.betas[n - 1]
            alpha_n = self.alphas[n - 1]
            alpha_cumprod_n = self.alphas_cumprod[n - 1]

            predicted_noise = self.denoiser(
                a_n, n, graph_emb, message_embs=message_embs
            )

            # Eq. 30: compute mean
            coeff = beta_n / torch.sqrt(1.0 - alpha_cumprod_n)
            mean = (1.0 / torch.sqrt(alpha_n)) * (a_n - coeff * predicted_noise)

            if n > 1:
                # Eq. 29: posterior variance beta_tilde_n
                alpha_cumprod_prev = self.alphas_cumprod[n - 2]
                beta_tilde_n = (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_n) * beta_n
                noise = torch.randn_like(a_n)
                a_n = mean + torch.sqrt(beta_tilde_n) * noise
            else:
                a_n = mean

        # Normalize action to [0, 1]
        a_0 = torch.sigmoid(a_n)
        return a_0

    def forward(self, graph_emb: torch.Tensor,
                message_embs: torch.Tensor = None):
        return self.reverse_diffusion(graph_emb, message_embs=message_embs)

    def parse_action(self, a_0: torch.Tensor, n_tasks: int = 5, n_relays: int = 5, n_mcs: int = 3):
        bw = a_0[:, :n_tasks]
        relay = a_0[:, n_tasks:n_tasks + n_tasks * n_relays].reshape(-1, n_tasks, n_relays)
        mcs = a_0[:, n_tasks + n_tasks * n_relays:].reshape(-1, n_tasks, n_mcs)
        return {"bandwidth": bw, "relay": relay, "mcs": mcs}


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int = 256, action_dim: int = 45, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor, action: torch.Tensor):
        if graph_emb.shape[0] != action.shape[0]:
            if graph_emb.shape[0] == 1:
                graph_emb = graph_emb.expand(action.shape[0], -1)
            elif action.shape[0] == 1:
                action = action.expand(graph_emb.shape[0], -1)
            else:
                raise ValueError(
                    f"CriticNetwork batch size mismatch: "
                    f"graph_emb={graph_emb.shape}, action={action.shape}"
                )
        inp = torch.cat([graph_emb, action], dim=-1)
        return self.net(inp)
