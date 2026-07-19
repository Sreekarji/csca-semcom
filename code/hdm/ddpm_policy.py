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
            nn.Linear(graph_emb_dim + hidden_dim, hidden_dim),
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
            # Average message embeddings across tasks sharing same CSCA
            # message_embs: [n_total_tasks, 256] — reduce to [n_tasks, 256]
            n_total = message_embs.shape[0]
            if n_total != self.n_tasks:
                # Group tasks by CSCA and average embeddings
                tasks_per_csca = n_total // self.n_tasks
                if tasks_per_csca > 0 and n_total % self.n_tasks == 0:
                    msg_grouped = message_embs.view(self.n_tasks, tasks_per_csca, -1).mean(dim=1)
                else:
                    # Fallback: interpolate to n_tasks
                    msg_grouped = message_embs[:self.n_tasks] if n_total >= self.n_tasks else \
                                  message_embs.mean(dim=0, keepdim=True).expand(self.n_tasks, -1)
            else:
                msg_grouped = message_embs
            t_emb_expanded = t_emb.expand(msg_grouped.shape[0], -1)
            task_input = torch.cat([msg_grouped, t_emb_expanded], dim=-1)
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

        global_input = torch.cat([graph_emb, t_emb], dim=-1)
        relay_mcs_noise = self.global_head(global_input)
        if relay_mcs_noise.shape[0] != a_n.shape[0]:
            relay_mcs_noise = relay_mcs_noise.expand(a_n.shape[0], -1)

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
        self.n_tasks = n_tasks

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
        Paper-faithful log_prob: Eq. 28-29.
        Record μ_θ(a_n, n, G^L_t) at each actual denoising step.
        log π_θ(a_t|s_t) = Σ_{n=1}^{N} log N(a_{n-1}; μ_θ, β̃_n I)
        """
        device = graph_emb.device
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if graph_emb.shape[0] == 1 and batch_size > 1:
            graph_emb = graph_emb.expand(batch_size, -1)

        # Run full trajectory WITH gradients, recording μ_θ at each step
        a_n = torch.randn(batch_size, self.action_dim, device=device)
        
        mu_list = []
        a_list = [a_n]  # a_N is the starting point

        for n in range(self.N, 0, -1):
            beta_n = self.betas[n - 1]
            alpha_n = self.alphas[n - 1]
            alpha_cumprod_n = self.alphas_cumprod[n - 1]

            # Predict noise WITH gradients (so actor loss flows through denoiser)
            pred_noise = self.denoiser(a_n, n, graph_emb, message_embs=message_embs)

            # Eq. 30: μ_θ(a_n, n, G^L_t)
            coeff = beta_n / torch.sqrt(1.0 - alpha_cumprod_n)
            mu = (1.0 / torch.sqrt(alpha_n)) * (a_n - coeff * pred_noise)
            mu_list.append(mu)

            if n > 1:
                # Eq. 29: posterior variance β̃_n
                alpha_cumprod_prev = self.alphas_cumprod[n - 2]
                beta_tilde_n = (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_n) * beta_n
                noise = torch.randn_like(a_n)
                a_n = mu.detach() + torch.sqrt(beta_tilde_n) * noise
            else:
                a_n = mu.detach()
            a_list.append(a_n)

        a_0 = torch.sigmoid(a_n)

        # Eq. 28-29: log π_θ = Σ_{n=1}^{N} log N(a_{n-1}; μ_θ(a_n,n,G), β̃_n I)
        log_prob = torch.zeros(batch_size, 1, device=device)
        for i, (mu_i, beta_i_idx) in enumerate(zip(mu_list, range(self.N - 1, -1, -1))):
            beta_n = self.betas[beta_i_idx]
            alpha_cumprod_n = self.alphas_cumprod[beta_i_idx]
            if beta_i_idx > 0:
                alpha_cumprod_prev = self.alphas_cumprod[beta_i_idx - 1]
                beta_tilde_n = (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_n) * beta_n
            else:
                beta_tilde_n = beta_n
            # a_{n-1} is a_list[i+1]
            a_prev = a_list[i + 1].detach()
            # log N(a_{n-1}; μ_i, β̃_n I) = -0.5 * ||a_{n-1} - μ_i||² / β̃_n
            diff = a_prev - mu_i
            log_prob_step = -0.5 * (diff ** 2 / (beta_tilde_n + 1e-8)).sum(dim=-1, keepdim=True)
            log_prob = log_prob + log_prob_step

        # Normalize by both N (steps) and action_dim to keep actor_loss in [-1, 1] range
        # Without this: log_prob ≈ -19.5, actor_loss ≈ ±58, training oscillates
        # With this: log_prob ≈ -0.07, actor_loss ≈ ±0.2, training is stable
        log_prob = log_prob / (self.N * self.action_dim)
        return a_0.detach(), log_prob

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
