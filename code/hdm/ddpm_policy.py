import torch
import torch.nn as nn
import numpy as np

class MLPDenoiser(nn.Module):
    """
    MLP-based denoiser for DDPM policy generation.
    Sun et al. 2026 explicitly uses MLP (not UNet) for faster inference.
    Input: noisy action a_n, timestep n, graph embedding GL_t
    Output: predicted noise theta
    """

    def __init__(
        self,
        action_dim: int = 45,
        graph_emb_dim: int = 128,
        hidden_dim: int = 256,
        n_denoising_steps: int = 6,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps = n_denoising_steps

        # Timestep embedding
        self.time_emb = nn.Embedding(n_denoising_steps + 1, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(action_dim + graph_emb_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, a_n: torch.Tensor, n: int, graph_emb: torch.Tensor):
        t_emb = self.time_emb(torch.tensor(n, device=a_n.device))
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0).expand(a_n.shape[0], -1)
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0).expand(a_n.shape[0], -1)
        inp = torch.cat([a_n, graph_emb, t_emb], dim=-1)
        return self.net(inp)


class HDMPolicy(nn.Module):
    """
    HDM: HAN + DDPM policy network.
    Implements Algorithm 2 from Sun et al. 2026.
    Action space: at = {BWt, Πt, Θt}
      BWt: bandwidth allocation (Nm values)
      Πt: relay selection (Nm x Nr values)
      Θt: MCS selection (Nm x NMCS values)
    With Nm=5, Nr=5, NMCS=3: action_dim = 5 + 25 + 15 = 45
    """

    def __init__(
        self,
        action_dim: int = 45,
        graph_emb_dim: int = 128,
        n_denoising_steps: int = 6,
        beta_min: float = 0.01,
        beta_max: float = 0.5,
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

    def reverse_diffusion(self, graph_emb: torch.Tensor, batch_size: int = None):
        """
        Reverse denoising process: a_N ~ N(0,I) -> a_0
        Eq. 28-30 from paper.
        """
        device = graph_emb.device
        # Auto-detect batch size from graph_emb
        if batch_size is None:
            batch_size = graph_emb.shape[0] if graph_emb.dim() > 1 else 1
        a_n = torch.randn(batch_size, self.action_dim, device=device)

        for n in range(self.N, 0, -1):
            beta_n = self.betas[n - 1]
            alpha_n = self.alphas[n - 1]
            alpha_cumprod_n = self.alphas_cumprod[n - 1]

            predicted_noise = self.denoiser(a_n, n, graph_emb)

            # Eq. 30: compute mean
            mean = (1.0 / torch.sqrt(alpha_n)) * (
                a_n - (beta_n / torch.sqrt(1.0 - alpha_cumprod_n)) * predicted_noise
            )

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

    def forward(self, graph_emb: torch.Tensor):
        return self.reverse_diffusion(graph_emb)

    def parse_action(self, a_0: torch.Tensor, n_tasks: int = 5, n_relays: int = 5, n_mcs: int = 3):
        bw = a_0[:, :n_tasks]
        relay = a_0[:, n_tasks:n_tasks + n_tasks * n_relays].reshape(-1, n_tasks, n_relays)
        mcs = a_0[:, n_tasks + n_tasks * n_relays:].reshape(-1, n_tasks, n_mcs)
        return {"bandwidth": bw, "relay": relay, "mcs": mcs}


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int = 128, action_dim: int = 45, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor, action: torch.Tensor):
        inp = torch.cat([graph_emb, action], dim=-1)
        return self.net(inp)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = HDMPolicy(action_dim=45, graph_emb_dim=128, n_denoising_steps=6).to(device)
    critic = CriticNetwork(state_dim=128, action_dim=45).to(device)

    graph_emb = torch.randn(1, 128, device=device)
    action = policy(graph_emb)
    value = critic(graph_emb, action)

    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"Value: {value.item():.4f}")
    parsed = policy.parse_action(action)
    print(f"Bandwidth: {parsed['bandwidth'].shape}")
    print(f"Relay: {parsed['relay'].shape}")
    print(f"MCS: {parsed['mcs'].shape}")
    print("DDPM policy test passed.")