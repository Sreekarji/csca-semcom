"""
DASC Baseline (Simplified) for CSCA comparison.
Implements a simplified version of Diffusion-based Audio Semantic Communication.

Paper reference: Grassucci et al. 2024 (Fig 6 comparison)
Simplified: mel spectrogram encoder -> AWGN channel -> decoder
Metric: cosine similarity of mel spectrograms
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MELAutoencoder(nn.Module):
    """
    Simplified audio semantic encoder/decoder.
    Encoder: mel spectrogram features -> compressed representation
    Channel: AWGN noise
    Decoder: compressed representation -> reconstructed mel features
    """

    def __init__(self, input_dim: int = 80, hidden_dim: int = 64, compressed_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim

        # Encoder: mel features -> compressed
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, compressed_dim),
        )

        # Decoder: compressed -> reconstructed mel
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, noise_std=0.1):
        z = self.encode(x)
        # Power normalize
        power = torch.mean(z * z).sqrt()
        if power > 1:
            z = z / power
        # Channel noise
        z_noisy = z + torch.normal(0, noise_std, size=z.shape).to(z.device)
        x_recon = self.decode(z_noisy)
        return x_recon, z, z_noisy


class DASCBaseline:
    """
    DASC baseline for audio semantic communication comparison.
    
    Key metrics from paper Fig 6:
    - Semantic accuracy: cosine similarity of mel features
    - Compression ratio: compressed_dim / input_dim
    """

    def __init__(self, input_dim: int = 80, compressed_dim: int = 16, snr_db: float = 10.0):
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.snr_db = snr_db
        self.noise_std = self._snr_to_noise(snr_db)

        self.model = MELAutoencoder(
            input_dim=input_dim,
            hidden_dim=64,
            compressed_dim=compressed_dim,
        ).to(DEVICE)

        self.compression_ratio = compressed_dim / input_dim

        print(f"[DASC] Initialized: {input_dim} -> {compressed_dim} features")
        print(f"[DASC] Compression ratio: {self.compression_ratio:.1%}")

    def _snr_to_noise(self, snr_db):
        snr = 10 ** (snr_db / 10)
        return 1 / np.sqrt(2 * snr)

    def _generate_mel_features(self, n_frames: int = 50) -> torch.Tensor:
        """Generate synthetic mel spectrogram features for testing."""
        # Simulate mel spectrogram: random features in typical mel range
        return torch.randn(1, n_frames, self.input_dim, device=DEVICE) * 0.5

    def compute_semantic_accuracy(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Cosine similarity of mel feature vectors."""
        orig_flat = original.reshape(-1).detach().cpu()
        recon_flat = reconstructed.reshape(-1).detach().cpu()
        min_len = min(len(orig_flat), len(recon_flat))
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat[:min_len].unsqueeze(0),
            recon_flat[:min_len].unsqueeze(0)
        ).item()
        return max(0, cos_sim)

    def evaluate(self, n_samples: int = 50, snr_range: list = None) -> dict:
        """Evaluate DASC across SNR range."""
        if snr_range is None:
            snr_range = [0, 5, 10, 15, 20, 25]

        results = {
            "compression_ratio": self.compression_ratio,
            "accuracy_by_snr": {},
        }

        for snr in snr_range:
            noise_std = self._snr_to_noise(snr)
            accuracies = []

            for _ in range(n_samples):
                mel = self._generate_mel_features()
                with torch.no_grad():
                    recon, z, z_noisy = self.model(mel, noise_std=noise_std)
                acc = self.compute_semantic_accuracy(mel, recon)
                accuracies.append(acc)

            results["accuracy_by_snr"][snr] = np.mean(accuracies)

        return results


def create_dasc_baseline():
    """Factory function to create DASC baseline."""
    return DASCBaseline(input_dim=80, compressed_dim=16, snr_db=10.0)


if __name__ == "__main__":
    print("=" * 60)
    print("DASC BASELINE TEST (Simplified Audio SemCom)")
    print("=" * 60)

    dasc = create_dasc_baseline()

    print(f"\nCompression ratio: {dasc.compression_ratio:.1%}")
    print(f"Architecture: mel({dasc.input_dim}) -> compressed({dasc.compressed_dim}) -> mel({dasc.input_dim})")

    print("\nSNR sweep:")
    results = dasc.evaluate(n_samples=50)
    for snr, acc in sorted(results["accuracy_by_snr"].items()):
        print(f"  SNR={snr}dB: accuracy={acc:.4f}")

    print("\nDASC baseline test complete.")
