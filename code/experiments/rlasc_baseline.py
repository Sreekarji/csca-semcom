"""
RL-ASC Baseline (Simplified) for CSCA comparison.
Implements a simplified version of RL-based image Semantic Coding.

Paper reference: Huang et al. 2023 (Fig 6 comparison)
Simplified: CNN autoencoder for images -> AWGN channel -> reconstruction
Metric: PSNR (as paper uses for images, Eq. 37)
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageAutoencoder(nn.Module):
    """
    Simplified image semantic encoder/decoder.
    Encoder: CNN -> latent representation
    Channel: AWGN noise
    Decoder: transposed CNN -> reconstructed image
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 32x32 -> 4x4xlatent
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

        # Decoder: latent -> 32x32
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),  # 32x32
            nn.Sigmoid(),
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


class RLASCBaseline:
    """
    RL-ASC baseline for image semantic communication comparison.
    
    Key metrics from paper Fig 6:
    - Semantic accuracy: PSNR > 22dB considered accurate (Eq. 37)
    - Compression ratio: latent_dim / (3 * 32 * 32)
    """

    def __init__(self, image_size: int = 32, latent_dim: int = 64, snr_db: float = 10.0):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.snr_db = snr_db
        self.noise_std = self._snr_to_noise(snr_db)

        self.model = ImageAutoencoder(
            in_channels=3,
            latent_dim=latent_dim,
        ).to(DEVICE)

        # Compression: latent / original pixels
        original_dim = 3 * image_size * image_size
        self.compression_ratio = latent_dim / original_dim

        print(f"[RL-ASC] Initialized: {image_size}x{image_size}x3 -> {latent_dim}D latent")
        print(f"[RL-ASC] Compression ratio: {self.compression_ratio:.1%}")

    def _snr_to_noise(self, snr_db):
        snr = 10 ** (snr_db / 10)
        return 1 / np.sqrt(2 * snr)

    def load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        return tensor

    def compute_psnr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """
        Compute PSNR (Peak Signal-to-Noise Ratio) — Eq. 37 from paper.
        Paper uses PSNR > 22dB as threshold for image accuracy.
        """
        mse = torch.mean((original - reconstructed) ** 2).item()
        if mse < 1e-10:
            return 100.0  # Perfect reconstruction
        psnr = 10 * np.log10(1.0 / mse)  # Assuming [0,1] range
        return psnr

    def evaluate(self, image_dir: str = None, n_samples: int = 50, snr_range: list = None) -> dict:
        """Evaluate RL-ASC across SNR range."""
        if snr_range is None:
            snr_range = [0, 5, 10, 15, 20, 25]

        # Load images or generate synthetic
        images = []
        if image_dir and os.path.exists(image_dir):
            files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))][:n_samples]
            for f in files:
                try:
                    img = self.load_image(os.path.join(image_dir, f))
                    images.append(img)
                except Exception:
                    pass

        if not images:
            # Generate synthetic images
            for _ in range(n_samples):
                images.append(torch.rand(1, 3, self.image_size, self.image_size, device=DEVICE))

        results = {
            "compression_ratio": self.compression_ratio,
            "psnr_by_snr": {},
            "accuracy_by_snr": {},  # PSNR > 22dB threshold
        }

        for snr in snr_range:
            noise_std = self._snr_to_noise(snr)
            psnr_values = []

            for img in images:
                with torch.no_grad():
                    recon, z, z_noisy = self.model(img, noise_std=noise_std)
                psnr = self.compute_psnr(img, recon)
                psnr_values.append(psnr)

            avg_psnr = np.mean(psnr_values)
            results["psnr_by_snr"][snr] = avg_psnr
            results["accuracy_by_snr"][snr] = np.mean([1 if p > 22 else 0 for p in psnr_values])

        return results


def create_rlasc_baseline():
    """Factory function to create RL-ASC baseline."""
    return RLASCBaseline(image_size=32, latent_dim=64, snr_db=10.0)


if __name__ == "__main__":
    print("=" * 60)
    print("RL-ASC BASELINE TEST (Simplified Image SemCom)")
    print("=" * 60)

    rlasc = create_rlasc_baseline()

    image_dir = r"D:\MP2\data\raw\images"
    print(f"\nCompression ratio: {rlasc.compression_ratio:.1%}")
    print(f"Architecture: image(32x32x3) -> latent({rlasc.latent_dim}) -> image(32x32x3)")

    print("\nSNR sweep:")
    results = rlasc.evaluate(image_dir=image_dir, n_samples=20)
    for snr in sorted(results["psnr_by_snr"].keys()):
        psnr = results["psnr_by_snr"][snr]
        acc = results["accuracy_by_snr"][snr]
        print(f"  SNR={snr}dB: PSNR={psnr:.2f}dB, accuracy(>22dB)={acc:.1%}")

    print("\nRL-ASC baseline test complete.")
