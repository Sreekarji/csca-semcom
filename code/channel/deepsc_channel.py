"""
DeepSC semantic communication channel.
Wraps the trained DeepSC encoder-channel-decoder for inference.

Usage:
    ch = DeepSCChannel()
    result = ch.transmit("the quick brown fox", snr_db=10)
    # result = {"decoded_text": "...", "similarity": 0.85, "compression_ratio": 0.125}
"""

import os
import sys
import json
import torch
import numpy as np

DEEPSC_REPO = r"D:\MP2\repos\DeepSC"
sys.path.insert(0, DEEPSC_REPO)

# Import DeepSC model
from models.transceiver import DeepSC
from utils import SNR_to_noise


class PowerNormalize(torch.nn.Module):
    """Normalize transmit power to 1."""
    def forward(self, x):
        power = torch.mean(x ** 2)
        return x / torch.sqrt(power)


class Channels:
    """AWGN / Rayleigh / Rician channel models."""
    @staticmethod
    def AWGN(x, n_var):
        noise = torch.randn_like(x) * n_var
        return x + noise

    @staticmethod
    def Rayleigh(x, n_var):
        h = torch.randn(x.shape[0], 1, x.shape[2], device=x.device) / np.sqrt(2)
        h = torch.abs(h)
        noise = torch.randn_like(x) * n_var
        return h * x + noise

    @staticmethod
    def Rician(x, n_var, K=3):
        h_i = torch.randn(x.shape[0], 1, x.shape[2], device=x.device)
        h_r = torch.randn(x.shape[0], 1, x.shape[2], device=x.device) + np.sqrt(K)
        h = torch.sqrt(h_r ** 2 + h_i ** 2) / np.sqrt(K + 1)
        noise = torch.randn_like(x) * n_var
        return h * x + noise


class DeepSCChannel:
    """
    Wraps trained DeepSC encoder-channel-decoder.
    Input: raw text string
    Output: decoded text + quality metrics
    """

    def __init__(
        self,
        checkpoint_path: str = r"D:\MP2\models\deepsc\text\best_model.pth",
        vocab_path: str = r"D:\MP2\repos\DeepSC\europarl\vocab.json",
        device: str = None,
        channel_type: str = "AWGN",
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.channel_type = channel_type
        self.checkpoint_path = checkpoint_path
        self.power_norm = PowerNormalize()
        self.channels = Channels()

        # Load vocab (structure: {"token_to_idx": {"<PAD>": 0, ...}})
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        if isinstance(raw_vocab, dict) and "token_to_idx" in raw_vocab:
            self.vocab = raw_vocab["token_to_idx"]
        else:
            self.vocab = raw_vocab
        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.pad_idx = self.vocab.get("<PAD>", 0)
        self.start_idx = self.vocab.get("<START>", 1)
        self.end_idx = self.vocab.get("<END>", 2)
        self.vocab_size = len(self.vocab)

        # Model hyperparameters (from main.py defaults)
        self.d_model = 128
        self.num_heads = 8
        self.dff = 512
        self.num_layers = 4
        self.max_len = self.vocab_size  # checkpoint was trained with max_len=vocab_size

        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"DeepSC checkpoint not found: {self.checkpoint_path}")

        self.model = DeepSC(
            num_layers=self.num_layers,
            src_vocab_size=self.vocab_size,
            trg_vocab_size=self.vocab_size,
            src_max_len=self.max_len,
            trg_max_len=self.max_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
        ).to(self.device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"[DeepSC] Loaded: {self.checkpoint_path}")

    def tokenize(self, text: str) -> list:
        """Convert text to token indices."""
        words = text.lower().strip().split()
        tokens = [self.start_idx]
        for w in words:
            tokens.append(self.vocab.get(w, self.vocab.get("<UNK>", 3)))
        tokens.append(self.end_idx)
        return tokens

    def detokenize(self, tokens: list) -> str:
        """Convert token indices back to text."""
        words = []
        for t in tokens:
            if t == self.end_idx:
                break
            if t in (self.pad_idx, self.start_idx, 0):
                continue
            words.append(self.idx2word.get(t, "<UNK>"))
        return " ".join(words)

    def transmit(self, text: str, snr_db: float = 10.0) -> dict:
        """
        Transmit text through DeepSC: tokenize → encode → channel → decode → detokenize.
        """
        tokens = self.tokenize(text)
        src = torch.tensor([tokens], dtype=torch.long, device=self.device)
        trg_inp = src[:, :-1]  # shifted input for decoder

        # Masks matching DeepSC's create_masks() format
        src_mask = (src == self.pad_idx).unsqueeze(-2).float().to(self.device)  # [1, 1, seq_len]
        trg_mask = (trg_inp == self.pad_idx).unsqueeze(-2).float().to(self.device)
        look_ahead = torch.from_numpy(
            np.triu(np.ones((1, trg_inp.size(1), trg_inp.size(1))), k=1).astype('uint8')
        ).float().to(self.device)
        combined_mask = torch.max(trg_mask, look_ahead)

        n_var = SNR_to_noise(snr_db)

        with torch.no_grad():
            # Encoder
            enc_output = self.model.encoder(src, src_mask)
            channel_enc_output = self.model.channel_encoder(enc_output)
            Tx_sig = self.power_norm(channel_enc_output)

            # Channel
            if self.channel_type == "AWGN":
                Rx_sig = self.channels.AWGN(Tx_sig, n_var)
            elif self.channel_type == "Rayleigh":
                Rx_sig = self.channels.Rayleigh(Tx_sig, n_var)
            else:
                Rx_sig = self.channels.AWGN(Tx_sig, n_var)

            # Decoder
            channel_dec_output = self.model.channel_decoder(Rx_sig)
            dec_output = self.model.decoder(trg_inp, channel_dec_output, combined_mask, src_mask)
            pred = self.model.dense(dec_output)  # [1, seq_len-1, vocab_size]

            # Greedy decode
            pred_tokens = pred.argmax(dim=-1).squeeze(0).tolist()

        decoded_text = self.detokenize(pred_tokens)

        return {
            "original": text,
            "decoded": decoded_text,
            "snr_db": snr_db,
            "channel": self.channel_type,
            "compression_ratio": 16.0 / self.d_model,  # 16-dim channel / 128-dim
        }


if __name__ == "__main__":
    ch = DeepSCChannel()
    result = ch.transmit("the quick brown fox jumps over the lazy dog", snr_db=10)
    print(f"Original:   {result['original']}")
    print(f"Decoded:    {result['decoded']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Channel:    {result['channel']} @ {result['snr_db']}dB")
