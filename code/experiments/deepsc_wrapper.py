import os
import sys
import torch
import numpy as np

DEEPSC_PATH = r"D:\MP2\repos\DeepSC"
sys.path.insert(0, DEEPSC_PATH)

# Checkpoint search order: wrapper dir first (has best_model.pth), then repo ckpt dir
_CHECKPOINT_DIRS = [
    r"D:\MP2\models\deepsc\text",
    r"D:\MP2\repos\DeepSC\ckpt",
]


class DeepSCWrapper:
    """
    Wrapper around trained DeepSC model for use in experiments.
    Uses the repo's transceiver.DeepSC directly (no forward() method,
    inference via components + greedy decode).

    NOTE: The repo's DeepSC passes num_vocab as max_len to PositionalEncoding.
    This is a known quirk — the PE buffer is shape [1, num_vocab, d_model]
    but only [:, :seq_len] is used at inference time.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        vocab_path: str = r"D:\MP2\repos\DeepSC\europarl\vocab.json",
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path  # If None, auto-search
        self.vocab_path = vocab_path
        self.model = None
        self.token_to_idx = None
        self.idx_to_token = None
        self.pad_idx = None
        self.start_idx = None
        self.end_idx = None
        self.num_vocab = None

    def _find_checkpoint(self, epoch: int = None) -> str:
        """Find checkpoint file, searching multiple directories."""
        if self.checkpoint_path and os.path.isdir(self.checkpoint_path):
            dirs_to_search = [self.checkpoint_path] + _CHECKPOINT_DIRS
        else:
            dirs_to_search = _CHECKPOINT_DIRS

        # Priority: best_model.pth, then requested epoch, then latest
        for d in dirs_to_search:
            if not os.path.isdir(d):
                continue
            # Try best_model.pth first
            best = os.path.join(d, "best_model.pth")
            if os.path.exists(best):
                return best
            # Try specific epoch
            if epoch is not None:
                ckpt = os.path.join(d, f"checkpoint_{str(epoch).zfill(2)}.pth")
                if os.path.exists(ckpt):
                    return ckpt
            # Try latest
            ckpts = sorted([f for f in os.listdir(d) if f.endswith(".pth")])
            if ckpts:
                return os.path.join(d, ckpts[-1])

        raise FileNotFoundError(
            f"No DeepSC checkpoints found in any of: {dirs_to_search}"
        )

    def load_model(self, epoch: int = None):
        """Load the trained DeepSC model from checkpoint."""
        import json
        from models.transceiver import DeepSC

        # Load vocab
        with open(self.vocab_path) as f:
            vocab = json.load(f)
        self.token_to_idx = vocab["token_to_idx"]
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.num_vocab = len(self.token_to_idx)
        self.pad_idx = self.token_to_idx.get("<PAD>", 0)
        self.start_idx = self.token_to_idx.get("<START>", 1)
        self.end_idx = self.token_to_idx.get("<END>", 2)

        # Find and load checkpoint
        ckpt_file = self._find_checkpoint(epoch)

        # Construct model with correct params
        # IMPORTANT: repo passes num_vocab as max_len (PE buffer is [1, num_vocab, d_model])
        self.model = DeepSC(
            num_layers=4,
            src_vocab_size=self.num_vocab,
            trg_vocab_size=self.num_vocab,
            src_max_len=self.num_vocab,   # repo quirk — PE shape matches vocab size
            trg_max_len=self.num_vocab,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1,
        ).to(self.device)

        state = torch.load(ckpt_file, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"DeepSC loaded from {ckpt_file}")
        print(f"  Vocab size: {self.num_vocab}, params: {sum(p.numel() for p in self.model.parameters()):,}")

    def _tokenize(self, text: str, max_len: int = 30) -> torch.Tensor:
        """Convert text to token tensor using DeepSC vocab."""
        words = text.lower().split()
        tokens = [self.start_idx]
        for w in words:
            tokens.append(self.token_to_idx.get(w, self.token_to_idx.get("<UNK>", 3)))
        tokens.append(self.end_idx)
        # Truncate and pad
        tokens = tokens[:max_len]
        tokens += [self.pad_idx] * (max_len - len(tokens))
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def _detokenize(self, token_ids: list) -> str:
        """Convert token IDs back to text."""
        words = []
        for idx in token_ids:
            tok = self.idx_to_token.get(idx, "<UNK>")
            if tok == "<END>":
                break
            if tok not in ("<START>", "<PAD>"):
                words.append(tok)
        return " ".join(words)

    def encode_decode(self, text: str, snr_db: float = 10.0) -> dict:
        """
        Encode text through DeepSC, pass through AWGN channel, and decode.
        Uses the repo's greedy_decode logic since DeepSC has no forward().

        Returns dict with original text, decoded text, and channel info.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first.")

        import math

        # SNR to noise variance
        snr_linear = 10 ** (snr_db / 10)
        n_var = 1 / np.sqrt(2 * snr_linear)

        src = self._tokenize(text)

        with torch.no_grad():
            # Encode
            src_mask = (src == self.pad_idx).unsqueeze(-2).float().to(self.device)
            enc_out = self.model.encoder(src, src_mask)
            ch_enc = self.model.channel_encoder(enc_out)

            # Power normalize
            power = torch.mean(ch_enc * ch_enc, dim=-1, keepdim=True).sqrt()
            ch_enc_norm = ch_enc / torch.clamp(power, min=1.0)

            # AWGN channel
            noise = torch.normal(0, n_var, size=ch_enc_norm.shape).to(self.device)
            rx_sig = ch_enc_norm + noise

            # Channel decode
            ch_dec = self.model.channel_decoder(rx_sig)

            # Greedy decode (auto-regressive)
            decoded = self._greedy_decode(ch_dec, src_mask, max_len=30)

        decoded_text = self._detokenize(decoded[0].cpu().tolist())

        # Compute semantic similarity
        sim = self._compute_similarity(text, decoded_text)

        return {
            "original": text,
            "decoded": decoded_text,
            "snr_db": snr_db,
            "noise_var": float(n_var),
            "semantic_similarity": sim,
        }

    def _greedy_decode(self, memory, src_mask, max_len=30):
        """Auto-regressive greedy decode using the decoder."""
        import sys as _sys
        _sys.path.insert(0, DEEPSC_PATH)
        from utils import subsequent_mask

        batch_size = memory.size(0)
        outputs = torch.ones(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.start_idx)

        for _ in range(max_len - 1):
            trg_mask = (outputs == self.pad_idx).unsqueeze(-2).float()
            look_ahead = subsequent_mask(outputs.size(1)).float().to(self.device)
            combined_mask = torch.max(trg_mask, look_ahead)

            dec_out = self.model.decoder(outputs, memory, combined_mask, None)
            pred = self.model.dense(dec_out)
            _, next_word = torch.max(pred[:, -1:, :], dim=-1)
            outputs = torch.cat([outputs, next_word], dim=1)

        return outputs

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_sent_model'):
                self._sent_model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = self._sent_model.encode([text1, text2], convert_to_tensor=True)
            return torch.nn.functional.cosine_similarity(
                embs[0].unsqueeze(0), embs[1].unsqueeze(0)
            ).item()
        except Exception:
            # Fallback: word overlap ratio
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / max(len(words1), len(words2))

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Public API for semantic similarity."""
        return self._compute_similarity(text1, text2)


if __name__ == "__main__":
    print("=" * 60)
    print("DEEPSC WRAPPER TEST")
    print("=" * 60)

    wrapper = DeepSCWrapper()
    wrapper.load_model()

    test_sentences = [
        "send the image quickly",
        "deliver with high resolution",
        "reliable transfer needed",
        "high quality audio streaming",
        "low latency video call",
    ]

    print("\nEncode-Decode test (SNR=10dB):")
    for sent in test_sentences:
        result = wrapper.encode_decode(sent, snr_db=10)
        print(f"  Original:  {result['original']}")
        print(f"  Decoded:   {result['decoded']}")
        print(f"  Sim:       {result['semantic_similarity']:.4f}")
        print()

    print("SNR sweep test:")
    for snr in [0, 5, 10, 15, 20]:
        result = wrapper.encode_decode("send the image quickly", snr_db=snr)
        print(f"  SNR={snr}dB: sim={result['semantic_similarity']:.4f}, "
              f"decoded='{result['decoded'][:50]}'")

    print("\nDeepSC wrapper test complete.")
