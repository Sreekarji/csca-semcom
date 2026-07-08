"""
DeepSC Baseline for CSCA comparison.
Implements the DeepSC architecture from Xie et al. 2021 (Fig 6 comparison).

Uses the repo's transceiver.DeepSC directly via DeepSCWrapper.
The repo model has NO forward() method — inference uses encoder -> channel_encoder
-> channel -> channel_decoder -> greedy_decode pipeline.

Compression ratio: 12.5% (d_model=128 -> 16 channel symbols)

TRAINED WEIGHTS AVAILABLE:
  - D:\MP2\models\deepsc\text\best_model.pth (50 epochs, Europarl)
  - Vocab: D:\MP2\repos\DeepSC\europarl\vocab.json (22234 tokens)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, r"D:\MP2\code\experiments")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepSCBaseline:
    """
    DeepSC baseline for text semantic communication.
    Compression: 12.5% (d_model=128 -> 16 channel symbols)

    Uses DeepSCWrapper for actual model inference.
    """

    def __init__(self, d_model=128, channel="AWGN", snr_db=10.0):
        self.d_model = d_model
        self.channel = channel
        self.snr_db = snr_db
        self.compression_ratio = 16 / d_model  # 12.5%
        self.wrapper = None

        print(f"[DeepSC] d_model={d_model}, channel={channel}, SNR={snr_db}dB")
        print(f"[DeepSC] Compression: {d_model} -> 16 = {self.compression_ratio:.1%}")

    def load_model(self):
        """Load trained DeepSC via wrapper."""
        from deepsc_wrapper import DeepSCWrapper
        self.wrapper = DeepSCWrapper()
        self.wrapper.load_model()
        print("[DeepSC] Model loaded successfully")

    def encode(self, text):
        """Encode text — returns compression info."""
        return {
            "channel_symbols": 16,  # d_model -> 16
            "compression_ratio": self.compression_ratio,
        }

    def encode_decode(self, text, snr_db=None):
        """Full encode-decode through channel."""
        if self.wrapper is None:
            self.load_model()
        snr = snr_db if snr_db is not None else self.snr_db
        return self.wrapper.encode_decode(text, snr_db=snr)

    def evaluate(self, texts, snr_range=None):
        """Evaluate across SNR range."""
        if self.wrapper is None:
            self.load_model()

        if snr_range is None:
            snr_range = [0, 5, 10, 15, 20, 25]

        results = {"compression_ratio": self.compression_ratio, "accuracy_by_snr": {}}

        for snr in snr_range:
            sims = []
            for text in texts[:20]:
                r = self.wrapper.encode_decode(text, snr_db=snr)
                sims.append(r.get("semantic_similarity", 0.0))
            results["accuracy_by_snr"][snr] = float(np.mean(sims))

        return results


if __name__ == "__main__":
    print("=" * 60)
    print("DEEPSC BASELINE TEST (using trained weights)")
    print("=" * 60)

    deepsc = DeepSCBaseline()
    deepsc.load_model()

    sentences = [
        "send it within 1 second",
        "deliver with high resolution",
        "reliable transfer needed",
        "send photo quickly",
        "high quality audio",
    ]

    print(f"\nTesting with {len(sentences)} sentences:")
    for i, sent in enumerate(sentences):
        enc = deepsc.encode(sent)
        print(f"  {i+1}. \"{sent[:60]}\"")
        print(f"     Compression: {enc['compression_ratio']:.1%}, symbols: {enc['channel_symbols']}")

    print(f"\nEncode-Decode test (SNR=10dB):")
    for sent in sentences:
        result = deepsc.encode_decode(sent, snr_db=10)
        print(f"  \"{sent[:40]}\" -> \"{result['decoded'][:40]}\"")
        print(f"    Sim: {result.get('semantic_similarity', 'N/A')}")

    print(f"\nSNR sweep (semantic similarity):")
    results = deepsc.evaluate(sentences)
    for snr, sim in sorted(results["accuracy_by_snr"].items()):
        print(f"  SNR={snr}dB: similarity={sim:.4f}")

    print("\nDeepSC baseline test complete.")
