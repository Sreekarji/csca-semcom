# Implementation Notes

## Key Differences from Paper

| Component | Paper | This Implementation | Reason |
|-----------|-------|---------------------|--------|
| LAM | LLaVA-NeXT-Interleave (7B, 8-bit) | Qwen2-VL-7B (4-bit GGUF) | Consumer GPU (6GB VRAM) |
| Channel | Full 3GPP TR 38.901 mmWave MIMO | Simplified numpy model | GPU compute constraints |
| Training | ~100 episodes (Table II) | 500 episodes | Extended for convergence |
| Actor loss | Eq. 33 (paper) | DDPO-SF approximation | Implementation complexity |
| Compression | DeepSC neural codec | MSS + DeepSC hybrid | Vocabulary mismatch |

## What Works Correctly
- CSCQI metric (Eq. 17) — exact match
- HAN graph structure (5 node types, 3 edge types) — matches paper
- DDPM noise schedule (Eq. 31-32) — matches paper
- N=6 optimal denoising steps — confirmed matches paper Fig 12a
- Scale comparison trend — HDM advantage grows with n (matches Section VI.C)
- All baselines trained: SAC, PPO, AC, DeepSC

## Training Configuration (Table II)
- HAN: 3 layers, 256 hidden, 6 heads
- DDPM: N=6 denoising steps, MLP 256-dim
- Batch size: 256
- Learning rate: 0.001
- Decay rate (gamma): 0.95
- Intent range: delay 50-1000ms, quality 90-100%
