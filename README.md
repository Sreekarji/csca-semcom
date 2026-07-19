# CSCA-SemCom: Edge LAM Agent-Empowered Cognitive Multimodal Semantic Communication

Reproduction of Sun et al., *IEEE Transactions on Mobile Computing*, 2026.  
**"Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication"**

> Student project — BITS Pilani, 5th Sem ECE | Supervisor: Dr. Sandeep Joshi  
> Hardware: RTX 4050 6GB VRAM | Python 3.12 | Windows

---

## Paper Claim

| Metric | Paper Result vs Baselines |
|--------|--------------------------|
| Intent Satisfaction Rate (ISR) | +42.19% over SAC/PPO/AC |
| Semantic Accuracy | +29.75% |
| Communication Delay | -33.40% |

---

## Architecture

```
Raw State (channel + intents)
        ↓
HAN (3 layers, 256-dim, 8 heads)     ← Heterogeneous Graph Attention Network
                                        Eq. 21-26 from paper
        ↓
Graph Embedding G^L_t [1×256]
Message Embeddings [n_tasks×256]
        ↓
HDM Policy (DDPM, N=6 denoising steps)   ← Paper Algorithm 2
                                        Eq. 28-32 from paper
        ↓
Action: {BW [5], Relay [5×5], MCS [5×3]}
        ↓
MultiCSCAEnvironment (3GPP channel)
        ↓
CSCQI Reward (Eq. 17) + ISR
```

---

## Key Implementation Decisions

| Gap | Fix Applied |
|-----|-------------|
| Hard negative reward for unmet intent | `cscqi.py`: violations → negative CSCQI |
| Paper log-prob (Eq. 28-29) | `ddpm_policy.py`: collect_trajectory with reparameterization |
| BW softmax differentiation | `sim_channel.py`: BW_SOFTMAX_TEMP=0.5 |
| Message embeddings grouped by CSCA | `ddpm_policy.py` + `mlp_policy.py`: mean pooling |
| Congestion curriculum | `hdm_trainer_congestion.py`: tpc cycles [1,2,4,6,10] |
| Actor loss scaling | `hdm_trainer.py`: advantage scaled ×10 before actor update |

---

## Environment Parameters

| Parameter | Value |
|-----------|-------|
| Bandwidth | 5 MHz total |
| Difficulty | medium (delay 0.5–1.2s, quality 0.55–0.75) |
| n_cscas | 5 |
| n_relays | 5 |
| n_mcs | 3 |
| Denoising steps N | 6 |
| HAN hidden dim | 256 |
| HAN layers | 3 |
| HAN heads | 8 |
| Action dim | 45 (5 BW + 25 relay + 15 MCS) |

---

## Oracle Gap Analysis

Proves the learning problem is real — uniform allocation is deeply suboptimal:

| tasks_per_csca | Oracle ISR | Uniform ISR | Gap |
|---------------|------------|-------------|-----|
| 1 | 0.644 | 0.202 | +219% |
| 4 | 1.000 | 0.054 | +1769% |
| 10 | 0.805 | 0.023 | +3368% |
| 20 | 0.383 | 0.011 | +3321% |

---

## Current Results

### ISR vs Tasks (congestion-trained models, medium difficulty)

| tpc | total | HDM | MLP | SAC | PPO | AC | Static |
|-----|-------|-----|-----|-----|-----|-----|--------|
| 1 | 5 | 0.727 | 0.766 | 0.780 | 0.759 | 0.770 | 0.756 |
| 2 | 10 | 0.507 | 0.527 | 0.516 | 0.530 | 0.531 | 0.546 |
| 4 | 20 | 0.365 | 0.387 | 0.396 | 0.404 | 0.405 | 0.420 |
| 6 | 30 | 0.253 | 0.292 | 0.316 | 0.332 | 0.349 | 0.369 |
| 8 | 40 | 0.166 | 0.196 | 0.215 | 0.244 | 0.252 | 0.284 |
| 10 | 50 | 0.170 | 0.221 | 0.243 | 0.268 | 0.299 | 0.325 |
| 12 | 60 | 0.164 | 0.220 | 0.264 | 0.297 | 0.335 | 0.381 |
| 15 | 75 | 0.102 | 0.158 | 0.203 | 0.241 | 0.291 | 0.363 |
| 20 | 100 | 0.049 | 0.094 | 0.150 | 0.202 | 0.275 | 0.305 |

### Baseline Training Results (2000 episodes each, congestion curriculum)

| Method | Final avg ISR | Best ISR |
|--------|---------------|----------|
| SAC | 0.497 | 0.502 |
| PPO | 0.500 | 0.508 |
| AC | 0.497 | 0.510 |

### Known Issue

**Static uniform allocation currently beats all learned methods at tpc >= 2.** The oracle gap test proves non-uniform allocation can achieve 80-100% ISR where uniform gets 2-5%, but the learned policies (HDM, MLP, SAC, PPO, AC) have not yet learned to exploit this gap. The training approach needs further work — likely requiring the full paper action space (BW + relay + MCS jointly) and/or environment persistence (Eqs. 19-20 intent decay under queuing).

---

## File Structure

```
D:\MP2
├── ddpm_policy.py              # HDM: DDPM policy + CriticNetwork
├── hdm_trainer.py              # HDM Actor-Critic training loop
├── hdm_trainer_congestion.py   # Training with congestion curriculum
├── hdm_continue_v2.py          # Resume from best checkpoint
├── han_network.py              # Heterogeneous Attention Network
├── csc_graph_builder.py        # CSC graph construction (Eq. 21)
├── mlp_policy.py               # HAN+MLP ablation policy
├── mlp_trainer.py              # MLP training loop
├── sim_channel.py              # 3GPP wireless channel simulation
├── cscqi.py                    # CSCQI metric (Eq. 17) + ISR
├── baselines.py                # SAC, PPO, AC, Static baselines
├── baseline_trainer_v2.py      # Baseline training with curriculum
├── mcs_table.py                # 3GPP TS 38.214 MCS tables
├── relay_selection.py          # Semantic relay selection
├── shaped_reward.py            # Reward shaping utilities
├── reproducibility.py          # Seed setting
├── joshi_eval_v2.py            # Full evaluation across tpc values
├── run_all_experiments.py      # Paper Fig 9a/9c/12a/13 experiments
├── oracle_gap_test.csv         # Oracle vs uniform baseline analysis
└── results/software/
    └── checkpoints/
        ├── hdm_best.pt                 # Best HDM checkpoint
        ├── hdm_congestion_best.pt      # Best congestion-trained HDM
        └── mlp_best.pt                 # Best MLP checkpoint
```

---

## How to Run

```bash
# 1. Train HDM with congestion curriculum (2000 episodes, ~2 hours)
python hdm_trainer_congestion.py

# 2. Train baselines with same curriculum
python baseline_trainer_v2.py

# 3. Continue HDM from best checkpoint (3000 more episodes)
python hdm_continue_v2.py

# 4. Full evaluation
python joshi_eval_v2.py

# 5. All paper experiments (Fig 9a, 9c, 12a, 13)
python run_all_experiments.py
```

---

## Dependencies

```bash
pip install torch torch-geometric numpy matplotlib sentence-transformers
```

---

## Status

- [x] HAN + DDPM pipeline implemented
- [x] Paper Eq. 17, 28-32 implemented faithfully  
- [x] Hard negative reward (Sec V.A.3)
- [x] Oracle gap analysis proves learning opportunity exists
- [x] HDM beats Static at tpc=1 medium difficulty
- [x] Congestion curriculum training implemented
- [ ] Learned policies beat Static at tpc=10+ (known issue)
- [ ] Paper Fig 9a ISR gap reproduced at tpc=10+
- [ ] Full paper results table matched

---

## Reference

Y. Sun et al., "Edge Large AI Model Agent-Empowered Cognitive Multimodal 
Semantic Communication," *IEEE Trans. Mobile Comput.*, vol. 25, no. 1, 
Jan. 2026. DOI: 10.1109/TMC.2025.3590723
