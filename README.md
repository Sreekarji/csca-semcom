# CSCA — Cognitive Semantic Communication Architecture

Implementation of **"Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication"** (Sun et al., IEEE TMC, Vol. 25, No. 1, Jan 2026).

## Architecture

### Layer 1: LAM (Left Brain)
- **Qwen2-VL-7B** (4-bit quantized) for intent parsing
- **Local Knowledge Base (LKB)** — 20 IG1253-format templates with cosine retrieval + CrossEncoder reranking
- **RAG with Self-RAG** — ISREL + ISSUP reflection tokens
- **Source Simplifier** — Algorithm 1 (MSS selection, BERT embeddings, Eq. 4)
- **MIM** — Message Importance Metric (Eq. 5)
- **MCS Selection** — 3GPP TS 38.214 based (Eq. 6)

### Layer 2: HDM (Right Brain)
- **HAN** — Heterogeneous Attention Network (2-layer HANConv, 128-dim, 8 heads)
- **DDPM Policy** — MLP denoiser, 6 steps, noise schedule (Eq. 31-32)
- **Actor-Critic Training** — Algorithm 2 with curriculum learning
- **CSC Graph** — 5 node types, 7 edge types

### Layer 3: Channel Model
- **Path loss**: 128.1 + 37.6·log₁₀(d) — 3GPP model
- **Shadow fading**: N(0, 8²) — 3GPP TR 38.901
- **SINR**: φ/(σ² + ρ) — Eq. 9
- **MCS table**: 3GPP TS 38.214 (Tables 1-3, 28-30 entries)
- **Finite blocklength rate**: Eq. 10
- **Delay**: D_S/ν — Eq. 12
- **Distortion**: exp(-0.1·SINR_dB) — Eq. 14 proxy
- **Inter-cell interference**: top-6 RSRP — Eq. 8
- **Relay selection**: Section IV.B, Eq. 15 approximation

## Datasets
| Dataset | Source | Size |
|---------|--------|------|
| Text | Stanford Sentiment Treebank | 2,000 sentences |
| Audio | VoxCeleb1 | 4,874 WAV files |
| Images | Oxford Buildings | 3,678 JPG files |
| DeepSC | Europarl | 73,472 sentence pairs |

## Training
- **HDM**: 7,000 episodes (curriculum: easy→medium→hard)
- **SAC**: 2,000 episodes
- **PPO**: 2,000 episodes
- **Actor-Critic**: 2,000 episodes
- **DeepSC**: 50 epochs on Europarl

## Experiments
- Fig 9a: ISR vs number of tasks
- Fig 9c: Delay vs SINR
- Fig 12a: CSCQI convergence
- Fig 13: Ablation study
- Scale comparison: n=5, 10, 15 CSCA nodes

## Quick Start
```powershell
cd D:\MP2
.\.venv\Scripts\Activate.ps1

# Train HDM
python code\hdm\hdm_trainer.py

# Run all experiments
python code\experiments\run_all_experiments.py

# Multimodal evaluation
python code\experiments\multimodal_eval.py
```

## Project Structure
```
code/
├── csca_pipeline.py          # Full pipeline
├── lam/                      # Layer 1: LAM
│   ├── intent_parser.py      # Qwen2-VL intent parsing
│   ├── lkb.py                # Local Knowledge Base
│   ├── rag_intent_parser.py  # RAG + Self-RAG
│   ├── source_simplifier.py  # Algorithm 1 (MSS)
│   └── modality_alignment.py # Modality alignment
├── hdm/                      # Layer 2: HDM
│   ├── han_network.py        # HAN graph network
│   ├── ddpm_policy.py        # DDPM policy network
│   ├── hdm_trainer.py        # Training loop (Algorithm 2)
│   └── csc_graph_builder.py  # CSC graph construction
├── channel/                  # Layer 3: Channel
│   ├── sim_channel.py        # 3GPP channel simulation
│   ├── relay_selection.py    # Relay selection (Eq. 15)
│   └── mcs_table.py          # 3GPP MCS tables
├── evaluation/               # Metrics and rewards
│   ├── cscqi.py              # CSCQI, ISR, semantic accuracy
│   └── shaped_reward.py      # Shaped reward for training
├── experiments/              # Experiment scripts
│   ├── run_all_experiments.py
│   ├── multimodal_eval.py
│   ├── train_baselines.py
│   └── baselines.py
└── utils/
    └── reproducibility.py    # Seed setting
```

## Dependencies
- Python 3.12, PyTorch 2.6 (CUDA 12.4)
- PyTorch Geometric 2.8
- Transformers 5.12.1, Sentence-Transformers 5.6
- llama-cpp-python (CUDA), openai-whisper
- Stable Diffusion 2.1 (diffusers 0.39)

## Hardware
- GPU: NVIDIA RTX 4050 (6GB VRAM)
- RAM: 16GB
- OS: Windows 11
