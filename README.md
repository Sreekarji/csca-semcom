# CSCA: Cognitive Semantic Communication Agent

Implementation of the paper:
> Y. Sun et al., "Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication," *IEEE Transactions on Mobile Computing*, Vol. 25, No. 1, Jan 2026. DOI: 10.1109/TMC.2025.3590723

**Author:** Sreekar Balagoni, B.Tech ECE, Vasavi College of Engineering  
**Research Internship:** ViSRI Lab, BITS Pilani, under Dr. Sandeep Joshi  

---

## Overview

This repository implements the CSCA (Cognitive SemCom Agent) framework for personalized multimodal wireless communication. The system uses a large AI model (LAM) as the left brain for intent understanding, and a Heterogeneous Diffusion Model (HDM) as the right brain for communication policy generation.

### Architecture

```
User Intent (text/audio/image)
        ↓
[Layer 1: LAM Left Brain]
  - Qwen2-VL-7B for intent parsing
  - Local Knowledge Base (LKB) with RAG
  - Algorithm 1: Minimum Synonymous Subsequence
        ↓
[Layer 2: HDM Right Brain]
  - HAN: Heterogeneous Graph Attention Network (3 layers, 256-dim)
  - DDPM: Denoising Diffusion Policy (N=6 denoising steps)
  - Actor-Critic training with DDPO-SF loss
        ↓
[Layer 3: Channel]
  - 3GPP-compliant: path loss 128.1 + 37.6·log10(d), shadow fading N(0,8²)
  - Finite blocklength rate model
  - CSCQI metric (Eq. 17) for intent satisfaction measurement
```

---

## Repository Structure

```
csca/
├── code/
│   ├── lam/              # Layer 1: LAM components
│   │   ├── intent_parser.py
│   │   ├── rag_intent_parser.py
│   │   ├── lkb.py
│   │   ├── source_simplifier.py
│   │   └── modality_alignment.py
│   ├── hdm/              # Layer 2: HDM components
│   │   ├── han_network.py
│   │   ├── ddpm_policy.py
│   │   ├── hdm_trainer.py
│   │   └── csc_graph_builder.py
│   ├── channel/          # Layer 3: Channel simulation
│   │   ├── sim_channel.py
│   │   ├── relay_selection.py
│   │   └── deepsc_channel.py
│   ├── evaluation/       # Metrics
│   │   ├── cscqi.py
│   │   ├── shaped_reward.py
│   │   └── dataset_loader.py
│   ├── experiments/      # Experiment scripts
│   │   ├── run_all_experiments.py
│   │   ├── multimodal_eval.py
│   │   ├── train_baselines.py
│   │   └── baselines.py
│   ├── utils/
│   │   └── reproducibility.py
│   └── csca_pipeline.py  # End-to-end pipeline
├── results/
│   └── software/
│       └── final/        # Key result plots and CSVs
├── repos/                # External repos (DeepSC, LAMMSC, PDI-Diffusion)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric
pip install -r requirements.txt
```

---

## Models Required

Download separately (not included due to size):

| Model | Purpose | Source |
|-------|---------|--------|
| Qwen2-VL-7B-Instruct Q4_K_M | LAM intent parsing | HuggingFace |
| openai/whisper-base | Audio transcription | HuggingFace |
| sd2-community/stable-diffusion-2-1 | Image reconstruction | HuggingFace |
| all-MiniLM-L6-v2 | Semantic similarity | HuggingFace |

Place models in `models/` directory.

---

## Usage

```bash
# Run end-to-end demo
python code/csca_pipeline.py

# Train HDM
python code/hdm/hdm_trainer.py

# Run all experiments
python code/experiments/run_all_experiments.py

# Multimodal evaluation
python code/experiments/multimodal_eval.py
```

---

## Key Result: HAN Graph Attention Provides Structural Advantage

The ablation study confirms that HAN's heterogeneous graph attention 
provides genuine benefit that grows with network scale:

| Network Size | Full HDM | Without HAN | HDM Advantage |
|-------------|----------|-------------|---------------|
| n=5  tasks  | 0.095    | 0.091       | +4.4%         |
| n=10 tasks  | 0.032    | 0.026       | +23.1%        |
| n=15 tasks  | 0.015    | 0.008       | +87.5%        |
| n=18 tasks  | 0.009    | 0.003       | +200%         |
| n=20 tasks  | 0.006    | 0.002       | +200%         |

This matches the paper's Fig 13a qualitative finding:
HAN contribution becomes increasingly important as network complexity grows.

### Multimodal Evaluation (real datasets)

| Modality | Semantic Similarity | Compression | Dataset |
|----------|-------------------|-------------|---------|
| Text     | 0.310 (DeepSC)    | 0.125       | SST-2 (2000 sentences) |
| Audio    | 0.988             | 0.945       | VoxCeleb (4874 clips) |
| Image    | 0.999             | 0.979       | Oxford Buildings (3678 images) |

Note: Text similarity is low because DeepSC was trained on Europarl (European Parliament)
and evaluates poorly on SST movie review sentences. Audio and image use MSS compression.

### CSCQI Convergence
N=6 denoising steps confirmed optimal — matches paper Fig 12a exactly.

### Known Limitations
- ISR gap: ~20% vs paper's ~90% — attributed to channel environment calibration
  (bandwidth, data size, SINR ranges not fully specified in paper)
- Actor loss instability: DDPO-SF loss clamps to ±100 during training
- Text DeepSC: trained on Europarl, not evaluated domain (SST)
- No full 3GPP mmWave MIMO: simplified numpy channel model used

---

## Key Implementation Notes

- LLaVA-NeXT-Interleave (paper) replaced with Qwen2-VL-7B Q4_K_M (consumer GPU compatible)
- Full 3GPP mmWave MIMO replaced with simplified 3GPP channel model (numpy)
- Actor loss uses DDPO-SF (score function) formulation
- Trained on consumer hardware: RTX 4050 6GB VRAM

---

## Citation

```bibtex
@article{sun2026edge,
  title={Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication},
  author={Sun, Yan and others},
  journal={IEEE Transactions on Mobile Computing},
  volume={25},
  number={1},
  year={2026},
  doi={10.1109/TMC.2025.3590723}
}
```
