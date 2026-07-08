# CSCA Project — Complete Context for New Chat

## What This Project Is

**CSCA (Cognitive Semantic Communication Architecture)** is an implementation of the paper:

> Y. Sun et al., "Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication," IEEE TMC, Vol. 25, No. 1, Jan 2026 (DOI: 10.1109/TMC.2025.3590723)

**Author:** Sreekar (VU39SR), 5th Semester ECE, VCE Hyderabad
**Guide:** Dr. Sandeep Joshi, BITS Pilani
**Hardware:** RTX 4050 6GB VRAM | 16GB RAM | Windows 11 + WSL2
**Windows Path:** D:\MP2
**WSL2 Path:** /mnt/d/MP2
**Venv:** D:\MP2\.venv\ (Python 3.12, CUDA 12.4)

The paper proposes a three-layer architecture for intent-driven multimodal semantic communication:
- **Layer 1 (LAM / Left Brain):** Large AI Model for intent parsing and modality alignment
- **Layer 2 (HDM / Right Brain):** HAN + DDPM policy for resource allocation
- **Layer 3:** Channel simulation with 3GPP parameters

---

## Directory Structure

```
D:\MP2\
  code\
    csca_pipeline.py          # Full pipeline: LAM -> HDM -> Channel -> CSCQI
    lam\
      intent_parser.py         # Qwen2-VL-7B for intent parsing
      lkb.py                   # Local Knowledge Base (20 templates, RAG retrieval)
      rag_intent_parser.py     # RAG + Self-RAG (ISREL/ISSUP reflection tokens)
      source_simplifier.py     # Algorithm 1: MSS selection, MIM (Eq.5), MCS (Eq.6)
      modality_alignment.py    # Text/Image/Audio -> unified text
      whisper_fallback.py      # Whisper transcription without ffmpeg
    hdm\
      han_network.py           # HAN (2-layer HANConv, 128-dim, 8 heads)
      ddpm_policy.py           # DDPM policy (MLP denoiser, 6 steps)
      hdm_trainer.py           # Actor-Critic training (Algorithm 2)
      csc_graph_builder.py     # CSC graph (5 node types, 7 edge types)
    channel\
      sim_channel.py           # 3GPP channel (path loss, shadow fading, SINR, delay)
      relay_selection.py       # Semantic relay selection (Eq. 15 approximation)
      mcs_table.py             # 3GPP TS 38.214 MCS tables (3 tables, 28-30 entries)
    evaluation\
      cscqi.py                 # CSCQI (Eq.17), ISR, semantic accuracy, compression ratio
      shaped_reward.py         # Shaped reward for HDM training
      dataset_loader.py        # Dataset loading utilities
    utils\
      reproducibility.py       # set_seed(42) for all experiments
    experiments\
      run_all_experiments.py   # Main experiment suite (Fig 9a, 9c, 12a, 13)
      multimodal_eval.py       # Multimodal evaluation (Fig 6) — HAS ISSUES, see below
      train_baselines.py       # SAC, PPO, AC baseline training (2000 episodes each)
      baselines.py             # SAC, AC, PPO, Static baseline classes
      deepsc_baseline.py       # DeepSC baseline (uses deepsc_wrapper.py)
      deepsc_wrapper.py        # DeepSC wrapper (loads trained repo model)
      dasc_baseline.py         # DASC baseline (mel spectrogram autoencoder)
      rlasc_baseline.py        # RL-ASC baseline (CNN autoencoder for images)
      image_eval.py            # Image PSNR evaluation (SD 2.1) — not yet run end-to-end
      psnr_eval_lightweight.py # Lightweight PSNR (CNN autoencoder, no SD)

  data\raw\
    sst_sentences.json         # 2000 SST sentences (text dataset)
    sst2_500.json              # 500 SST2 sentences
    audio_voxceleb\            # 4874 real VoxCeleb WAV files
    images_oxford\             # 3678 real Oxford Buildings JPG files
    google_landmark_test_000.tar  # 285 MB Google Landmarks archive

  results\software\
    checkpoints\               # HDM and baseline checkpoints
      hdm_ep2000.pt           # Best retrained HDM (Jul 8, ep7000 total)
      sac_trained.pt           # SAC baseline (arch: [256, 256, 45])
      ppo_trained.pt           # PPO baseline (arch: [256, 256, 45])
      ac_trained.pt            # AC baseline (arch: [128, 45])
    fig9a_isr_vs_tasks.png     # ISR vs tasks plot
    fig9c_delay_vs_sinr.png    # Delay vs SINR plot
    fig12a_cscqi_convergence.png  # CSCQI convergence plot
    fig13_ablation.png         # Ablation study plot
    scale_comparison_n5_n10_n15.png  # Scale comparison plot
    results_summary.csv        # Summary table
    final\                     # Multimodal eval outputs (when run)

  repos\
    DeepSC\                    # DeepSC repo (Xie et al. 2021) — trained 50 epochs
      europarl\vocab.json      # 22234 token vocab
      ckpt\checkpoint_50.pth   # DeepSC trained weights
    LAMMSC\                    # Paper's reference implementation repo

  models\
    Qwen.Qwen2-VL-7B.Q4_K_M.gguf  # LAM model (4-bit quantized)
    whisper\                   # Whisper base model (282 MB)
    stable-diffusion\          # SD 2.1 v2-1_768-ema-pruned.ckpt (4.86 GB)
      sd21_config\             # Config files cached from HuggingFace

  log.txt                      # Implementation log (2400+ lines)
```

---

## What Has Been Implemented (77% of paper)

### Core Architecture (IMPLEMENTED)
- Full CSCA pipeline: LAM -> HDM -> Channel -> CSCQI
- LAM: Qwen2-VL-7B Q4_K_M via llama-cpp-python (CUDA)
- LKB: 20 IG1253-format intent templates, cosine retrieval + CrossEncoder reranking
- RAG: ISREL + ISSUP reflection tokens with retry
- HDM: 2-layer HANConv (128-dim, 8 heads) + 6-step DDPM MLP policy
- CSC Graph: 5 node types (csca, relay, message, base_station, init), 7 edge types

### Algorithms (IMPLEMENTED)
- Algorithm 1: MSS selection (BERT cosine sim, eta=0.85)
- Algorithm 2: HDM training (Actor-Critic + DDPM + HAN)
- Source simplification with MIM (Eq. 5) and MCS selection (Eq. 6)

### Channel Model (IMPLEMENTED)
- Path loss: 128.1 + 37.6*log10(d_km) — exact 3GPP
- Shadow fading: N(0, 8^2) — 3GPP TR 38.901
- Clusters: LOS=12, NLOS=19, rays=20 — 3GPP TR 36.873
- Noise PSD: -174 dBm/Hz
- MCS: 3GPP TS 38.214 Tables 1,2,3 (28-30 entries)
- Finite blocklength rate (Eq. 10), delay (Eq. 12-13), distortion (Eq. 14)
- Inter-cell interference (Eq. 8): top-6 RSRP

### Training (COMPLETED)
- HDM: 5000 episodes original + 2000 episodes retrained (ep7000 total)
  - Used shaped reward with delay penalty
  - Curriculum learning (easy/medium/hard phases)
  - Experience replay (capacity=10000, batch=32)
  - LR scheduler: warmup + cosine decay
- SAC: 2000 episodes
- PPO: 2000 episodes
- AC: 2000 episodes
- DeepSC: 50 epochs on Europarl (in DeepSC repo)

### Experiments (COMPLETED — Prompt 7)
All experiments ran successfully:
- Fig 9a: ISR vs tasks (8 task counts, 5 methods, 3 seeds x 200 episodes)
- Fig 9c: Delay vs SINR (6 SNR points)
- Fig 12a: CSCQI convergence (from checkpoint)
- Fig 13: Ablation (HDM vs no-HAN vs no-DDPM)
- Scale comparison: n=5, n=10, n=15
- Summary table: all methods, ISR/CSCQI/Delay

### Results Summary (from Prompt 7 run)
| Method | ISR | CSCQI | Delay |
|--------|-----|-------|-------|
| HDM | 0.455 | 2.57 | 1.07s |
| SAC | 0.456 | 2.57 | 1.07s |
| AC | 0.446 | 2.57 | 1.09s |
| PPO | 0.454 | 2.57 | 1.08s |
| Static | 0.453 | 2.57 | 1.08s |

---

## Known Issues and Gaps

### 1. HDM Does Not Outperform Baselines (CRITICAL)
HDM ISR ≈ baselines. Root cause: proportional bandwidth allocation.
- HDM outputs BW values via sigmoid in [0,1]
- `sim_channel.py` step() normalizes: `bw_frac / sum(all_fracs) * total_bw`
- When HDM outputs uniform values (all ~1.0), each task gets equal BW = same as static
- The reward function and environment don't create enough differentiation
- **This needs a fundamentally different approach to BW allocation**

### 2. Delay Is Constant Across SINR
Delay = 1.07s regardless of SINR. Root cause: proportional BW allocation gives
each task the same BW regardless of HDM output or SINR.

### 3. Multimodal Evaluation Stuck (Prompt 8)
`multimodal_eval.py` hangs because:
- MSS Algorithm 1 computes SentenceTransformer embeddings for every subsequence
- For 100 sentences x 6 SNR x ~120 subsequences = 72,000 embedding calls
- Each call takes ~50ms = ~60 minutes just for text modality
- **Fix needed: skip MSS for evaluation (use word-drop), or batch embeddings**

### 4. BERT vs SentenceTransformer
- `source_simplifier.py` uses BERT (`bert-base-uncased`) for Eq. 4 semantic distance
- `cscqi.py` uses SentenceTransformer (`all-MiniLM-L6-v2`) for semantic accuracy
- Paper uses Qwen2-VL as LAM — we have it loaded via llama-cpp-python
- **Should consolidate to one embedding model**

### 5. Image Evaluation Not Run End-to-End
- `image_eval.py` exists, SD 2.1 load test passed
- Needs Qwen2-VL for description + SD for reconstruction
- Each image requires 2 model loads/unloads (6GB VRAM constraint)
- **Not yet run**

### 6. DeepSC Integration
- DeepSC repo has trained weights (50 epochs Europarl)
- `deepsc_wrapper.py` loads the repo's model correctly
- DeepSC has NO forward() method — inference via components (encoder -> channel_encoder -> channel -> channel_decoder -> greedy_decode)
- **Working but not integrated into main experiment suite**

### 7. Relay Selection (Eq. 15)
- Paper defines semantic mutual information with probability distributions
- We approximate using cosine similarity / string overlap
- Paper says distributions come from LKB deployment (we don't have this table)
- **Approximation is reasonable, not exact**

---

## What Needs To Be Done Next

### Priority 1: Fix Multimodal Evaluation (Prompt 8)
The `multimodal_eval.py` script is stuck. Options:
- **Option A:** Skip MSS entirely, use word-drop compression for evaluation
- **Option B:** Batch all SentenceTransformer embeddings at once (not per-subsequence)
- **Option C:** Use SentenceTransformer for everything (replace BERT in source_simplifier.py)

### Priority 2: Fix HDM vs Baseline Differentiation
The proportional BW allocation makes all methods equivalent. Need:
- Different BW allocation scheme (not proportional)
- Or different reward signal that penalizes uniform allocation
- Or different environment where intelligent allocation actually matters

### Priority 3: Run Image Evaluation
- `image_eval.py` is ready (SD 2.1 load test passed)
- Needs Qwen2-VL + SD 2.1 sequential loading
- Estimated: ~2-3 min per image, ~15 min for 5 images

### Priority 4: Run DeepSC Comparison
- DeepSC wrapper works
- Need to integrate into experiment suite for Fig 6 comparison

---

## Key Code Patterns

### How to Load Trained HDM
```python
# In run_all_experiments.py
han, hdm = load_trained_hdm(n_tasks=5, n_relays=5)
# Loads newest checkpoint by modification time (hdm_ep2000.pt)
```

### How to Load Trained Baselines
```python
bl = load_trained_baseline("SAC", 45, DEVICE)
# Auto-detects architecture from checkpoint keys
# Returns wrapper with get_action() method
```

### How to Parse Actions (handles dimension mismatch)
```python
parsed = parse_action(action_raw, n_tasks, n_relays, n_mcs)
# Pads when too small, truncates when too large
result = env.step(parsed, state)
```

### How Channel Simulation Works
```python
from sim_channel import WirelessChannel, MultiCSCAEnvironment
channel = WirelessChannel(bandwidth_hz=10e6)
result = channel.simulate_channel(
    tx_power_dbm=23, distance_km=0.5,
    data_size_bits=1e6, target_snr_db=10
)
# Returns: sinr_db, rate_bps, delay_s, distortion, mcs_index
```

### How to Run Experiments
```powershell
cd D:\MP2
.\.venv\Scripts\Activate.ps1
python code\experiments\run_all_experiments.py
```

---

## Paper Verification Report

Total components: 65
- Implemented: 50 (77%)
- Partial: 11 (17%)
- Missing: 4 (6%)
- Critical gaps: 4

Full report at: D:\MP2\results\software\paper_verification_report.txt

---

## Datasets

| Dataset | Paper | Our Implementation | Status |
|---------|-------|-------------------|--------|
| Text | Stanford Sentiment Treebank | 2000 SST sentences | EXACT |
| Audio | VoxCeleb1 | 4874 VoxCeleb WAV files | EXACT |
| Images | Google Landmarks v2 | 3678 Oxford Buildings JPGs | CLOSE (proxy) |
| DeepSC | Europarl | 73,472 sentences | EXACT |

---

## Evaluation Metrics Used

- **ISR** (Intent Satisfaction Rate): fraction of tasks where delay <= intent AND quality <= intent
- **CSCQI** (Eq. 17): w_tau * (tau_max - tau_S)/(tau_max - tau_S_int) + w_vartheta * (vartheta_max - vartheta_S)/(vartheta_max - vartheta_S_int)
- **Semantic Accuracy**: cosine similarity via SentenceTransformer
- **Compression Ratio**: simplified_words / original_words
- **PSNR** (Eq. 37): for images (implemented in RL-ASC baseline, not main pipeline)
- **Communication Delay** (Eq. 12): D_S / nu

---

## Dependencies

```
torch 2.6.0+cu124
torch-geometric 2.8.0
transformers 5.12.1
sentence-transformers 5.6.0
diffusers 0.39.0 (upgraded from 0.38.0 for transformers 5.x compat)
llama-cpp-python 0.3.32 (CUDA)
openai-whisper 20250625
pytorch_lightning 2.6.5
```

---

## Fixes Applied During This Session

1. **relay_selection.py:** Added `import torch` (was missing), changed compute_symbol_similarity to fast string overlap by default
2. **sim_channel.py:** Message features now encode real task info (data_size_norm, delay_intent, quality_intent, urgency) instead of random values
3. **shaped_reward.py:** Strong delay penalty (-2x overshoot), weights 0.6 delay / 0.4 quality, range [-2.0, 2.0]
4. **hdm_trainer.py:** Added checkpoint resume, 2000 episodes with checkpoint every 200
5. **run_all_experiments.py:** load_trained_hdm sorts by modification time, load_trained_baseline auto-detects architecture, parse_action handles dimension mismatch, BaselineWrapper with get_action()
6. **source_simplifier.py:** Temporarily changed BERT to SentenceTransformer (REVERTED — back to BERT)
7. **cscqi.py:** Temporarily added model parameter (REVERTED — back to original)
8. **diffusers single_file_utils.py:** Patched stabilityai -> sd2-community (HF repo deleted)
