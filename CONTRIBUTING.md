# Contributing to CSCA

## Known Issues / Open Problems

### 1. ISR Gap (High Priority)
**Problem:** Our ISR is ~20% vs paper's ~90%.  
**Suspected cause:** Channel environment miscalibration — bandwidth per CSCA, 
data size range, and SINR distribution not fully specified in the paper.  
**Status:** Awaiting guidance from paper authors.  
**How to help:** If you have access to the paper's simulation code or can 
reproduce the exact Table II settings, please open an issue.

### 2. Actor Loss Instability (High Priority)
**Problem:** DDPO-SF actor loss clamps to ±100 during training.  
**Suspected cause:** Gradient flow issue in collect_trajectory() or 
advantage normalization instability.  
**Status:** Working fix in progress.

### 3. DeepSC Text Similarity (Medium Priority)
**Problem:** Text similarity is 0.31 because DeepSC was trained on Europarl, 
not the SST evaluation domain.  
**Fix:** Fine-tune DeepSC on SST or use LAM-generated descriptions for evaluation.

### 4. Channel Model (Low Priority)
**Problem:** Using simplified numpy channel instead of full 3GPP TR 38.901 mmWave MIMO.  
**Fix:** DeepMIMO integration planned after core algorithm converges.

## Development Setup
See README.md for installation instructions.

## Running Tests
```bash
python -m pytest code/ -v  # Unit tests
python code/evaluation/cscqi.py  # CSCQI metric tests
python code/channel/sim_channel.py  # Channel simulation tests
```
