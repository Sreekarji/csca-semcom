"""
Central path configuration for CSCA project.
All absolute paths go here. Import this in every module instead of hardcoding.
"""
from pathlib import Path

# Project roots
MP2_ROOT     = Path(r"D:\MP2")
CODE_ROOT    = MP2_ROOT / "code"

# Model paths
MINIML_PATH  = MP2_ROOT / "all-MiniLM-L6-v2"
DEEPSC_PATH  = MP2_ROOT / "models" / "deepsc" / "text" / "best_model.pth"
QWEN_PATH    = MP2_ROOT / "models" / "Qwen.Qwen2-VL-7B.Q4_K_M.gguf"

# Data paths
DATA_PATH    = MP2_ROOT / "data"
CHECKPOINT_PATH = MP2_ROOT / "checkpoints"
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

# Repo paths
DEEPSC_REPO  = MP2_ROOT / "repos" / "DeepSC"
