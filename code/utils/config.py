"""Shared paths and device config."""
import os
import torch

# Auto-detect project root (parent of the directory containing this file)
ROOT    = os.environ.get("MP3_ROOT", os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT    = os.path.join(ROOT, "checkpoints")
LOG_PATH = os.path.join(ROOT, "log.txt")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg: str):
    from datetime import datetime
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
