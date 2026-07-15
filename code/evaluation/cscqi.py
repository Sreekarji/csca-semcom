import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\MP2\models"

import numpy as np

# Paper's parameter settings (Table II)
TAU_MAX = 10.0       # Maximum delay in seconds
VARTHETA_MAX = 1.0   # Maximum distortion (normalized)
W_TAU = 0.5          # Delay weight (equal weights per paper)
W_VARTHETA = 0.5     # Quality weight

_semantic_model = None

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        local_path = r"D:\MP2\all-MiniLM-L6-v2"
        try:
            print(f"[cscqi] Loading SentenceTransformer from: {local_path}", flush=True)
            _semantic_model = SentenceTransformer(local_path, device="cpu")
            print("[cscqi] SentenceTransformer loaded successfully", flush=True)
        except Exception as e:
            print(f"[cscqi] SentenceTransformer failed: {type(e).__name__}: {e}", flush=True)
            raise RuntimeError(f"SentenceTransformer load failed: {e}")
    return _semantic_model


def compute_cscqi(
    tau_S: float,
    vartheta_S: float,
    tau_S_int: float,
    vartheta_S_int: float,
    w_tau: float = W_TAU,
    w_vartheta: float = W_VARTHETA,
) -> float:
    delay_score = (TAU_MAX - tau_S) / max(TAU_MAX - tau_S_int, 1e-8)
    quality_score = (VARTHETA_MAX - vartheta_S) / max(VARTHETA_MAX - vartheta_S_int, 1e-8)
    cscqi = w_tau * delay_score + w_vartheta * quality_score
    return cscqi


def compute_isr(tasks: list) -> float:
    if not tasks:
        return 0.0
    satisfied = 0
    for t in tasks:
        tau_s = t.get("tau_S", 999)
        vartheta_s = t.get("vartheta_S", 999)
        tau_int = t.get("tau_S_int", 0)
        vartheta_int = t.get("vartheta_S_int", 0)
        if tau_s <= tau_int and vartheta_s <= vartheta_int:
            satisfied += 1
    return satisfied / len(tasks)


def compute_semantic_accuracy(sent_text: str, recv_text: str) -> float:
    """
    Semantic accuracy using sentence-level cosine similarity.
    Loads all-MiniLM-L6-v2 from local disk — no network required.
    Returns value in [0, 1].
    """
    model = get_semantic_model()
    embs = model.encode([sent_text, recv_text], convert_to_numpy=True)
    norm0 = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)
    norm1 = embs[1] / (np.linalg.norm(embs[1]) + 1e-8)
    cos_sim = float(np.dot(norm0, norm1))
    return float(max(0.0, cos_sim))


def compute_compression_ratio(
    original_text: str,
    simplified_text: str = None,
    original_bits: float = None,
    transmitted_bits: float = None,
) -> float:
    if original_bits is not None and transmitted_bits is not None:
        if original_bits == 0:
            return 0.0
        return transmitted_bits / original_bits
    if simplified_text is not None and original_text:
        orig_words = len(original_text.split())
        simp_words = len(simplified_text.split())
        if orig_words == 0:
            return 0.0
        return simp_words / orig_words
    return 0.0


def is_intent_satisfied(
    tau_S: float,
    vartheta_S: float,
    tau_S_int: float,
    vartheta_S_int: float,
) -> bool:
    return tau_S <= tau_S_int and vartheta_S <= vartheta_S_int


def adjust_intent(
    delay_intent: float,
    quality_intent: float,
    tau_w: float = 0.0,
    omega1: float = 0.05,
    omega2: float = 0.02,
) -> tuple:
    """
    Intent adjustment under high-traffic scenarios (Eq. 19-20).
    
    When traffic load is high, intents are relaxed to avoid message failure:
    - Eq. 19: tau_S,int = tau_S,int * exp(omega1 * tau_w)  (relax delay — allow more time)
    - Eq. 20: vartheta_S,int = vartheta_S,int * exp(-omega2 * tau_w)  (relax quality — allow lower quality)
    
    Args:
        delay_intent: original delay intent in seconds
        quality_intent: original quality intent [0,1]
        tau_w: waiting time proxy (traffic load - 0.5, clamped to [0,1])
        omega1: attenuation factor for delay intent (paper: 0.05)
        omega2: attenuation factor for quality intent (paper: 0.02)
    
    Returns:
        (adjusted_delay_intent, adjusted_quality_intent)
    """
    import math
    
    # Eq. 19: relax delay intent (increase allowed delay)
    adjusted_delay = delay_intent * math.exp(omega1 * tau_w)
    
    # Eq. 20: relax quality intent (decrease quality requirement)
    adjusted_quality = quality_intent * math.exp(-omega2 * tau_w)
    
    # Clamp to reasonable ranges
    adjusted_delay = min(adjusted_delay, 10.0)  # Max 10 seconds
    adjusted_quality = max(adjusted_quality, 0.5)  # Min 50% quality
    
    return adjusted_delay, adjusted_quality
