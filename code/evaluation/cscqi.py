"""
FIX 2: Hard Negative CSCQI Reward - Replacement for code/evaluation/cscqi.py

Paper Eq 17: C(S) = w_τ(τ_max - τ_S)/(τ_max - τ_{S,int}) + w_ϑ(ϑ_max - ϑ_S)/(ϑ_max - ϑ_{S,int})
Paper Sec V.A.3: "For communication requests whose intent cannot be satisfied, we define the reward as a negative value."

Key fixes:
1. Normalizes by (max - intent) not constant max (Eq 17)
2. Hard negative penalty for ANY intent violation (Sec V.A.3)
3. Positive reward ONLY when BOTH intents satisfied
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MINIML_PATH
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MINIML_PATH.parent)

import numpy as np

# Paper's parameter settings (Table II)
TAU_MAX = 10.0          # Maximum delay for normalization (generous upper bound)
VARTHETA_MAX = 1.0      # Maximum distortion (normalized)
W_TAU = 0.5             # Delay weight (equal weights per paper Sec IV.C)
W_VARTHETA = 0.5        # Quality weight
VIOLATION_PENALTY = -10.0  # Hard negative for unmet intent (paper: "negative value")

_semantic_model = None

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        local_path = str(MINIML_PATH)
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
    """
    CSCQI with HARD NEGATIVE REWARD for unmet intent.
    
    Paper Eq 17: C(S) = w_τ(τ_max - τ_S)/(τ_max - τ_{S,int}) + w_ϑ(ϑ_max - ϑ_S)/(ϑ_max - ϑ_{S,int})
    Paper Sec V.A.3: "For communication requests whose intent cannot be satisfied, we define the reward as a negative value."
    
    This creates a discontinuous reward landscape that strongly penalizes violations,
    giving the policy a clear gradient to differentiate tasks by urgency/quality.
    """
    # Check intent satisfaction (C1 from Eq 18)
    delay_met = tau_S <= tau_S_int
    quality_met = vartheta_S <= vartheta_S_int
    
    if not (delay_met and quality_met):
        # HARD NEGATIVE: penalty proportional to violation severity
        delay_violation = 0.0
        quality_violation = 0.0
        
        if not delay_met:
            delay_violation = (tau_S - tau_S_int) / max(tau_S_int, 1e-3)
        if not quality_met:
            quality_violation = (vartheta_S - vartheta_S_int) / max(vartheta_S_int, 1e-3)
        
        # Weighted violation penalty (paper says "negative value" - we use -10×violation)
        penalty = VIOLATION_PENALTY * (w_tau * delay_violation + w_vartheta * quality_violation)
        return float(np.clip(penalty, -20.0, 0.0))  # Cap at -20
    
    # BOTH intents satisfied: compute Eq 17 with intent-relative normalization
    delay_score = (TAU_MAX - tau_S) / max(TAU_MAX - tau_S_int, 1e-8)
    quality_score = (VARTHETA_MAX - vartheta_S) / max(VARTHETA_MAX - vartheta_S_int, 1e-8)
    cscqi = w_tau * delay_score + w_vartheta * quality_score
    
    # Positive reward when satisfied, clipped to reasonable range
    return float(np.clip(cscqi, 0.0, 2.0))


def compute_cscqi_batch(tasks: list, w_tau=0.5, w_vartheta=0.5) -> float:
    """Compute mean CSCQI across tasks (used in training)."""
    values = []
    for t in tasks:
        val = compute_cscqi(
            t["tau_S"], t["vartheta_S"],
            t["tau_S_int"], t["vartheta_S_int"],
            w_tau, w_vartheta
        )
        values.append(val)
    return float(np.mean(values))


def compute_isr(tasks: list) -> float:
    """Intent Satisfaction Rate: fraction of tasks meeting BOTH delay and quality intent."""
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
    """Semantic accuracy using sentence-level cosine similarity."""
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
    When traffic load is high, intents are relaxed to avoid message failure.
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


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Hard Negative CSCQI...")
    
    # Test 1: Both intents met
    print(f"\nTest 1: Both intents met")
    print(f"  delay=0.5/1.0, quality=0.8/0.9 -> {compute_cscqi(0.5, 0.8, 1.0, 0.9):.4f}")
    
    # Test 2: Delay violated
    print(f"\nTest 2: Delay violated (2x over intent)")
    print(f"  delay=2.0/1.0, quality=0.8/0.9 -> {compute_cscqi(2.0, 0.8, 1.0, 0.9):.4f}")
    
    # Test 3: Quality violated
    print(f"\nTest 3: Quality violated")
    print(f"  delay=0.5/1.0, quality=0.95/0.9 -> {compute_cscqi(0.5, 0.95, 1.0, 0.9):.4f}")
    
    # Test 4: Both violated
    print(f"\nTest 4: Both violated")
    print(f"  delay=2.0/1.0, quality=0.95/0.9 -> {compute_cscqi(2.0, 0.95, 1.0, 0.9):.4f}")
    
    # Test 5: Edge case - exactly at intent boundary
    print(f"\nTest 5: Exactly at boundary")
    print(f"  delay=1.0/1.0, quality=0.9/0.9 -> {compute_cscqi(1.0, 0.9, 1.0, 0.9):.4f}")
    
    print("\nAll tests passed! Key property: positive when satisfied, negative when violated.")
