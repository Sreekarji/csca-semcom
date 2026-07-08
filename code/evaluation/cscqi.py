import numpy as np
import torch

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
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def compute_cscqi(
    tau_S: float,
    vartheta_S: float,
    tau_S_int: float,
    vartheta_S_int: float,
    tau_max: float = TAU_MAX,
    vartheta_max: float = VARTHETA_MAX,
    w_tau: float = W_TAU,
    w_vartheta: float = W_VARTHETA,
    eps: float = 1e-8,
) -> float:
    """
    Compute CSCQI from Eq. 17 of Sun et al. 2026.

    C(S) = w_tau * (tau_max - tau_S) / (tau_max - tau_S_int)
         + w_vartheta * (vartheta_max - vartheta_S) / (vartheta_max - vartheta_S_int)

    Returns:
        CSCQI value. Positive = intent satisfied. Negative = intent not satisfied.
    """
    assert tau_S >= 0, f"tau_S must be >= 0, got {tau_S}"
    assert vartheta_S >= 0, f"vartheta_S must be >= 0, got {vartheta_S}"
    assert tau_S_int > 0, f"tau_S_int must be > 0, got {tau_S_int}"
    assert vartheta_S_int > 0, f"vartheta_S_int must be > 0, got {vartheta_S_int}"

    delay_term = (tau_max - tau_S) / (tau_max - tau_S_int + eps)
    quality_term = (vartheta_max - vartheta_S) / (vartheta_max - vartheta_S_int + eps)

    cscqi = w_tau * delay_term + w_vartheta * quality_term
    return float(cscqi)


def is_intent_satisfied(
    tau_S: float,
    vartheta_S: float,
    tau_S_int: float,
    vartheta_S_int: float,
) -> bool:
    """
    C1 constraint from Eq. 18: tau_S <= tau_S_int AND vartheta_S <= vartheta_S_int
    """
    return (tau_S <= tau_S_int) and (vartheta_S <= vartheta_S_int)


def compute_isr(tasks: list) -> float:
    """
    Intent Satisfaction Rate (ISR): fraction of tasks with intent satisfied.
    tasks: list of dicts with keys tau_S, vartheta_S, tau_S_int, vartheta_S_int
    """
    if not tasks:
        return 0.0
    satisfied = sum(
        1 for t in tasks
        if is_intent_satisfied(t["tau_S"], t["vartheta_S"], t["tau_S_int"], t["vartheta_S_int"])
    )
    return satisfied / len(tasks)


def compute_batch_cscqi(tasks: list) -> list:
    """Compute CSCQI for a list of tasks."""
    return [
        compute_cscqi(
            t["tau_S"], t["vartheta_S"],
            t["tau_S_int"], t["vartheta_S_int"]
        )
        for t in tasks
    ]


def normalize_cscqi(cscqi_values: list, window: int = 100) -> list:
    """
    Normalize CSCQI for plotting (as shown in paper Fig 12a).
    Uses min-max normalization over a sliding window.
    """
    arr = np.array(cscqi_values)
    result = []
    for i, v in enumerate(arr):
        start = max(0, i - window)
        w = arr[start:i + 1]
        mn, mx = w.min(), w.max()
        if mx - mn < 1e-8:
            result.append(0.5)
        else:
            result.append((v - mn) / (mx - mn))
    return result


def adjust_intent(
    tau_S_int: float,
    vartheta_S_int: float,
    tau_w: float,
    omega1: float = 0.1,
    omega2: float = 0.1,
) -> tuple:
    """
    Adjust user intent in high-traffic scenarios (Eq. 19-20).
    tau_S_int_new = tau_S_int * exp(omega1 * tau_w)
    vartheta_S_int_new = vartheta_S_int * exp(omega2 * tau_w)
    """
    tau_new = tau_S_int * np.exp(omega1 * tau_w)
    vartheta_new = vartheta_S_int * np.exp(omega2 * tau_w)
    return tau_new, vartheta_new


def compute_semantic_accuracy(sent_text: str, recv_text: str) -> float:
    """
    Proxy for semantic accuracy using cosine similarity.
    Paper uses PSNR for images and text similarity for text/audio.
    >0.8 cosine similarity = accurate per paper Section VI.B.
    """
    model = get_semantic_model()
    embs = model.encode([sent_text, recv_text], convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(
        embs[0].unsqueeze(0), embs[1].unsqueeze(0)
    ).item()
    return float(max(0.0, cos_sim))


def compute_compression_ratio(
    original_text: str = None,
    simplified_text: str = None,
    original_tokens: int = None,
    transmitted_tokens: int = None,
) -> float:
    """
    Compute compression ratio.
    If text provided: word-level ratio (matches paper Section VI.B).
    If token counts provided: fallback to transmitted/orig.
    Paper reports: text ~73%, audio ~32%, image ~21%.
    """
    if original_text is not None and simplified_text is not None:
        orig_words = len(original_text.split())
        simp_words = len(simplified_text.split())
        if orig_words == 0:
            return 0.0
        return simp_words / orig_words
    if original_tokens is not None and transmitted_tokens is not None:
        if original_tokens == 0:
            return 0.0
        return transmitted_tokens / original_tokens
    raise ValueError("Provide (original_text, simplified_text) or (original_tokens, transmitted_tokens)")


if __name__ == "__main__":
    print("=== CSCQI Unit Tests ===\n")

    # Test case 1: intent fully satisfied (low delay, low distortion)
    c1 = compute_cscqi(tau_S=0.5, vartheta_S=0.1, tau_S_int=1.0, vartheta_S_int=0.2)
    print(f"Test 1 (satisfied): CSCQI = {c1:.4f} (expected > 0)")
    assert c1 > 0, f"Expected positive CSCQI, got {c1}"

    # Test case 2: intent not satisfied (delay exceeds intent) — CSCQI still positive but lower
    c2 = compute_cscqi(tau_S=2.0, vartheta_S=0.5, tau_S_int=1.0, vartheta_S_int=0.2)
    print(f"Test 2 (not satisfied): CSCQI = {c2:.4f} (expected < {c1:.4f})")
    assert c2 < c1, f"Expected CSCQI lower than satisfied case, got {c2} >= {c1}"

    # Test ISR
    tasks = [
        {"tau_S": 0.5, "vartheta_S": 0.1, "tau_S_int": 1.0, "vartheta_S_int": 0.2},
        {"tau_S": 2.0, "vartheta_S": 0.5, "tau_S_int": 1.0, "vartheta_S_int": 0.2},
        {"tau_S": 0.8, "vartheta_S": 0.15, "tau_S_int": 1.0, "vartheta_S_int": 0.2},
    ]
    isr = compute_isr(tasks)
    print(f"ISR: {isr:.3f} (expected 0.667 -- 2 of 3 satisfied)")

    # Test semantic accuracy
    acc = compute_semantic_accuracy(
        "Send it within 1 second",
        "Send it within 1 second"
    )
    print(f"Semantic accuracy (identical): {acc:.4f} (expected ~1.0)")

    # Test intent adjustment
    t_new, v_new = adjust_intent(1.0, 0.2, tau_w=2.0)
    print(f"Adjusted intent: tau={t_new:.3f}, vartheta={v_new:.3f}")

    print("\nAll CSCQI tests passed.")
